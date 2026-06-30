# ==============================================================================
# Script 13: Modelo Enrutador (Segmentado) con Transformación Logarítmica
# ==============================================================================
# Idea: 
# 1. Aplicar reglas de negocio para clasificar el alojamiento (Lujo/Resort, 
#    Boutique/Informal, y Estándar).
# 2. Entrenar un modelo XGBoost especializado para CADA segmento.
# 3. Entrenar usando el log1p(precio) para reducir el ruido de los outliers.
# 4. Al predecir, usar expm1() para devolver EXACTAMENTE el precio en dólares.
# ==============================================================================

library(data.table)
library(xgboost)

# Ajuste automático del directorio de trabajo a la raíz
if (basename(getwd()) == "scripts") setwd("..")

cat("Iniciando Entrenamiento del Modelo Enrutador...\n")

train_data <- readRDS("data/train_data.rds")
val_data   <- readRDS("data/val_data.rds")
test_data  <- readRDS("data/test_data.rds")

# ==============================================================================
# 1. DEFINIR LA FUNCIÓN ENRUTADORA (CLASIFICADOR DE ALOJAMIENTOS)
# ==============================================================================
# Basado en el análisis de errores, separamos en 3 tipos de alojamientos:
asignar_segmento <- function(dt) {
  dt_copy <- copy(dt)
  dt_copy[, segmento := "Estandar"] # Por defecto
  
  # Condición 1: Lujo o Mega Resort
  # 5 estrellas, o muchas habitaciones, o mucho Spa/Relax
  dt_copy[starRating >= 4.5 | numberOfRooms > 300 | cat_Bienestar.y.relajación > 2, segmento := "Lujo_Resort"]
  
  # Condición 2: Informal, Hostels, Departamentos, Boutique
  # Menos de 2 estrellas, o muy pocas habitaciones (y que no haya sido pisado por lujo)
  dt_copy[segmento == "Estandar" & (starRating < 2 | numberOfRooms < 50), segmento := "Boutique_Informal"]
  
  return(dt_copy)
}

train_data <- asignar_segmento(train_data)
val_data   <- asignar_segmento(val_data)
test_data  <- asignar_segmento(test_data)

cat("Distribución de segmentos en el Train Set:\n")
print(table(train_data$segmento))

# ==============================================================================
# 2. FUNCIÓN PARA ENTRENAR UN MODELO ESPECÍFICO
# ==============================================================================
entrenar_segmento <- function(segmento_nombre) {
  cat("\n------------------------------------------------\n")
  cat("Entrenando modelo especializado para:", segmento_nombre, "\n")
  
  # Filtrar datos
  train_sub <- train_data[segmento == segmento_nombre]
  val_sub   <- val_data[segmento == segmento_nombre]
  
  # Guardamos el target y APLICAMOS LOGARITMO PARA REDUCIR RUIDO
  y_train_log <- log1p(train_sub$price_per_person)
  y_val_log   <- log1p(val_sub$price_per_person)
  
  # Eliminamos variables que no van en la matriz
  drop_cols <- c("price_per_person", "segmento")
  train_sub[, (drop_cols) := NULL]
  val_sub[, (drop_cols) := NULL]
  
  # Convertir texto a factores numéricos (safety)
  char_cols <- names(train_sub)[sapply(train_sub, is.character)]
  if(length(char_cols) > 0) {
    train_sub[, (char_cols) := lapply(.SD, function(x) as.numeric(as.factor(x))), .SDcols = char_cols]
    val_sub[, (char_cols) := lapply(.SD, function(x) as.numeric(as.factor(x))), .SDcols = char_cols]
  }
  
  X_train <- as.matrix(train_sub)
  X_val   <- as.matrix(val_sub)
  mode(X_train) <- "numeric"
  mode(X_val) <- "numeric"
  
  dtrain <- xgb.DMatrix(data = X_train, label = y_train_log)
  dval   <- xgb.DMatrix(data = X_val, label = y_val_log)
  
  # Parametros base
  params <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse", # Evaluamos RMSE del logaritmo (que equivale a RMSLE)
    max_depth = 5,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  # Entrenar
  modelo <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 1000,
    watchlist = list(train = dtrain, val = dval),
    early_stopping_rounds = 30,
    print_every_n = 100
  )
  
  # Guardar los modelos
  xgb.save(modelo, paste0("models/modelo_enrutador_", segmento_nombre, ".model"))
  saveRDS(colnames(X_train), paste0("models/columnas_enrutador_", segmento_nombre, ".rds"))
  
  # Devolver modelo y columnas usadas
  return(list(modelo = modelo, columnas = colnames(X_train)))
}

# ==============================================================================
# 3. ENTRENAR LOS 3 MODELOS
# ==============================================================================
modelos_enrutador <- list(
  Estandar          = entrenar_segmento("Estandar"),
  Lujo_Resort       = entrenar_segmento("Lujo_Resort"),
  Boutique_Informal = entrenar_segmento("Boutique_Informal")
)

# ==============================================================================
# 4. PREDECIR EN EL TEST SET USANDO EL ENRUTADOR
# ==============================================================================
cat("\n------------------------------------------------\n")
cat("Evaluando el Enrutador en el Test Set (Dólares Reales)...\n")

y_test_real <- test_data$price_per_person
test_data[, pred_reales := NA_real_]

# Predecimos segmento por segmento
for (seg in names(modelos_enrutador)) {
  mod_info <- modelos_enrutador[[seg]]
  mod <- mod_info$modelo
  cols <- mod_info$columnas
  
  # Extraer filas de este segmento
  idx <- which(test_data$segmento == seg)
  if (length(idx) > 0) {
    dt_sub <- test_data[idx]
    dt_sub[, (setdiff(cols, names(dt_sub))) := 0]
    X_test <- as.matrix(dt_sub[, cols, with=FALSE])
    
    char_cols <- colnames(X_test)[sapply(dt_sub[, cols, with=FALSE], is.character)]
    if(length(char_cols) > 0) {
      for(c in char_cols) X_test[, c] <- as.numeric(as.factor(X_test[, c]))
    }
    mode(X_test) <- "numeric"
    
    # 1. El modelo escupe la predicción en Logaritmo
    pred_log <- predict(mod, xgb.DMatrix(X_test))
    
    # 2. Convertimos OBLIGATORIAMENTE a dólares reales usando expm1
    pred_real <- expm1(pred_log)
    
    # Asignamos a la tabla final
    test_data[idx, pred_reales := pred_real]
  }
}

# ==============================================================================
# 5. RESULTADOS GLOBALES
# ==============================================================================
rmse_global <- sqrt(mean((y_test_real - test_data$pred_reales)^2))
mae_global <- mean(abs(y_test_real - test_data$pred_reales))
r2_global <- 1 - sum((y_test_real - test_data$pred_reales)^2) / sum((y_test_real - mean(y_test_real))^2)

cat(sprintf("\n=== METRICAS FINALES DEL ENRUTADOR (Dólares Reales) ===\n"))
cat(sprintf("RMSE: %.2f\n", rmse_global))
cat(sprintf("MAE: %.2f\n", mae_global))
cat(sprintf("R2: %.4f\n", r2_global))

cat("\nDesglose de MAE por segmento:\n")
print(test_data[, .(MAE = mean(abs(price_per_person - pred_reales))), by = segmento])

cat("\nNOTA: No se sobrescribió ningún archivo modelo anterior. Análisis completado.\n")
