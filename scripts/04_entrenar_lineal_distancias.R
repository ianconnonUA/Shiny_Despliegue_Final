# =======================================================
# Ajuste automtico del directorio de trabajo a la raz
if (basename(getwd()) == "scripts") setwd("..")
# =======================================================
# ==============================================================================
# Script de Entrenamiento y Evaluación: INGENIERÍA DE CARACTERÍSTICAS ESPACIALES
# Objetivo: Crear indicadores de distancia a puntos turísticos de Río de Janeiro 
#           usando latitud/longitud y probar si mejoran el rendimiento del modelo.
# ==============================================================================

library(data.table)
library(glmnet)
library(xgboost)

cat("Iniciando pipeline con Ingeniería de Distancias...\n")

# ------------------------------------------------------------------------------
# 1. CARGA DE DATOS Y REDUCCIÓN OPCIONAL
# ------------------------------------------------------------------------------
train_data <- readRDS("data/train_data.rds")
val_data   <- readRDS("data/val_data.rds")
test_data  <- readRDS("data/test_data.rds")

# ==============================================================================
# OPCIONAL: REDUCCIÓN DE TAMAÑO MUESTRAL A ~15.000 DATOS (Para pruebas rápidas)
# ==============================================================================
# set.seed(42)
# total_rows <- nrow(train_data) + nrow(val_data) + nrow(test_data)
# if (total_rows > 15000) {
#   frac <- 15000 / total_rows
#   train_data <- train_data[sample(.N, floor(nrow(train_data) * frac))]
#   val_data   <- val_data[sample(.N, floor(nrow(val_data) * frac))]
#   test_data  <- test_data[sample(.N, floor(nrow(test_data) * frac))]
#   cat("\n[!] AVISO: Entrenando con una muestra reducida de ~15.000 observaciones.\n")
# }
# ==============================================================================

# ------------------------------------------------------------------------------
# 2. FEATURE ENGINEERING: DISTANCIAS A PUNTOS TURÍSTICOS (HAVERSINE)
# ------------------------------------------------------------------------------
cat("\nCalculando distancias a puntos turísticos clave de Río de Janeiro...\n")

# Coordenadas de los principales atractivos de Río de Janeiro
puntos_interes <- list(
  Cristo_Redentor = c(lat = -22.9519, lon = -43.2105),
  Pan_de_Azucar   = c(lat = -22.9492, lon = -43.1563),
  Copacabana      = c(lat = -22.9711, lon = -43.1822),
  Ipanema         = c(lat = -22.9868, lon = -43.2024),
  Aeropuerto_GIG  = c(lat = -22.8134, lon = -43.2494),
  Centro_Lapa     = c(lat = -22.9142, lon = -43.1812),
  Leblon          = c(lat = -22.9877, lon = -43.2219),
  Barra_da_Tijuca = c(lat = -23.0096, lon = -43.3308),
  Botafogo        = c(lat = -22.9491, lon = -43.1834),
  Flamengo        = c(lat = -22.9348, lon = -43.1725),
  Jardin_Botanico = c(lat = -22.9673, lon = -43.2285),
  Maracana        = c(lat = -22.9121, lon = -43.2302),
  Museo_Manana    = c(lat = -22.8945, lon = -43.1795),
  Parque_Lage     = c(lat = -22.9602, lon = -43.2127),
  Aeropuerto_SDU  = c(lat = -22.9110, lon = -43.1631),
  Arpoador        = c(lat = -22.9892, lon = -43.1917),
  Fuerte_Copa     = c(lat = -22.9859, lon = -43.1868),
  Escalera_Selaron= c(lat = -22.9155, lon = -43.1794)
)

# Función vectorizada para calcular distancia Haversine en Kilómetros
calcular_haversine <- function(lat1, lon1, lat2, lon2) {
  R <- 6371 # Radio de la Tierra en km
  lat1_rad <- lat1 * pi / 180
  lon1_rad <- lon1 * pi / 180
  lat2_rad <- lat2 * pi / 180
  lon2_rad <- lon2 * pi / 180
  
  dlat <- lat2_rad - lat1_rad
  dlon <- lon2_rad - lon1_rad
  
  a <- sin(dlat/2)^2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)^2
  c <- 2 * atan2(sqrt(a), sqrt(1-a))
  return(R * c)
}

# Aplicamos la función a todos los datasets para cada punto de interés
agregar_distancias <- function(dt) {
  if (all(c("latitude", "longitude") %in% names(dt))) {
    for (nombre in names(puntos_interes)) {
      punto_lat <- puntos_interes[[nombre]]["lat"]
      punto_lon <- puntos_interes[[nombre]]["lon"]
      
      col_nombre <- paste0("dist_", nombre, "_km")
      dt[, (col_nombre) := calcular_haversine(latitude, longitude, punto_lat, punto_lon)]
    }
    
    # Opcional: Distancia mínima al mar (aproximada tomando Copa o Ipanema)
    dt[, dist_playa_min_km := pmin(dist_Copacabana_km, dist_Ipanema_km, dist_Leblon_km, dist_Barra_da_Tijuca_km, dist_Botafogo_km, dist_Flamengo_km, dist_Arpoador_km, na.rm=TRUE)]
  } else {
    warning("Las columnas 'latitude' y 'longitude' no están presentes. No se calcularon distancias.")
  }
  return(dt)
}

train_data <- agregar_distancias(train_data)
val_data   <- agregar_distancias(val_data)
test_data  <- agregar_distancias(test_data)

# ------------------------------------------------------------------------------
# 3. PREPARACIÓN DE MATRICES
# ------------------------------------------------------------------------------
target_var <- "price_per_person"

y_train <- train_data[[target_var]]
y_val   <- val_data[[target_var]]
y_test  <- test_data[[target_var]]

prepare_matrix <- function(dt, target) {
  dt_copy <- copy(dt)
  dt_copy[, (target) := NULL]
  char_cols <- names(dt_copy)[sapply(dt_copy, is.character)]
  if (length(char_cols) > 0) {
    dt_copy[, (char_cols) := lapply(.SD, function(x) as.numeric(as.factor(x))), .SDcols = char_cols]
  }
  # Manejar posibles NAs en latitude/longitude
  num_cols <- names(dt_copy)[sapply(dt_copy, is.numeric)]
  for (col in num_cols) {
    if (any(is.na(dt_copy[[col]]))) {
      dt_copy[is.na(get(col)), (col) := median(dt_copy[[col]], na.rm = TRUE)]
    }
  }
  return(as.matrix(dt_copy))
}

cat("Preparando matrices de predictores (X)...\n")
X_train <- prepare_matrix(train_data, target_var)
X_val   <- prepare_matrix(val_data, target_var)
X_test  <- prepare_matrix(test_data, target_var)

# ------------------------------------------------------------------------------
# 4. MODELO BASELINE (ELASTIC NET)
# ------------------------------------------------------------------------------
cat("\n[1/2] Entrenando Modelo Lineal (Elastic Net)...\n")

set.seed(42)
lasso_model <- cv.glmnet(X_train, y_train, alpha = 0.5, family = "gaussian", nfolds = 5)
lasso_preds <- predict(lasso_model, s = lasso_model$lambda.min, newx = X_test)

# ------------------------------------------------------------------------------
# 5. MODELO AVANZADO (XGBOOST CON NUEVOS FEATURES)
# ------------------------------------------------------------------------------
cat("\n[2/2] Entrenando Modelo Avanzado (XGBoost con Datos Espaciales)...\n")

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dval   <- xgb.DMatrix(data = X_val, label = y_val)
dtest  <- xgb.DMatrix(data = X_test, label = y_test)

evals_list <- list(train = dtrain, validation = dval)

xgb_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  tree_method = "hist",     
  eta = 0.05,               
  max_depth = 6,            
  subsample = 0.8,          
  colsample_bytree = 0.8,   
  min_child_weight = 5      
)

set.seed(42)
xgb_model <- suppressWarnings(xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 1500, 
  watchlist = evals_list,
  early_stopping_rounds = 15, 
  verbose = 0 
))

xgb_preds <- predict(xgb_model, dtest)

# ------------------------------------------------------------------------------
# 6. EVALUACIÓN Y RESULTADOS
# ------------------------------------------------------------------------------
cat("\nEvaluando modelos sobre el Test Set...\n")

calc_rmse <- function(y, y_hat) sqrt(mean((y - y_hat)^2))
calc_mae  <- function(y, y_hat) mean(abs(y - y_hat))
calc_r2   <- function(y, y_hat) {
  ss_res <- sum((y - y_hat)^2)
  ss_tot <- sum((y - mean(y))^2)
  return(1 - (ss_res / ss_tot))
}

lasso_metrics <- c(RMSE = calc_rmse(y_test, lasso_preds), MAE = calc_mae(y_test, lasso_preds), R2 = calc_r2(y_test, lasso_preds))
xgb_metrics <- c(RMSE = calc_rmse(y_test, xgb_preds), MAE = calc_mae(y_test, xgb_preds), R2 = calc_r2(y_test, xgb_preds))

metricas_df <- data.frame(
  Modelo = c("Elastic Net (Espacial)", "XGBoost (Espacial)"),
  RMSE   = c(lasso_metrics["RMSE"], xgb_metrics["RMSE"]),
  MAE    = c(lasso_metrics["MAE"], xgb_metrics["MAE"]),
  R2     = c(lasso_metrics["R2"], xgb_metrics["R2"])
)

cat("\n====================================================\n")
cat("MÉTRICAS COMPARATIVAS CON INGENIERÍA ESPACIAL\n")
cat("====================================================\n")
print(metricas_df)

cat("\n====================================================\n")
cat("IMPORTANCIA DE LAS NUEVAS VARIABLES DE DISTANCIA (TOP 20)\n")
cat("====================================================\n")
importance_matrix <- xgb.importance(feature_names = colnames(X_train), model = xgb_model)
top_features <- head(importance_matrix, 20)
print(top_features)

# Guardar a disco
write.csv(metricas_df, "metricas_distancias_espaciales.csv", row.names = FALSE)

# ==============================================================================
# 7. GUARDADO DE MODELO LINEAL HAVERSINE PARA LA WEB APP
# ==============================================================================
cat("\nGuardando modelo Lineal Haversine (Elastic Net) para la Web App...\n")
saveRDS(lasso_model, "models/modelo_linear_haversine.rds")
saveRDS(colnames(X_train), "models/columnas_linear_haversine.rds")

cat("\nPipeline finalizado.\n")
