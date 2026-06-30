# =======================================================
# Ajuste autom�tico del directorio de trabajo a la ra�z
if (basename(getwd()) == "scripts") setwd("..")
# =======================================================
# ==============================================================================
# Script de Entrenamiento y Evaluación de Modelos (Lasso vs XGBoost)
# Rol: Científico de Datos Senior
# Objetivo: Entrenar, comparar y evaluar un baseline (Lasso) vs XGBoost para 
#           predecir 'price_per_person'.
# ==============================================================================

# Cargar las librerías necesarias
# data.table: Manipulación rápida de datos
# glmnet: Entrenamiento de modelos penalizados (Lasso/Ridge)
# xgboost: Gradient Boosting escalable y de alto rendimiento
library(data.table)
library(glmnet)
library(xgboost)

cat("Iniciando pipeline de modelado predictivo...\n")

# ------------------------------------------------------------------------------
# 1. CARGA DE DATOS Y PREPARACIÓN DE MATRICES
# ------------------------------------------------------------------------------
train_data <- readRDS("data/train_data.rds")
val_data <- readRDS("data/val_data.rds")
test_data <- readRDS("data/test_data.rds")

# ==============================================================================
# OPCIONAL: REDUCCIÓN DE TAMAÑO MUESTRAL A ~15.000 DATOS (Para pruebas rápidas)
# Para usar el dataset completo de 150k, simplemente comenta desde aquí...
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
# ...hasta aquí.
# ==============================================================================

target_var <- "price_per_person"

# Separar el vector objetivo 'y' para cada set
y_train <- train_data[[target_var]]
y_val   <- val_data[[target_var]]
y_test  <- test_data[[target_var]]

# Función auxiliar para preparar la matriz de predictores (X). 
# glmnet y xgboost requieren matrices numéricas, no data.frames.
prepare_matrix <- function(dt, target) {
  dt_copy <- copy(dt)
  dt_copy[, (target) := NULL] # Removemos la variable objetivo
  
  # Si quedan variables de tipo caracter (que no fueron purgadas), las codificamos.
  # Para un modelo más estricto se usaría One-Hot Encoding, pero la conversión a
  # numérico (Label Encoding) es aceptable para XGBoost.
  char_cols <- names(dt_copy)[sapply(dt_copy, is.character)]
  if (length(char_cols) > 0) {
    dt_copy[, (char_cols) := lapply(.SD, function(x) as.numeric(as.factor(x))), .SDcols = char_cols]
  }
  
  return(as.matrix(dt_copy))
}

cat("Preparando matrices de predictores (X)...\n")
X_train <- prepare_matrix(train_data, target_var)
X_val   <- prepare_matrix(val_data, target_var)
X_test  <- prepare_matrix(test_data, target_var)

# ------------------------------------------------------------------------------
# 2. MODELO BASELINE (REGRESIÓN LASSO)
# ------------------------------------------------------------------------------
# Lasso (alpha = 1) no solo predice, sino que realiza selección de variables al 
# forzar que los coeficientes menos importantes sean exactamente cero (Regularización L1).
cat("\n[1/2] Entrenando Modelo Baseline (Regresión Lasso)...\n")

set.seed(42) # Reproducibilidad en validación cruzada
# cv.glmnet con alpha = 0.5 (Elastic Net). Mezcla penalización L1 (Lasso, selección de variables)
# y L2 (Ridge, contracción de coeficientes correlacionados) para mejorar la generalización.
lasso_model <- cv.glmnet(X_train, y_train, alpha = 0.5, family = "gaussian", nfolds = 5)

# Predecimos sobre el test set utilizando el mejor lambda encontrado (lambda.min)
lasso_preds <- predict(lasso_model, s = lasso_model$lambda.min, newx = X_test)

# ------------------------------------------------------------------------------
# 3. MODELO AVANZADO (XGBOOST)
# ------------------------------------------------------------------------------
# XGBoost es ideal para relaciones no lineales e interacciones complejas.
cat("\n[2/2] Entrenando Modelo Avanzado (XGBoost)...\n")

# Conversión al formato optimizado interno de xgboost (DMatrix)
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dval   <- xgb.DMatrix(data = X_val, label = y_val)
dtest  <- xgb.DMatrix(data = X_test, label = y_test)

# Definimos el watchlist para monitorear el error en train y validation durante el ajuste.
evals_list <- list(train = dtrain, validation = dval)

# Hiperparámetros optimizados para generalización y velocidad:
xgb_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  tree_method = "hist",     # Aceleración extrema
  eta = 0.05,               # Learning rate conservador
  max_depth = 6,            # Profundidad moderada para evitar sobreajuste
  subsample = 0.8,          # Muestreo de filas (reduce varianza/overfitting)
  colsample_bytree = 0.8,   # Muestreo de columnas por árbol (reduce correlación entre árboles)
  min_child_weight = 5      # Exige un mínimo de peso en cada hoja (poda árboles muy específicos)
)

set.seed(42)
# Entrenamiento del modelo
xgb_model <- suppressWarnings(xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 1500, # Límite alto arbitrario; el modelo frenará solo gracias al early_stopping.
  watchlist = evals_list,
  early_stopping_rounds = 15, # Detenerse si el error de validación no mejora en 15 iteraciones seguidas.
  print_every_n = 50 # Reducir verbosidad en la consola
))

# Predicción sobre el test set (XGBoost usa automáticamente la mejor iteración)
xgb_preds <- predict(xgb_model, dtest)

# ------------------------------------------------------------------------------
# 4. EVALUACIÓN Y EXPORTACIÓN DE MÉTRICAS
# ------------------------------------------------------------------------------
cat("\nEvaluando modelos sobre el Test Set...\n")

# Funciones de evaluación de negocio y estadística
calc_rmse <- function(y, y_hat) sqrt(mean((y - y_hat)^2)) # Sensible a grandes errores
calc_mae  <- function(y, y_hat) mean(abs(y - y_hat))      # Promedio del error absoluto
calc_r2   <- function(y, y_hat) {                         # Proporción de varianza explicada
  ss_res <- sum((y - y_hat)^2)
  ss_tot <- sum((y - mean(y))^2)
  return(1 - (ss_res / ss_tot))
}

# Métricas Lasso
lasso_metrics <- c(
  RMSE = calc_rmse(y_test, lasso_preds),
  MAE  = calc_mae(y_test, lasso_preds),
  R2   = calc_r2(y_test, lasso_preds)
)

# Métricas XGBoost
xgb_metrics <- c(
  RMSE = calc_rmse(y_test, xgb_preds),
  MAE  = calc_mae(y_test, xgb_preds),
  R2   = calc_r2(y_test, xgb_preds)
)

# Armado del DataFrame comparativo
metricas_df <- data.frame(
  Modelo = c("Elastic Net (Baseline Mejorado)", "XGBoost (Fuerte Generalización)"),
  RMSE   = c(lasso_metrics["RMSE"], xgb_metrics["RMSE"]),
  MAE    = c(lasso_metrics["MAE"], xgb_metrics["MAE"]),
  R2     = c(lasso_metrics["R2"], xgb_metrics["R2"])
)

# Guardar a disco
write.csv(metricas_df, "metricas_evaluacion_modelos.csv", row.names = FALSE)
cat("Métricas de rendimiento exportadas a: metricas_evaluacion_modelos.csv\n")
print(metricas_df)

# ------------------------------------------------------------------------------
# 5. EXTRACCIÓN DE IMPORTANCIA DE VARIABLES (XGBOOST)
# ------------------------------------------------------------------------------
cat("\nExtrayendo importancia de variables (Gain)...\n")

# xgb.importance calcula el Gain (mejora en accuracy traida por una feature), 
# Cover (número de observaciones relativas) y Frequency.
importance_matrix <- xgb.importance(feature_names = colnames(X_train), model = xgb_model)

# Filtramos el top 20
top_20_importance <- head(importance_matrix, 20)

# Exportar a disco
write.csv(top_20_importance, "importancia_variables_xgboost.csv", row.names = FALSE)
cat("Top 20 variables más importantes exportadas a: importancia_variables_xgboost.csv\n")

# ==============================================================================
# 6. GUARDADO DE MODELO SIMPLE PARA LA WEB APP
# ==============================================================================
cat("\nGuardando modelo Elastic Net Simple (Baseline) para la Web App...\n")
saveRDS(lasso_model, "models/modelo_simple.rds")
saveRDS(colnames(X_train), "models/columnas_modelo_simple.rds")

cat("\nPipeline finalizado exitosamente.\n")
