# =======================================================
# Ajuste automtico del directorio de trabajo a la raz
if (basename(getwd()) == "scripts") setwd("..")
# =======================================================
library(data.table)
library(xgboost)

set.seed(123)

# 1. Carga de Datos
cat("Cargando datos...\n")
train_data <- readRDS("data/train_data.rds")
val_data   <- readRDS("data/val_data.rds")
test_data  <- readRDS("data/test_data.rds")

# ==============================================================================
# OPCIONAL: REDUCCIÓN DE TAMAÑO MUESTRAL A ~15.000 DATOS (Para pruebas rápidas)
# Para usar el dataset completo de 150k, simplemente comenta desde aquí...
# ==============================================================================
set.seed(123)
total_rows <- nrow(train_data) + nrow(val_data) + nrow(test_data)
if (total_rows > 15000) {
  frac <- 15000 / total_rows
  train_data <- train_data[sample(.N, floor(nrow(train_data) * frac))]
  val_data   <- val_data[sample(.N, floor(nrow(val_data) * frac))]
  test_data  <- test_data[sample(.N, floor(nrow(test_data) * frac))]
  cat("\n[!] AVISO: Entrenando con una muestra reducida de ~15.000 observaciones.\n")
}
# ==============================================================================
# ...hasta aquí.
# ==============================================================================

# ------------------------------------------------------------------------------
# 2. FEATURE ENGINEERING: DISTANCIAS A PUNTOS TURÍSTICOS (HAVERSINE)
# ------------------------------------------------------------------------------
cat("\nCalculando distancias a puntos turísticos clave de Río de Janeiro...\n")

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

calcular_haversine <- function(lat1, lon1, lat2, lon2) {
  R <- 6371 # Radio Tierra en km
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

agregar_distancias <- function(dt) {
  if (all(c("latitude", "longitude") %in% names(dt))) {
    for (nombre in names(puntos_interes)) {
      punto_lat <- puntos_interes[[nombre]]["lat"]
      punto_lon <- puntos_interes[[nombre]]["lon"]
      col_nombre <- paste0("dist_", nombre, "_km")
      dt[, (col_nombre) := calcular_haversine(latitude, longitude, punto_lat, punto_lon)]
    }
    # Indicador de distancia mínima a la playa
    dt[, dist_playa_min_km := pmin(dist_Copacabana_km, dist_Ipanema_km, dist_Leblon_km, dist_Barra_da_Tijuca_km, dist_Botafogo_km, dist_Flamengo_km, dist_Arpoador_km, na.rm=TRUE)]
  } else {
    warning("Las variables de coordenadas no existen. Imposible calcular distancias.")
  }
  return(dt)
}

train_data <- agregar_distancias(train_data)
val_data   <- agregar_distancias(val_data)
test_data  <- agregar_distancias(test_data)

target_var <- "price_per_person"

y_train <- train_data[[target_var]]
y_val   <- val_data[[target_var]]
y_test  <- test_data[[target_var]]

# ------------------------------------------------------------------------------
# 3. Preparacion de Matrices
# ------------------------------------------------------------------------------
prepare_matrix <- function(dt, target) {
  dt_copy <- copy(dt)
  dt_copy[, (target) := NULL]
  char_cols <- names(dt_copy)[sapply(dt_copy, is.character)]
  if (length(char_cols) > 0) {
    dt_copy[, (char_cols) := lapply(.SD, function(x) as.numeric(as.factor(x))), .SDcols = char_cols]
  }
  
  num_cols <- names(dt_copy)[sapply(dt_copy, is.numeric)]
  for (col in num_cols) {
    if (any(is.na(dt_copy[[col]]))) {
      dt_copy[is.na(get(col)), (col) := median(dt_copy[[col]], na.rm = TRUE)]
    }
  }
  return(as.matrix(dt_copy))
}

cat("Preparando matrices...\n")
X_train <- prepare_matrix(train_data, target_var)
X_val   <- prepare_matrix(val_data,   target_var)
X_test  <- prepare_matrix(test_data,  target_var)

X_trainval <- rbind(X_train, X_val)
y_trainval  <- c(y_train, y_val)

dtrain_cv <- xgb.DMatrix(data = X_trainval, label = y_trainval)
dtest     <- xgb.DMatrix(data = X_test,     label = y_test)

n_threads <- max(1L, parallel::detectCores() - 1L)

# ------------------------------------------------------------------------------
# 4. Random Search de 20 combinaciones
# ------------------------------------------------------------------------------
set.seed(123)
n_combinaciones <- 20

grid_random <- data.table(
  max_depth        = sample(c(4, 5, 6, 8),            n_combinaciones, replace = TRUE),
  eta              = sample(c(0.03, 0.05, 0.08, 0.1), n_combinaciones, replace = TRUE),
  subsample        = sample(c(0.7, 0.8, 0.9, 1.0),   n_combinaciones, replace = TRUE),
  colsample_bytree = sample(c(0.7, 0.8, 0.9, 1.0),   n_combinaciones, replace = TRUE),
  min_child_weight = sample(c(1, 3, 5, 10),            n_combinaciones, replace = TRUE)
)

grid_random[, cv_rmse     := as.numeric(NA)]
grid_random[, best_nrounds := as.integer(NA)]

cat(sprintf("\nIniciando Random Search (xgb.cv) sobre %d combinaciones (%d threads)...\n\n",
            n_combinaciones, n_threads))

for (i in seq_len(n_combinaciones)) {

  params <- list(
    objective        = "reg:squarederror",
    eval_metric      = "rmse",
    tree_method      = "hist",
    nthread          = n_threads,
    max_depth        = grid_random$max_depth[i],
    eta              = grid_random$eta[i],
    subsample        = grid_random$subsample[i],
    colsample_bytree = grid_random$colsample_bytree[i],
    min_child_weight = grid_random$min_child_weight[i],
    gamma            = 0,
    lambda           = 1,
    alpha            = 0
  )

  tryCatch({
    cv_result <- suppressWarnings(xgb.cv(
      params               = params,
      data                 = dtrain_cv,
      nrounds              = 500,
      nfold                = 5,           
      early_stopping_rounds = 25,
      verbose              = 0,           
      showsd               = FALSE
    ))

    log_cv <- cv_result$evaluation_log
    rmse_col <- grep("test.*mean", names(log_cv), value = TRUE)[1]

    if (!is.null(rmse_col) && !is.na(rmse_col)) {
      rmse_vals <- log_cv[[rmse_col]]
      best_pos  <- which.min(rmse_vals)
      grid_random$cv_rmse[i]     <- rmse_vals[best_pos]
      grid_random$best_nrounds[i] <- as.integer(log_cv$iter[best_pos])
    }

  }, error = function(e) {
    cat(sprintf("  Combinacion %d fallo: %s\n", i, conditionMessage(e)))
  })

  cat(sprintf(
    "  [%2d/%d] depth=%d  eta=%.2f  sub=%.1f  col=%.1f  mcw=%2d  -> cv_RMSE: %s\n",
    i, n_combinaciones,
    grid_random$max_depth[i], grid_random$eta[i],
    grid_random$subsample[i], grid_random$colsample_bytree[i],
    grid_random$min_child_weight[i],
    ifelse(is.na(grid_random$cv_rmse[i]), "NA",
           sprintf("%.4f (iter %d)", grid_random$cv_rmse[i],
                   grid_random$best_nrounds[i]))
  ))
}

# ------------------------------------------------------------------------------
# 5. Seleccion del Mejor Modelo
# ------------------------------------------------------------------------------
if (all(is.na(grid_random$cv_rmse))) stop("Ninguna combinacion produjo resultados validos.")

mejor_idx    <- which.min(grid_random$cv_rmse)
mejor_config <- grid_random[mejor_idx]

cat("\n====================================================\n")
cat("MEJOR CONFIGURACION ENCONTRADA\n")
cat("====================================================\n")
print(mejor_config[, .(max_depth, eta, subsample, colsample_bytree,
                        min_child_weight, best_nrounds,
                        cv_RMSE = cv_rmse)])

# ------------------------------------------------------------------------------
# 6. Entrenamiento Final sobre train + val combinados
# ------------------------------------------------------------------------------
cat("\nEntrenando modelo final...\n")

params_finales <- list(
  objective        = "reg:squarederror",
  eval_metric      = "rmse",
  tree_method      = "hist",
  nthread          = n_threads,
  max_depth        = mejor_config$max_depth,
  eta              = mejor_config$eta,
  subsample        = mejor_config$subsample,
  colsample_bytree = mejor_config$colsample_bytree,
  min_child_weight = mejor_config$min_child_weight,
  gamma            = 0,
  lambda           = 1,
  alpha            = 0
)

set.seed(123)
modelo_final <- suppressWarnings(xgb.train(
  params  = params_finales,
  data    = dtrain_cv,
  nrounds = mejor_config$best_nrounds,
  verbose = 0
))

# ------------------------------------------------------------------------------
# 7. Evaluacion Final en Test Set
# ------------------------------------------------------------------------------
preds_test <- predict(modelo_final, dtest)

calc_rmse <- function(y, y_hat) sqrt(mean((y - y_hat)^2))
calc_mae  <- function(y, y_hat) mean(abs(y - y_hat))
calc_r2   <- function(y, y_hat) 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)

metricas <- data.frame(
  Metrica = c("RMSE", "MAE", "R2"),
  Valor   = round(c(
    calc_rmse(y_test, preds_test),
    calc_mae(y_test,  preds_test),
    calc_r2(y_test,   preds_test)
  ), 4)
)

cat("====================================================\n")
cat("METRICAS FINALES (TEST SET) CON INGENIERIA ESPACIAL\n")
cat("====================================================\n")
print(metricas, row.names = FALSE)
cat("====================================================\n")

# ------------------------------------------------------------------------------
# 8. Importancia de Variables (Incluyendo las nuevas de distancia)
# ------------------------------------------------------------------------------
cat("\nTop 15 Variables más importantes del modelo final:\n")
importance_matrix <- xgb.importance(feature_names = colnames(X_train), model = modelo_final)
print(head(importance_matrix, 15))

# ==============================================================================
# 9. GUARDADO DE MODELO PARA LA WEB APP
# ==============================================================================
cat("\nGuardando modelo Boosting Espacial (Tuned) para la Web App...\n")
xgb.save(modelo_final, "models/modelo_boosting_espacial.model")
saveRDS(colnames(X_train), "models/columnas_boosting_espacial.rds")
cat("\nPipeline finalizado.\n")
