# =======================================================
# Ajuste automtico del directorio de trabajo a la raz
if (basename(getwd()) == "scripts") setwd("..")
# =======================================================
library(data.table)
library(xgboost)
library(stats)

set.seed(123)

# ==============================================================================
# 1. CARGA DE DATOS Y REDUCCIÓN DE MUESTRA (15k)
# ==============================================================================
cat("Cargando datos...\n")
train_data <- readRDS("data/train_data.rds")
val_data   <- readRDS("data/val_data.rds")
test_data  <- readRDS("data/test_data.rds")

# ==============================================================================
# OPCIONAL: REDUCCIÓN DE TAMAÑO MUESTRAL A ~15.000 DATOS (Para pruebas rápidas)
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

# ==============================================================================
# 2. INGENIERÍA DE DATOS AVANZADA
# ==============================================================================
cat("\nAplicando Ingeniería de Datos Avanzada...\n")

target_var <- "price_per_person"

# 2.1 TRANSFORMACIÓN LOGARÍTMICA DEL PRECIO (Reduce influencia de outliers extremos)
# Usamos log1p(x) que es log(x + 1) por seguridad matemática.
y_train <- log1p(train_data[[target_var]])
y_val   <- log1p(val_data[[target_var]])
y_test  <- log1p(test_data[[target_var]])

# Removemos el target de los data.tables
train_data[, (target_var) := NULL]
val_data[, (target_var) := NULL]
test_data[, (target_var) := NULL]

# Función maestra para aplicar transformaciones a un dataset
aplicar_ingenieria <- function(dt, centros_kmeans = NULL) {
  dt_copy <- copy(dt)
  
  # A. INTERACCIONES PSICOLÓGICAS Y DE NEGOCIO
  if (all(c("adults", "children") %in% names(dt_copy))) {
    dt_copy[, is_family_trip := as.numeric(children > 0)]
    dt_copy[, capacity_density := adults + children]
  }
  
  if ("stars" %in% names(dt_copy)) {
    dt_copy[, is_luxury := as.numeric(stars >= 4)]
  }
  
  # B. DISTANCIAS GEOGRÁFICAS (HAVERSINE)
  if (all(c("latitude", "longitude") %in% names(dt_copy))) {
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
      R <- 6371 
      lat1_rad <- lat1 * pi / 180; lon1_rad <- lon1 * pi / 180
      lat2_rad <- lat2 * pi / 180; lon2_rad <- lon2 * pi / 180
      dlat <- lat2_rad - lat1_rad; dlon <- lon2_rad - lon1_rad
      a <- sin(dlat/2)^2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)^2
      return(R * (2 * atan2(sqrt(a), sqrt(1-a))))
    }
    
    for (nombre in names(puntos_interes)) {
      punto <- puntos_interes[[nombre]]
      col_nombre <- paste0("dist_", nombre, "_km")
      dt_copy[, (col_nombre) := calcular_haversine(latitude, longitude, punto["lat"], punto["lon"])]
    }
    dt_copy[, dist_playa_min_km := pmin(dist_Copacabana_km, dist_Ipanema_km, dist_Leblon_km, dist_Barra_da_Tijuca_km, dist_Botafogo_km, dist_Flamengo_km, dist_Arpoador_km, na.rm=TRUE)]
    
    # C. MICRO-ZONIFICACIÓN ESPACIAL (CLUSTERING)
    # Si estamos en entrenamiento, calculamos los centroides.
    # Si estamos en val/test, asignamos al centroide más cercano.
    if (is.null(centros_kmeans)) {
      coords_clean <- na.omit(dt_copy[, .(latitude, longitude)])
      if (nrow(coords_clean) > 0) {
        set.seed(42)
        km_model <- kmeans(coords_clean, centers = 30, iter.max = 50)
        centros_kmeans <- km_model$centers
        
        # Asignar cluster
        dt_copy[, micro_barrio_cluster := kmeans(dt_copy[, .(latitude, longitude)], centers = centros_kmeans)$cluster]
      }
    } else {
      # Función para asignar cluster basado en distancia euclidiana a centroides pre-calculados
      asignar_cluster <- function(lat, lon) {
        if (is.na(lat) || is.na(lon)) return(NA_integer_)
        dists <- (centros_kmeans[,1] - lat)^2 + (centros_kmeans[,2] - lon)^2
        return(which.min(dists))
      }
      dt_copy[, micro_barrio_cluster := mapply(asignar_cluster, latitude, longitude)]
    }
  }
  
  return(list(data = dt_copy, centros = centros_kmeans))
}

cat("Procesando Train set...\n")
res_train <- aplicar_ingenieria(train_data, centros_kmeans = NULL)
train_data <- res_train$data
centros_guardados <- res_train$centros

cat("Procesando Validation y Test sets...\n")
val_data   <- aplicar_ingenieria(val_data, centros_kmeans = centros_guardados)$data
test_data  <- aplicar_ingenieria(test_data, centros_kmeans = centros_guardados)$data


# ==============================================================================
# 3. PREPARACIÓN DE MATRICES XGBOOST
# ==============================================================================
prepare_matrix <- function(dt) {
  dt_copy <- copy(dt)
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

X_train <- prepare_matrix(train_data)
X_val   <- prepare_matrix(val_data)
X_test  <- prepare_matrix(test_data)

X_trainval <- rbind(X_train, X_val)
y_trainval  <- c(y_train, y_val)

dtrain_cv <- xgb.DMatrix(data = X_trainval, label = y_trainval)
dtest     <- xgb.DMatrix(data = X_test,     label = y_test)

n_threads <- max(1L, parallel::detectCores() - 1L)

# ==============================================================================
# 4. RANDOM SEARCH (Predecir LOG-PRICE)
# ==============================================================================
set.seed(123)
n_combinaciones <- 20
grid_random <- data.table(
  max_depth        = sample(c(4, 5, 6, 8),            n_combinaciones, replace = TRUE),
  eta              = sample(c(0.03, 0.05, 0.08, 0.1), n_combinaciones, replace = TRUE),
  subsample        = sample(c(0.7, 0.8, 0.9, 1.0),   n_combinaciones, replace = TRUE),
  colsample_bytree = sample(c(0.7, 0.8, 0.9, 1.0),   n_combinaciones, replace = TRUE),
  min_child_weight = sample(c(1, 3, 5, 10),            n_combinaciones, replace = TRUE)
)
grid_random[, cv_rmse_log := as.numeric(NA)]
grid_random[, best_nrounds := as.integer(NA)]

cat(sprintf("\nIniciando Tuning XGBoost sobre %d combinaciones (optimizando RMSLE)...\n\n", n_combinaciones))

for (i in seq_len(n_combinaciones)) {
  params <- list(
    objective        = "reg:squarederror",
    eval_metric      = "rmse", # RMSE sobre log(precio) equivale a RMSLE
    tree_method      = "hist",
    nthread          = n_threads,
    max_depth        = grid_random$max_depth[i],
    eta              = grid_random$eta[i],
    subsample        = grid_random$subsample[i],
    colsample_bytree = grid_random$colsample_bytree[i],
    min_child_weight = grid_random$min_child_weight[i],
    gamma            = 0, lambda = 1, alpha = 0
  )

  tryCatch({
    cv_result <- suppressWarnings(xgb.cv(
      params = params, data = dtrain_cv, nrounds = 500, nfold = 5,           
      early_stopping_rounds = 25, verbose = 0, showsd = FALSE
    ))
    
    log_cv <- cv_result$evaluation_log
    rmse_col <- grep("test.*mean", names(log_cv), value = TRUE)[1]
    
    if (!is.null(rmse_col) && !is.na(rmse_col)) {
      rmse_vals <- log_cv[[rmse_col]]
      best_pos  <- which.min(rmse_vals)
      grid_random$cv_rmse_log[i] <- rmse_vals[best_pos]
      grid_random$best_nrounds[i] <- as.integer(log_cv$iter[best_pos])
    }
  }, error = function(e) { cat(sprintf("  Comb %d fallo: %s\n", i, conditionMessage(e))) })

  cat(sprintf("  [%2d/%d] depth=%d eta=%.2f sub=%.1f col=%.1f -> cv_RMSLE: %s\n",
    i, n_combinaciones, grid_random$max_depth[i], grid_random$eta[i], grid_random$subsample[i], grid_random$colsample_bytree[i],
    ifelse(is.na(grid_random$cv_rmse_log[i]), "NA", sprintf("%.4f", grid_random$cv_rmse_log[i]))))
}

# ==============================================================================
# 5. ENTRENAMIENTO FINAL Y EVALUACIÓN
# ==============================================================================
mejor_idx    <- which.min(grid_random$cv_rmse_log)
mejor_config <- grid_random[mejor_idx]

cat("\nEntrenando modelo final maestro...\n")
params_finales <- list(
  objective = "reg:squarederror", eval_metric = "rmse", tree_method = "hist", nthread = n_threads,
  max_depth = mejor_config$max_depth, eta = mejor_config$eta, subsample = mejor_config$subsample,
  colsample_bytree = mejor_config$colsample_bytree, min_child_weight = mejor_config$min_child_weight
)

set.seed(123)
modelo_final <- suppressWarnings(xgb.train(params = params_finales, data = dtrain_cv, nrounds = mejor_config$best_nrounds, verbose = 0))

# PREDICCIÓN (El modelo devuelve el LOG del precio)
preds_log_test <- predict(modelo_final, dtest)

# TRANSFORMACIÓN INVERSA (expm1 para revertir log1p y obtener dólares reales)
preds_test_reales <- expm1(preds_log_test)
y_test_reales     <- expm1(y_test)

calc_rmse <- function(y, y_hat) sqrt(mean((y - y_hat)^2))
calc_mae  <- function(y, y_hat) mean(abs(y - y_hat))
calc_r2   <- function(y, y_hat) 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)

metricas <- data.frame(
  Metrica = c("RMSE (Real $)", "MAE (Real $)", "R2"),
  Valor   = round(c(
    calc_rmse(y_test_reales, preds_test_reales),
    calc_mae(y_test_reales,  preds_test_reales),
    calc_r2(y_test_reales,   preds_test_reales)
  ), 4)
)

cat("\n====================================================\n")
cat("MÉTRICAS FINALES (DE VUELTA EN DÓLARES REALES)\n")
cat("====================================================\n")
print(metricas, row.names = FALSE)

cat("\n====================================================\n")
cat("IMPORTANCIA DE VARIABLES (Nuevos Features Incluidos)\n")
cat("====================================================\n")
print(head(xgb.importance(feature_names = colnames(X_train), model = modelo_final), 15))

# ==============================================================================
# 6. GUARDADO DE MODELO AVANZADO PARA LA WEB APP
# ==============================================================================
cat("\nGuardando modelo Maestro y artefactos espaciales para la Web App...\n")
xgb.save(modelo_final, "models/modelo_boosting_avanzado.model")
saveRDS(centros_guardados, "models/centros_kmeans.rds")
saveRDS(colnames(X_train), "models/columnas_boosting_avanzado.rds")
cat("\nPipeline maestro finalizado. Modelos listos para Producción.\n")
