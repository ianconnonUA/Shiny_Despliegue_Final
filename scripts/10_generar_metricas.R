# =======================================================
# Ajuste automtico del directorio de trabajo a la raz
if (basename(getwd()) == "scripts") setwd("..")
# =======================================================
library(data.table)
library(xgboost)
library(glmnet)
library(ggplot2)

# Crear directorio si no existe
dir.create("resultados", showWarnings = FALSE)

cat("Cargando test_data.rds...\n")
test_data <- readRDS("data/test_data.rds")

# Separamos el target
y_true <- test_data$price_per_person
test_data[, price_per_person := NULL]

# Coordenadas turísticas para el modelo Haversine
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

# Agregar las distancias al dataset de test (ya que el modelo Haversine las necesita)
if (all(c("latitude", "longitude") %in% names(test_data))) {
  for (nombre in names(puntos_interes)) {
    pt <- puntos_interes[[nombre]]
    col_name <- paste0("dist_", nombre, "_km")
    test_data[, (col_name) := calcular_haversine(latitude, longitude, pt["lat"], pt["lon"])]
  }
  test_data[, dist_playa_min_km := pmin(dist_Copacabana_km, dist_Ipanema_km, dist_Leblon_km, dist_Barra_da_Tijuca_km, dist_Botafogo_km, dist_Flamengo_km, dist_Arpoador_km, na.rm=TRUE)]
}

# Agregar features avanzados para el modelo 7
if (all(c("adults", "children") %in% names(test_data))) {
  test_data[, is_family_trip := as.numeric(children > 0)]
  test_data[, capacity_density := adults + children]
}
if ("stars" %in% names(test_data)) {
  test_data[, is_luxury := as.numeric(stars >= 4)]
}
centros_kmeans <- tryCatch(readRDS("models/centros_kmeans.rds"), error = function(e) NULL)
if (!is.null(centros_kmeans) && all(c("latitude", "longitude") %in% names(test_data))) {
  asignar_cluster <- function(lat, lon) {
    if (is.na(lat) || is.na(lon)) return(NA_integer_)
    dists <- (centros_kmeans[,1] - lat)^2 + (centros_kmeans[,2] - lon)^2
    return(which.min(dists))
  }
  test_data[, micro_barrio_cluster := mapply(asignar_cluster, latitude, longitude)]
}

# Función principal para evaluar modelo y generar PNGs
evaluar_modelo <- function(nombre_modelo, modelo_file, cols_file, es_xgb) {
  cat("\nEvaluando:", nombre_modelo, "...\n")
  
  if (!file.exists(modelo_file) || !file.exists(cols_file)) {
    cat("  -> ERROR: Archivos no encontrados para", nombre_modelo, "\n")
    return(NULL)
  }
  
  cols <- readRDS(cols_file)
  dt_subset <- copy(test_data)
  
  # Rellenar columnas faltantes con 0 si existieran (safety check)
  faltantes <- setdiff(cols, names(dt_subset))
  if (length(faltantes) > 0) {
    cat("  -> Advertencia: Faltan", length(faltantes), "columnas en test_data. Rellenando con 0.\n")
    dt_subset[, (faltantes) := 0]
  }
  
  X_test <- as.matrix(dt_subset[, cols, with=FALSE])
  mode(X_test) <- "numeric"
  
  # Predicción según tipo de modelo
  if (es_xgb) {
    mod <- xgb.load(modelo_file)
    y_pred <- predict(mod, xgb.DMatrix(X_test))
  } else {
    mod <- readRDS(modelo_file)
    y_pred <- as.numeric(predict(mod, s = mod$lambda.min, newx = X_test))
  }
  
  if (nombre_modelo == "7_Tuning_Avanzado") {
    y_pred <- expm1(y_pred)
  }
  
  # Cálculo de métricas
  rmse <- sqrt(mean((y_true - y_pred)^2))
  mae <- mean(abs(y_true - y_pred))
  r2 <- 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
  
  cat(sprintf("  -> RMSE: %.2f | MAE: %.2f | R2: %.4f\n", rmse, mae, r2))
  
  # ==========================================
  # GRÁFICO 1: REAL VS PREDICHO
  # ==========================================
  plot_data <- data.frame(Real = y_true, Predicho = y_pred)
  p1 <- ggplot(plot_data, aes(x = Real, y = Predicho)) +
    geom_point(alpha = 0.2, color = "dodgerblue") +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) +
    labs(title = paste("Precio Real vs Predicho -", nombre_modelo),
         subtitle = sprintf("Métricas: RMSE = %.2f | MAE = %.2f | R² = %.4f", rmse, mae, r2),
         x = "Precio Real (USD)",
         y = "Precio Predicho (USD)") +
    theme_minimal() +
    coord_cartesian(xlim = c(0, quantile(y_true, 0.95)), ylim = c(0, quantile(y_true, 0.95))) # Zoom para ocultar outliers extremos
    
  ggsave(filename = paste0("resultados/1_Dispersion_", nombre_modelo, ".png"), plot = p1, width = 8, height = 6)
  
  # ==========================================
  # GRÁFICO 2: DISTRIBUCIÓN DE ERRORES
  # ==========================================
  plot_data$Error <- plot_data$Predicho - plot_data$Real
  p2 <- ggplot(plot_data, aes(x = Error)) +
    geom_histogram(bins = 60, fill = "coral", color = "black", alpha = 0.7) +
    labs(title = paste("Distribución de los Errores -", nombre_modelo),
         subtitle = "Error = Precio Predicho - Precio Real. (Errores > 0 significan que el modelo cobró de más)",
         x = "Error (USD)",
         y = "Frecuencia") +
    theme_minimal() +
    coord_cartesian(xlim = c(-150, 150))
    
  ggsave(filename = paste0("resultados/2_Errores_", nombre_modelo, ".png"), plot = p2, width = 8, height = 6)
  
  return(data.frame(Modelo = nombre_modelo, RMSE = rmse, MAE = mae, R2 = r2))
}

# Evaluar los 6 modelos activos
res1 <- evaluar_modelo("1_Lineal_Basico", "models/modelo_simple.rds", "models/columnas_modelo_simple.rds", es_xgb = FALSE)
res2 <- evaluar_modelo("2_Lineal_Espacial", "models/modelo_linear_haversine.rds", "models/columnas_linear_haversine.rds", es_xgb = FALSE)
res3 <- evaluar_modelo("3_XGBoost_Basico", "models/modelo_xgb_basico.model", "models/columnas_xgb_basico.rds", es_xgb = TRUE)
res4 <- evaluar_modelo("4_XGBoost_Espacial", "models/modelo_xgb_distancias.model", "models/columnas_xgb_distancias.rds", es_xgb = TRUE)
res5 <- evaluar_modelo("5_Tuning_Basico", "models/modelo_boosting_basico.model", "models/columnas_boosting_basico.rds", es_xgb = TRUE)
res6 <- evaluar_modelo("6_Tuning_Espacial", "models/modelo_boosting_espacial.model", "models/columnas_boosting_espacial.rds", es_xgb = TRUE)
res7 <- evaluar_modelo("7_Tuning_Avanzado", "models/modelo_boosting_avanzado.model", "models/columnas_boosting_avanzado.rds", es_xgb = TRUE)

evaluar_enrutador <- function() {
  cat("\nEvaluando: 8_Enrutador_Especialista ...\n")
  dt_copy <- copy(test_data)
  dt_copy[, segmento := "Estandar"]
  dt_copy[starRating >= 4.5 | numberOfRooms > 300 | cat_Bienestar.y.relajación > 2, segmento := "Lujo_Resort"]
  dt_copy[segmento == "Estandar" & (starRating < 2 | numberOfRooms < 50), segmento := "Boutique_Informal"]
  
  y_pred_final <- rep(NA_real_, nrow(dt_copy))
  
  for (seg in c("Estandar", "Lujo_Resort", "Boutique_Informal")) {
    mod_file <- paste0("models/modelo_enrutador_", seg, ".model")
    cols_file <- paste0("models/columnas_enrutador_", seg, ".rds")
    if (file.exists(mod_file) && file.exists(cols_file)) {
      mod <- xgb.load(mod_file)
      cols <- readRDS(cols_file)
      idx <- which(dt_copy$segmento == seg)
      if (length(idx) > 0) {
        dt_sub <- dt_copy[idx]
        dt_sub[, (setdiff(cols, names(dt_sub))) := 0]
        X <- as.matrix(dt_sub[, cols, with = FALSE])
        char_cols <- colnames(X)[sapply(dt_sub[, cols, with=FALSE], is.character)]
        if(length(char_cols) > 0) {
          for(c in char_cols) X[, c] <- as.numeric(as.factor(X[, c]))
        }
        mode(X) <- "numeric"
        pred_log <- predict(mod, xgb.DMatrix(X))
        y_pred_final[idx] <- expm1(pred_log)
      }
    }
  }
  
  if (any(is.na(y_pred_final))) return(NULL)
  
  rmse <- sqrt(mean((y_true - y_pred_final)^2))
  mae <- mean(abs(y_true - y_pred_final))
  r2 <- 1 - sum((y_true - y_pred_final)^2) / sum((y_true - mean(y_true))^2)
  cat(sprintf("  -> RMSE: %.2f | MAE: %.2f | R2: %.4f\n", rmse, mae, r2))
  
  plot_data <- data.frame(Real = y_true, Predicho = y_pred_final)
  p1 <- ggplot(plot_data, aes(x = Real, y = Predicho)) + geom_point(alpha = 0.2, color = "dodgerblue") + geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) + labs(title = "Precio Real vs Predicho - 8_Enrutador_Especialista", subtitle = sprintf("Métricas: RMSE = %.2f | MAE = %.2f | R² = %.4f", rmse, mae, r2), x = "Precio Real (USD)", y = "Precio Predicho (USD)") + theme_minimal() + coord_cartesian(xlim = c(0, quantile(y_true, 0.95)), ylim = c(0, quantile(y_true, 0.95)))
  ggsave(filename = "resultados/1_Dispersion_8_Enrutador.png", plot = p1, width = 8, height = 6)
  
  plot_data$Error <- plot_data$Predicho - plot_data$Real
  p2 <- ggplot(plot_data, aes(x = Error)) + geom_histogram(bins = 60, fill = "coral", color = "black", alpha = 0.7) + labs(title = "Distribución de los Errores - 8_Enrutador", subtitle = "Error = Precio Predicho - Precio Real.", x = "Error (USD)", y = "Frecuencia") + theme_minimal() + coord_cartesian(xlim = c(-150, 150))
  ggsave(filename = "resultados/2_Errores_8_Enrutador.png", plot = p2, width = 8, height = 6)
  
  return(data.frame(Modelo = "8_Enrutador_Especialista", RMSE = rmse, MAE = mae, R2 = r2))
}
res8 <- evaluar_enrutador()

resultados_df <- rbind(res1, res2, res3, res4, res5, res6, res7, res8)
write.csv(resultados_df, "resultados/metricas_resumen+enrutado.csv", row.names = FALSE)
cat("\n====================================================\n")
cat("¡Métricas y Gráficos generados con éxito en la carpeta 'resultados/'!\n")
cat("====================================================\n")
