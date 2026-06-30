library(data.table)
library(xgboost)
library(dplyr)

# Ajuste automático del directorio de trabajo a la raíz
if (basename(getwd()) == "scripts") setwd("..")

# 1. Cargar test data
test_data <- readRDS("data/test_data.rds")
y_true <- test_data$price_per_person

# Función para predecir con un modelo XGBoost dado
predecir_xgb <- function(modelo_path, cols_path, dt) {
  mod <- xgb.load(modelo_path)
  cols <- readRDS(cols_path)
  
  dt_subset <- copy(dt)
  faltantes <- setdiff(cols, names(dt_subset))
  if (length(faltantes) > 0) {
    dt_subset[, (faltantes) := 0]
  }
  
  X <- as.matrix(dt_subset[, cols, with = FALSE])
  mode(X) <- "numeric"
  return(predict(mod, xgb.DMatrix(X)))
}

# 2. Generar predicciones para los Modelos 3, 4 y 5
# Nota: Asumimos que test_data ya tiene las distancias calculadas. 
# Si no, las calculamos rápido para el modelo 4.
puntos_interes <- list(
  Cristo_Redentor = c(lat = -22.9519, lon = -43.2105),
  Copacabana      = c(lat = -22.9711, lon = -43.1822),
  Ipanema         = c(lat = -22.9868, lon = -43.2024),
  Leblon          = c(lat = -22.9877, lon = -43.2219),
  Barra_da_Tijuca = c(lat = -23.0096, lon = -43.3308),
  Botafogo        = c(lat = -22.9491, lon = -43.1834),
  Flamengo        = c(lat = -22.9348, lon = -43.1725),
  Arpoador        = c(lat = -22.9892, lon = -43.1917)
  # Solo necesitamos las que el modelo 4 usa realmente.
)

calcular_haversine <- function(lat1, lon1, lat2, lon2) {
  R <- 6371
  lat1_rad <- lat1 * pi / 180; lon1_rad <- lon1 * pi / 180
  lat2_rad <- lat2 * pi / 180; lon2_rad <- lon2 * pi / 180
  dlat <- lat2_rad - lat1_rad; dlon <- lon2_rad - lon1_rad
  a <- sin(dlat/2)^2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)^2
  return(R * (2 * atan2(sqrt(a), sqrt(1-a))))
}

if (!"dist_Copacabana_km" %in% names(test_data)) {
  for (nombre in names(puntos_interes)) {
    pt <- puntos_interes[[nombre]]
    col_name <- paste0("dist_", nombre, "_km")
    test_data[, (col_name) := calcular_haversine(latitude, longitude, pt["lat"], pt["lon"])]
  }
  test_data[, dist_playa_min_km := pmin(dist_Copacabana_km, dist_Ipanema_km, dist_Leblon_km, dist_Barra_da_Tijuca_km, dist_Botafogo_km, dist_Flamengo_km, dist_Arpoador_km, na.rm=TRUE)]
}

pred_3 <- predecir_xgb("models/modelo_xgb_basico.model", "models/columnas_xgb_basico.rds", test_data)
pred_4 <- predecir_xgb("models/modelo_xgb_distancias.model", "models/columnas_xgb_distancias.rds", test_data)
pred_5 <- predecir_xgb("models/modelo_boosting_basico.model", "models/columnas_boosting_basico.rds", test_data)

# 3. Crear un dataframe consolidado de errores
errores <- data.table(
  Real = y_true,
  Estrellas = test_data$starRating,
  Rating = test_data$avgRating,
  Adultos = test_data$adults,
  Anticipacion = test_data$anticipation,
  
  Error_Abs_M3 = abs(y_true - pred_3),
  Error_Abs_M4 = abs(y_true - pred_4),
  Error_Abs_M5 = abs(y_true - pred_5),
  
  Sesgo_M3 = pred_3 - y_true,  # Positivo = Predice de mas, Negativo = Predice de menos
  Sesgo_M4 = pred_4 - y_true,
  Sesgo_M5 = pred_5 - y_true
)

# 4. Análisis por Segmentos
cat("\n=== ANALISIS DE ERRORES: MODELOS 3, 4 y 5 ===\n")

cat("\n1. ERROR MEDIO ABSOLUTO (MAE) POR RANGO DE PRECIO REAL:\n")
# Discretizamos el precio real en cuantiles
errores[, Rango_Precio := cut(Real, breaks = quantile(Real, probs = seq(0, 1, 0.2)), include.lowest = TRUE, labels = c("Muy Barato", "Barato", "Medio", "Caro", "Muy Caro"))]
agrupado_precio <- errores[, .(
  N = .N,
  MAE_M3 = mean(Error_Abs_M3),
  MAE_M4 = mean(Error_Abs_M4),
  MAE_M5 = mean(Error_Abs_M5)
), by = Rango_Precio][order(Rango_Precio)]
print(agrupado_precio)

cat("\n2. SESGO (PRED - REAL) POR RANGO DE PRECIO REAL:\n")
# ¿Está subestimando o sobreestimando?
sesgo_precio <- errores[, .(
  Sesgo_Medio_M3 = mean(Sesgo_M3),
  Sesgo_Medio_M4 = mean(Sesgo_M4),
  Sesgo_Medio_M5 = mean(Sesgo_M5)
), by = Rango_Precio][order(Rango_Precio)]
print(sesgo_precio)

cat("\n3. ERROR MEDIO ABSOLUTO (MAE) POR ESTRELLAS:\n")
errores[, Estrellas_Redondeadas := round(Estrellas)]
agrupado_estrellas <- errores[, .(
  N = .N,
  MAE_M3 = mean(Error_Abs_M3),
  MAE_M4 = mean(Error_Abs_M4),
  MAE_M5 = mean(Error_Abs_M5)
), by = Estrellas_Redondeadas][order(Estrellas_Redondeadas)]
print(agrupado_estrellas)

cat("\n4. CORRELACION DEL ERROR ABSOLUTO CON OTRAS VARIABLES (¿Qué aumenta el error?):\n")
cor_vars <- c("Real", "Estrellas", "Rating", "Adultos", "Anticipacion")
cor_matrix <- cor(errores[, ..cor_vars], errores[, .(Error_Abs_M3, Error_Abs_M4, Error_Abs_M5)], use = "pairwise.complete.obs")
print(round(cor_matrix, 3))

cat("\n=== CONCLUSION RAPIDA ===\n")
cat("- Si el sesgo es muy negativo en 'Muy Caro', significa que los modelos no logran predecir los hoteles de lujo (predicen de menos).\n")
cat("- Si la correlacion con 'Real' es alta, significa que a mayor precio, mayor es el error en dólares.\n")
