# =======================================================
# Ajuste automtico del directorio de trabajo a la raz
if (basename(getwd()) == "scripts") setwd("..")
# =======================================================
library(data.table)
library(xgboost)

cat("Iniciando entrenamiento de XGBoost Espacial (Con Distancias)...\n")

train_data <- readRDS("data/train_data.rds")
val_data <- readRDS("data/val_data.rds")
test_data <- readRDS("data/test_data.rds")

# Coordenadas
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

agregar_distancias <- function(dt) {
  if (all(c("latitude", "longitude") %in% names(dt))) {
    for (nombre in names(puntos_interes)) {
      pt <- puntos_interes[[nombre]]
      col_nombre <- paste0("dist_", nombre, "_km")
      dt[, (col_nombre) := calcular_haversine(latitude, longitude, pt["lat"], pt["lon"])]
    }
    dt[, dist_playa_min_km := pmin(dist_Copacabana_km, dist_Ipanema_km, dist_Leblon_km, dist_Barra_da_Tijuca_km, dist_Botafogo_km, dist_Flamengo_km, dist_Arpoador_km, na.rm=TRUE)]
  }
  return(dt)
}

train_data <- agregar_distancias(train_data)
val_data   <- agregar_distancias(val_data)
test_data  <- agregar_distancias(test_data)

target_var <- "price_per_person"

y_train <- train_data[[target_var]]
y_val   <- val_data[[target_var]]

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

X_train <- prepare_matrix(train_data, target_var)
X_val   <- prepare_matrix(val_data, target_var)

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dval   <- xgb.DMatrix(data = X_val, label = y_val)

xgb_params <- list(
  objective = "reg:squarederror",
  eta = 0.05,
  max_depth = 6
)

set.seed(42)
xgb_model <- suppressWarnings(xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 1500,
  watchlist = list(train = dtrain, validation = dval),
  early_stopping_rounds = 15,
  verbose = 0
))

cat("\nGuardando modelo XGBoost Espacial (Con Distancias) para la Web App...\n")
xgb.save(xgb_model, "models/modelo_xgb_distancias.model")
saveRDS(colnames(X_train), "models/columnas_xgb_distancias.rds")

cat("Pipeline finalizado exitosamente.\n")
