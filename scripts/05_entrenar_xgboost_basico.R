# =======================================================
# Ajuste automático del directorio de trabajo a la raíz
if (basename(getwd()) == "scripts") setwd("..")
# =======================================================
library(data.table)
library(xgboost)

cat("Iniciando entrenamiento de XGBoost BÃ¡sico (Sin Distancias)...\n")

train_data <- readRDS("data/train_data.rds")
val_data <- readRDS("data/val_data.rds")
test_data <- readRDS("data/test_data.rds")

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
  return(as.matrix(dt_copy))
}

X_train <- prepare_matrix(train_data, target_var)
X_val   <- prepare_matrix(val_data, target_var)
X_test  <- prepare_matrix(test_data, target_var)

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

cat("\nGuardando modelo XGBoost BÃ¡sico (Sin Distancias) para la Web App...\n")
xgb.save(xgb_model, "models/modelo_xgb_basico.model")
saveRDS(colnames(X_train), "models/columnas_xgb_basico.rds")

cat("Pipeline finalizado exitosamente.\n")
