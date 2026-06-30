source('app.R')
library(data.table)
library(xgboost)
cols_enrutador_Estandar <- readRDS('models/columnas_enrutador_Estandar.rds')
dt_pred <- data.table(matrix(0, nrow = 1, ncol = length(cols_enrutador_Estandar)))
setnames(dt_pred, cols_enrutador_Estandar)
dt_pred[, starRating := 4]
dt_pred[, numberOfRooms := 100]
dt_pred[, latitude := -22.9]
dt_pred[, longitude := -43.2]

dt_copy <- copy(dt_pred)
dt_copy[, segmento := 'Estandar']
if ('starRating' %in% names(dt_copy) && 'numberOfRooms' %in% names(dt_copy)) {
  dt_copy[starRating >= 4.5 | numberOfRooms > 300, segmento := 'Lujo_Resort']
  dt_copy[segmento == 'Estandar' & (starRating < 2 | numberOfRooms < 50), segmento := 'Boutique_Informal']
}
seg <- 'Estandar'
mod_info <- xgb.load('models/modelo_enrutador_Estandar.model')
cols_info <- readRDS('models/columnas_enrutador_Estandar.rds')
idx <- which(dt_copy$segmento == seg)
dt_sub <- dt_copy[idx]
dt_sub[, (setdiff(cols_info, names(dt_sub))) := 0]

X <- as.matrix(dt_sub[, cols_info, with = FALSE])
mode(X) <- 'numeric'
predict(mod_info, xgb.DMatrix(X))
