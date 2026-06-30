library(data.table)
library(xgboost)
errores <- readRDS('data/test_data.rds')
y_true <- errores$price_per_person

mod5 <- xgb.load('models/modelo_boosting_basico.model')
cols5 <- readRDS('models/columnas_boosting_basico.rds')
dt_subset <- copy(errores)
faltantes <- setdiff(cols5, names(dt_subset))
if(length(faltantes)>0) dt_subset[, (faltantes):=0]
X5 <- as.matrix(dt_subset[, cols5, with = FALSE])
mode(X5) <- 'numeric'
pred5 <- predict(mod5, xgb.DMatrix(X5))

errores[, MAE := abs(y_true - pred5)]

cat('\n=== ERROR POR TAMAÑO DEL HOTEL ===\n')
errores[, Tamanio := cut(numberOfRooms, breaks = c(0, 50, 150, 300, Inf), labels = c('Boutique/Chico (<50)', 'Mediano (50-150)', 'Grande (150-300)', 'Resort/Gigante (>300)'))]
print(errores[!is.na(Tamanio), .(N = .N, Error_Dolares = mean(MAE)), by = Tamanio][order(Tamanio)])

cat('\n=== ERROR POR CATEGORIA DE ESTRELLAS ===\n')
print(errores[, .(N = .N, Error_Dolares = mean(MAE)), by = .(Estrellas=round(starRating))][order(Estrellas)])

cat('\n=== ERROR POR AMENITIES (SPA/RELAX) ===\n')
errores[, Tiene_Spa := cat_Bienestar.y.relajación > 2]
print(errores[, .(N = .N, Error_Dolares = mean(MAE)), by = Tiene_Spa])
