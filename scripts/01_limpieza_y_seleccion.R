# =======================================================
# Ajuste automático del directorio de trabajo a la raíz
if (basename(getwd()) == "scripts") setwd("..")
# =======================================================
library(data.table)

# 1. Carga y cruce del dataset
hoteles_raw <- fread("data/raw/hoteles.csv")
rio_hotels_raw <- hoteles_raw[destination_code == "RIO" | main_city_oid == 6381]
rio_hotels_raw <- unique(rio_hotels_raw, by = "hotel_id")

rio_searches_raw <- fread("data/raw/Rio.csv")

df <- merge(rio_searches_raw, rio_hotels_raw, by.x = "hid", by.y = "hotel_id", all.x = FALSE)

# Calcular la variable objetivo price_per_person
df[, price_per_person := price_by_night / (adults + children)]

# Guardar dimensiones originales
orig_rows <- nrow(df)
orig_cols <- ncol(df)

# 2. Eliminacion por NAs y Varianza cercana a cero
na_pcts <- colMeans(is.na(df))
columnas_na_eliminar <- names(df)[na_pcts > 0.70]
df[, (columnas_na_eliminar) := NULL]

nzv_cols <- c()
for (col in names(df)) {
  if (col %in% c("price_per_person", "price_by_night", "hid", "searchid")) next
  tabla_frecuencia <- table(df[[col]], useNA = "no")
  if (length(tabla_frecuencia) > 0) {
    max_proporcion <- max(tabla_frecuencia) / sum(tabla_frecuencia)
    if (max_proporcion >= 0.95) {
      nzv_cols <- c(nzv_cols, col)
    }
  }
}
df[, (nzv_cols) := NULL]

# 3. Agrupacion de amenities por categoria
amenities_desc <- fread("amenities_descriptions.csv")
categorias <- unique(amenities_desc$amenity_category.descriptions.es)

for (cat_name in categorias) {
  if (is.na(cat_name) || cat_name == "") next
  ids_en_categoria <- amenities_desc[amenity_category.descriptions.es == cat_name, id]
  cols_a_sumar <- intersect(names(df), ids_en_categoria)
  col_categoria_nueva <- paste0("cat_", make.names(cat_name))
  
  if (length(cols_a_sumar) > 0) {
    mat_tmp <- as.matrix(df[, ..cols_a_sumar, with = FALSE])
    mat_tmp[is.na(mat_tmp)] <- 0
    df[[col_categoria_nueva]] <- rowSums(mat_tmp)
    df[, (cols_a_sumar) := NULL]
  }
}

# 4. Tratamiento de outliers en price_per_person
p1 <- quantile(df$price_per_person, 0.01, na.rm = TRUE)
p99 <- quantile(df$price_per_person, 0.99, na.rm = TRUE)
df <- df[price_per_person >= p1 & price_per_person <= p99]

# 5. Imputacion de NAs en variables numericas con mediana y creacion de dummies
columnas_na_num <- names(df)[sapply(df, function(x) is.numeric(x) && any(is.na(x)))]
columnas_na_num <- setdiff(columnas_na_num, "price_per_person")

for (col in columnas_na_num) {
  col_dummy <- paste0("is_missing_", col)
  df[[col_dummy]] <- as.numeric(is.na(df[[col]]))
  valor_mediana <- median(df[[col]], na.rm = TRUE)
  df[is.na(get(col)), (col) := valor_mediana]
}

# 6. Purga de Fugas (Data Leakage)
columnas_fuga <- c("hid", "searchid", "name", "detail", "date", "position", 
                   "price_by_night", "price_by_night_adult", "price_by_night_person", 
                   "min_query_price")

columnas_existentes <- intersect(columnas_fuga, names(df))
if (length(columnas_existentes) > 0) {
  df[, (columnas_existentes) := NULL]
}

# 7. Submuestreo a 150.000 registros para modelos
set.seed(42)
if (nrow(df) > 150000) {
  df_modelos <- df[sample(.N, 150000)]
} else {
  df_modelos <- df
}

# Guardar dataset completo limpio y el reducido para modelos
saveRDS(df, "data/dataset_limpio.rds")
saveRDS(df_modelos, "dataset_modelos_150k.rds")

# 8. Particion de Datos (Train/Val/Test sobre los 150.000)
n_modelos <- nrow(df_modelos)
idx <- sample(seq_len(n_modelos))

train_size <- floor(0.70 * n_modelos)
val_size <- floor(0.15 * n_modelos)

train_idx <- idx[1:train_size]
val_idx <- idx[(train_size + 1):(train_size + val_size)]
test_idx <- idx[(train_size + val_size + 1):n_modelos]

train_data <- df_modelos[train_idx]
val_data <- df_modelos[val_idx]
test_data <- df_modelos[test_idx]

# 9. Exportacion de los splits
saveRDS(train_data, "data/train_data.rds")
saveRDS(val_data, "data/val_data.rds")
saveRDS(test_data, "data/test_data.rds")

# 10. Resumen estructurado en consola
cat("====================================================\n")
cat("RESUMEN DE LIMPIEZA, PREPROCESAMIENTO Y PARTICION\n")
cat("====================================================\n")
cat(sprintf("Dimension Cruda Original: %d filas, %d columnas\n", orig_rows, orig_cols))
cat(sprintf("Dimension Limpia Total: %d filas, %d columnas\n", nrow(df), ncol(df)))
cat(sprintf("Variables eliminadas por exceso de NAs: %d\n", length(columnas_na_eliminar)))
cat(sprintf("Variables eliminadas por varianza cercana a cero: %d\n", length(nzv_cols)))
cat(sprintf("Variables de Data Leakage eliminadas: %d\n", length(columnas_existentes)))
cat("====================================================\n")
cat("PARTICION DE DATOS PARA MODELOS (Max 150.000)\n")
cat("====================================================\n")
train_pct <- (nrow(train_data) / n_modelos) * 100
val_pct <- (nrow(val_data) / n_modelos) * 100
test_pct <- (nrow(test_data) / n_modelos) * 100

cat(sprintf("Total de datos para modelos: %d filas\n", n_modelos))
cat(sprintf("Train set: %d filas (%.2f%%)\n", nrow(train_data), train_pct))
cat(sprintf("Validation set: %d filas (%.2f%%)\n", nrow(val_data), val_pct))
cat(sprintf("Test set: %d filas (%.2f%%)\n", nrow(test_data), test_pct))
cat("====================================================\n")
cat("Summary de price_per_person (Train set):\n")
print(summary(train_data$price_per_person))
cat("====================================================\n")
