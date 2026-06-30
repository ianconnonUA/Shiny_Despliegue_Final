# =======================================================
# Ajuste autom�tico del directorio de trabajo a la ra�z
if (basename(getwd()) == "scripts") setwd("..")
# =======================================================
library(data.table)

cat("Cargando el set de prueba (test_data.rds)...\n")
# Usamos el test_data porque ya pasó por el script de limpieza inicial
# y tiene exactamente el formato y las variables que los modelos esperan.
test_data <- readRDS("data/test_data.rds")

# Elegir cuántas filas queremos en nuestra muestra de prueba
cantidad_filas <- 20

cat(sprintf("Extrayendo %d filas completamente al azar...\n", cantidad_filas))
set.seed(Sys.time()) # Usar el reloj para que siempre dé una muestra distinta
muestra_aleatoria <- test_data[sample(.N, cantidad_filas)]

# Eliminar la variable objetivo (el precio) porque justamente la idea
# es que la aplicación Shiny lo adivine usando el resto de los datos.
if ("price_per_person" %in% names(muestra_aleatoria)) {
  muestra_aleatoria[, price_per_person := NULL]
}

# Guardar el resultado en un archivo CSV
archivo_salida <- "muestra_para_shiny.csv"
fwrite(muestra_aleatoria, archivo_salida)

cat("====================================================\n")
cat(sprintf("¡Éxito! Se ha creado el archivo: %s\n", archivo_salida))
cat("====================================================\n")
cat("Ya podés ir a la página web (app.R), hacer clic en 'Browse...' \n")
cat("y subir este archivo para probar la predicción por lote.\n")
