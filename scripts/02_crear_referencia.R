# =======================================================
# Ajuste autom�tico del directorio de trabajo a la ra�z
if (basename(getwd()) == "scripts") setwd("..")
# =======================================================
library(data.table)

cat("Cargando nombres de hoteles y amenidades...\n")
# hoteles.csv contiene el 'hotel_id', 'name' y las amenidades
hoteles <- fread("data/raw/hoteles.csv", select = c("hotel_id", "name", "BREAKFST", "PARKING", "AIR", "PISC", "ROOMSVC"))
hoteles <- unique(hoteles, by = "hotel_id")

cat("Cargando búsquedas de Río para extraer coordenadas, estrellas y precios...\n")
# Rio.csv contiene la info geoespacial, el rating y el precio.
rio_searches <- fread("data/raw/Rio.csv", select = c("hid", "starRating", "latitude", "longitude", "price_by_night", "adults", "children"))

# Calcular el precio por persona real
rio_searches[, price_per_person := price_by_night / (adults + children)]

# Colapsar las múltiples búsquedas en un solo valor promedio por hotel
cat("Calculando promedios por hotel...\n")
precios_promedio <- rio_searches[, .(
  precio_promedio_persona_noche = mean(price_per_person, na.rm=TRUE),
  stars = mean(starRating, na.rm=TRUE),
  latitude = mean(latitude, na.rm=TRUE),
  longitude = mean(longitude, na.rm=TRUE)
), by = hid]

cat("Cruzando datos y generando archivo de referencia...\n")
ref_df <- merge(precios_promedio, hoteles, by.x="hid", by.y="hotel_id", all.x=FALSE)
ref_df <- unique(ref_df, by="hid")

saveRDS(ref_df, "data/hoteles_referencia.rds")
cat("¡Éxito! Archivo 'hoteles_referencia.rds' creado. Ahora la App Shiny puede mostrar hoteles similares.\n")
