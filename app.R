library(shiny)
library(xgboost)
library(data.table)
library(leaflet)
library(glmnet)

# ==============================================================================
# 1. CARGA DE MODELOS Y ARTEFACTOS
# ==============================================================================
# Intentamos cargar los modelos generados por los scripts de entrenamiento.
modelo_simple <- tryCatch(readRDS("models/modelo_simple.rds"), error = function(e) NULL)
modelo_haversine <- tryCatch(readRDS("models/modelo_linear_haversine.rds"), error = function(e) NULL)
modelo_xgb_basico <- tryCatch(xgb.load("models/modelo_xgb_basico.model"), error = function(e) NULL)
modelo_xgb_distancias <- tryCatch(xgb.load("models/modelo_xgb_distancias.model"), error = function(e) NULL)

modelo_boosting_basico <- tryCatch(xgb.load("models/modelo_boosting_basico.model"), error = function(e) NULL)
modelo_boosting_espacial <- tryCatch(xgb.load("models/modelo_boosting_espacial.model"), error = function(e) NULL)
modelo_boosting_avanzado <- tryCatch(xgb.load("models/modelo_boosting_avanzado.model"), error = function(e) NULL)

cols_simple <- tryCatch(readRDS("models/columnas_modelo_simple.rds"), error = function(e) NULL)
cols_haversine <- tryCatch(readRDS("models/columnas_linear_haversine.rds"), error = function(e) NULL)
cols_xgb_basico <- tryCatch(readRDS("models/columnas_xgb_basico.rds"), error = function(e) NULL)
cols_xgb_distancias <- tryCatch(readRDS("models/columnas_xgb_distancias.rds"), error = function(e) NULL)

cols_boosting_basico <- tryCatch(readRDS("models/columnas_boosting_basico.rds"), error = function(e) NULL)
cols_boosting_espacial <- tryCatch(readRDS("models/columnas_boosting_espacial.rds"), error = function(e) NULL)
cols_boosting_avanzado <- tryCatch(readRDS("models/columnas_boosting_avanzado.rds"), error = function(e) NULL)

modelo_enrutador_Estandar <- tryCatch(xgb.load("models/modelo_enrutador_Estandar.model"), error = function(e) NULL)
modelo_enrutador_Lujo <- tryCatch(xgb.load("models/modelo_enrutador_Lujo_Resort.model"), error = function(e) NULL)
modelo_enrutador_Boutique <- tryCatch(xgb.load("models/modelo_enrutador_Boutique_Informal.model"), error = function(e) NULL)

cols_enrutador_Estandar <- tryCatch(readRDS("models/columnas_enrutador_Estandar.rds"), error = function(e) NULL)
cols_enrutador_Lujo <- tryCatch(readRDS("models/columnas_enrutador_Lujo_Resort.rds"), error = function(e) NULL)
cols_enrutador_Boutique <- tryCatch(readRDS("models/columnas_enrutador_Boutique_Informal.rds"), error = function(e) NULL)

centros_kmeans <- tryCatch(readRDS("models/centros_kmeans.rds"), error = function(e) NULL)
hoteles_ref <- tryCatch(readRDS("data/hoteles_referencia.rds"), error = function(e) NULL)

# Función Haversine para distancias
calcular_haversine <- function(lat1, lon1, lat2, lon2) {
  R <- 6371 
  lat1_rad <- lat1 * pi / 180; lon1_rad <- lon1 * pi / 180
  lat2_rad <- lat2 * pi / 180; lon2_rad <- lon2 * pi / 180
  dlat <- lat2_rad - lat1_rad; dlon <- lon2_rad - lon1_rad
  a <- sin(dlat/2)^2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)^2
  return(R * (2 * atan2(sqrt(a), sqrt(1-a))))
}

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

# ==============================================================================
# 2. UI (INTERFAZ DE USUARIO)
# ==============================================================================
ui <- fluidPage(
  titlePanel("🔮 Predicción de Precios de Hoteles en Río de Janeiro"),
  
  sidebarLayout(
    sidebarPanel(
      h4("🔍 Auditoría: Buscar Hotel Real"),
      selectizeInput("buscar_hotel", "Elegí un hotel para comparar:", choices = NULL),
      wellPanel(
        p("Precio Real Histórico (Persona/Noche):", style = "margin-bottom: 2px; font-size: 12px; color: #7f8c8d;"),
        h3(textOutput("precio_real_auditoria"), style = "color: #e74c3c; font-weight: bold; margin-top: 0px;")
      ),
      hr(),
      h4("Configuración del Modelo"),
      selectInput("modelo_select", "Elegir Modelo:", 
                  choices = c("1. Lineal Básico (Sin Distancias)" = "simple", 
                              "2. Lineal Espacial (Con Distancias)" = "haversine",
                              "3. XGBoost Básico (Sin Distancias)" = "xgb_basico",
                              "4. XGBoost Espacial (Con Distancias)" = "xgb_distancias",
                              "5. XGBoost Tuning Básico (Sin Distancias)" = "boosting_basico",
                              "6. XGBoost Tuning Espacial (Con Distancias)" = "boosting_espacial",
                              "7. XGBoost Tuning Avanzado (Completo)" = "boosting_avanzado",
                              "8. XGBoost Enrutador Especialista (Recomendado)" = "enrutador")),
      hr(),
      h4("Datos de la Búsqueda"),
      fileInput("archivo_csv", "Opcional: Subir CSV para Predicción Múltiple", accept = c(".csv")),
      helpText("Si subís un CSV, ignorará los controles manuales de abajo e intentará predecir para todas las filas del archivo. El CSV debe contener las variables necesarias (estrellas, lat, lon, amenities, etc.)."),
      hr(),
      h5("O ingresar manualmente:"),
      numericInput("adults", "Cantidad de Adultos:", value = 2, min = 1, max = 10),
      numericInput("children", "Cantidad de Niños:", value = 0, min = 0, max = 10),
      numericInput("stars", "Categoría (Estrellas):", value = 3, min = 1, max = 5),
      hr(),
      h5("Amenidades Principales (Top Variables):"),
      checkboxInput("amen_breakfst", "Incluye Desayuno (BREAKFST)", value = FALSE),
      checkboxInput("amen_parking", "Estacionamiento (PARKING)", value = FALSE),
      checkboxInput("amen_air", "Aire Acondicionado (AIR)", value = FALSE),
      checkboxInput("amen_pisc", "Piscina (PISC)", value = FALSE),
      checkboxInput("amen_roomsvc", "Servicio al Cuarto (ROOMSVC)", value = FALSE),
      hr(),
      h4("Ubicación Geográfica"),
      p("Haz clic en el mapa para actualizar las coordenadas, o ingresalas manualmente."),
      numericInput("latitude", "Latitud:", value = -22.9711, step = 0.0001),
      numericInput("longitude", "Longitud:", value = -43.1822, step = 0.0001),
      hr(),
      actionButton("predict_btn", "Generar Predicción", class = "btn-primary btn-lg", width = "100%")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Predicción Manual",
                 br(),
                 h3("Ubicación del Hotel"),
                 leafletOutput("mapa", height = "400px"),
                 br(),
                 h3("Resultados del Algoritmo"),
                 br(),
                 wellPanel(
                   h4("Precio Estimado por Persona (Por noche):"),
                   h1(textOutput("precio_predicho"), style = "color: #2c3e50; font-weight: bold;"),
                   h4("Precio Estimado TOTAL (Por noche):"),
                   h2(textOutput("precio_total_predicho"), style = "color: #27ae60; font-weight: bold;")
                 ),
                 br(),
                 h4("🏨 5 Hoteles Reales Similares en esa zona:"),
                 tableOutput("hoteles_similares"),
                 br(),
                 h5("Detalles del procesamiento:"),
                 verbatimTextOutput("log_proceso")
        ),
        tabPanel("Predicción Múltiple (CSV)",
                 br(),
                 h3("Resultados de la predicción por lote"),
                 p("Aquí se muestran las predicciones para el archivo subido:"),
                 dataTableOutput("tabla_lote")
        ),
        tabPanel("Importancia de Variables (Top 20)",
                 br(),
                 h3("Las 20 variables más importantes del modelo seleccionado"),
                 p("Estas son las variables que el algoritmo considera más relevantes para definir el precio:"),
                 tableOutput("tabla_importancia")
        ),
        tabPanel("Comparativa de Modelos (Métricas)",
                 br(),
                 h3("Métricas de Desempeño"),
                 p("Métricas extraídas directamente de los testeos con el set de validación puro (MAE, R2, RMSE)."),
                 tableOutput("tabla_metricas")
        )
      )
    )
  )
)

# ==============================================================================
# 3. SERVER (BACKEND Y LÓGICA DE PREDICCIÓN)
# ==============================================================================
server <- function(input, output, session) {
  
  # --- INICIALIZAR BÚSQUEDA DE HOTELES ---
  if (!is.null(hoteles_ref)) {
    updateSelectizeInput(session, "buscar_hotel", 
                         choices = c("Escribí para buscar..." = "", sort(hoteles_ref$name)), 
                         server = TRUE)
  }
  
  observeEvent(input$buscar_hotel, {
    req(input$buscar_hotel)
    hotel_data <- hoteles_ref[name == input$buscar_hotel][1]
    
    if (nrow(hotel_data) > 0) {
      updateNumericInput(session, "stars", value = round(hotel_data$stars))
      updateNumericInput(session, "latitude", value = hotel_data$latitude)
      updateNumericInput(session, "longitude", value = hotel_data$longitude)
      
      if ("BREAKFST" %in% names(hotel_data)) updateCheckboxInput(session, "amen_breakfst", value = as.logical(hotel_data$BREAKFST))
      if ("PARKING" %in% names(hotel_data)) updateCheckboxInput(session, "amen_parking", value = as.logical(hotel_data$PARKING))
      if ("AIR" %in% names(hotel_data)) updateCheckboxInput(session, "amen_air", value = as.logical(hotel_data$AIR))
      if ("PISC" %in% names(hotel_data)) updateCheckboxInput(session, "amen_pisc", value = as.logical(hotel_data$PISC))
      if ("ROOMSVC" %in% names(hotel_data)) updateCheckboxInput(session, "amen_roomsvc", value = as.logical(hotel_data$ROOMSVC))
      
      output$precio_real_auditoria <- renderText({
        paste0("$ ", round(hotel_data$precio_promedio_persona_noche, 2), " USD")
      })
    }
  })
  
  # --- LÓGICA DEL MAPA INTERACTIVO (LEAFLET) ---
  output$mapa <- renderLeaflet({
    m <- leaflet() %>%
      addProviderTiles(providers$CartoDB.Positron) %>%
      setView(lng = -43.1822, lat = -22.9711, zoom = 12) # Centrado en Río
    
    # Agregar los puntos turísticos fijos
    for (nombre in names(puntos_interes)) {
      punto <- puntos_interes[[nombre]]
      m <- addAwesomeMarkers(m, lng = punto["lon"], lat = punto["lat"], 
                             icon = awesomeIcons(icon = 'camera', markerColor = 'blue', library = 'fa'),
                             popup = paste("Punto de Referencia:", gsub("_", " ", nombre)))
    }
    
    # Agregar marcador inicial del hotel (editable)
    m <- addMarkers(m, lng = -43.1822, lat = -22.9711, layerId = "hotel_marker",
                    popup = "Ubicación seleccionada del Hotel")
    
    return(m)
  })
  
  # Capturar clics en el mapa para mover el pin y actualizar los inputs
  observeEvent(input$mapa_click, {
    click <- input$mapa_click
    
    # Actualizar cajas numéricas
    updateNumericInput(session, "latitude", value = click$lat)
    updateNumericInput(session, "longitude", value = click$lng)
    
    # Mover el pin rojo
    leafletProxy("mapa") %>%
      clearMarkers() %>%
      addMarkers(lng = click$lng, lat = click$lat, layerId = "hotel_marker",
                 popup = "Ubicación seleccionada del Hotel")
    
    # Re-dibujar los puntos de referencia porque clearMarkers() borra todo (si no filtramos por group)
    # Forma más limpia: re-agregar los fijos.
    for (nombre in names(puntos_interes)) {
      punto <- puntos_interes[[nombre]]
      leafletProxy("mapa") %>% 
        addAwesomeMarkers(lng = punto["lon"], lat = punto["lat"], 
                          icon = awesomeIcons(icon = 'camera', markerColor = 'blue', library = 'fa'),
                          popup = paste("Punto de Referencia:", gsub("_", " ", nombre)))
    }
  })
  
  # Si el usuario tipea a mano en las cajas numéricas, también movemos el pin
  observeEvent(c(input$latitude, input$longitude), {
    leafletProxy("mapa") %>%
      clearMarkers() %>%
      addMarkers(lng = input$longitude, lat = input$latitude, layerId = "hotel_marker",
                 popup = "Ubicación seleccionada del Hotel")
                 
    for (nombre in names(puntos_interes)) {
      punto <- puntos_interes[[nombre]]
      leafletProxy("mapa") %>% 
        addAwesomeMarkers(lng = punto["lon"], lat = punto["lat"], 
                          icon = awesomeIcons(icon = 'camera', markerColor = 'blue', library = 'fa'),
                          popup = paste("Punto de Referencia:", gsub("_", " ", nombre)))
    }
  })
  
  # --- LÓGICA DE PREDICCIÓN ---
  observeEvent(input$predict_btn, {
    req(input$modelo_select)
    
    # 1. Chequear que los modelos estén cargados
    if (input$modelo_select == "simple" && (is.null(modelo_simple) || is.null(cols_simple))) {
      output$precio_predicho <- renderText("Error: El modelo simple no se encuentra.")
      output$log_proceso <- renderText("Por favor ejecuta 'scripts/03_entrenar_lineal_basico.R' primero.")
      return()
    }
    if (input$modelo_select == "haversine" && (is.null(modelo_haversine) || is.null(cols_haversine))) {
      output$precio_predicho <- renderText("Error: El modelo lineal haversine no se encuentra.")
      output$log_proceso <- renderText("Por favor ejecuta 'scripts/04_entrenar_lineal_distancias.R' primero.")
      return()
    }
    if (input$modelo_select == "xgb_basico" && (is.null(modelo_xgb_basico) || is.null(cols_xgb_basico))) {
      output$precio_predicho <- renderText("Error: El modelo XGBoost Básico no se encuentra.")
      output$log_proceso <- renderText("Por favor ejecuta 'scripts/05_entrenar_xgboost_basico.R' primero.")
      return()
    }
    if (input$modelo_select == "xgb_distancias" && (is.null(modelo_xgb_distancias) || is.null(cols_xgb_distancias))) {
      output$precio_predicho <- renderText("Error: El modelo XGBoost Espacial no se encuentra.")
      output$log_proceso <- renderText("Por favor ejecuta 'scripts/06_entrenar_xgboost_distancias.R' primero.")
      return()
    }
    if (input$modelo_select == "boosting_basico" && (is.null(modelo_boosting_basico) || is.null(cols_boosting_basico))) {
      output$precio_predicho <- renderText("Error: El modelo boosting básico no se encuentra.")
      output$log_proceso <- renderText("Por favor ejecuta 'scripts/07_tuning_xgboost_basico.R' primero.")
      return()
    }
    if (input$modelo_select == "boosting_espacial" && (is.null(modelo_boosting_espacial) || is.null(cols_boosting_espacial))) {
      output$precio_predicho <- renderText("Error: El modelo boosting espacial no se encuentra.")
      output$log_proceso <- renderText("Por favor ejecuta 'scripts/08_tuning_xgboost_espacial.R' primero.")
      return()
    }
    if (input$modelo_select == "boosting_avanzado" && (is.null(modelo_boosting_avanzado) || is.null(cols_boosting_avanzado))) {
      output$precio_predicho <- renderText("Error: El modelo boosting avanzado no se encuentra.")
      output$log_proceso <- renderText("Por favor ejecuta 'scripts/09_tuning_xgboost_avanzado.R' primero.")
      return()
    }
    
    # 2. Seleccionar modelo y columnas correspondientes
    columnas <- switch(input$modelo_select,
                       "simple" = cols_simple,
                       "haversine" = cols_haversine,
                       "xgb_basico" = cols_xgb_basico,
                       "xgb_distancias" = cols_xgb_distancias,
                       "boosting_basico" = cols_boosting_basico,
                       "boosting_espacial" = cols_boosting_espacial,
                       "boosting_avanzado" = cols_boosting_avanzado,
                       "enrutador" = list(Estandar = cols_enrutador_Estandar, Lujo_Resort = cols_enrutador_Lujo, Boutique_Informal = cols_enrutador_Boutique))
                       
    modelo   <- switch(input$modelo_select,
                       "simple" = modelo_simple,
                       "haversine" = modelo_haversine,
                       "xgb_basico" = modelo_xgb_basico,
                       "xgb_distancias" = modelo_xgb_distancias,
                       "boosting_basico" = modelo_boosting_basico,
                       "boosting_espacial" = modelo_boosting_espacial,
                       "boosting_avanzado" = modelo_boosting_avanzado,
                       "enrutador" = list(Estandar = modelo_enrutador_Estandar, Lujo_Resort = modelo_enrutador_Lujo, Boutique_Informal = modelo_enrutador_Boutique))
    
    log_text <- paste0("Modelo seleccionado: ", input$modelo_select, "\n")
    
    # 3. Determinar el dataset a predecir
    es_lote <- !is.null(input$archivo_csv)
    
    if (es_lote) {
      dt_pred <- fread(input$archivo_csv$datapath)
      log_text <- paste0(log_text, "Modo Lote (CSV): Procesando ", nrow(dt_pred), " filas...\n")
      # Asegurar que todas las columnas del modelo existan, rellenar con 0 si faltan
      columnas_base <- if(input$modelo_select == "enrutador") cols_enrutador_Estandar else columnas
      cols_faltantes <- setdiff(columnas_base, names(dt_pred))
      if (length(cols_faltantes) > 0) {
        dt_pred[, (cols_faltantes) := 0]
      }
    } else {
      # 3. Crear un data.table base con todas las columnas en 0 (valores por defecto seguros)
      columnas_base <- if(input$modelo_select == "enrutador") cols_enrutador_Estandar else columnas
      dt_pred <- data.table(matrix(0, nrow = 1, ncol = length(columnas_base)))
      setnames(dt_pred, columnas_base)
      
      # 4. Asignar los valores ingresados por el usuario
      if ("adults" %in% names(dt_pred)) dt_pred[, adults := input$adults]
      if ("children" %in% names(dt_pred)) dt_pred[, children := input$children]
      if ("stars" %in% names(dt_pred)) dt_pred[, stars := input$stars]
      if ("starRating" %in% names(dt_pred)) dt_pred[, starRating := input$stars]
      if ("latitude" %in% names(dt_pred)) dt_pred[, latitude := input$latitude]
      if ("longitude" %in% names(dt_pred)) dt_pred[, longitude := input$longitude]
      
      # Asignar amenidades ingresadas
      if ("BREAKFST" %in% names(dt_pred)) dt_pred[, BREAKFST := as.numeric(input$amen_breakfst)]
      if ("PARKING" %in% names(dt_pred)) dt_pred[, PARKING := as.numeric(input$amen_parking)]
      if ("AIR" %in% names(dt_pred)) dt_pred[, AIR := as.numeric(input$amen_air)]
      if ("PISC" %in% names(dt_pred)) dt_pred[, PISC := as.numeric(input$amen_pisc)]
      if ("ROOMSVC" %in% names(dt_pred)) dt_pred[, ROOMSVC := as.numeric(input$amen_roomsvc)]
      
      # Asignar un valor estándar de habitaciones para que el Enrutador no asuma siempre que es un hotel chico
      if ("numberOfRooms" %in% names(dt_pred)) dt_pred[, numberOfRooms := 100]
      
      # VARIABLES POR DEFECTO PARA MEJORAR LA PREDICCIÓN MANUAL EN XGBOOST
      # Si no tenemos estos datos, le decimos al modelo explícitamente que "faltan"
      miss_cols <- grep("is_missing", names(dt_pred), value = TRUE)
      if (length(miss_cols) > 0) dt_pred[, (miss_cols) := 1]
      
      # Valores lógicos típicos para búsquedas (0 días arruina la predicción)
      if ("duration" %in% names(dt_pred)) dt_pred[, duration := 1]
      if ("anticipation" %in% names(dt_pred)) dt_pred[, anticipation := 15]
      if ("AR" %in% names(dt_pred)) dt_pred[, AR := 1] # Asumimos origen AR por defecto
    }
    
    # 5. INGENIERÍA DE DATOS AL VUELO
    if (input$modelo_select %in% c("haversine", "xgb_distancias", "boosting_espacial", "boosting_avanzado", "enrutador")) {
      log_text <- paste0(log_text, "Calculando Distancias Geográficas Haversine...\n")
      
      # Distancias Haversine (Aplica a modelos con distancias)
      if (all(c("latitude", "longitude") %in% names(dt_pred))) {
        for (nombre in names(puntos_interes)) {
          punto <- puntos_interes[[nombre]]
          col_nombre <- paste0("dist_", nombre, "_km")
          if (col_nombre %in% names(dt_pred)) {
            dist <- calcular_haversine(dt_pred$latitude, dt_pred$longitude, punto["lat"], punto["lon"])
            dt_pred[, (col_nombre) := dist]
          }
        }
        if ("dist_playa_min_km" %in% names(dt_pred)) {
          dt_pred[, dist_playa_min_km := pmin(dt_pred$dist_Copacabana_km, dt_pred$dist_Ipanema_km, na.rm=TRUE)]
        }
      }
    }
    
    if (input$modelo_select %in% c("boosting_avanzado", "enrutador")) {
      log_text <- paste0(log_text, "Calculando Clústeres y Variables Avanzadas...\n")
      if (all(c("adults", "children") %in% names(dt_pred))) {
        dt_pred[, is_family_trip := as.numeric(children > 0)]
        dt_pred[, capacity_density := adults + children]
      }
      if ("stars" %in% names(dt_pred)) {
        dt_pred[, is_luxury := as.numeric(stars >= 4)]
      }
      if ("starRating" %in% names(dt_pred)) {
        dt_pred[, is_luxury := as.numeric(starRating >= 4)]
      }
      if (!is.null(centros_kmeans)) {
        asignar_cluster <- function(lat, lon) {
          if (is.na(lat) || is.na(lon)) return(NA_integer_)
          dists <- (centros_kmeans[,1] - lat)^2 + (centros_kmeans[,2] - lon)^2
          return(which.min(dists))
        }
        if (all(c("latitude", "longitude") %in% names(dt_pred))) {
          dt_pred[, micro_barrio_cluster := mapply(asignar_cluster, latitude, longitude)]
        }
      }
    }
    
    # 6. Preparar Matriz Final y Predecir
    if (input$modelo_select != "enrutador") {
      dt_pred_final <- dt_pred[, columnas, with = FALSE]
      matriz_final <- as.matrix(dt_pred_final)
      mode(matriz_final) <- "numeric"
    } else {
      matriz_final <- NULL
    }
    
    # Manejar predicciones condicionalmente según el tipo de objeto (glmnet o xgboost)
    prediccion_raw <- tryCatch({
      if (input$modelo_select == "enrutador") {
        dt_copy <- copy(dt_pred)
        dt_copy[, segmento := "Estandar"]
        if ("starRating" %in% names(dt_copy) && "numberOfRooms" %in% names(dt_copy)) {
            dt_copy[starRating >= 4.5 | numberOfRooms > 300, segmento := "Lujo_Resort"]
            if ("cat_Bienestar.y.relajación" %in% names(dt_copy)) dt_copy[cat_Bienestar.y.relajación > 2, segmento := "Lujo_Resort"]
            dt_copy[segmento == "Estandar" & (starRating < 2 | numberOfRooms < 50), segmento := "Boutique_Informal"]
        }
        y_pred_final <- rep(NA_real_, nrow(dt_copy))
        for (seg in c("Estandar", "Lujo_Resort", "Boutique_Informal")) {
            mod_info <- modelo[[seg]]
            cols_info <- columnas[[seg]]
            idx <- which(dt_copy$segmento == seg)
            if (length(idx) > 0) {
               dt_sub <- dt_copy[idx]
               dt_sub[, (setdiff(cols_info, names(dt_sub))) := 0]
               X <- as.matrix(dt_sub[, cols_info, with = FALSE])
               mode(X) <- "numeric"
               y_pred_final[idx] <- expm1(predict(mod_info, xgb.DMatrix(X)))
            }
        }
        y_pred_final
      } else if (input$modelo_select %in% c("simple", "haversine")) {
        # Para glmnet, necesitamos llamar a predict con 'newx' y especificar 's' (lambda.min)
        pred_glm <- predict(modelo, s = modelo$lambda.min, newx = matriz_final)
        as.numeric(pred_glm)
      } else {
        # Para xgboost
        dpred <- xgb.DMatrix(data = matriz_final)
        predict(modelo, dpred)
      }
    }, error = function(e) {
      NULL
    })
    
    if (is.null(prediccion_raw)) {
      output$log_proceso <- renderText("Error al calcular la predicción. Asegurá que las columnas tengan valores válidos.")
      return()
    }
    
    # 7. Post-procesamiento (No hay log-transform en estos tres modelos)
    precio_persona <- prediccion_raw
    
    precio_total <- precio_persona * (input$adults + input$children)
    
    # 8. Mostrar Resultados (Modo Lote o Manual)
    if (es_lote) {
      # Si es CSV, mostrar en tabla_lote
      dt_resultados <- copy(dt_pred)
      dt_resultados[, Precio_Persona_Predicho_USD := round(precio_persona, 2)]
      
      output$tabla_lote <- renderDataTable({
        dt_resultados
      }, options = list(pageLength = 10, scrollX = TRUE))
      
      output$precio_predicho <- renderText("Ver Pestaña 'Predicción Múltiple'")
      output$precio_total_predicho <- renderText("")
      output$hoteles_similares <- renderTable({ NULL })
      
    } else {
      # Si es manual, mostrar en los paneles principales
      output$precio_predicho <- renderText({
        paste0("$ ", format(round(precio_persona, 2), nsmall = 2), " USD")
      })
      
      output$precio_total_predicho <- renderText({
        paste0("$ ", format(round(precio_total, 2), nsmall = 2), " USD")
      })
      
      output$tabla_lote <- renderDataTable({ NULL })
      
      # 9. Buscar 5 Hoteles Similares
      if (!is.null(hoteles_ref)) {
        similares <- hoteles_ref[stars == input$stars]
        if (nrow(similares) < 5) similares <- hoteles_ref[abs(stars - input$stars) <= 1]
        similares[, dist_sq := (latitude - input$latitude)^2 + (longitude - input$longitude)^2]
        top_5 <- head(similares[order(dist_sq)], 5)
        
        output$hoteles_similares <- renderTable({
          data.frame(
            Hotel = top_5$name,
            Estrellas = top_5$stars,
            `Precio Historico (Persona/Noche)` = paste0("$ ", round(top_5$precio_promedio_persona_noche, 1))
          )
        })
      } else {
        output$hoteles_similares <- renderTable({
          data.frame(Mensaje = "No se encontró hoteles_referencia.rds. Ejecutá crear_referencia.R primero.")
        })
      }
    }
    
    # 10. Actualizar Tabla de Importancia de Variables (Top 20)
    output$tabla_importancia <- renderTable({
      if (input$modelo_select %in% c("simple", "haversine")) {
        # Para glmnet sacamos los coeficientes distintos de cero
        coefs <- coef(modelo, s = modelo$lambda.min)
        imp_dt <- data.table(
          Variable = rownames(coefs),
          Coeficiente = as.numeric(coefs)
        )
        # Filtrar el intercepto y los coeficientes en cero
        imp_dt <- imp_dt[Variable != "(Intercept)" & Coeficiente != 0]
        # Ordenar por el valor absoluto del coeficiente
        imp_dt[, Impacto_Absoluto := abs(Coeficiente)]
        head(imp_dt[order(-Impacto_Absoluto), .(Variable, Coeficiente)], 20)
      } else {
        # Para xgboost
        imp <- xgb.importance(feature_names = columnas, model = modelo)
        head(imp, 20)
      }
    })
    
    # 11. Cargar CSV de métricas si existe
    output$tabla_metricas <- renderTable({
      if (file.exists("resultados/metricas_resumen+enrutado.csv")) {
        read.csv("resultados/metricas_resumen+enrutado.csv")
      } else {
        data.frame(Mensaje = "Las métricas no han sido generadas. Corré scripts/10_generar_metricas.R")
      }
    })
    
    output$log_proceso <- renderText({ log_text })
    
  })
}

# Ejecutar la App
shinyApp(ui = ui, server = server)
