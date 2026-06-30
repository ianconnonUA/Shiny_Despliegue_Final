$ErrorActionPreference = "Stop"
$dir = "C:\Users\ianvc_utdtct5\OneDrive\Escritorio\UA\2doAño\Taller de resolucion de Problemas\Despegar -searches with R\Shiny_Despliegue_Final"


Write-Host "Creando carpetas..."
New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null
New-Item -ItemType Directory -Force -Path "scripts" | Out-Null

Write-Host "Moviendo archivos..."
Move-Item "Rio.csv", "hoteles.csv" "data\raw\" -ErrorAction SilentlyContinue
Move-Item "*.rds" "data\" -Exclude "modelo_*", "columnas_*", "centros_*" -ErrorAction SilentlyContinue
Move-Item "modelo_*", "columnas_*", "centros_*" "models\" -ErrorAction SilentlyContinue
Move-Item "*.R" "scripts\" -Exclude "app.R", "organizar.ps1" -ErrorAction SilentlyContinue

function Replace-StringInFile {
    param([string]$path, [string]$search, [string]$replace)
    if (Test-Path $path) {
        $content = Get-Content $path -Raw
        $newContent = [regex]::Replace($content, [regex]::Escape($search), $replace)
        if ($content -ne $newContent) {
            Set-Content $path -Value $newContent -NoNewline
        }
    }
}

Write-Host "Actualizando rutas en app.R..."
Replace-StringInFile "app.R" '"modelo_' '"models/modelo_'
Replace-StringInFile "app.R" '"columnas_' '"models/columnas_'
Replace-StringInFile "app.R" '"centros_' '"models/centros_'
Replace-StringInFile "app.R" '"hoteles_referencia.rds"' '"data/hoteles_referencia.rds"'

# Renombrar scripts para darle orden lógico
Write-Host "Renombrando y ordenando scripts..."
Rename-Item "scripts\limpieza_y_seleccion.R" "01_limpieza_y_seleccion.R" -ErrorAction SilentlyContinue
Rename-Item "scripts\crear_referencia.R" "02_crear_referencia.R" -ErrorAction SilentlyContinue
Rename-Item "scripts\entrenar_lineal_basico.R" "03_entrenar_lineal_basico.R" -ErrorAction SilentlyContinue
Rename-Item "scripts\entrenar_lineal_distancias.R" "04_entrenar_lineal_distancias.R" -ErrorAction SilentlyContinue
Rename-Item "scripts\entrenar_xgboost_basico.R" "05_entrenar_xgboost_basico.R" -ErrorAction SilentlyContinue
Rename-Item "scripts\entrenar_xgboost_distancias.R" "06_entrenar_xgboost_distancias.R" -ErrorAction SilentlyContinue
Rename-Item "scripts\random_search_xgboost.R" "07_tuning_xgboost_basico.R" -ErrorAction SilentlyContinue
Rename-Item "scripts\random_search_xgboost_con_distancias.R" "08_tuning_xgboost_espacial.R" -ErrorAction SilentlyContinue
Rename-Item "scripts\random_search_xgboost_avanzado.R" "09_tuning_xgboost_avanzado.R" -ErrorAction SilentlyContinue
Rename-Item "scripts\generar_metricas_modelos.R" "10_generar_metricas.R" -ErrorAction SilentlyContinue
Rename-Item "scripts\generar_muestra_test.R" "11_generar_muestra_csv.R" -ErrorAction SilentlyContinue

Write-Host "Actualizando mensajes de error de scripts en app.R..."
Replace-StringInFile "app.R" "entrenar_lineal_basico.R" "scripts/03_entrenar_lineal_basico.R"
Replace-StringInFile "app.R" "entrenar_lineal_distancias.R" "scripts/04_entrenar_lineal_distancias.R"
Replace-StringInFile "app.R" "entrenar_xgboost_basico.R" "scripts/05_entrenar_xgboost_basico.R"
Replace-StringInFile "app.R" "entrenar_xgboost_distancias.R" "scripts/06_entrenar_xgboost_distancias.R"
Replace-StringInFile "app.R" "random_search_xgboost.R" "scripts/07_tuning_xgboost_basico.R"
Replace-StringInFile "app.R" "random_search_xgboost_con_distancias.R" "scripts/08_tuning_xgboost_espacial.R"
Replace-StringInFile "app.R" "generar_metricas_modelos.R" "scripts/10_generar_metricas.R"

Write-Host "Actualizando rutas en los scripts .R..."
Get-ChildItem -Path "scripts\*.R" | ForEach-Object {
    $f = $_.FullName
    Replace-StringInFile $f '"train_data.rds"' '"data/train_data.rds"'
    Replace-StringInFile $f '"val_data.rds"' '"data/val_data.rds"'
    Replace-StringInFile $f '"test_data.rds"' '"data/test_data.rds"'
    Replace-StringInFile $f '"hoteles_referencia.rds"' '"data/hoteles_referencia.rds"'
    Replace-StringInFile $f '"Rio.csv"' '"data/raw/Rio.csv"'
    Replace-StringInFile $f '"hoteles.csv"' '"data/raw/hoteles.csv"'
    Replace-StringInFile $f '"dataset_limpio.rds"' '"data/dataset_limpio.rds"'
    
    Replace-StringInFile $f '"modelo_' '"models/modelo_'
    Replace-StringInFile $f '"columnas_' '"models/columnas_'
    Replace-StringInFile $f '"centros_' '"models/centros_'
}

Write-Host "¡Organización completada!"
