@echo off
REM ============================================================
REM  PASO 2b — Ejecutar TruFor sobre el holdout (498 imágenes)
REM  Tiempo estimado: ~5-10 min con GPU | ~60-90 min con CPU
REM ============================================================
cd /d E:\PycharmProjects\DocVerify\sota_comparison

echo.
echo [PASO 2b] Ejecutando inferencia TruFor...
echo Con GPU: ~5-10 minutos
echo Sin GPU: ~60-90 minutos (usar --no_gpu solo si no hay GPU disponible)
echo.

REM Verificar que el holdout existe
if not exist holdout_images\attack (
    echo [ERROR] No se encontro holdout_images\
    echo Ejecuta primero: python 00_export_holdout.py
    pause
    exit /b 1
)

REM Verificar que los pesos existen
if not exist trufor_weights\trufor.pth.tar (
    echo [ERROR] No se encontraron pesos trufor_weights\trufor.pth.tar
    pause
    exit /b 1
)

REM Crear directorio de salida
mkdir trufor_output 2>nul

REM Obtener rutas absolutas en formato Docker (C:\... -> /c/...)
set "IMAGES_PATH=%CD%\holdout_images"
set "OUTPUT_PATH=%CD%\trufor_output"
set "WEIGHTS_PATH=%CD%\trufor_weights"

REM Convertir rutas Windows a formato /c/... para Docker en WSL2
REM (reemplazar C:\ por /c/)
set "DRIVE=%IMAGES_PATH:~0,1%"
set "DRIVE_LOWER=%DRIVE%"
call :tolower DRIVE_LOWER

set "IMAGES_DOCKER=/%DRIVE_LOWER%/%IMAGES_PATH:~3%"
set "OUTPUT_DOCKER=/%DRIVE_LOWER%/%OUTPUT_PATH:~3%"
set "WEIGHTS_DOCKER=/%DRIVE_LOWER%/%WEIGHTS_PATH:~3%"

REM Reemplazar backslashes
set "IMAGES_DOCKER=%IMAGES_DOCKER:\=/%"
set "OUTPUT_DOCKER=%OUTPUT_DOCKER:\=/%"
set "WEIGHTS_DOCKER=%WEIGHTS_DOCKER:\=/%"

echo Imagenes : %IMAGES_DOCKER%
echo Salida   : %OUTPUT_DOCKER%
echo Pesos    : %WEIGHTS_DOCKER%
echo.

docker run --rm --gpus all ^
    -v "%IMAGES_DOCKER%:/input:ro" ^
    -v "%OUTPUT_DOCKER%:/output" ^
    -v "%WEIGHTS_DOCKER%:/weights:ro" ^
    trufor_docverify ^
    -in /input ^
    -out /output ^
    -exp trufor_ph3 ^
    TEST.MODEL_FILE "/weights/trufor.pth.tar"

if errorlevel 1 (
    echo.
    echo [INFO] Si el error es por GPU, intenta sin GPU:
    echo   Edita este bat y cambia "--gpus all" por "" (elimina esa linea)
    echo.
    echo [ERROR] Fallo en la inferencia TruFor.
    pause
    exit /b 1
)

echo.
echo [OK] Inferencia completada. Resultados en: trufor_output\
echo.
echo [SIGUIENTE] En PyCharm, ejecuta la Run Configuration: sota_01_parse_output
echo O desde terminal: python sota_comparison\01_run_trufor.py --skip_docker
pause
goto :eof

:tolower
for %%a in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do (
    set "%1=!%1:%%a=%%a!"
)
goto :eof
