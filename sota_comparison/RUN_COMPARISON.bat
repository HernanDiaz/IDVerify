@echo off
REM ============================================================
REM  DocVerify — TruFor Comparison Pipeline
REM  Haz doble clic o ejecuta desde terminal en:
REM    E:\PycharmProjects\DocVerify\
REM ============================================================

cd /d E:\PycharmProjects\DocVerify

echo.
echo ============================================================
echo  DocVerify SOTA Comparison — TruFor vs DocVerify
echo ============================================================
echo.

REM Activar entorno virtual
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [OK] Entorno virtual activado
) else (
    echo [WARN] No se encontro venv\ - usando Python del sistema
)

echo.
echo [PASO 1/3] Exportando holdout de FantasyID...
echo ------------------------------------------------------------
python sota_comparison\00_export_holdout.py
if errorlevel 1 (
    echo [ERROR] Fallo en paso 1. Revisa el output anterior.
    pause
    exit /b 1
)

echo.
echo [PASO 2/3] Configurando y ejecutando TruFor...
echo ------------------------------------------------------------
echo NOTA: Este paso necesita Docker. Si no esta instalado,
echo       ve a https://www.docker.com/products/docker-desktop/
echo.
python sota_comparison\01_run_trufor.py
if errorlevel 1 (
    echo [ERROR] Fallo en paso 2. Revisa el output anterior.
    echo Posibles causas:
    echo   - Docker no instalado o no corriendo
    echo   - Sin conexion a internet para descargar pesos
    echo   - Error al construir la imagen Docker
    pause
    exit /b 1
)

echo.
echo [PASO 3/3] Calculando metricas y generando tabla comparativa...
echo ------------------------------------------------------------
python sota_comparison\02_eval_comparison.py
if errorlevel 1 (
    echo [ERROR] Fallo en paso 3. Revisa el output anterior.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  COMPLETADO
echo ============================================================
echo  Resultados en:
echo    sota_comparison\trufor_metrics.csv
echo    sota_comparison\comparison_table.csv
echo    sota_comparison\comparison_table.tex   ^<-- para el paper
echo ============================================================
echo.
pause
