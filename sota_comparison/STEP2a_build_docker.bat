@echo off
REM ============================================================
REM  PASO 2a — Construir imagen Docker de TruFor
REM  Tiempo estimado: 15-25 minutos (primera vez)
REM  Una vez construida, las siguientes veces usan caché (<1 min)
REM ============================================================
cd /d E:\PycharmProjects\DocVerify\sota_comparison

echo.
echo [PASO 2a] Construyendo imagen Docker de TruFor...
echo Primera vez: ~15-25 minutos (descarga + compilacion de mmcv-full)
echo Puedes monitorizar en Docker Desktop.
echo.

docker build -t trufor_docverify -f Dockerfile .

if errorlevel 1 (
    echo.
    echo [ERROR] Fallo al construir la imagen Docker.
    echo Posibles causas:
    echo   - Docker Desktop no esta corriendo
    echo   - Sin conexion a internet
    echo   - Error de compilacion de mmcv-full
    echo.
    echo Revisa el output anterior para mas detalles.
    pause
    exit /b 1
)

echo.
echo [OK] Imagen 'trufor_docverify' construida correctamente.
echo [SIGUIENTE] Ejecuta: STEP2b_run_inference.bat
pause
