@echo off
setlocal
cd /d "%~dp0"
title Prediccion de cancer de mama

where uv >nul 2>nul
if errorlevel 1 (
    echo No se encontro uv, necesario para preparar Python 3.12.
    echo Instala uv desde https://docs.astral.sh/uv/getting-started/installation/
    echo y vuelve a abrir este archivo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Preparando el entorno de Python 3.12...
    uv venv --python 3.12 .venv
    if errorlevel 1 goto error
)

echo Comprobando e instalando dependencias...
uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
if errorlevel 1 goto error

if not exist "entrada.json" (
    ".venv\Scripts\python.exe" predecir.py --plantilla entrada.json
    if errorlevel 1 goto error
)

echo.
echo Se abrira entrada.json. Completa las 30 mediciones y guarda el archivo.
echo Cierra el Bloc de notas para ejecutar la prediccion.
start "" /wait notepad.exe "entrada.json"

echo.
echo Resultado del modelo:
".venv\Scripts\python.exe" predecir.py entrada.json
if errorlevel 1 goto error
echo.
pause
exit /b 0

:error
echo.
echo Ocurrio un error al preparar o ejecutar el modelo.
pause
exit /b 1
