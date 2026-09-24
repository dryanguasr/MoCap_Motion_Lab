@echo off
cd /d "%~dp0"
echo Prueba limitada al primer intercambio. Se reanudan las correcciones guardadas.
echo Al terminar, revisaremos el diagnostico antes de procesar otros clips.
".venv\Scripts\python.exe" scripts\process_ball_tracking_real.py --manifest data\annotations\rallies\professionals.json --clip-id rally_001_part_01 --interactive
if errorlevel 1 (
    echo El procesamiento termino con un error. Revise el mensaje anterior.
) else (
    echo Sesion guardada. Los siguientes intercambios no se han procesado.
    echo Si termino el primero, el resumen y el video diagnostico estan en data\processed.
)
pause
