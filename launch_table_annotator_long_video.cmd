@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "scripts\annotate_table_geometry.py" "data\raw\Ma-Long-and-Fan-Zhendong-Training-T2-Diamond-2019-Malaysia.mp4"
echo.
echo The table annotator has closed. Press any key to close this window.
pause >nul
