@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\prepare_rally_clips.py "data\raw\Ma-Long-and-Fan-Zhendong-Training-T2-Diamond-2019-Malaysia.mp4" --manifest data\annotations\rallies\professionals.json --review --export
if errorlevel 1 pause
