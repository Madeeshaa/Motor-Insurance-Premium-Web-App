@echo off
title Premium Calculation Backend (FastAPI)
echo Starting the Motor Insurance Premium FastAPI Server...
echo Make sure you have installed 'fastapi' and 'uvicorn' via pip.
uvicorn backend:app --reload --host localhost --port 8000
pause
