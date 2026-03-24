@echo off
title Premium Calculation Frontend Server
echo Starting local web server on port 8080...
echo Open http://localhost:8080 in your web browser.
start http://localhost:8080
python -m http.server 8080
pause
