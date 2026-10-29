@echo off
chcp 65001 > nul

rem Folder with the GGUF models; llama-server.exe must be on PATH
set "MODELS=D:\ML Enviroment\models"

start "Embedder :5002" llama-server.exe ^
  -m "%MODELS%\user-bge-m3-q8_0.gguf" ^
  --embedding --pooling cls -ngl 99 -c 8192 ^
  -b 8192 -ub 8192 ^
  --port 5002

timeout /t 10 /nobreak > nul

start "Reranker :5003" llama-server.exe ^
  -m "%MODELS%\bge-reranker-v2-m3-Q8_0.gguf" ^
  --reranking --pooling rank -ngl 99 -c 8192 ^
  -b 2048 -ub 2048 ^
  --port 5003

pause