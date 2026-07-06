@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 校园规章制度问答AI助手
call .venv\Scripts\python run.py


pause
