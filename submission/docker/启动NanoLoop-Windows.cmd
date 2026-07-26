@echo off
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%NanoLoop-Control.ps1" -Action start
if errorlevel 1 (
  echo.
  echo 启动失败。请保留本窗口并按部署手册排查。
  pause
)
