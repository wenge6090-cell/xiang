@echo off
echo 停止 llama-server...
taskkill /f /im llama-server.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ 服务器已停止
) else (
    echo ⚠ 服务器未运行
)
pause
