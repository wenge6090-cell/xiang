@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set PORT=8080
set PROJECT_DIR=%~dp0
set LLAMA_DIR=%PROJECT_DIR%llama.cpp
set SERVER=%LLAMA_DIR%\build\bin\Release\llama-server.exe
set MODEL_DIR=%PROJECT_DIR%models
set MODEL_FILE=%MODEL_DIR%\Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf

:: 检查 llama-server.exe
if not exist "%SERVER%" (
    echo ✗ 找不到 llama-server.exe
    echo   请先运行 build_llama_win.bat 编译
    pause
    exit /b 1
)

:: 查找 GGUF 模型
set MODEL=
if exist "%MODEL_FILE%" (
    set MODEL=%MODEL_FILE%
) else (
    for /r "%MODEL_DIR%" %%f in (*.gguf) do (
        set MODEL=%%f
        goto :found
    )
)
:found

if "%MODEL%"=="" (
    echo ✗ 在 %MODEL_DIR% 下未找到 .gguf 模型文件
    echo   请将模型放入 %MODEL_DIR%
    pause
    exit /b 1
)

echo ============================================
echo  启动 llama.cpp server (Vulkan GPU)
echo ============================================
echo  模型: %MODEL%
echo  端口: %PORT%
echo  GPU : 全部层 (-ngl 99)
echo ============================================
echo.
echo  请在 浏览器 或 另一个终端 中运行:
echo    run_exp_win.bat --http
echo.

"%SERVER%" -m "%MODEL%" --host 0.0.0.0 --port %PORT% -ngl 99 --threads 8 --ctx-size 65536

pause
