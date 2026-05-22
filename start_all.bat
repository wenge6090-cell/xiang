@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set PROJECT_DIR=%~dp0
set LLAMA_DIR=%PROJECT_DIR%llama.cpp
set SERVER=%LLAMA_DIR%\build\bin\Release\llama-server.exe
set MODEL_DIR=%PROJECT_DIR%models
set MODEL_FILE=%MODEL_DIR%\Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
set LLAMA_PORT=8080
set CHAT_PORT=3001
set UI_PORT=5173

:: ── 检查依赖 ─────────────────────────────────

:: 1) llama-server.exe
if not exist "%SERVER%" (
    echo [错误] 找不到 llama-server.exe
    echo        请先运行 build_llama_win.bat 编译
    pause
    exit /b 1
)

:: 2) GGUF 模型
set MODEL=
if exist "%MODEL_FILE%" (
    set MODEL=%MODEL_FILE%
) else (
    for /r "%MODEL_DIR%" %%f in (*.gguf) do (
        set MODEL=%%f
        goto :model_found
    )
)
:model_found
if "%MODEL%"=="" (
    echo [错误] 在 %MODEL_DIR% 下未找到 .gguf 模型文件
    pause
    exit /b 1
)

:: 3) Rust 项目
if not exist "%PROJECT_DIR%Cargo.toml" (
    echo [错误] 找不到 Cargo.toml
    pause
    exit /b 1
)

:: 4) 前端依赖
if not exist "%PROJECT_DIR%chat-ui\node_modules" (
    echo [信息] 安装前端依赖...
    cd /d "%PROJECT_DIR%chat-ui"
    call npm install
    if errorlevel 1 (
        echo [错误] npm install 失败
        pause
        exit /b 1
    )
    cd /d "%PROJECT_DIR%"
)

:: ── 启动服务 ─────────────────────────────────

echo ============================================
echo  归藏 - 一键启动
echo ============================================
echo  1. llama.cpp server    → 新窗口 (port %LLAMA_PORT%)
echo  2. xiang-chat 后端     → 新窗口 (port %CHAT_PORT%)
echo  3. chat-ui 前端        → 新窗口 (port %UI_PORT%)
echo ============================================
echo.

:: ── 1. 启动 llama.cpp server ──
echo [1/3] 启动 llama.cpp server (Vulkan GPU)...
start "llama-server" "%SERVER%" -m "%MODEL%" --host 0.0.0.0 --port %LLAMA_PORT% -ngl 99 --threads 8 --ctx-size 65536

:: ── 2. 等待 server 就绪 ──
echo 等待 llama.cpp server 就绪...
set WAIT_COUNT=0
:wait_loop
timeout /t 2 /nobreak >nul
set /a WAIT_COUNT+=1
>nul 2>&1 curl -s http://localhost:%LLAMA_PORT%/health || goto :check_timeout
echo [OK] llama.cpp server 已就绪 (尝试 %WAIT_COUNT% 次)
goto :server_ready

:check_timeout
if %WAIT_COUNT% geq 60 (
    echo [错误] llama.cpp server 启动超时
    pause
    exit /b 1
)
goto :wait_loop

:server_ready

:: ── 3. 启动 xiang-chat 后端 ──
echo [2/3] 启动 xiang-chat 后端...
:: 使用 /d 参数设置工作目录，避免嵌套引号问题
start "xiang-chat" /d "%PROJECT_DIR%" cmd /c "cargo run -p xiang-chat"

:: 等待后端就绪（vocab 发现需要 60+ 次 tokenize 调用，预留充足时间）
echo 等待 xiang-chat 后端就绪（词汇发现中，可能需要 30-60 秒）...
set WAIT_COUNT=0
:wait_chat
timeout /t 2 /nobreak >nul
set /a WAIT_COUNT+=1
>nul 2>&1 curl -s http://localhost:%CHAT_PORT%/api/state || goto :check_chat_timeout
echo [OK] xiang-chat 后端已就绪 (尝试 %WAIT_COUNT% 次)
goto :chat_ready

:check_chat_timeout
if %WAIT_COUNT% geq 90 (
    echo [错误] xiang-chat 后端启动超时（%WAIT_COUNT% 次尝试）
    pause
    exit /b 1
)
goto :wait_chat

:chat_ready

:: ── 4. 启动 chat-ui 前端 ──
echo [3/3] 启动 chat-ui 前端...
start "chat-ui" /d "%PROJECT_DIR%chat-ui" cmd /c "npx vite --host"

echo.
echo ============================================
echo  全部服务已启动
echo ============================================
echo.
echo  前端界面: http://localhost:%UI_PORT%
echo  (Vite 自动代理 /api 到 :%CHAT_PORT%)
echo.
echo  直接访问后端 API:
echo    POST http://localhost:%CHAT_PORT%/api/raw
echo    POST http://localhost:%CHAT_PORT%/api/constrained
echo    GET  http://localhost:%CHAT_PORT%/api/state
echo    GET  http://localhost:%CHAT_PORT%/api/reset
echo.
echo  关闭窗口即可停止各服务
echo ============================================
echo.

:: 在当前窗口打开浏览器
start http://localhost:%UI_PORT%

echo 按任意键退出本窗口（服务将继续在后台运行）...
pause >nul
