@echo off
chcp 65001 >nul

set PROJECT_DIR=%~dp0

:: 检查 Rust
where rustc >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ 未找到 Rust，请先安装: https://rustup.rs
    pause
    exit /b 1
)

echo ============================================
echo  象语言约束引擎 — 实验运行 (GPU模式)
echo ============================================
echo.
echo  用法:
echo    run_exp_win.bat                          全部实验 (Mock)
echo    run_exp_win.bat gatekeeper               指定实验
echo    run_exp_win.bat --http                   全部实验 (GPU)
echo    run_exp_win.bat gatekeeper --http        指定实验 (GPU)
echo.

set EXPERIMENT=all
set USE_HTTP=

if not "%1"=="" (
    if "%1"=="--http" (
        set USE_HTTP=--http http://localhost:8080
    ) else (
        set EXPERIMENT=%1
        if not "%2"=="" (
            if "%2"=="--http" set USE_HTTP=--http http://localhost:8080
        )
    )
)

cd /d "%PROJECT_DIR%"

echo  实验: %EXPERIMENT%
if not "%USE_HTTP%"=="" (
    echo  后端: GPU (llama.cpp Vulkan @ %USE_HTTP%)
) else (
    echo  后端: Mock (CPU)
)
echo.

cargo run --features http_backend -p xiang-experiments -- run --experiment %EXPERIMENT% %USE_HTTP%

if %errorlevel% neq 0 (
    echo.
    echo ✗ 运行失败
    pause
)
