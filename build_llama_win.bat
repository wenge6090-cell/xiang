@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set PROJECT_DIR=%~dp0
set LLAMA_DIR=%PROJECT_DIR%llama.cpp

echo ============================================
echo  编译 llama.cpp — Vulkan 后端 (AMD RX 6650 XT)
echo ============================================
echo.

:: 1. 克隆 llama.cpp
if not exist "%LLAMA_DIR%" (
    echo [1/4] 克隆 llama.cpp...
    cd /d "%PROJECT_DIR%"
    git clone https://github.com/ggerganov/llama.cpp
) else (
    echo [1/4] llama.cpp 已存在: %LLAMA_DIR%
)

:: 2. 检查 Vulkan SDK
if "%VULKAN_SDK%"=="" (
    echo [信息] VULKAN_SDK 环境变量未设置
    echo   请安装 Vulkan SDK: winget install KhronosGroup.VulkanSDK
    echo   或手动设置: set VULKAN_SDK=C:\VulkanSDK\1.4.350.0
)

:: 3. 创建 build 目录
echo [2/4] 配置 CMake (Vulkan)...
cd /d "%LLAMA_DIR%"
if not exist build mkdir build
cd build

:: 4. CMake 配置 + 编译
echo [3/4] CMake 配置中...
cmake .. -DLLAMA_VULKAN=1
if %errorlevel% neq 0 (
    echo ✗ CMake 配置失败（可能缺少 Vulkan SDK）
    pause
    exit /b %errorlevel%
)

echo.
echo [4/4] 编译中 (Release)... 请稍候
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo ✗ 编译失败
    pause
    exit /b %errorlevel%
)

:: 5. 验证
echo.
echo [验证] 编译产物...
if exist "%LLAMA_DIR%\build\bin\Release\llama-server.exe" (
    echo ✅ llama-server.exe 编译成功
) else (
    echo ⚠ llama-server.exe 未找到，检查 build\bin\Release\ 目录
)
if exist "%LLAMA_DIR%\build\bin\Release\ggml-vulkan.dll" (
    echo ✅ ggml-vulkan.dll 已生成 (Vulkan GPU 支持)
) else (
    echo ⚠ ggml-vulkan.dll 未生成 (Vulkan 编译失败)
)

echo.
echo ============================================
echo  编译完成！
echo.
echo  运行服务器:
echo    start_server_win.bat
echo  运行实验 (另一个终端):
echo    run_exp_win.bat --http
echo ============================================
pause
