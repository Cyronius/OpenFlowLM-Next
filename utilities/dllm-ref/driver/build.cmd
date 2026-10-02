@echo off
REM Configure and build dllm-ref (the diffusion-LM reference driver, main.cpp) against the
REM patched llama.cpp, Release, from a VS 2022 developer environment.
REM   GGML_VULKAN    ON (default): the Vulkan backend, which needs the LunarG Vulkan SDK
REM                  (VULKAN_SDK set by its installer). OFF: CPU only, same results, slower.
REM   LLAMA_CPP_DIR  the llama.cpp tree (default: %DLLM_ROOT%\llama.cpp-dllm)
REM   BUILD_DIR      default: build, next to this file. Delete it to switch generators.
REM   VCVARS64       vcvars64.bat (default: VS 2022 BuildTools)
REM Ninja if it is on PATH (vcvars puts VS's own there), else the VS generator.
REM Output: %BUILD_DIR%\bin\dllm-ref.exe (Ninja) or %BUILD_DIR%\bin\Release\dllm-ref.exe (VS).
setlocal
cd /d "%~dp0"
if "%VCVARS64%"=="" set "VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS64%" >nul
if errorlevel 1 goto :vcfail
where cmake >nul 2>nul
if errorlevel 1 goto :nocmake

if "%GGML_VULKAN%"=="" set "GGML_VULKAN=ON"
if /i "%GGML_VULKAN%"=="ON" if "%VULKAN_SDK%"=="" goto :nosdk
if "%BUILD_DIR%"=="" set "BUILD_DIR=build"
REM LLAMA_ARG keeps its quotes, so an unset LLAMA_CPP_DIR adds no argument at all.
set "LLAMA_ARG="
if not "%LLAMA_CPP_DIR%"=="" set LLAMA_ARG="-DLLAMA_CPP_DIR=%LLAMA_CPP_DIR%"

where ninja >nul 2>nul
if errorlevel 1 goto :vsgen
echo [dllm-ref] configure: Ninja, Release, GGML_VULKAN=%GGML_VULKAN%
cmake -S . -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=%GGML_VULKAN% %LLAMA_ARG%
if errorlevel 1 goto :cfgfail
goto :build

:vsgen
echo [dllm-ref] configure: Visual Studio 17 2022, GGML_VULKAN=%GGML_VULKAN%
cmake -S . -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 -DGGML_VULKAN=%GGML_VULKAN% %LLAMA_ARG%
if errorlevel 1 goto :cfgfail

:build
cmake --build "%BUILD_DIR%" --config Release --target dllm-ref --parallel
if errorlevel 1 goto :buildfail
echo [dllm-ref] built: %BUILD_DIR%\bin
exit /b 0

:vcfail
echo [dllm-ref] vcvars64.bat failed: set VCVARS64
exit /b 1
:nocmake
echo [dllm-ref] cmake is not on PATH: install VS's "C++ CMake tools" component or CMake itself
exit /b 1
:nosdk
echo [dllm-ref] GGML_VULKAN=ON but VULKAN_SDK is not set: install the LunarG Vulkan SDK (and open a new
echo            shell), or set GGML_VULKAN=OFF for a CPU-only build
exit /b 1
:cfgfail
echo [dllm-ref] cmake configure failed
exit /b 1
:buildfail
echo [dllm-ref] build failed
exit /b 1
