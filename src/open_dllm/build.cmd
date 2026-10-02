@echo off
REM Build and run the diffusion-LM policy tests (no NPU, no XRT): each replays a fixture
REM recorded from the model's own reference decode loop (utilities\dllm-ref\ref_*.py).
REM   VCVARS64  vcvars64.bat (default: VS 2022 BuildTools)
setlocal
cd /d "%~dp0"
if "%VCVARS64%"=="" set "VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS64%" >nul
if errorlevel 1 goto :vcfail
if not exist out mkdir out
set "FIX=..\..\specs\open-engine\tests\fixtures"

for %%T in (sdar llada2) do (
    if not exist out\%%T mkdir out\%%T
    echo [open_dllm] %%T_test
    cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS /W4 ^
       /I ".." /I "..\include" ^
       %%T_test.cpp policy_%%T.cpp sampling.cpp /Fe:out\%%T_test.exe /Fo:out\%%T\
    if errorlevel 1 goto :clfail
    out\%%T_test.exe %FIX%\dllm_%%T.json
    if errorlevel 1 goto :testfail
)
echo [open_dllm] all policy tests pass
exit /b 0

:vcfail
echo [open_dllm] vcvars64.bat failed: set VCVARS64
exit /b 1
:clfail
echo [open_dllm] compile failed
exit /b 1
:testfail
echo [open_dllm] a policy test FAILED
exit /b 1
