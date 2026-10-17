@echo off
REM Build kokoro_cli.exe and the open_tts unit tests, and run the tests. Host only: no XRT, no device.
setlocal
cd /d "%~dp0"
if "%VCVARS64%"=="" set "VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS64%" >nul
if errorlevel 1 (
    echo [open_tts] vcvars64 failed: "%VCVARS64%"
    exit /b 1
)
if not exist out mkdir out
set "CLFLAGS=/nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /utf-8 /D_CRT_SECURE_NO_WARNINGS /openmp /arch:AVX2 /I . /I ..\include"
set "ENGINE=safetensors.cpp ops.cpp kokoro.cpp g2p_en.cpp tts_engine.cpp"

cl %CLFLAGS% %ENGINE% cli.cpp /Fe:out\kokoro_cli.exe /Fo:out\
if errorlevel 1 goto :clfail

cl %CLFLAGS% ops_test.cpp ops.cpp safetensors.cpp kokoro.cpp /Fe:out\ops_test.exe /Fo:out\
if errorlevel 1 goto :clfail
out\ops_test.exe testdata
if errorlevel 1 goto :testfail

cl %CLFLAGS% g2p_test.cpp g2p_en.cpp /Fe:out\g2p_test.exe /Fo:out\
if errorlevel 1 goto :clfail
out\g2p_test.exe testdata
if errorlevel 1 goto :testfail

echo [open_tts] OK -^> out\kokoro_cli.exe
exit /b 0
:clfail
echo [open_tts] compile FAILED
exit /b 1
:testfail
echo [open_tts] a test FAILED
exit /b 1
