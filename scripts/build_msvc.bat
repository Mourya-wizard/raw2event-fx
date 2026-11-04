@echo off
rem Windows build: loads the MSVC environment for THIS process only, then builds with the venv's cmake + ninja.
rem Finds any Visual Studio / Build Tools install (2019+) with the C++ workload via vswhere.
set "ROOT=%~dp0.."
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
rem Git Bash does not pass ProgramFiles(x86) through to cmd
if not exist "%VSWHERE%" set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSDIR="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (echo No Visual Studio C++ toolset found. Install "Desktop development with C++" or VS Build Tools. & exit /b 1)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
"%ROOT%\.venv\Scripts\cmake.exe" -S "%ROOT%" -B "%ROOT%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="%ROOT%\.venv\Scripts\ninja.exe" || exit /b 1
"%ROOT%\.venv\Scripts\cmake.exe" --build "%ROOT%\build" || exit /b 1
