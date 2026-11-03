@echo off
rem Windows build: loads the MSVC environment for THIS process only, then builds with the venv's cmake + ninja.
set "ROOT=%~dp0.."
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
"%ROOT%\.venv\Scripts\cmake.exe" -S "%ROOT%" -B "%ROOT%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="%ROOT%\.venv\Scripts\ninja.exe" || exit /b 1
"%ROOT%\.venv\Scripts\cmake.exe" --build "%ROOT%\build" || exit /b 1
