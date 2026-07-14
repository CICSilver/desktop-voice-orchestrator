@echo off
setlocal

rem Always run from the project root so config, models, and web assets resolve.
cd /d "%~dp0"

set "VOICE_FRONTEND=%~dp0build\windows-x64\Release\voice_frontend.exe"

if not exist "%VOICE_FRONTEND%" (
    echo [ERROR] voice_frontend.exe was not found:
    echo         %VOICE_FRONTEND%
    echo.
    echo Build the Release preset first, then run this script again.
    pause
    exit /b 1
)

if not exist "%~dp0config\default.toml" (
    echo [ERROR] config\default.toml was not found.
    echo Run this script from a complete project checkout.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo Starting Voice Frontend live mode...
    echo Press Ctrl+C to stop.
    echo.
    "%VOICE_FRONTEND%" live
) else (
    "%VOICE_FRONTEND%" %*
)

set "VOICE_FRONTEND_EXIT=%ERRORLEVEL%"
if not "%VOICE_FRONTEND_EXIT%"=="0" (
    echo.
    echo [ERROR] Voice Frontend exited with code %VOICE_FRONTEND_EXIT%.
    pause
)

exit /b %VOICE_FRONTEND_EXIT%
