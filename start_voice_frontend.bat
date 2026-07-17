@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

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
    powershell -NoProfile -NonInteractive -Command ^
        "$connection = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if ($null -eq $connection) { exit 0 }; $owner = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue; if ($null -ne $owner -and $owner.ProcessName -eq 'voice_frontend') { Write-Host ('[INFO] Voice Frontend is already running on port 8765 (PID {0}).' -f $connection.OwningProcess); exit 10 }; $name = if ($null -ne $owner) { $owner.ProcessName } else { 'unknown' }; Write-Host ('[ERROR] Debug port 8765 is occupied by PID {0} ({1}).' -f $connection.OwningProcess, $name); exit 11"
    set "PORT_CHECK_EXIT=!ERRORLEVEL!"
    if "!PORT_CHECK_EXIT!"=="10" (
        echo Use the existing Voice Frontend window, or stop that process before restarting.
        pause
        exit /b 0
    )
    if not "!PORT_CHECK_EXIT!"=="0" (
        echo Stop the process using port 8765, then run this script again.
        pause
        exit /b 1
    )

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
