@echo off
echo ======================================================================
echo  CustomerIQ - 1-Click Git Commit & Push
echo ======================================================================
echo.
"C:\Program Files\Git\cmd\git.exe" add .
set /p commit_msg="Enter commit message (or press Enter for default): "
if "%commit_msg%"=="" set commit_msg=Update CustomerIQ project
"C:\Program Files\Git\cmd\git.exe" commit -m "%commit_msg%"
"C:\Program Files\Git\cmd\git.exe" push origin main
echo.
echo ======================================================================
echo  Success! All changes pushed to:
echo  https://github.com/riteshgujar073-coder/CustomerIQ
echo ======================================================================
pause
