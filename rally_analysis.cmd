@echo off
"%~dp0runtime\python.exe" "%~dp0run.py" %*
exit /b %errorlevel%
