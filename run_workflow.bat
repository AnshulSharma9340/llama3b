@echo off
echo ==========================================
echo   MEDICAL NANO TUNER - AUTOMATED PIPELINE
echo ==========================================

echo.
echo [Step 1] Preparing Task Mixture...
python src/task_mixture.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

echo.
echo [Step 2] Starting QLoRA Training...
python src/train_engine.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

echo.
echo ==========================================
echo   TRAINING SUCCESSFUL! 
echo ==========================================
echo.
set /p run_inf="Do you want to run Inference now? (y/n): "
if "%run_inf%"=="y" (
    python src/inference.py
)

GOTO END

:ERROR
echo.
echo ❌ SOMETHING WENT WRONG. CHECK THE LOGS ABOVE.
pause

:END
pause