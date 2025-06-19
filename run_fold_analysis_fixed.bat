@echo off
REM ===================================================================
REM Fixed Batch script to run validation.py and main.py with different n_folds
REM This script will test fold numbers from 2 to 75 and generate analysis
REM Using explicit iteration to avoid batch loop issues
REM ===================================================================

setlocal EnableDelayedExpansion

REM Set the base output directory for fold determination analysis
set "BASE_FIGURES_DIR=figures\analysis\fold_determination"

REM Create the base directory if it doesn't exist
if not exist "%BASE_FIGURES_DIR%" mkdir "%BASE_FIGURES_DIR%"

REM Log file for the batch process
set "LOG_FILE=!BASE_FIGURES_DIR!\fold_analysis_log.txt"

REM Initialize log file
echo ============================================= > "!LOG_FILE!"
echo Fold Determination Analysis - Started at: >> "!LOG_FILE!"
echo %date% %time% >> "!LOG_FILE!"
echo ============================================= >> "!LOG_FILE!"

echo Starting fold determination analysis...
echo Results will be saved in: !BASE_FIGURES_DIR!
echo Check log file: !LOG_FILE!

REM Create a simple counter method to ensure all folds are processed
set /a "fold_num=2"
set /a "total_folds=25"
set /a "current_count=0"

:fold_loop
if !fold_num! gtr 75 goto :fold_loop_end

set /a "current_count+=1"
set /a "progress=!current_count! * 100 / !total_folds!"

set "fold_dir=!BASE_FIGURES_DIR!\fold_!fold_num!"

echo.
echo [!progress!%%] Processing fold number: !fold_num! (!current_count!/!total_folds!)
echo [!progress!%%] Processing fold number: !fold_num! (!current_count!/!total_folds!) >> "!LOG_FILE!"

REM Create directory for this fold's results
if not exist "!fold_dir!" mkdir "!fold_dir!"

REM Run validation.py with current fold number
echo   ^> Running validation with !fold_num! folds...
echo   ^> Running validation with !fold_num! folds... >> "!LOG_FILE!"
python validation.py --n_folds !fold_num! --output_dir "output/exp_fold_!fold_num!" --figures_dir "!fold_dir!" >> "!LOG_FILE!" 2>&1

if !errorlevel! neq 0 (
    echo   ^> ERROR: validation.py failed for !fold_num! folds >> "!LOG_FILE!"
    echo   ^> ERROR: validation.py failed for !fold_num! folds
    goto :continue_loop
)

REM Run main.py with current fold number  
echo   ^> Running main analysis with !fold_num! folds...
echo   ^> Running main analysis with !fold_num! folds... >> "!LOG_FILE!"
python main.py --n_folds !fold_num! --output_dir "output/exp_fold_!fold_num!" --figures_dir "!fold_dir!" >> "!LOG_FILE!" 2>&1

if !errorlevel! neq 0 (
    echo   ^> ERROR: main.py failed for !fold_num! folds >> "!LOG_FILE!"
    echo   ^> ERROR: main.py failed for !fold_num! folds
    goto :continue_loop
)

echo   ^> SUCCESS: Completed fold !fold_num! analysis >> "!LOG_FILE!"
echo   ^> SUCCESS: Completed fold !fold_num! analysis

:continue_loop
REM Increment fold number and continue
set /a "fold_num+=1"

REM Brief pause to avoid overwhelming the system
timeout /t 1 /nobreak > nul

goto :fold_loop

:fold_loop_end

echo.
echo =============================================
echo Fold analysis completed!
echo =============================================
echo.
echo Running final correlation analysis...
echo Running final correlation analysis... >> "!LOG_FILE!"

REM Generate the final correlation vs folds analysis
python extract_fold_correlations.py --start_fold 2 --end_fold 75 --base_output_dir "output" --figs_dir "!BASE_FIGURES_DIR!" >> "!LOG_FILE!" 2>&1

if !errorlevel! equ 0 (
    echo.
    echo SUCCESS: Fold determination analysis completed!
    echo SUCCESS: Fold determination analysis completed! >> "!LOG_FILE!"
    echo.
    echo Results saved in: !BASE_FIGURES_DIR!
    echo Check the correlation_vs_folds_curve.png for the analysis!
) else (
    echo.
    echo ERROR: Final analysis failed. Check log file for details.
    echo ERROR: Final analysis failed. Check log file for details. >> "!LOG_FILE!"
)

echo.
echo Completed at: %date% %time% >> "!LOG_FILE!"
echo Press any key to exit...
pause > nul
