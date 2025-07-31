@echo off
REM ===================================================================
REM Batch script to run validation.py and main.py with different n_folds
REM This script will test fold numbers from 2 to 75 and generate analysis
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

REM Initialize results file for correlation data
set "RESULTS_FILE=!BASE_FIGURES_DIR!\correlation_vs_folds_results.csv"
echo n_folds,subject_1_correlation,subject_2_correlation,mean_correlation,std_correlation > "!RESULTS_FILE!"

REM Counter for progress tracking
set /a "total_folds=74"
set /a "current_fold=0"

REM Loop through fold numbers from 2 to 75
for /L %%f in (2,1,75) do (
    set /a "current_fold+=1"
    set "current_fold_num=%%f"
    set "fold_dir=!BASE_FIGURES_DIR!\fold_!current_fold_num!"
    
    REM Calculate progress percentage
    set /a "progress=!current_fold! * 100 / !total_folds!"
    
    echo.
    echo [!progress!%%] Processing fold number: !current_fold_num! (!current_fold!/!total_folds!)
    echo [!progress!%%] Processing fold number: !current_fold_num! (!current_fold!/!total_folds!) >> "!LOG_FILE!"
    
    REM Create directory for this fold's results
    if not exist "!fold_dir!" mkdir "!fold_dir!"
      REM Run validation.py with current fold number
    echo   ^> Running validation with !current_fold_num! folds...
    echo   ^> Running validation with !current_fold_num! folds... >> "!LOG_FILE!"
    python validation.py --n_folds !current_fold_num! --output_dir "output/exp_fold_!current_fold_num!" --figures_dir "!fold_dir!" >> "!LOG_FILE!" 2>&1
    
    if !errorlevel! neq 0 (
        echo   ^> ERROR: validation.py failed for !current_fold_num! folds >> "!LOG_FILE!"
        echo   ^> ERROR: validation.py failed for !current_fold_num! folds
        goto :continue_loop
    )
    
    REM Run main.py with current fold number  
    echo   ^> Running main analysis with !current_fold_num! folds...
    echo   ^> Running main analysis with !current_fold_num! folds... >> "!LOG_FILE!"
    python main.py --n_folds !current_fold_num! --output_dir "output/exp_fold_!current_fold_num!" --figures_dir "!fold_dir!" >> "!LOG_FILE!" 2>&1
    
    if !errorlevel! neq 0 (
        echo   ^> ERROR: main.py failed for !current_fold_num! folds >> "!LOG_FILE!"
        echo   ^> ERROR: main.py failed for !current_fold_num! folds
        goto :continue_loop
    )
    
    :continue_loop
    
    REM Brief pause to avoid overwhelming the system
    timeout /t 1 /nobreak > nul
)

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
