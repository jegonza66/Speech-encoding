@echo off
setlocal enabledelayedexpansion

REM === User options: set to true or false ===
set run_load=true
set run_main=true

REM === Temporal shift values ===
@REM set "temporal_shifts=-0.75 -0.5 -0.3 -0.25 -0.2 -0.1 -0.078 -0.05 0 0.05 0.075 0.1 0.2 0.25 0.3 0.5 0.75"
@REM set "temporal_shifts=0.05 0.078 0.1 0.2 0.25 0.3 0.5 0.75"
set "temporal_shifts=0.2"


for %%T in (%temporal_shifts%) do (
    echo ==============================
    echo Processing temporal_shift=%%T
    echo ==============================

    set "save_dir=saves\experiments\temporal_shift_%%T"
    set "output_dir=output\experiments\temporal_shift_%%T"
    set "figure_dir=figures\experiments\temporal_shift_%%T"

    REM === Create directories if they do not exist ===
    if not exist "!save_dir!" mkdir "!save_dir!"
    if not exist "!output_dir!" mkdir "!output_dir!"
    if not exist "!figure_dir!" mkdir "!figure_dir!"

    REM === Run load.py ===
    if /i "%run_load%"=="true" (
        echo Running load.py with temporal_shift %%T...
        python load.py --temporal_shift %%T --saves_dir "!save_dir!" --output_dir "!output_dir!" --figures_dir "!figure_dir!"
        if errorlevel 1 (
            echo load.py failed for temporal_shift %%T. Exiting.
            exit /b 1
        )
    )

    REM === Run main.py SAME VAL, EXTERNAL VAL, RIDGE
    if /i "%run_main%"=="true" (
        echo Running main.py with temporal_shift %%T...
        python main.py --solver "ridge" --set_alpha 500.0 --temporal_shift %%T --saves_dir "!save_dir!" --output_dir "!output_dir!" --figures_dir "!figure_dir!"
        if errorlevel 1 (
            echo main.py failed for temporal_shift %%T. Exiting.
            exit /b 1
        )
    )
)

echo Pipeline completed.
exit /b 0
