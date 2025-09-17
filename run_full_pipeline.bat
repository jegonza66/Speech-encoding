@echo off
setlocal enabledelayedexpansion

REM === User options: set to true or false ===
set run_load=true
set run_validation=true
set run_permutations=false
set run_main=true
set NUMOW=8

REM === Run load.py ===
if /i "%run_load%"=="true" (
    echo Running load.py...
    if PARALLEL_LOAD(
        python load.py --number_of_workers %NUMOW% --parallel_load
    ) else (
        python load.py --number_of_workers %NUMOW% --no-parallel_load
    )
    
    if errorlevel 1 (
        echo load.py failed. Exiting.
        exit /b 1
    )
)

@REM REM === Run validation.py ===
@REM if /i "%run_validation%"=="true" (
@REM     echo Running validation.py...
@REM     python validation.py --solver "ridge-laplacian"
@REM     if errorlevel 1 (
@REM         echo validation.py failed. Exiting.
@REM         exit /b 1
@REM     )
@REM )
REM === Run validation.py ===
if /i "%run_validation%"=="true" (
    echo Running validation.py...
    python validation.py --solver "ridge"
    if errorlevel 1 (
        echo validation.py failed. Exiting.
        exit /b 1
    )
)

REM === Run random_permutations.py 
if /i "%run_permutations%"=="true" (
    echo Running random_permutations.py...
    python random_permutations.py
    if errorlevel 1 (
        echo random_permutations.py failed. Exiting.
        exit /b 1
    )
)

REM === Run main.py SAME VAL, EXTERNAL VAL, RIDGE
if /i "%run_main%"=="true" (
    echo Running main.py...
    if "%run_permutations%"=="true" (
        python main.py --same_validation_subjects --external_validation --solver "ridge" --statistical_test
    ) else (
        python main.py --same_validation_subjects --external_validation --solver "ridge" --no-statistical_test
    )
    if errorlevel 1 (
        echo main.py failed. Exiting.
        exit /b 1
    )
)
@REM REM === Run main.py SAME VAL, EXTERNAL VAL, RIDGE-LAPLACIAN
@REM if /i "%run_main%"=="true" (
@REM     echo Running main.py...
@REM     if "%run_permutations%"=="true" (
@REM         python main.py --same_validation_subjects --external_validation --solver "ridge-laplacian" --statistical_test
@REM     ) else (
@REM         python main.py --same_validation_subjects --external_validation --solver "ridge-laplacian" --no-statistical_test
@REM     )
@REM     if errorlevel 1 (
@REM         echo main.py failed. Exiting.
@REM         exit /b 1
@REM     )
@REM )

@REM REM === Run main.py DIFF VAL, NO EXTERNAL VAL, RIDGE-LAPLACIAN
@REM if /i "%run_main%"=="true" (
@REM     echo Running main.py...
@REM     if "%run_permutations%"=="true" (
@REM         python main.py --no-same_validation_subjects --no-external_validation --solver "ridge-laplacian" --statistical_test
@REM     ) else (
@REM         python main.py --no-same_validation_subjects --no-external_validation --solver "ridge-laplacian" --no-statistical_test
@REM     )
@REM     if errorlevel 1 (
@REM         echo main.py failed. Exiting.
@REM         exit /b 1
@REM     )
@REM )

REM === Run main.py DIFF VAL, NO EXTERNAL VAL, RIDGE
if /i "%run_main%"=="true" (
    echo Running main.py...
    if "%run_permutations%"=="true" (
        python main.py --no-same_validation_subjects --no-external_validation --solver "ridge" --statistical_test
    ) else (
        python main.py --no-same_validation_subjects --no-external_validation --solver "ridge" --no-statistical_test
    )
    if errorlevel 1 (
        echo main.py failed. Exiting.
        exit /b 1
    )
)

echo Pipeline completed.
exit /b 0

