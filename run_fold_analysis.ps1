# ===================================================================
# PowerShell script to run validation.py and main.py with different n_folds
# This script will test fold numbers from 2 to 75 and generate analysis
# ===================================================================

param(
    [int]$StartFold = 2,
    [int]$EndFold = 75,
    [string]$BaseDir = "figures\analysis\fold_determination"
)

# Set error action preference
$ErrorActionPreference = "Continue"

# Create the base directory if it doesn't exist
if (!(Test-Path $BaseDir)) {
    New-Item -ItemType Directory -Path $BaseDir -Force | Out-Null
}

# Log file for the batch process
$LogFile = Join-Path $BaseDir "fold_analysis_log.txt"

# Initialize log file
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
@"
=============================================
Fold Determination Analysis - Started at: $timestamp
=============================================
"@ | Out-File -FilePath $LogFile -Encoding UTF8

Write-Host "Starting fold determination analysis..." -ForegroundColor Green
Write-Host "Results will be saved in: $BaseDir" -ForegroundColor Yellow
Write-Host "Check log file: $LogFile" -ForegroundColor Yellow

# Initialize results file for correlation data
$ResultsFile = Join-Path $BaseDir "correlation_vs_folds_results.csv"
"n_folds,subject_1_correlation,subject_2_correlation,mean_correlation,std_correlation" | Out-File -FilePath $ResultsFile -Encoding UTF8

# Counter for progress tracking
$totalFolds = $EndFold - $StartFold + 1
$currentFold = 0

# Loop through fold numbers
for ($f = $StartFold; $f -le $EndFold; $f++) {
    $currentFold++
    $foldDir = Join-Path $BaseDir "fold_$f"
    
    # Calculate progress percentage
    $progress = [int](($currentFold * 100) / $totalFolds)
    
    Write-Host ""
    Write-Host "[$progress%] Processing fold number: $f ($currentFold/$totalFolds)" -ForegroundColor Cyan
    "[$progress%] Processing fold number: $f ($currentFold/$totalFolds)" | Add-Content -Path $LogFile
    
    # Create directory for this fold's results
    if (!(Test-Path $foldDir)) {
        New-Item -ItemType Directory -Path $foldDir -Force | Out-Null
    }
    
    # Run validation.py with current fold number
    Write-Host "  > Running validation with $f folds..." -ForegroundColor Gray
    "  > Running validation with $f folds..." | Add-Content -Path $LogFile
    
    try {
        $validationResult = & python validation.py --n_folds $f 2>&1
        $validationResult | Add-Content -Path $LogFile
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  > ERROR: validation.py failed for $f folds" -ForegroundColor Red
            "  > ERROR: validation.py failed for $f folds" | Add-Content -Path $LogFile
            continue
        }
    }
    catch {
        Write-Host "  > ERROR: validation.py failed for $f folds - $_" -ForegroundColor Red
        "  > ERROR: validation.py failed for $f folds - $_" | Add-Content -Path $LogFile
        continue
    }
    
    # Run main.py with current fold number  
    Write-Host "  > Running main analysis with $f folds..." -ForegroundColor Gray
    "  > Running main analysis with $f folds..." | Add-Content -Path $LogFile
    
    try {
        $mainResult = & python main.py --n_folds $f 2>&1
        $mainResult | Add-Content -Path $LogFile
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  > ERROR: main.py failed for $f folds" -ForegroundColor Red
            "  > ERROR: main.py failed for $f folds" | Add-Content -Path $LogFile
            continue
        }
    }
    catch {
        Write-Host "  > ERROR: main.py failed for $f folds - $_" -ForegroundColor Red
        "  > ERROR: main.py failed for $f folds - $_" | Add-Content -Path $LogFile
        continue
    }
    
    # Move generated figures to the fold-specific directory
    # Looking for figures in the default path structure
    $sourceFigures = "figures\mtrf_ridge_torch\External\stims_Normalize_EEG_Standarize\tmin-0.2_tmax0.6\All\Envelope"
    
    if (Test-Path $sourceFigures) {
        Write-Host "  > Moving figures to $foldDir..." -ForegroundColor Gray
        "  > Moving figures to $foldDir..." | Add-Content -Path $LogFile
        
        try {
            Copy-Item -Path "$sourceFigures\*" -Destination $foldDir -Recurse -Force
            Remove-Item -Path $sourceFigures -Recurse -Force
        }
        catch {
            Write-Host "  > WARNING: Failed to move figures - $_" -ForegroundColor Yellow
            "  > WARNING: Failed to move figures - $_" | Add-Content -Path $LogFile
        }
    }
    else {
        Write-Host "  > WARNING: No figures found in expected location" -ForegroundColor Yellow
        "  > WARNING: No figures found in expected location" | Add-Content -Path $LogFile
    }
    
    # Run the correlation extractor to get results for this fold
    Write-Host "  > Extracting correlation results for $f folds..." -ForegroundColor Gray
    "  > Extracting correlation results for $f folds..." | Add-Content -Path $LogFile
    
    try {
        $extractResult = & python extract_fold_correlations.py --n_folds $f --output_dir $foldDir 2>&1
        $extractResult | Add-Content -Path $LogFile
    }
    catch {
        Write-Host "  > WARNING: Correlation extraction failed - $_" -ForegroundColor Yellow
        "  > WARNING: Correlation extraction failed - $_" | Add-Content -Path $LogFile
    }
    
    # Brief pause to avoid overwhelming the system
    Start-Sleep -Seconds 1
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "Fold analysis completed!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Running final correlation analysis..." -ForegroundColor Yellow
"Running final correlation analysis..." | Add-Content -Path $LogFile

# Generate the final correlation vs folds analysis
try {
    $analysisResult = & python analyze_fold_correlations.py --base_dir $BaseDir 2>&1
    $analysisResult | Add-Content -Path $LogFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "SUCCESS: Fold determination analysis completed!" -ForegroundColor Green
        "SUCCESS: Fold determination analysis completed!" | Add-Content -Path $LogFile
        Write-Host ""
        Write-Host "Results saved in: $BaseDir" -ForegroundColor Yellow
        Write-Host "Check the correlation_vs_folds_curve.png for the analysis!" -ForegroundColor Yellow
    }
    else {
        Write-Host ""
        Write-Host "ERROR: Final analysis failed. Check log file for details." -ForegroundColor Red
        "ERROR: Final analysis failed. Check log file for details." | Add-Content -Path $LogFile
    }
}
catch {
    Write-Host ""
    Write-Host "ERROR: Final analysis failed - $_" -ForegroundColor Red
    "ERROR: Final analysis failed - $_" | Add-Content -Path $LogFile
}

$endTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"Completed at: $endTimestamp" | Add-Content -Path $LogFile

Write-Host ""
Write-Host "Analysis completed at: $endTimestamp" -ForegroundColor Green
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
