param(
  [int]$Rows = 50000,
  [int]$Trials = 15,
  [int]$Warmups = 3,
  [string]$Python = "python",
  [string]$Output
)

$ErrorActionPreference = "Stop"

if ($Rows -le 0 -or $Trials -le 0 -or $Warmups -le 0) {
  throw "Rows, Trials, and Warmups must be positive integers."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonCommand = Get-Command $Python -ErrorAction Stop
$pythonPath = $pythonCommand.Source
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
if ([string]::IsNullOrWhiteSpace($Output)) {
  $Output = Join-Path $PSScriptRoot "results\cross-library-$stamp.csv"
}
$outputPath = [System.IO.Path]::GetFullPath($Output)
$outputDirectory = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("vectored-bench-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempRoot | Out-Null
$exePath = Join-Path $tempRoot "vectored-bench.exe"
$dartStdOut = Join-Path $tempRoot "dart.stdout.csv"
$dartStdErr = Join-Path $tempRoot "dart.stderr.txt"
$pythonStdOut = Join-Path $tempRoot "python.stdout.csv"
$pythonStdErr = Join-Path $tempRoot "python.stderr.txt"

try {
  Push-Location $repoRoot
  try {
    Write-Host "Compiling Dart AOT benchmark..."
    & dart compile exe bin/benchmark.dart -o $exePath
    if ($LASTEXITCODE -ne 0) {
      throw "Dart AOT compilation failed with exit code $LASTEXITCODE."
    }

    Write-Host "Running Vectored and Matrix2D..."
    $dartProcess = Start-Process -FilePath $exePath `
      -ArgumentList @("--dataframe-csv", "--rows=$Rows", "--trials=$Trials") `
      -Wait -PassThru -NoNewWindow `
      -RedirectStandardOutput $dartStdOut `
      -RedirectStandardError $dartStdErr
    if ($dartProcess.ExitCode -ne 0) {
      Get-Content $dartStdErr | Write-Host
      throw "Dart benchmark failed with exit code $($dartProcess.ExitCode)."
    }
    Get-Content $dartStdErr | Write-Host
    $dartRows = @(Get-Content $dartStdOut)

    Write-Host "Running Pandas and Polars with $pythonPath..."
    $pythonProcess = Start-Process -FilePath $pythonPath `
      -ArgumentList @(
        "benchmarks/benchmark_python.py",
        "--rows=$Rows",
        "--trials=$Trials",
        "--warmups=$Warmups"
      ) `
      -Wait -PassThru -NoNewWindow `
      -RedirectStandardOutput $pythonStdOut `
      -RedirectStandardError $pythonStdErr
    if ($pythonProcess.ExitCode -ne 0) {
      Get-Content $pythonStdErr | Write-Host
      throw "Python benchmark failed with exit code $($pythonProcess.ExitCode). Install packages with: $pythonPath -m pip install -r benchmarks/requirements-python.txt"
    }
    Get-Content $pythonStdErr | Write-Host
    $pythonRows = @(Get-Content $pythonStdOut)
  }
  finally {
    Pop-Location
  }

  $csvLines = @("runtime,library,operation,rows,mean_ms,stddev_ms")
  foreach ($line in @($dartRows) + @($pythonRows)) {
    $value = [string]$line
    if ($value -match "^(dart|python),") {
      $csvLines += $value
    }
  }
  if ($csvLines.Count -le 1) {
    throw "No benchmark CSV records were produced."
  }

  $rawOutputPath = [System.IO.Path]::Combine(
    $outputDirectory,
    ([System.IO.Path]::GetFileNameWithoutExtension($outputPath) + ".raw.csv")
  )
  [System.IO.File]::WriteAllLines(
    $rawOutputPath,
    $csvLines,
    (New-Object System.Text.UTF8Encoding($false))
  )

  $records = @($csvLines | ConvertFrom-Csv)
  $comparisonRows = foreach ($group in ($records | Group-Object {
    "$($_.operation)|$($_.rows)"
  })) {
    $first = $group.Group[0]
    $comparison = [ordered]@{
      operation = $first.operation
      rows = $first.rows
    }
    foreach ($library in @(
      "vectored", "vectored-view", "vectored-list-map", "vectored-list-columns",
      "pandas", "polars", "matrix2d", "list", "list-map", "list-columns"
    )) {
      $record = $group.Group |
        Where-Object { $_.library -eq $library } |
        Select-Object -First 1
      $prefix = $library.Replace("-", "_")
      $comparison["${prefix}_mean_ms"] = if ($null -eq $record) { "" } else { $record.mean_ms }
      $comparison["${prefix}_stddev_ms"] = if ($null -eq $record) { "" } else { $record.stddev_ms }
    }
    [pscustomobject]$comparison
  }
  $comparisonCsv = @($comparisonRows | ConvertTo-Csv -NoTypeInformation)
  [System.IO.File]::WriteAllLines(
    $outputPath,
    $comparisonCsv,
    (New-Object System.Text.UTF8Encoding($false))
  )
  Write-Host ""
  Write-Host "Saved comparison results to $outputPath" -ForegroundColor Green
  Write-Host "Saved raw results to $rawOutputPath"
  Write-Host "Rows: $Rows | Trials: $Trials | Warmups: $Warmups"
}
finally {
  if (Test-Path -LiteralPath $tempRoot) {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force
  }
}
