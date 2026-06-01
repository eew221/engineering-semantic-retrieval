param(
    [int]$WaitPid = 0
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = "python"
$logDir = Join-Path $root "outputs\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

function Write-Stage {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$stamp] $Message"
    Write-Host $line
    Add-Content -LiteralPath (Join-Path $logDir "extension_queue.log") -Value $line
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$CliArgs
    )

    Write-Stage "START $Name"
    $outLog = Join-Path $logDir "$Name.out.log"
    $errLog = Join-Path $logDir "$Name.err.log"
    & $python $CliArgs 1>> $outLog 2>> $errLog
    if ($LASTEXITCODE -ne 0) {
        Write-Stage "FAIL $Name exit=$LASTEXITCODE"
        throw "Step failed: $Name"
    }
    Write-Stage "DONE $Name"
}

if ($WaitPid -gt 0) {
    Write-Stage "Waiting for PID $WaitPid to finish before continuing queue"
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
    Write-Stage "PID $WaitPid finished"
}

Set-Location $root
$env:TRANSFORMERS_NO_TF = "1"
$env:USE_TF = "0"

Invoke-Step -Name "eval_lambda_t_01" -CliArgs @(
    "-u", "scripts/evaluate_retrieval.py",
    "--config", "$root\configs\bridge_retrieval_lambda_t_01_3epoch.yaml"
)

Invoke-Step -Name "train_lambda_t_05" -CliArgs @(
    "-u", "scripts/train_retrieval.py",
    "--config", "$root\configs\bridge_retrieval_lambda_t_05_3epoch.yaml"
)

Invoke-Step -Name "eval_lambda_t_05" -CliArgs @(
    "-u", "scripts/evaluate_retrieval.py",
    "--config", "$root\configs\bridge_retrieval_lambda_t_05_3epoch.yaml"
)

Invoke-Step -Name "eval_sdi_zeroshot" -CliArgs @(
    "-u", "scripts/evaluate_retrieval.py",
    "--config", "$root\configs\sdi_eval_zeroshot.yaml",
    "--no-checkpoint"
)

Invoke-Step -Name "eval_sdi_full3epoch" -CliArgs @(
    "-u", "scripts/evaluate_retrieval.py",
    "--config", "$root\configs\sdi_eval_full3epoch.yaml",
    "--checkpoint", "$root\outputs\checkpoints\bridge_engineering_semantic_retrieval_full_3epoch.pt"
)

Invoke-Step -Name "train_full_5epoch" -CliArgs @(
    "-u", "scripts/train_retrieval.py",
    "--config", "$root\configs\bridge_retrieval_full_5epoch.yaml"
)

Invoke-Step -Name "eval_full_5epoch" -CliArgs @(
    "-u", "scripts/evaluate_retrieval.py",
    "--config", "$root\configs\bridge_retrieval_full_5epoch.yaml"
)

Write-Stage "Queue finished"
