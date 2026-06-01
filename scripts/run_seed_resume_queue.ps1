$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$root = Split-Path -Parent $scriptDir
$python = "D:\conda\python.exe"
$logDir = Join-Path $root "outputs\logs"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$jobs = @(
    @{
        Name = "lambda_t_05_seed7_resume"
        TrainConfig = Join-Path $root "configs\bridge_retrieval_lambda_t_05_seed7.yaml"
        EvalConfig = Join-Path $root "configs\bridge_retrieval_lambda_t_05_seed7.yaml"
        TrainOnly = $false
    },
    @{
        Name = "lambda_t_05_seed21"
        TrainConfig = Join-Path $root "configs\bridge_retrieval_lambda_t_05_seed21.yaml"
        EvalConfig = Join-Path $root "configs\bridge_retrieval_lambda_t_05_seed21.yaml"
        TrainOnly = $false
    }
)

foreach ($job in $jobs) {
    $trainOut = Join-Path $logDir ("resume_train_" + $job.Name + ".out.log")
    $trainErr = Join-Path $logDir ("resume_train_" + $job.Name + ".err.log")
    $evalOut = Join-Path $logDir ("resume_eval_" + $job.Name + ".out.log")
    $evalErr = Join-Path $logDir ("resume_eval_" + $job.Name + ".err.log")

    Write-Output ("[QUEUE] train " + $job.Name)
    & $python -u (Join-Path $root "scripts\train_retrieval.py") --config $job.TrainConfig 1>> $trainOut 2>> $trainErr

    if (-not $job.TrainOnly) {
        Write-Output ("[QUEUE] eval " + $job.Name)
        & $python -u (Join-Path $root "scripts\evaluate_retrieval.py") --config $job.EvalConfig 1>> $evalOut 2>> $evalErr
    }
}

Write-Output "[QUEUE] all resumed jobs completed"
