$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$root = Split-Path -Parent $scriptDir
$python = "D:\conda\python.exe"
$logDir = Join-Path $root "outputs\logs"

$jobs = @(
    @{
        Name = "hardneg_supcon_pair_3epoch"
        Config = Join-Path $root "configs\bridge_retrieval_hardneg_supcon_pair_3epoch.yaml"
        Mode = "eval_only"
    },
    @{
        Name = "altweights_3epoch"
        Config = Join-Path $root "configs\bridge_retrieval_altweights_3epoch.yaml"
        Mode = "train_eval"
    },
    @{
        Name = "lambda_t_05_seed7"
        Config = Join-Path $root "configs\bridge_retrieval_lambda_t_05_seed7.yaml"
        Mode = "train_eval"
    },
    @{
        Name = "lambda_t_05_seed21"
        Config = Join-Path $root "configs\bridge_retrieval_lambda_t_05_seed21.yaml"
        Mode = "train_eval"
    }
)

foreach ($job in $jobs) {
    $trainOut = Join-Path $logDir ("remaining_train_" + $job.Name + ".out.log")
    $trainErr = Join-Path $logDir ("remaining_train_" + $job.Name + ".err.log")
    $evalOut = Join-Path $logDir ("remaining_eval_" + $job.Name + ".out.log")
    $evalErr = Join-Path $logDir ("remaining_eval_" + $job.Name + ".err.log")

    if ($job.Mode -eq "train_eval") {
        Write-Output ("[QUEUE] train " + $job.Name)
        & $python -u (Join-Path $root "scripts\train_retrieval.py") --config $job.Config 1>> $trainOut 2>> $trainErr
    }

    Write-Output ("[QUEUE] eval " + $job.Name)
    & $python -u (Join-Path $root "scripts\evaluate_retrieval.py") --config $job.Config 1>> $evalOut 2>> $evalErr
}

Write-Output "[QUEUE] remaining jobs completed"
