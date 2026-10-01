# Orchestrator: run all 6 Hopper jobs, max 3 concurrent.
# Usage: powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Continue"
$py = "C:\Users\hahaha\anaconda3\python.exe"
Set-Location $PSScriptRoot

$jobs = @(
  @{cfg="configs/sac_hopper_baseline.yaml";    seed=0; out="logs/hopper_baseline_s0"},
  @{cfg="configs/sac_hopper_baseline.yaml";    seed=1; out="logs/hopper_baseline_s1"},
  @{cfg="configs/sac_hopper_baseline.yaml";    seed=2; out="logs/hopper_baseline_s2"},
  @{cfg="configs/sac_hopper_rewardscale.yaml"; seed=0; out="logs/hopper_rewardscale_s0"},
  @{cfg="configs/sac_hopper_rewardscale.yaml"; seed=1; out="logs/hopper_rewardscale_s1"},
  @{cfg="configs/sac_hopper_rewardscale.yaml"; seed=2; out="logs/hopper_rewardscale_s2"}
)

$running = @()
foreach ($j in $jobs) {
  $p = Start-Process -FilePath $py -ArgumentList "train.py","--config",$j.cfg,"--seed",$j.seed,"--out",$j.out `
        -NoNewWindow -PassThru `
        -RedirectStandardError "$($j.out).err.txt" -RedirectStandardOutput "$($j.out).out.txt"
  $running += $p
  Write-Host "started $($j.out) pid=$($p.Id)"
  if ($running.Count -ge 3) {
    $running | Wait-Process
    $running = @()
    Write-Host "--- wave done ---"
  }
}
if ($running.Count -gt 0) { $running | Wait-Process }
Write-Host "ALL DONE"
