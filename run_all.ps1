# Orchestrator: run all 9 Hopper jobs (3 arms x 3 seeds), max 3 concurrent.
# Waits for >=8GB free RAM before each wave.
# Usage: powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

# Portable interpreter: honour $env:PYTHON if set, else fall back to "python" on PATH.
$py = $env:PYTHON
if (-not $py) { $py = "python" }
$pyCmd = Get-Command $py -ErrorAction SilentlyContinue
if (-not $pyCmd) {
    Write-Error "Python not found: set `$env:PYTHON to your interpreter or put python on PATH (tried: '$py')."
    exit 1
}
$py = $pyCmd.Source
Write-Host "using interpreter: $py"

$jobs = @(
  @{cfg="configs/sac_hopper_baseline.yaml";    seed=0; out="logs/hopper_baseline_s0"},
  @{cfg="configs/sac_hopper_baseline.yaml";    seed=1; out="logs/hopper_baseline_s1"},
  @{cfg="configs/sac_hopper_baseline.yaml";    seed=2; out="logs/hopper_baseline_s2"},
  @{cfg="configs/sac_hopper_rewardscale.yaml"; seed=0; out="logs/hopper_rewardscale_s0"},
  @{cfg="configs/sac_hopper_rewardscale.yaml"; seed=1; out="logs/hopper_rewardscale_s1"},
  @{cfg="configs/sac_hopper_rewardscale.yaml"; seed=2; out="logs/hopper_rewardscale_s2"},
  @{cfg="configs/sac_hopper_fixedalpha.yaml";  seed=0; out="logs/hopper_fixedalpha_s0"},
  @{cfg="configs/sac_hopper_fixedalpha.yaml";  seed=1; out="logs/hopper_fixedalpha_s1"},
  @{cfg="configs/sac_hopper_fixedalpha.yaml";  seed=2; out="logs/hopper_fixedalpha_s2"}
)

function Wait-FreeRam($needGB) {
  while ($true) {
    $os = Get-CimInstance Win32_OperatingSystem
    $freeGB = [math]::Round($os.FreePhysicalMemory/1MB, 1)
    if ($freeGB -ge $needGB) { Write-Host "free RAM ${freeGB}GB >= ${needGB}GB, launching"; return }
    Write-Host "free RAM ${freeGB}GB < ${needGB}GB, waiting 30s..."
    Start-Sleep 30
  }
}

$running = @()
foreach ($j in $jobs) {
  Wait-FreeRam 8
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
