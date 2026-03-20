param(
    [int]$TempLimit = 83,
    [int]$Interval = 30,
    [string]$LogFile = "C:\Users\JungHyun\Desktop\brain\BrainSimulation\gpu_monitor.log"
)

"$(Get-Date): GPU Monitor started (temp limit=${TempLimit}C, interval=${Interval}s)" | Out-File $LogFile

while ($true) {
    try {
        # GPU 3D utilization (same as Task Manager)
        $samples = (Get-Counter "\GPU Engine(*engtype_3D)\Utilization Percentage" -SampleInterval 1 -ErrorAction Stop).CounterSamples
        $gpu3d = [math]::Round(($samples | Measure-Object -Property CookedValue -Sum).Sum, 1)

        # nvidia-smi for temp/power/mem
        $nvsmi = nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits 2>$null
        if ($nvsmi) {
            $parts = $nvsmi -split ','
            $temp = $parts[0].Trim()
            $power = $parts[1].Trim()
            $memUsed = $parts[2].Trim()
            $memTotal = $parts[3].Trim()
        } else {
            $temp = "?"; $power = "?"; $memUsed = "?"; $memTotal = "?"
        }

        $ts = Get-Date -Format "HH:mm:ss"
        $line = "$ts | 3D:${gpu3d}% | ${temp}C | ${power}W | ${memUsed}/${memTotal}MB"
        $line | Out-File $LogFile -Append

        $tempInt = [int]$temp
        if ($tempInt -ge $TempLimit) {
            "$ts | *** OVERHEAT ${temp}C >= ${TempLimit}C — KILLING TRAINING ***" | Out-File $LogFile -Append
            wsl -d Ubuntu-24.04 -- bash -c "pkill -f forager_brain" 2>$null
            "$ts | Training killed." | Out-File $LogFile -Append
            break
        }

        # Check if training still running (allow 3 consecutive misses for build time)
        $running = wsl -d Ubuntu-24.04 -- bash -c "pgrep -f forager_brain" 2>$null
        if (-not $running) {
            if (-not $script:missCount) { $script:missCount = 0 }
            $script:missCount++
            if ($script:missCount -ge 3) {
                "$ts | Training not found (3 checks). Monitor exiting." | Out-File $LogFile -Append
                break
            }
        } else {
            $script:missCount = 0
        }
    } catch {
        $ts = Get-Date -Format "HH:mm:ss"
        "$ts | Error: $_" | Out-File $LogFile -Append
    }

    Start-Sleep -Seconds ($Interval - 1)
}

"$(Get-Date): GPU Monitor stopped." | Out-File $LogFile -Append
