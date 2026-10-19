$ErrorActionPreference = 'Stop'
$escaped = [regex]::Escape((Split-Path -Parent $MyInvocation.MyCommand.Path))
$targets = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -in @('python.exe','pythonw.exe','cmd.exe')) -and
    (($_.CommandLine -match 'main\.py' -and $_.CommandLine -match $escaped) -or ($_.Name -eq 'cmd.exe' -and $_.CommandLine -match 'run_server\.bat' -and $_.CommandLine -match $escaped))
}
$portOwners = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
$pids = @($targets | Select-Object -ExpandProperty ProcessId) + @($portOwners)
$pids = $pids | Where-Object { $_ -and $_ -ne $PID } | Sort-Object -Unique
foreach ($processId in $pids) {
    try {
        $process = Get-Process -Id $processId -ErrorAction Stop
        if ($process.ProcessName -in @('python','pythonw','cmd')) {
            Stop-Process -Id $processId -ErrorAction Stop
            Write-Host "[SYSTEM] Stopped server process $processId."
        } else {
            Write-Warning "Port 8000 is held by PID $processId ($($process.ProcessName)); left untouched."
        }
    } catch { Write-Warning "Could not stop PID $processId: $_" }
}
if (-not $pids) { Write-Host '[SYSTEM] No matching server process found.' }
