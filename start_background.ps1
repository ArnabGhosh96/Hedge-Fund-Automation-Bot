$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$batchPath = Join-Path $projectDir 'run_server.bat'
if (-not (Test-Path -LiteralPath $batchPath)) { throw "Missing server launcher: $batchPath" }
$commandLine = 'cmd.exe /c ""' + $batchPath + '""'
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $commandLine }
if ($result.ReturnValue -ne 0) { throw "Could not start server process (WMI return code $($result.ReturnValue))." }
Write-Host "[SYSTEM] Trading dashboard launched in background (PID $($result.ProcessId))."
