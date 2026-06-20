# Register a Windows Scheduled Task that runs daily_run.py once per weekday.
#
# Why 08:30 ET (default): pre-market for US equities. Yesterday's close has
# fully settled in the Yahoo/Stooq feed, but the regular session has not
# started, so today's bar will not be partial. Adjust -Hour / -Minute as needed.
#
# Default: cheap daily run (no LLM judgments). Pass -WithJudge to add the
# 7-agent LB panel (~$0.70-$1.10 per run).
#
# Usage (run from an elevated PowerShell prompt is NOT required — this uses
# the per-user task store):
#
#     pwsh -File scripts\register_daily_task.ps1                 # default 08:30 local time
#     pwsh -File scripts\register_daily_task.ps1 -Hour 7 -Minute 0
#     pwsh -File scripts\register_daily_task.ps1 -WithJudge      # full daily incl. LB
#     pwsh -File scripts\register_daily_task.ps1 -DryRun         # print the registration plan, don't register
#
# After registering, verify in Task Scheduler GUI:
#     Win+R -> taskschd.msc -> Task Scheduler Library -> stocktrading-daily-run

param(
    [int]   $Hour     = 8,
    [int]   $Minute   = 30,
    [string]$TaskName = "stocktrading-daily-run",
    [switch]$WithJudge,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# ---- Resolve paths ----------------------------------------------------------
$RepoRoot  = (Resolve-Path "$PSScriptRoot\..").Path
$Python    = (Get-Command python).Source
$Script    = Join-Path $RepoRoot "daily_run.py"
$LogDir    = Join-Path $RepoRoot "data\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LogFile   = Join-Path $LogDir "daily_run_stdout.log"
$ErrFile   = Join-Path $LogDir "daily_run_stderr.log"

if (-not (Test-Path $Script)) {
    throw "daily_run.py not found at $Script"
}

# ---- Build the action -------------------------------------------------------
$Args = if ($WithJudge) { "daily_run.py --judge" } else { "daily_run.py" }

# Use cmd.exe wrapper so we can redirect stdout/stderr to log files. Task
# Scheduler's own logging is opaque; tee'd files are infinitely easier to read.
$CmdExe  = "$env:WINDIR\System32\cmd.exe"
$CmdLine = "/c `"`"$Python`" $Args >> `"$LogFile`" 2>> `"$ErrFile`"`""

$Action  = New-ScheduledTaskAction `
    -Execute $CmdExe `
    -Argument $CmdLine `
    -WorkingDirectory $RepoRoot

# ---- Trigger: every weekday at HH:MM (local time) ---------------------------
$TriggerTime = (Get-Date -Hour $Hour -Minute $Minute -Second 0)
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At $TriggerTime

# ---- Settings: don't run on battery, retry on failure, stop after 30 min ----
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 10)

# ---- Principal: run as current user, only when logged in -------------------
$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

# ---- Plan ------------------------------------------------------------------
Write-Host ""
Write-Host "Registration plan:"
Write-Host "  Task name        : $TaskName"
Write-Host "  Run user         : $env:USERDOMAIN\$env:USERNAME"
Write-Host "  Working directory: $RepoRoot"
Write-Host "  Python           : $Python"
Write-Host "  Args             : $Args"
Write-Host "  Trigger          : Every Mon-Fri at $($TriggerTime.ToString('HH:mm')) (local time)"
Write-Host "  Stdout log       : $LogFile"
Write-Host "  Stderr log       : $ErrFile"
Write-Host ""

if ($DryRun) {
    Write-Host "Dry run — no changes made. Re-run without -DryRun to register."
    return
}

# Replace any existing task with the same name
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Task '$TaskName' already exists. Replacing..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Daily stocktrading pipeline: refresh -> scanner -> options_tracker -> todays_actions" `
    | Out-Null

Write-Host "Registered. Verify in Task Scheduler (Win+R -> taskschd.msc)."
Write-Host "To run manually right now:"
Write-Host "    Start-ScheduledTask -TaskName $TaskName"
Write-Host "To unregister later:"
Write-Host "    pwsh -File scripts\unregister_daily_task.ps1"
