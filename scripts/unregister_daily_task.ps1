# Remove the stocktrading-daily-run scheduled task.
#
# Usage:
#     pwsh -File scripts\unregister_daily_task.ps1

param(
    [string]$TaskName = "stocktrading-daily-run"
)

$ErrorActionPreference = "Stop"

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Unregistered '$TaskName'."
} else {
    Write-Host "No task named '$TaskName' found — nothing to do."
}
