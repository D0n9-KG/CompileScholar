# CompileScholar full backup to NAS (pre-upgrade restore point). ASCII only: Windows PowerShell 5 reads scripts as ANSI.
# Excludes: .env and .env.bak* (credentials; NAS is a shared drive), git interrupted-op leftovers (tmp_pack_*, tmp_obj_*, gc.pid),
# venvs / node_modules (rebuildable), one trash script containing a key.
$src  = 'C:\Users\D0n9\Desktop\CompileScholar'
$dst  = '\\192.168.199.138\Share400T\personal\jhd0n9\CompileScholar-backup-20261004'
$logd = "$dst\_backup_logs"
New-Item -ItemType Directory -Force -Path $logd | Out-Null
$st   = "$logd\status.txt"
$lgit = "$logd\git.log"
$lwt  = "$logd\worktree.log"
"START $(Get-Date -Format s)" | Out-File -Encoding ascii $st

& robocopy "$src\.git" "$dst\repo\.git" /E /COPY:DAT /DCOPY:T /MT:16 /R:2 /W:2 /NFL /NDL /NP /XF tmp_pack_* tmp_obj_* gc.pid "/LOG:$lgit"
"GIT_DONE rc=$LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append -Encoding ascii $st

& robocopy "$src" "$dst\repo" /E /COPY:DAT /DCOPY:T /MT:16 /R:2 /W:2 /NFL /NDL /NP /XD "$src\.git" venv311 .venv venv node_modules /XF .env .env.bak* _tmp_cst_test.py "/LOG:$lwt"
"WORKTREE_DONE rc=$LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append -Encoding ascii $st
"ALLDONE $(Get-Date -Format s)" | Out-File -Append -Encoding ascii $st
