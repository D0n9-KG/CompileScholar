# CompileScholar 整仓备份到 NAS（升级前恢复点）。先拷 .git（历史），再拷工作区其余部分。
# 排除：.env（含第三方 API key，NAS 为共享盘）、git 中断残留 tmp_pack_*/tmp_obj_*/gc.pid、venv / node_modules（可重建）。
$ErrorActionPreference = 'Continue'
$src = 'C:\Users\D0n9\Desktop\CompileScholar'
$dst = '\\192.168.199.138\Share400T\personal\jhd0n9\CompileScholar-backup-20261004'
$log = Join-Path $dst '_backup_logs'
New-Item -ItemType Directory -Force -Path $log | Out-Null
"START $(Get-Date -Format s)" | Out-File (Join-Path $log 'status.txt')

robocopy (Join-Path $src '.git') (Join-Path $dst 'repo\.git') /E /COPY:DAT /DCOPY:T /MT:16 /R:2 /W:2 /NFL /NDL /NP `
  /XF tmp_pack_* tmp_obj_* gc.pid /LOG:(Join-Path $log 'git.log')
"GIT_DONE rc=$LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append (Join-Path $log 'status.txt')

robocopy $src (Join-Path $dst 'repo') /E /COPY:DAT /DCOPY:T /MT:16 /R:2 /W:2 /NFL /NDL /NP `
  /XD (Join-Path $src '.git') venv311 .venv venv node_modules `
  /XF .env /LOG:(Join-Path $log 'worktree.log')
"WORKTREE_DONE rc=$LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append (Join-Path $log 'status.txt')
"ALLDONE $(Get-Date -Format s)" | Out-File -Append (Join-Path $log 'status.txt')
