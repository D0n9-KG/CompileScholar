# W6 S8 cold storage: copy retired experiment dirs to the NAS, verify file count + total bytes + sha256 of every file,
# then (only with -Delete) remove the local copy. ASCII only (PowerShell 5 reads scripts as ANSI).
param([switch]$Delete)
$ErrorActionPreference = 'Stop'
$src  = 'C:\Users\D0n9\Desktop\CompileScholar\.research_tmp'
$dst  = '\\192.168.199.138\Share400T\personal\jhd0n9\CompileScholar-cold-20261004'
$dirs = @('experiments\archive', 'experiments\baselines', 'experiments\e2_need_gap', 'experiments\stageB',
          'archive_oneoff', 'benchmark-audit-0926')
New-Item -ItemType Directory -Force -Path "$dst\_logs" | Out-Null
$report = @()
foreach ($d in $dirs) {
  $s = Join-Path $src $d
  if (-not (Test-Path $s)) { $report += "$d : missing (skipped)"; continue }
  $t = Join-Path $dst $d
  & robocopy $s $t /E /COPY:DAT /DCOPY:T /MT:16 /R:2 /W:2 /NFL /NDL /NP "/LOG:$dst\_logs\$($d -replace '\\','_').log" | Out-Null
  $rc = $LASTEXITCODE
  $sf = Get-ChildItem $s -Recurse -File -Force
  $tf = Get-ChildItem $t -Recurse -File -Force
  $sb = ($sf | Measure-Object Length -Sum).Sum
  $tb = ($tf | Measure-Object Length -Sum).Sum
  # per-file sha256 (relative path -> hash), compared both ways
  $bad = 0
  foreach ($f in $sf) {
    $rel = $f.FullName.Substring($s.Length)
    $g = Join-Path $t $rel
    if (-not (Test-Path -LiteralPath $g)) { $bad++; continue }
    if ((Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $g -Algorithm SHA256).Hash) { $bad++ }
  }
  $ok = ($rc -lt 8) -and ($sf.Count -eq $tf.Count) -and ($sb -eq $tb) -and ($bad -eq 0)
  $line = "{0} : robocopy rc={1} files {2}/{3} bytes {4}/{5} hash_mismatch={6} -> {7}" -f $d, $rc, $tf.Count, $sf.Count, $tb, $sb, $bad, $(if ($ok) {'VERIFIED'} else {'FAILED'})
  $report += $line
  if ($ok -and $Delete) {
    Remove-Item -LiteralPath $s -Recurse -Force
    $report += "$d : local copy deleted"
  }
}
$report | Out-File -Encoding ascii "$dst\_logs\report.txt"
$report
