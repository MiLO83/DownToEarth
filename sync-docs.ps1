# sync-docs.ps1 -- one-command sync for the DownToEarth doc set
#
# What this does, in order:
#   1. Detect which of README.md / LYRA2_PROPOSAL.md / DUMMIES.md changed
#      vs the last commit. (Pure no-op if nothing changed.)
#   2. Copy changed .md files from repo root -> voxgaussian/viewer/public/
#   3. Commit + push the .md sync (ASCII-only message, Cloudflare-safe)
#   4. Deploy to Cloudflare Pages so the /readme /proposal /dummies URLs
#      reflect the new markdown
#   5. For each changed doc, re-render the matching PDF via Edge headless
#      print from the now-fresh live URL
#   6. Copy regenerated PDFs back to repo root
#   7. Commit + push the PDF regen, redeploy
#   8. Print the public URLs + commit SHAs
#
# Usage from any directory:
#   pwsh -File C:\Users\rxcam\Documents\DownToEarth\sync-docs.ps1
#
# Or with options:
#   .\sync-docs.ps1                              # auto-detect changes, full flow
#   .\sync-docs.ps1 -Message "docs: ..."         # custom commit message
#   .\sync-docs.ps1 -SkipPdfs                    # markdown-only, skip PDF regen
#   .\sync-docs.ps1 -SkipDeploy                  # commit+push but don't deploy
#   .\sync-docs.ps1 -Force                       # regenerate ALL PDFs even if no MD changes
#   .\sync-docs.ps1 -DryRun                      # show what would happen, change nothing
#
# Exit codes:
#   0 = success (or no-op)
#   1 = unrecoverable error
#   2 = partial success (e.g. deploy failed but commit succeeded)

[CmdletBinding()]
param(
    [string]$Message = "",
    [switch]$SkipPdfs,
    [switch]$SkipDeploy,
    [switch]$Force,
    [switch]$DryRun
)

# ─── Configuration ─────────────────────────────────────────────────────────

$RepoRoot       = "C:\Users\rxcam\Documents\DownToEarth"
$PublicDir      = "$RepoRoot\voxgaussian\viewer\public"
$WranglerProj   = "downtoearth"
$WranglerBranch = "main"
$BaseUrl        = "https://downtoearth-9lq.pages.dev"
$GitUserName    = "MiLO83"
$GitUserEmail   = "edisonzghost@gmail.com"

# Map of doc -> (live URL for PDF render, expected PDF filename).
# All three docs assume markdown at $RepoRoot\<source> and the public-dir
# copy at $PublicDir\<source>.
$Docs = @(
    @{ Md = "README.md";          Url = "$BaseUrl/readme";    Pdf = "README.pdf" },
    @{ Md = "LYRA2_PROPOSAL.md";  Url = "$BaseUrl/proposal";  Pdf = "LYRA2_PROPOSAL.pdf" },
    @{ Md = "DUMMIES.md";         Url = "$BaseUrl/dummies";   Pdf = "DUMMIES.pdf" },
    @{ Md = "DUMMIESV3.md";       Url = "$BaseUrl/dummiesv3"; Pdf = "DUMMIESV3.pdf" }
)

# ─── Helpers ───────────────────────────────────────────────────────────────

function Write-Step  { param($Text) Write-Host "[->] $Text" -ForegroundColor Cyan }
function Write-Ok    { param($Text) Write-Host "[OK] $Text" -ForegroundColor Green }
function Write-Warn  { param($Text) Write-Host "[!!] $Text" -ForegroundColor Yellow }
function Write-Fail  { param($Text) Write-Host "[xx] $Text" -ForegroundColor Red }
function Write-Skip  { param($Text) Write-Host "[..] $Text" -ForegroundColor DarkGray }

function Find-Edge {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
        "${env:ProgramFiles}\Microsoft\Edge\Application\msedge.exe"
    )
    foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
    return $null
}

function Sanitize-AsciiOnly {
    param([string]$Text)
    # Replace common UTF-8 typography with ASCII fallbacks for Cloudflare API safety
    $clean = $Text
    $clean = $clean -replace [char]0x2014, '-'    # em dash -> hyphen
    $clean = $clean -replace [char]0x2013, '-'    # en dash -> hyphen
    $clean = $clean -replace [char]0x2192, '->'   # right arrow
    $clean = $clean -replace [char]0x2194, '<->'  # bidirectional arrow
    $clean = $clean -replace [char]0x2191, '^'    # up arrow
    $clean = $clean -replace [char]0x2193, 'v'    # down arrow
    $clean = $clean -replace [char]0x00B3, '^3'   # superscript 3
    $clean = $clean -replace [char]0x00B2, '^2'   # superscript 2
    $clean = $clean -replace [char]0x00D7, 'x'    # multiplication sign
    $clean = $clean -replace [char]0x2018, "'"    # left single quote
    $clean = $clean -replace [char]0x2019, "'"    # right single quote
    $clean = $clean -replace [char]0x201C, '"'    # left double quote
    $clean = $clean -replace [char]0x201D, '"'    # right double quote
    $clean = $clean -replace [char]0x2026, '...'  # ellipsis
    return $clean
}

function Get-ChangedMarkdowns {
    Push-Location $RepoRoot
    try {
        # Detect .md files in $Docs that differ from HEAD (staged or unstaged)
        $changed = @()
        foreach ($d in $Docs) {
            $rootPath   = Join-Path $RepoRoot $d.Md
            $publicPath = Join-Path $PublicDir $d.Md
            $rootDiff   = git diff --quiet HEAD -- $d.Md 2>$null; $rootDirty = ($LASTEXITCODE -ne 0)
            $publicRel  = "voxgaussian/viewer/public/$($d.Md)"
            $publicDiff = git diff --quiet HEAD -- $publicRel 2>$null; $publicDirty = ($LASTEXITCODE -ne 0)
            # Also detect "root differs from public" (forgot to sync earlier)
            $rootHash   = if (Test-Path $rootPath)   { (Get-FileHash $rootPath   -Algorithm SHA1).Hash } else { $null }
            $publicHash = if (Test-Path $publicPath) { (Get-FileHash $publicPath -Algorithm SHA1).Hash } else { $null }
            $rootVsPublic = ($rootHash -ne $publicHash)
            if ($rootDirty -or $publicDirty -or $rootVsPublic -or $Force) {
                $changed += [PSCustomObject]@{
                    Md          = $d.Md
                    Url         = $d.Url
                    Pdf         = $d.Pdf
                    RootDirty   = $rootDirty
                    PublicDirty = $publicDirty
                    OutOfSync   = $rootVsPublic
                }
            }
        }
        return $changed
    } finally {
        Pop-Location
    }
}

function Invoke-WranglerDeploy {
    Push-Location $RepoRoot
    try {
        Write-Step "Deploying to Cloudflare Pages ..."
        if ($DryRun) { Write-Skip "  (dry run, skipping)"; return $true }
        $output = & npx wrangler pages deploy voxgaussian/viewer/public --project-name=$WranglerProj --branch=$WranglerBranch 2>&1 | Out-String
        if ($output -match "Deployment complete") {
            Write-Ok "Cloudflare Pages deployed"
            return $true
        } else {
            Write-Fail "Wrangler deploy did not report success. Output:"
            Write-Host $output -ForegroundColor DarkRed
            return $false
        }
    } finally {
        Pop-Location
    }
}

function Render-Pdf {
    param(
        [string]$Url,
        [string]$OutPath,
        [string]$EdgePath
    )
    if ($DryRun) { Write-Skip "  (dry run) would render $Url -> $OutPath"; return $true }
    & $EdgePath --headless=new --disable-gpu --no-sandbox --virtual-time-budget=22000 --print-to-pdf-no-header "--print-to-pdf=$OutPath" $Url 2>&1 | Out-Null
    Start-Sleep -Milliseconds 800
    if (Test-Path $OutPath) {
        return $true
    } else {
        return $false
    }
}

# ─── Main flow ─────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "======== sync-docs.ps1 ========" -ForegroundColor White
Write-Host "  Repo:     $RepoRoot"
Write-Host "  Public:   $PublicDir"
Write-Host "  Project:  $WranglerProj"
if ($DryRun)     { Write-Warn "DryRun mode -- nothing will be modified" }
if ($SkipPdfs)   { Write-Warn "SkipPdfs -- markdown only" }
if ($SkipDeploy) { Write-Warn "SkipDeploy -- commit/push but no wrangler deploy" }
if ($Force)      { Write-Warn "Force -- treating all docs as changed" }
Write-Host ""

# 1. Detect changes
Write-Step "Detecting changed markdowns ..."
$changedDocs = Get-ChangedMarkdowns
if ($changedDocs.Count -eq 0) {
    Write-Ok "Nothing to sync. All docs in sync with HEAD."
    exit 0
}
Write-Host ("    Changed: " + (($changedDocs | ForEach-Object { $_.Md }) -join ", ")) -ForegroundColor White

# 2. Sync MD: root -> public
Write-Step "Syncing markdown root -> public ..."
foreach ($d in $changedDocs) {
    $src = Join-Path $RepoRoot $d.Md
    $dst = Join-Path $PublicDir $d.Md
    if (-not (Test-Path $src)) {
        Write-Fail "  Source missing: $src"
        exit 1
    }
    if ($DryRun) { Write-Skip "  $($d.Md): would copy" }
    else { Copy-Item -Force $src $dst; Write-Ok "  $($d.Md)" }
}

# 3. Commit markdown changes + push
if (-not $Message) {
    $changedList = ($changedDocs | ForEach-Object { $_.Md }) -join ", "
    $Message = "docs: sync $changedList"
}
$Message = Sanitize-AsciiOnly $Message
Write-Step "Committing + pushing markdown sync ..."
if ($DryRun) {
    Write-Skip "  would commit with message: $Message"
} else {
    Push-Location $RepoRoot
    try {
        foreach ($d in $changedDocs) {
            git add ($d.Md) 2>&1 | Out-Null
            git add ("voxgaussian/viewer/public/" + $d.Md) 2>&1 | Out-Null
        }
        $cmtOut = git -c "user.name=$GitUserName" -c "user.email=$GitUserEmail" commit -m $Message 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) {
            if ($cmtOut -match "nothing to commit") {
                Write-Skip "  Nothing changed in git terms; skipping"
            } else {
                Write-Fail "  git commit failed:"
                Write-Host $cmtOut -ForegroundColor DarkRed
                exit 1
            }
        } else {
            Write-Ok ("  committed: " + (git log -1 --format='%h %s'))
            $pushOut = git push 2>&1 | Out-String
            if ($LASTEXITCODE -ne 0) {
                Write-Fail "  git push failed:"
                Write-Host $pushOut -ForegroundColor DarkRed
                exit 1
            }
            Write-Ok "  pushed to origin"
        }
    } finally {
        Pop-Location
    }
}

# 4. Deploy (so PDF-render URLs reflect new MD)
if (-not $SkipDeploy) {
    if (-not (Invoke-WranglerDeploy)) {
        Write-Warn "Continuing despite deploy issue; PDFs may reflect stale URLs."
    }
} else {
    Write-Skip "Skipping deploy per -SkipDeploy"
}

# 5. Regenerate PDFs
if (-not $SkipPdfs) {
    $edge = Find-Edge
    if (-not $edge) {
        Write-Fail "Microsoft Edge not found at standard install paths."
        Write-Fail "PDF regeneration skipped; MD sync is complete."
        exit 2
    }
    Write-Step "Regenerating PDFs via Edge headless ($edge) ..."
    foreach ($d in $changedDocs) {
        $publicPdf = Join-Path $PublicDir $d.Pdf
        $rootPdf   = Join-Path $RepoRoot  $d.Pdf
        Write-Host ("    {0} -> {1}" -f $d.Url, $d.Pdf) -ForegroundColor DarkGray
        if (Render-Pdf -Url $d.Url -OutPath $publicPdf -EdgePath $edge) {
            $kb = if (Test-Path $publicPdf) { (Get-Item $publicPdf).Length / 1024 } else { 0 }
            Copy-Item -Force $publicPdf $rootPdf
            Write-Ok ("    {0,-22} {1,8:N1} KB" -f $d.Pdf, $kb)
        } else {
            Write-Fail "    $($d.Pdf): render failed"
        }
    }

    # 6+7. Commit + push PDFs, redeploy
    Write-Step "Committing PDFs + redeploying ..."
    if ($DryRun) {
        Write-Skip "  would commit PDFs"
    } else {
        Push-Location $RepoRoot
        try {
            foreach ($d in $changedDocs) {
                git add ($d.Pdf) 2>&1 | Out-Null
                git add ("voxgaussian/viewer/public/" + $d.Pdf) 2>&1 | Out-Null
            }
            $cmtOut = git -c "user.name=$GitUserName" -c "user.email=$GitUserEmail" commit -m "docs: regenerate PDFs" 2>&1 | Out-String
            if ($LASTEXITCODE -ne 0) {
                if ($cmtOut -match "nothing to commit") {
                    Write-Skip "  PDFs identical to previous; nothing to commit"
                } else {
                    Write-Warn "  git commit (PDFs) issue:"
                    Write-Host $cmtOut -ForegroundColor DarkYellow
                }
            } else {
                Write-Ok ("  committed: " + (git log -1 --format='%h %s'))
                git push 2>&1 | Out-Null
                Write-Ok "  pushed PDFs"
                if (-not $SkipDeploy) {
                    Invoke-WranglerDeploy | Out-Null
                }
            }
        } finally {
            Pop-Location
        }
    }
} else {
    Write-Skip "Skipping PDF regen per -SkipPdfs"
}

# 8. Final report
Write-Host ""
Write-Host "======== Done ========" -ForegroundColor White
Write-Host "  Public URLs:"
foreach ($d in $changedDocs) {
    Write-Host ("    " + $d.Url) -ForegroundColor Cyan
    $pdfUrl = "$BaseUrl/$($d.Pdf)"
    Write-Host ("    " + $pdfUrl) -ForegroundColor Cyan
}
Write-Host ""

exit 0
