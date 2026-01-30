# Push konkani_asr to https://github.com/mokshkopikar/amchi_asr
# Run: .\push_to_mokshkopikar.ps1
#
# AUTHENTICATION: You must be logged into GitHub. If prompted:
#   - Use a Personal Access Token (PAT) as password: https://github.com/settings/tokens
#   - Or use: git config --global credential.helper manager (Windows Credential Manager)
#   - Or switch to SSH: git remote set-url moksh git@github.com:mokshkopikar/amchi_asr.git

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Ensure moksh remote exists
$remotes = git remote
if ($remotes -notmatch "moksh") {
    git remote add moksh https://github.com/mokshkopikar/amchi_asr.git
} else {
    git remote set-url moksh https://github.com/mokshkopikar/amchi_asr.git
}

# Stage and commit any uncommitted changes
$status = git status --porcelain
if ($status) {
    Write-Host "Staging and committing changes..."
    git add -A
    git commit -m "Sync: update konkani_asr code"
}

# Push to mokshkopikar/amchi_asr (empty repo - push master)
Write-Host "Pushing to https://github.com/mokshkopikar/amchi_asr ..."
$pushResult = git push moksh master 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Push failed. Trying master:main (in case repo uses main as default)..." -ForegroundColor Yellow
    git push moksh master:main 2>&1
}
if ($LASTEXITCODE -eq 0) {
    Write-Host "Done. Check https://github.com/mokshkopikar/amchi_asr" -ForegroundColor Green
} else {
    Write-Host "Push failed. Ensure you are logged into GitHub (PAT or SSH)." -ForegroundColor Red
    exit 1
}
