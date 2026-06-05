# AgriMind - GitHub Push Script
# Calistirmak icin: sag tik -> "PowerShell ile Calistir"
# Veya terminal'de: cd "C:\Users\user\Desktop\YoruzuyAI\Side Projects\AgriMind" && powershell -ExecutionPolicy Bypass -File push_to_github.ps1

Set-Location "C:\Users\user\Desktop\YoruzuyAI\Side Projects\AgriMind"

Write-Host "=== AgriMind GitHub Push ===" -ForegroundColor Cyan

# Index lock varsa kaldir
if (Test-Path ".git\index.lock") {
    Remove-Item ".git\index.lock" -Force
    Write-Host "Index lock temizlendi." -ForegroundColor Yellow
}

# Git config
git config user.email "ahmettalhaymn@gmail.com"
git config user.name "talhaymn7"

# Tracked olmamasi gereken dosyalari index'ten cikar
Write-Host "`nTracking'den cikarilanlar:" -ForegroundColor Yellow
git rm --cached codex.md 2>$null
git rm --cached data/hemp_training.csv 2>$null
git rm --cached -r REPORT/ 2>$null
git rm --cached -r .claude/ 2>$null
git rm --cached -r memory/ 2>$null
git rm --cached AI_INFO.md 2>$null
git rm --cached report_photos/ -r 2>$null

# Stage et
Write-Host "`nDosyalar stage ediliyor..." -ForegroundColor Yellow
git add .gitignore README.md

# Geri kalan modified dosyalari da ekle (CRLF farkliliklari dahil)
git add -A

# Status goster
Write-Host "`nCommit edilecekler:" -ForegroundColor Cyan
git status --short

# Commit
Write-Host "`nCommit yapiliyor..." -ForegroundColor Yellow
git commit -m "Add hemp yield prediction pipeline with nested XGBoost classifier and regressor

- Two-stage XGBoost pipeline: suitability classifier (F1=0.961) + yield regressor (R2=0.70)
- Synthetic training dataset generator (2,000 samples, agronomic formula-based)
- Historical correction layer with exponential-weighted cycle data
- Hemp prescription engine: N/P/K and irrigation recommendations (dekar units)
- NASA POWER and FAOSTAT external data ingestion pipelines
- Pluggable AI provider system (rule-based, ML, LLM, stub)
- Updated README and .gitignore; removed report and AI session files"

# Push
Write-Host "`nGitHub'a push yapiliyor..." -ForegroundColor Yellow
git push origin main

Write-Host "`n=== Tamamlandi ===" -ForegroundColor Green
Write-Host "Repo: https://github.com/talhaymn7/sci305-final-project" -ForegroundColor Cyan
