# compile.ps1 — Compile DocVerify paper
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File compile.ps1

Set-Location $PSScriptRoot

$pdflatex = "C:\Users\herdi\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
$bibtex   = "C:\Users\herdi\AppData\Local\Programs\MiKTeX\miktex\bin\x64\bibtex.exe"

Write-Host "=== Pasada 1: pdflatex ===" -ForegroundColor Cyan
& $pdflatex --enable-installer -interaction=nonstopmode main.tex
if ($LASTEXITCODE -ne 0) { Write-Warning "Pasada 1 con warnings/errores (codigo $LASTEXITCODE)" }

Write-Host "`n=== BibTeX ===" -ForegroundColor Cyan
& $bibtex main
if ($LASTEXITCODE -ne 0) { Write-Warning "BibTeX con warnings (codigo $LASTEXITCODE)" }

Write-Host "`n=== Pasada 2: pdflatex ===" -ForegroundColor Cyan
& $pdflatex --enable-installer -interaction=nonstopmode main.tex

Write-Host "`n=== Pasada 3: pdflatex ===" -ForegroundColor Cyan
& $pdflatex --enable-installer -interaction=nonstopmode main.tex

Write-Host "`n=== Resultado ===" -ForegroundColor Cyan
if (Test-Path "main.pdf") {
    $size = (Get-Item "main.pdf").Length
    Write-Host "PDF generado: main.pdf ($([math]::Round($size/1KB, 1)) KB)" -ForegroundColor Green
} else {
    Write-Host "ERROR: main.pdf no generado" -ForegroundColor Red
    Write-Host "Revisa main.log para detalles"
}
