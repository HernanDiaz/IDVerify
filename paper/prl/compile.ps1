# Compile PRL submission (Pattern Recognition Letters)
# Requires: pdflatex, bibtex, and prletters.sty from Elsevier in the same folder
Set-Location $PSScriptRoot
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
Write-Host "Done. Output: main.pdf"
