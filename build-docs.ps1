#!/usr/bin/env powershell

# Script de construction automatisé pour la documentation MkDocs
# Usage: .\build-docs.ps1

param(
    [switch]$Clean,
    [switch]$Serve,
    [switch]$Open,
    [switch]$Deploy
)

Write-Host "🏗️  Script de construction de la documentation Valeur Résiduelle" -ForegroundColor Blue
Write-Host "================================================================" -ForegroundColor Blue

# Vérifier qu'on est dans le bon dossier
if (-not (Test-Path "mkdocs.yml")) {
    Write-Host "❌ Erreur: mkdocs.yml non trouvé. Assurez-vous d'être dans le dossier racine du projet." -ForegroundColor Red
    exit 1
}

# Activer l'environnement virtuel si il existe
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "🔧 Activation de l'environnement virtuel..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Environnement virtuel activé" -ForegroundColor Green
    }
} else {
    Write-Host "⚠️  Aucun environnement virtuel trouvé (.venv)" -ForegroundColor Yellow
}

# Vérifier que MkDocs est installé
try {
    $mkdocsVersion = & mkdocs --version 2>$null
    Write-Host "✅ MkDocs trouvé: $mkdocsVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ MkDocs non trouvé. Installation..." -ForegroundColor Red
    pip install mkdocs mkdocs-material mkdocs-jupyter pymdown-extensions mkdocs-mermaid2-plugin
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Erreur lors de l'installation de MkDocs" -ForegroundColor Red
        exit 1
    }
}

# Nettoyer si demandé
if ($Clean -or (Test-Path "site")) {
    Write-Host "🧹 Nettoyage de l'ancien build..." -ForegroundColor Yellow
    if (Test-Path "site") {
        Remove-Item -Recurse -Force site
        Write-Host "✅ Ancien dossier 'site' supprimé" -ForegroundColor Green
    }
}

# Mode serveur de développement
if ($Serve) {
    Write-Host "🚀 Lancement du serveur de développement..." -ForegroundColor Blue
    Write-Host "📱 La documentation sera accessible sur http://localhost:8000" -ForegroundColor Cyan
    Write-Host "⌨️  Appuyez sur Ctrl+C pour arrêter" -ForegroundColor Yellow
    mkdocs serve
    exit 0
}

# Construction de la documentation
Write-Host "🏗️  Construction de la documentation..." -ForegroundColor Blue

$buildArgs = @("build")
if ($Clean) {
    $buildArgs += "--clean"
}

& mkdocs @buildArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Documentation construite avec succès !" -ForegroundColor Green
    
    # Statistiques
    $fileCount = (Get-ChildItem -Recurse site\).Count
    $sizeKB = [math]::Round((Get-ChildItem -Recurse site\ | Measure-Object -Property Length -Sum).Sum / 1KB, 2)
    
    Write-Host "📊 Statistiques:" -ForegroundColor Cyan
    Write-Host "   📁 Fichiers générés: $fileCount" -ForegroundColor White
    Write-Host "   💾 Taille totale: $sizeKB KB" -ForegroundColor White
    Write-Host "   📂 Dossier de sortie: site/" -ForegroundColor White
    
    # Déploiement GitHub Pages
    if ($Deploy) {
        Write-Host "🚀 Déploiement sur GitHub Pages..." -ForegroundColor Blue
        mkdocs gh-deploy --clean
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Déployé sur GitHub Pages" -ForegroundColor Green
        } else {
            Write-Host "❌ Erreur lors du déploiement" -ForegroundColor Red
        }
    }
    
    # Ouvrir automatiquement
    if ($Open -or (-not $Deploy)) {
        $response = Read-Host "📱 Voulez-vous ouvrir la documentation ? (Y/n)"
        if ($response -eq "" -or $response -eq "y" -or $response -eq "Y") {
            Write-Host "🌐 Ouverture de la documentation..." -ForegroundColor Blue
            start site\index.html
        }
    }
    
    Write-Host ""
    Write-Host "🎉 Construction terminée !" -ForegroundColor Green
    Write-Host "📋 Commandes utiles:" -ForegroundColor Yellow
    Write-Host "   .\build-docs.ps1 -Serve    # Mode développement" -ForegroundColor White
    Write-Host "   .\build-docs.ps1 -Clean    # Nettoyage + build" -ForegroundColor White
    Write-Host "   .\build-docs.ps1 -Deploy   # Déployer sur GitHub" -ForegroundColor White
    Write-Host "   start site\index.html      # Ouvrir la doc" -ForegroundColor White
    
} else {
    Write-Host "❌ Erreur lors de la construction" -ForegroundColor Red
    Write-Host "💡 Essayez: mkdocs build --verbose pour plus de détails" -ForegroundColor Yellow
    exit 1
}