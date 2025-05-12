# Eski eğitim dosyalarını temizle
Write-Host "Eski eğitim dosyaları temizleniyor..." -ForegroundColor Yellow

# Silinecek dizinler

$directories = @(
    "C:/msworks/APLR_Final/data/processed",
    "C:/msworks/APLR_Final/notebooks/runs",
    "C:/msworks/APLR_Final/models"
)
# Her dizini kontrol et ve sil
foreach ($dir in $directories) {
    if (Test-Path $dir) {
        Write-Host "Siliniyor: $dir" -ForegroundColor Red
        Remove-Item -Path $dir -Recurse -Force
    } else {
        Write-Host "Dizin bulunamadı: $dir" -ForegroundColor Gray
    }
}

Write-Host "`nTemizlik tamamlandı!" -ForegroundColor Green
Write-Host "Artık yeni bir eğitime başlayabilirsiniz." -ForegroundColor Green 