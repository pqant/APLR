# Plaka tespiti için YOLOv8 model eğitimini devam ettirme
# Bu script eğitimi en son kaydedilen noktadan devam ettirir

# Ortam değişkenlerini ve yolları ayarla
$env:PYTHONPATH = "."

# Varsayılan parametreler (komut satırından değiştirilebilir)
param(
    [string]$dataset_root = "data/raw/ufpr-alpr",  # Veri setinin ana dizini
    [string]$output_path = "data/processed",      # İşlenmiş verilerin ve modelin kaydedileceği dizin
    [int]$epochs = 100,                          # Toplam eğitim epoch sayısı
    [int]$batch_size = 16,                       # Batch boyutu
    [int]$img_size = 640                         # Giriş görüntü boyutu
)

# Çıktı dizininin varlığını kontrol et
if (-not (Test-Path $output_path)) {
    Write-Error "Çıktı dizini bulunamadı. Lütfen önce sıfırdan eğitimi başlatın."
    exit 1
}

# Eğitimi son kaydedilen noktadan devam ettir
python train_ufpr.py `
    --dataset_root $dataset_root `
    --output_path $output_path `
    --epochs $epochs `
    --batch_size $batch_size `
    --img_size $img_size `
    --resume 