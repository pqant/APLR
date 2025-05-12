# Plaka tespiti için YOLOv8 modelini sıfırdan eğitme
# Bu script eğitimi en baştan başlatır

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

# Çıktı dizini yoksa oluştur
if (-not (Test-Path $output_path)) {
    New-Item -ItemType Directory -Path $output_path -Force
}

# Sıfırdan eğitimi başlat
python train_ufpr.py `
    --dataset_root $dataset_root `
    --output_path $output_path `
    --epochs $epochs `
    --batch_size $batch_size `
    --img_size $img_size 