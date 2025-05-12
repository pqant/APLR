# Plaka Tanıma Sistemi

Bu proje, YOLOv8 ve Tesseract OCR kullanarak gerçek zamanlı plaka tanıma sistemi geliştirmeyi amaçlamaktadır. Sistem, UFPR-ALPR veri seti üzerinde eğitilmiş ve test edilmiştir.

## Özellikler

- Gerçek zamanlı plaka tespiti (YOLOv8)
- Yüksek doğruluklu OCR (Tesseract)
- Görüntü ön işleme ve iyileştirme
- Çoklu plaka tespiti ve tanıma
- Detaylı performans değerlendirmesi
- UFPR-ALPR veri seti desteği

## Gereksinimler

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Tesseract OCR
- NumPy
- Matplotlib
- scikit-learn

## Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. Tesseract OCR'ı yükleyin:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

## Kullanım

### Tek Görüntü İşleme

```bash
python demo.py --image path/to/image.jpg --tesseract-path "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Veri Seti Üzerinde Eğitim

```bash
python train_ufpr.py --dataset_root path/to/ufpr_dataset --output_path path/to/output
```

### Toplu İşleme

```bash
python src/detect_and_recognize.py --dataset path/to/dataset --model path/to/model.pt
```

## Proje Yapısı

```
├── src/
│   ├── detect_and_recognize.py  # Ana işleme modülü
│   ├── plate_detection.py       # Plaka tespit modülü
│   ├── ocr.py                   # OCR işleme modülü
│   ├── preprocessing.py         # Görüntü ön işleme
│   └── evaluate.py              # Performans değerlendirme
├── data/
│   └── raw/                     # Ham veri seti
├── results/                     # İşleme sonuçları
├── demo.py                      # Demo script
└── train_ufpr.py               # UFPR veri seti eğitim scripti
```

## Performans

Sistem, UFPR-ALPR veri seti üzerinde test edilmiştir. Detaylı performans metrikleri için `results` klasörüne bakınız.

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir özellik dalı oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Dalınıza push edin (`git push origin feature/amazing-feature`)
5. Bir Pull Request açın
