import os
import argparse
from src.plate_detection import PlateDetector

def main():
    parser = argparse.ArgumentParser(description='UFPR-ALPR veri seti üzerinde YOLO modelini eğit')
    parser.add_argument('--dataset_root', type=str, required=True, help='training/validation/testing klasörlerini içeren UFPR-ALPR veri seti kök dizini')
    parser.add_argument('--output_path', type=str, required=True, help='Hazırlanan veri setini ve eğitilen modeli kaydetmek için yol')
    parser.add_argument('--epochs', type=int, default=100, help='Eğitim epoch sayısı')
    parser.add_argument('--batch_size', type=int, default=16, help='Eğitim için toplu iş boyutu')
    parser.add_argument('--img_size', type=int, default=640, help='Model için giriş görüntüsü boyutu')
    parser.add_argument('--resume', action='store_true', help='Son kontrol noktasından eğitime devam et')
    
    args = parser.parse_args()
    
    # Tespit modülünü başlat
    detector = PlateDetector()
    
    # Veri seti yoksa hazırla
    dataset_yaml = os.path.join(args.output_path, 'dataset.yaml')
    if not os.path.exists(dataset_yaml) or not args.resume:
        print("Veri seti hazırlanıyor...")
        prepared_dataset_path = detector.prepare_ufpr_dataset(args.dataset_root, args.output_path)
        print(f"Veri seti şurada hazırlandı: {prepared_dataset_path}")
    else:
        prepared_dataset_path = args.output_path
        print("Mevcut veri seti kullanılıyor...")
    
    # Modeli eğit
    print("Model eğitiliyor...")
    model_path = detector.train_custom_model(
        prepared_dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    print(f"Eğitim tamamlandı. Model şuraya kaydedildi: {model_path}")

if __name__ == "__main__":
    main() 