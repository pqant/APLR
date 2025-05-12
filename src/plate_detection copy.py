import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import shutil
from tqdm import tqdm
import time

class PlateDetector:
    def __init__(self, model_path=None):
        """
        YOLOv8 kullanarak plaka tespit modülünü başlat
        
        Parametreler:
            model_path: Eğitilmiş YOLOv8 model dosyasının yolu. None ise, önceden eğitilmiş modeli kullan
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Önceden eğitilmiş YOLOv8 modelini kullan
            self.model = YOLO('yolov8n.pt')
        
        # Varsayılan güven eşiği
        self.conf_threshold = 0.25
    
    def set_confidence_threshold(self, conf_threshold):
        """
        Tespit için güven eşiğini ayarla
        
        Parametreler:
            conf_threshold: Güven eşiği değeri (0-1)
        """
        self.conf_threshold = conf_threshold
    
    def detect(self, image):
        """
        Görüntüdeki plakaları tespit et
        
        Parametreler:
            image: Giriş görüntüsü (BGR formatında)
            
        Dönüş:
            detected_plates: Plaka bölgelerinin listesi [x1, y1, x2, y2, güven]
            annotated_image: Tespit kutuları çizilmiş görüntü
        """
        # Çıkarım yap
        results = self.model(image, conf=self.conf_threshold)[0]
        
        detected_plates = []
        annotated_image = image.copy()
        
        # Sonuçları işle
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            
            # Tespiti listeye ekle
            detected_plates.append([int(x1), int(y1), int(x2), int(y2), confidence])
            
            # Tespiti görüntüye çiz
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated_image, f"Plaka: {confidence:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return detected_plates, annotated_image
    
    def extract_plate_regions(self, image, detected_plates):
        """
        Görüntüden plaka bölgelerini çıkar
        
        Parametreler:
            image: Giriş görüntüsü
            detected_plates: Tespit edilen plaka bölgeleri listesi [x1, y1, x2, y2, güven]
            
        Dönüş:
            Kırpılmış plaka görüntüleri listesi
        """
        plate_images = []
        
        for plate in detected_plates:
            x1, y1, x2, y2, _ = plate
            plate_img = image[y1:y2, x1:x2]
            plate_images.append(plate_img)
        
        return plate_images
    
    def train_custom_model(self, dataset_path, epochs=50, batch_size=16, img_size=640):
        """
        Plaka tespiti için özel YOLOv8 modeli eğit
        
        Parametreler:
            dataset_path: YOLO formatında veri seti yolu
            epochs: Eğitim epoch sayısı
            batch_size: Toplu iş boyutu
            img_size: Model için giriş görüntüsü boyutu
            
        Dönüş:
            Kaydedilen modelin yolu veya hata durumunda None
        """
        # Veri seti yolunu kontrol et
        if not os.path.exists(dataset_path):
            print(f"Hata: Veri seti yolu mevcut değil: {dataset_path}")
            return None
            
        # Veri seti yaml dosyasını kontrol et
        yaml_path = os.path.join(dataset_path, 'dataset.yaml')
        if not os.path.exists(yaml_path):
            print(f"Hata: dataset.yaml dosyası bulunamadı: {yaml_path}")
            return None
            
        # Erişim izinlerini kontrol et
        try:
            with open(yaml_path, 'r') as f:
                yaml_content = f.read()
                print(f"YAML dosyası okunabildi: {yaml_path}")
                
            # Dizin erişim izinlerini kontrol et
            for split in ['train', 'val', 'test']:
                split_dir = os.path.join(dataset_path, split)
                if os.path.exists(split_dir):
                    try:
                        files = os.listdir(split_dir)
                        print(f"{split} dizini erişilebilir ({len(files)} öğe)")
                    except PermissionError:
                        print(f"Hata: {split} dizinine erişim izni yok")
                        return None
        except PermissionError:
            print(f"Hata: YAML dosyasına erişim izni yok: {yaml_path}")
            return None
        except Exception as e:
            print(f"Veri seti kontrol edilirken beklenmeyen hata: {str(e)}")
            return None
            
        # Alternatif çıktı dizini oluştur (izin sorunlarını önlemek için)
        output_dir = os.path.join(os.getcwd(), 'runs', 'train', f'yolov8_plate_{int(time.time())}')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Eğitim çıktıları şuraya kaydedilecek: {output_dir}")
        
        # Yeni bir model başlat
        try:
            model = YOLO('yolov8n.pt')
            print("YOLOv8n modeli başarıyla yüklendi")
        except Exception as e:
            print(f"Model yüklenirken hata: {str(e)}")
            return None
        
        # Modeli eğit
        try:
            print(f"Eğitim başlatılıyor: {epochs} epochs, {batch_size} batch size, {img_size} image size")
            print(f"Veri seti: {dataset_path}")
            print(f"Cihaz: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            
            results = model.train(
                data=yaml_path,  # Doğrudan yaml dosyasını kullan
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                save=True,
                project=os.path.dirname(output_dir),
                name=os.path.basename(output_dir),
                device='0' if torch.cuda.is_available() else 'cpu'
            )
            
            # Kaydedilen modelin yolunu al
            saved_model_path = str(Path(results.save_dir) / 'weights' / 'best.pt')
            
            # Mevcut modeli güncelle
            self.model = YOLO(saved_model_path)
            
            print(f"Eğitim başarıyla tamamlandı. Model kaydedildi: {saved_model_path}")
            return saved_model_path
            
        except PermissionError as e:
            print(f"Eğitim sırasında izin hatası: {str(e)}")
            print("Çözüm önerisi: Farklı bir çıktı dizini kullanın veya mevcut dizine yazma izni verin")
            return None
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"GPU bellek hatası: {str(e)}")
                print("Çözüm önerisi: batch_size değerini düşürün veya CPU kullanın")
            else:
                print(f"Eğitim sırasında çalışma zamanı hatası: {str(e)}")
            return None
        except Exception as e:
            print(f"Eğitim sırasında beklenmeyen hata: {str(e)}")
            print(f"Hata türü: {type(e).__name__}")
            return None

    def prepare_ufpr_dataset(self, dataset_root, output_path):
        """
        UFPR-ALPR veri setini YOLO eğitimi için hazırla
        
        Parametreler:
            dataset_root: training/validation/testing klasörlerini içeren UFPR-ALPR veri seti kök dizini
            output_path: Hazırlanan veri setini kaydetmek için yol
            
        Dönüş:
            Hazırlanan veri setinin yolu
        """
        print("Veri seti hazırlama başladı...")
        start_time = time.time()
        
        # Çıktı dizinlerini oluştur
        splits = ['train', 'val', 'test']
        for split in splits:
            os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)
        
        # UFPR bölümlerini bizim bölümlerimizle eşleştir
        split_mapping = {
            'training': 'train',
            'validation': 'val',
            'testing': 'test'
        }
        
        total_images = 0
        processed_images = 0
        
        # Önce toplam görüntü sayısını hesapla
        print("Toplam görüntü sayısı hesaplanıyor...")
        for ufpr_split, our_split in split_mapping.items():
            split_path = os.path.join(dataset_root, ufpr_split)
            if not os.path.exists(split_path):
                print(f"Uyarı: {split_path} mevcut değil")
                continue
            
            for track_folder in os.listdir(split_path):
                track_path = os.path.join(split_path, track_folder)
                if not os.path.isdir(track_path):
                    continue
                
                total_images += len([f for f in os.listdir(track_path) if f.endswith('.png')])
        
        print(f"Toplam {total_images} görüntü işlenecek")
        
        # Her bir bölümü işle
        for ufpr_split, our_split in split_mapping.items():
            split_path = os.path.join(dataset_root, ufpr_split)
            if not os.path.exists(split_path):
                print(f"Uyarı: {split_path} mevcut değil")
                continue
            
            print(f"\n{ufpr_split} bölümü işleniyor...")
            
            # Her bir izleme klasörünü işle
            for track_folder in tqdm(os.listdir(split_path), desc=f"{ufpr_split} izlemeleri"):
                track_path = os.path.join(split_path, track_folder)
                if not os.path.isdir(track_path):
                    continue
                
                # İzlemedeki her bir görüntüyü işle
                for file in os.listdir(track_path):
                    if not file.endswith('.png'):
                        continue
                    
                    # İlgili açıklama dosyasını al
                    txt_file = file.replace('.png', '.txt')
                    txt_path = os.path.join(track_path, txt_file)
                    
                    if not os.path.exists(txt_path):
                        continue
                    
                    # Açıklama dosyasını oku
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Plaka köşelerini ayrıştır
                    plate_corners = None
                    for line in lines:
                        if line.startswith('corners:'):
                            corners = line.split(':')[1].strip().split()
                            plate_corners = []
                            for corner in corners:
                                x, y = map(int, corner.split(','))
                                plate_corners.append([x, y])
                            break
                    
                    if plate_corners is None:
                        continue
                    
                    # Köşelerden sınırlayıcı kutuyu hesapla
                    x_coords = [p[0] for p in plate_corners]
                    y_coords = [p[1] for p in plate_corners]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    
                    # Boyutları almak için görüntüyü oku
                    img_path = os.path.join(track_path, file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    img_height, img_width = img.shape[:2]
                    
                    # YOLO formatına dönüştür (normalize edilmiş merkez x, merkez y, genişlik, yükseklik)
                    center_x = (x1 + x2) / 2 / img_width
                    center_y = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # YOLO açıklaması oluştur
                    yolo_annotation = f"0 {center_x} {center_y} {width} {height}"
                    
                    # Görüntüyü ve açıklamayı kaydet
                    output_img_path = os.path.join(output_path, our_split, 'images', file)
                    output_label_path = os.path.join(output_path, our_split, 'labels', file.replace('.png', '.txt'))
                    
                    shutil.copy2(img_path, output_img_path)
                    with open(output_label_path, 'w') as f:
                        f.write(yolo_annotation)
                    
                    processed_images += 1
                    if processed_images % 100 == 0:
                        elapsed_time = time.time() - start_time
                        print(f"\nİşlenen görüntü: {processed_images}/{total_images} ({processed_images/total_images*100:.1f}%)")
                        print(f"Geçen süre: {elapsed_time/60:.1f} dakika")
                        print(f"Tahmini kalan süre: {(elapsed_time/processed_images * (total_images-processed_images))/60:.1f} dakika")
        
        # dataset.yaml dosyası oluştur
        yaml_content = f"""
path: {output_path}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['license_plate']
"""
        with open(os.path.join(output_path, 'dataset.yaml'), 'w') as f:
            f.write(yaml_content)
        
        total_time = time.time() - start_time
        print(f"\nVeri seti hazırlama tamamlandı!")
        print(f"Toplam süre: {total_time/60:.1f} dakika")
        print(f"İşlenen toplam görüntü: {processed_images}")
        
        return output_path 