#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plaka Tanıma Demo
------------------------------

Bu basit bir demo scriptidir, plaka tespiti ve tanıma sistemini gösterir.
"""

import os
import sys
import argparse
import cv2
import matplotlib.pyplot as plt
import pytesseract

# src dizinini yola ekle
sys.path.append(os.path.abspath('src'))

# Özel modülleri içe aktar
from src.plate_detection import PlateDetector
from src.ocr import PlateOCR
from src.preprocessing import preprocess_image_for_plate_detection, preprocess_plate_for_ocr

def parse_arguments():
    """Komut satırı argümanlarını ayrıştır"""
    parser = argparse.ArgumentParser(description='Plaka Tanıma Demo')
    parser.add_argument('--image', type=str, help='İşlenecek görüntünün yolu')
    parser.add_argument('--tesseract-path', type=str, 
                        default=r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                        help='Tesseract uygulamasının yolu')
    parser.add_argument('--model', type=str, help='Özel YOLOv8 model dosyası')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Tespit için güven eşiği')
    parser.add_argument('--no-display', action='store_true', help='Sonuçları gösterme')
    
    return parser.parse_args()

def demo(image_path, tesseract_path, model_path=None, conf_threshold=0.25, display=True):
    """
    Plaka tanıma demosunu çalıştır
    
    Parametreler:
        image_path: Giriş görüntüsünün yolu
        tesseract_path: Tesseract uygulamasının yolu
        model_path: Özel YOLOv8 model dosyası (isteğe bağlı)
        conf_threshold: Tespit için güven eşiği
        display: Sonuçları gösterip göstermeme
    """
    print("Plaka Tanıma Demo")
    print("-" * 30)
    
    # Görüntünün var olup olmadığını kontrol et
    if not os.path.exists(image_path):
        print(f"Hata: {image_path} konumunda görüntü bulunamadı")
        return
    
    # Tesseract kurulumunu kontrol et
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    try:
        # Tespit modülünü başlat
        print("Tespit modülü başlatılıyor...")
        detector = PlateDetector(model_path=model_path)
        detector.set_confidence_threshold(conf_threshold)
        
        # OCR modülünü başlat
        print("OCR modülü başlatılıyor...")
        ocr = PlateOCR(tesseract_path=tesseract_path)
        
        # Görüntüyü oku
        print(f"Görüntü okunuyor: {image_path}...")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hata: {image_path} konumundaki görüntü okunamadı")
            return
        
        # Görüntüyü ön işle
        print("Görüntü ön işleniyor...")
        preprocessed = preprocess_image_for_plate_detection(image)
        
        # Plakaları tespit et
        print("Plakalar tespit ediliyor...")
        detected_plates, annotated_image = detector.detect(image)
        
        if not detected_plates:
            print("Görüntüde plaka tespit edilemedi.")
            # Orijinal görüntüyü göster
            if display:
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title("Orijinal Görüntü - Plaka Tespit Edilemedi")
                plt.axis('off')
                plt.show()
            return
        
        print(f"{len(detected_plates)} adet plaka tespit edildi")
        
        # Plaka bölgelerini çıkar
        plate_images = detector.extract_plate_regions(image, detected_plates)
        
        # Her bir tespit edilen plakayı işle
        for i, plate_img in enumerate(plate_images):
            print(f"\n{i+1}. plaka işleniyor...")
            
            # Plakayı OCR için ön işle
            processed_plate = preprocess_plate_for_ocr(plate_img)
            
            # Plaka karakterlerini tanı
            plate_text, confidence = ocr.recognize_plate(processed_plate)
            
            # OCR sonuçlarını analiz et
            final_text, is_valid = ocr.analyze_results(plate_text, plate_text)
            
            # Plaka konumunu al
            x1, y1, x2, y2, det_conf = detected_plates[i]
            
            # İşaretlenmiş görüntüye tanınan metni ekle
            cv2.putText(annotated_image, final_text, (x1, y2 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Sonucu yazdır
            print(f"Plaka {i+1}: {final_text}")
            print(f"OCR Güveni: {confidence:.2f}")
            print(f"Tespit Güveni: {det_conf:.2f}")
            print(f"Geçerli format: {is_valid}")
        
        # Sonuçları kaydet
        os.makedirs('results/images', exist_ok=True)
        os.makedirs('results/plates', exist_ok=True)
        
        # İşaretlenmiş görüntüyü kaydet
        result_path = os.path.join('results/images', f"result_{os.path.basename(image_path)}")
        cv2.imwrite(result_path, annotated_image)
        print(f"\nİşaretlenmiş görüntü kaydedildi: {result_path}")
        
        # Her bir plakayı ayrı kaydet
        for i, plate_img in enumerate(plate_images):
            plate_path = os.path.join('results/plates', f"plate_{i}_{os.path.basename(image_path)}")
            cv2.imwrite(plate_path, plate_img)
            print(f"Plaka görüntüsü kaydedildi: {plate_path}")
        
        # Sonuçları göster
        if display:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.title("Tespit Edilen Plakalar")
            plt.axis('off')
            plt.show()
            
            # Her bir plakayı ayrı göster
            if len(plate_images) > 0:
                plt.figure(figsize=(15, 3))
                for i, plate_img in enumerate(plate_images):
                    plt.subplot(1, len(plate_images), i+1)
                    plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
                    plt.title(f"Plaka {i+1}")
                    plt.axis('off')
                plt.tight_layout()
                plt.show()
    
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        raise

if __name__ == "__main__":
    args = parse_arguments()
    
    demo(
        image_path=args.image,
        tesseract_path=args.tesseract_path,
        model_path=args.model,
        conf_threshold=args.conf_threshold,
        display=not args.no_display
    ) 