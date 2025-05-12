import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Özel modülleri içe aktar
from preprocessing import preprocess_image_for_plate_detection, preprocess_plate_for_ocr
from plate_detection import PlateDetector
from ocr import PlateOCR
from evaluate import EvaluationMetrics

def process_single_image(image_path, detector, ocr, save_results=True, display=False):
    """
    Tek bir görüntüde plaka tespiti ve tanıma işlemi yap
    
    Parametreler:
        image_path: Giriş görüntüsünün yolu
        detector: Başlatılmış PlateDetector nesnesi
        ocr: Başlatılmış PlateOCR nesnesi
        save_results: Sonuçların diske kaydedilip kaydedilmeyeceği
        display: Sonuçların gösterilip gösterilmeyeceği
        
    Dönüş:
        recognized_plates: Tanınan plakaların metin ve konumlarını içeren liste
        result_image: Tespit kutuları ve tanınan metinlerle işaretlenmiş görüntü
    """
    # Görüntüyü oku
    image = cv2.imread(image_path)
    if image is None:
        print(f"Hata: {image_path} konumundaki görüntü okunamadı")
        return [], None
    
    # Görüntüyü ön işle
    preprocessed = preprocess_image_for_plate_detection(image)
    
    # Plakaları tespit et
    detected_plates, annotated_image = detector.detect(image)
    
    # Plaka bölgelerini çıkar
    plate_images = detector.extract_plate_regions(image, detected_plates)
    
    recognized_plates = []
    
    # Her bir tespit edilen plakayı işle
    for i, plate_img in enumerate(plate_images):
        # Plakayı OCR için ön işle
        processed_plate = preprocess_plate_for_ocr(plate_img)
        
        # Plaka karakterlerini tanı
        plate_text, ocr_confidence = ocr.recognize_plate(processed_plate)
        
        # OCR sonuçlarını analiz et
        final_text, is_valid = ocr.analyze_results(plate_text, plate_text)
        
        # Plaka konumunu al
        x1, y1, x2, y2, det_conf = detected_plates[i]
        
        # İşaretlenmiş görüntüye tanınan metni ekle
        cv2.putText(annotated_image, final_text, (x1, y2 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Sonuçları listeye ekle
        recognized_plates.append({
            'text': final_text,
            'position': [x1, y1, x2, y2],
            'detection_confidence': det_conf,
            'ocr_confidence': ocr_confidence,
            'is_valid': is_valid
        })
    
    # Sonuçları kaydet
    if save_results:
        os.makedirs('results/images', exist_ok=True)
        os.makedirs('results/plates', exist_ok=True)
        
        # İşaretlenmiş görüntüyü kaydet
        result_path = os.path.join('results/images', f"result_{os.path.basename(image_path)}")
        cv2.imwrite(result_path, annotated_image)
        
        # Her bir plakayı ayrı kaydet
        for i, plate_img in enumerate(plate_images):
            plate_path = os.path.join('results/plates', f"plate_{i}_{os.path.basename(image_path)}")
            cv2.imwrite(plate_path, plate_img)
    
    # Sonuçları göster
    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.title("Tespit Edilen Plakalar")
        plt.axis('off')
        plt.show()
    
    return recognized_plates, annotated_image

def process_dataset(dataset_path, detector, ocr, ground_truth=None):
    """
    Görüntü veri setini işle ve performansı değerlendir
    
    Parametreler:
        dataset_path: Veri seti dizininin yolu
        detector: Başlatılmış PlateDetector nesnesi
        ocr: Başlatılmış PlateOCR nesnesi
        ground_truth: Gerçek etiket dosyasının yolu (isteğe bağlı)
        
    Dönüş:
        Değerlendirme sonuçlarını içeren EvaluationMetrics nesnesi
    """
    # Değerlendirme metriklerini başlat
    evaluator = EvaluationMetrics()
    
    # Gerçek etiketleri yükle (varsa)
    gt_data = {}
    if ground_truth and os.path.exists(ground_truth):
        import json
        with open(ground_truth, 'r') as f:
            gt_data = json.load(f)
    
    # Görüntü dosyalarının listesini al
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(dataset_path).glob(f'*{ext}')))
    
    total_images = len(image_files)
    processed_images = 0
    
    print(f"{total_images} görüntü işlenecek...")
    
    start_time = time.time()
    
    for img_path in image_files:
        # Görüntüyü işle
        img_filename = os.path.basename(img_path)
        recognized_plates, _ = process_single_image(str(img_path), detector, ocr, save_results=True, display=False)
        
        # Gerçek etiketler varsa değerlendirme metriklerini güncelle
        if img_filename in gt_data:
            gt_info = gt_data[img_filename]
            
            # Değerlendirme için formatla
            gt_boxes = [box['position'] for box in gt_info['plates']]
            gt_texts = [plate['text'] for plate in gt_info['plates']]
            
            detected_boxes = [plate['position'] + [plate['detection_confidence']] for plate in recognized_plates]
            detected_texts = [plate['text'] for plate in recognized_plates]
            
            # Tespiti değerlendir
            precision, recall, f1 = evaluator.evaluate_detection(gt_boxes, detected_boxes)
            
            # Plaka tespit edildiyse OCR'ı değerlendir
            if detected_texts and gt_texts:
                char_acc, exact_acc = evaluator.evaluate_ocr(gt_texts[:len(detected_texts)], detected_texts)
                
                # Tespit sonucunu ekle
                for i, plate in enumerate(recognized_plates):
                    if i < len(gt_info['plates']):
                        evaluator.add_detection_result(
                            img_filename,
                            gt_info['plates'][i],
                            {
                                'position': plate['position'],
                                'confidence': plate['detection_confidence']
                            },
                            {
                                'text': plate['text'],
                                'confidence': plate['ocr_confidence']
                            }
                        )
        
        processed_images += 1
        if processed_images % 10 == 0:
            print(f"{processed_images}/{total_images} görüntü işlendi")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"İşlem {processing_time:.2f} saniyede tamamlandı")
    print(f"Görüntü başına ortalama süre: {processing_time/total_images:.2f} saniye")
    
    # Değerlendirme sonuçlarını görselleştir ve kaydet
    plot_path, csv_path = evaluator.plot_results()
    print(f"Değerlendirme sonuçları {plot_path} ve {csv_path} konumlarına kaydedildi")
    
    return evaluator

def parse_arguments():
    """
    Komut satırı argümanlarını ayrıştır
    
    Dönüş:
        Ayrıştırılmış argümanlar
    """
    parser = argparse.ArgumentParser(description='Plaka Tespiti ve Tanıma')
    parser.add_argument('--image', type=str, help='İşlenecek tek görüntünün yolu')
    parser.add_argument('--dataset', type=str, help='Veri seti dizininin yolu')
    parser.add_argument('--model', type=str, help='Eğitilmiş YOLOv8 model dosyasının yolu')
    parser.add_argument('--ground-truth', type=str, help='Gerçek etiket dosyasının yolu')
    parser.add_argument('--tesseract-path', type=str, help='Tesseract uygulamasının yolu')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Tespit için güven eşiği')
    parser.add_argument('--display', action='store_true', help='Sonuçları göster')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Tespit modülünü başlat
    detector = PlateDetector(model_path=args.model)
    detector.set_confidence_threshold(args.conf_threshold)
    
    # OCR modülünü başlat
    ocr = PlateOCR(tesseract_path=args.tesseract_path)
    
    if args.image:
        # Tek görüntüyü işle
        recognized_plates, result_image = process_single_image(
            args.image, detector, ocr, save_results=True, display=args.display
        )
        
        # Sonuçları yazdır
        print("Tanınan plakalar:")
        for plate in recognized_plates:
            print(f"Metin: {plate['text']} (Güven: {plate['ocr_confidence']:.2f})")
    
    elif args.dataset:
        # Veri setini işle
        evaluator = process_dataset(args.dataset, detector, ocr, args.ground_truth)
    
    else:
        print("Hata: --image veya --dataset argümanı belirtilmeli")
        exit(1) 