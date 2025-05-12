import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import json
from datetime import datetime
import csv

class EvaluationMetrics:
    def __init__(self, save_dir='results'):
        """
        Değerlendirme metrikleri modülünü başlat
        
        Parametreler:
            save_dir: Değerlendirme sonuçlarının kaydedileceği dizin
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Değerlendirme sonuçlarını takip et
        self.results = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'ocr_correct': 0,
            'ocr_incorrect': 0,
            'total_plates': 0,
            'plate_detections': []
        }
    
    def evaluate_detection(self, ground_truth_boxes, detected_boxes, iou_threshold=0.5):
        """
        Plaka tespiti performansını değerlendir
        
        Parametreler:
            ground_truth_boxes: Gerçek plaka konumları listesi
            detected_boxes: Tespit edilen plaka konumları listesi
            iou_threshold: IoU eşik değeri
            
        Dönüş:
            precision, recall, f1 değerleri
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Her bir gerçek plaka için en iyi eşleşmeyi bul
        matched_detections = set()
        
        for gt_box in ground_truth_boxes:
            best_iou = 0
            best_match = None
            
            for i, det_box in enumerate(detected_boxes):
                if i in matched_detections:
                    continue
                
                iou = self._calculate_iou(gt_box[:4], det_box[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
            
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_detections.add(best_match)
            else:
                false_negatives += 1
        
        # Eşleşmeyen tespitler yanlış pozitiflerdir
        false_positives = len(detected_boxes) - len(matched_detections)
        
        # Sonuçları kaydet
        self.results['true_positives'] += true_positives
        self.results['false_positives'] += false_positives
        self.results['false_negatives'] += false_negatives
        
        # Metrikleri hesapla
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def evaluate_ocr(self, ground_truth_texts, recognized_texts):
        """
        OCR performansını değerlendir
        
        Parametreler:
            ground_truth_texts: Gerçek plaka metinleri listesi
            recognized_texts: Tanınan plaka metinleri listesi
            
        Dönüş:
            karakter_doğruluğu, tam_eşleşme_doğruluğu
        """
        # Doğru karakter tanımalarını say
        character_correct = 0
        character_total = 0
        exact_matches = 0
        
        for gt_text, rec_text in zip(ground_truth_texts, recognized_texts):
            if gt_text == rec_text:
                exact_matches += 1
                self.results['ocr_correct'] += 1
            else:
                self.results['ocr_incorrect'] += 1
            
            # Karakter seviyesinde doğruluğu hesapla
            min_len = min(len(gt_text), len(rec_text))
            character_correct += sum(1 for i in range(min_len) if gt_text[i] == rec_text[i])
            character_total += max(len(gt_text), len(rec_text))
        
        # Metrikleri hesapla
        character_accuracy = character_correct / character_total if character_total > 0 else 0
        exact_match_accuracy = exact_matches / len(ground_truth_texts) if len(ground_truth_texts) > 0 else 0
        
        return character_accuracy, exact_match_accuracy
    
    def _calculate_iou(self, box1, box2):
        """
        İki sınırlayıcı kutu arasındaki Kesişim/Birleşim (IoU) oranını hesapla
        
        Parametreler:
            box1: Birinci kutu [x1, y1, x2, y2]
            box2: İkinci kutu [x1, y1, x2, y2]
            
        Dönüş:
            IoU değeri
        """
        # Kesişim dikdörtgeninin koordinatlarını belirle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        # Kesişim olup olmadığını kontrol et
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Kesişim alanını hesapla
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Her iki kutunun alanını hesapla
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Birleşim alanını hesapla
        union_area = box1_area + box2_area - intersection_area
        
        # IoU'yu hesapla
        iou = intersection_area / union_area
        
        return iou
    
    def add_detection_result(self, image_name, ground_truth, detection, recognition):
        """
        Tespit sonucunu değerlendirmeye ekle
        
        Parametreler:
            image_name: Test görüntüsünün adı
            ground_truth: Gerçek bilgiler (kutu, metin)
            detection: Tespit sonucu (kutu, güven)
            recognition: OCR sonucu (metin, güven)
        """
        self.results['plate_detections'].append({
            'image_name': image_name,
            'ground_truth': ground_truth,
            'detection': detection,
            'recognition': recognition
        })
    
    def plot_results(self, title="Plaka Tespiti ve Tanıma Sonuçları"):
        """
        Değerlendirme sonuçlarını görselleştir
        
        Parametreler:
            title: Grafikler için başlık
            
        Dönüş:
            Kaydedilen grafik dosyalarının yolu
        """
        # Dosya kaydetmek için zaman damgası oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Tespit metriklerini hesapla
        detection_metrics = self._calculate_overall_metrics()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Grafik 1: Tespit metrikleri
        metrics = ['Hassasiyet', 'Duyarlılık', 'F1-Skoru', 'Doğruluk']
        values = [
            detection_metrics['precision'], 
            detection_metrics['recall'], 
            detection_metrics['f1'],
            detection_metrics['accuracy']
        ]
        
        ax1.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        ax1.set_ylim(0, 1.0)
        ax1.set_title('Tespit Performans Metrikleri')
        ax1.set_ylabel('Skor')
        
        for i, v in enumerate(values):
            ax1.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        # Grafik 2: OCR doğruluğu
        ocr_metrics = ['Karakter Doğruluğu', 'Tam Eşleşme']
        ocr_values = [
            detection_metrics['character_accuracy'],
            detection_metrics['exact_match_accuracy']
        ]
        
        ax2.bar(ocr_metrics, ocr_values, color=['#9b59b6', '#1abc9c'])
        ax2.set_ylim(0, 1.0)
        ax2.set_title('OCR Performans Metrikleri')
        ax2.set_ylabel('Doğruluk')
        
        for i, v in enumerate(ocr_values):
            ax2.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Grafiği kaydet
        plot_path = os.path.join(self.save_dir, f'metrics_plot_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Metrikleri CSV'ye kaydet
        csv_path = os.path.join(self.save_dir, f'metrics_{timestamp}.csv')
        self._save_metrics_to_csv(detection_metrics, csv_path)
        
        return plot_path, csv_path
    
    def _calculate_overall_metrics(self):
        """
        Genel performans metriklerini hesapla
        
        Dönüş:
            Metrikler sözlüğü
        """
        # Tespit metrikleri
        tp = self.results['true_positives']
        fp = self.results['false_positives']
        fn = self.results['false_negatives']
        
        # Hassasiyet, duyarlılık ve F1 skorunu hesapla
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Doğruluğu hesapla
        total = tp + fp + fn
        accuracy = tp / total if total > 0 else 0
        
        # OCR metrikleri
        total_ocr = self.results['ocr_correct'] + self.results['ocr_incorrect']
        character_accuracy = self.results['ocr_correct'] / total_ocr if total_ocr > 0 else 0
        exact_match_accuracy = self.results['ocr_correct'] / self.results['total_plates'] if self.results['total_plates'] > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'character_accuracy': character_accuracy,
            'exact_match_accuracy': exact_match_accuracy
        }
    
    def _save_metrics_to_csv(self, metrics, csv_path):
        """
        Değerlendirme metriklerini CSV dosyasına kaydet
        
        Parametreler:
            metrics: Kaydetmek için metrikler sözlüğü
            csv_path: Kaydetmek için CSV dosyasının yolu
        """
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metrik', 'Değer'])
            for metric, value in metrics.items():
                writer.writerow([metric, f'{value:.4f}']) 