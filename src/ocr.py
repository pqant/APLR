import cv2
import numpy as np
import pytesseract
import re

class PlateOCR:
    def __init__(self, tesseract_path=None):
        """
        Plaka OCR modülünü başlat
        
        Parametreler:
            tesseract_path: Tesseract uygulamasının yolu (Windows'ta gerekli)
        """
        # Tesseract yolunu ayarla (eğer belirtildiyse)
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def recognize_plate_oldddd(self, plate_image, preprocess=True):
        """
        Plaka görüntüsündeki karakterleri tanı
        
        Parametreler:
            plate_image: Kırpılmış plaka görüntüsü
            preprocess: OCR öncesi ön işleme yapılıp yapılmayacağı
            
        Dönüş:
            plate_text: Tanınan plaka metni
            confidence: Genel güven skoru
        """
        if preprocess:
            # OCR için özel ön işleme uygula
            processed_image = self._preprocess_for_ocr(plate_image)
        else:
            processed_image = plate_image
        
        # Tesseract parametrelerini yapılandır
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        # OCR işlemini gerçekleştir
        ocr_result = pytesseract.image_to_data(processed_image, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # Metin ve güven değerlerini çıkar
        plate_text = ""
        confidence_sum = 0
        confidence_count = 0
        
        for i in range(len(ocr_result['text'])):
            if int(ocr_result['conf'][i]) > 0:  # Düşük güvenilirlikli sonuçları filtrele
                plate_text += ocr_result['text'][i]
                confidence_sum += float(ocr_result['conf'][i])
                confidence_count += 1
        
        # Ortalama güveni hesapla
        confidence = confidence_sum / confidence_count if confidence_count > 0 else 0
        
        # Plaka metnini temizle
        plate_text = self._clean_plate_text(plate_text)
        
        return plate_text, confidence
    
    def recognize_plate(self, processed_plate_image):
        """
        Recognizes characters from the preprocessed plate image.
        """
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        # PSM Modları:
        # --psm 6: Assume a single uniform block of text. (Genel metin blokları için)
        # --psm 7: Treat the image as a single text line. (Plakalar için genellikle iyi)
        # --psm 8: Treat the image as a single word.
        # --psm 11: Sparse text. Find as much text as possible in no particular order.
        # --psm 13: Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

        # tessedit_char_whitelist: Sadece bu karakterleri tanımasını sağlar.
        # Türkiye'deki plakalarda genellikle büyük harfler ve rakamlar bulunur.
        # İhtiyaç duyarsanız Türkçe karakterleri de ekleyebilirsiniz: ABCDEFGHIJKLMNOPQRSTUVWXYZÖÇŞİĞÜ0123456789
        # Ancak UFPR-ALPR veri seti Brezilya plakalarını içeriyorsa, oradaki karakter setine göre ayarlayın.
        # Genellikle sadece büyük harf ve rakamlar yeterli olur.

        try:
            text = pytesseract.image_to_string(processed_plate_image, lang='eng', config=custom_config) # 'eng' veya 'por' deneyin
            # Güven skorunu almak için image_to_data kullanabilirsiniz, ancak bu daha karmaşıktır.
            # Basitlik için image_to_string'den dönen metni kullanıyoruz.
            # Tesseract doğrudan bir "güven skoru" vermez image_to_string ile.
            # image_to_data ile kelime bazında güven skorları alınabilir.
            # Şimdilik temsili bir güven skoru (örneğin, karakter sayısı veya desen eşleşmesi)
            # analyze_results içinde hesaplanabilir veya basitçe 0.0 döndürülebilir.
            confidence = 0.0 # Bu değeri analyze_results'ta daha anlamlı hale getirebilirsiniz.
            if text.strip(): # Eğer metin boş değilse, basit bir güven temsili
                confidence = len(text.strip()) / 10.0 # Örnek bir hesaplama, geliştirilmeli
                confidence = min(confidence, 1.0)


        except Exception as e:
            print(f"Tesseract OCR hatası: {e}")
            text = ""
            confidence = 0.0
        
        return text.strip(), confidence # .strip() ile baş ve sondaki boşlukları kaldırın    
    
    def _preprocess_for_ocr(self, image):
        """
        Daha iyi OCR sonuçları için plaka görüntüsünü ön işle
        
        Parametreler:
            image: Giriş plaka görüntüsü
            
        Dönüş:
            Ön işlenmiş görüntü
        """
        # Gerekirse gri tonlamaya çevir
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Gürültüyü azaltmak için Gaussian bulanıklaştırma uygula
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # İkili görüntü oluşturmak için eşikleme uygula
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Kontrastı artır
        kernel = np.ones((1, 1), np.uint8)
        erode = cv2.erode(thresh, kernel, iterations=1)
        dilate = cv2.dilate(erode, kernel, iterations=1)
        
        return dilate
    
    def _clean_plate_text(self, text):
        """
        Geçerli bir plaka formatı elde etmek için OCR sonucunu temizle
        
        Parametreler:
            text: Ham OCR metni
            
        Dönüş:
            Temizlenmiş metin
        """
        # Boşlukları kaldır
        text = text.strip().replace(" ", "")
        
        # Özel karakterleri kaldır
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        return text
    
    def analyze_results_olddddd(self, original_text, processed_text):
        """
        OCR sonuçlarını analiz et ve gerekirse düzeltmeler öner
        
        Parametreler:
            original_text: Orijinal OCR sonucu
            processed_text: İşlenmiş ve temizlenmiş OCR sonucu
            
        Dönüş:
            final_text: Son önerilen plaka metni
            is_valid: Plaka formatının geçerli olup olmadığı
        """
        # Plaka formatının geçerli olup olmadığını kontrol et
        # Brezilya plaka formatı: 3 harf + 4 rakam (örneğin AXV8804)
        br_pattern = r'^[A-Z]{3}\d{4}$'
        
        if re.match(br_pattern, processed_text):
            is_valid = True
        else:
            is_valid = False
        
        # Basitlik için, işlenmiş metni final metin olarak döndür
        final_text = processed_text
        
        return final_text, is_valid 
    

    # src/ocr.py dosyasındaki analyze_results metodu için öneri (PlateOCR sınıfı içinde):

    def analyze_results(self, ocr_text, original_detected_text_unused=None): # İkinci parametreye şimdilik ihtiyacımız yok gibi
        """
        Analyzes the OCR results to clean and validate the plate text.
        """

        import re

        # 1. Sadece izin verilen karakterleri tut (whitelist ile zaten yapılıyor ama çift kontrol)
        cleaned_text = re.sub(r'[^A-Z0-9]', '', ocr_text.upper()) # Sadece büyük harf ve rakam

        # 2. Uzunluk Kontrolü (Plaka formatına göre)
        # Örneğin, Türkiye'deki genel formatlar (örn: 34 ABC 123, 34 AB 1234)
        # UFPR-ALPR Brezilya plakaları için format farklı olacaktır (genellikle ABC1D23 veya ABC1234)
        # Bu kısmı veri setinizdeki plaka formatına göre uyarlayın.
        is_valid = False
        # Basit bir örnek: Eğer 6-8 karakter uzunluğundaysa ve çoğunluğu rakam/harf ise geçerli sayalım.
        if 6 <= len(cleaned_text) <= 8: # Brezilya formatları için 7 daha yaygın
             # Daha karmaşık desen kontrolleri eklenebilir (regex ile)
            if re.match(r'^[A-Z]{2,3}[0-9]{1,2}[A-Z0-9]{1,2}$', cleaned_text) or \
               re.match(r'^[A-Z]{3}[0-9]{4}$', cleaned_text) or \
               re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', cleaned_text): # Brezilya Mercosur formatı (ABC1D23)
                is_valid = True
        
        # Daha gelişmiş analizler eklenebilir (örneğin, karakter olasılıkları, desen eşleştirme vb.)
        
        return cleaned_text, is_valid 