import cv2
import numpy as np

def grayscale(image):
    """
    Görüntüyü gri tonlamaya çevir
    
    Parametreler:
        image: Giriş görüntüsü
        
    Dönüş:
        Gri tonlamalı görüntü
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Gürültüyü azaltmak için Gaussian bulanıklaştırma uygula
    
    Parametreler:
        image: Giriş görüntüsü
        kernel_size: Gaussian çekirdeğinin boyutu
        
    Dönüş:
        Bulanıklaştırılmış görüntü
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_histogram_equalization(image):
    """
    Kontrastı iyileştirmek için histogram eşitleme uygula
    
    Parametreler:
        image: Gri tonlamalı giriş görüntüsü
        
    Dönüş:
        Histogram eşitlenmiş görüntü
    """
    return cv2.equalizeHist(image)

def detect_edges(image, low_threshold=50, high_threshold=150):
    """
    Canny kenar tespiti uygula
    
    Parametreler:
        image: Gri tonlamalı giriş görüntüsü
        low_threshold: Histerezis prosedürü için alt eşik değeri
        high_threshold: Histerezis prosedürü için üst eşik değeri
        
    Dönüş:
        Kenar görüntüsü
    """
    return cv2.Canny(image, low_threshold, high_threshold)

def apply_morphological_operations(image, operation='close', kernel_size=5):
    """
    Morfolojik işlemler uygula (genişletme, erozyon, açma, kapama)
    
    Parametreler:
        image: İkili giriş görüntüsü
        operation: İşlem türü ('dilate', 'erode', 'open', 'close')
        kernel_size: Yapısal elemanın boyutu
        
    Dönüş:
        İşlenmiş görüntü
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'dilate':
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == 'erode':
        return cv2.erode(image, kernel, iterations=1)
    elif operation == 'open':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:
        return image

def preprocess_image_for_plate_detection(image):
    """
    Plaka tespiti için görüntüyü ön işle
    
    Parametreler:
        image: Giriş renkli görüntüsü
        
    Dönüş:
        Ön işlenmiş görüntü
    """
    # Gri tonlamaya çevir
    gray = grayscale(image)
    
    # Gürültüyü azaltmak için Gaussian bulanıklaştırma uygula
    blurred = apply_gaussian_blur(gray)
    
    # Kontrastı iyileştirmek için histogram eşitleme uygula
    equalized = apply_histogram_equalization(blurred)
    
    # Ön işlenmiş görüntüyü döndür
    return equalized

def preprocess_plate_for_ocr_v1(plate_image):
    """
    OCR için tespit edilen plaka görüntüsünü ön işle
    
    Parametreler:
        plate_image: Kırpılmış plaka görüntüsü
        
    Dönüş:
        OCR için hazır ön işlenmiş plaka görüntüsü
    """
    # Henüz değilse gri tonlamaya çevir
    if len(plate_image.shape) == 3:
        plate_image = grayscale(plate_image)
    
    # Histogram eşitleme uygula
    equalized = apply_histogram_equalization(plate_image)
    
    # İkili görüntü elde etmek için eşikleme uygula
    _, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Gürültüyü gidermek için morfolojik işlemler uygula
    processed = apply_morphological_operations(binary, 'close', 3)
    
    return processed 

# src/preprocessing.py dosyasındaki preprocess_plate_for_ocr fonksiyonu için öneri:
import cv2
import numpy as np

def preprocess_plate_for_ocr(plate_img):
    """
    Preprocesses the license plate image for better OCR results.
    """
    # 1. Gri Tonlama (Grayscale Conversion)
    gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # 2. Yeniden Boyutlandırma (Resizing) - İsteğe bağlı ama genellikle faydalı
    # Karakter yüksekliğinin en az 30-40 piksel olmasını hedefleyin.
    # Oranları koruyarak yeniden boyutlandırma:
    height, width = gray_plate.shape
    # Örneğin, yüksekliği 100 piksel yapalım (oranı koruyarak)
    target_height = 100 
    scale_ratio = target_height / height
    target_width = int(width * scale_ratio)
    resized_plate = cv2.resize(gray_plate, (target_width, target_height), interpolation=cv2.INTER_CUBIC) # INTER_CUBIC veya INTER_LINEAR deneyin

    # 3. Gürültü Azaltma (Noise Reduction) - Duruma göre
    # blurred_plate = cv2.medianBlur(resized_plate, 3) # 3 veya 5 gibi tek bir çekirdek boyutu
    # VEYA daha kenar koruyucu bir filtre:
    blurred_plate = cv2.bilateralFilter(resized_plate, 9, 75, 75)


    # 4. Binarizasyon (Binarization) - Bu adım çok önemli!
    # Otsu's binarization genellikle iyi bir başlangıçtır.
    # Arka planın mı metnin mi daha açık olduğuna göre THRESH_BINARY veya THRESH_BINARY_INV kullanın.
    # Plakalarda genellikle koyu metin açık arka plan üzerindedir.
    _, binary_plate = cv2.threshold(blurred_plate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Alternatif olarak adaptiveThreshold deneyebilirsiniz:
    # binary_plate = cv2.adaptiveThreshold(blurred_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                                     cv2.THRESH_BINARY_INV, 11, 2) # Parametreleri ayarlamanız gerekebilir

    # 5. İsteğe Bağlı: Morfolojik İşlemler (Morphological Operations)
    # Gürültüyü daha da azaltmak veya karakterleri birleştirmek/ayırmak için.
    # kernel = np.ones((1,1),np.uint8) # Kernel boyutunu ayarlayın
    # binary_plate = cv2.erode(binary_plate, kernel, iterations = 1)
    # binary_plate = cv2.dilate(binary_plate, kernel, iterations = 1)

    return binary_plate    