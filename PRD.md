## 📄 Plaka Tanıma Uygulaması – Ürün Gereksinimleri Dokümanı (PRD)

### 1. **Amaç**

Bu proje, genel kabul görmüş veri kümeleri üzerinde çalışarak plaka tanıma algoritmalarını uygulamayı amaçlamaktadır. Hedef, YOLOv8 derin öğrenme modeli kullanılarak araç plakalarının tespit edilmesi ve bu plakalardaki karakterlerin tanınmasıdır. Nihai çıktılar doğruluk ve başarım analizleriyle birlikte kısa bir teknik rapor şeklinde sunulacaktır.

---

### 2. **Kapsam**

* Sadece **statik görüntüler** üzerinde çalışılacaktır (video, canlı akış yoktur).
* Plaka tespiti YOLOv8 ile yapılacaktır.
* Karakter tanıma klasik OCR (örneğin Tesseract) veya basit CNN ile gerçekleştirilebilir.
* Tüm sonuçlar raporlanacak, görseller ve analizler dosyalanacaktır.

---

### 3. **Zorunlu Teknolojiler ve Kısıtlar**

| Alan                    | Tercih                                                                  |
| ----------------------- | ----------------------------------------------------------------------- |
| Derin Öğrenme Modeli    | ✅ **YOLOv8 (Ultralytics)** – zorunludur                               |
| Programlama Dili        | Python                                                                  |
| Kütüphaneler            | `ultralytics`, `opencv-python`, `pytesseract`, `matplotlib`, `numpy`    |
| Derin Öğrenme Çerçevesi | PyTorch (YOLOv8 gereği)                                                |
| Kullanım Ortamı         | Jupyter Notebook veya `.py` script                                      |

---

### 4. **Kullanılacak Veri Kümeleri**

Projede UFPR-ALPR veri seti kullanılacaktır:

**UFPR-ALPR Dataset**
🔗 [https://web.inf.ufpr.br/vri/databases/ufpr-alpr/](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/)
➤ Gerçek dünya görüntüleri içerir, RGB ve etiketli plaka verisi sağlar.

---

### 5. **İş Akışı ve Aşamalar**

#### 5.1. Görüntü Ön İşleme

* Gri tonlama, gürültü azaltma
* Histogram eşitleme, kenar bulma

#### 5.2. Plaka Tespiti

* YOLOv8 ile eğitimli veya yeniden eğitilmiş model üzerinden plaka kutusu tespiti

#### 5.3. Karakter Tanıma

* OCR ile karakterlerin çıkarılması (`pytesseract`)
* Alternatif olarak: Basit CNN tabanlı tanıma (isteğe bağlı)

#### 5.4. Değerlendirme ve Analiz

* Doğruluk, precision, recall gibi metriklerle analiz
* Örnek görüntülerle sonuçların gösterimi
* Başarı oranlarının grafiksel sunumu

---

### 6. **Beklenen Çıktılar**

* Her test görseli için:

  * Orijinal görüntü
  * Tespit edilen plaka kutusuyla birlikte işlenmiş çıktı
  * Tanınan plaka karakter dizisi
  * Performans analizi (% doğruluk, precision/recall)
  * PDF formatında teknik rapor:
  * Kullanılan yöntemler
  * Elde edilen sonuçlar
  * Karşılaşılan problemler ve çözüm yolları

---

### 7. **Teslimat İçeriği**

* Kodlar: `.ipynb` veya `.py`
* Görseller: Girdi ve çıktıların yer aldığı klasör
* Rapor: PDF (2–5 sayfa)
* README: Projenin nasıl çalıştırılacağı