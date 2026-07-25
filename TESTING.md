# TESTING

Test senaryoları, sonuçları ve sistemin sınırları. Kararların gerekçeleri ve süreç
için DEVLOG.md'ye bakın; bu doküman son durumu ve ölçülen sonuçları özetler.

Kapsam notu: şu an yalnızca **parsing / OCR katmanı** ölçüldü (Gün 2 itibarıyla).
Retrieval, generation ve abstention bölümleri kendi değerlendirme düzenekleri
kurulunca eklenecek.

---

## 1. Parsing / OCR 

### Metodoloji

- **Metrik: CER** (Character Error Rate)
- **Referans tasarımı:** Aynı belgenin dijital sürümünün metin katmanı ground truth
  olarak kullanılıyor. Manuel transkripsiyon yok.
- **Dil ayrımı:** Hata metriği Türkçe ve İngilizce olarak ayrı raporlandı.
- **worst-case** = İki dil arasında daha yüksek CER üreten sonuç dikkate alındı.

### OCR Testi (Gün 2)

9 belgenin tüm sayfaları için (metin katmanlı belgeler dahil) dört yöntem denendi ve dil ayrımlı 
CER değerleri aşağıda paylaşıldı. Varsayılan olarak seçilen modelin 
sebebi DEVLOG.md dosyasında aktarılmıştır.

| Yöntem | CER TR | CER EN | worst-case | s/belge |
|---|---|---|---|---|
| **ocr_raw** | %10.6 | %14.1 | %14.1 | 18.3 |
| ocr_preprocessed | %13.7 | %13.3 | %13.7 | 19.1 |
| easyocr | %9.4 | %47.6 | %47.6 | 41.5 |
| rapidocr | %12.2 | %50.3 | %50.3 | 25.5 | 

### Tesseract parametre taraması (Gün 2)

Varsayılan yöntemin (ocr_raw) üç ekseni tarandı: tessdata (fast/best) × psm (3/4/6)
× DPI (200/300/400). Sonuçlar aşağıdaki tabloda paylaşıldı.

| config | CER TR | CER EN | worst-case | s/belge |
|---|---|---|---|---|
| **fast_psm3_300** | %10.63 | %14.12 | %14.12 | 17.9 |
| fast_psm3_200 | %11.75 | %13.33 | %13.33 | 14.6 |
| fast_psm3_400 | %10.62 | %14.48 | %14.48 | 20.8 |
| best_psm3_300 | %10.70 | %14.82 | %14.82 | 35.5 |
| fast_psm4_300 | %10.61 | %39.48 | %39.48 | 17.9 |
| best_psm4_300 | %10.65 | %40.27 | %40.27 | 35.8 |
| fast_psm6_300 | %10.02 | %46.50 | %46.50 | 17.4 |
| best_psm6_300 | %9.99 | %47.45 | %47.45 | 35.3 |

## Sınırlar

Parsing / OCR:
- Tablolar: OCR satır-sütun yapısını düzleştiriyor, CER bunu tam yansıtmıyor.
- Grafik içi veri: yalnızca grafikte olan değerler metin katmanında da OCR'da da güvenilir çıkmıyor.
- Çok sütunlu düzen: arXiv gibi belgelerde CER yükseliyor (~%14).
- Fotoğraf tespiti: koşullu ön işleme kararı verildi, ancak girdinin fotoğraf olduğunu tespit mekanizması henüz kurulmadı.