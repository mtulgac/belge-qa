# TESTING

Test senaryoları, sonuçları ve sistemin sınırları. Kararların gerekçeleri ve süreç
için DEVLOG.md'ye bakın; bu doküman son durumu ve ölçülen sonuçları özetler.

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

### Fotoğraf tespiti (Gün 3)

Ön işleme yalnız fotoğrafta kazandırdığı için ingest'te koşullu çalışıyor. Kapının
hangi sinyale bağlanacağı ölçüldü; "doğru metod" sütunu CER metriğine göre seçildi.

| belge | yayılım farkı | seçim | CER ham | CER ön | doğru metod |
|---|---|---|---|---|---|
| taranmış | 0.0 | ham  | %0.34 | %0.30 | berabere |
| sentetik gölgeli | 135.0 | ön  | %26.14 | %0.27 | ön |
| gerçek fotoğraf | 72.0 | ön  | %21.50 | %3.73 | ön |
| ekran görüntüsü | 0.0 | ham  | %3.27 | %33.75 | ham |
| dijital render | 0.0 | ham  | %0.30 | %0.30 | berabere |

- Aydınlatma yayılımı bazlı metod seçimi: **3/3 doğru** (diğer ikisi berabere). Eşik 25.0.
---

## 2. Retrieval (dense)

### Metodoloji

- **Ölçüm birimi çapa:** golden set'te cevabın yeri `dosya + sayfa + alıntı` ile
  işaretli. Bir chunk "doğru" sayılıyorsa, alıntıyı içeriyor demektir. Metrik chunk
  boyutundan bağımsız. Chunking boyutu değişince yeniden etiketleme gerekmiyor.
- **Metrikler:** recall@k, precision@k, MRR; TR ve EN **ayrı**.
- **Kapsam:** bugün yalnız dense retrieval (kosinüs). Hibrit ve reranking yok.
- **Varyant grubu:** `tobb_yonetmelik.pdf | tobb_taranmis.pdf | karma.pdf` aynı içeriğin
  üç formatı olduğu için indekste aynı anda yalnız biri bulunuyor. Ana metrikler dijital varyantla
  ölçüldü.
- **`izole`:** yalnız q061'de kullanılıyor (aynı bilgi hem EN hem de TR bültende de var). Ölçüm amaçlı;
  çalışan sistemde belge gizleyen bir filtre yok.
- **İndeks:** 311 chunk, hedef 800 / örtüşme 150 karakter, `CHUNK_MAX` 1600. En uzun chunk
  411 token. 1600 sınırı e5-small'ın 512 token'lık penceresi için kondu; donan model bge-m3'ün
  penceresi 8192 olduğu için bu gerekçe artık geçerli değil (bkz. DEVLOG Gün 3).

### Embedding modeli karşılaştırması (Gün 3)

Aynı korpus, aynı golden set, aynı chunking; tek değişken model. k=5.

| model | embed_dim | recall@5 | TR | EN | çapraz dil | MRR | AUC | ms/sorgu | s/chunk |
|---|---|---|---|---|---|---|---|---|---|
| **BAAI/bge-m3** | 1024 | %90.0 | %93.8 | %75.0 | %100 | 0.68 | 0.790 | 35 | ~0.46 |
| intfloat/multilingual-e5-base | 768 | %70.0 | %62.5 | %100 | %33.3 | 0.54 | 0.745 | 15 | ~0.18 |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | %65.0 | %62.5 | %75.0 | %66.7 | 0.45 | 0.660 | 8 | ~0.11 |
| intfloat/multilingual-e5-small | 384 | %62.5 | %59.4 | %75.0 | %0.0 | 0.47 | 0.765 | 9 | ~0.10 |


### Dense retrieval sonuçları (bge-m3, k=5) (Gün 3)

| | recall@1 | recall@3 | recall@5 | recall@10 | n |
|---|---|---|---|---|---|
| tümü | %50.0 | %77.5 | %90.0 | %95.0 | 20 |
| TR | %56.2 | %78.1 | %93.8 | %93.8 | 16 |
| EN | %25.0 | %75.0 | %75.0 | %100 | 4 |

| kategori | recall@5 | precision@5 | MRR | n |
|---|---|---|---|---|
| çapraz dil | %100 | %20.0 | 0.56 | 3 |
| çok belge | %100 | %40.0 | 0.62 | 2 |
| tablo | %100 | %30.0 | 0.88 | 4 |
| tek belge | %88.9 | %22.2 | 0.68 | 9 |
| çelişki | %50.0 | %30.0 | 0.50 | 2 |

k=5'te kaçan çapalar: q009 (arXiv/EN) ve q041'in iki çapası (çelişki sorusu, iki belgeden
iki değer birden gerekiyor).

### Reddetme eşiği için skor dağılımı (Gün 3)

| | min | ortalama | medyan | max | n |
|---|---|---|---|---|---|
| yanitla | 0.490 | 0.671 | 0.680 | 0.782 | 20 |
| reddet | 0.343 | 0.568 | 0.572 | 0.667 | 10 |

10 reddet satırının 8'i yanıtlanabilirlerin skor aralığında; ayrılabilirlik AUC 0.790.
**Tek bir kosinüs eşiği abstention kapısı olarak yeterli değil.**

### OCR'ın retrieval'a etkisi (Gün 3)

Aynı 30 soru, tek değişken yönetmelik belgesinin formatı.

| varyant | OCR'lı chunk | recall@5 | TR | EN | MRR | AUC |
|---|---|---|---|---|---|---|
| tobb_yonetmelik (dijital) | 0 | %90.0 | %93.8 | %75.0 | 0.68 | 0.790 |
| tobb_taranmis (taranmış) | 76 | %90.0 | %93.8 | %75.0 | 0.68 | 0.790 |
| karma (s.2 ve 4 taranmış) | 15 | %90.0 | %93.8 | %75.0 | 0.68 | 0.790 |


## Sınırlar

Parsing / OCR:
- Grafik içi veri: yalnızca grafikte olan değerler metin katmanında da OCR'da da güvenilir çıkmıyor.
- Çok sütunlu düzen: arXiv gibi belgelerde CER yükseliyor (~%14).
- Fotoğraf tespiti: kapı kuruldu ve kalibre edildi, ancak korpusta **tek gerçek fotoğraf**
  var; ikinci pozitif örnek sentetik. Farklı aydınlatma koşullarında yeniden bakılmalı.

Retrieval:
- **Örneklem küçük:** 20 yanıtlanabilir soru; EN yalnız 4, çapraz dil 3 satır. Bu
  sütunlarda tek bir soru 25-33 puan oynatıyor, tek başlarına karar verilecek sayılar
  değil. Kararlar TR (n=16) ve iki dilin worst-case'i üzerinden verildi.
- **Çelişki kategorisi en zayıf halka** (%50): iki belgeden iki değerin birden gelmesi
  gerekiyor, tek eşleşme yetmiyor.
- **Reddetme eşiği henüz ayarlanmadı:** kosinüs skoru ayrım için yetersiz (AUC 0.790,
  8/10 örtüşme). Reranker skoru ölçülene kadar reddetme davranışı ölçülmüş sayılmaz.
- **OCR maliyeti sonucu temiz taramayla sınırlı:** varyant grubundaki tarama %0.13 CER.
  Daha kötü OCR'da retrieval'ın nerede kırıldığı ölçülmedi.
- **Chunk parametreleri taranmadı:** 800/150 makul başlangıç değerleri, ölçümle
  seçilmediler. `CHUNK_MAX=1600` ölçümle kondu ama e5-small'ın penceresine göre; bge-m3'te
  o kısıt yok, dolayısıyla bugünkü değerin ölçülmüş bir gerekçesi de yok.
- Tablo soruları doğru chunk'ı bulma seviyesinde %100; düzleşmiş tablodan doğru hücrenin
  okunması generation aşamasında ayrıca test edilecek.