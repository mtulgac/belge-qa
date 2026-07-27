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

## 2. Retrieval

### Metodoloji

- **Ölçüm birimi çapa:** golden set'te cevabın yeri `dosya + sayfa + alıntı` ile
  işaretli. Bir chunk "doğru" sayılıyorsa, alıntıyı içeriyor demektir. Metrik chunk
  boyutundan bağımsız. Chunking boyutu değişince yeniden etiketleme gerekmiyor.
- **Metrikler:** recall@k, precision@k, MRR; TR ve EN **ayrı**.
- **AUC** = bir `yanitla` satırının bir `reddet` satırını geçme olasılığı (Mann-Whitney); 
1.0 tam ayrım, 0.5 yazı-tura. 
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

### Hibrit retrieval: BM25 + RRF (Gün 4)

**RRF parametre taraması** (18 kombinasyon; derinlik × RRF sabiti × ağırlık, seçilmiş satırlar):

| depth / k / ağırlık (dense:bm25) | recall@5 | TR | EN | MRR | çapraz dil |
|---|---|---|---|---|---|
| 50 / 60 / 1:1 (başlangıç*) | %82.5 | %78.1 | %100 | 0.67 | %66.7 |
| 50 / 10 / 1:1 | %82.5 | %84.4 | %75.0 | 0.67 | %100 |
| 20 / 60 / 1:1 | %87.5 | %84.4 | %100 | 0.68 | %100 |
| 10 / 10 / 1:1 | %90.0 | %87.5 | %100 | 0.69 | %100 |
| **20 / 10 / 2:1** | **%92.5** | %90.6 | %100 | **0.72** | %100 |
| 20 / 10 / 3:1 | %92.5 | %90.6 | %100 | 0.72 | %100 |

\* Başlangıç ayarının kaynağı üç farklı yer: `k=60` RRF'i öneren makaleden
  (Cormack, Clarke & Buettcher, 2009); **eşit ağırlık** orijinal formülün kendisi
  (`Σ 1/(k+sıra)`, ağırlık kavramı yok — ağırlık ekleyerek formülün dışına çıkıldı);
  `depth=50` ise yalnızca yaygın bir pratik değer, ölçüye dayanmıyor. Yani bu satır
  "literatürün önerdiği ayar" değil.

**Hibrit Retrieval Karşılaştırma (k=5, dijital varyant, bge-m3):**

| retriever | recall@5 | TR | EN | precision@5 | MRR | çapraz dil | AUC |
|---|---|---|---|---|---|---|---|
| dense | %90.0 | **%93.8** | %75.0 | %26.0 | 0.68 | %100 | 0.790 |
| bm25 tek başına | %62.5 | %59.4 | %75.0 | %21.0 | 0.47 | **%0** | 0.640 |
| **hibrit d20/k10/w2:1** | **%92.5** | %90.6 | **%100** | %26.0 | **0.72** | %100 | 0.445* |

\* Hibritin AUC'si RRF skoru üzerinden; dense'inkiyle **karşılaştırılamaz**. RRF skoru
füzyon sabitiyle sınırlı, 30 sorunun tamamı 0.0909'luk bir aralıkta (0.182–0.273) ve
beraberliklerle dolu.

**Hibrit retrieval gecikmesi (Gün 4)**

| retriever | medyan | ortalama | min | max |
|---|---|---|---|---|
| dense | 34.4 ms | 35.9 ms | 30.9 | 52.2 |
| bm25 | 0.0 ms | 0.2 ms | 0.0 | 14.2 |
| **hibrit** | **33.9 ms** | 35.0 ms | 31.0 | 56.0 |


### OCR'ın retrieval'a etkisi (Gün 3 dense, Gün 4 hibrit)

Aynı 30 soru, tek değişken yönetmelik belgesinin formatı.

| varyant | OCR'lı chunk | dense r@5 | dense TR | hibrit r@5 | hibrit TR |
|---|---|---|---|---|---|
| tobb_yonetmelik (dijital) | 0 | %90.0 | %93.8 | %92.5 | %90.6 |
| tobb_taranmis (taranmış) | 76 | %90.0 | %93.8 | %92.5 | %90.6 |
| karma (s.2 ve 4 taranmış) | 15 | %90.0 | %93.8 | %92.5 | %90.6 |

İki retriever'da da **soru bazında tek bir değişiklik yok**. Hibritin dense'ten daha
kırılgan olması beklenirdi (BM25'te tek karakterlik OCR hatası token'ı tamamen
kaçırtır, dense'te vektör kademeli bozulur) fakat bu korpusta gözlenmedi, çünkü varyant
grubundaki tarama %0.13 CER ile fazla temiz.

### Cross-encoder rerank (Gün 4)


**Recall (k=5, depth 20, dijital varyant, bge-m3 embedding):**

| retriever | recall@5 | TR | EN | MRR |
|---|---|---|---|---|
| dense | %90.0 | %93.8 | %75.0 | 0.68 |
| hibrit | %92.5 | %90.6 | %100 | 0.72 |
| rerank bge · hibrit | %95.0 | %93.8 | %100 | 0.80 |
| **rerank bge · dense** | **%97.5** | **%96.9** | %100 | 0.81 |
| rerank mmarco · hibrit | %95.0 | %93.8 | %100 | 0.83 |
| **rerank mmarco · dense** | **%97.5** | **%96.9** | %100 | 0.85 |

**Recall (k=5, depth 10, dijital varyant, bge-m3 embedding):**

| retriever | recall@5 | TR | EN | MRR |
|---|---|---|---|---|
| dense | %90.0 | %93.8 | %75.0 | 0.68 |
| hibrit | %92.5 | %90.6 | %100 | 0.72 |
| rerank bge · hibrit | %95.0 | %93.8 | %100 | 0.80 |
| **rerank bge · dense** | **%95.0** | **%93.8** | %100 | 0.82 |
| rerank mmarco · hibrit | %95.0 | %93.8 | %100 | 0.83 |
| **rerank mmarco · dense** | **%95.0** | **%93.8** | %100 | 0.83 |


**Depth (aday havuzu) — operasyon noktası (seçilen model bge):**

| depth | recall@5 | gecikme/soru (bu CPU) |
|---|---|---|
| 20 | %97.5 | ~6.9 s |
| **10** | **%95.0** | **~3.3 s** |



**Gecikme (CPU):**

Golden set'in tamamı (30 soru) için reranking toplam süresi:

| model | depth 10 | depth 20 |
|---|---|---|
| mmarco (118M) | 87.3 s (2909 ms/soru) | 193.2 s (6442 ms/soru) |
| bge (568M) | 97.8 s (3259 ms/soru) | 206.1 s (6870 ms/soru) |

**Reddetme eşiği — reranker skoru vs kosinüs kapısı, EŞLEŞMELİ (iki bağımsız AUC değil):**

| model | AUC(rerank) | AUC(cosine) | eşleşmeli Δ | %95 CI |
|---|---|---|---|---|
| bge | 0.840 | 0.790 | +0.050 | [−0.075, +0.185] |
| mmarco | 0.800 | 0.790 | +0.010 | [−0.095, +0.105] |


Tek eşik maliyeti (reranker skoru, [0,1]):

| model | t=0.5 (uydurma / gereksiz-red) | 0 uydurma eşiği → feda |
|---|---|---|
| bge | 3 / 3 | 0.952 → 7/20 yanıtlanabilir |
| mmarco | 4 / 0 | 0.999 → 14/20 yanıtlanabilir |



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
- **OCR maliyeti sonucu temiz taramayla sınırlı:** varyant grubundaki tarama %0.13 CER.
  Daha kötü OCR'da retrieval'ın nerede kırıldığı ölçülmedi.
- **Chunk parametreleri taranmadı:** 800/150 makul başlangıç değerleri, ölçümle
  seçilmediler. `CHUNK_MAX=1600` ölçümle kondu ama e5-small'ın penceresine göre; bge-m3'te
  o kısıt yok, dolayısıyla bugünkü değerin ölçülmüş bir gerekçesi de yok.
- Tablo soruları doğru chunk'ı bulma seviyesinde %100; düzleşmiş tablodan doğru hücrenin
  okunması generation aşamasında ayrıca test edilecek.
- **Reddetme eşiği hâlâ çözülmedi.** Kosinüs yetersizdi (AUC 0.790); reranker skoru
  ölçüldü ve en iyi aday sinyal ama eşleşmeli testte kosinüsü n=20/10'da kesin yenmiyor
  (Δ +0.05, CI 0'ı kapsıyor), ve tek eşik kuyruk çakışması yüzünden çalışmıyor
  (0 uydurma için 7-14/20 yanıtlanabilir cevap feda ediliyor). 
