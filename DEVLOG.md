# DEVLOG

Çalışma günlüğü. Kararlar, denediklerim, takıldığım yerler.
Sistemin son hali ve sınırları için TESTING.md dokümanına bakabilirsiniz.

---

## Gün 1 — 23.07.2026

### Dokümandan çıkardığım adımlar

Dokümanı okudum ve sistemi ayrı aşamalara böldüm. Her aşama kendi içinde değerlendirilsin ve kendi hata
metriği olsun istedim, böylece yanlış bir cevap geldiğinde hangi modülün bozulduğunu bulabilirim.

| İster | Modül | Metrik |
|---|---|---|
| PDF + JPG/PNG belgelerden metin okuma | Parsing / OCR | CER |
| Türkçe ve İngilizce desteği| Her modülü etkiliyor | metrikler TR/EN ayrı |
| Doğal dilde soru-cevap | Retrieval + generation | precision@k, recall@k, faithfulness |
| Belgede olmayanı üretmeme | Threshold kullanma + grounding | abstention matrisi |
| Arayüz | Web UI | — |
| Kolay çalıştırılabilirlik | Docker Compose | temiz kurulum testi |


### Akış diyagramları

**İndeksleme (Offline):** Belge yükleme → Belge tipi tespiti → Parsing / OCR → Chunking
(sayfa no ile) → Embedding → İndeksleme

**Sorgu (Online):** Kullanıcı sorusu → Hibrit retrieval (top-k) → Re-ranking (top-m) → Relevance Thresholding →
İlişki yetersizse reddet, yeterliyse top-m chunk'ı LLM'e aktar -> Yanıt üretimi

### Bugünkü testler

İlk gün iskeletin kafamda oturması adına ufak bir test script oluşturdum. Bu script üzerinden yüklenen belgeyi PyMuPDF ile 
parse ettikten sonra chunklara bölüp, multilingual-e5-small modeliyle vektörlere çevirip sonrasında kullanıcı sorgusunu da 
aynı modelle vektöre dönüştürdükten sonra cosine similarity ile top-5 chunk'ı alarak onları local'de ollama üzerinden çalışan
Qwen2.5:1.5b modeline verdim. İki farklı PDF (Türkçe ve İngilizce) ve iki farklı dilde sorgu denedim.

| (Doküman-> Sorgu Dili) | Senaryo | Sonuç |
|---|---|---|
| TR→TR | Cevap belgede mevcut| Doğru cevabı Türkçe olarak getirdi|
| TR→TR | Alakasız soru (Hava durumu) | Doğru şekilde soruyu Türkçe olarak reddetti |
| EN→EN | Cevap belgede mevcut | Doğru cevabı İngilizce olarak getirdi |
| EN→TR | Cevap belgede mevcut | Doğru cevap geldi ama İngilizce olarak getirdi |
| EN→TR | Belgede bulunmayan şirket hakkında bilgi istendi | Cevabı Türkçe verdi ama uydurdu |

Bulgular:

- **Uydurma cevap.** Hava durumunu sorunca reddetti çünkü hiçbir chunk
  eşleşmedi. Ama belgede geçmeyen bir şirketin kuruluş tarihini sorunca uydurdu.
  Sorun cevabın yanlış olması değil, belgeye dayanmaması. Buradan çıkan karar: 
  Gold dataset hazırlanırken yanıtlanamaz sorularda beklenen çıktı doğru cevap değil, reddetme.
- **Cosine skoru eşik için zayıf.** Doğru chunk'larda 0.84, uydurma vakada 0.77.
  Fark var ama çok ciddi bir fark değil. e5 modeli dar bir aralıkta çalışıyor, alakasız metinde bile 0.7
  üstü veriyor. Cross-encoder kullanmamız gerektiği ortada.
- **Dil tutarsızlığı.** Türkçe sorgu + İngilizce belge olunca model bir soruda Türkçe, bir
  soruda İngilizce cevap verdi. Prompt'ta hangi dilde cevap vereceği yazmıyor.
  İlk testte düzeltmedim, ama not aldım.
- **Chunk'lar sayfa sınırında kesiliyor.** Sayfa sonunda başlayan cümle ikiye
  bölünüyor. İlk test için düzeltmedim ama not aldım.

Not: Seçilen hiçbir model nihai değil, hızlıca test edilmesi için seçilmiş modeller. 
Her bir modül için farklı seçenekleri deneyip, en iyi performans veren model ve 
parametreler seçilecek.

### Erken kararlar

**Hibrit retrieval + cross-encoder.** Sadece cosine similarity kullanmayacağım. 
Türkçe'de özel isim ve numara aramalarında zayıf kalıyor, BM25 bunları yakalıyor. 
Reranker skorunu ayrıca relevance thresholding için kullanacağım.

**LLM'de on-prem.** Sistemin şirket belgeleri için kullanılacağını varsayıyorum
ve güvenlik sebebiyle LLM için on-prem şekilde local'de çalışacak model kullanacağım.

**VLM ile OCR yapmayacağım.** Sistem on-prem çalışacak ve zaten LLM kullanıyoruz.
Bir de belgenin taranması tarafında VLM kullanırsam belgenin yüklenip indekslenmesi
ve buna bağlı olarak kullanıcının belgeden istediği bilgiyi alma süresi çok uzayabilir. 

**Dil bazlı metrik hesaplama.** Bütün metrikleri TR ve EN ayrı 
raporlayacağım. Ortalama alırsam İngilizce'nin iyi sonucu Türkçe'yi gizler.

**Tek embedding modeli.** TR ve EN için ayrı model kullanırsam iki ayrı vektör
uzayı olur ve Türkçe soruyla İngilizce belgede arama yapamam. Bu senaryo gerçek
kullanımda olacak.

**Language classifier kullanmıyorum.** Kısa sorularda dil tespiti güvenilmez.
Yanlış tespit, hiç tespit etmemekten kötü çünkü sessizce bozuyor. Cevap dili
için prompt'a bir satır yazmak yeterli olacaktır diye düşünüyorum.

### Açık sorular

- Çoklu belge yükleme kısmını nasıl yapabilirim?
- Multi-turn dialog şeklinde yaparsam nasıl test edebilirim?
- Farklı belge türlerinde (JPG ya da taranmış) parsing performansı nasıl değişecek? 
Tabloları nasıl parse edebilirim?

---

## Gün 2 — 24.07.2026

Bugün, kullanıcı tarafından yüklenecek belgelerin anlamlı metne çevrilmesi 
(Document Ingestion) kısmında çeşitli testler yaparak, bu modülün netleştirilmesi ile 
uğraştım. Öncelikle, farklı belge tiplerindeki başarımı ölçebilmek için örnek belgelerimi
içeren korpusu ayarladım. Bu korpus, 10 belgeden oluşuyor. İçerisinde Türkçe, İngilizce, düz metin,
tablolu, taranmış ve fotoğrafı çekilmiş belgeler bulunuyor. Başarım hesaplarken ground
truth gerektiği için, aynı belgenin farklı versiyonlarını kullandım. Başarım metriği olarak
**CER** (Character Error Rate) seçtim, referans olarak da dijital PDF'in metin katmanını 
kullandım. Daha önce aldığım karar gereği, bütün metrikleri Türkçe ve İngilizce olarak
ayrı raporladım. Buradaki amacım, en iyi parsing yöntemini seçip onunla sonraki aşamalara
devam etmekti.

### needs_ocr kalibrasyonu

Öncelikle, yüklenen belgede OCR gerekip gerekmediğini anlamamız gerekiyor. Bu sebeple birkaç farklı kontrol mekanizmasından oluşan needs_ocr.py dosyasını oluşturdum. Özet olarak bu dosya, yüklenen belgenin her bir sayfasında yeterli metin bulunup bulunmadığına, bozulmuş karakterlerin oranına ve sayfadaki resimlerin sayfaya oranına bakarak, sayfanın OCR ile taranıp taranmamasına karar veriyor. Fakat bu script, 5 farklı parametre içeriyor. Bu parametrelerin kalibrasyon testi için de calibrate.py dosyasını oluşturdum. Elimizdeki örnek belgeler üzerinde OCR kullanımı hakkında doğru kararın verilip verilmediğine bakarak, parametre değerlerini güncelledim. Bu noktadan sonra, OCR gerektiğinde hangi OCR metodunu kullanacağımı seçmek için OCR testi yaptım.

### OCR Testi

Metin katmanı olmadığında nasıl ilerleyebileceğimizi ölçmek adına, birkaç farklı metod denedim.
Bu metodları:
- Tesseract OCR, (ocr_raw)
- Ön İşlemeli Tesseract OCR (ocr_preprocessed)
- EasyOCR (easyocr)
- RapidOCR (rapidocr)

olarak belirledim. Aday OCR modellerinin seçilmesinde, GPU ve VLM kullanmama kararım çok belirleyici
oldu. Bu 4 yöntemi tüm korpus üzerinde denedim (parser.py, parse_all.py ve evaluate.py dosyaları oluşturuldu) ve aşağıdaki bulgulara ulaştım. Önişleme yönteminde yapılan iyileştirmeler sebebiyle bu test yaklaşık 3 kez tekrarlandı. Son testin sonuçlarını TESTING.MD içerisinde paylaştım. 

Bulgular:

- **Kazanan ocr_raw.** Fotoğraf dışındaki neredeyse her belgede ocr_raw önde; ön işleme kaliteli taramada metin kaybettiriyor. Tek istisna kamerayla çekilmiş belge: orada ön işleme %21.50 → %3.73 kazandırıyor. Bu yüzden ön işlemeyi koşullu kullanacağım, fotoğrafta açık, diğerlerinde kapalı. (ocr_preprocessed'in ortalama CER'i daha düşük görünüyor ama bu tek belgenin etkisi.)

- **EasyOCR ve RapidOCR elendi.** arXiv'den aldığımız makalenin çok sütunlu düzeninde bu iki model de çöküş yaşadı. Ayrıca diğer belgelerde de içerik kaybı mevcut (anadolu yönetmeliğinde ~6k, tobb yönetmeliğinde ~3.3k karakter eksik). 


### Tesseract parametre taraması (sweep_tesseract.py)

OCR konusunda Tesseract ile ilerleyeceğim netleştikten sonra, bu modelden optimum performansı almak adına farklı
parametreler ile test yapmak istedim. Üç ekseni birden taradım: tessdata (fast/best) × psm (3/4/6) × DPI (200/300/400, kazanan
üzerinde). Metodoloji evaluate.py ile birebir aynı olduğu için baseline model aynı sonuçları üretti. Diğer sonuçlar iste TESTING.md 
içerisinde paylaşıldı.

Bulgular:

- **tessdata_best hipotezi çürüdü.** "Fast Türkçe'de kötü" iddiası bu korpusta
  tutmuyor: TR başa baş (%10.63 vs %10.70), EN'de best biraz daha kötü, üstelik
  2 kat yavaş (17.9s → 35.5s). 
- **psm 3 şart.** psm 4 ve 6 çok sütunlu arXiv'i parçalıyor (hata değeri %39-47'ye
  fırlıyor). psm 6'nın tek sütunlu ekran görüntüsündeki kazancı (%0.7 vs %3.3)
  bunu karşılamıyor.
- **DPI 200'ün "kazanması" (*) referans gürültüsü.** Ortalama en düşük worst-case
hata metriği DPI için 200'ü gösteriyor ama belge bazında bakınca kaynak tek: 
tuik_cpi_en'de düşük DPI grafik etiketlerinin bir kısmını hiç okuyamıyor; 
referans metin katmanında zaten olmayan metin azalınca CER "iyileşiyor". 
Sayfayı daha az okumak daha iyi OCR değil. Temiz referanslı belgelerde 300 DPI net önde 
(tobb_taranmis %0.13 vs %0.33/%0.36; arXiv %4.91 vs %5.13).
- **Süreç dersi:** Ortalama metrik (dil bazında worst-case ortalama) gürültülü metin katmanı
içeren belgeler tarafından yanıltılabiliyor. Yöntemi seçmeden önce belge bazında kırılıma bakmak şart.

### Kararlar: 

- Metin katmanı düzgün olan belgelerde metin katmanı kullanılacak, OCR gereken belgelerde fotoğraf için ocr_preprocessed, 
diğerleri için ocr_raw kullanılacak.

- ocr_raw için mevcut config (fast + psm 3 + 300 DPI) donduruldu. Bu değerler artık tahmine dayalı değil ölçüme dayalı
olarak seçildi.

- Bugün kullanılan kodlar, sistemin nihai kodunda kaynak olarak kullanılacak.


### Açık sorular

- Ön işleme fotoğraf olan belgelerde kullanılacak ama fotoğrafı ingest aşamasında
  nasıl tespit edilecek? Aday: her görüntüyü iki geçişle (ham + ön işlemeli) OCR'layıp
  `ocr_confidence` yüksek olanı seçmek ama maliyeti ölçülecek.

- Tablolar nasıl parse edilecek? Tesseract yapıyı düzleştiriyor, CER metriği bunu tam ölçmüyor. 
Retrieval aşamasında tablo sorularıyla ayrıca test edilecek.

---

## Gün 3 — 25.07.2026

Bugün, önceki gün netleşen parsing/OCR katmanının üstüne metin indeksleme ve basit arama
sistemini kurdum. Bu sistemin, ve bundan sonraki adımların objektif şekilde değerlendirilebilmesi
için 'Golden set' adını verdiğimiz test veri setini oluşturdum. Bu veri seti, örnek belgeler içerisinden
sorulabilecek 30 farklı soruyu, bu soruların cevabının hangi belgede, hangi sayfada ve tam olarak
nasıl geçtiğini içeriyor. Sorular zorluk kategorisine göre bölündü. Sonrasında, aşağıdaki akışı takip
ederek sistemi test ettim:

belge → sayfa bazında yönlendirme → chunking → embedding → indeks → dense retrieval -> evaluation

Arama sisteminin kapsamını bugün için bilerek dar tuttum: **bugün yalnızca dense retrieval var**, 
hibrit (BM25+RRF) ve cross-encoder rerank yok. Bunun sebebi, basit sistemde alınan performansı
gördükten sonra, eklenecek modüllerin katkısını görerek eklemek olacak.

Sistemin nihai koduna ulaşmak adına dünkü kodları da taşıdım: sistem kodu artık `src/app/` 
altında bir paket (`ocr.py`, `needs_ocr.py`, `config.py`, `ingest.py`, `retrieval.py`)
şeklinde geliştirilecek, ölçüm yaptığımız kodlar kökte kalmaya devam ediyor ve diğer kodları 
`from app.x import` ile çağırıyor.

### Fotoğraf Tespiti

İlk olarak, Gün 2'nin açık maddesi olan fotoğraf tespitini de kapattım. OCR ile işlenecek bir 
belgenin fotoğraf olup olmadığını, sayfadaki aydınlatma yayılımından bulmayı denedim. 

Ön işleme adımında halihazırda tahmin edilen arka planda persentil farkına bakıp, seçilen eşik değerine göre
fotoğraf tespiti yapıyorum. Fotoğraf olan belgelerde bulunan gölge sebebiyle fark yüksek çıkıyor, 
tarama ya da ekran görüntüsünde fark sıfıra yakın. 

Burada kullanılacak eşik değerini elimizdeki taranmış  belgeler ve fotoğraf üzerinde test ettim. 
Korpusta tek gerçek fotoğraf olduğu için eşiği seçerken temiz bir taranmış sayfaya sentetik aydınlatma 
uygulayıp aynı ölçümü tekrarladım. Seçilen eşik değeri ile yapılan test sonuçları TESTING.md'ye eklendi.

### Golden set ve varyant kararı

Golden set manuel şekilde hazırlandı ama `check_golden.py` kodu ile doğruladım. Alıntı gerçekten o 
belgenin o sayfasında mı ve ingest kodunun ürettiği metinde bulunabiliyor mu kontrollerini yaptım.

Asıl karar korpusla ilgili çıktı. `tobb_taranmis.pdf`, `karma.pdf` ve `tobb_yonetmelik.pdf`
dosyaları birebir **aynı içeriğe sahip**; üçü de bu belgeler ile alakalı sorulan soruların 
cevaplarının 6/6'sını içeriyorlar. Üçü birden indekste olursa aynı cevap üç kopya dönüyor ve 
precision ölçülemez hale geliyor.

- **Varyant grubu.** Bu üç dosya tek bir grup; indekste **aynı anda yalnız biri** bulunuyor.
  Ana retrieval testi hep varsayılan (dijital) versiyon ile koşuyor. Taranmış ve karma sürümler
  için ise aynı 30 soru ile ayrı bir test yaparak, OCR'ın retrieval'a herhangi bir etkisi olup
  olmadığına baktım. 
- **`yonetmelik_ss.png` ve `yonetmelik_foto.jpeg` korpus dışı.** İkisi de yönetmeliğin
  yalnız 1. sayfası ve OCR testi için vardı.


### Chunking

- Chunk'ları sayfa sayfa değil belge genelinde bölüyorum, böylece sayfa sınırında başlayan
cümle bölünmüyor (Gün 1'de not ettiğim sorun). Sayfa numarası offset'ten geri hesaplanıyor.
Chunk metadata'sında dosya, sayfa aralığı ve hangi yöntemle çıkarıldığı var.

- Burada cümle sınırı granülaritesi tek başına yetmiyor. "Tanımlar MADDE 3- (1) ... a) ... b) ..." 
gibi maddelerde cümle sonu hiç yok ve tek bir chunk 3302 karaktere kadar çıkıyordu. O sırada
kullandığım e5-small'ın context window değeri 512 token, fazlası vektöre hiç girmiyordu.

- `CHUNK_MAX=1600` parametresi ekleyip uzun cümleleri kelime sınırında böldüm. Sonra tokenizer 
ile doğruladım: 311 chunk'ın en uzunu 411 token, 512'yi aşan yok.

- **Ama bu gerekçe, model seçimiyle birlikte düştü.** Aynı gün donan bge-m3'ün penceresi 8192
token, yani o 3302 karakterlik chunk bu modelde hiç kırpılmazdı. Parametreyi yine de
kaldırmadım: chunk boyutu yalnız pencereyi değil retrieval hassasiyetini de etkiliyor, 1600
hâlâ makul bir tavan olabilir. Ancak bu haliyle ölçülmüş bir değer değil, hedef/örtüşme
(800/150) parametreleri ile birlikte chunk taramasına giriyor.


### Embedding modeli seçimi

CPU'da hızlı çalışabilecek modelleri aynı korpus, aynı golden set ve aynı chunking ile karşılaştırdım.
Test sonuçları TESTING.md'ye eklendi.

*(Gün 6 düzeltmesi: buradaki süre ölçümleri aslında CPU'da değilmiş. sentence-transformers
Apple Silicon'da cihazı otomatik seçiyor ve embedding MPS'te koştu. recall/MRR/AUC
cihazdan bağımsız, model seçimi değişmiyor; ama ms/sorgu ve s/chunk değerleri GPU'lu sayılar.)*

Bulgular:

- **bge-m3 seçildi.** İki dilin worst-case'inde de önde. Bedeli sorgu başına 35 ms 
ve chunk başına ~0.46 s indeksleme.

- Çapraz dil sütunu tek başına yanıltıcı (n=3). Doğru chunk'ın sırası: e5-small
  89/172/181, e5-base 3/106/8, MiniLM 3/6/3, bge-m3 3/1/3.

- İlk ölçümde q061 (çapraz dil, çapa `tuik_cpi_en.pdf`) kaçırma görünüyordu. Alınan chunk'a baktığımda
sistem aslında **doğru cevabı veriyor**, ama cevabı Türkçe bültenden getiriyor. 2. sıradaki chunk cevabı içeriyor. 
Burada kaçırma olarak görünmesinin sebebi, golden set üzerinde istenen kaynak belgenin İngilizce olması. 

- Bu sebeple golden set'e opsiyonel `izole` alanı ekledim: soru test edilirken izole alanında adı geçen belgeler
indeksten düşürülüyor. Şu an yalnızca q061'de kullanılıyor ve bu sadece **test için** olacak. Çalışan sistemde 
belge gizleyen bir filtre olmayacak. 

- İzole alanın etkisi dört modelde de yeniden ölçüldü: recall@k bge-m3 %85.0 → %90.0, e5-base %65.0 → %70.0;
MiniLM ve e5-small değişmedi (biri zaten buluyordu, diğerinin doğru chunk'ı çok uzaktaydı).
TESTING.md son sonuçları içeriyor.

- **AUC** = bir `yanitla` satırının bir `reddet` satırını geçme olasılığı (ölçekten bağımsız). 
Ham skor farkı modeller arası kıyaslanamıyor: e5 skorları dar bir banda sıkışıyor.



### Dense retrieval

| Dil | recall@1 | recall@3 | recall@5 | recall@10 |
|---|---|---|---|---|
| TR (16) | %56.2 | %78.1 | %93.8 | %93.8 |
| EN (4) | %25.0 | %75.0 | %75.0 | %100 |

- Kategori bazında (k=5): çapraz dil %100, çok belge %100, tablo %100, tek belge %88.9,
çelişki %50. 

- k=5'te kaçan sadece üç çapa var: q009 (arXiv, İngilizce) ve q041'in iki çapası birden.
q041 çelişki sorusu, yani iki farklı belgeden iki değeri birden getirmesi gerekiyor.

- **Tablo sorularının %100 çıkması dikkat çekici.** Gün 2'de "OCR tabloyu düzleştiriyor, CER
bunu ölçmüyor, retrieval'da ayrıca test edilecek" diye not düşmüştüm. En azından doğru
chunk'ı bulma seviyesinde tablo sorunu görünmüyor. 

### Reddetme eşiği: kosinüs skoru yetmiyor

Golden set içerisinde yanıtlanması ve reddedilmesi gereken sorularda çıkan kosinüs skorlarının
dağılımını aşağıda paylaştım:

| soru tipi | min skor | ortalama skor | medyan skor | max skor |
|---|---|---|---|---|
| yanitla (20) | 0.490 | 0.671 | 0.680 | 0.782 |
| reddet (10) | 0.343 | 0.568 | 0.572 | 0.667 |

Dağılımlar ayrışıyor ama örtüşüyor: 10 reddet satırının 8'i yanıtlanabilirlerin skor
aralığının içinde. Gün 1'de tek örnekle "cosine eşik için zayıf" demiştim (0.84 vs 0.77); 
şimdi 30 soruda ölçüldü ve en iyi modelde bile tek bir kosinüs eşiği yanıtlanabilir/yanıtlanamaz 
ayrımını yapamıyor. Cross-encoder kararı gerekçelendi.

### OCR'ın retrieval'a maliyeti

Aynı 30 soruyu, aynı belgenin üç farklı formatıyla denedim (dijital, taranmış ve karma). Sonuçları
TESTING.md içerisinde de paylaştım.

Bulgular:

- Sadece nihai recall skoru aynı değil, **soru bazında da tek bir değişiklik yok** 
(Sonucun aynı kalıp altında bir sorunun bozulup başkasının düzelmesi ihtimalini elemek için
soru bazında da kontrol ettim). 

- Nedenini de ölçtüm: karşılık gelen chunk'lar arasında ortalama CER taranmışta %0.14, 
karmada %0.01; vektörler 0.999+ kosinüsle örtüşüyor. Bu kadar küçük bir
kayma 311 chunk'lık sıralamayı hiç değiştirmiyor.

**Sonucun sınırı:** bu "OCR retrieval'ı etkilemiyor" demek değil, "%0.13 CER'lik temiz tarama
etkilemiyor" demek. Varyant grubunda kötü tarama yok.

### Kararlar

- Golden set 30 soruda bırakıldı; dağılım korpusla uyumlu olsun diye TR ağırlıklı kaldı
  (TR 23 / EN 7).
- Embedding modeli **BAAI/bge-m3** olarak seçildi.
- Ön işleme için fotoğraf tespiti eşiği: **aydınlatma yayılımı ≥ 25.0**.
- Aynı yönetmeliğin üç formatından (varyant grubu) indekste yalnız biri olacak. O da dijital
versiyonu olacak.
- `CHUNK_MAX=1600` kaldı ama **gerekçesi dondurulmadı**: e5-small'ın 512 token'lık penceresi
için konmuştu, bge-m3'ün penceresi 8192. Chunk taramasında hedef/örtüşme ile birlikte ele alınacak.


### Açık sorular

- Hibrit retrieval (BM25 + RRF) ve cross-encoder rerank ne kadar katkı verecek? Ölçüm
  düzeneği hazır, tek yapılacak aynı testi güncellenen retrieval üzerinde koşmak.
- Reddetme eşiği neyin üstüne kurulacak? Kosinüs yetmiyor; reranker skorunun
  ayrımı ne kadar iyileştirdiği ölçülecek.
- Chunk boyutu/örtüşmesi taranacak. Çapalar chunk'tan bağımsız olduğu için golden set'i
  yeniden etiketlemeye gerek yok.
- Retrieval hangi OCR kalitesinde kırılıyor? Temiz taramada kırılmıyor; sentetik gölgeli
  bozuk bir varyant üretip kırılma noktası ölçülebilir.

---

## Gün 4 — 26.07.2026

Dün dense retrieval kısmı bitirildikten sonra, retrieval modülünü bitirmek için hibrit 
retrieval (BM25 + Reciprocal Rank Fusion (RRF)) ve reranking modüllerini hazırladım. Önce hibrit retrieval eklendi,
golden set üzerinde dense model ile performansı karşılaştırıldı. Sonrasında reranker, hem
sadece dense, hem de hibrit retrieval üzerine eklenerek retrieval aşaması için nihai karar
alındı. Hibrit retrieval için BM25 modeli farklı parametrelerle test edildi. Reranker için ise
2 farklı model ve farklı derinlik değerleri (top-k) için denendi.

### BM25

Öncelikle, sisteme BM25'in nasıl entegre edilebileceğini netleştirdim, çünkü kullanacağım tokenizer
kararı buna bağlıydı. Şu an eklediğimiz belgeler ve golden set, belgenin dilinin bilindiğini varsayıyor.

Fakat sistem gerçek senaryoda gelen belgenin dilini bilmeyeceği için, gelen her belgede Türkçe bazlı
köke inme yapamam. Bu sebeple, BM25'i dense retrieval'ın zayıf olduğu yerde değil, yapısal olarak yapamadığı 
yerde kullanıyorum: özel isim, madde numarası, not eşiği gibi tam eşleşme gereken şeylerde. Anlamca
yakın olmak orada işe yaramıyor. Bunun sonucu olarak, BM25'e stemming ya da ek kesme koymadım. 


### RRF

Hibrit retrieval esnasında, iki retrieval yönteminden gelen skorları toplamak yerine 
chunk sıralamalarını RRF ile birleştirdim. Bu kararın gerekçesi, iki yöntemin skor dağılımlarının
çok farklı olması. `capraz_dil` sorularında (TR soru → EN belge) BM25'in terim
örtüşmesi **tanım gereği sıfır**. Dense o üç soruda %100. Ağırlıklı skor
toplamı o sıfırları hesaba katıp dense'in halihazırda tuttuğu satırları seyreltirdi;
RRF'te ise bir chunk'ı sıralamayan retriever ona hiç katkı vermiyor.

Varsayılan parametrelerle yapılan ilk testte hibrit retrieval dense'ten kötü performans gösterdi: 
`depth=50`, `k=60`, eşit ağırlıkla recall@5 %82.5 (dense %90.0), TR %78.1 ve `capraz_dil` %100'den %66.7'ye düştü.

Teşhis: BM25 en emin olduğu birkaç chunk sonrasında gürültü üretmeye başlıyor, ama o gürültü
RRF'te dense'in isabetleriyle **aynı ağırlıkta oy kullanıyor**. Eşit ağırlıklı füzyon
varsayılan değil bir varsayım, ve BM25 tek başına %62.5 iken yanlış bir varsayım.

18 farklı parametre seti denedim (derinlik × RRF sabiti × ağırlık). İki eksen belirleyici çıktı:
derinliği 20'ye indirmek ve dense'i 2:1 ağırlıklandırmak.

### Hibrit retrieval

Optimum parametreleri bulduktan sonra, retriever testimi gerçekleştirdim. Sonuçlar aşağıdaki 
tabloda ve TESTING.md'de görülebilir.

| retriever | recall@5 | TR | EN | MRR | çapraz dil |
|---|---|---|---|---|---|
| dense | %90.0 | **%93.8** | %75.0 | 0.68 | %100 |
| bm25 tek başına | %62.5 | %59.4 | %75.0 | 0.47 | **%0** |
| hibrit (varsayılan ayar) | %82.5 | %78.1 | %100 | 0.67 | %66.7 |
| **hibrit d20/k10/w2:1** | **%92.5** | %90.6 | **%100** | **0.72** | %100 |

Hibrit retrieval kullanmanın toplam recall'a kazandırdığı %2.5 puan. Ama soru bazında 
baktığımda bunun tam olarak ne olduğu görünüyor: 

q009 (EN, arXiv) %0 → %100 kazanıldı,  q011 (TR, çok belge) %100 → %50 kaybedildi. 
Kazanılan soru tam da BM25'nin kullanılma sebebi, cevap "Munkres and
Kalman filter" gibi özel isimler içeriyor. 

18 parametre setini golden set için tarayıp en iyisini seçtim, yani seçim 20 örneklem üzerinde
seçildi, overfit riski taşıyor. Bunu risk olarak yazıyorum.

Rerank testi öncesinde, hibrit retrieval kullanmaya karar vermiştim. Sebepleri şu şekilde:

- gecikme maliyeti sıfıra yakın, hibrit 33.9 ms, dense 34.4 ms çünkü ikinci bir embedding
yok, aynı sorgu vektörü BM25 ile birleşiyor.
- model seçimini "TR ve iki dilin worst-case'ine göre" yapıyorduk. 
worst-case dense'te %75.0, hibritte %90.6. TR'ye tek başına
bakarsan dense önde (%93.8 > %90.6), yani bu bir kazanç değil takas.
- MRR 0.68 iken 0.72 oldu. MRR her soruda sürekli bir değer, recall gibi
5 puanlık basamaklarla zıplamıyor, dolayısıyla tek soruya daha az duyarlı.

Rerank testi sonunda bu kararım değişti, sebeplerini o kısımda anlatacağım.


## Cross-encoder rerank

Hibriti kullanmaya karar verdikten sonra aynı gün reranker'ı kurup hibrit ve sadece dense
modele karşı test ettim. Sıra şu şekildeydi: **önce gecikme, sonra recall, sonra reddetme sinyali.**
Cross-encoder dense'ten yapısal olarak pahalı çünkü dense sorguyu bir kez gömüyor,
cross-encoder ise her adayı ayrı bir forward pass'ten geçiriyor ve gecikmesi kabul
edilemezse recall katkısına bakmanın anlamı yok. Testler halihazırda uzun sürdüğü için
iki farklı model ile ilerledim:

- "BAAI/bge-reranker-v2-m3" (568M)
- "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" (118M) 

### Gecikme

Gecikme değerleri golden set'i dense retrieval'dan gelen adaylarla uçtan uca ölçünce:
20 chunk derinliğinde mmarco 6.4 s/soru ve bge 6.9 s/soru ortalama gecikmeye sahip, 
çünkü gerçek chunk'lar ~512 token ve reranker maliyeti bu boyda çok artıyor. 
mmarco'nun avantajı ancak **kısa chunk'ta** açılıyor (maliyeti dizi uzunluğuyla büyüyor, 
bge tavanda sabit). Derinlik 10 olduğunda da iki model de soru başına süre ~3 saniye gecikmeye
sahip. Yani **gecikme, modeli seçmiyor.**

 
### Recall

Reranker recall@5 değerini %92.5'ten %97.5'e çıkardı (TR 96.9, EN 100, MRR 0.85). 
Soru bazında da hibritin kaybettiği q011 geri geldi, dense-base'de q041 (çelişki) de
düzeldi ve **kayıp yok.** Hibritin "bir soru kazanç / bir soru kayıp" takası kapandı.

Asıl sürpriz: reranker'a hibrit yerine **dense** adayları vermek daha iyi (97.5 vs 95.0).
Reranker dense retrieval ve 20 derinlik üzerinde, hibriti eklerken gerekçe gösterdiğim q009'u 
(özel isimler, BM25'in işi) BM25 olmadan topluyor, çünkü q009'un cevabı dense'in top-20'sinde mevcut, 
sadece top-5'te değildi; reranker onu yukarı çekiyor. 

Gecikme değeri derinlik 20 olduğunda yüksek olduğu için (yaklaşık 6.5 saniye), 
derinlik 10 olduğunda recall tekrar test edildi. Dense + reranker için recall@5 değeri 
%95 çıktı (derinlik 20 iken 97.5). Bu düşüş, gecikmede sağladığı faydaya göre tolere edilebilir bir düşüş.


### Reddetme sinyali

- Reranker skorunun asıl merak edilen tarafı reddetme eşiğiydi çünkü kosinüs yetmiyordu
(AUC 0.790). Reranker'ın skor dağılımı görsel olarak çok daha ayrık (bge yanitla medyan 0.99 vs
reddet 0.19). Ama **AUC değerine** baktığımızda reranker'ın kosinüsü kesin yendiği söylenemez. (0.84)

- Neden bu kadar ayrık dağılım tek eşikle çözülmüyor sorusu soru bazlı bakıldığında ortaya çıkıyor
Bir 'reddet' sorusu 0.95+ skorluyor (bge 0.952, mmarco 0.999), bir yanitla sorusu ~0 skor alıyor. 
Eşik 0.5'te olursa 3 uydurma + 3 gereksiz reddetme anlamına geliyor.

- Uydurmayı sıfırlamak (çalışma amacı) eşiği o aykırının üstüne çekmeyi, 
o da yanıtlanabilirlerin 7/20'sini (bge) ya da 14/20'sini (mmarco) feda etmeyi
gerektiriyor. Tek skaler eşik bu bedeli ödemeden çalışmıyor, önemli nokta bge burada mmarco'dan
çok daha iyi.

### Kararlar

- Retrieval mimarisi: **dense → reranker, hibrit/BM25 yok.** Hibrit bir ara basamak testi olarak kaldı, 
nihai sisteme dahil değil.

- Rerank modeli: **bge.** Kritik nokta: golden set 30 soruda iki model de recall için %97.5 tavanına vuruyor,
yani test seti bu iki modeli **ayırt edemiyor** Görülmemiş belgelerde (asıl kullanım) bge daha güvenilir.
Üstelik reddetme değerleri de ölçülü olarak bge lehine çıktı (AUC 0.840 vs 0.800).

- **Derinlik 10 seçildi** (%95.0, ~3 s/soru): Derinlik 20'nin %97.5'i +2.5 puan ama tek yanlış cevap
  (q041 çelişki), 2× gecikmeye değmiyor. Derinlik 10'da recall, dense ve hibrit için eşit, "BM25 gereksiz"
  kararı korunuyor. Reddetme eşiği derinlikten bağımsız.

### Açık sorular


- **Reddetme hâlâ çözülmedi.** Tek eşik yetmiyor. Aday yönler: (1) asimetrik eşik +
  ikinci sinyal (top1−top2 marjı, ya da kosinüsle 2-özellikli kapı); (2) asıl backstop
  **generation'da LLM'in grounding/citation kontrolü** (Gün 5). O reddet aykırısını da
  incelemek gerek (konu korpusta var ama cevap yok mu?).
- Model freeze mmarco'da mı: golden_total sonrası.

---

## Gün 5 — 27.07.2026

Retrieval katmanı son haliyle netleştikten sonra, bugün sistemi uçtan uca tamamladım.
Son kalan kısımlar LLM kullanarak cevap üretimi ve soru reddetme kısmıydı. Case çalışmasının
amacı, sorulan soruya kaynak göstererek sorunun dilinde ve **belgeye dayalı** (uydurmayan) 
bir cevap üreten, sorunun cevabı belgede yoksa da soruyu reddeden bir sistem yapmaktı.
Cevap üretme kısmı için Ollama destekli 4 farklı model denendi, 2 model üzerinde karar kılındı.
Kullanıcı hızlı ya da daha güvenilir model arasında seçim yapabilecek. Buna ek olarak, seçilen
modellerin reasoning kapabilitesi olduğu için, 'think' parametresinin sonuca etkisi izlendi.
Modellerin performansının ölçümü için `eval_generation.py` ve (opsiyonel) `judge_generation.py` eklendi.
Bugünün kararları öncekiler gibi ölçümle alındı. Cevap üretimi sonrasında sistem Docker'a taşındı.

### Cevap üretme kurulumu

Cevap üretme kodu arayüzden bağımsız şekilde `pipeline.answer(...)` fonksiyonuna bağlı.
CLI, arayüz ve evaluation kodlarının hepsi bu fonksiyonu çağıracak, böylece retrieval 
,reddetme, cevap üretim yolu tek yerde tanımlı. LLM'e verdiğim prompt kuralları: 
- (1) yalnız verilen pasajlardan cevapla, 
- (2) her iddiayı **pasaj numarasıyla** kaynak göster, 
- (3) cevap pasajlarda yoksa `[YANITLANAMADI]` dön, 
- (4) kaynaklar çelişiyorsa ikisini de kaynağıyla sun, 
- (5) soruyu sorunun dilinde cevapla.

Belge kaynak gösterme kısmını pasaj numarasına (`[1]`, `[3]`) çevirdim; gerçek dosya ve sayfayı ben hit
listesinden eşliyorum. Modelin gösterdiği numara yalnız ona verilen bir pasajı işaret edebilir.

Çoklu-tur konuşmayı bu çalışmada yapmadım çünkü hazırladığım golden set tek soru-cevap destekliyor. 

### LLM seçimi

Model seçimimi ilk günden aldığım On-prem/Ollama kararına göre yaptım. Seçilecek modelin 
CPU'da çalışması lazım, model boyutu pratikte ~12B ile sınırlı. Türkçe cevap üretimi çok kritik. 
Bu sebeplerle model adayları olarak qwen3.5:4b/9b ve  gemma4:e4b/12b modellerini seçtim.

Kullanacağım modelleri embedding/reranker aşamalarındaki gibi **ölçerek** seçtim; 
`eval_generation.py` golden set üzerinde reddetme matrisini ve cevap doğruluğunu TR/EN ayrı ölçüyor. 
 Sonuçları TESTING.md içerisinde paylaştım.

Bulgular:

- **gemma4:12b elendi:** soru başına 50-160 s (medyan ~65 s), interaktif kullanım
  imkânsız. Ama offline yargıç olarak geri döndü (aşağıda). *(Gün 6 düzeltmesi: bunu
  "CPU'da" diye kaydetmiştim; local Ollama aslında M2'nin GPU'sunu kullanıyormuş.
  GPU'da bile bu sürelerdeyse GPU'suz CPU'da daha yavaş. Eleme kararı daha da güçlenerek
  geçerli.)*
- **qwen3.5:9b elendi:** cevap doğruluğunda tepedekilerle eşit (15/20) ama **uydurma 3/10**
  (q022, q033, q034). Projenin ilk önceliği sıfıra yakın uydurma; aynı boyuttaki gemma4:e4b
  1/10 yapıyor.
- **Kalan iki model (qwen3.5:4b - gemma4:e4b) neredeyse eşit** (judge ile ikisi de 16/20). 
Bu yüzden birini seçmek yerine ikisini de tutuyorum: iki mod, kullanıcı seçsin. 

- `turbo` = qwen3.5:4b (medyan 14 s), `large` = gemma4:e4b (medyan 37 s). 
Varsayılan `large`, çünkü faithfulness öncelik; UI'da hız için `turbo`'ya geçilebilir.

### Reddetme kapısı: neden iki aşamalı

3. ve 4. günde tek skaler eşiğin (önce kosinüs, sonra reranker skoru) yetmediğini ölçmüştüm.
Bugün reranker top-1 skorlarını soru soru çıkarınca neden yetmediği somutlaştı: reranker `capraz_dil`
kategorisinde yanıtlanması gereken soruları 0.003-0.012 aralığında skorluyor 
(reranker onları **sıralıyor** ama mutlak skor neredeyse sıfır), 
buna karşılık `yanlis_oncul` kategorisinde reddedilmesi gereken bir soru (q033) **0.9519** skoru aldı.
Tek bir eşik kullanımı ya uydurmaya izin verir ya 7/20 yanıtlanabilir soruyu feda ederdi.

Buna ek olarak top-5 chunk'ın ortalama skorunu denedim (top-1 gürültülüyse ortalama daha kararlı olur mu diye).
Tersine çıktı, ortalama **iki sinyali de** bozdu (tablo TESTING'de). Bunun sebebi, cevabı sadece tek bir chunk'ta
olan ve yanıtlanması gereken sorular, ortalamada reddet bandına düşüyor, sorularının konusu belgede geçen ama 
reddedilmesi gereken sorular yüksek kalıyor. reranker top-1 en iyi sinyal, ama tek başına yetmiyor.

Kararım iki aşamalı kapı, uydurma-öncelikli:

- **Aşama 1:** reranker top-1 skoru < `0.003` ise, LLM'i çağırmadan reddet. Eşiği
  **sıfır-yanlış-red** noktasına koydum (en zayıf yanıtlanabilir soru 0.0032). Bu şekilde hiçbir
  yanıtlanabilir soru düşürmez ve yalnız bariz alan-dışı sorular (q071/q072) elenir.
  
- **Aşama 2:** LLM grounding, pasajlarda cevap yoksa `[YANITLANAMADI]`.
  Skor bandının içinde kalan 8 reddet sorusunu (q033 aykırısı dahil) bu yakalıyor.

Sonuç: seçilen iki modelinde de **uydurma 1/10**. Tek kaçan **q022** (arXiv makalesi, "video stream
accuracy", reddedilmesi gereken soru), üç modelde de uyduruldu. Bu soru LLM grounding'in de sınırı oldu,
yalnız eşiğin değil; düzeltilebilir hata olarak not ettim.

### Doğruluk ölçümü ve LLM-as-judge

Yanıtlanabilir soruların cevaplarının doğruluğunu golden set içinde bulunan referans alıntılarla 
karşılaştırıp başarıyı otomatik ölçmek istedim. İlk yapılan testlerde YANLIS_CEVAP etiketli
bazı cevapların aslında doğru olduğunu görünce üç ayrı kırılma çıktı:

- **çapraz dil:** referans İngilizce ("increased by 45.14% for housing"), cevap Türkçe olduğunda eşleşme
  tanım gereği sıfır.
- **paraphrasing:** modelin cevabı "dokuz dönemdir" ama referans "en çok dokuz dönemde" olursa yanlış olarak
işaretleniyor.
- **sayı formatı:** `45,14` vs `45.14` yine yanlış olarak işaretlendi.

Metrik **her modeli** düşük gösteriyordu. Normalizasyon (NFKC) ve birkaç metin düzeltmesi ile bariz hataları
düzelttim. gemma4:e4b modelinin doğru cevap sayısı 12'den → 15'e, qwen3.5:4b ise 5'ten → 14'e sıçradı.

Deterministik eşleştiricinin bariz hataları düzelse bile bir **alt sınır mevcut**: paraphrasing
ölçülemiyor. Bu sebebple opsiyonel bir LLM yargıç ekledim (`judge_generation.py`). Tasarım ilkeleri:

- **Offline**, LLM yargıcı kaydedilmiş cevaplar üzerinde çalışır, bu sayede model yeniden çalışmaz.
- **Sonuç JSON'ı ASLA değiştirmez**, düzeltmeyi kodda/bellekte yapar, yargıyı ayrı dosya olan
 `<model>.judge.json`'a yazar.
- Yargıç, değerlendirilen modelden **farklı ve güçlü** olmalı (bir model kendini yargılarsa
  kayırır). Cevap üretimi testinde gecikmeden elenen **gemma4:12b**, offline olduğu için burada ideal
  yargıç, çünkü gecikme önemli değil.
- **İstenen bilgi cevapta varsa, fazladan bilgi cevabı yanlış yapmaz**, uydurma ya da
  çelişki olmadıkça.

Yargıç yanlış olarak işaretlenen doğru cevapları çevirdiği gibi, birkaç vakada da doğru işaretlenen
cevabı yanlış olarak çevirdi. Bu vakaları da inceledim:

- **Deterministik false-positive'i doğru yakalama** (yargıç haklı): q051 (1,37 sayısı var ama
  yanlış tarihe bağlanmış). Deterministik eşleştiricinin zayıflığı: sayı doğru, bağlam yanlış.
- **Fazla katılık** Cevapta aranan bilgi vardı, fakat fazladan **doğru** bilgi cezalandırılıyordu.
  Bu kısmı yukarıdaki kurala gevşetince düzeldi.

- Buna ek olarak bir soruda (q052) qwen3.5:4b modeli 22 bin karakterlik bir **tekrar döngüsüne** girmiş;
yargıç bunu "aynı paragrafın çok sayıda tekrarı" diye yorumlamış ve sonuç vermemiş.  Parser'ı, yargıçtan
sonuç yoksa sessizce YANLIS saymak yerine **"parse edilemedi"** olarak işaretleyecek
şekilde düzelttim.

Bu son bulgu üzerine üretim modellerine de bir çıktı sınırı (`num_predict = 512`) koydum:
grounded bir cevap birkaç yüz token, 512 meşru cevabı kesmiyor ama degenerasyon döngüsünü ve
worst-case gecikmeyi (q052'de 266 s) engelliyor.

### Depth=20 denemesi

4.Gün'de reddettiğim chunk depth=20'yi cevap üretme seviyesinde tekrar denedim (q041 sorusunu
kurtarır mı diye). Kurtarmadı: q041 iki modelde de hâlâ yanlış, çünkü aday havuzu 20 olsa da,
doğru cevabı içeren chunk top-5'e girmiyor. Toplam skor da artmadı (gemma4:e4b 15 → 14'e düştü bile). 
Depth 10 kararı cevap üretiminde de doğrulandı.

## Docker ve thinking politikası

Cevap üretme kısmını bitirince sistemi kolay ayağa kaldırma ve sonuçların tekrarlanabilirliği adına
sistemi Docker'a taşıdım. Docker imaj hazırlanıp, sistem çtan uca denenirken large modelde
beklemediğim bir bug çıktı ve bu blok ikiye ayrıldı: Docker imajı doğrulamak, ve doğrularken
bulduğum gizli bir **thinking** sorununu çözmek. İkincisi generation'ın 5. gün kararlarından
birini (`num_predict=512`) revize ettirdi, o yüzden buraya yazıyorum.

### Docker imajı

- **turbo (qwen3.5:4b) uçtan uca çalışıyor**. Retrieval + reranker + HF cache + ollama +
  doğru citation eşleme.
- **large (gemma4:e4b) modeli ~8 GB Docker VM'inde OOM hatası aldı.** ollama yüklerken tensörler 8.5 GB
  (`load_tensors` 5903 + 2827 MiB), 7.75 GB'lik VM'e sığmıyor. Docker Desktop belleğini 12 GB'a 
  çıkarınca çalıştı. **Dağıtım notu:** large modeli VM ≥12 GB RAM ister, turbo 8 GB'ta çalışır.

### Gemma modelinden boş cevap: gizli thinking × çıktı sınırı

Docker'da gemma modeli ile "Haziran 2026 için enflasyon oranı nedir?" sorusunun cevabı boş döndü. Fakat bu
bir hata değil, sadece boş satır. Sebebi ham yanıtın metadata'sından çıkardım: 
gemma4:e4b bir **thinking modeli**; ürettiği token'lar `message.thinking`'e gidiyor, 
`content` boş, `done=length`. Yani `num_predict=512` tavanı sadece reasoning'e
gidiyor ve **cevap başlamadan** kesiliyor.

En kritik soru: bu problemi **golden set neden yakalamadı?** 30 sorunun hiçbirinde boş cevap yok.
Fark sorunun kesinliğinde: golden sorular çok belirli sorular soruyor ("bir önceki aya göre yüzde kaç")
,gemma az düşünüyor, bu sebeple thinking+cevap 512'ye sığıyor. Benim test için yazdığım soru hem 
**belirsiz** (hangi oran: aylık/yıllık/12-ay/özel-kapsam?) hem de **belgede olmayan bir terim** 
("enflasyon"; belgeler "TÜFE" diyor) içeriyor. gemma uzun uzun düşünüyor, tavanı aşıyor. 

Çıkarılan ders: golden set gerçek kullanıcının belirsiz sorularını yeterince temsil etmiyormuş, 
bug'ı maskeleyen buydu. 

Gemma thinking açık şekildeyken worst case ~1106 token kullandı (açık uçlu İngilizce soru). Marjin
olarak yüzde 40'lık bir pay bırakarak max token limitini (num_predict) 1536 token seçtim.
Qwen ise thinking kapalı olduğu için 512 sadece cevap için yeterli. Ayrıca artık boş cevap geldiğinde
`grounded=False` (gerekçeli red) oluyor.

### Thinking on/off: Model bazında, ölçümle

'think' parametresinin seçilen modellere etkisini golden set'te ölçtüm (tablolar TESTING'de).

- **gemma think=False:** cevaplanabilir sorularda **eşit performans** (15/20) ve **2.7× daha hızlı** 
(medyan 13.6 vs 36.6 s) ama **uydurma 1/10 → 4/10.** Cevap öncesi düşünme kapanınca 
yanlış-öncül/cevaplanamaz sorular (q024/q033/q034) cevaplanmaya başlıyor. Uydurma olmaması birinci
öncelik → **gemma'da think açık kalıyor** (çıktı sınırı 1536 ile).
- **qwen think=True:** qwen zaten `think=False` idi. Açınca reasoning'i **felaket uzun**,
  tek bir soru için 4600-6000+ token; 6000 tavanı bile tek başına thinking'e yetmiyor.
  Gecikme tek başına diskalifiye sebebi, bu sebeple **qwen think=False kalıyor.** 
  Tek bir soru bile çok uzun sürdüğü için tüm seti test etmedim.

Sonuç simetrik ve savunulabilir: **her modelin düşünme ayarı rolüyle gerekçeli**, large model kalite
(think on, ~36 s, uydurma 1/10), turbo model hızlı (think off, ~14 s).

### Kararlar

- **Cevap üretme: iki mod.** `turbo` = qwen3.5:4b, `large` = gemma4:e4b, varsayılan `large`.
  Elenenler: qwen3.5:9b (uydurma 3/10), gemma4:12b (gecikme → offline yargıç).
- **Reddetme: iki aşamalı.** Stage-1 reranker top-1 eşiği `0.003` (sıfır-yanlış-red), stage-2
  LLM grounding. Uydurma 1/10 (residual q022).
- **Doğruluk ölçümü:** Deterministik anahtar-eşleşme (savunulabilir alt sınır) + opsiyonel
  gemma4:12b yargıç.
- **Depth 10 korundu**; RERANK_DEPTH=20 kazanç getirmedi.
- **Docker:** iki servis, model'ler named volume, ilk çalışmada provizyon. large modeli için 
  Docker VM ≥~12 GB RAM (ölçüldü), turbo 8 GB.
- **Model bazlı token limiti `num_predict`:** gemma 1536 (thinking bütçesi + cevap), qwen 512 (cevap +
  degenerasyon sınırı). Boş `content` artık `grounded=False`.
- **Thinking model bazında:** large açık, turbo kapalı.

### Açık sorular

- **q022 residual:** Soru belge ile ilgili fakat yanıtlanamaz, üç modelde de uyduruldu. 
  Prompt'ta daha katı grounding mi, yoksa retrieval'dan "cevap yok" sinyali mi gerek? Zaman kalırsa ölçülecek.
- **q041 çelişki:** retrieval (3,00 chunk'ı top-5 dışında) + generation birlikte. Çelişki
  kategorisi için GEN_K/depth artırılmalı mı? Runtime dil/kategori-kör olduğu için global olur.
- **q051/q052:** tablo hücresi (yanlış hücre seçimi) ve turbo degenerasyonu. İkisi de
  generation tarafında.
- Docker'da large modelin **gerçek CPU inference gecikmesi** ölçülmedi, local ~36 s MPS/GPU'lu;
  Docker CPU daha yavaş olacak, arayüzde bu daha da önemli.
- Chunk boyutu/örtüşmesi hâlâ taranmadı.

---

## Gün 6 — 28.07.2026

Bugün sistemin arayüzünü Streamlit ile tamamladım. Minimum arayüz isterleri birden fazla ve
farklı türlerde belge yükleme ve indeksleme, soru sorma, cevapta kaynak belge + sayfa
görüntüleyebilme, reddedilen sorunun net gösterimi ve turbo/large model seçimiydi.
Arayüz geliştirmesi öncesinde repoyu düzenledim ve tekrarlanabilirlik için kullanılan 
kütüphanelerin sürümlerini sabitledim. Arayüz bittikteon sonra Docker imaj testlerinde
beklemediğim bir gecikme farkı çıktı; bu farkın teşhisi bugünün en öğretici bulgusu oldu.

### Repo düzeni ve sürüm sabitleme

- Çalışma boyunca kullandığım ölçüm kodlarının tamamı `eval/` klasörüne taşındı. Scriptler 
artık çalışma dizininden bağımsız (path'ler `config.ROOT` üzerinden tanımlanıyor). Taşımanın
hiçbir şeyi bozmadığını testlerle doğruladım. Dense ve reranked retrieval eval kodları 
önceki sonuçların birebir aynısını üretti.

- `pyproject.toml`'da listelenen kütüphaneler, kurulu sürümlere `==` ile sabitlendi. 
`torch` ve `transformers` daha önce dolaylı bağımlılıktı; model davranışını doğrudan
belirledikleri için açıkça listeye alındı, Docker imajı da aynı sürümleri kuruyor. 
Gerekçe basit: raporladığım sonuçlar, kurulumu sıfırdan yapan birinde de aynı çıkmalı.

### Arayüz kurulumu

Framework olarak Streamlit seçtim çünkü arayüz isteri bir ürünün frontend'i değil 
çalışan bir demo. Streamlit sayesinde arayüz tek dosyayla çözülüyor ve kod dili aynı kalıyor.
Arayüz (`webui.py`) yalnız `pipeline.answer`'ı çağırıyor, bu sayede retrieval, reddetme ve 
cevap üretiminin tek yerde tanımlı olması kararı arayüzde de korunmuş oldu. 
UI kodu hiçbir arama/üretim mantığı içermiyor.

### Arayüzdeki "alakasız pasajlar"

Arayüzdeki testlerde düşük skorlu (0.01 ve altı) pasajları görünce ilk bakışta bir hata gibi durdu.
Soruyla ilgisiz pasajlar neden modele gidiyor diye düşündüm. Fakat oluşturduğumuz sistem 
her soruda en yüksek skorlu 5 pasajı modele veriyor ve düşük skorlular da recall güvencesi. 
Bunları bir skor eşiğiyle elemek Gün 5 ölçümüyle doğrudan çelişiyor: çapraz-dil sorularında **doğru cevabı
içeren** pasaj 0.003-0.012 skorluyor; buradaki 0.003'ü eleyecek her eşik o cevapları da
elerdi. Düzeltme pasajları filtrelemek değil sunumu netleştirmek oldu: LLM tarafından cevapta gerçekten
kullanılan pasajlar ✅ ile işaretleniyor ve arayüzde tek cümlelik açıklama var. Model zaten
yalnız ilgili pasajı kullanıp kaynak gösteriyordu; sorun davranış değil, arayüzün bunu
anlatmamasıydı.

### Docker gecikme teşhisi: GPU farkı

Docker'da soru başına turbo ~60-75 s, large ~150 s ölçtüm; local testlerde aynı sorular 14 / 37 s
sürüyordu. Gün 5'in açık sorusu ("Docker'da gerçek CPU gecikmesi") böylece cevaplandı ama sebep
beklediğim gibi "sanallaştırma yükü" değil: **local Ollama, M2'nin entegre GPU'sunu Metal
üzerinden kullanıyormuş** (`ollama ps` → `PROCESSOR: 100% GPU`), Docker Desktop'ın Linux
VM'ine ise GPU geçmiyor (container logu: `inference compute id=cpu`). Yani şimdiye kadar
"local CPU" sandığım LLM sayıları aslında GPU'luydu; Dockerdaki süreler GPU'suz on-prem
CPU'nun dürüst maliyeti. Reranker'da aynı tuzağı 4. günde yakalamıştım (MPS), LLM tarafında
ancak Docker karşılaştırması görünür kıldı.

### Chunk parametre taraması

Retrieval aşamasında yapmadığım chunk boyutu ve örtüşme parametreleri için bir test yaptım.
12 farklı parametre kombinasyonunda yaptığım testlerde, şu anki parametreleri değiştirmeme
sebep olacak bir fark çıkmadı. (Sonuçlar TESTING.md'de)

Bulgular:

- **Chunk boyutunun tutarlı bir etkisi yok.** recall@5 %80-92.5 arasında zıplıyor ama
  bir örüntü yok, komşu kombinasyonlar arasındaki 5-10 puanlık farklar tek bir sorunun 
  chunk sınırına denk gelip gelmemesinden geliyor (n=20'de bir soru 5 puan).
- Tek tutarlı örüntü: **küçük chunk (600) arXiv sorusunu (q009, EN) dense top-5'e
  sokuyor** (EN %75 → %100), karşılığında q011 (TR) düşüyor (TR %93.8 → %90.6). Gün
  4'teki hibrit retrieval takasının birebir aynısı. Ama reranker q009'u dense'in top-10'undan
  zaten kurtarıyor; yani bu kazanç uçtan uca sistemde **tekrarlanıyor**.
- **Karar: 800/150 parametreleri korundu.** Değişiklik uçtan uca kazanç vaat etmiyor; mevcut değerler
  artık "makul başlangıç" değil, **ölçülüp yerinde bırakılmış** değerler.

### Kararlar

- Arayüz: **Streamlit, tek dosya, yalnızca `pipeline.answer`** üzerinden cevap üretimi.
- Runtime indeks **boş başlar**, aynı isimli doküman yüklenirse güncelle; runtime indeks
ölçüm indekslerinden tamamen ayrı. Docker'da named volume ile kalıcı.
- **Tek tur konuşma + görsel geçmiş**; çoklu-tur ölçülemediği için eklenmedi (sistem sınırı).
- İlerleme **aşama aşama**, cevap **streaming** şeklinde gösteriliyor.
- Tanılama panelinde **pasaj filtresi yok** LLM tarafından kullanılan kaynak ✅ ile işaretleniyor.
- Bağımlılıklar **sürüm sabitli**; ölçüm scriptleri `eval/` altında.
- **Reddedilen soruda kaynak/pasaj gösterilmez**.
- **Varsayılan tier turbo** Gün 5 revizyonu, UI deneyimi sonrasında tercihim değişti.
- Gün 3-5'teki süre ölçümlerinin cihaz etiketleri düzeltildi (embedding MPS, reranker CPU,
  LLM Metal).

### Açık sorular

- Generation testindeki hatalar (q022, q041, q051, q052) incelenemedi.
- Ollama ~5 dk boşta kalan modeli bellekten atıyor; tier değişimi/uzun ara sonrası ilk soru
  model yükleme süresi ödüyor. `keep_alive` ile sabitleme ölçülmedi (iki model birden
  bellekte: RAM maliyeti var).

---

## Gün 7 — 29.07.2026

Bugün geliştirme günü değil teslim günüydü. Kod dünkü haliyle tamamlanmıştı; bugün tüm
proje kodunun üstünden son bir kez geçtim, sistemi başka bir makinede repodan klonlayıp
test ettim, demo videosunu ve sunumu hazırladım, README dosyasını son haline getirdim.

### Kod okuması

Commit öncesi tüm modüllerin üstünden geçtim. Davranış değiştiren bir bulgu çıkmadı;
temizlik eski yorumlarla sınırlı kaldı. 
Kararların kodda değil DEVLOG/TESTING'de yaşaması bu okumayı kolaylaştırdı.

### Başka makinede temiz kurulum testi

Sistemi kendi geliştirme ortamım dışında bir makinede iki yoldan da sıfırdan denedim:

- **Docker:** `docker compose up -d --build`, model provizyonu, web arayüzünden belge
  yükleme + soru-cevap. Problem bulunmadı.
- **Local:** venv kurulumu + host Ollama + Streamlit. Problem bulunmadı.

### Baştan başlasam neyi farklı yapardım

- **Golden set'e baştan belirsiz sorular koyardım.** 30 sorunun hepsi net hedefli
  ("bir önceki aya göre yüzde kaç" gibi) ve bu, gemma'nın thinking bütçesinin cevabı
  yutması bug'ını sonuna kadar maskeledi. Gerçek kullanıcının sorduğu türden belirsiz,
  belgede geçmeyen terimli sorulardan birkaç tane sette olsaydı bug Docker testinde değil
  5. gün ölçümünde yakalanırdı.
- **Cihazı ilk gün sabitleyip öyle ölçerdim.** "CPU sandığım süre aslında GPU'ymuş"
  düzeltmesini üç ayrı yerde yaşadım: reranker bench'inde (MPS, 4. gün fark ettim),
  embedding sürelerinde ve LLM gecikmelerinde (Metal, ancak Docker karşılaştırması
  görünür kıldı). Kararlar değişmedi ama süre etiketlerini geriye dönük düzeltmek zorunda
  kaldım; baştan `device` sabitlemek bunların hepsini önlerdi.
- **Hibrit retrieval'a ayırdığım sürenin bir kısmını çelişki kategorisine verirdim.**
  BM25 + RRF taraması öğreticiydi ama sonuç "reranker zaten kapsıyor" oldu; buna karşılık
  çelişki kategorisi (iki belgeden iki değeri kaynağıyla getirme) en zayıf halka olarak
  kaldı ve ona özel bir iyileştirme denemeye zaman kalmadı.