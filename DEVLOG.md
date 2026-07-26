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