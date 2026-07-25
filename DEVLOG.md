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
  `ocr_confidence` yüksek olanı seçmek — maliyeti ölçülecek.

- Tablolar nasıl parse edilecek? Tesseract yapıyı düzleştiriyor, CER metriği bunu tam ölçmüyor. 
Retrieval aşamasında tablo sorularıyla ayrıca test edilecek.