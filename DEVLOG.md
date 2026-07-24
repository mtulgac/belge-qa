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