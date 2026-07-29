# Doküman Bilgi Erişimi (RAG)

Bu repo, Case çalışması kapsamında kullanıcının PDF/JPG/PNG formatında belgeler yükleyip doğal dilde soru sorabildiği, cevapları **yalnız belgelere dayalı** ve **Türkçe + İngilizce** çalışan bir doküman bilgi erişim sistemini içermektedir. 

Tasarım kararları ve ölçümler: [`DEVLOG.md`](DEVLOG.md) ve [`TESTING.md`](TESTING.md).

---

## 1. Docker ile çalıştırma (sıfırdan)

Repoyu yeni klonlayan biri için uçtan uca akış. İki servis ayağa kalkar: `ollama` (LLM
sunucusu) ve `app` (web UI + QA uygulaması). Model'ler image'a gömülü değil, ilk açılışta
named volume'lara inip orada kalır, sonrası offline çalışır.

### Gereksinimler

- **Docker Desktop** (compose v2).
- **Docker VM belleği ≥ 12 GB** `large` model (gemma4:e4b) belleğe 8.5 GB ağırlık yüklüyor;
  8 GB VM'de OOM oluyor. Yalnız `turbo` kullanıldığında 8 GB yeter. 
  
### İilk sefer image build + model indirme

```bash
git clone <repo-url>
cd belge_qa
docker compose up -d --build
```

İlk kurulumda `PROVISION=1` iki LLM'i Ollama volume'una **sırayla** çeker:
gemma4:e4b (~9.6 GB) sonra qwen3.5:4b (~3.4 GB). Ağ hızına göre **~15-20 dk** sürebilir.
İlerlemeyi izlemek için:

```bash
docker compose logs -f app               # "provizyon tamam." görünene kadar
docker compose exec ollama ollama list   # iki model de listelenmeli
```

> İki model listede görünmeden soru sorulursa `model not found` döner, indirme bitmemiştir.

### Web arayüzü

Arayüze tarayıcıda **http://localhost:8501** linkinden erişilebilir. 
Arayüz yalnızca yüklenen belgelerde arama yapar (indeks boş başlar); cevabın altında kaynak
(belge + sayfa) çipleri, cevap belgede yoksa açık bir **YANITLANAMADI** kartı görülür.

### CLI (opsiyonel, ölçüm korpusu üzerinde)

Web arayüzü yalnızca yüklenen belgelerde arar; CLI ise `data/samples/` altındaki ölçüm
korpusunda çalışır. Repo hazır indeks içermediği için CLI'da çalışmadan önce indeksin bir kez
oluşturulması gerekiyor (indeks container içinde kalır, `--build` ile image yeniden
oluşturulursa tekrar gerekir):

```bash
docker compose exec app python -m app.ingest                               # bir kez: indeksi kur
docker compose exec app python -m app.cli "azami süre kaç dönem?"          # turbo (varsayılan)
docker compose exec app python -m app.cli --tier large "azami süre kaç dönem?"
docker compose exec app python -m app.cli                                  # etkileşimli mod
```

### Durdur / tekrar başlat / güncelle / sıfırla

```bash
docker compose stop
docker compose up -d             # anında; model volume'ları duruyor, tekrar inmez
docker compose up -d --build     # kod değişince bu çalıştırılacak
docker compose down -v           # TAM sıfırlama: volume'lar (modeller + indeks) dahil silinir
```

---

## 2. Yerel çalıştırma (Docker'sız, Streamlit)

Geliştirme ve hızlı demo için bu yol tercih edilebilir. LLM host'taki Ollama'da Metal/GPU kullanır, 
soru cevaplama hızı Docker VM'den yaklaşık 4 kat hızlıdır (TESTING.md)

### Gereksinimler ve kurulum

```bash
# 1. Sistem bağımlılıkları (macOS)
brew install tesseract           # OCR; tur+eng dil paketleri dahil
brew install ollama              # LLM sunucusu (ya da ollama.com'dan uygulama)

# 2. Python ortamı (3.11)
python3.11 -m venv .venv
.venv/bin/pip install -e .
```

### Ollama: sunucu + modeller (bir kez)

Uygulama LLM'e `http://localhost:11434`'te çalışan Ollama üzerinden bağlanır; sunucunun
açık ve iki modelin de çekili olması gerekir:

```bash
ollama serve                     # sunucuyu başlat (macOS uygulaması kuruluysa zaten çalışır;

ollama pull qwen3.5:4b           # turbo, varsayılan model (~3.4 GB)
ollama pull gemma4:e4b           # large, reasoning model (~9.6 GB)

ollama list                      # iki model de listede görünmeli
```

Modeller bir kez iner, `~/.ollama`'da kalır. Notlar:

- Yalnız `turbo` ile denenmek istenirse `gemma4:e4b`'yi çekmeden başlanabilir. `large`
  seçildiğinde model yoksa cevap `model not found` hatasıyla döner.

### Web arayüzü

```bash
.venv/bin/streamlit run src/app/webui.py
```

Tarayıcıda **http://localhost:8501** (kullanım yukarıdaki *Web arayüzü* bölümüyle aynı).
İlk çalıştırmada embedder + reranker Hugging Face'ten iner (~4.5 GB) ve süreç başına bir kez
belleğe yüklenir (~15-20 s açılış ısıtması).

### CLI

CLI `data/samples/` altındaki ölçüm korpusunda arar; ilk kullanımdan önce indeksi bir kez
oluşturmak gerekiyor (repo hazır indeks içermez):

```bash
.venv/bin/python -m app.ingest                                   # bir kez: indeksi kur
.venv/bin/python -m app.cli "azami süre kaç dönem?"              # turbo (varsayılan)
.venv/bin/python -m app.cli --tier large "azami süre kaç dönem?" # large (reasoning)
.venv/bin/python -m app.cli                                      # etkileşimli mod
```

### Testler ve ölçüm scriptleri

```bash
.venv/bin/python -m pytest tests/    # modelden bağımsız testler (9 test)
```

Ölçüm kodları `eval/` altındadır, hangi scriptin hangi soruyu cevapladığı ve tüm sonuçlar
[`TESTING.md`](TESTING.md)'de.
