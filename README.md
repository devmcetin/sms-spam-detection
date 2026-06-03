# Data Science Project 34 — SMS Spam Tespiti (Naive Bayes)

**Modül**: ML-04 (Sınıflandırma 2) • **Süre**: 3-4 saat

## 🎯 Proje Senaryosu

Bir telekom şirketinde **data scientist** olarak çalışıyorsun. Müşterilere her gün milyonlarca SMS geliyor ve bunların önemli bir kısmı **spam**: "FREE prize", "Call now to win", "Click this link" gibi reklam ve dolandırıcılık mesajları. Şirket, gelen SMS'leri **spam** ve **ham** (normal mesaj) olarak ayıran bir filtre kurmanı istiyor.

Senin görevin: bir SMS metnine bakıp **spam mı, normal mı** olduğunu tahmin eden bir model kurmak. Doğru çalışırsa spam'ler kullanıcıya ulaşmadan ayrı klasöre gider, normal mesajlar engellenmeden geçer.

Bunun için **Naive Bayes** kullanacaksın — metin sınıflandırmanın klasik baseline'ı. Gmail ve Hotmail'in eski spam filtreleri tam olarak böyle çalışıyordu: her kelimenin spam olma olasılığını sayar, mesajdaki kelimeleri birleştirip karar verir. Basit ama metin için şaşırtıcı derecede güçlü.

Veride mesajların **~%13'ü spam**, geri kalanı normal. Yani hafif **dengesiz (imbalanced)** bir problem. Spam filtresinde **recall önemli**: bir spam'i kaçırıp gelen kutusuna sızdırmak istemiyoruz.

Bu projede ML-04 dersinde öğrendiklerini birleştirip uygulayacaksın:
- ✅ **Gerçek dünya veri çekme** (URL → zip → TSV cache)
- ✅ **Metin → sayı** dönüşümü (`CountVectorizer` — bag of words)
- ✅ **MultinomialNB** (metin için Naive Bayes)
- ✅ **stop_words** (anlamsız kelimeleri eleme)
- ✅ **Stratified train/test split**
- ✅ **Precision / Recall / F1** (spam'de recall kritik)
- ✅ **`feature_log_prob_` ile "en spam kelimeler"**
- ✅ **sklearn Pipeline** (vektörize + model tek akışta)

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd data-science-project-34

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Dependencies
pip install -r requirements.txt

# Auto test runner (dosya değişince çalışır)
python watch.py

# Manuel test
pytest tests/test_question.py -v
```

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

Skorunun Kaizu hesabına yazılması için **`kaizu_config.py`** dosyasını aç ve **`USER_ID`** alanını kendi user_id'nle değiştir:

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 714 # ← Bu projeye ait, dokunma
```

User_id'ni Kaizu profilinden bulabilirsin (Profile → Settings → User ID).

Skor göndermek için tüm testleri toplu çalıştırmalısın:

```bash
python tests/test_question.py
```

Bu komut tüm testleri çalıştırır, **passed/total oranını otomatik Kaizu'ya gönderir**. Geliştirme sırasında `pytest -v` kullanmaya devam edebilirsin (skor göndermez).

## 📚 Dataset — UCI SMS Spam Collection

### Kaynak
- **UCI Machine Learning Repository** — [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- Akademik kaynak: Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). *Contributions to the study of SMS spam filtering: new collection and results.* Proceedings of the 11th ACM Symposium on Document Engineering (DocEng'11).

### Veri Çekme Yöntemi (önemli)
Veri seti **repo'da YOK** — projeyi clone ettiğin gün **runtime'da indireceksin**. Gerçek dünyada veriler S3 / API / FTP üzerinden gelir; statik dosya olarak gönderilmez. Bu yüzden `fetch_sms_data()` fonksiyonun:

1. Hedef URL: `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`
2. `data/` klasörüne zip'i indirir (`requests.get` — SSL hatası olursa `urllib` fallback)
3. Zip'i açar → içinde **uzantısız** `SMSSpamCollection` dosyası var (TSV formatında)
4. Cache: dosya zaten varsa **tekrar indirme**, hemen path döndür

> **Not — SSL fallback:** UCI sunucusunun sertifikası bazı sistemlerde doğrulanamıyor. `requests` `SSLError` fırlatırsa, kod `ssl` + `urllib.request` ile sertifika doğrulamasını kapatıp tekrar indirir. Bu sadece bu güvenilir akademik kaynak için bir workaround.

### Boyut & Target
- **5572 mesaj × 2 sütun**
- Target: `label` (`ham` / `spam`) — mesaj normal mi spam mi?
- Class dağılımı:
  - `ham`: ~4825 (%86.6) ← normal mesaj
  - `spam`: ~747 (%13.4) ← **azınlık sınıf**
- **Hafif dengesiz** — sadece accuracy yanıltıcı olabilir. Precision/Recall/F1 kullan.

### Dosya Formatı
- **TSV** (tab ile ayrılmış), başlık satırı YOK
- Her satır: `label<TAB>mesaj metni`
- `pd.read_csv(path, sep='\t', names=['label', 'message'])` ile oku.

### Örnek Satırlar (3 örnek)
```
label | message
ham   | Ok lar... Joking wif u oni...
ham   | Even my brother is not like to speak with me. They treat me like aids patent.
spam  | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)
```
Spam mesajlarda **FREE**, **win**, **Text/Txt**, **Call**, telefon numaraları ve para/ödül vurgusu sık görülür. Modelin tam da bu kalıpları öğrenmesini bekliyoruz.

### Domain Notu
Veriler **2011 yılında** İngiltere ve Singapur kaynaklı gerçek SMS'lerden toplanmış (kısmen Grumbletext web sitesinden, kısmen NUS SMS Corpus'tan). Mesajlar İngilizce ve günlük dil (kısaltmalı, "u", "wif", "lar" gibi). Spam'ler dönemin SMS dolandırıcılık kalıplarını yansıtıyor: premium-rate numaralara "Text X to 8xxxx" çağrıları, sahte yarışma/ödül duyuruları. Bu yüzden **kelime sayımı (bag of words)** tek başına çok güçlü bir sinyal.

## 📋 Görevler (`tasks/task_manager.py`)

`task_manager.py` dosyasındaki **14 fonksiyonu** sırayla doldur. Her task altta testler pass olana kadar düzenlenmeli.

1. **`fetch_sms_data(cache_dir='data')`** — URL'den zip indir, aç, TSV path döndür (cache'li)
2. **`load_sms_data(path)`** — `pd.read_csv(path, sep='\t', names=['label','message'])`
3. **`explore_data(df)`** — total, ham_count, spam_count, spam_rate
4. **`encode_labels(df)`** — `label`: spam→1, ham→0 (yeni `target` sütunu)
5. **`split_data(X, y)`** — 80/20, stratify=y, random_state=42
6. **`build_vectorizer_pipeline()`** — CountVectorizer(stop_words='english') + MultinomialNB
7. **`train_model(pipe, X_train, y_train)`** — fit + dön
8. **`evaluate_model(pipe, X_test, y_test)`** — accuracy, precision, recall, f1, confusion matrix
9. **`predict_message(pipe, text)`** — tek mesaj için tahmin + spam olasılığı
10. **`top_spam_words(pipe, n=10)`** — `feature_log_prob_` farkıyla en spam n kelime
11. **`top_ham_words(pipe, n=10)`** — tersi, en ham n kelime
12. **`compare_with_without_stopwords(...)`** — stop_words='english' vs None F1 karşılaştır
13. **`predict_batch(pipe, messages)`** — birden çok mesaj için tahmin listesi
14. **`run_pipeline()`** — uçtan uca akış, özet dict dön

## 🎓 Öğrenme Hedefleri

Bu projeyi bitirdiğinde:
- [x] Gerçek dünyada **veri çekme** (URL + zip + cache) yapabileceksin
- [x] **Metni sayıya** çevirebileceksin (`CountVectorizer`, bag of words)
- [x] **MultinomialNB** ile metin sınıflandırma yapabileceksin
- [x] **stop_words** temizliğinin etkisini ölçebileceksin
- [x] **Stratified split** ile dengesiz veride sınıf oranını koruyabileceksin
- [x] **Precision / Recall / F1** hesaplayıp spam'de neden recall önemli anlayacaksın
- [x] **`feature_log_prob_`** ile modelin "en spam kelimelerini" çıkarabileceksin
- [x] **sklearn Pipeline** ile vektörize + model adımlarını tek akışta birleştirebileceksin

## 🧪 Testler

Test dosyası: `tests/test_question.py` (17 test)

Tümü pass olmalı:
- Dataset fetch + zip açma (internet gerekli, ilk test indirir)
- Satır sayısı (5572) ve label değerleri doğru mu
- Spam oranı ~%13 tespit edilmiş mi
- `target` 0/1 encode edilmiş mi (spam=1, ham=0)
- Stratified split korunmuş mu
- Pipeline tipi doğru mu (CountVectorizer + MultinomialNB)
- F1 > 0.85 (model çok güçlü)
- Spam/ham örnek mesajlar doğru sınıflanıyor mu
- En spam kelimeler mantıklı mı ('free', 'call', 'txt' gibi)

## 📊 Beklenen Sonuçlar

```
Spam rate: ~%13.4
Test F1: ~0.90-0.95 (MultinomialNB SMS spam'de çok güçlü)
Test Recall: ~0.85-0.95 (spam'leri yüksek oranda yakalar)
En spam kelimeler: free, call, txt, claim, mobile, win, prize ...
En ham kelimeler: günlük konuşma / kişisel kelimeler
```

## 💡 İpuçları

- **İlk testte internet** gerekli (zip indirme). Sonraki testler cache'den okur.
- `pd.read_csv(path, sep='\t', names=['label', 'message'])` — tab ayırıcı + başlık yok
- Pipeline ham string listesi alır; CountVectorizer otomatik vektörize eder — sen elle vektörize etme
- `predict_message`'ta tek string değil **liste** geç: `pipe.predict_proba([text])`
- `feature_log_prob_[1]` = spam, `[0]` = ham. Farkları en büyük kelimeler en "spam"
- `np.argsort(diff)[::-1][:n]` → en büyük n değerin indeksleri
- Spam filtresinde **recall** önemli: spam'i kaçırmak (false negative) gelen kutuna sızar
- `precision_score`, `recall_score`, `f1_score` — hepsi `sklearn.metrics`

## 🚫 Dikkat

- `tests/test_question.py` dosyasını **değiştirme**
- `random_state=42` değerini değiştirme (testler fail olur)
- `_solution/` klasörü yok (DB'de saklanır, dersin haftası geçince açılır)
- `data/` klasörü repo'ya **gitmez** (.gitignore'da exclude)
- Dokunabileceğin **2 dosya**: `tasks/task_manager.py` (kodu yaz) + `kaizu_config.py` (sadece USER_ID)
