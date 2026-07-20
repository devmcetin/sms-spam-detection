"""
DS-34 — SMS Spam Tespiti (Naive Bayes)
Bir telekom şirketinde data scientist'sin. Gelen SMS'leri spam/ham (normal mesaj)
ayıran bir filtre kuruyorsun. Naive Bayes — metin sınıflandırmanın klasik
baseline'ı (Gmail'in eski spam filtreleri böyleydi).

Her fonksiyonun pass kısmını doldur. Testleri çalıştır, hepsi geçene kadar
iterate et: `python watch.py` veya `pytest tests/test_question.py -v`
"""

import os
import ssl
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import requests

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


# 1. UCI SMS Spam Collection veri setini indir (cache'li)
def fetch_sms_data(cache_dir='data'):
    """
    UCI'den sms+spam+collection.zip indir, aç, TSV dosyasının path'ini döndür.

    Akış:
    1. cache_dir mevcut değilse oluştur (os.makedirs(cache_dir, exist_ok=True))
    2. Hedef dosya path: cache_dir/SMSSpamCollection (uzantısız bir TSV dosyası)
    3. Bu dosya zaten varsa → hemen path'i döndür (cache hit, tekrar indirme)
    4. Yoksa:
       a. URL'den zip indir:
          https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
       b. Zip'i cache_dir'a aç (zipfile.ZipFile + extractall) — içinde
          uzantısız 'SMSSpamCollection' dosyası var (TSV: label<TAB>mesaj)
    5. cache_dir/SMSSpamCollection path'ini döndür

    Args:
        cache_dir: lokal cache klasörü (default 'data')

    Returns:
        str: SMSSpamCollection dosyasının tam yolu

    İpucu: import os, zipfile, requests
    - İndirme: r = requests.get(URL, timeout=60); r.raise_for_status()
      with open(zip_path, 'wb') as f: f.write(r.content)
    - DİKKAT: UCI sertifikası bazı sistemlerde SSL hatası verir. requests
      requests.exceptions.SSLError fırlatırsa, ssl + urllib.request ile
      sertifika doğrulamasını kapatıp tekrar dene (fallback):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(URL, context=ctx, timeout=60) as resp: ...
    URL: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
    """
    os.makedirs(cache_dir, exist_ok=True)
    data_path = os.path.join(cache_dir, 'SMSSpamCollection')

    if os.path.exists(data_path):
        return data_path

    zip_path = os.path.join(cache_dir, 'sms+spam+collection.zip')
    if not os.path.exists(zip_path):
        try:
            r = requests.get(UCI_URL, timeout=60)
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                f.write(r.content)
        except requests.exceptions.SSLError:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(UCI_URL, context=ctx, timeout=60) as resp:
                with open(zip_path, 'wb') as f:
                    f.write(resp.read())

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(cache_dir)

    return data_path


# 2. TSV'yi DataFrame olarak yükle
def load_sms_data(path):
    """
    SMSSpamCollection dosyasını oku. Format: tab ile ayrılmış (TSV),
    başlık satırı YOK, iki sütun: label ve mesaj metni.

    Args:
        path: TSV dosya yolu (fetch_sms_data'dan dönen)

    Returns:
        pd.DataFrame: 5572 satır × 2 sütun ['label', 'message']

    İpucu: pd.read_csv(path, sep='\\t', names=['label', 'message'])
    - sep='\\t' → tab ayırıcı
    - names=[...] → dosyada başlık olmadığı için sütun adlarını biz veriyoruz
    """
    return pd.read_csv(path, sep="\t", names=["label", "message"])


# 3. Veriyi keşfet — label dağılımı
def explore_data(df):
    """
    Temel keşif metriği üret.

    Returns:
        dict: {
            'total': int (toplam mesaj sayısı),
            'ham_count': int ('ham' = normal mesaj sayısı),
            'spam_count': int ('spam' mesaj sayısı),
            'spam_rate': float (spam_count / total — ~0.13)
        }

    İpucu:
    - df['label'].value_counts() → 'ham' ve 'spam' sayıları
    - total = len(df)
    """
    
    counts = df["label"].value_counts()
    total = len(df)

    return {
        "total": total,
        "ham_count": counts["ham"],
        "spam_count": counts["spam"],
        "spam_rate": counts["spam"] / total
    }


# 4. Label'ları encode et (spam/ham → 1/0)
def encode_labels(df):
    """
    'label' sütununu binary target'a çevir: 'spam' → 1, 'ham' → 0.
    Yeni bir 'target' sütununa yaz (orijinal 'label' kalsın).

    Args:
        df: 'label' sütunu olan DataFrame

    Returns:
        pd.DataFrame: yeni 'target' sütunu (int 0/1) eklenmiş kopya

    İpucu:
    - out = df.copy()
    - out['target'] = out['label'].map({'spam': 1, 'ham': 0}).astype(int)
    """
    
    df_copy = df.copy()
    df_copy["target"] = df_copy["label"].map({"ham": 0, "spam": 1})
    
    return df_copy


# 5. Train/test split (stratified)
def split_data(X, y):
    """
    train_test_split kullan:
    - X: mesaj metinleri (pd.Series — ham string'ler, vektörize EDİLMEMİŞ)
    - y: target (0/1)
    - test_size=0.2
    - stratify=y (spam/ham oranı train ve test'te korunsun — dengesiz veri)
    - random_state=42 (tekrarlanabilirlik)

    Returns:
        tuple: (X_train, X_test, y_train, y_test)

    İpucu: from sklearn.model_selection import train_test_split
    """
    return train_test_split(X, y, random_state=42, stratify=y, test_size=0.2)


# 6. CountVectorizer + MultinomialNB pipeline kur
def build_vectorizer_pipeline():
    """
    sklearn Pipeline:
    - 'cv': CountVectorizer(stop_words='english')
            → metni "bag of words" sayım vektörüne çevirir
            → stop_words='english' → 'the', 'a', 'is' gibi anlamsız
              İngilizce kelimeleri eler
    - 'nb': MultinomialNB()
            → kelime sayımları için klasik Naive Bayes (metin sınıflandırma
              için standart seçim)

    Returns:
        sklearn.pipeline.Pipeline

    İpucu:
    - from sklearn.feature_extraction.text import CountVectorizer
    - from sklearn.naive_bayes import MultinomialNB
    - from sklearn.pipeline import Pipeline
    - Pipeline ham string listesini alır, CountVectorizer otomatik vektörize eder.
    """
    return Pipeline([
        ("cv", CountVectorizer(stop_words="english")),
        ("nb", MultinomialNB())
    ])


# 7. Modeli eğit
def train_model(pipe, X_train, y_train):
    """
    Pipeline'ı fit et ve döndür.

    Args:
        pipe: build_vectorizer_pipeline'dan dönen pipeline
        X_train: mesaj metinleri (ham string'ler)
        y_train: target (0/1)

    Returns:
        Pipeline: fit edilmiş pipeline

    İpucu: pipe.fit(X_train, y_train); return pipe
    """
    return pipe.fit(X_train, y_train)


# 8. Modeli değerlendir
def evaluate_model(pipe, X_test, y_test):
    """
    Test setinde tahmin yap, metrikleri hesapla.

    DİKKAT: Spam filtresinde RECALL önemli — spam'i kaçırmamak istiyoruz
    (yüksek recall = daha az spam gelen kutuna sızıyor).

    Returns:
        dict: {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'confusion_matrix': np.array (2x2)
        }

    İpucu:
    - y_pred = pipe.predict(X_test)
    - sklearn.metrics'ten: accuracy_score, precision_score, recall_score,
      f1_score, confusion_matrix
    - precision/recall/f1'de zero_division=0 kullan
    """
    y_pred = pipe.predict(X_test)
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }


# 9. Tek bir mesaj için tahmin yap
def predict_message(pipe, text):
    """
    Bir SMS metni al, spam/ham tahmini + spam olasılığı döndür.

    Args:
        pipe: eğitilmiş pipeline
        text: str (bir SMS mesajı)

    Returns:
        dict: {
            'prediction': int (0=ham, 1=spam),
            'label': str ('spam' veya 'ham'),
            'spam_probability': float (spam olma olasılığı)
        }

    İpucu:
    - Pipeline tek string değil, liste bekler → [text] geç
    - proba = pipe.predict_proba([text])[0, 1]  # pozitif (spam) sınıf olasılığı
    - pred = int(pipe.predict([text])[0])
    """
    pred = pipe.predict([text])[0]
    proba = pipe.predict_proba([text])[0, 1]
    
    return {
        "prediction": pred,
        "label": "spam" if pred == 1 else "ham",
        "spam_probability": proba
    }


# 10. En "spam" kelimeler
def top_spam_words(pipe, n=10):
    """
    Modelin spam'i en çok ayırt eden n kelimesini bul.

    MultinomialNB.feature_log_prob_ → her sınıf için her kelimenin log
    olasılığı. feature_log_prob_[1] = spam sınıfı, feature_log_prob_[0] = ham.
    İkisinin FARKI en büyük olan kelimeler → "en spam" kelimeler
    (örn. 'free', 'call', 'txt', 'win', 'prize' gibi bekleriz).

    Args:
        pipe: eğitilmiş pipeline
        n: kaç kelime döndürülecek (default 10)

    Returns:
        list: n adet kelime (str), en spam'den aza doğru sıralı

    İpucu:
    - cv = pipe.named_steps['cv']; nb = pipe.named_steps['nb']
    - feature_names = cv.get_feature_names_out()
    - diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    - top_idx = np.argsort(diff)[::-1][:n]  # büyükten küçüğe ilk n
    - [feature_names[i] for i in top_idx]
    """
    
    cv = pipe.named_steps["cv"]
    nb = pipe.named_steps["nb"]
    
    feature_names = cv.get_feature_names_out()
    
    diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    top_idx = np.argsort(diff)[::-1][:n]

    return [feature_names[i] for i in top_idx]


# 11. En "ham" kelimeler
def top_ham_words(pipe, n=10):
    """
    top_spam_words'ün tersi — normal (ham) mesajları en çok ayırt eden n kelime.

    Args:
        pipe: eğitilmiş pipeline
        n: kaç kelime (default 10)

    Returns:
        list: n adet kelime (str), en ham'dan aza doğru sıralı

    İpucu:
    - diff = nb.feature_log_prob_[0] - nb.feature_log_prob_[1]  # ham - spam
    - top_idx = np.argsort(diff)[::-1][:n]
    """
    
    cv = pipe.named_steps["cv"]
    nb = pipe.named_steps["nb"]
    
    feature_names = cv.get_feature_names_out()
    
    diff = nb.feature_log_prob_[0] - nb.feature_log_prob_[1]
    top_idx = np.argsort(diff)[::-1][:n]
    
    return [feature_names[i] for i in top_idx]


# 12. Stopwords var/yok karşılaştırması
def compare_with_without_stopwords(X_train, X_test, y_train, y_test):
    """
    İki pipeline eğit ve test F1'lerini karşılaştır:
    - with_stopwords:    CountVectorizer(stop_words='english')
    - without_stopwords: CountVectorizer(stop_words=None)  # hiç kelime elenmez

    Her ikisi de MultinomialNB ile. Amaç: stopword temizliğinin F1'e etkisini
    gözlemlemek.

    Returns:
        dict: {
            'with_stopwords': float (F1),
            'without_stopwords': float (F1)
        }

    İpucu: İki ayrı Pipeline kur, fit et, pipe.predict(X_test) → f1_score.
    """
    
    results = {}
    
    for label, sw in [("with_stopwords", "english"), ("without_stopwords", None)]:
        pipeline = Pipeline([
            ("cv", CountVectorizer(stop_words=sw)),
            ("nb", MultinomialNB())
        ]).fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        
        results[label] = f1
    
    return results


# 13. Toplu tahmin
def predict_batch(pipe, messages):
    """
    Birden çok mesaj için tahmin yap.

    Args:
        pipe: eğitilmiş pipeline
        messages: list of str (SMS mesajları)

    Returns:
        list: her mesaj için predict_message çıktısı (list of dict)

    İpucu: [predict_message(pipe, m) for m in messages]
    """
    
    return [predict_message(pipe, message) for message in messages]


# 14. Tüm pipeline'ı uçtan uca çalıştır
def run_pipeline():
    """
    Uçtan uca akış:
    1. fetch_sms_data → load_sms_data
    2. explore_data (spam_rate al)
    3. encode_labels
    4. split_data (X = df['message'], y = df['target'])
    5. build_vectorizer_pipeline → train_model
    6. evaluate_model (test F1, recall)
    7. top_spam_words (en spam kelime)
    8. predict_message ile iki örnek mesaj test et (biri spam, biri ham)

    Örnek mesajlar (önerilen):
    - Spam: "WINNER!! You have won a FREE prize. Call now to claim your reward!"
    - Ham:  "Hey, are we still meeting for lunch tomorrow?"

    Returns:
        dict: {
            'spam_rate': float,
            'test_f1': float,
            'test_recall': float,
            'top_spam_word': str (top_spam_words listesinin ilki),
            'sample_spam_pred': dict (predict_message çıktısı),
            'sample_ham_pred': dict (predict_message çıktısı)
        }
    """
        
    # 1. fetch_sms_data → load_sms_data
    path = fetch_sms_data()
    df = load_sms_data(path)
    
    # 2. explore_data (spam_rate al)
    info = explore_data(df)
    
    # 3. encode_labels
    df = encode_labels(df)
    
    # 4. split_data (X = df['message'], y = df['target'])
    X_train, X_test, y_train, y_test = split_data(df["message"], df["target"])
    
    # 5. build_vectorizer_pipeline → train_model
    pipeline = build_vectorizer_pipeline()
    pipeline = train_model(pipeline, X_train, y_train)
    
    # 6. evaluate_model (test F1, recall)
    metrics = evaluate_model(pipeline, X_test, y_test)
    
    # 7. top_spam_words (en spam kelime)
    top_spams = top_spam_words(pipeline)
    
    # 8. predict_message ile iki örnek mesaj test et (biri spam, biri ham)
    sample_spam = predict_message(
        pipeline,
        "WINNER!! You have won a FREE prize. Call now to claim your reward!",
    )
    sample_ham = predict_message(
        pipeline,
        "Hey, are we still meeting for lunch tomorrow?",
    )
    
    return {
        "spam_rate": info["spam_rate"],
        "test_f1": metrics["f1"],
        "test_recall": metrics["recall"],
        "top_spam_word": top_spams[0],
        "sample_spam_pred": sample_spam,
        "sample_ham_pred": sample_ham
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 Pipeline Sonuçları:")
    print(f"  Spam rate        : {result['spam_rate']:.2%}")
    print(f"  Test F1          : {result['test_f1']:.4f}")
    print(f"  Test Recall      : {result['test_recall']:.4f}")
    print(f"  En spam kelime   : {result['top_spam_word']}")
    print(f"  Spam örnek tahmin: {result['sample_spam_pred']['label']} "
          f"(p={result['sample_spam_pred']['spam_probability']:.3f})")
    print(f"  Ham örnek tahmin : {result['sample_ham_pred']['label']} "
          f"(p={result['sample_ham_pred']['spam_probability']:.3f})")
