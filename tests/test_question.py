import pytest
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    fetch_sms_data, load_sms_data, explore_data, encode_labels,
    split_data, build_vectorizer_pipeline, train_model, evaluate_model,
    predict_message, top_spam_words, top_ham_words,
    compare_with_without_stopwords, predict_batch, run_pipeline,
)


# ──────────────────────────────────────────────────────
# Modül-seviye cache — testler arası tekrar indirme/eğitme yok
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def data_path():
    """İlk testte internet'ten indirir, sonraki tüm testler cache'den okur."""
    return fetch_sms_data()


@pytest.fixture(scope="module")
def raw_df(data_path):
    return load_sms_data(data_path)


@pytest.fixture(scope="module")
def split(raw_df):
    df = encode_labels(raw_df)
    return split_data(df['message'], df['target'])


@pytest.fixture(scope="module")
def trained_pipeline(split):
    X_train, X_test, y_train, y_test = split
    pipe = build_vectorizer_pipeline()
    return train_model(pipe, X_train, y_train)


# 1. fetch_sms_data
def test_fetch_sms_data_returns_path(data_path):
    assert isinstance(data_path, str)
    assert os.path.exists(data_path)
    assert data_path.endswith('SMSSpamCollection')


# 2. load_sms_data
def test_load_sms_data_shape(raw_df):
    assert isinstance(raw_df, pd.DataFrame)
    assert raw_df.shape == (5572, 2)
    assert list(raw_df.columns) == ['label', 'message']


# 3. load — label değerleri
def test_load_sms_data_labels(raw_df):
    assert set(raw_df['label'].unique()) == {'ham', 'spam'}


# 4. explore_data
def test_explore_data_structure(raw_df):
    info = explore_data(raw_df)
    assert set(info.keys()) >= {'total', 'ham_count', 'spam_count', 'spam_rate'}
    assert info['total'] == 5572
    assert info['ham_count'] > info['spam_count']
    assert info['ham_count'] + info['spam_count'] == info['total']
    assert 0.10 < info['spam_rate'] < 0.16  # ~%13


# 5. encode_labels
def test_encode_labels_binary(raw_df):
    df = encode_labels(raw_df)
    assert 'target' in df.columns
    assert set(df['target'].unique()) == {0, 1}
    assert df['target'].dtype in (np.int64, np.int32, int)
    # spam → 1, ham → 0
    assert df[df['label'] == 'spam']['target'].iloc[0] == 1
    assert df[df['label'] == 'ham']['target'].iloc[0] == 0


# 6. split_data
def test_split_data_stratified(raw_df):
    df = encode_labels(raw_df)
    X_train, X_test, y_train, y_test = split_data(df['message'], df['target'])
    # 80/20
    assert abs(len(X_train) / len(df) - 0.8) < 0.01
    # Stratify: train ve test spam oranı yakın olmalı
    assert abs(y_train.mean() - y_test.mean()) < 0.01


# 7. build_vectorizer_pipeline
def test_build_vectorizer_pipeline_type():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    pipe = build_vectorizer_pipeline()
    assert isinstance(pipe, Pipeline)
    names = dict(pipe.steps)
    assert 'cv' in names and 'nb' in names
    assert isinstance(names['cv'], CountVectorizer)
    assert isinstance(names['nb'], MultinomialNB)
    assert names['cv'].stop_words == 'english'


# 8. train_model
def test_train_model_fits(trained_pipeline, split):
    X_train, X_test, y_train, y_test = split
    preds = trained_pipeline.predict(X_test[:5])
    assert len(preds) == 5


# 9. evaluate_model
def test_evaluate_model(trained_pipeline, split):
    _, X_test, _, y_test = split
    res = evaluate_model(trained_pipeline, X_test, y_test)
    assert set(res.keys()) >= {
        'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'
    }
    assert 0 <= res['accuracy'] <= 1
    assert res['confusion_matrix'].shape == (2, 2)
    # Naive Bayes SMS spam'de çok güçlü
    assert res['f1'] > 0.85


# 10. predict_message — spam
def test_predict_message_spam(trained_pipeline):
    out = predict_message(
        trained_pipeline,
        "Congratulations! You won a FREE entry to claim your cash prize, call now!",
    )
    assert set(out.keys()) >= {'prediction', 'label', 'spam_probability'}
    assert out['prediction'] == 1
    assert out['label'] == 'spam'
    assert 0 <= out['spam_probability'] <= 1
    assert out['spam_probability'] > 0.5


# 11. predict_message — ham
def test_predict_message_ham(trained_pipeline):
    out = predict_message(
        trained_pipeline,
        "Ok i will call you when i reach home, see you soon",
    )
    assert out['prediction'] == 0
    assert out['label'] == 'ham'
    assert out['spam_probability'] < 0.5


# 12. top_spam_words
def test_top_spam_words(trained_pipeline):
    words = top_spam_words(trained_pipeline, n=10)
    assert isinstance(words, list)
    assert len(words) == 10
    assert all(isinstance(w, str) for w in words)
    # Spam-tipik kelimelerden en az biri olmalı
    spammy = {'free', 'call', 'txt', 'text', 'win', 'won', 'prize', 'claim',
              'mobile', 'cash', 'reply', 'stop', 'urgent', 'award'}
    assert len(set(w.lower() for w in words) & spammy) >= 1


# 13. top_ham_words
def test_top_ham_words(trained_pipeline):
    words = top_ham_words(trained_pipeline, n=10)
    assert isinstance(words, list)
    assert len(words) == 10
    # Ham ve spam kelimeleri farklı olmalı
    spam_words = top_spam_words(trained_pipeline, n=10)
    assert set(words) != set(spam_words)


# 14. compare_with_without_stopwords
def test_compare_with_without_stopwords(split):
    X_train, X_test, y_train, y_test = split
    cmp = compare_with_without_stopwords(X_train, X_test, y_train, y_test)
    assert set(cmp.keys()) >= {'with_stopwords', 'without_stopwords'}
    # İkisi de iyi performans göstermeli
    assert cmp['with_stopwords'] > 0.8
    assert cmp['without_stopwords'] > 0.8


# 15. predict_batch
def test_predict_batch(trained_pipeline):
    messages = [
        "WINNER! Claim your free prize now by calling this number",
        "Hey what time are you coming over tonight?",
    ]
    out = predict_batch(trained_pipeline, messages)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all('prediction' in o for o in out)
    # İlki spam, ikincisi ham
    assert out[0]['prediction'] == 1
    assert out[1]['prediction'] == 0


# 16. run_pipeline
def test_run_pipeline_full():
    result = run_pipeline()
    assert set(result.keys()) >= {
        'spam_rate', 'test_f1', 'test_recall',
        'top_spam_word', 'sample_spam_pred', 'sample_ham_pred'
    }
    assert result['test_f1'] > 0.85
    assert 0.10 < result['spam_rate'] < 0.16
    # Örnek tahminler doğru
    assert result['sample_spam_pred']['prediction'] == 1
    assert result['sample_ham_pred']['prediction'] == 0
    # En spam kelime mantıklı
    assert isinstance(result['top_spam_word'], str)
    assert len(result['top_spam_word']) > 0


# 17. End-to-end sanity (recall yüksek olmalı — spam kaçırmıyoruz)
def test_run_pipeline_recall():
    result = run_pipeline()
    assert result['test_recall'] > 0.7


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
