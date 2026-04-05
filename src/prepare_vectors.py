import os
import numpy as np
import pandas as pd
import time
import json
from data_loader import get_clean_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# Создаем папки, если их нет
os.makedirs("../data/processed/", exist_ok=True)

# Словарь для хранения времени векторизации
vectorization_time = {}


def save_vec(name, X_train, X_test, duration):
    """Вспомогательная функция для сохранения векторов и времени"""
    np.save(f"../data/processed/train_{name}_vectors.npy", X_train)
    np.save(f"../data/processed/test_{name}_vectors.npy", X_test)

    # Сохраняем время в словарь
    vectorization_time[name] = round(duration, 2)
    print(f"--- Метод {name} сохранен (Время: {duration:.2f} сек) ---")


def main():
    print("Загрузка данных...")
    data = get_clean_20newsgroups()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    np.save("../data/processed/y_train.npy", y_train)
    np.save("../data/processed/y_test.npy", y_test)

    # --- 1 & 2: Binary и Bag of Words ---
    print("\nВекторизация: Binary & BoW...")
    for name, is_binary in [("binary", True), ("bow", False)]:
        start = time.time()
        vec = CountVectorizer(binary=is_binary, stop_words="english", max_features=5000)
        xt = vec.fit_transform(X_train_raw).toarray()
        xv = vec.transform(X_test_raw).toarray()
        duration = time.time() - start
        save_vec(name, xt, xv, duration)

    # --- 3 & 4: TF-IDF Standard и Bigrams ---
    print("\nВекторизация: TF-IDF...")
    tfidf_params = [("tfidf_std", (1, 1)), ("tfidf_bigrams", (1, 2))]
    for name, ngram in tfidf_params:
        start = time.time()
        vec = TfidfVectorizer(
            ngram_range=ngram, stop_words="english", max_features=5000
        )
        xt = vec.fit_transform(X_train_raw).toarray()
        xv = vec.transform(X_test_raw).toarray()
        duration = time.time() - start
        save_vec(name, xt, xv, duration)

    # --- 5: Word2Vec (Average) ---
    print("\nВекторизация: Word2Vec...")
    start = time.time()
    tokenized_train = [t.lower().split() for t in X_train_raw]
    tokenized_test = [t.lower().split() for t in X_test_raw]
    w2v = Word2Vec(
        sentences=tokenized_train, vector_size=100, window=5, min_count=2, workers=4
    )

    def get_avg_vec(tokens):
        v = [w2v.wv[w] for w in tokens if w in w2v.wv]
        return np.mean(v, axis=0) if v else np.zeros(100)

    xt = np.array([get_avg_vec(t) for t in tokenized_train])
    xv = np.array([get_avg_vec(t) for t in tokenized_test])
    duration = time.time() - start
    save_vec("w2v", xt, xv, duration)

    # --- 6: Doc2Vec ---
    print("\nВекторизация: Doc2Vec...")
    start = time.time()
    tagged = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_train)]
    d2v = Doc2Vec(tagged, vector_size=100, epochs=20)
    xt = np.array([d2v.infer_vector(t) for t in tokenized_train])
    xv = np.array([d2v.infer_vector(t) for t in tokenized_test])
    duration = time.time() - start
    save_vec("d2v", xt, xv, duration)

    # --- 7: BERT ---
    print("\nВекторизация: BERT (это долго)...")
    start = time.time()
    bert = SentenceTransformer("all-MiniLM-L6-v2")
    xt = bert.encode(X_train_raw, show_progress_bar=True)
    xv = bert.encode(X_test_raw, show_progress_bar=True)
    duration = time.time() - start
    save_vec("bert", xt, xv, duration)

    # Сохраняем все замеры времени в файл
    with open("../data/processed/vectorization_time.json", "w") as f:
        json.dump(vectorization_time, f)
    print("\nВсе замеры времени сохранены в vectorization_time.json")


if __name__ == "__main__":
    main()
