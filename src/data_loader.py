from sklearn.datasets import fetch_20newsgroups
def get_clean_20newsgroups():
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    # Здесь можно добавить что то
    return data

def get_methods() -> list:
    '''Возвращает список методов, которые используются'''
    return ["binary", "bow", "tfidf_std", "tfidf_bigrams", "w2v","d2v","bert"]