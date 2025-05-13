from typing import List, Dict, Any, Union
from string import punctuation


def load_stop_words(path: str) -> Union[List[str], None]:
    if path is None:
        return None

    with open(path, 'r') as fp:
        raw = fp.readlines()

    return [x.lower().strip() for x in raw]


def remove_stop_words(order: Dict[str, Any], stop_words: Union[List[str], None]) -> None:
    if stop_words is None:
        return

    for field in order:
        if isinstance(order[field], str):
            raw = order[field].split()
            filtered = []
            for word in raw:
                if word.lower() not in stop_words:
                    filtered.append(word)
            order[field] = " ".join(filtered)

def convert_to_clean_ngrams_set(string: str, stopwords: list = None):
    ngrams = set([s.lower().translate(str.maketrans('', '', punctuation)) for s in string.split(" ")])
    if stopwords is not None:
        clean_ngrams = []
        for ngram in ngrams:
            if ngram not in stopwords:
                clean_ngrams.append(ngram)
        ngrams = set(clean_ngrams)
    return ngrams
