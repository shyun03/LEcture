import re
from konlpy.tag import Okt
okt = Okt()

def normalize_text(text, synonym_dict):
    for word, categories in synonym_dict.items():
        for cat in categories:
            if word in text:
                text = text.replace(word, cat)
    return text

def preprocess_input(text, synonym_dict):
    text = normalize_text(text, synonym_dict)
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\s]", "", text)
    tokens = okt.pos(text, stem=True)
    words = [w for w, t in tokens if t in ["Noun", "Adjective", "Verb"] and len(w) > 1]
    return " ".join(sorted(set(words)))
