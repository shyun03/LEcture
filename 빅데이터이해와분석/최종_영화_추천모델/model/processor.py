import re  # 정규 표현식 처리용 표준 라이브러리
from konlpy.tag import Okt  # 한국어 형태소 분석을 위한 KoNLPy의 Okt 클래스 임포트

okt = Okt()  # Okt 형태소 분석기 인스턴스 생성

def normalize_text(text, synonym_dict):  # 텍스트 내 단어를 동의어 사전 기준으로 정규화하는 함수
    for word, categories in synonym_dict.items():  # 사전에 등록된 (단어, 대응 카테고리 리스트) 반복
        for cat in categories:  # 각 카테고리(표준어) 반복
            if word in text:  # 텍스트에 동의어 단어가 포함되어 있으면
                text = text.replace(word, cat)  # 카테고리(표준어)로 교체
    return text  # 정규화된 텍스트 반환


def preprocess_input(text, synonym_dict):  # 사용자 입력 텍스트 전처리 함수
    # 1) 동의어 기반 정규화 수행
    text = normalize_text(text, synonym_dict)
    # 2) 한글, 영문, 숫자, 공백만 남기고 기타 특수문자는 제거
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\s]", "", text)
    # 3) 형태소 분석으로 명사, 형용사, 동사만 추출 (어간 처리 포함)
    tokens = okt.pos(text, stem=True)
    # 4) 길이가 2 이상인 단어만 필터링
    words = [w for w, t in tokens if t in ["Noun", "Adjective", "Verb"] and len(w) > 1]
    # 5) 중복 제거 후 정렬하여 공백으로 연결해 최종 전처리 문자열 반환
    return " ".join(sorted(set(words)))
