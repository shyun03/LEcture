from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF 벡터화를 위한 클래스 임포트
from sklearn.metrics.pairwise import cosine_similarity    # 코사인 유사도 계산 함수 임포트
from sentence_transformers import SentenceTransformer       # 문장 임베딩 모델 클래스 임포트
import numpy as np                                        # 넘파이 배열 연산을 위한 모듈 임포트

class Recommender:
    def __init__(self, df):
        self.df = df
        # TF-IDF 벡터라이저 초기화: 최대 문서 비율 0.8, 최소 문서 빈도 2, 1~2그램 사용
        self.vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, ngram_range=(1, 2))
        # 데이터프레임의 '통합텍스트' 컬럼에 fit 및 transform 수행하여 TF-IDF 행렬 생성
        self.tfidf_mat = self.vectorizer.fit_transform(df["통합텍스트"])

        # SentenceTransformer 모델 로드 (all-MiniLM-L6-v2)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # 코퍼스(통합텍스트) 문장 임베딩 계산
        self.corpus_embeds = self.model.encode(
            df["통합텍스트"].tolist(), batch_size=64
        )
        # 감정 키워드(최종키워드) 문장 임베딩 계산
        self.emotion_embeds = self.model.encode(
            df["최종키워드"].tolist(), batch_size=64
        )

    def recommend(self, query_emb, tfidf_query_vec, rating_filters, alpha=0.5, top_n=5):
        # 쿼리 TF-IDF 벡터와 코퍼스 TF-IDF 행렬 간 코사인 유사도 계산
        tf_sims = cosine_similarity(tfidf_query_vec, self.tfidf_mat).flatten()
        # 쿼리 임베딩과 코퍼스 임베딩 간 코사인 유사도 계산 (의미 기반)
        sem_sims = cosine_similarity(query_emb, self.corpus_embeds).flatten()
        # 쿼리 임베딩과 감정 임베딩 간 코사인 유사도 계산 (감정 키워드 기반)
        emo_sims = cosine_similarity(query_emb, self.emotion_embeds).flatten()

        # 최종 유사도 조합: 감정 유사도 70%, TF-IDF와 의미 유사도 조합 30%
        combined = 0.7 * emo_sims + 0.3 * (
            alpha * tf_sims + (1 - alpha) * sem_sims
        )

        # '모두 보기' 옵션이 있으면 모든 행을 True로, 아니면 등급 필터 적용
        if "모두 보기" in rating_filters:
            mask = np.ones(len(self.df), dtype=bool)
        else:
            # 데이터프레임의 '등급' 컬럼 값이 rating_filters 리스트에 있는지 여부로 마스크 생성
            mask = self.df["등급"].isin(rating_filters).to_numpy()

        # 필터된 항목은 유사도 유지, 아니면 -1로 마스킹 처리
        sims_masked = np.where(mask, combined, -1)
        # 유사도가 높은 상위 top_n 개의 인덱스를 내림차순으로 정렬
        idxs = sims_masked.argsort()[-top_n:][::-1]

        # top_n 개의 추천 결과를 데이터프레임 형태로 반환
        return self.df.iloc[idxs].copy()
