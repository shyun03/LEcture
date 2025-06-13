from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

class Recommender:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, ngram_range=(1, 2))
        self.tfidf_mat = self.vectorizer.fit_transform(df["통합텍스트"])

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.corpus_embeds = self.model.encode(df["통합텍스트"].tolist(), batch_size=64)
        self.emotion_embeds = self.model.encode(df["최종키워드"].tolist(), batch_size=64)

    def recommend(self, query_emb, tfidf_query_vec, rating_filters, alpha=0.5, top_n=5):
        tf_sims = cosine_similarity(tfidf_query_vec, self.tfidf_mat).flatten()
        sem_sims = cosine_similarity(query_emb, self.corpus_embeds).flatten()
        emo_sims = cosine_similarity(query_emb, self.emotion_embeds).flatten()

        combined = 0.7 * emo_sims + 0.3 * (alpha * tf_sims + (1 - alpha) * sem_sims)

        if "모두 보기" in rating_filters:
            mask = np.ones(len(self.df), dtype=bool)
        else:
            mask = self.df["등급"].isin(rating_filters).to_numpy()

        sims_masked = np.where(mask, combined, -1)
        idxs = sims_masked.argsort()[-top_n:][::-1]

        return self.df.iloc[idxs].copy()
