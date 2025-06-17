import streamlit as st
import pandas as pd
from model.loader import load_data, load_synonym_dict
from model.processor import normalize_text, preprocess_input
from model.recommender import Recommender

st.set_page_config(page_title="🎬 오늘의 당신에게 어울리는 영화를 추천합니다", layout="wide")

# ────────────── CSS 호버 스타일 추가 ──────────────
st.markdown(
    """
    <style>
    .recommend-box {
        padding: 10px;
        border-radius: 10px;
        transition: 0.3s ease;
    }
    .recommend-box:hover {
        background-color: #e0e7ff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .recommend-box:hover .movie-title {
        color: #e74c3c !important;
        background-color: rgba(231, 76, 60, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
        transition: 0.3s ease;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ────────────── 데이터 로드 및 전처리 ──────────────
df = load_data()
synonym_dict = load_synonym_dict()

df["최종키워드"] = df["최종키워드"].fillna("").apply(lambda x: normalize_text(x, synonym_dict))
df["통합텍스트"] = (
    df["영화제목"] + " " + df["장르"] + " " + df["상영시간"] + " " +
    df["등급"] + " " + df["개봉일"] + " " + df["최종키워드"]
)

@st.cache_resource
def load_recommender(df):
    return Recommender(df)

recommender = load_recommender(df)

# ────────────── session_state 초기화 ──────────────
if "c_all" not in st.session_state:
    st.session_state.c_all = True
if "c_general" not in st.session_state:
    st.session_state.c_general = True
if "c_12" not in st.session_state:
    st.session_state.c_12 = True
if "c_15" not in st.session_state:
    st.session_state.c_15 = True
if "c_adult" not in st.session_state:
    st.session_state.c_adult = True

if "selected_genres" not in st.session_state:
    st.session_state.selected_genres = []

if "show_results" not in st.session_state:
    st.session_state.show_results = False

# ────────────── 체크박스 변경 콜백 ──────────────
def on_all_change():
    if st.session_state.c_all:
        st.session_state.c_general = True
        st.session_state.c_12 = True
        st.session_state.c_15 = True
        st.session_state.c_adult = True
    else:
        st.session_state.c_general = False
        st.session_state.c_12 = False
        st.session_state.c_15 = False
        st.session_state.c_adult = False

def on_specific_change():
    if (st.session_state.c_general and st.session_state.c_12 and
        st.session_state.c_15 and st.session_state.c_adult):
        st.session_state.c_all = True
    else:
        st.session_state.c_all = False

def on_genre_change():
    st.session_state.selected_genres = st.session_state.genre_multiselect

# ────────────── 사이드바 UI ──────────────
st.sidebar.header("🔧 옵션 설정")

top_n = st.sidebar.slider("추천 개수", 3, 10, 5)
alpha = st.sidebar.slider("📌 추천 기준 설정", 0.0, 1.0, 0.5, 0.1, help="오른쪽일수록 키워드 위주, 왼쪽은 감정 표현 위주")

st.sidebar.markdown("**연령 등급 선택**")
st.sidebar.checkbox("모두 보기", value=st.session_state.c_all, key="c_all", on_change=on_all_change)
st.sidebar.checkbox("전체관람가", value=st.session_state.c_general, key="c_general", on_change=on_specific_change)
st.sidebar.checkbox("12세이상관람가", value=st.session_state.c_12, key="c_12", on_change=on_specific_change)
st.sidebar.checkbox("15세이상관람가", value=st.session_state.c_15, key="c_15", on_change=on_specific_change)
st.sidebar.checkbox("청소년 관람불가", value=st.session_state.c_adult, key="c_adult", on_change=on_specific_change)

st.sidebar.markdown("**보고 싶지 않은 장르 제외 (복수 선택 가능)**")
genres_list = [
    "드라마", "코미디", "액션", "스릴러", "공포", "SF", "판타지", "애니메이션",
    "로맨스", "다큐멘터리", "뮤지컬", "가족", "어드벤처", "범죄", "미스터리", "전쟁"
]
st.sidebar.multiselect(
    "제외할 장르를 선택하세요",
    genres_list,
    default=st.session_state.selected_genres,
    key="genre_multiselect",
    on_change=on_genre_change
)

# ────────────── 메인 UI ──────────────
st.title("🎬 오늘의 당신에게 어울리는 영화를 추천합니다")

query = st.text_input("기분을 자유롭게 표현해보세요! (예: 오늘 너무 힘들고 지쳤어요)", key="input_query")
search_btn = st.button("추천 받기")


def run_recommend():
    if not st.session_state.input_query.strip():
        st.warning("오늘의 기분을 입력해 주세요!")
        st.session_state.show_results = False
        return

    selected_ratings = []
    if st.session_state.c_all:
        selected_ratings = ["전체관람가", "12세이상관람가", "15세이상관람가", "청소년관람불가"]
    else:
        if st.session_state.c_general: selected_ratings.append("전체관람가")
        if st.session_state.c_12: selected_ratings.append("12세이상관람가")
        if st.session_state.c_15: selected_ratings.append("15세이상관람가")
        if st.session_state.c_adult: selected_ratings.append("청소년관람불가")

    if not selected_ratings:
        st.warning("최소 하나의 등급을 선택하세요!")
        st.session_state.show_results = False
        return


    proc = preprocess_input(st.session_state.input_query, synonym_dict)
    query_vec = recommender.vectorizer.transform([proc])
    query_emb = recommender.model.encode([st.session_state.input_query])

    results = recommender.recommend(query_emb, query_vec, selected_ratings, alpha, top_n)

    # 장르 필터링: 선택된 장르 중 하나라도 포함된 경우만
    results = results[~results["장르"].apply(lambda g: any(genre in g for genre in st.session_state.selected_genres))]


    if results.empty:
        st.info("조건에 맞는 영화가 없습니다.")
        st.session_state.show_results = False
        return

    st.session_state.results = results
    st.session_state.show_results = True

if search_btn:
    run_recommend()

if st.session_state.show_results:
    results = st.session_state.results
    for idx, row in results.iterrows():
        st.markdown(f"""
        <div class="recommend-box">
            <h3 class="movie-title">{row['영화제목']} ({row['개봉일'][:4]})</h3>
            <p>장르: {row['장르']} | 등급: {row['등급']} | 상영시간: {row['상영시간']}</p>
            <p>설명: {row['최종키워드']}</p>
        </div>
        """, unsafe_allow_html=True)

