import streamlit as st
from model.loader import load_data, load_synonym_dict
from model.processor import normalize_text, preprocess_input
from model.recommender import Recommender

st.set_page_config(page_title="🎬 오늘의 당신에게 어울리는 영화를 추천합니다", layout="wide")

# ────────────── 데이터 및 사전 로딩 ──────────────
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

# ────────────── 사이드바 옵션 구성 ──────────────
st.sidebar.header("🔧 옵션 설정")
top_n = st.sidebar.slider("추천 개수", 3, 10, 5)

alpha = st.sidebar.slider(
    "📌 추천 기준 설정", 0.0, 1.0, 0.5, 0.1,
    help="슬라이더를 오른쪽으로 움직이면 키워드(단어) 위주, 왼쪽은 감정 표현(문장) 위주 추천입니다."
)

st.sidebar.markdown("**연령 등급 선택**")

def on_all_change():
    if st.session_state.c_all:
        st.session_state.c_general = False
        st.session_state.c_12 = False
        st.session_state.c_15 = False
        st.session_state.c_adult = False

def on_specific_change():
    if (st.session_state.c_general or st.session_state.c_12 or
        st.session_state.c_15 or st.session_state.c_adult):
        st.session_state.c_all = False

c_all     = st.sidebar.checkbox("모두 보기",        value=True,  key="c_all",     on_change=on_all_change)
c_general = st.sidebar.checkbox("전체관람가",       value=False, key="c_general", on_change=on_specific_change)
c_12      = st.sidebar.checkbox("12세이상관람가",   value=False, key="c_12",      on_change=on_specific_change)
c_15      = st.sidebar.checkbox("15세이상관람가",   value=False, key="c_15",      on_change=on_specific_change)
c_adult   = st.sidebar.checkbox("청소년 관람불가",  value=False, key="c_adult",   on_change=on_specific_change)

selected = []
if c_all:
    selected = ["모두 보기"]
else:
    if c_general: selected.append("전체관람가")
    if c_12:      selected.append("12세이상관람가")
    if c_15:      selected.append("15세이상관람가")
    if c_adult:   selected.append("청소년관람불가")

# ────────────── 메인 인터페이스 ──────────────
st.title("🎬 오늘의 당신에게 어울리는 영화를 추천합니다")

query = st.text_input("기분을 자유롭게 표현해보세요! (예: 오늘 너무 힘들고 지쳤어요)")
search_btn = st.button("추천 받기")

if search_btn:
    if not query.strip():
        st.warning("오늘의 기분을 입력해 주세요!")
    elif not selected:
        st.warning("최소 하나의 등급을 선택하세요!")
    else:
        proc = preprocess_input(query, synonym_dict)
        query_vec = recommender.vectorizer.transform([proc])
        query_emb = recommender.model.encode([query])

        results = recommender.recommend(query_emb, query_vec, selected, alpha, top_n)

        if results.empty:
            st.info("조건에 맞는 영화가 없습니다.")
        else:
            for _, row in results.iterrows():
                st.subheader(f"{row['영화제목']}  ({row['장르']} / {row['등급']})")
                st.write(f"- 개봉일: {row['개봉일']}  |  상영시간: {row['상영시간']}")
                st.write(f"- 키워드: {row['최종키워드']}")
                st.write(f"> {row['전체리뷰'][:150]}…")
                st.markdown("---")
