import streamlit as st  # Streamlit 라이브러리 임포트
import pandas as pd  # 데이터프레임 처리를 위한 pandas 임포트
from model.loader import load_data, load_synonym_dict  # 데이터 및 동의어 사전 로드 함수 임포트
from model.processor import normalize_text, preprocess_input  # 텍스트 전처리 함수 임포트
from model.recommender import Recommender  # 추천 엔진 클래스 임포트

# 페이지 설정: 타이틀과 레이아웃 지정
st.set_page_config(page_title="🎬 오늘의 당신에게 어울리는 영화를 추천합니다", layout="wide")

# ────────────── CSS 호버 스타일 추가 ──────────────
st.markdown(
    """
    <style>
    .recommend-box {  /* 추천 박스 기본 스타일 */
        padding: 10px;  /* 내부 여백 지정 */
        border-radius: 10px;  /* 둥근 모서리 적용 */
        transition: 0.3s ease;  /* 호버 시 부드러운 전환 */
    }
    .recommend-box:hover {  /* 호버 시 배경 및 그림자 효과 */
        background-color: #e0e7ff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .recommend-box:hover .movie-title {  /* 호버 시 제목 강조 */
        color: #e74c3c !important;
        background-color: rgba(231, 76, 60, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
        transition: 0.3s ease;
    }
    </style>
    """,
    unsafe_allow_html=True  # HTML 렌더링 허용
)

# ────────────── 데이터 로드 및 전처리 ──────────────
df = load_data()  # 영화 데이터 로드
synonym_dict = load_synonym_dict()  # 동의어 사전 로드

# 최종키워드 컬럼 결측값 처리 및 정규화 적용
df["최종키워드"] = df["최종키워드"].fillna("").apply(
    lambda x: normalize_text(x, synonym_dict)  # 단어 정규화 함수 적용
)
# 통합텍스트 컬럼 생성: 제목, 장르, 상영시간 등 결합
df["통합텍스트"] = (
    df["영화제목"] + " " + df["장르"] + " " + df["상영시간"] + " " +
    df["등급"] + " " + df["개봉일"] + " " + df["최종키워드"]
)

@st.cache_resource  # 리소스 캐싱: 추천기 초기화 비용 줄임
def load_recommender(df):
    return Recommender(df)  # Recommender 객체 생성

recommender = load_recommender(df)  # 전역으로 추천기 로드

# ────────────── session_state 초기화 ──────────────
# 각 체크박스 상태를 session_state에 저장하여 유지
for key in ["c_all","c_general","c_12","c_15","c_adult"]:
    if key not in st.session_state:
        st.session_state[key] = True  # 기본값 모두 True

# 선택된 장르 리스트 초기화
if "selected_genres" not in st.session_state:
    st.session_state.selected_genres = []
# 결과 표시 여부 초기화
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# ────────────── 체크박스 변경 콜백 ──────────────
def on_all_change():
    # 전체보기 체크박스 변경 시 개별 등급 모두 동기화
    state = st.session_state.c_all
    st.session_state.c_general = state
    st.session_state.c_12 = state
    st.session_state.c_15 = state
    st.session_state.c_adult = state

def on_specific_change():
    # 개별 등급 변경 시 전체보기 체크박스 상태 업데이트
    all_selected = (
        st.session_state.c_general and st.session_state.c_12 and
        st.session_state.c_15 and st.session_state.c_adult
    )
    st.session_state.c_all = all_selected

def on_genre_change():
    # 멀티셀렉트 변경 시 session_state에 저장
    st.session_state.selected_genres = st.session_state.genre_multiselect

# ────────────── 사이드바 UI ──────────────
st.sidebar.header("🔧 옵션 설정")  # 사이드바 헤더
# 추천 개수 슬라이더
top_n = st.sidebar.slider("추천 개수", 3, 10, 5)
# 가중치 조정 슬라이더
alpha = st.sidebar.slider(
    "📌 추천 기준 설정", 0.0, 1.0, 0.5, 0.1,
    help="오른쪽일수록 키워드 위주, 왼쪽은 감정 표현 위주"
)

# 연령 등급 체크박스
st.sidebar.markdown("**연령 등급 선택**")
st.sidebar.checkbox(
    "모두 보기", value=st.session_state.c_all,
    key="c_all", on_change=on_all_change
)
for label, key in [
    ("전체관람가","c_general"), ("12세이상관람가","c_12"),
    ("15세이상관람가","c_15"), ("청소년 관람불가","c_adult")
]:
    st.sidebar.checkbox(label, value=st.session_state[key], key=key, on_change=on_specific_change)

# 제외할 장르 멀티셀렉트
st.sidebar.markdown("**보고 싶지 않은 장르 제외 (복수 선택 가능)**")
genres_list = ["드라마","코미디","액션","스릴러","공포","SF","판타지",
               "애니메이션","로맨스","다큐멘터리","뮤지컬","가족",
               "어드벤처","범죄","미스터리","전쟁"]
st.sidebar.multiselect(
    "제외할 장르를 선택하세요", genres_list,
    default=st.session_state.selected_genres,
    key="genre_multiselect", on_change=on_genre_change
)

# ────────────── 메인 UI ──────────────
st.title("🎬 오늘의 당신에게 어울리는 영화를 추천합니다")  # 페이지 타이틀

# 사용자 입력 받기: 기분 텍스트
query = st.text_input(
    "기분을 자유롭게 표현해보세요! (예: 오늘 너무 힘들고 지쳤어요)",
    key="input_query"
)
search_btn = st.button("추천 받기")  # 추천 버튼

# 추천 실행 함수 정의
def run_recommend():
    # 입력 공백 체크
    if not st.session_state.input_query.strip():
        st.warning("오늘의 기분을 입력해 주세요!")
        st.session_state.show_results = False
        return

    # 선택된 등급 리스트 생성
    selected_ratings = []
    if st.session_state.c_all:
        selected_ratings = ["전체관람가","12세이상관람가","15세이상관람가","청소년관람불가"]
    else:
        for key, label in [("c_general","전체관람가"),("c_12","12세이상관람가"),
                           ("c_15","15세이상관람가"),("c_adult","청소년관람불가")]:
            if st.session_state[key]: selected_ratings.append(label)

    # 등급 미선택 시 경고
    if not selected_ratings:
        st.warning("최소 하나의 등급을 선택하세요!")
        st.session_state.show_results = False
        return

    # 입력 전처리 및 벡터화
    proc = preprocess_input(st.session_state.input_query, synonym_dict)
    query_vec = recommender.vectorizer.transform([proc])
    query_emb = recommender.model.encode([st.session_state.input_query])

    # 추천 실행
    results = recommender.recommend(query_emb, query_vec, selected_ratings, alpha, top_n)

    # 장르 제외 필터 적용
    results = results[~results["장르"].apply(
        lambda g: any(genre in g for genre in st.session_state.selected_genres)
    )]

    # 추천 결과 없을 때 안내
    if results.empty:
        st.info("조건에 맞는 영화가 없습니다.")
        st.session_state.show_results = False
        return

    # 결과 저장 및 플래그 설정
    st.session_state.results = results
    st.session_state.show_results = True

# 버튼 클릭 시 추천 실행
if search_btn:
    run_recommend()

# 추천 결과 출력
if st.session_state.show_results:
    for idx, row in st.session_state.results.iterrows():
        st.markdown(f"""
        <div class="recommend-box">
            <h3 class="movie-title">{row['영화제목']} ({row['개봉일'][:4]})</h3>
            <p>장르: {row['장르']} | 등급: {row['등급']} | 상영시간: {row['상영시간']}</p>
            <p>설명: {row['최종키워드']}</p>
        </div>
        """, unsafe_allow_html=True)  # HTML 마크업으로 스타일 적용
