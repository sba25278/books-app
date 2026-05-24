import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(
    page_title="Book Dashboard",
    layout="wide"
)

st.title("📚 Book Analytics Dashboard")


# =====================================================
# SIMPLE BACKGROUND STYLE (LIGHT CREAM)
# =====================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f3ea;
    }

    h1, h2, h3 {
        color: #4a3b2a;
    }

    div.stButton > button {
        background-color: #b8744f;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #9c5f3f;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    books = pd.read_csv("books_clean.csv")
    trending = pd.read_csv("trending_clean.csv")

    books["timestamp"] = pd.to_datetime(books["timestamp"], errors="coerce")

    return books, trending


books, trending = load_data()


# =====================================================
# CONTENT MODEL
# =====================================================
@st.cache_data
def build_model(df):

    df = df.dropna(subset=["content", "book_title"]).copy()
    df = df.sample(min(5000, len(df)), random_state=42).reset_index(drop=True)

    tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
    matrix = tfidf.fit_transform(df["content"])

    sim = cosine_similarity(matrix)

    book_index = pd.Series(
        range(len(df)),
        index=df["book_title"]
    ).to_dict()

    return sim, book_index, df


content_sim, book_idx, df_cb = build_model(books)


# =====================================================
# FIXED RECOMMENDATION (NO DUPLICATES)
# =====================================================
def recommend_books(title, top_n=5):

    if title not in book_idx:
        return pd.DataFrame()

    idx = book_idx[title]

    scores = list(enumerate(content_sim[idx]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    seen = set()
    results = []

    for i, score in scores:
        book = df_cb.iloc[i]["book_title"]

        if book != title and book not in seen:
            seen.add(book)
            results.append((i, score))

        if len(results) == top_n:
            break

    rows = [r[0] for r in results]
    scores = [r[1] for r in results]

    recs = df_cb.iloc[rows][["book_title", "rating", "categories"]].copy()
    recs["similarity"] = [round(float(s), 2) for s in scores]

    recs.columns = ["Book Title", "Rating", "Genre", "Match Score"]

    return recs.reset_index(drop=True)


# =====================================================
# GENRE RECOMMENDATION (NO DUPLICATES)
# =====================================================
def recommend_by_genre(genre):

    df = books[
        books["categories"].astype(str).str.contains(genre, case=False, na=False)
    ]

    df = df.drop_duplicates(subset=["book_title"])

    df = df[["book_title", "rating", "categories"]]
    df = df.sort_values("rating", ascending=False).head(5).reset_index(drop=True)

    df.columns = ["Book Title", "Rating", "Genre"]

    return df


# =====================================================
# NAVIGATION (RESTORED BUTTON)
# =====================================================
col1, col2 = st.columns(2)

with col1:
    home_btn = st.button("Home")

with col2:
    trending_btn = st.button("Top 100 Trending Books")

if "page" not in st.session_state:
    st.session_state.page = "home"

if home_btn:
    st.session_state.page = "home"

if trending_btn:
    st.session_state.page = "trending"


# =====================================================
# HOME PAGE
# =====================================================
if st.session_state.page == "home":

    st.header("Overview")

    st.divider()

    # ---------------- INPUTS ----------------
    col1, col2 = st.columns(2)

    with col1:
        selected_book = st.selectbox(
            "Select a Book",
            sorted(df_cb["book_title"].dropna().unique())
        )

    with col2:
        selected_genre = st.selectbox(
            "Select a Genre",
            sorted(books["categories"].dropna().unique())
        )

    # ---------------- RECOMMENDATIONS ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Book Recommendations")

        if selected_book:
            recs = recommend_books(selected_book)
            st.dataframe(recs, use_container_width=True)

    with col2:
        st.subheader("Genre Recommendations")

        if selected_genre:
            recs = recommend_by_genre(selected_genre)
            st.dataframe(recs, use_container_width=True)

    st.divider()

    # ---------------- GRAPHS ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Genres")

        g = books["categories"].value_counts().head(8).reset_index()
        g.columns = ["Genre", "Count"]

        fig = px.bar(
            g,
            x="Genre",
            y="Count",
            color="Genre",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Authors")

        a = books["store"].value_counts().head(8).reset_index()
        a.columns = ["Author", "Count"]

        fig = px.bar(
            a,
            x="Author",
            y="Count",
            color="Author",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------- TREND OVER TIME ----------------
    st.subheader("Reading Trends Over Time")

    clean = books.dropna(subset=["timestamp", "categories"]).copy()

    top5 = clean["categories"].value_counts().head(5).index

    trend = clean[clean["categories"].isin(top5)].copy()

    trend["month"] = trend["timestamp"].dt.to_period("M").astype(str)

    chart = trend.groupby(["month", "categories"]).size().unstack(fill_value=0)

    st.line_chart(chart, use_container_width=True)


# =====================================================
# TRENDING PAGE
# =====================================================
if st.session_state.page == "trending":

    st.header("Top 100 Trending Books")

    st.dataframe(
        trending.reset_index(drop=True),
        use_container_width=True
    )
