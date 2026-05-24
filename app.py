import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Book Analytics Dashboard",
    layout="wide"
)


# =====================================================
# POWER BI STYLE THEME
# =====================================================
st.markdown(
    """
    <style>

    /* Dashboard background */
    .stApp {
        background-color: #f4f6f8;
    }

    /* Section titles */
    h1 {
        font-size: 40px;
        color: #1f2a44;
        font-weight: 700;
        margin-bottom: 10px;
    }

    h2, h3 {
        color: #1f2a44;
        font-weight: 600;
    }

    p, div, label {
        font-size: 18px;
        color: #2b2b2b;
    }

    /* Card container */
    .bi-card {
        background-color: white;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #e1e5ea;
        box-shadow: 0px 1px 4px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #2f5597;
        color: white;
        font-size: 18px;
        border-radius: 6px;
        padding: 10px;
        width: 100%;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #1f3f73;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# TITLE (Power BI style header area)
# =====================================================
st.title("Book Analytics Dashboard")
st.write("Interactive insights into books, genres, and reading trends")


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
def build_content_model(df):

    df = df.dropna(subset=["content", "book_title"]).copy()
    df = df.sample(min(5000, len(df)), random_state=42).reset_index(drop=True)

    tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
    tfidf_matrix = tfidf.fit_transform(df["content"])

    sim = cosine_similarity(tfidf_matrix)

    book_index = df.reset_index().set_index("book_title")["index"].to_dict()

    return sim, book_index, df


content_sim, book_idx, df_cb = build_content_model(books)


# =====================================================
# RECOMMENDATION SYSTEM
# =====================================================
def recommend_books(title, top_n=5):

    if title not in book_idx:
        return pd.DataFrame()

    idx = book_idx[title]

    scores = list(enumerate(content_sim[idx].flatten()))

    scores = sorted(scores, key=lambda x: float(x[1]), reverse=True)[1:top_n + 1]

    indices = [i[0] for i in scores]

    recs = df_cb.iloc[indices][["book_title", "rating", "categories"]].copy()

    recs["similarity"] = [round(float(i[1]), 2) for i in scores]

    recs.columns = ["Book Title", "Rating", "Genre", "Match Score"]

    return recs.reset_index(drop=True)


# =====================================================
# GENRE RECOMMENDATION
# =====================================================
def recommend_by_genre(genre):

    genre_df = books[
        books["categories"].astype(str).str.contains(genre, case=False, na=False)
    ]

    genre_df = (
        genre_df[["book_title", "rating", "categories"]]
        .drop_duplicates()
        .sort_values("rating", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    genre_df.columns = ["Book Title", "Rating", "Genre"]

    return genre_df


# =====================================================
# DASHBOARD LAYOUT (POWER BI STYLE GRID)
# =====================================================

col1, col2 = st.columns([1, 1])

with col1:
    selected_book = st.selectbox(
        "Select Book",
        sorted(df_cb["book_title"].dropna().unique())
    )

with col2:
    selected_genre = st.selectbox(
        "Select Genre",
        sorted(books["categories"].dropna().unique())
    )


st.divider()


# =====================================================
# KPI STYLE CARDS (like Power BI tiles)
# =====================================================
k1, k2, k3 = st.columns(3)

k1.markdown('<div class="bi-card"><h3>Total Books</h3><h2>{}</h2></div>'.format(len(books)), unsafe_allow_html=True)
k2.markdown('<div class="bi-card"><h3>Genres</h3><h2>{}</h2></div>'.format(books["categories"].nunique()), unsafe_allow_html=True)
k3.markdown('<div class="bi-card"><h3>Authors</h3><h2>{}</h2></div>'.format(books["store"].nunique()), unsafe_allow_html=True)


st.divider()


# =====================================================
# RECOMMENDATION + GENRE (CARDS)
# =====================================================
c1, c2 = st.columns(2)

with c1:

    st.subheader("Recommended Books")

    if selected_book:

        recs = recommend_books(selected_book)

        for _, r in recs.iterrows():

            st.markdown(
                f"""
                <div class="bi-card">
                    <h3> {r['Book Title']}</h3>
                    <p> Rating: {r['Rating']}</p>
                    <p> Match Score: {r['Match Score']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


with c2:

    st.subheader("Genre Recommendations")

    if selected_genre:

        recs = recommend_by_genre(selected_genre)

        for _, r in recs.iterrows():

            st.markdown(
                f"""
                <div class="bi-card">
                    <h3> {r['Book Title']}</h3>
                    <p> Rating: {r['Rating']}</p>
                    <p> Genre: {r['Genre']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


st.divider()


# =====================================================
# POWER BI STYLE CHART PANELS
# =====================================================
c1, c2 = st.columns(2)

with c1:

    st.subheader("Top Genres")

    data = books["categories"].value_counts().head(8).reset_index()
    data.columns = ["Genre", "Count"]

    fig = px.bar(
        data,
        x="Genre",
        y="Count",
        color="Genre",
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    st.plotly_chart(fig, use_container_width=True)


with c2:

    st.subheader("Top Authors")

    data = books["store"].value_counts().head(8).reset_index()
    data.columns = ["Author", "Count"]

    fig = px.bar(
        data,
        x="Author",
        y="Count",
        color="Author",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    st.plotly_chart(fig, use_container_width=True)


st.divider()


# =====================================================
# TREND LINE (POWER BI PANEL)
# =====================================================
st.subheader("Reading Trends Over Time")

books_clean = books.dropna(subset=["timestamp", "categories"]).copy()

top5 = books_clean["categories"].value_counts().head(5).index

trend = books_clean[books_clean["categories"].isin(top5)].copy()

trend["month"] = trend["timestamp"].dt.to_period("M").astype(str)

chart = trend.groupby(["month", "categories"]).size().unstack(fill_value=0)

st.line_chart(chart, use_container_width=True)


st.divider()


# =====================================================
# TRENDING TABLE PANEL
# =====================================================
st.subheader("Trending Books")

st.dataframe(
    trending.reset_index(drop=True),
    use_container_width=True
)
