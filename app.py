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

st.title("Book Analytics Dashboard")


# =====================================================
# ACCESSIBLE THEME + GRAPH FRAMES (65+ FRIENDLY)
# =====================================================
st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f3ea;
    }

    h1, h2, h3 {
        color: #2f2f2f;
    }

    .book-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #e6d9c8;
        box-shadow: 1px 1px 6px rgba(0,0,0,0.05);
    }

    .graph-frame {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e6d9c8;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    div.stButton > button {
        background-color: #4a6fa5;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #3c5c8c;
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
# HIGH-CONTRAST 65+ COLOUR PALETTE
# (red/orange/brown/black/grey - high readability)
# =====================================================
ACCESSIBLE_COLOURS = [
    "#c94c4c",  
    "#e07a5f",  
    "#f2a65a",  
    "#f4c095",  
    "#c06c84",  
    "#6a994e",  
    "#8e7cc3",  
    "#6f4e37"  
]


# =====================================================
# RECOMMENDATION FUNCTION (NO DUPLICATES)
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
            results.append(i)

        if len(results) == top_n:
            break

    return df_cb.iloc[results][["book_title", "rating", "categories"]].reset_index(drop=True)


# =====================================================
# GENRE RECOMMENDATION
# =====================================================
def recommend_by_genre(genre):

    df = books[
        books["categories"].astype(str).str.contains(genre, case=False, na=False)
    ]

    df = df.drop_duplicates(subset=["book_title"])
    df = df.sort_values("rating", ascending=False).head(5)

    return df[["book_title", "rating", "categories"]].reset_index(drop=True)


# =====================================================
# NAVIGATION
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
# CARD DISPLAY FUNCTION
# =====================================================
def show_cards(df):
    for _, row in df.iterrows():
        st.markdown(
            f"""
            <div class="book-card">
                <b>{row['book_title']}</b><br>
                  Rating: {row['rating']}<br>
                  Genre: {row['categories']}
            </div>
            """,
            unsafe_allow_html=True
        )


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
            show_cards(recommend_books(selected_book))

    with col2:
        st.subheader("Genre Recommendations")
        if selected_genre:
            show_cards(recommend_by_genre(selected_genre))

    st.divider()

    # =================================================
    # TOP GENRES (ACCESSIBLE COLOURS + NO AXIS CLUTTER)
    # =================================================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="graph-frame">', unsafe_allow_html=True)
        st.subheader("Top Genres")

        g = books["categories"].value_counts().head(8).reset_index()
        g.columns = ["Genre", "Count"]

        fig = px.bar(
            g,
            x="Genre",
            y="Count",
            color="Genre",
            color_discrete_sequence=ACCESSIBLE_COLOURS
        )

        fig.update_layout(
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=30, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="graph-frame">', unsafe_allow_html=True)
        st.subheader("Top Authors")

        a = books["store"].value_counts().head(8).reset_index()
        a.columns = ["Author", "Count"]

        fig = px.bar(
            a,
            x="Author",
            y="Count",
            color="Author",
            color_discrete_sequence=ACCESSIBLE_COLOURS
        )

        fig.update_layout(
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=30, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # =================================================
    # TREND OVER TIME + SLIDER (NEW)
    # =================================================
    st.markdown('<div class="graph-frame">', unsafe_allow_html=True)
    st.subheader("Reading Trends Over Time")

    clean = books.dropna(subset=["timestamp", "categories"]).copy()

    top5 = clean["categories"].value_counts().head(5).index
    trend = clean[clean["categories"].isin(top5)].copy()

    trend["month"] = trend["timestamp"].dt.to_period("M").astype(str)

    min_date = clean["timestamp"].min().date()
    max_date = clean["timestamp"].max().date()

    date_range = st.slider(
        "Select time period",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    start_date, end_date = date_range

    trend = trend[
        (trend["timestamp"].dt.date >= start_date) &
        (trend["timestamp"].dt.date <= end_date)
    ]

    chart = trend.groupby(["month", "categories"]).size().unstack(fill_value=0)

    fig = px.line(chart, markers=True, color_discrete_sequence=ACCESSIBLE_COLOURS)

    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================
# TRENDING PAGE
# =====================================================
if st.session_state.page == "trending":

    st.header("Top 100 Trending Books")

    st.dataframe(
        trending.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        height=900
    )
