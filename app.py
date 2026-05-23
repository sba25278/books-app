import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# PAGE SETUP (SENIOR FRIENDLY)
# -----------------------------
st.set_page_config(
    page_title='Book Discovery Platform',
    layout='wide'
)

# warm background styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fbf6ef;
        font-size: 20px;
    }

    h1, h2, h3 {
        color: #5a3e2b;
    }

    div.stButton > button {
        background-color: #c97b63;
        color: white;
        font-size: 20px;
        padding: 12px;
        border-radius: 12px;
    }

    div.stButton > button:hover {
        background-color: #a65c45;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Book Discovery Platform')
st.write('A simple way to explore books, recommendations, and trends')


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    books = pd.read_csv('books_clean.csv')
    trending = pd.read_csv('trending_clean.csv')
    return books, trending


books, trending = load_data()


# -----------------------------
# CONTENT MODEL
# -----------------------------
@st.cache_data
def build_content_model(df):

    df = df.dropna(subset=['content', 'book_title']).copy()
    df = df.sample(min(5000, len(df)), random_state=42)

    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    tfidf_matrix = tfidf.fit_transform(df['content'])

    sim = cosine_similarity(tfidf_matrix)

    book_index = pd.Series(df.index, index=df['book_title']).drop_duplicates()

    return sim, book_index, df


content_sim, book_idx, df_cb = build_content_model(books)


# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------
def recommend_books(title, top_n=10):

    if title not in book_idx:
        return pd.DataFrame()

    idx = book_idx[title]

    scores = list(enumerate(content_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    indices = [i[0] for i in scores]

    recs = df_cb.iloc[indices][['book_title', 'rating']].copy()
    recs['similarity'] = [i[1] for i in scores]

    return recs.drop_duplicates(subset='book_title')


# -----------------------------
# SIMPLE NAVIGATION (BIG BUTTONS)
# -----------------------------
st.markdown("## Choose an option")

col1, col2, col3 = st.columns(3)

with col1:
    rec_btn = st.button("📚 Find Similar Books")

with col2:
    trend_btn = st.button("📈 Trending Books")

with col3:
    insight_btn = st.button("📊 Book Insights")


menu = None
if rec_btn:
    menu = "rec"
elif trend_btn:
    menu = "trend"
elif insight_btn:
    menu = "insight"


# -----------------------------
# 1. RECOMMENDATIONS
# -----------------------------
if menu == "rec":

    st.header("Find Similar Books")

    st.write("Type a book you enjoy and we will find similar ones.")

    book_name = st.text_input("Enter book title")

    if book_name:
        results = recommend_books(book_name)

        if results.empty:
            st.warning("Book not found. Try another title.")
        else:
            st.dataframe(results, use_container_width=True)


# -----------------------------
# 2. TRENDING BOOKS
# -----------------------------
if menu == "trend":

    st.header("Most Popular Books")

    st.write("These are the most read books in the collection.")

    st.dataframe(trending, use_container_width=True)


if menu == 'Insights Dashboard':

    st.header('Book Insights Overview')

    st.markdown(
        """
        <div style='font-size:20px; line-height:1.6; color:#5a3e2b;'>
        This section shows which books, authors, and genres are most popular,
        and how reading trends change over time.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # =====================================================
    # TOP BOOKS LIST
    # =====================================================
    st.subheader('Most Popular Books')

    top_books = books['book_title'].value_counts().head(10)

    top_books_df = top_books.reset_index()
    top_books_df.columns = ['Book Title', 'Reader Count']

    st.dataframe(
        top_books_df,
        use_container_width=True,
        height=400
    )

    st.write("")


    # =====================================================
    # GENRES + AUTHORS
    # =====================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Top Genres / Categories')

        top_categories = (
            books['categories']
            .value_counts()
            .head(10)
        )

        st.bar_chart(top_categories)

    with col2:
        st.subheader('Top Authors')

        top_authors = (
            books['store']
            .value_counts()
            .head(10)
        )

        st.bar_chart(top_authors)


    st.write("")


    # =====================================================
    # TIME SERIES TREND
    # =====================================================
    st.subheader('Top Genre Engagement Over Time')

    st.caption(
        "This shows how the most popular genres change over time."
    )

    books['timestamp'] = pd.to_datetime(books['timestamp'], errors='coerce')
    books_clean = books.dropna(subset=['timestamp', 'categories']).copy()

    top5_genres = books_clean['categories'].value_counts().head(5).index

    trend_df = books_clean[books_clean['categories'].isin(top5_genres)]

    trend_df['month'] = trend_df['timestamp'].dt.to_period('M').astype(str)

    genre_time = (
        trend_df
        .groupby(['month', 'categories'])
        .size()
        .unstack(fill_value=0)
    )

    st.line_chart(genre_time)
