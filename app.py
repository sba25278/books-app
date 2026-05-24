import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(
    page_title='Book Discovery Platform',
    layout='wide'
)

st.title('Book Discovery Platform')


# ---------------- WHITE THEME ----------------
st.markdown(
    """
    <style>

    .stApp {
        background-color: white;
    }

    h1, h2, h3 {
        color: #2f2f2f;
    }

    div.stButton > button {
        background-color: #d9a066;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 8px;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #b97c4b;
        color: white;
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
    books = pd.read_csv('books_clean.csv')
    trending = pd.read_csv('trending_clean.csv')
    return books, trending


books, trending = load_data()


# =====================================================
# CONTENT MODEL
# =====================================================
@st.cache_data
def build_content_model(df):

    df = df.dropna(subset=['content', 'book_title']).copy()

    df = df.sample(min(5000, len(df)), random_state=42)

    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=2000
    )

    tfidf_matrix = tfidf.fit_transform(df['content'])

    sim = cosine_similarity(tfidf_matrix)

    return sim, df


content_sim, df_cb = build_content_model(books)


# =====================================================
# RECOMMENDATION FUNCTION
# =====================================================
def recommend_books(title, top_n=10):

    matches = df_cb[df_cb['book_title'] == title]

    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]

    scores = list(enumerate(content_sim[idx]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )[1:top_n + 1]

    indices = [i[0] for i in scores]

    recs = df_cb.iloc[indices][
        ['book_title', 'rating', 'categories']
    ].copy()

    recs['similarity'] = [round(i[1], 3) for i in scores]

    recs.columns = [
        'Book Title',
        'Rating',
        'Category',
        'Similarity'
    ]

    recs = recs.reset_index(drop=True)

    return recs


# =====================================================
# GENRE RECOMMENDATION
# =====================================================
def recommend_by_genre(genre):

    genre_df = books[
        books['categories']
        .astype(str)
        .str.contains(genre, case=False, na=False)
    ]

    genre_df = genre_df[
        ['book_title', 'rating', 'categories']
    ].drop_duplicates()

    genre_df.columns = ['Book Title', 'Rating', 'Category']

    return genre_df.sort_values(
        by='Rating',
        ascending=False
    ).head(10).reset_index(drop=True)


# =====================================================
# NAVIGATION
# =====================================================
menu = st.sidebar.radio(
    'Navigation',
    [
        'Insights Dashboard',
        'Book Recommendations',
        'Trending Books'
    ]
)


# =====================================================
# RECOMMENDATIONS PAGE
# =====================================================
if menu == 'Book Recommendations':

    st.header('Find Similar Books')

    book_name = st.selectbox(
        'Select a book',
        sorted(df_cb['book_title'].dropna().unique())
    )

    if book_name:

        results = recommend_books(book_name)

        st.dataframe(results, use_container_width=True, height=400)

    st.divider()

    genre = st.selectbox(
        'Or explore by genre',
        sorted(books['categories'].dropna().unique())
    )

    if genre:

        genre_results = recommend_by_genre(genre)

        st.dataframe(genre_results, use_container_width=True, height=400)


# =====================================================
# TRENDING PAGE
# =====================================================
if menu == 'Trending Books':

    st.header('Top 100 Trending Books')

    st.dataframe(
        trending.reset_index(drop=True),
        use_container_width=True,
        height=600
    )


# =====================================================
# INSIGHTS DASHBOARD
# =====================================================
if menu == 'Insights Dashboard':

    st.header('Book Insights Overview')

    # ---------------- TOP BOOKS ----------------
    st.subheader('Most Popular Books')

    top_books = (
        books['book_title']
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_books.columns = ['Book Title', 'Count']

    st.dataframe(top_books, use_container_width=True, height=350)


    # ---------------- GENRES + AUTHORS ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Top Genres')

        top_genres = (
            books['categories']
            .value_counts()
            .head(10)
        )

        st.bar_chart(top_genres)

    with col2:
        st.subheader('Top Authors')

        top_authors = (
            books['store']
            .value_counts()
            .head(10)
        )

        st.bar_chart(top_authors)


    # ---------------- TIME SERIES ----------------
    st.subheader('Genre Trends Over Time')

    books['timestamp'] = pd.to_datetime(
        books['timestamp'],
        errors='coerce'
    )

    books_clean = books.dropna(
        subset=['timestamp', 'categories']
    ).copy()

    top5_genres = (
        books_clean['categories']
        .value_counts()
        .head(5)
        .index
    )

    trend_df = books_clean[
        books_clean['categories'].isin(top5_genres)
    ].copy()

    # DOUBLE-ENDED SLIDER
    min_date = trend_df['timestamp'].min().date()
    max_date = trend_df['timestamp'].max().date()

    date_range = st.slider(
        "Select time range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    start_date, end_date = date_range

    filtered = trend_df[
        (trend_df['timestamp'].dt.date >= start_date) &
        (trend_df['timestamp'].dt.date <= end_date)
    ].copy()

    filtered['month'] = filtered['timestamp'].dt.to_period('M')

    genre_time = (
        filtered
        .groupby(['month', 'categories'])
        .size()
        .unstack(fill_value=0)
    )

    genre_time.index = genre_time.index.astype(str)

    st.line_chart(genre_time, use_container_width=True)
