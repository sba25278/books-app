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

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f8f4ee;
    }

    h1, h2, h3 {
        color: #5b4636;
    }

    div.stButton > button {
        background-color: #c9895b;
        color: white;
        font-size: 20px;
        border-radius: 10px;
        padding: 10px;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #a96f45;
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

    book_index = pd.Series(
        df.index,
        index=df['book_title']
    ).drop_duplicates()

    return sim, book_index, df


content_sim, book_idx, df_cb = build_content_model(books)


# =====================================================
# BOOK RECOMMENDATION FUNCTION
# =====================================================
def recommend_books(title, top_n=10):

    if title not in book_idx:
        return pd.DataFrame()

    idx = book_idx[title]

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

    recs['similarity'] = [i[1] for i in scores]

    return recs.drop_duplicates(subset='book_title')


# =====================================================
# GENRE RECOMMENDATION FUNCTION
# =====================================================
def recommend_by_genre(genre):

    genre_df = books[
        books['categories']
        .astype(str)
        .str.contains(
            genre,
            case=False,
            na=False
        )
    ]

    genre_df = (
        genre_df[['book_title', 'rating', 'categories']]
        .drop_duplicates()
        .sort_values(by='rating', ascending=False)
        .head(10)
    )

    return genre_df


# =====================================================
# BIG NAVIGATION BUTTONS
# =====================================================
col1, col2 = st.columns(2)

with col1:
    home_btn = st.button('Home Dashboard')

with col2:
    trending_btn = st.button('Top 100 Trending Books')


# default page
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if home_btn:
    st.session_state.page = 'home'

if trending_btn:
    st.session_state.page = 'trending'


# =====================================================
# HOME PAGE
# =====================================================
if st.session_state.page == 'home':

    st.header('Book Recommendations and Insights')

    st.write(
        'Search for books, explore genres, and discover reading trends.'
    )

    st.divider()

    # =====================================================
    # RECOMMENDATION BAR
    # =====================================================
    st.subheader('Find Similar Books')

    col1, col2 = st.columns(2)

    # ---------------- BOOK DROPDOWN ----------------
    with col1:

        selected_book = st.selectbox(
            'Choose a Book',
            sorted(df_cb['book_title'].dropna().unique())
        )

        if selected_book:

            recommendations = recommend_books(selected_book)

            st.write('Recommended Books')

            st.dataframe(
                recommendations,
                use_container_width=True,
                height=350
            )

    # ---------------- GENRE DROPDOWN ----------------
    with col2:

        genres = (
            books['categories']
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
        )

        selected_genre = st.selectbox(
            'Choose a Genre',
            genres
        )

        if selected_genre:

            genre_recs = recommend_by_genre(selected_genre)

            st.write('Popular Books in this Genre')

            st.dataframe(
                genre_recs,
                use_container_width=True,
                height=350
            )

    st.divider()

    # =====================================================
    # GENRES + AUTHORS SIDE BY SIDE
    # =====================================================
    col1, col2 = st.columns(2)

    # ---------------- TOP GENRES ----------------
    with col1:

        st.subheader('Most Popular Genres')

        top_categories = (
            books['categories']
            .value_counts()
            .head(10)
        )

        st.bar_chart(top_categories)

    # ---------------- TOP AUTHORS ----------------
    with col2:

        st.subheader('Most Popular Authors')

        top_authors = (
            books['store']
            .value_counts()
            .head(10)
        )

        st.bar_chart(top_authors)

    st.divider()

    # =====================================================
    # BOOKS OVER TIME
    # =====================================================
    st.subheader('Popular Genres Over Time')

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
    ]

    trend_df['month'] = (
        trend_df['timestamp']
        .dt.to_period('M')
        .astype(str)
    )

    genre_time = (
        trend_df
        .groupby(['month', 'categories'])
        .size()
        .unstack(fill_value=0)
    )

    st.line_chart(genre_time)


# =====================================================
# TRENDING BOOKS PAGE
# =====================================================
if st.session_state.page == 'trending':

    st.header('Top 100 Trending Books')

    st.write(
        'Explore the most popular and trending books.'
    )

    st.dataframe(
        trending,
        use_container_width=True,
        height=700
    )
