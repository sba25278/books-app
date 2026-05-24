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
    page_title='Book Discovery Platform',
    layout='wide'
)

st.title('Book Discovery Platform')


# ---------------- WARM CREAM THEME ----------------
st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f1e6;  /* softer cream */
    }

    h1, h2, h3 {
        color: #5b4636;
    }

    p, div {
        font-size: 18px;
        color: #3d2b1f;
    }

    div.stButton > button {
        background-color: #c9895b;
        color: white;
        font-size: 22px;
        border-radius: 12px;
        padding: 14px;
        width: 100%;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #a96f45;
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

    books['timestamp'] = pd.to_datetime(books['timestamp'], errors='coerce')

    return books, trending


books, trending = load_data()


# =====================================================
# CONTENT MODEL
# =====================================================
@st.cache_data
def build_content_model(df):

    df = df.dropna(subset=['content', 'book_title']).copy()

    df = df.sample(min(5000, len(df)), random_state=42).reset_index(drop=True)

    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    tfidf_matrix = tfidf.fit_transform(df['content'])

    sim = cosine_similarity(tfidf_matrix)

    book_index = pd.Series(range(len(df)), index=df['book_title']).drop_duplicates()

    return sim, book_index, df


content_sim, book_idx, df_cb = build_content_model(books)


# =====================================================
# RECOMMENDATION FUNCTION
# =====================================================
def recommend_books(title, top_n=10):

    if title not in book_idx:
        return pd.DataFrame()

    idx = book_idx[title]

    scores = list(enumerate(content_sim[idx]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    indices = [i[0] for i in scores]

    recs = df_cb.iloc[indices][['book_title', 'rating', 'categories']].copy()

    recs['similarity'] = [round(i[1], 2) for i in scores]

    recs.columns = ['Book Title', 'Reader Rating', 'Genre', 'Similarity Score']

    return recs.drop_duplicates(subset='Book Title').reset_index(drop=True)


# =====================================================
# GENRE RECOMMENDATION
# =====================================================
def recommend_by_genre(genre):

    genre_df = books[
        books['categories'].astype(str).str.contains(genre, case=False, na=False)
    ]

    genre_df = (
        genre_df[['book_title', 'rating', 'categories']]
        .drop_duplicates()
        .sort_values(by='rating', ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    genre_df.columns = ['Book Title', 'Reader Rating', 'Genre']

    return genre_df


# =====================================================
# NAVIGATION BUTTONS
# =====================================================
col1, col2 = st.columns(2)

with col1:
    home_btn = st.button('Home Dashboard')

with col2:
    trending_btn = st.button('Top 100 Trending Books')


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

    st.header('Book Recommendations and Reading Insights')

    st.divider()

    # =================================================
    # RECOMMENDATION SECTION
    # =================================================
    st.subheader('Find Your Next Book')

    col1, col2 = st.columns(2)

    # BOOK RECS
    with col1:

        selected_book = st.selectbox(
            'Choose a Book',
            sorted(df_cb['book_title'].dropna().unique())
        )

        if selected_book:
            recommendations = recommend_books(selected_book)

            st.dataframe(recommendations, use_container_width=True, height=350)

    # GENRE RECS
    with col2:

        selected_genre = st.selectbox(
            'Choose a Genre',
            sorted(books['categories'].dropna().unique())
        )

        if selected_genre:
            genre_recs = recommend_by_genre(selected_genre)

            st.dataframe(genre_recs, use_container_width=True, height=350)


    st.divider()

    # =================================================
    # WARM COLOUR BAR CHARTS
    # =================================================
    col1, col2 = st.columns(2)

    with col1:

        st.subheader('Most Popular Genres')

        top_categories = books['categories'].value_counts().head(10).reset_index()
        top_categories.columns = ['Genre', 'Count']

        fig = px.bar(
            top_categories,
            x='Genre',
            y='Count',
            color='Genre',
            color_discrete_sequence=px.colors.sequential.Oranges
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader('Most Popular Authors')

        top_authors = books['store'].value_counts().head(10).reset_index()
        top_authors.columns = ['Author', 'Count']

        fig = px.bar(
            top_authors,
            x='Author',
            y='Count',
            color='Author',
            color_discrete_sequence=px.colors.sequential.Burg
        )

        st.plotly_chart(fig, use_container_width=True)


    st.divider()

    # =================================================
    # DOUBLE-ENDED SLIDER TIME SERIES
    # =================================================
    st.subheader('Popular Genres Over Time')

    books_clean = books.dropna(subset=['timestamp', 'categories']).copy()

    top5_genres = books_clean['categories'].value_counts().head(5).index

    trend_df = books_clean[books_clean['categories'].isin(top5_genres)].copy()

    min_date = trend_df['timestamp'].min().date()
    max_date = trend_df['timestamp'].max().date()

    date_range = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    start_date, end_date = date_range

    filtered = trend_df[
        (trend_df['timestamp'].dt.date >= start_date) &
        (trend_df['timestamp'].dt.date <= end_date)
    ]

    filtered['month'] = filtered['timestamp'].dt.to_period('M').astype(str)

    genre_time = (
        filtered
        .groupby(['month', 'categories'])
        .size()
        .unstack(fill_value=0)
    )

    st.line_chart(genre_time, use_container_width=True)


# =====================================================
# TRENDING PAGE
# =====================================================
if st.session_state.page == 'trending':

    st.header('Top 100 Trending Books')

    st.dataframe(
        trending.reset_index(drop=True),
        use_container_width=True,
        height=700
    )
