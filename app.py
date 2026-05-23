import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title='Book Discovery Platform',
    layout='wide'
)

st.title('Book Discovery Platform')


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
# CONTENT MODEL (LIGHTWEIGHT)
# -----------------------------
@st.cache_data
def build_content_model(df):

    df = df.dropna(subset=['content', 'book_title']).copy()

    # reduce size for stability
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
# SIDEBAR NAVIGATION
# -----------------------------
menu = st.sidebar.radio(
    'Navigation',
    [   'Insights Dashboard',
        'Book Recommendations',
        'Trending Books'
    ]
)


# -----------------------------
# 1. RECOMMENDATIONS
# -----------------------------
if menu == 'Book Recommendations':

    st.header('Find Similar Books')

    book_name = st.text_input('Enter a book title')

    if book_name:

        results = recommend_books(book_name)

        if results.empty:
            st.warning('Book not found in dataset')
        else:
            st.dataframe(results, use_container_width=True)


# -----------------------------
# 2. TRENDING BOOKS
# -----------------------------
if menu == 'Trending Books':

    st.header('Top 100 Trending Books')

    st.dataframe(trending, use_container_width=True)


# -----------------------------
# 3. INSIGHTS DASHBOARD (STREAMLIT ONLY)
# -----------------------------
if menu == 'Insights Dashboard':

    st.header('Book Insights Overview')

    # =====================================================
    # TOP BOOKS LIST
    # =====================================================
    st.subheader('Most Popular Books')

    top_books = books['book_title'].value_counts().head(10)

    top_books_df = top_books.reset_index()
    top_books_df.columns = ['Book Title', 'Reader Count']

    st.dataframe(top_books_df, use_container_width=True)


    # =====================================================
    # SORTED GENRE + AUTHOR CHARTS
    # =====================================================
    col1, col2 = st.columns(2)

    # ---------------- CATEGORIES / GENRES ----------------
    cats = books[['categories']].dropna().copy()

    cats['categories'] = cats['categories'].astype(str)

    cats['categories'] = cats['categories'].str.replace('[', '', regex=False)
    cats['categories'] = cats['categories'].str.replace(']', '', regex=False)
    cats['categories'] = cats['categories'].str.replace("'", '', regex=False)

    cats['categories'] = cats['categories'].str.split(',')

    cats = cats.explode('categories')

    cats['categories'] = cats['categories'].str.strip()

    cats = cats[cats['categories'] != '']

    # remove generic Books category
    cats = cats[cats['categories'].str.lower() != 'books']

    # sorted top categories
    top_categories = (
        cats['categories']
        .value_counts()
        .sort_values(ascending=True)
        .tail(10)
    )

    col1.subheader('Top Genres / Categories')
    col1.bar_chart(top_categories)


    # ---------------- AUTHORS ----------------
    top_authors = (
        books['store']
        .value_counts()
        .sort_values(ascending=True)
        .tail(10)
    )

    col2.subheader('Top Authors')
    col2.bar_chart(top_authors)


    # =====================================================
    # SALES / ENGAGEMENT OVER TIME
    # =====================================================
    st.subheader('Top Genre Engagement Over Time')

    # prepare dates
    books['timestamp'] = pd.to_datetime(books['timestamp'])

    # get top 5 genres
    top5_genres = (
        cats['categories']
        .value_counts()
        .head(5)
        .index
    )

    # rebuild dataset with exploded categories
    trend_df = books[['timestamp', 'categories']].dropna().copy()

    trend_df['categories'] = trend_df['categories'].astype(str)

    trend_df['categories'] = trend_df['categories'].str.replace('[', '', regex=False)
    trend_df['categories'] = trend_df['categories'].str.replace(']', '', regex=False)
    trend_df['categories'] = trend_df['categories'].str.replace("'", '', regex=False)

    trend_df['categories'] = trend_df['categories'].str.split(',')

    trend_df = trend_df.explode('categories')

    trend_df['categories'] = trend_df['categories'].str.strip()

    trend_df = trend_df[
        trend_df['categories'].isin(top5_genres)
    ]

    # =====================================================
    # DATE RANGE FILTER
    # =====================================================
    min_date = trend_df['timestamp'].min().date()
    max_date = trend_df['timestamp'].max().date()

    date_range = st.slider(
        'Select Time Period',
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    start_date, end_date = date_range

    filtered_trend = trend_df[
        (trend_df['timestamp'].dt.date >= start_date) &
        (trend_df['timestamp'].dt.date <= end_date)
    ]

    # monthly aggregation
    filtered_trend['month'] = (
        filtered_trend['timestamp']
        .dt.to_period('M')
        .astype(str)
    )

    genre_time = (
        filtered_trend
        .groupby(['month', 'categories'])
        .size()
        .unstack(fill_value=0)
    )

    st.line_chart(genre_time)
