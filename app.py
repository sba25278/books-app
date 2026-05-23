import pandas as pd
import numpy as np
import streamlit as st

import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mlxtend.frequent_patterns import apriori, association_rules


st.set_page_config(page_title='Book Recommendation System', layout='wide')

st.title('Book Recommendation System')


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    books = pd.read_csv('books_clean.csv')
    trending = pd.read_csv('trending_clean.csv')
    return books, trending


books, trending = load_data()


# -----------------------------
# KPI CARDS (NEW)
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric('Total Books', len(books))
col2.metric('Total Users', books['user_id'].nunique())
col3.metric('Average Rating', round(books['rating'].mean(), 2))
col4.metric('Trending Titles', trending['book title'].nunique())


# -----------------------------
# CONTENT MODEL
# -----------------------------
def build_content_model(df):

    df = df.dropna(subset=['content', 'book_title'])

    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    tfidf_matrix = tfidf.fit_transform(df['content'])

    sim = cosine_similarity(tfidf_matrix)

    book_index = pd.Series(df.index, index=df['book_title']).drop_duplicates()

    return sim, book_index, df


content_sim, book_idx, df_cb = build_content_model(books)


def recommend_content(title, top_n=10):

    if title not in book_idx:
        return pd.DataFrame()

    idx = book_idx[title]

    scores = list(enumerate(content_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    indices = [i[0] for i in scores]

    recs = df_cb.iloc[indices][['book_title', 'rating']].copy()
    recs['similarity'] = [i[1] for i in scores]

    return recs.drop_duplicates(subset=['book_title'])


# -----------------------------
# BASKET BUILD (MEMORY SAFE)
# -----------------------------
def build_basket(df):

    df = df[df['rating'] >= 4]
    df = df[df['verified_purchase'] == True]

    df = df[['user_id', 'book_title']].drop_duplicates()

    user_counts = df['user_id'].value_counts()
    df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]

    book_counts = df['book_title'].value_counts()
    df = df[df['book_title'].isin(book_counts[book_counts >= 10].index)]

    basket = df.groupby(['user_id', 'book_title']).size().unstack(fill_value=0)

    return basket.astype(bool)


basket = build_basket(books)


# -----------------------------
# APRIORI
# -----------------------------
def run_apriori(basket):

    freq = apriori(
        basket,
        min_support=0.01,
        use_colnames=True,
        low_memory=True
    )

    rules = association_rules(
        freq,
        metric='lift',
        min_threshold=1.0
    )

    return rules


rules = run_apriori(basket)


# -----------------------------
# SIDEBAR
# -----------------------------
menu = st.sidebar.radio(
    'Select View',
    [
        'Content-Based Recommendations',
        'Market Basket Analysis',
        'Trending Books',
        'Analytics Dashboard'
    ]
)


# -----------------------------
# CONTENT-BASED
# -----------------------------
if menu == 'Content-Based Recommendations':

    st.header('Content-Based Book Recommendations')

    book_name = st.text_input('Enter book title')

    if book_name:
        recs = recommend_content(book_name)

        if recs.empty:
            st.warning('Book not found in dataset')
        else:
            st.dataframe(recs, use_container_width=True)


# -----------------------------
# MARKET BASKET
# -----------------------------
if menu == 'Market Basket Analysis':

    st.header('Books Frequently Bought Together')

    min_lift = st.slider('Minimum Lift', 1.0, 10.0, 2.0)

    filtered_rules = rules[
        (rules['lift'] >= min_lift) &
        (rules['confidence'] >= 0.3)
    ]

    st.dataframe(
        filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20),
        use_container_width=True
    )


# -----------------------------
# TRENDING (CLEAN VISUALS)
# -----------------------------
if menu == 'Trending Books':

    st.header('Top 100 Trending Books')

    st.dataframe(trending, use_container_width=True)


# -----------------------------
# ANALYTICS DASHBOARD (FULL UPGRADE)
# -----------------------------
if menu == 'Analytics Dashboard':

    st.header('Book Engagement Insights')

    st.write(
        'This dashboard highlights popularity patterns across books, genres, and authors.'
    )

    # ---------------- TOP BOOKS ----------------
    st.subheader('Most Popular Books')

    top_books = trending['book title'].value_counts().head(10).reset_index()
    top_books.columns = ['Book', 'Count']

    fig1 = px.bar(top_books, x='Count', y='Book', orientation='h', text='Count')
    fig1.update_layout(height=450)

    st.plotly_chart(fig1, use_container_width=True)


    # ---------------- GENRES ----------------
    st.subheader('Top Genres')

    genres = trending['genre'].value_counts().head(10).reset_index()
    genres.columns = ['Genre', 'Count']

    fig2 = px.bar(genres, x='Count', y='Genre', orientation='h', text='Count')
    fig2.update_layout(height=450)

    st.plotly_chart(fig2, use_container_width=True)


    # ---------------- AUTHORS ----------------
    st.subheader('Top Authors')

    authors = trending['author'].value_counts().head(10).reset_index()
    authors.columns = ['Author', 'Count']

    fig3 = px.bar(authors, x='Count', y='Author', orientation='h', text='Count')
    fig3.update_layout(height=450)

    st.plotly_chart(fig3, use_container_width=True)
