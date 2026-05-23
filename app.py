import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    recs = recs.drop_duplicates(subset=['book_title'])

    return recs


# -----------------------------
# BASKET BUILD (MEMORY SAFE)
# -----------------------------
def build_basket(df):

    df = df[df['rating'] >= 4]
    df = df[df['verified_purchase'] == True]

    df = df[['user_id', 'book_title']].drop_duplicates()

    # reduce sparsity
    user_counts = df['user_id'].value_counts()
    df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]

    book_counts = df['book_title'].value_counts()
    df = df[df['book_title'].isin(book_counts[book_counts >= 10].index)]

    basket = df.groupby(['user_id', 'book_title']).size().unstack(fill_value=0)

    # convert to binary safely
    basket = basket > 0

    return basket


basket = build_basket(books)


# -----------------------------
# APRIORI (SAFE SETTINGS)
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
# SIDEBAR MENU
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

        recs = recommend_content(book_name, top_n=10)

        if recs.empty:
            st.write('Book not found in dataset')
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
# TRENDING
# -----------------------------
if menu == 'Trending Books':

    st.header('Top 100 Trending Books')

    st.dataframe(trending, use_container_width=True)


# -----------------------------
# ANALYTICS DASHBOARD
# -----------------------------
if menu == 'Analytics Dashboard':

    st.header('Dataset Insights from Top 100 Trending Books')

    col1, col2, col3 = st.columns(3)

    # POPULAR BOOKS
    top_books = trending['book title'].value_counts().head(10)

    fig1, ax1 = plt.subplots()
    top_books.plot(kind='bar', ax=ax1)
    ax1.set_title('Most Popular Trending Books')
    ax1.tick_params(axis='x', rotation=45)

    col1.pyplot(fig1)


    # GENRES
    genre_counts = trending['genre'].value_counts().head(10)

    fig2, ax2 = plt.subplots()
    genre_counts.plot(kind='bar', ax=ax2)
    ax2.set_title('Top Genres')

    col2.pyplot(fig2)


    # AUTHORS
    author_counts = trending['author'].value_counts().head(10)

    fig3, ax3 = plt.subplots()
    author_counts.plot(kind='bar', ax=ax3)
    ax3.set_title('Top Authors')

    col3.pyplot(fig3)
