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


# =====================================================
# SENIOR-FRIENDLY STYLE
# =====================================================
st.markdown(
    """
    <style>

    .stApp {
        background-color: #fbf3e6;
    }

    h1 {
        color: #4a3324;
        font-size: 42px;
    }

    h2, h3 {
        color: #5b4636;
        font-size: 28px;
    }

    p, div, label {
        font-size: 20px;
        color: #3d2b1f;
    }

    div.stButton > button {
        background-color: #c9895b;
        color: white;
        font-size: 20px;
        border-radius: 12px;
        padding: 12px;
        width: 100%;
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
# CONTENT MODEL (SAFE)
# =====================================================
@st.cache_data
def build_content_model(df):

    df = df.dropna(subset=['content', 'book_title']).copy()

    df = df.sample(min(5000, len(df)), random_state=42).reset_index(drop=True)

    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    tfidf_matrix = tfidf.fit_transform(df['content'])

    sim = cosine_similarity(tfidf_matrix)

    book_index = df.reset_index().set_index('book_title')['index'].to_dict()

    return sim, book_index, df


content_sim, book_idx, df_cb = build_content_model(books)


# =====================================================
# RECOMMENDATION FUNCTION
# =====================================================
def recommend_books(title, top_n=5):

    if title not in book_idx:
        return pd.DataFrame()

    idx = book_idx[title]

    scores = list(enumerate(content_sim[idx].flatten()))

    scores = sorted(
        scores,
        key=lambda x: float(x[1]),
        reverse=True
    )[1:top_n + 1]

    indices = [i[0] for i in scores]

    recs = df_cb.iloc[indices][['book_title', 'rating', 'categories']].copy()

    recs['similarity'] = [round(float(i[1]), 2) for i in scores]

    recs.columns = ['Book Title', 'Rating', 'Genre', 'Match Score']

    return recs.reset_index(drop=True)


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
        .head(5)
        .reset_index(drop=True)
    )

    genre_df.columns = ['Book Title', 'Rating', 'Genre']

    return genre_df


# =====================================================
# NAVIGATION
# =====================================================
col1, col2 = st.columns(2)

with col1:
    home_btn = st.button("Home")

with col2:
    trending_btn = st.button("Trending Books")

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

    st.header("Find Books You’ll Enjoy")

    st.write(
        "Choose a book or genre below. We’ll show simple recommendations."
    )

    st.divider()


    # =================================================
    # RECOMMENDATION AREA
    # =================================================
    st.subheader("Book Recommendations")

    col1, col2 = st.columns(2)

    with col1:

        selected_book = st.selectbox(
            "Choose a book",
            sorted(df_cb['book_title'].dropna().unique())
        )

        if selected_book:

            results = recommend_books(selected_book)

            st.write("### Because you liked this book, you may also like:")

            for _, row in results.iterrows():

                st.markdown(
                    f"""
                    <div style="
                        background-color:#fff7ec;
                        padding:16px;
                        border-radius:12px;
                        margin-bottom:12px;
                        border-left:6px solid #c9895b;
                    ">

                    <h3 style="margin-bottom:5px;">
                        {row['Book Title']}
                    </h3>

                    <p>
                        Rating: {row['Rating']}  
                        Match: {row['Match Score']}
                    </p>

                    <p>
                        This book is similar in theme, style, and content to
                        <b>{selected_book}</b>.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


    with col2:

        selected_genre = st.selectbox(
            "Choose a genre",
            sorted(books['categories'].dropna().unique())
        )

        if selected_genre:

            genre_recs = recommend_by_genre(selected_genre)

            st.write("### Popular books in this genre:")

            st.dataframe(genre_recs, use_container_width=True)


    st.divider()


    # =================================================
    # CHARTS (WARM COLOURS)
    # =================================================
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Most Popular Genres")

        data = books['categories'].value_counts().head(8).reset_index()
        data.columns = ['Genre', 'Count']

        fig = px.bar(
            data,
            x='Genre',
            y='Count',
            color='Genre',
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Most Popular Authors")

        data = books['store'].value_counts().head(8).reset_index()
        data.columns = ['Author', 'Count']

        fig = px.bar(
            data,
            x='Author',
            y='Count',
            color='Author',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        st.plotly_chart(fig, use_container_width=True)


    st.divider()


    # =================================================
    # TIME SLIDER CHART
    # =================================================
    st.subheader("Reading Trends Over Time")

    books_clean = books.dropna(subset=['timestamp', 'categories']).copy()

    top5 = books_clean['categories'].value_counts().head(5).index

    trend = books_clean[books_clean['categories'].isin(top5)].copy()

    min_d = trend['timestamp'].min().date()
    max_d = trend['timestamp'].max().date()

    date_range = st.slider(
        "Select time period",
        min_value=min_d,
        max_value=max_d,
        value=(min_d, max_d)
    )

    start, end = date_range

    filtered = trend[
        (trend['timestamp'].dt.date >= start) &
        (trend['timestamp'].dt.date <= end)
    ].copy()

    filtered['month'] = filtered['timestamp'].dt.to_period('M').astype(str)

    chart = filtered.groupby(['month', 'categories']).size().unstack(fill_value=0)

    st.line_chart(chart, use_container_width=True)


# =====================================================
# TRENDING PAGE
# =====================================================
if st.session_state.page == 'trending':

    st.header("Trending Books")

    st.dataframe(
        trending.reset_index(drop=True),
        use_container_width=True
    )
