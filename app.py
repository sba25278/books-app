import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components

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
# ACCESSIBLE THEME
# =====================================================
st.markdown("""
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

div.stButton > button {
    background-color: #4a6fa5;
    color: white;
    font-size: 18px;
    border-radius: 8px;
    padding: 10px;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)


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
# MODEL
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
# COLOURS
# =====================================================
ACCESSIBLE_COLOURS = [
    "#c94c4c",
    "#e07a5f",
    "#f2a65a",
    "#6a994e",
    "#8b5e3c",
    "#c06c84",
    "#8e7cc3",
    "#2f2f2f"
]


# =====================================================
# TEXT-TO-SPEECH CONTROLS (FULL)
# =====================================================
def audio_controls(text):

    html_code = f"""
    <div style="display:flex; gap:10px; margin:10px 0;">
        <button onclick="startSpeech()">🔊 Read</button>
        <button onclick="pauseSpeech()">⏸ Pause</button>
        <button onclick="resumeSpeech()">▶ Resume</button>
        <button onclick="stopSpeech()">⏹ Stop</button>
    </div>

    <script>
    var utterance;

    function startSpeech() {{
        window.speechSynthesis.cancel();
        utterance = new SpeechSynthesisUtterance(`{text}`);
        utterance.rate = 0.9;
        window.speechSynthesis.speak(utterance);
    }}

    function pauseSpeech() {{
        window.speechSynthesis.pause();
    }}

    function resumeSpeech() {{
        window.speechSynthesis.resume();
    }}

    function stopSpeech() {{
        window.speechSynthesis.cancel();
    }}
    </script>
    """

    components.html(html_code, height=120)


# =====================================================
# RECOMMENDATIONS FUNCTION (CONTENT BASED)
# =====================================================
def recommend_books(title, top_n=5):

    if title not in book_idx:
        return pd.DataFrame()

    idx = book_idx[title]

    scores = list(enumerate(content_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []
    seen = set()

    for i, score in scores:
        book = df_cb.iloc[i]["book_title"]

        if book != title and book not in seen:
            seen.add(book)
            results.append(i)

        if len(results) == top_n:
            break

    return df_cb.iloc[results][["book_title", "rating", "categories"]].reset_index(drop=True)


# =====================================================
# GENRE RECOMMENDATION FUNCTION
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
        st.markdown(f"""
        <div class="book-card">
            <b>{row['book_title']}</b><br>
            Rating: {row['rating']}<br>
            Genre: {row['categories']}
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# HOME PAGE
# =====================================================
if st.session_state.page == "home":

    st.header("Overview")
    st.divider()

    # 🔊 PAGE OVERVIEW AUDIO
    audio_controls(
        "Book analytics dashboard home page. "
        "This page includes book recommendations, genre recommendations, top genres, top authors, and reading trends over time."
    )

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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Book Recommendations")
        if selected_book:
            rec = recommend_books(selected_book)
            show_cards(rec)

            audio_controls(
                "Content based recommendations. " + rec.to_string(index=False)
            )

    with col2:
        st.subheader("Genre Recommendations")
        if selected_genre:
            rec2 = recommend_by_genre(selected_genre)
            show_cards(rec2)

            audio_controls(
                "Genre based recommendations. " + rec2.to_string(index=False)
            )

    st.divider()


    # =================================================
    # TOP GENRES + AUTHORS
    # =================================================
    col1, col2 = st.columns(2)

    with col1:
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
            plot_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
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
            plot_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()


    # =================================================
    # TREND OVER TIME
    # =================================================
    st.subheader("Reading Trends Over Time")

    clean = books.dropna(subset=["timestamp", "categories"]).copy()
    clean["year"] = clean["timestamp"].dt.year

    top5 = clean["categories"].value_counts().head(5).index
    trend = clean[clean["categories"].isin(top5)].copy()

    min_year = int(clean["year"].min())
    max_year = int(clean["year"].max())

    year_range = st.slider(
        "Select year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    start_year, end_year = year_range

    trend = trend[(trend["year"] >= start_year) & (trend["year"] <= end_year)]

    chart = trend.groupby(["year", "categories"]).size().unstack(fill_value=0)

    fig = px.line(
        chart,
        markers=True,
        color_discrete_sequence=ACCESSIBLE_COLOURS
    )

    fig.update_layout(
        xaxis=dict(title="Year", showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# TRENDING PAGE
# =====================================================
if st.session_state.page == "trending":

    st.header("Top 100 Trending Books")

    # 🔊 PAGE OVERVIEW AUDIO
    audio_controls(
        "Top 100 trending books page. This table shows the most popular books currently trending."
    )

    st.dataframe(
        trending.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        height=900
    )
