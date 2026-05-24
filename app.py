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


# Accessible theme
# https://dashboards.mysidewalk.com/style-guide-for-dashboards/bar-charts-old
# https://medium.com/@verinamk/streamlit-for-beginners-build-your-first-dashboard-58b764a62a2d
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



# big app + lots of data= using @ stops very long load tim
# https://docs.streamlit.io/develop/concepts/architecture/caching
# https://medium.com/@heyamit10/benefits-of-using-streamlit-cache-for-faster-apps-006632d673ef
@st.cache_data
def load_data():
    books = pd.read_csv("books_clean.csv")
    trending = pd.read_csv("trending_clean.csv")
    books["timestamp"] = pd.to_datetime(books["timestamp"], errors="coerce")
    return books, trending

books, trending = load_data()

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


# https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
# https://dashboards.mysidewalk.com/style-guide-for-dashboards/color

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


#https://discuss.streamlit.io/t/text-to-speech-in-streamlt-cloud/66848
# https://medium.com/@pavlo_sydorenko/add-text-to-speech-to-your-web-app-with-5-lines-of-python-code-8c4707f2dc93
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


# recs : book and genre based
# using content based due to data sparsity
# Based on M. Iqbals Lectures
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


def recommend_by_genre(genre):

    df = books[
        books["categories"].astype(str).str.contains(genre, case=False, na=False)
    ]

    df = df.drop_duplicates(subset=["book_title"])
    df = df.sort_values("rating", ascending=False).head(5)

    return df[["book_title", "rating", "categories"]].reset_index(drop=True)


# buttons for navif=gation are easier and more intuitive for old people
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


# card display looks cleaner
# https://discuss.streamlit.io/t/new-component-streamlit-product-card/113494
# Also easier to read according to Gran Joan
#            - rem make text slightly less contrast for eye fatigue

def show_cards(df):
    for _, row in df.iterrows():
        st.markdown(f"""
        <div class="book-card">
            <b>{row['book_title']}</b><br>
            Rating: {row['rating']}<br>
            Genre: {row['categories']}
        </div>
        """, unsafe_allow_html=True)


# home
if st.session_state.page == "home":

    st.header("Overview")
    st.divider()

    # 🔊 PAGE OVERVIEW AUDIO
    audio_controls(
        "Book analytics dashboard home page. "
        "This page includes book recommendations, genre recommendations, top genres of all time, top authors of all time, and reading trends over time."
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


    #top genre and author graphs side by side = can zoom for bigger
    # colours are best pastel and low contrast, minimal blues and greens
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


    # More complex data but allows more tech savvy 65 year olds to investigate more
    # Plus like most 65 year olds have dealt with tech these days
    
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


# Top 100 books trending in 2023 according to nyt
# try to find more updated version if possible?
if st.session_state.page == "trending":

    st.header("Top 100 Trending Books")

    # 🔊 PAGE OVERVIEW AUDIO
    audio_controls(
        "Top 100 trending books page. This table shows the most popular books currently trending."
    )

    st.subheader("Trending Insights")

    # =================================================
    # CLEAN DATA + MULTI-LABEL PARSING (ROBUST)
    # =================================================
    trending = trending.copy()

    # ensure numeric price (CRITICAL FIX)
    trending["book price"] = pd.to_numeric(trending["book price"], errors="coerce")

    # split cleaned genre column
    trending["genre_split"] = trending["genre_clean"].astype(str).str.split("|")

    trending_exploded = trending.explode("genre_split")

    # remove empty values
    trending_exploded = trending_exploded.dropna(subset=["genre_split"])
    trending_exploded = trending_exploded[trending_exploded["genre_split"] != ""]

    # remove duplicate book-genre pairs (CRITICAL FIX for mean inflation)
    trending_exploded = trending_exploded.drop_duplicates(
        subset=["book title", "genre_split"]
    )

    # =================================================
    # AVG PRICE BY GENRE (FIXED)
    # =================================================
    price_by_genre = (
        trending_exploded
        .groupby("genre_split", as_index=False)["book price"]
        .mean()
        .sort_values("book price", ascending=False)
        .head(10)
    )

    price_by_genre.columns = ["Genre", "Avg Price"]

    fig2 = px.bar(
        price_by_genre,
        x="Genre",
        y="Avg Price",
        color="Genre",
        color_discrete_sequence=ACCESSIBLE_COLOURS
    )

    fig2.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, title=""),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=20, b=10)
    )

    st.plotly_chart(fig2, use_container_width=True)

    # =================================================
    # TABLE (WITH CLEAN GENRE COLUMN IN CORRECT POSITION)
    # =================================================

    trending_display = trending.copy()

    # convert list → readable string
    trending_display["Genre"] = trending_display["genre_split"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else ""
    )

    # build column order
    cols = trending_display.columns.tolist()

    # remove unwanted/raw columns from display order
    for col in ["genre", "genre_clean", "genre_split"]:
        if col in cols:
            cols.remove(col)

    # insert Genre right after book title
    insert_pos = cols.index("book title") + 1
    cols.insert(insert_pos, "Genre")

    trending_display = trending_display[cols]

    st.dataframe(
        trending_display.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        height=900
    )
