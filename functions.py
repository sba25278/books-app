# functions.py
# For Contnet based filtering
import pandas as pd
import numpy as np

def recommend_content_based(book_title, top_n, df, book_idx, content_sim):

    if book_title not in book_idx:
        raise ValueError(f"Book '{book_title}' not found.")

    idx = book_idx[book_title]

    sim_scores = list(enumerate(content_sim[idx].flatten()))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1: top_n + 1]

    book_indices = [i for i, score in sim_scores]
    scores = [score for i, score in sim_scores]

    recs = df.iloc[book_indices][
        ["parent_asin", "book_title", "categories"]
    ].copy()

    recs["similarity_score"] = scores

    return recs.reset_index(drop=True)

# For User - User Collaboration

def recommend_user_user(
    user_id,
    user_item,
    user_sim_df,
    df,
    top_n=10,
    min_sim=0.05
):

    if user_id not in user_item.index:
        raise ValueError(f"User {user_id} not found.")

    sims = user_sim_df.loc[user_id].drop(user_id, errors="ignore")
    sims = sims[sims > min_sim].sort_values(ascending=False)

    if sims.empty:
        return pd.DataFrame(columns=["parent_asin", "book_title", "categories", "predicted_rating"])

    target_ratings = user_item.loc[user_id]
    unseen_items = target_ratings[target_ratings.isna()].index

    predictions = {}

    for item in unseen_items:

        neighbor_ratings = user_item.loc[sims.index, item].dropna()

        if len(neighbor_ratings) == 0:
            continue

        aligned_sims = sims.loc[neighbor_ratings.index]

        denom = aligned_sims.abs().sum()

        if denom == 0:
            continue

        pred = np.dot(neighbor_ratings.values, aligned_sims.values) / denom
        predictions[item] = pred

    if len(predictions) == 0:
        return pd.DataFrame(columns=["parent_asin", "book_title", "categories", "predicted_rating"])

    recs = (
        pd.DataFrame(predictions.items(), columns=["parent_asin", "predicted_rating"])
        .sort_values("predicted_rating", ascending=False)
        .head(top_n)
    )

    recs = recs.merge(
        df[["parent_asin", "book_title", "categories"]],
        on="parent_asin",
        how="left"
    )

    return recs.reset_index(drop=True)

# Item-item collaborative Filtering

def recommend_item_item(
    user_id,
    user_item,
    item_sim_df,
    df,
    top_n=10,
    min_user_rating=4.0
):

    if user_id not in user_item.index:
        raise ValueError(f"User {user_id} not found.")

    user_ratings = user_item.loc[user_id].dropna()

    liked_items = user_ratings[user_ratings >= min_user_rating]
    unseen_items = user_item.loc[user_id][user_item.loc[user_id].isna()].index

    if len(liked_items) == 0:
        return pd.DataFrame(columns=["parent_asin", "book_title", "categories", "predicted_rating"])

    scores = {}

    for candidate in unseen_items:

        sim_sum = 0.0
        weighted_sum = 0.0

        for liked_item, rating in liked_items.items():

            if candidate not in item_sim_df.index or liked_item not in item_sim_df.columns:
                continue

            sim = item_sim_df.loc[candidate, liked_item]

            if sim <= 0:
                continue

            weighted_sum += sim * rating
            sim_sum += sim

        if sim_sum > 0:
            scores[candidate] = weighted_sum / sim_sum

    if len(scores) == 0:
        return pd.DataFrame(columns=["parent_asin", "book_title", "categories", "predicted_rating"])

    recs = (
        pd.DataFrame(scores.items(), columns=["parent_asin", "predicted_rating"])
        .sort_values("predicted_rating", ascending=False)
        .head(top_n)
    )

    recs = recs.merge(
        df[["parent_asin", "book_title", "categories"]],
        on="parent_asin",
        how="left"
    )

    return recs.reset_index(drop=True)