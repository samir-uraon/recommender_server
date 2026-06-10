import pandas as pd
import ast
import pickle
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# =========================
# Load Datasets
# =========================

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")

movies = movies[
    [
        "movie_id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew",
    ]
]

movies.dropna(inplace=True)

# =========================
# Helper Functions
# =========================

def convert(text):
    return [i["name"] for i in ast.literal_eval(text)]


def convert_cast(text):
    return [i["name"] for i in ast.literal_eval(text)[:3]]


def fetch_director(text):
    for i in ast.literal_eval(text):
        if i["job"] == "Director":
            return [i["name"]]
    return []


# =========================
# Feature Engineering
# =========================

movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert_cast)
movies["crew"] = movies["crew"].apply(fetch_director)

movies["overview"] = movies["overview"].apply(lambda x: x.split())

for feature in ["genres", "keywords", "cast", "crew"]:
    movies[feature] = movies[feature].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )

movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["crew"]
)

new_df = movies[["movie_id", "title", "tags"]].copy()

new_df["tags"] = new_df["tags"].apply(
    lambda x: " ".join(x).lower()
)

# =========================
# Stemming
# =========================

ps = PorterStemmer()

def stem(text):
    return " ".join(
        ps.stem(word) for word in text.split()
    )

new_df["tags"] = new_df["tags"].apply(stem)

# =========================
# Vectorization
# =========================

cv = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = cv.fit_transform(
    new_df["tags"]
).toarray()

vectors = vectors.astype(np.uint8)

print("Vector Shape:", vectors.shape)

# =========================
# Similarity Matrix
# =========================

similarity = cosine_similarity(vectors)

print("Similarity Shape:", similarity.shape)

# =========================
# Save Only Top 100 Similar Movies
# =========================

top_similarities = []

for i in range(len(similarity)):

    # Top 100 movie indices
    top_indices = np.argsort(
        similarity[i]
    )[-101:-1][::-1]

    recommendations = [
        (
            int(idx),
            float(similarity[i][idx])
        )
        for idx in top_indices
    ]

    top_similarities.append(recommendations)

print("Top 100 Similarities Generated")

# =========================
# Save Files
# =========================

with open("movies.pkl", "wb") as f:
    pickle.dump(
        new_df,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

with open("similarity.pkl", "wb") as f:
    pickle.dump(
        top_similarities,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print("\n✅ Files Saved Successfully")
print(f"Movies Shape: {new_df.shape}")
print(f"Similarity Entries: {len(top_similarities)}")