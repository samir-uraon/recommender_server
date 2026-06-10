import os
import pickle
import requests
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import unquote
from functools import lru_cache


# =========================
# APP INIT
# =========================
app = FastAPI()

# =========================
# ENV CONFIG
# =========================
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "54b90aab")

# =========================
# CORS FIX
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD DATA
# =========================
with open("movies.pkl", "rb") as f:
    movies = pickle.load(f)

similarity = pickle.load(open("similarity.pkl", "rb"))


@lru_cache(maxsize=500)
def fetch_movie_details(movie_title: str):
    try:
        url = (
            f"https://www.omdbapi.com/"
            f"?t={movie_title}&apikey={OMDB_API_KEY}"
        )

        response = requests.get(url, timeout=5)
        data = response.json()

        if data.get("Response") == "True":
            return {
                "title": data.get("Title"),
                "poster": data.get("Poster"),
                "year": data.get("Year"),
                "rating": data.get("imdbRating"),
            }

    except Exception as e:
        print("OMDB Error:", e)

    return {
        "title": movie_title,
        "poster": None,
        "year": None,
        "rating": None,
    }

# =========================
# RECOMMENDATION ENGINE
# =========================
def recommend(movie_name: str):
    match = movies[movies["title"] == movie_name]

    if match.empty:
        raise ValueError("Movie not found")

    movie_index = match.index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        enumerate(distances),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    results = []

    for i in movie_list:
        title = movies.iloc[i[0]].title
        results.append(fetch_movie_details(title))

    return results

# =========================
# ROUTES
# =========================

@app.get("/")
def home():
    return {"message": "Movie Recommendation API Running"}

@app.get("/movies")
def get_movies():
    return {"movies": movies["title"].tolist()}

@app.get("/recommend/{movie_name}")
def get_recommendations(movie_name: str):
    try:
        movie_name = unquote(movie_name)

        recommendations = recommend(movie_name)

        return {
            "success": True,
            "selected_movie": movie_name,
            "recommendations": recommendations
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
