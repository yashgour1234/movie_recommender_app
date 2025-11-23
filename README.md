 Movie Recommendation System (Genre-Weighted + Cosine Similarity)

A personalized movie recommendation system built using Streamlit, Python, and Machine Learning techniques.
The system dynamically recommends movies based on user-selected genre preferences (in % weights), using cosine similarity between user taste and movie genre vectors.

This project uses TMDB movie data, encoded genre vectors, and a modern Streamlit UI.
All posters are fetched and preprocessed offline, making the app fast & deployment-friendly.

Live Demo
https://movierecommenderapp-er2fgmszbcvunkupwcfdia.streamlit.app/


Features

Interactive Genre Preference Sliders

Users choose how much they like each genre (0–100%):

Action

Romance

Comedy

Horror

Science Fiction

The app instantly updates recommendations.

ML-Powered Recommendation Engine

Movies are encoded using multi-hot genre vectors

User preferences become a weighted vector

Uses cosine similarity to match movies

Combines similarity with movie rating (vote_average)

final_score = 0.8 * similarity + 0.2 * (rating / 10)


Returns top recommendations ranked by match strength

High-Quality Posters

All posters are fetched from TMDB using a pre-processing script.
This avoids live API calls and makes the app deployment-ready.

Modern Netflix-Style UI

Attractive movie cards

Responsive 3-column grid

Dark theme

Clean typography

Works on mobile & desktop


How It Works (Quick Overview)
1. Load Dataset

The CSV contains:

Title

Overview

TMDB poster path

Genre encodings

Rating

2. User Selects Genre Weights

Example:

Action: 80%
Comedy: 60%
Sci-Fi: 90%

3. Cosine Similarity Calculation

Each movie’s genre vector is compared with the user vector.

4. Ranking

Higher similarity → higher recommendation.
Ratings influence ranking slightly.

5. Display Results

Movies are shown with:

Poster

Title

Rating

Genre list

Clean card UI

