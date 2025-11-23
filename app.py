import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load & Clean Data ----------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    genre_cols = ['Action', 'Romance', 'Comedy', 'Horror', 'Science Fiction']

    # Ensure genre columns exist
    for col in genre_cols:
        if col not in df.columns:
            df[col] = 0

    # Convert genre columns to integer (0/1)
    for col in genre_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Convert the 'genres' column (list-like strings) into readable text
    if 'genres' in df.columns:
        df['genres'] = df['genres'].apply(
            lambda x: x.replace("[", "")
                      .replace("]", "")
                      .replace("'", "")
                      .strip()
            if isinstance(x, str) else "Unknown"
        )

    # Remove old similarity scores if present
    for col in ["similarity_score", "final_score"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df, genre_cols


# Load your dataset
df, genre_cols = load_data("tmdb_movies_posters_fixed.csv")


# ---------- Streamlit UI ----------
st.title("🎬 Movie Recommendation System")
st.subheader("Customized by Your Genre Preferences (%)")

st.sidebar.header("🎛 Adjust Your Genre Preferences")

# Sidebar sliders
user_weights = []
for genre in genre_cols:
    default_val = int(df[genre].mean() * 100)
    val = st.sidebar.slider(genre, 0, 100, default_val)
    user_weights.append(val)

# Normalize user preferences
user_vector = np.array(user_weights) / 100.0


# ---------- Cosine Similarity ----------
# Movie matrix
movie_matrix = df[genre_cols].astype(float).values

# Normalize movie vectors so multi-genre movies don't dominate
movie_matrix = movie_matrix / np.clip(movie_matrix.sum(axis=1, keepdims=True), 1, None)

# Compute cosine similarity
similarities = cosine_similarity([user_vector], movie_matrix).flatten()

df["similarity_score"] = similarities

# Combine with rating (weighted)
df["final_score"] = 0.8 * df["similarity_score"] + 0.2 * (df["vote_average"] / 10)


# ---------- Display Recommendations ----------
top_n = st.sidebar.slider("Number of Recommendations", 1, 20, 5)
top_movies = df.sort_values("final_score", ascending=False).head(top_n)


st.subheader("🎥 Recommended Movies")

# Modern 3-column grid layout
cols = st.columns(3)

for i, (_, row) in enumerate(top_movies.iterrows()):
    col = cols[i % 3]  # pick column based on index
    
    with col:
        st.markdown(
            f"""
            <div style="
                background-color: #1f1f1f;
                border-radius: 12px;
                padding: 10px;
                margin-bottom: 20px;
                text-align: center;
                box-shadow: 0 4px 10px rgba(0,0,0,0.4);
            ">
            """,
            unsafe_allow_html=True,
        )

        # Poster Image
        poster = row.get("poster_path")
        if isinstance(poster, str) and len(poster) > 3:
            st.image(f"https://image.tmdb.org/t/p/w500{poster}", use_container_width=True)
        else:
            st.image(
                "https://via.placeholder.com/300x450.png?text=No+Poster",
                use_container_width=True
            )

        # Title & Rating
        st.markdown(
            f"""
            <h4 style="color:white; margin-top:10px;">{row.title}</h4>
            <p style="color:#d4d4d4; font-size:15px;">
                ⭐ {row.vote_average:.1f}  
            </p>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)


# for _, row in top_movies.iterrows():
#     col1, col2 = st.columns([1, 4])

#     with col1:
#         poster = row.get("poster_path")
#         if isinstance(poster, str) and len(poster) > 3:
#             st.image(f"https://image.tmdb.org/t/p/w500{poster}", width=120)
#         else:
#             st.write("No Poster")

#     with col2:
#         st.markdown(f"**{row.title}**  \n⭐ {row.vote_average:.1f}")
#         st.write(row.genres)


# st.sidebar.markdown(f"📦 Total Movies Loaded: **{len(df)}**")
