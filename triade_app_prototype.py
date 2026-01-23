import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- CONFIGURATION PAGE & DESIGN COMPACT ---
st.set_page_config(page_title="La Triade", page_icon="🎬", layout="wide")

st.markdown(f"""
<style>
    .stApp {{
        background-color: #445566;
        color: white;
    }}
    h1, h2, h3, h4, h5, p, span, label {{
        color: white !important;
        text-align: center;
    }}
    /* Centrage et réduction des espaces */
    [data-testid="column"] {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        gap: 5px;
    }}
    .movie-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
        text-align: center;
    }}
    .movie-poster {{
        border-radius: 10px;
        width: 150px; /* Taille réduite pour tout voir sur une page */
        transition: transform 0.2s;
        box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    }}
    .movie-poster:hover {{
        transform: scale(1.03);
    }}
    .description-text {{
        font-size: 0.8rem;
        font-style: italic;
        margin-top: 8px;
        max-width: 220px;
        line-height: 1.2;
        color: #d1d1d1 !important;
    }}
    a {{ text-decoration: none !important; }}
    hr {{ margin: 10px 0 !important; }}
</style>
""", unsafe_allow_html=True)

# --- CHARGEMENT ---
@st.cache_data
def load_data():
    df = pd.read_csv('Triade_ULTIMATE.csv')
    df.columns = [c.lower() for c in df.columns]
    
    # Sécurité colonnes
    expected = ['genres', 'keywords', 'director', 'cast', 'all_themes', 'overview', 'name', 'category']
    for c in expected:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna('').astype(str)

    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    
    df['search_label'] = df['name'] + " (" + df['year'].apply(lambda x: str(x).replace('.0', '')) + ")"
    
    # --- ALGORITHME MIS À JOUR ---
    def create_soup(x):
        # Keywords x5 (Priorité haute), Thèmes x2 (Priorité basse)
        keywords = (x['keywords'] + " ") * 5
        themes = (x['all_themes'] + " ") * 2
        genres = (x['genres'] + " ") * 2
        crew = x['director'] + " " + x['cast']
        return (keywords + themes + genres + crew).lower()

    df['soup'] = df.apply(create_soup, axis=1)
    return df

@st.cache_resource
def get_vectorizer(df):
    return CountVectorizer(stop_words='english').fit_transform(df['soup'])

df = load_data()
count_matrix = get_vectorizer(df)
indices = pd.Series(df.index, index=df['search_label']).drop_duplicates()

if 'offset' not in st.session_state: st.session_state.offset = 0

def get_combined_recs(search_labels):
    all_sim_scores = None
    selected_indices = []
    for label in search_labels:
        idx = indices[label]
        actual_idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx
        selected_indices.append(actual_idx)
        cos_sim = cosine_similarity(count_matrix[actual_idx], count_matrix)[0]
        all_sim_scores = cos_sim if all_sim_scores is None else all_sim_scores + cos_sim
            
    sim_scores = sorted(list(enumerate(all_sim_scores)), key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores if i[0] not in selected_indices]
    return df.iloc[movie_indices[0:150]]

# --- INTERFACE ---
st.markdown("<h1 style='margin-bottom:0; font-size: 2.2rem;'>🎬 La Triade</h1>", unsafe_allow_html=True)

selected_labels = st.multiselect(
    "Ajoute tes films favoris (max 4) :",
    options=df['search_label'].sort_values().unique().tolist(),
    max_selections=4,
    placeholder="Rechercher un film..."
)

if selected_labels:
    results = get_combined_recs(selected_labels)
    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    def draw_movie(category, cat_filter, col):
        recs = results[results['category'].str.lower() == cat_filter.lower()]
        if len(recs) > st.session_state.offset:
            movie = recs.iloc[st.session_state.offset]
            
            url = movie['film_url'] if 'film_url' in movie and movie['film_url'] != "" else "#"
            img = movie['poster_url'] if movie['poster_url'] != "" else "https://via.placeholder.com/150x225?text=Image+Manquante"

            with col:
                st.markdown(f"### {category}")
                st.markdown(f'''
                    <div class="movie-container">
                        <a href="{url}" target="_blank">
                            <img src="{img}" class="movie-poster">
                            <h4 style="margin:8px 0 2px 0; font-size:1.1rem;">{movie['name']}</h4>
                        </a>
                        <p style="margin-bottom:5px; font-size:0.9rem;">{str(movie['year'])[:4]} | ⭐ {movie['rating']}</p>
                    </div>
                ''', unsafe_allow_html=True)
                
                if movie['overview'] and movie['overview'] != "":
                    desc = movie['overview'][:150] + "..." if len(movie['overview']) > 150 else movie['overview']
                    st.markdown(f'<p class="description-text">{desc}</p>', unsafe_allow_html=True)
        else:
            col.warning(f"Plus de {category}.")

    draw_movie("LA VALEUR SÛRE", "Blockbuster", col1)
    draw_movie("LE CHOIX CULTE", "Culte", col2)
    draw_movie("LA PÉPITE", "Pépite", col3)

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("🔄 Voir d'autres résultats"):
        st.session_state.offset += 1
        st.rerun()
else:
    st.info("Entre un ou plusieurs films pour voir ta Triade s'afficher.")
