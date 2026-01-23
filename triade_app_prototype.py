import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- CONFIGURATION PAGE & DESIGN ---
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
    /* Alignement vertical et horizontal parfait */
    [data-testid="column"] {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        gap: 10px;
    }}
    .movie-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
        text-align: center;
    }}
    .movie-poster {{
        border-radius: 12px;
        width: 180px;
        transition: transform 0.3s, box-shadow 0.3s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }}
    .movie-poster:hover {{
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0,0,0,0.7);
    }}
    .tag-style {{
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 3px 10px;
        margin: 3px;
        display: inline-block;
        font-size: 0.75rem;
        color: #00d4ff !important; /* Bleu cyan pour plus de modernité */
        font-weight: 600;
    }}
    .description-text {{
        font-size: 0.85rem;
        font-style: italic;
        margin-top: 10px;
        max-width: 250px;
        line-height: 1.4;
        color: #e0e0e0 !important;
    }}
    a {{ text-decoration: none !important; }}
</style>
""", unsafe_allow_html=True)

# --- CHARGEMENT ---
@st.cache_data
def load_data():
    df = pd.read_csv('Triade_ULTIMATE.csv')
    df.columns = [c.lower() for c in df.columns]
    
    # Sécurisation des colonnes (évite le KeyError 'overview')
    expected = ['genres', 'keywords', 'director', 'cast', 'all_themes', 'overview', 'name', 'category']
    for c in expected:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna('').astype(str)

    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    
    # Label pour la barre de recherche : "Nom (Année)"
    df['search_label'] = df['name'] + " (" + df['year'].apply(lambda x: str(x).replace('.0', '')) + ")"
    
    # --- ALGORITHME : PRIORITÉ AUX KEYWORDS (x5) ---
    def create_soup(x):
        # On inverse les poids : Keywords x5, Thèmes x3
        keywords = (x['keywords'] + " ") * 5
        themes = (x['all_themes'] + " ") * 3
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
    input_soups = []
    selected_indices = []
    for label in search_labels:
        idx = indices[label]
        # Gestion propre de l'index (entier ou série)
        actual_idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx
        selected_indices.append(actual_idx)
        input_soups.append(df.iloc[actual_idx]['soup'])
        cos_sim = cosine_similarity(count_matrix[actual_idx], count_matrix)[0]
        all_sim_scores = cos_sim if all_sim_scores is None else all_sim_scores + cos_sim
            
    sim_scores = sorted(list(enumerate(all_sim_scores)), key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores if i[0] not in selected_indices]
    return df.iloc[movie_indices[0:150]], " ".join(input_soups)

# --- INTERFACE ---
st.markdown("<h1 style='margin-bottom:0;'>🎬 La Triade</h1>", unsafe_allow_html=True)

# Barre de recherche multi-sélection (max 4)
selected_labels = st.multiselect(
    "Ajoute jusqu'à 4 films favoris :",
    options=df['search_label'].sort_values().unique().tolist(),
    max_selections=4,
    placeholder="Commence à taper le nom d'un film..."
)

if selected_labels:
    results, combined_input_soup = get_combined_recs(selected_labels)
    st.write("---")
    col1, col2, col3 = st.columns(3)
    
    def draw_movie(category, cat_filter, col):
        recs = results[results['category'].str.lower() == cat_filter.lower()]
        if len(recs) > st.session_state.offset:
            movie = recs.iloc[st.session_state.offset]
            
            # Nettoyage des tags (extraction propre des mots sans quotes)
            movie_words = re.findall(r'\w+', movie['soup'])
            input_words = set(re.findall(r'\w+', combined_input_soup))
            common_tags = []
            for w in movie_words:
                if w in input_words and len(w) > 3 and w not in common_tags:
                    common_tags.append(w)
                if len(common_tags) >= 5: break
            
            url = movie['film_url'] if 'film_url' in movie and movie['film_url'] != "" else "#"
            img = movie['poster_url'] if movie['poster_url'] != "" else "https://via.placeholder.com/180x270?text=Image+Indisponible"

            with col:
                st.markdown(f"### {category}")
                # POSTER ET TITRE CLIQUABLES
                st.markdown(f'''
                    <div class="movie-container">
                        <a href="{url}" target="_blank">
                            <img src="{img}" class="movie-poster">
                            <h4 style="margin:15px 0 5px 0; font-size:1.2rem;">{movie['name']}</h4>
                        </a>
                        <p style="margin-bottom:10px; font-weight:bold;">{str(movie['year'])[:4]} | ⭐ {movie['rating']}</p>
                    </div>
                ''', unsafe_allow_html=True)
                
                # Tags propres
                tags_html = "".join([f'<span class="tag-style">#{t}</span>' for t in common_tags])
                st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)
                
                # Description
                if movie['overview'] and movie['overview'] != "":
                    desc = movie['overview'][:160] + "..." if len(movie['overview']) > 160 else movie['overview']
                    st.markdown(f'<p class="description-text">{desc}</p>', unsafe_allow_html=True)
        else:
            col.warning(f"Plus de {category} disponible.")

    draw_movie("La Valeur Sûre", "Blockbuster", col1)
    draw_movie("Le Choix Culte", "Culte", col2)
    draw_movie("La Pépite", "Pépite", col3)

    st.write("---")
    if st.button("🔄 Pas convaincu ? Voir d'autres résultats"):
        st.session_state.offset += 1
        st.rerun()
else:
    st.info("Utilise la barre de recherche ci-dessus pour lancer ta recommandation.")
