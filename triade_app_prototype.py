import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="La Triade", page_icon="🎬", layout="wide")

# Couleurs Letterboxd
APP_BG = "#14181c"
DESC_BG = "#242c34"
DESC_TEXT = "#93a0ae"
HIGHLIGHT_ORANGE = "#f59331"
HIGHLIGHT_BLUE = "#3fb8ef"
HIGHLIGHT_GREEN = "#00ba2e"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');

    .stApp {{ background-color: {APP_BG}; }}
    
    h1 {{ 
        font-family: 'Bebas Neue', cursive; 
        font-size: 5rem !important; 
        color: white !important; 
        text-align: center;
        margin-bottom: 30px;
    }}
    
    /* Centrage forcé des colonnes et de leur contenu */
    [data-testid="column"] {{
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }}

    .poster-img {{
        width: 160px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.6);
        display: block;
        margin: 0 auto;
    }}

    .movie-title {{
        font-family: 'Bebas Neue', cursive;
        font-size: 1.8rem;
        color: {HIGHLIGHT_BLUE} !important;
        margin-top: 15px;
        text-align: center;
    }}

    .desc-container {{
        background-color: {DESC_BG};
        padding: 15px;
        border-radius: 8px;
        margin: 15px 10px;
        min-height: 120px;
    }}
    .desc-container p {{
        color: {DESC_TEXT} !important;
        font-size: 0.88rem;
        text-align: justify !important;
        line-height: 1.4;
        margin: 0;
    }}

    .credits-text {{
        font-size: 0.75rem;
        color: white !important;
        margin: 2px 0;
        opacity: 0.8;
    }}

    /* CENTRAGE DU BOUTON REFRESH */
    .stButton {{
        display: flex;
        justify-content: center;
        width: 100%;
        margin: 40px 0;
    }}
    
    .stButton > button {{
        background-color: {HIGHLIGHT_ORANGE} !important;
        color: white !important;
        border-radius: 5px;
        padding: 12px 60px !important;
        font-family: 'Bebas Neue';
        font-size: 1.6rem;
        border: none;
    }}
    
    /* Barre de recherche */
    div[data-baseweb="select"] > div {{
        background-color: {DESC_BG} !important;
    }}
</style>
""", unsafe_allow_html=True)

# Fonction de nettoyage pour limiter le cast à 3 personnes
def clean_credits(text, is_cast=False):
    if not text or text == "" or str(text).lower() == "nan": return "Non spécifié"
    # Nettoyage des crochets/guillemets résiduels
    clean = re.sub(r"[\[\]'\"()]", "", str(text))
    if is_cast:
        items = [i.strip() for i in clean.split(',')]
        return ", ".join(items[:3])
    return clean

@st.cache_data
def load_data():
    # Utilisation de ton nouveau fichier propre
    df = pd.read_csv('Triade_ULTIMATE_CLEAN.csv')
    df.columns = [c.lower().strip() for c in df.columns]
    for c in ['director', 'cast', 'description', 'minute', 'name', 'category']:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna('')
    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    df['search_label'] = df['name'].astype(str) + " (" + df['year'].astype(str).str.replace('.0', '', regex=False) + ")"
    # Soupe pour l'algorithme uniquement
    df['soup'] = df.apply(lambda x: (str(x['keywords'])+" ")*5 + (str(x['all_themes'])+" ")*2 + (str(x['genres'])+" ")*2 + str(x['director'])+" "+str(x['cast']).lower(), axis=1)
    return df

df = load_data()
count_matrix = CountVectorizer(stop_words='english').fit_transform(df['soup'])
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
    return df.iloc[movie_indices]

st.markdown("<h1>LA TRIADE</h1>", unsafe_allow_html=True)

selected_labels = st.multiselect(
    "RECHERCHE TES FILMS FAVORIS :",
    options=df['search_label'].sort_values().unique().tolist(),
    max_selections=4
)

if selected_labels:
    results = get_combined_recs(selected_labels)
    st.write("---")
    col1, col2, col3 = st.columns(3)
    
    def draw_movie(category, cat_filter, streamlit_col, highlight_color):
        recs = results[results['category'].str.lower() == cat_filter.lower()]
        if len(recs) > st.session_state.offset:
            movie = recs.iloc[st.session_state.offset]
            url = str(movie['film_url']) if 'film_url' in movie else "#"
            img = str(movie['poster_url']) if movie['poster_url'] != "" else "https://via.placeholder.com/160x240"
            
            with streamlit_col:
                st.markdown(f"<h2 style='color:{highlight_color} !important;'>{category}</h2>", unsafe_allow_html=True)
                st.markdown(f'<a href="{url}" target="_blank"><img src="{img}" class="poster-img"></a>', unsafe_allow_html=True)
                st.markdown(f'<a href="{url}" target="_blank"><div class="movie-title">{movie["name"]}</div></a>', unsafe_allow_html=True)
                
                year = str(movie['year'])[:4]
                try:
                    # Conversion propre de la durée
                    time_val = float(movie['minute']) if movie['minute'] != "" else 0
                    time = f"{int(time_val)} min" if time_val > 0 else ""
                except:
                    time = ""
                
                st.markdown(f"<p style='font-size:0.9rem; opacity:0.8; color:white; text-align:center;'>{year} | ⭐ {movie['rating']} {f'| {time}' if time else ''}</p>", unsafe_allow_html=True)
                
                st.markdown(f'<div class="desc-container"><p>{movie["description"][:280]}...</p></div>', unsafe_allow_html=True)
                
                # Affichage des crédits (limité à 3 pour le cast)
                st.markdown(f"<p class='credits-text'><b>Director:</b> {clean_credits(movie['director'])}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='credits-text'><b>Cast:</b> {clean_credits(movie['cast'], True)}</p>", unsafe_allow_html=True)

    draw_movie("LA VALEUR SÛRE", "Blockbuster", col1, HIGHLIGHT_ORANGE)
    draw_movie("LE CHOIX CULTE", "Culte", col2, HIGHLIGHT_BLUE)
    draw_movie("LA PÉPITE", "Pépite", col3, HIGHLIGHT_GREEN)

    # Bouton rafraîchir centré
    if st.button("VOIR D'AUTRES RÉSULTATS"):
        st.session_state.offset += 1
        st.rerun()
else:
    st.info("Sélectionne des films pour découvrir ta Triade.")
