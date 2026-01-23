import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="La Triade", page_icon="🎬", layout="wide")

# CSS pour forcer l'apparence
st.markdown(f"""
<style>
    .stApp {{
        background-color: #445566;
    }}
    /* Tout le texte de base en blanc */
    h1, h2, h3, h4, h5, p, span {{
        color: white !important;
    }}
    /* Centrage des colonnes Streamlit */
    [data-testid="column"] {{
        text-align: center;
    }}
    /* Style spécifique pour la description (fond clair, texte noir) */
    .desc-box {{
        background-color: #f8f9fa;
        color: #333333 !important;
        padding: 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-top: 10px;
        text-align: justify;
        line-height: 1.3;
    }}
    .desc-box p {{
        color: #333333 !important;
        margin: 0;
    }}
    /* Le bouton Refresh */
    .stButton>button {{
        background-color: #FF4B4B !important;
        color: white !important;
        font-weight: bold;
        border-radius: 20px;
        padding: 10px 30px;
        border: none;
    }}
    .movie-link {{
        text-decoration: none;
    }}
    .poster-img {{
        width: 150px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }}
</style>
""", unsafe_allow_html=True)

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
    
    # Algorithme : Keywords (x5) / Thèmes (x2)
    def create_soup(x):
        return ((x['keywords'] + " ") * 5 + (x['all_themes'] + " ") * 2 + (x['genres'] + " ") * 2 + x['director'] + " " + x['cast']).lower()

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
st.markdown("<center><h1>🎬 La Triade</h1></center>", unsafe_allow_html=True)

selected_labels = st.multiselect(
    "Quels films as-tu aimés ? (max 4)",
    options=df['search_label'].sort_values().unique().tolist(),
    max_selections=4
)

if selected_labels:
    results = get_combined_recs(selected_labels)
    st.write("---")
    
    col1, col2, col3 = st.columns(3)
    
    def draw_
