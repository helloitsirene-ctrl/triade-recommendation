import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="La Triade", page_icon="🎬", layout="wide")

st.markdown(f"""
<style>
    .stApp {{ background-color: #445566; }}
    h1, h2, h3, h4, h5, p, span {{ color: white !important; text-align: left; }}
    
    /* Conteneur pour chaque ligne de film */
    .movie-row {{
        display: flex;
        align-items: flex-start;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .poster-container {{
        flex: 0 0 150px;
        margin-right: 25px;
        text-align: center;
    }}

    .info-container {{
        flex: 1;
    }}

    .poster-img {{
        width: 150px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }}

    .description-text {{
        font-size: 0.95rem;
        color: #e0e0e0 !important;
        margin: 15px 0;
        line-height: 1.5;
        text-align: justify;
    }}

    .credits-text {{
        font-size: 0.85rem;
        color: #FFD700 !important;
        margin-top: 5px;
    }}

    .stButton>button {{
        background-color: #FF4B4B !important;
        color: white !important;
        border-radius: 20px;
        display: block;
        margin: 0 auto;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('Triade_ULTIMATE.csv')
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Sécurité colonnes
    expected = ['genres', 'keywords', 'director', 'cast', 'all_themes', 'overview', 'name', 'category', 'runtime']
    for c in expected:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna('')

    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    
    df['search_label'] = df['name'].astype(str) + " (" + df['year'].astype(str).str.replace('.0', '', regex=False) + ")"
    
    # Algo : Keywords (x5) / Thèmes (x2)
    def create_soup(x):
        return ((str(x['keywords']) + " ") * 5 + (str(x['all_themes']) + " ") * 2 + (str(x['genres']) + " ") * 2 + str(x['director']) + " " + str(x['cast'])).lower()

    df['soup'] = df.apply(create_soup, axis=1)
    return df

# Fonction pour convertir les minutes en HhMM
def format_runtime(minutes):
    try:
        m = int(float(minutes))
        if m == 0: return ""
        hours = m // 60
        mins = m % 60
        return f"{hours}h{mins:02d}"
    except:
        return ""

# Fonction pour nettoyer le cast (prendre les 3 premiers)
def clean_cast(cast_str):
    if not cast_str: return ""
    # Enlève les crochets et guillemets si présents
    clean = re.sub(r"[\[\]']", "", cast_str)
    items = [i.strip() for i in clean.split(',')]
    return ", ".join(items[:3])

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
        cos_sim = cosine_similarity(count_matrix
