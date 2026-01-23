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
    h1, h2, h3, h4, h5, p, span {{ color: white !important; text-align: center; }}
    
    /* Style des colonnes */
    [data-testid="column"] {{
        background-color: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 5px;
    }}
    
    .poster-img {{
        width: 140px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        margin-bottom: 10px;
    }}

    .description-text {{
        font-size: 0.85rem;
        color: #e0e0e0 !important;
        line-height: 1.3;
        text-align: justify;
        margin: 10px 0;
    }}

    .credits-text {{
        font-size: 0.8rem;
        color: #FFD700 !important;
        text-align: center;
        margin: 2px 0;
    }}

    .stButton {{ text-align: center; }}
    .stButton>button {{
        background-color: #FF4B4B !important;
        color: white !important;
        border-radius: 20px;
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)

# --- FONCTIONS DE NETTOYAGE ---
def clean_human_names(text, is_cast=False):
    if not text or text == "" or text == "nan": return "Non spécifié"
    # Enlever crochets, guillemets et parenthèses
    clean = re.sub(r"[\[\]'\"()]", "", str(text))
    # Si les noms sont collés (WoodyHarrelson), on tente de séparer par les majuscules
    # On ajoute un espace avant chaque majuscule qui suit une minuscule
    clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean)
    
    if is_cast:
        # Séparer par virgule et prendre les 3 premiers
        items = [i.strip() for i in clean.split(',')]
        return ", ".join(items[:3])
    return clean

def format_runtime(minutes):
    try:
        m = int(float(minutes))
        return f"{m // 60}h{m % 60:02d}" if m > 0 else ""
    except: return ""

@st.cache_data
def load_data():
    df = pd.read_csv('Triade_ULTIMATE.csv')
    df.columns = [c.lower().strip() for c in df.columns]
    for c in ['genres', 'keywords', 'director', 'cast', 'all_themes', 'description', 'name', 'category', 'runtime']:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna('')
    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    df['search_label'] = df['name'].astype(str) + " (" + df['year'].astype(str).str.replace('.0', '', regex=False) + ")"
    df['soup'] = df.apply(lambda x: ((str(x['keywords']) + " ") * 5 + (str(x['all_themes']) + " ") * 2 + (str(x['genres']) + " ") * 2 + str(x['director']) + " " + str(x['cast'])).lower(), axis=1)
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
    return df.iloc[movie_indices[0:150]]

# --- INTERFACE ---
st.markdown("<h1>🎬 La Triade</h1>", unsafe_allow_html=True)
selected_labels = st.multiselect("Choisis tes films (max 4) :", options=df['search_label'].sort_values().unique().tolist(), max_selections=4)

if selected_labels:
    results = get_combined_recs(selected_labels)
    st.write("---")
    col1, col2, col3 = st.columns(3)
    
    def draw_movie_col(category, cat_filter, streamlit_col):
        recs = results[results['category'].str.lower() == cat_filter.lower()]
        if len(recs) > st.session_state.offset:
            movie = recs.iloc[st.session_state.offset]
            url = str(movie['film_url'])
            img = str(movie['poster_url']) if movie['poster_url'] != "" else "https://via.placeholder.com/150x225"
            
            with streamlit_col:
                st.markdown(f"<h2 style='color:#FF4B4B !important; font-size:1.3rem;'>{category}</h2>", unsafe_allow_html=True)
                st.markdown(f'''
                    <a href="{url}" target="_blank" style="text-decoration:none;">
                        <img src="{img}" class="poster-img"><br>
                        <h4 style="margin:5px 0; font-size:1.1rem;">{movie['name']}</h4>
                    </a>
                ''', unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:0.85rem; opacity:0.8;'>{str(movie['year'])[:4]} | ⭐ {movie['rating']} | {format_runtime(movie['runtime'])}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='description-text'>{movie['description'][:250]}...</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='credits-text'><b>Dir:</b> {clean_human_names(movie['director'])}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='credits-text'><b>Cast:</b> {clean_human_names(movie['cast'], True)}</p>", unsafe_allow_html=True)
        else:
            streamlit_col.warning(f"Plus de {category}.")

    draw_movie_col("LA VALEUR SÛRE", "Blockbuster", col1)
    draw_movie_col("LE CHOIX CULTE", "Culte", col2)
    draw_movie_col("LA PÉPITE", "Pépite", col3)

    st.write("---")
    _, mid, _ = st.columns([1, 1, 1])
    if mid.button("🔄 Voir d'autres résultats"):
        st.session_state.offset += 1
        st.rerun()
else:
    st.info("Sélectionne des films pour afficher ta Triade.")
