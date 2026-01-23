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
    h1, h2, h3, h4, h5, p, span {{ color: white !important; }}
    
    /* Conteneur horizontal pour chaque film */
    .movie-row {{
        display: flex;
        align-items: flex-start;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .poster-container {{
        flex: 0 0 150px;
        margin-right: 30px;
        text-align: center;
    }}

    .info-container {{
        flex: 1;
        text-align: left;
    }}

    .poster-img {{
        width: 150px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        transition: transform 0.2s;
    }}
    
    .poster-img:hover {{ transform: scale(1.03); }}

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

    /* Bouton Refresh centré */
    .stButton {{
        text-align: center;
    }}
    
    .stButton>button {{
        background-color: #FF4B4B !important;
        color: white !important;
        border-radius: 20px;
        padding: 12px 40px;
        font-weight: bold;
    }}
    
    a {{ text-decoration: none !important; }}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Triade_ULTIMATE.csv')
    except:
        st.error("Fichier Triade_ULTIMATE.csv introuvable.")
        st.stop()
        
    df.columns = [c.lower().strip() for c in df.columns]
    
    # --- SÉCURITÉ COLONNES ---
    expected = ['genres', 'keywords', 'director', 'cast', 'all_themes', 'description', 'name', 'category', 'runtime', 'poster_url', 'film_url', 'rating']
    for c in expected:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna('')

    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    
    df['search_label'] = df['name'].astype(str) + " (" + df['year'].astype(str).str.replace('.0', '', regex=False) + ")"
    
    # Algo : Keywords prioritaires (x5) / Thèmes secondaires (x2)
    def create_soup(x):
        return ((str(x['keywords']) + " ") * 5 + (str(x['all_themes']) + " ") * 2 + (str(x['genres']) + " ") * 2 + str(x['director']) + " " + str(x['cast'])).lower()

    df['soup'] = df.apply(create_soup, axis=1)
    return df

def format_runtime(minutes):
    try:
        m = int(float(minutes))
        if m <= 0: return "Durée inconnue"
        return f"{m // 60}h{m % 60:02d}"
    except:
        return ""

def clean_cast(cast_str):
    if not cast_str or cast_str == "": return "Non spécifié"
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
        
        cos_sim = cosine_similarity(count_matrix[actual_idx], count_matrix)[0]
        
        if all_sim_scores is None:
            all_sim_scores = cos_sim
        else:
            all_sim_scores += cos_sim
    
    sim_scores = sorted(list(enumerate(all_sim_scores)), key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores if i[0] not in selected_indices]
    return df.iloc[movie_indices[0:150]]

# --- INTERFACE ---
st.markdown("<h1 style='text-align:center;'>🎬 La Triade</h1>", unsafe_allow_html=True)

selected_labels = st.multiselect("Tape tes films favoris (max 4) :", options=df['search_label'].sort_values().unique().tolist(), max_selections=4)

if selected_labels:
    results = get_combined_recs(selected_labels)
    st.write("---")
    
    def display_movie_row(category, cat_filter):
        recs = results[results['category'].str.lower() == cat_filter.lower()]
        if len(recs) > st.session_state.offset:
            movie = recs.iloc[st.session_state.offset]
            url = str(movie['film_url']) if movie['film_url'] != "" else "#"
            img = str(movie['poster_url']) if movie['poster_url'] != "" else "https://via.placeholder.com/150x225?text=Pas+d'image"
            
            st.markdown(f'''
                <div class="movie-row">
                    <div class="poster-container">
                        <a href="{url}" target="_blank">
                            <img src="{img}" class="poster-img">
                        </a>
                    </div>
                    <div class="info-container">
                        <h2 style="color:#FF4B4B !important; margin:0; font-size:1.5rem;">{category}</h2>
                        <a href="{url}" target="_blank">
                            <h3 style="margin:5px 0 0 0;">{movie['name']}</h3>
                        </a>
                        <p style="font-size:1rem; margin-top:5px; color:#ccc !important;">
                            {str(movie['year'])[:4]} | ⭐ {movie['rating']} | ⏱️ {format_runtime(movie['runtime'])}
                        </p>
                        <p class="description-text">{movie['description']}</p>
                        <p class="credits-text"><b>Director:</b> {movie['director']}</p>
                        <p class="credits-text"><b>Cast:</b> {clean_cast(movie['cast'])}</p>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.warning(f"Plus de {category} disponible.")

    display_movie_row("LA VALEUR SÛRE", "Blockbuster")
    display_movie_row("LE CHOIX CULTE", "Culte")
    display_movie_row("LA PÉPITE", "Pépite")

    st.write("---")
    _, mid_col, _ = st.columns([1, 1, 1])
    with mid_col:
        if st.button("🔄 Voir d'autres résultats"):
            st.session_state.offset += 1
            st.rerun()
else:
    st.info("Ajoute un ou plusieurs films pour générer ta Triade personnalisée.")
