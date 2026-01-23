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
    
    /* Conteneur pour aligner Poster et Texte côte à côte */
    .movie-box {{
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 10px;
        min-height: 350px;
    }}
    
    .poster-side {{
        flex: 0 0 120px;
        margin-right: 15px;
    }}

    .info-side {{
        flex: 1;
        text-align: left;
    }}

    .poster-img {{
        width: 120px;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }}

    .description-text {{
        font-size: 0.8rem;
        color: #e0e0e0 !important;
        line-height: 1.3;
        text-align: justify;
        margin: 8px 0;
    }}

    .credits-text {{
        font-size: 0.75rem;
        color: #FFD700 !important;
        margin: 2px 0;
    }}

    /* Bouton Reload centré */
    .stButton {{
        display: flex;
        justify-content: center;
        width: 100%;
        margin-top: 20px;
    }}
    
    .stButton>button {{
        background-color: #FF4B4B !important;
        color: white !important;
        border-radius: 20px;
        font-weight: bold;
        padding: 10px 40px;
    }}
    
    a {{ text-decoration: none !important; }}
</style>
""", unsafe_allow_html=True)

# --- FONCTIONS DE NETTOYAGE ---
def clean_names(text, is_cast=False):
    if not text or text == "" or str(text).lower() == "nan": 
        return "Non spécifié"
    
    # 1. Enlever les crochets, guillemets et parenthèses
    clean = re.sub(r"[\[\]'\"()]", "", str(text))
    
    # 2. Séparer les noms collés (ex: WoodyHarrelson -> Woody Harrelson)
    # On cherche une minuscule suivie d'une majuscule
    clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean)
    
    if is_cast:
        # Nettoyage supplémentaire pour le cast (souvent séparé par des virgules cachées)
        items = [i.strip() for i in clean.split(',')]
        return ", ".join(items[:3])
    return clean

@st.cache_data
def load_data():
    df = pd.read_csv('Triade_ULTIMATE.csv')
    df.columns = [c.lower().strip() for c in df.columns]
    
    # On utilise 'minute' pour le temps et 'description' pour le texte
    runtime_col = 'minute' if 'minute' in df.columns else 'runtime'
    
    expected = ['genres', 'keywords', 'director', 'cast', 'all_themes', 'description', 'name', 'category', runtime_col]
    for c in expected:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna('')

    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    
    df['search_label'] = df['name'].astype(str) + " (" + df['year'].astype(str).str.replace('.0', '', regex=False) + ")"
    
    # Algo : Keywords prioritaires (x5) / Thèmes secondaires (x2)
    def create_soup(x):
        return ((str(x['keywords'])+" ")*5 + (str(x['all_themes'])+" ")*2 + (str(x['genres'])+" ")*2 + str(x['director'])+" "+str(x['cast'])).lower()

    df['soup'] = df.apply(create_soup, axis=1)
    df['time_val'] = df[runtime_col]
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
st.markdown("<h1 style='text-align:center;'>🎬 La Triade</h1>", unsafe_allow_html=True)
selected_labels = st.multiselect("Choisis tes films (max 4) :", options=df['search_label'].sort_values().unique().tolist(), max_selections=4)

if selected_labels:
    results = get_combined_recs(selected_labels)
    st.write("---")
    
    # On crée 3 colonnes pour mettre les 3 recommandations côte à côte
    col1, col2, col3 = st.columns(3)
    
    def draw_movie_card(category, cat_filter, streamlit_col):
        recs = results[results['category'].str.lower() == cat_filter.lower()]
        if len(recs) > st.session_state.offset:
            movie = recs.iloc[st.session_state.offset]
            url = str(movie['film_url'])
            img = str(movie['poster_url']) if movie['poster_url'] != "" else "https://via.placeholder.com/150x225"
            
            # Affichage du temps
            try:
                t = int(float(movie['time_val']))
                time_str = f"{t} min" if t > 0 else ""
            except:
                time_str = ""

            with streamlit_col:
                st.markdown(f"<h3 style='color:#FF4B4B !important; text-align:center;'>{category}</h3>", unsafe_allow_html=True)
                st.markdown(f'''
                    <div class="movie-box">
                        <div class="poster-side">
                            <a href="{url}" target="_blank">
                                <img src="{img}" class="poster-img">
                            </a>
                        </div>
                        <div class="info-side">
                            <a href="{url}" target="_blank">
                                <h4 style="margin:0; font-size:1.1rem; color:white;">{movie['name']}</h4>
                            </a>
                            <p style="font-size:0.8rem; margin:5px 0; color:#ccc !important;">
                                {str(movie['year'])[:4]} | ⭐ {movie['rating']} | {time_str}
                            </p>
                            <p class="description-text">{movie['description'][:180]}...</p>
                            <p class="credits-text"><b>Dir:</b> {clean_names(movie['director'])}</p>
                            <p class="credits-text"><b>Cast:</b> {clean_names(movie['cast'], True)}</p>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
        else:
            streamlit_col.warning(f"Plus de {category}.")

    draw_movie_card("LA VALEUR SÛRE", "Blockbuster", col1)
    draw_movie_card("LE CHOIX CULTE", "Culte", col2)
    draw_movie_card("LA PÉPITE", "Pépite", col3)

    st.write("---")
    # Centrage du bouton Reload
    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        if st.button("🔄 Voir d'autres résultats"):
            st.session_state.offset += 1
            st.rerun()
else:
    st.info("Entre un ou plusieurs films pour générer ta Triade.")
