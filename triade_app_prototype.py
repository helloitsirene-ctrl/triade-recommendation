import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="La Triade", page_icon="🎬", layout="wide")

# Couleurs personnalisées
APP_BG = "#14181c"
DESC_BG = "#242c34"
DESC_TEXT = "#93a0ae"
HIGHLIGHT_ORANGE = "#f59331"
HIGHLIGHT_BLUE = "#3fb8ef"
HIGHLIGHT_GREEN = "#00ba2e"

st.markdown(f"""
<style>
    .stApp {{ background-color: {APP_BG}; }}
    h1, h2, h3, h4, h5, p, span, label {{ color: white !important; text-align: center; }}
    
    /* Centrage forcé des colonnes et posters */
    [data-testid="column"] {{
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }}
    
    .poster-img {{
        width: 150px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.6);
        margin: 0 auto 10px auto;
        display: block;
    }}

    /* Boîte de description avec tes couleurs */
    .desc-container {{
        background-color: {DESC_BG};
        color: {DESC_TEXT} !important;
        padding: 15px;
        border-radius: 8px;
        font-size: 0.85rem;
        text-align: justify;
        line-height: 1.4;
        margin: 15px 10px;
        min-height: 120px;
    }}
    .desc-container p {{
        color: {DESC_TEXT} !important;
        text-align: justify !important;
        margin: 0;
    }}

    .credits-text {{
        font-size: 0.8rem;
        color: #ffffff !important;
        margin: 2px 0;
        opacity: 0.9;
    }}
    .credits-text b {{ color: #FFD700 !important; }}

    /* Bouton Refresh Centré */
    .stButton {{
        display: flex;
        justify-content: center;
        width: 100%;
        margin-top: 30px;
    }}
    .stButton>button {{
        background-color: #ff4b4b !important;
        color: white !important;
        border-radius: 20px;
        padding: 10px 40px;
        font-weight: bold;
        border: none;
    }}
    
    a {{ text-decoration: none !important; }}
</style>
""", unsafe_allow_html=True)

# --- FONCTION NETTOYAGE DES NOMS ---
def clean_names_properly(text, max_items=None):
    if not text or text == "" or str(text).lower() == "nan":
        return "Non spécifié"
    
    # 1. Enlever les crochets et guillemets : ['WoodyHarrelson'] -> WoodyHarrelson
    clean = re.sub(r"[\[\]'\"()]", "", str(text))
    
    # 2. Séparer les noms collés (ex: WoodyHarrelson -> Woody Harrelson)
    # On insère un espace entre une minuscule et une majuscule
    clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean)
    
    if max_items:
        items = [i.strip() for i in clean.split(',')]
        return ", ".join(items[:max_items])
    return clean

@st.cache_data
def load_data():
    df = pd.read_csv('Triade_ULTIMATE.csv')
    df.columns = [c.lower().strip() for c in df.columns]
    
    # On vérifie la colonne minute
    time_col = 'minute' if 'minute' in df.columns else 'runtime'
    
    # Sécurité colonnes
    expected = ['genres', 'keywords', 'director', 'cast', 'all_themes', 'description', 'name', 'category', time_col]
    for c in expected:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna('')

    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    
    df['search_label'] = df['name'].astype(str) + " (" + df['year'].astype(str).str.replace('.0', '', regex=False) + ")"
    
    # Algo : Keywords x5 / Themes x2
    df['soup'] = df.apply(lambda x: ((str(x['keywords'])+" ")*5 + (str(x['all_themes'])+" ")*2 + (str(x['genres'])+" ")*2 + str(x['director'])+" "+str(x['cast'])).lower(), axis=1)
    return df, time_col

df, minute_column = load_data()
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
selected_labels = st.multiselect("Recherche tes films favoris :", options=df['search_label'].sort_values().unique().tolist(), max_selections=4)

if selected_labels:
    results = get_combined_recs(selected_labels)
    st.write("---")
    col1, col2, col3 = st.columns(3)
    
    def draw_movie_card(category, cat_filter, streamlit_col, highlight_color):
        recs = results[results['category'].str.lower() == cat_filter.lower()]
        if len(recs) > st.session_state.offset:
            movie = recs.iloc[st.session_state.offset]
            url = str(movie['film_url'])
            img = str(movie['poster_url']) if movie['poster_url'] != "" else "https://via.placeholder.com/150x225?text=Poster"
            
            # Temps
            t_val = str(movie[minute_column]).strip()
            time_display = f"{int(float(t_val))} min" if t_val and t_val != "" and t_val != "0" else ""

            with streamlit_col:
                # Titre de catégorie avec couleur spécifique
                st.markdown(f"<h2 style='color:{highlight_color} !important; font-size:1.4rem;'>{category}</h2>", unsafe_allow_html=True)
                
                # Bloc cliquable Poster + Titre
                st.markdown(f'''
                    <a href="{url}" target="_blank">
                        <img src="{img}" class="poster-img">
                        <h3 style="margin:5px 0; font-size:1.2rem; line-height:1.2; color:white !important;">{movie['name']}</h3>
                    </a>
                ''', unsafe_allow_html=True)
                
                # Infos
                st.markdown(f"<p style='font-size:0.9rem; opacity:0.8;'>{str(movie['year'])[:4]} | ⭐ {movie['rating']} {f'| {time_display}' if time_display else ''}</p>", unsafe_allow_html=True)
                
                # Description Box
                st.markdown(f'''
                    <div class="desc-container">
                        <p>{str(movie['description'])[:300]}...</p>
                    </div>
                ''', unsafe_allow_html=True)
                
                # Crédits avec nettoyage des noms collés
                st.markdown(f"<p class='credits-text'><b>Director:</b> {clean_names_properly(movie['director'])}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='credits-text'><b>Cast:</b> {clean_names_properly(movie['cast'], 3)}</p>", unsafe_allow_html=True)
        else:
            streamlit_col.warning(f"Plus de {category}.")

    draw_movie_card("LA VALEUR SÛRE", "Blockbuster", col1, HIGHLIGHT_ORANGE)
    draw_movie_card("LE CHOIX CULTE", "Culte", col2, HIGHLIGHT_BLUE)
    draw_movie_card("LA PÉPITE", "Pépite", col3, HIGHLIGHT_GREEN)

    st.write("---")
    # Centrage du bouton Refresh
    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        if st.button("🔄 Voir d'autres résultats"):
            st.session_state.offset += 1
            st.rerun()
else:
    st.info("Sélectionne des films pour découvrir ta Triade.")
