import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="La Triade", page_icon="🎬", layout="wide")

# Tes couleurs Letterboxd
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
    
    /* Centrage forcé */
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
        margin: 0 auto;
        display: block;
    }}

    /* Boîte de description personnalisée */
    .desc-container {{
        background-color: {DESC_BG};
        padding: 15px;
        border-radius: 8px;
        margin: 15px 10px;
        min-height: 100px;
    }}
    .desc-container p {{
        color: {DESC_TEXT} !important;
        font-size: 0.85rem;
        text-align: justify !important;
        line-height: 1.4;
        margin: 0;
    }}

    .credits-text {{
        font-size: 0.85rem;
        color: white !important;
        margin: 3px 0;
    }}
    .credits-label {{
        color: {DESC_TEXT} !important;
        font-weight: bold;
    }}

    /* Bouton Refresh Centré */
    .stButton {{
        display: flex;
        justify-content: center;
        padding-top: 20px;
    }}
    .stButton>button {{
        background-color: {HIGHLIGHT_ORANGE} !important;
        color: white !important;
        border-radius: 5px;
        padding: 10px 30px;
        border: none;
    }}
</style>
""", unsafe_allow_html=True)

# --- NETTOYAGE DES NOMS (Correction Woody Harrelson) ---
def fix_glued_names(text, limit=None):
    if not text or text == "" or str(text).lower() == "nan":
        return "Non spécifié"
    
    # On enlève les résidus de liste [ ' ' ]
    clean = re.sub(r"[\[\]'\"()]", "", str(text))
    
    # Correction des noms collés : insère un espace avant chaque Majuscule qui suit une minuscule
    # Exemple : WoodyHarrelson -> Woody Harrelson
    clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean)
    
    if limit:
        items = [i.strip() for i in clean.split(',')]
        return ", ".join(items[:limit])
    return clean

@st.cache_data
def load_data():
    df = pd.read_csv('Triade_ULTIMATE.csv')
    df.columns = [c.lower().strip() for c in df.columns]
    
    time_col = 'minute' if 'minute' in df.columns else 'runtime'
    expected = ['genres', 'keywords', 'director', 'cast', 'all_themes', 'description', 'name', 'category', time_col]
    
    for c in expected:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna('')

    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    
    df['search_label'] = df['name'].astype(str) + " (" + df['year'].astype(str).str.replace('.0', '', regex=False) + ")"
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
selected_labels = st.multiselect("Choisis tes films :", options=df['search_label'].sort_values().unique().tolist(), max_selections=4)

if selected_labels:
    results = get_combined_recs(selected_labels)
    st.write("---")
    col1, col2, col3 = st.columns(3)
    
    def draw_movie(category, cat_filter, streamlit_col, highlight_color):
        recs = results[results['category'].str.lower() == cat_filter.lower()]
        if len(recs) > st.session_state.offset:
            movie = recs.iloc[st.session_state.offset]
            url = str(movie['film_url'])
            img = str(movie['poster_url']) if movie['poster_url'] != "" else "https://via.placeholder.com/150x225"
            
            with streamlit_col:
                st.markdown(f"<h2 style='color:{highlight_color} !important;'>{category}</h2>", unsafe_allow_html=True)
                st.markdown(f'<a href="{url}" target="_blank"><img src="{img}" class="poster-img"></a>', unsafe_allow_html=True)
                st.markdown(f'<a href="{url}" target="_blank"><h3 style="margin-top:10px;">{movie["name"]}</h3></a>', unsafe_allow_html=True)
                
                year = str(movie['year'])[:4]
                time = f"{int(float(movie[minute_column]))} min" if movie[minute_column] else ""
                st.markdown(f"<p style='opacity:0.7;'>{year} | ⭐ {movie['rating']} | {time}</p>", unsafe_allow_html=True)
                
                st.markdown(f'<div class="desc-container"><p>{movie["description"][:280]}...</p></div>', unsafe_allow_html=True)
                
                # Crédits avec nettoyage des noms
                st.markdown(f"<p class='credits-text'><span class='credits-label'>Director:</span> {fix_glued_names(movie['director'])}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='credits-text'><span class='credits-label'>Cast:</span> {fix_glued_names(movie['cast'], 3)}</p>", unsafe_allow_html=True)
        else:
            streamlit_col.warning(f"Plus de {category}.")

    draw_movie("LA VALEUR SÛRE", "Blockbuster", col1, HIGHLIGHT_ORANGE)
    draw_movie("LE CHOIX CULTE", "Culte", col2, HIGHLIGHT_BLUE)
    draw_movie("LA PÉPITE", "Pépite", col3, HIGHLIGHT_GREEN)

    st.write("---")
    # Centrage forcé du bouton
    st.markdown("<center>", unsafe_allow_html=True)
    if st.button("🔄 Voir d'autres résultats"):
        st.session_state.offset += 1
        st.rerun()
    st.markdown("</center>", unsafe_allow_html=True)
else:
    st.info("Sélectionne des films pour commencer.")
