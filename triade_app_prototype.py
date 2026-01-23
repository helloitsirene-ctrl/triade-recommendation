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
    
    /* Centrage forcé dans les colonnes */
    [data-testid="column"] {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        text-align: center;
    }}
    
    .poster-img {{
        width: 150px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.6);
        margin-bottom: 10px;
    }}

    .description-text {{
        font-size: 0.85rem;
        color: #e0e0e0 !important;
        line-height: 1.4;
        text-align: justify;
        margin: 15px 0;
        padding: 0 10px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px;
    }}

    .credits-text {{
        font-size: 0.8rem;
        color: #FFD700 !important;
        margin: 2px 0;
    }}

    /* Centrage du bouton */
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
</style>
""", unsafe_allow_html=True)

# --- FONCTION NETTOYAGE DIRECT ---
def clean_raw_data(text, max_items=None):
    if not text or text == "" or str(text).lower() == "nan":
        return "Non spécifié"
    # Supprime les résidus de format liste ['Nom']
    clean = re.sub(r"[\[\]'\"()]", "", str(text))
    
    if max_items:
        # Pour le cast, on prend les X premiers
        items = [i.strip() for i in clean.split(',')]
        return ", ".join(items[:max_items])
    return clean

@st.cache_data
def load_data():
    df = pd.read_csv('Triade_ULTIMATE.csv')
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Sécurité sur les noms de colonnes
    runtime_col = 'minute' if 'minute' in df.columns else 'runtime'
    
    expected = ['genres', 'keywords', 'director', 'cast', 'all_themes', 'description', 'name', 'category', runtime_col]
    for c in expected:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna('')

    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    
    df['search_label'] = df['name'].astype(str) + " (" + df['year'].astype(str).str.replace('.0', '', regex=False) + ")"
    
    # Soupe pour l'IA (Keywords x5, Thèmes x2)
    df['soup'] = df.apply(lambda x: ((str(x['keywords'])+" ")*5 + (str(x['all_themes'])+" ")*2 + (str(x['genres'])+" ")*2 + str(x['director'])+" "+str(x['cast'])).lower(), axis=1)
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
            
            # Temps
            runtime_val = movie['minute'] if 'minute' in movie else movie.get('runtime', '')
            time_display = f"{int(float(runtime_val))} min" if runtime_val and str(runtime_val).strip() != "" else ""

            with streamlit_col:
                st.markdown(f"<h2 style='color:#FF4B4B !important; font-size:1.4rem;'>{category}</h2>", unsafe_allow_html=True)
                st.markdown(f'''
                    <a href="{url}" target="_blank" style="text-decoration:none;">
                        <img src="{img}" class="poster-img">
                        <h4 style="margin:5px 0; font-size:1.2rem; line-height:1.2; color:white;">{movie['name']}</h4>
                    </a>
                    <p style="font-size:0.9rem; opacity:0.8;">{str(movie['year'])[:4]} | ⭐ {movie['rating']} {f"| {time_display}" if time_display else ""}</p>
                    <div class="description-text">{str(movie['description'])[:300]}...</div>
                    <p class="credits-text"><b>Director:</b> {clean_raw_data(movie['director'])}</p>
                    <p class="credits-text"><b>Cast:</b> {clean_raw_data(movie['cast'], 3)}</p>
                ''', unsafe_allow_html=True)
        else:
            streamlit_col.warning(f"Plus de {category}.")

    draw_movie_col("LA VALEUR SÛRE", "Blockbuster", col1)
    draw_movie_col("LE CHOIX CULTE", "Culte", col2)
    draw_movie_col("LA PÉPITE", "Pépite", col3)

    st.write("---")
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("🔄 Voir d'autres résultats"):
            st.session_state.offset += 1
            st.rerun()
else:
    st.info("Sélectionne tes films pour générer ta Triade.")
