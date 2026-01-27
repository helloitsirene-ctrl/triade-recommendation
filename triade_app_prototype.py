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
SEARCH_GRAY = "#e0e0e0"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');

    .stApp {{ background-color: {APP_BG}; }}
    
    /* Titre Principal */
    h1 {{ 
        font-family: 'Bebas Neue', cursive; 
        font-size: 5rem !important; 
        color: white !important; 
        text-align: center;
        margin-bottom: 20px;
        width: 100%;
    }}

    /* BARRE DE RECHERCHE ET TAGS (Gris Clair) */
    div[data-baseweb="select"] > div {{
        background-color: {SEARCH_GRAY} !important;
        color: #14181c !important;
    }}
    span[data-baseweb="tag"] {{
        background-color: #b0b0b0 !important;
        color: #14181c !important;
    }}
    
    /* BOUTON RELOAD À DROITE */
    .stButton > button {{
        background-color: {HIGHLIGHT_ORANGE} !important;
        color: white !important;
        border-radius: 8px;
        width: 50px;
        height: 40px;
        font-size: 1.5rem !important;
        border: none;
        margin-top: 0px;
    }}

    /* CENTRAGE ABSOLU DES COLONNES */
    [data-testid="column"] {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        text-align: center;
    }}

    .poster-img {{
        width: 160px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.6);
        margin: 0 auto;
        display: block;
    }}

    .movie-title {{
        font-family: 'Bebas Neue', cursive;
        font-size: 2rem;
        color: {HIGHLIGHT_BLUE} !important;
        margin-top: 15px;
        text-align: center !important;
        width: 100%;
    }}

    .info-line {{
        text-align: center;
        width: 100%;
        color: white;
        opacity: 0.8;
        font-size: 0.9rem;
        margin: 5px 0;
    }}

    .desc-container {{
        background-color: {DESC_BG};
        padding: 15px;
        border-radius: 8px;
        margin: 15px auto;
        width: 90%;
        height: auto; /* Permet à la boîte de grandir selon le texte */
        min-height: 100px; /* Optionnel : garde une taille mini pour l'alignement */
    }}
    
    .desc-container p {{
        color: {DESC_TEXT} !important;
        font-size: 0.88rem;
        text-align: justify !important;
        line-height: 1.4;
        margin: 0;
    }}


    .credits-text {{
        font-size: 1.0rem;
        color: white !important;
        margin: 2px 0;
        opacity: 0.9;
        text-align: center !important;
        width: 100%;
    }}

        /* Couleur de la barre du slider (la partie sélectionnée) */
        div[data-baseweb="slider"] > div > div > div {{
            background-color: #00c030 !important; /* Vert Letterboxd par défaut */
        }}

        /* Couleur du petit bouton rond (le "handle") */
        div[data-baseweb="slider"] > div > div > div > div {{
            background-color: #ffffff !important;
            border: 2px solid #00c030 !important;
        }}
        
        /* Couleur des étiquettes (labels) du select_slider pour la durée */
        div[data-testid="stWidgetLabel"] p {{
            color: #ffffff !important;
            font-weight: bold;
        }}
        
        /* Couleur des options textuelles du slider de durée */
        div[data-testid="stMarkdownContainer"] p {{
            font-size: 0.9rem;
        }}

        /* 1. Supprimer le rouge au focus de la recherche et des genres */
    div[data-baseweb="select"] > div {{
        border-color: transparent !important;
        box-shadow: none !important;
    }}
    
    /* Couleur des étiquettes de genres (tags) en bleu pour éviter le rouge */
    span[data-baseweb="tag"] {{
        background-color: {HIGHLIGHT_BLUE} !important;
        color: white !important;
    }}

    /* 2. Configuration des Sliders (Note et Durée) */
    
    /* La barre (le rail) du slider en gris comme la recherche */
    div[data-baseweb="slider"] > div > div > div {{
        background-color: {SEARCH_GRAY} !important;
    }}

    /* Le bouton du slider : Blanc avec contour bleu */
    div[data-baseweb="slider"] > div > div > div > div {{
        background-color: #ffffff !important;
        border: 2px solid {HIGHLIGHT_BLUE} !important;
    }}
    
    /* 3. Force TOUS les textes des filtres en BLANC */
    div[data-testid="stWidgetLabel"] p, 
    div[data-testid="stMarkdownContainer"] p,
    div[data-baseweb="slider"] div {{
        color: white !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }}


/* --- STYLE DE L'EXPANDER (FILTRES AVANCÉS) --- */
    
    /* Supprimer le fond blanc et la bordure rouge/claire */
    div[data-testid="stExpander"] {
        background-color: transparent !important;
        border: 1px solid #444c56 !important; /* Petit contour gris discret */
        border-radius: 8px;
    }

    /* Forcer le texte du titre de l'expander en blanc */
    div[data-testid="stExpander"] summary p {
        color: white !important;
    }

    /* --- SLIDER GRIS & TEXTE BLANC --- */

    /* La barre du slider en gris (couleur de tes titres/recherche) */
    div[data-baseweb="slider"] > div > div > div {
        background-color: #2c3440 !important; /* Gris Letterboxd */
    }

    /* Le bouton du slider (Handle) toujours propre */
    div[data-baseweb="slider"] > div > div > div > div {
        background-color: #ffffff !important;
        border: 2px solid {HIGHLIGHT_BLUE} !important;
    }

    /* Forcer TOUS les textes des filtres en blanc */
    div[data-testid="stWidgetLabel"] p, 
    div[data-testid="stMarkdownContainer"] p,
    div[data-baseweb="slider"] div {
        color: white !important;
    }
    
    </style>
""", unsafe_allow_html=True)

def clean_credits(text, is_cast=False):
    if not text or text == "" or str(text).lower() == "nan": return "Non spécifié"
    clean = re.sub(r"[\[\]'\"()]", "", str(text))
    if is_cast:
        items = [i.strip() for i in clean.split(',')]
        return ", ".join(items[:3])
    return clean

@st.cache_data
def load_data():
    df = pd.read_csv('Triade_ULTIMATE_CLEAN.csv')
    df.columns = [c.lower().strip() for c in df.columns]
    for c in ['director', 'cast', 'description', 'minute', 'name', 'category']:
        if c not in df.columns: df[c] = ""
    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    df['search_label'] = df['name'].astype(str) + " (" + df['year'].astype(str).str.replace('.0', '', regex=False) + ")"
    # Soupe pour le calcul
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

# --- INTERFACE ---
st.markdown("<h1>LA TRIADE</h1>", unsafe_allow_html=True)

# Ligne de recherche et bouton (On garde celle-ci !)
c_search, c_btn = st.columns([10, 1])

with c_search:
    selected_labels = st.multiselect(
        "RECHERCHE TES FILMS FAVORIS :",
        # Remplace 'display_name' par le nom exact de ta colonne (probablement search_label)
        options=df['search_label'].sort_values().unique().tolist(), 
        max_selections=4
    )

with c_btn:
    st.markdown("<br>", unsafe_allow_html=True) # Petit ajustement pour aligner le bouton
    if st.button("🔄"):
        st.session_state.offset += 1
        st.rerun()

# --- FILTRES AVANCÉS ---
with st.expander("Filtres avancés"):
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        min_rating = st.slider("Note Letterboxd minimum", 0.0, 5.0, 3.0, 0.5)
        duration_choice = st.select_slider(
            "Durée du film",
            options=["Peu importe", "Court", "Moyen", "Long"],
            value="Peu importe",
            help="Court: <90min | Moyen: 90-130min | Long: >130min"
        )
    with f_col2:
        all_genres = sorted(list(set([g.strip() for sublist in df['genres'].str.split(',') for g in sublist if g])))
        # On nettoie les crochets et les guillemets avant de découper
        all_genres = sorted(list(set([g.replace("[", "").replace("]", "").replace("'", "").strip() for sublist in df['genres'].dropna().str.split(',') for g in sublist if g])))
        selected_genres = st.multiselect("Genres spécifiques", all_genres)

# --- LOGIQUE DE GÉNÉRATION ---
if selected_labels:
    results = get_combined_recs(selected_labels)
    
    # Filtre Note
    results = results[pd.to_numeric(results['rating'], errors='coerce') >= min_rating]
    
    # Filtre Durée
    if duration_choice != "Peu importe":
        results['minute'] = pd.to_numeric(results['minute'], errors='coerce')
        if duration_choice == "Court":
            results = results[results['minute'] < 90]
        elif duration_choice == "Moyen":
            results = results[(results['minute'] >= 90) & (results['minute'] <= 130)]
        elif duration_choice == "Long":
            results = results[results['minute'] > 130]

    # Filtre Genres
    if selected_genres:
        genre_pattern = '|'.join(selected_genres)
        results = results[results['genres'].str.contains(genre_pattern, case=False, na=False)]

    st.write("---")
    
    if results.empty:
        st.warning("Oups ! Aucun film ne correspond à tes filtres. Essaie d'être moins exigeant.")
    else:
        col1, col2, col3 = st.columns(3)

        # Définition de la fonction d'affichage
        def draw_movie(category, cat_filter, streamlit_col, highlight_color):
            recs = results[results['category'].str.lower() == cat_filter.lower()]
            if len(recs) > st.session_state.offset:
                movie = recs.iloc[st.session_state.offset]
                url = str(movie['film_url']) if 'film_url' in movie else "#"
                img = str(movie['poster_url']) if movie['poster_url'] != "" else "https://via.placeholder.com/160x240"
                
                with streamlit_col:
                    st.markdown(f"<h2 style='color:{highlight_color} !important; text-align: center;'>{category}</h2>", unsafe_allow_html=True)
                    st.markdown(f'<a href="{url}" target="_blank"><img src="{img}" class="poster-img"></a>', unsafe_allow_html=True)
                    st.markdown(f'<a href="{url}" target="_blank" style="text-decoration:none;"><div class="movie-title">{movie["name"]}</div></a>', unsafe_allow_html=True)
                    
                    year = str(movie['year'])[:4]
                    try:
                        time = f"{int(float(movie['minute']))} min" if movie['minute'] != "" else ""
                    except: time = ""
                    st.markdown(f"<div class='info-line'>{year} | ⭐ {movie['rating']} {f'| {time}' if time else ''}</div>", unsafe_allow_html=True)
                    st.markdown(f'<div class="desc-container"><p>{movie["description"]}</p></div>', unsafe_allow_html=True)
                    st.markdown(f"<div class='credits-text'><b>Director:</b> {clean_credits(movie['director'])}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='credits-text'><b>Cast:</b> {clean_credits(movie['cast'], True)}</div>", unsafe_allow_html=True)

        # Appels de la fonction (Bien alignés sous le else)
        draw_movie("LA VALEUR SÛRE", "Blockbuster", col1, HIGHLIGHT_ORANGE)
        draw_movie("LE CHOIX CULTE", "Culte", col2, HIGHLIGHT_BLUE)
        draw_movie("LA PÉPITE", "Pépite", col3, HIGHLIGHT_GREEN)

# Description méthodologique en bas de page
st.write("---")
st.markdown("""
<div style="opacity: 1; font-size: 0.85rem; text-align: justify; padding: 20px;">
    <strong>Comment est générée votre Triade ?</strong><br>
    Chaque recommandation est issue d'une analyse sémantique croisant thèmes, keywords et équipe technique. 
    Les films sont segmentés selon leur impact sur la communauté Letterboxd :<br>
    • 🟠 <strong>La Valeur Sûre</strong> : Un large nombre de spectateurs, le classique que tout le monde a vu.<br>
    • 🔵 <strong>Le Choix Culte</strong> : Un film souvent moins connu du grand public, mais qui fédère les utilisateurs l'ayant vu.<br>
    • 🟢 <strong>La Pépite</strong> : Un film très peu connu, mais ceux qui l'ont découvert en sont tombé amoureux.
</div>
""", unsafe_allow_html=True)
