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

# Seuils des catégories + zone de chevauchement (±10%)
BLOCKBUSTER_THRESHOLD = 450_000
CULTE_LOW_THRESHOLD = 50_000
OVERLAP_MARGIN = 0.10  # 10% de chevauchement aux frontières

# Stop words FR + EN (le vectorizer ne supporte qu'une langue à la fois nativement)
FRENCH_STOP_WORDS = [
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "mais",
    "donc", "car", "ni", "que", "qui", "quoi", "dont", "où", "ce", "cet",
    "cette", "ces", "son", "sa", "ses", "leur", "leurs", "mon", "ma", "mes",
    "ton", "ta", "tes", "notre", "nos", "votre", "vos", "il", "elle", "ils",
    "elles", "on", "nous", "vous", "je", "tu", "me", "te", "se", "lui",
    "y", "en", "à", "au", "aux", "dans", "par", "pour", "sur", "avec",
    "sans", "sous", "vers", "chez", "entre", "est", "sont", "été", "être",
    "avoir", "fait", "faire", "plus", "moins", "très", "bien", "aussi",
    "tout", "tous", "toute", "toutes", "comme", "si", "ne", "pas", "ni",
]

ENGLISH_STOP_WORDS = [
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "this", "that", "these", "those", "i", "you", "he", "she", "it",
    "we", "they", "them", "their", "his", "her", "its", "my", "your",
]

STOP_WORDS = list(set(FRENCH_STOP_WORDS + ENGLISH_STOP_WORDS))

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

    /* BARRE DE RECHERCHE ET TAGS */
    div[data-baseweb="select"] > div {{
        background-color: {SEARCH_GRAY} !important;
        color: #14181c !important;
        border-color: transparent !important;
        box-shadow: none !important;
    }}

    span[data-baseweb="tag"] {{
        background-color: {HIGHLIGHT_BLUE} !important;
        color: white !important;
    }}

    /* BOUTON RELOAD */
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

    /* CENTRAGE COLONNES */
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
        height: auto;
        min-height: 100px;
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

    /* SLIDERS : rail gris, handle blanc à contour bleu */
    div[data-baseweb="slider"] > div > div > div {{
        background-color: {SEARCH_GRAY} !important;
    }}

    div[data-baseweb="slider"] > div > div > div > div {{
        background-color: #ffffff !important;
        border: 2px solid {HIGHLIGHT_BLUE} !important;
    }}

    /* TEXTES BLANCS pour les labels des filtres (pas l'intérieur du slider) */
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stMarkdownContainer"] p {{
        color: white !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }}

    /* Valeurs min/max sous le slider — texte blanc visible sur fond sombre */
    div[data-testid="stSliderTickBarMin"],
    div[data-testid="stSliderTickBarMax"] {{
        color: white !important;
    }}

    /* Valeur courante du slider (la bulle au-dessus du handle) — texte sombre sur fond clair */
    div[data-baseweb="slider"] [role="slider"] + div,
    div[data-baseweb="slider"] div[data-testid="stThumbValue"] {{
        color: {APP_BG} !important;
    }}

    /* Tooltip du point d'interrogation (help) — fond sombre, texte blanc */
    div[data-baseweb="tooltip"] {{
        background-color: {DESC_BG} !important;
        color: white !important;
    }}

    div[data-baseweb="tooltip"] div {{
        background-color: {DESC_BG} !important;
        color: white !important;
    }}

    /* EXPANDER "Filtres avancés" — fond sombre, texte blanc */
    div[data-testid="stExpander"] {{
        background-color: {DESC_BG} !important;
        border: 1px solid #2c3440 !important;
        border-radius: 8px !important;
    }}

    /* Header de l'expander (la partie cliquable) */
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] details > summary {{
        background-color: {DESC_BG} !important;
        color: white !important;
    }}

    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span {{
        color: white !important;
        font-weight: bold !important;
    }}

    /* Contenu de l'expander une fois ouvert */
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        background-color: {DESC_BG} !important;
    }}

    /* Icône chevron de l'expander en blanc */
    div[data-testid="stExpander"] svg {{
        fill: white !important;
    }}
</style>
""", unsafe_allow_html=True)


def clean_credits(text, is_cast=False):
    if not text or text == "" or str(text).lower() == "nan":
        return "Non spécifié"
    clean = re.sub(r"[\[\]'\"()]", "", str(text))
    if is_cast:
        items = [i.strip() for i in clean.split(',')]
        return ", ".join(items[:3])
    return clean


def clean_genre_string(s):
    """Nettoie une chaîne de genres type "['Drama', 'Thriller']" -> liste propre."""
    if pd.isna(s) or s == "":
        return []
    cleaned = re.sub(r"[\[\]']", "", str(s))
    return [g.strip() for g in cleaned.split(',') if g.strip()]


@st.cache_data
def load_data():
    df = pd.read_csv('Triade_ULTIMATE_CLEAN.csv')
    df.columns = [c.lower().strip() for c in df.columns]

    # Colonnes obligatoires
    for c in ['director', 'cast', 'description', 'minute', 'name', 'category',
              'keywords', 'all_themes', 'genres', 'watches', 'rating']:
        if c not in df.columns:
            df[c] = ""

    # Année
    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]

    df['search_label'] = (
        df['name'].astype(str) + " (" +
        df['year'].astype(str).str.replace('.0', '', regex=False) + ")"
    )

    # Soupe pondérée — vectorisée (beaucoup plus rapide qu'apply)
    df['soup'] = (
        (df['keywords'].fillna('') + ' ') * 5 +
        (df['all_themes'].fillna('') + ' ') * 2 +
        (df['genres'].fillna('') + ' ') * 2 +
        df['director'].fillna('') + ' ' +
        df['cast'].fillna('').astype(str).str.lower()
    )

    # Catégories avec zone de chevauchement
    # Un film en zone frontière appartient à 2 catégories
    df['watches_num'] = pd.to_numeric(df['watches'], errors='coerce').fillna(0)

    bb_low = BLOCKBUSTER_THRESHOLD * (1 - OVERLAP_MARGIN)   # 405k
    bb_high = BLOCKBUSTER_THRESHOLD * (1 + OVERLAP_MARGIN)  # 495k
    pep_low = CULTE_LOW_THRESHOLD * (1 - OVERLAP_MARGIN)    # 45k
    pep_high = CULTE_LOW_THRESHOLD * (1 + OVERLAP_MARGIN)   # 55k

    def assign_categories(row):
        """Retourne la liste des catégories applicables à ce film."""
        cats = set()
        # On garde la catégorie pré-calculée (source de vérité)
        if isinstance(row['category'], str) and row['category']:
            cats.add(row['category'].strip().lower())

        # On ajoute les catégories voisines si on est en zone de chevauchement
        w = row['watches_num']
        if bb_low <= w <= bb_high:
            cats.update(['blockbuster', 'culte'])
        if pep_low <= w <= pep_high:
            cats.update(['culte', 'pépite'])

        return list(cats)

    df['categories'] = df.apply(assign_categories, axis=1)
    return df


@st.cache_resource
def build_model(soup_series):
    """Construit la matrice de comptage et l'index — caché pour ne calculer qu'une fois."""
    count_matrix = CountVectorizer(stop_words=STOP_WORDS).fit_transform(soup_series)
    return count_matrix


df = load_data()
count_matrix = build_model(df['soup'])
indices = pd.Series(df.index, index=df['search_label']).drop_duplicates()

# --- ÉTAT DE SESSION ---
if 'offset' not in st.session_state:
    st.session_state.offset = 0
if 'last_selection' not in st.session_state:
    st.session_state.last_selection = []


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

c_search, c_btn = st.columns([10, 1])

with c_search:
    selected_labels = st.multiselect(
        "RECHERCHE TES FILMS FAVORIS :",
        options=df['search_label'].sort_values().unique().tolist(),
        max_selections=4
    )

with c_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄"):
        st.session_state.offset += 1
        st.rerun()

# Reset offset si la sélection change
if selected_labels != st.session_state.last_selection:
    st.session_state.offset = 0
    st.session_state.last_selection = selected_labels

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
        # Parsing propre des genres, une seule fois
        all_genres_set = set()
        for genre_str in df['genres'].dropna():
            all_genres_set.update(clean_genre_string(genre_str))
        all_genres = sorted(all_genres_set)

        # Affichage avec majuscule, mais on garde la valeur originale pour le filtrage
        selected_genres_display = st.multiselect(
            "Genres spécifiques",
            options=all_genres,
            format_func=lambda g: g.capitalize()
        )
        selected_genres = selected_genres_display

        # Filtre par décennie
        decade_options = ["Avant 1950", "1950s", "1960s", "1970s", "1980s",
                          "1990s", "2000s", "2010s", "2020s"]
        selected_decades = st.multiselect("Décennie", options=decade_options)

# --- LOGIQUE DE GÉNÉRATION ---
if selected_labels:
    results = get_combined_recs(selected_labels)

    # Filtre Note
    results = results[pd.to_numeric(results['rating'], errors='coerce') >= min_rating]

    # Filtre Durée
    if duration_choice != "Peu importe":
        results = results.copy()
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

    # Filtre Décennie
    if selected_decades:
        results = results.copy()
        years_numeric = pd.to_numeric(results['year'], errors='coerce')

        decade_ranges = {
            "Avant 1950": (0, 1950),
            "1950s": (1950, 1960),
            "1960s": (1960, 1970),
            "1970s": (1970, 1980),
            "1980s": (1980, 1990),
            "1990s": (1990, 2000),
            "2000s": (2000, 2010),
            "2010s": (2010, 2020),
            "2020s": (2020, 2030),
        }

        decade_mask = pd.Series(False, index=results.index)
        for decade in selected_decades:
            low, high = decade_ranges[decade]
            decade_mask |= (years_numeric >= low) & (years_numeric < high)
        results = results[decade_mask]

    st.write("---")

    if results.empty:
        st.warning("Oups ! Aucun film ne correspond à tes filtres. Essaie d'être moins exigeant.")
    else:
        col1, col2, col3 = st.columns(3)

        def draw_movie(category_label, cat_filter, streamlit_col, highlight_color):
            # Filtre via la liste de catégories (zone de chevauchement)
            recs = results[results['categories'].apply(lambda c: cat_filter.lower() in c)]

            if len(recs) == 0:
                with streamlit_col:
                    st.markdown(
                        f"<h2 style='color:{highlight_color} !important; text-align: center;'>{category_label}</h2>",
                        unsafe_allow_html=True
                    )
                    st.info("Aucun film dans cette catégorie pour ces filtres.")
                return

            # Wrap-around : si offset trop grand, on reboucle
            movie = recs.iloc[st.session_state.offset % len(recs)]
            url = str(movie['film_url']) if 'film_url' in movie else "#"
            img = str(movie['poster_url']) if movie['poster_url'] != "" else "https://via.placeholder.com/160x240"

            with streamlit_col:
                st.markdown(
                    f"<h2 style='color:{highlight_color} !important; text-align: center;'>{category_label}</h2>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<a href="{url}" target="_blank"><img src="{img}" class="poster-img"></a>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<a href="{url}" target="_blank" style="text-decoration:none;"><div class="movie-title">{movie["name"]}</div></a>',
                    unsafe_allow_html=True
                )

                year = str(movie['year'])[:4]
                try:
                    time = f"{int(float(movie['minute']))} min" if movie['minute'] != "" else ""
                except (ValueError, TypeError):
                    time = ""
                st.markdown(
                    f"<div class='info-line'>{year} | ⭐ {movie['rating']} {f'| {time}' if time else ''}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="desc-container"><p>{movie["description"]}</p></div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div class='credits-text'><b>Director:</b> {clean_credits(movie['director'])}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div class='credits-text'><b>Cast:</b> {clean_credits(movie['cast'], True)}</div>",
                    unsafe_allow_html=True
                )

        draw_movie("LA VALEUR SÛRE", "Blockbuster", col1, HIGHLIGHT_ORANGE)
        draw_movie("LE CHOIX CULTE", "Culte", col2, HIGHLIGHT_BLUE)
        draw_movie("LA PÉPITE", "Pépite", col3, HIGHLIGHT_GREEN)

# Description méthodologique
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
