import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="La Triade", page_icon="🎬", layout="wide")

# CSS personnalisé pour le style
st.markdown("""
<style>
    a {text-decoration: none; color: #ff4b4b; font-weight: bold;}
    a:hover {text-decoration: underline;}
    img {border-radius: 15px; margin-bottom: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);}
    .stButton>button {width: 100%; border-radius: 20px;}
</style>
""", unsafe_allow_html=True)

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    df = pd.read_csv('Triade_ULTIMATE.csv')
    df.columns = [c.lower() for c in df.columns]
    
    # Réparation de l'année
    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    
    # Nettoyage des colonnes pour la soupe
    cols = ['genres', 'keywords', 'director', 'cast', 'all_themes']
    for c in cols:
        df[c] = df[c].fillna('').astype(str)

    # --- SOUPE AVEC PRIORITÉ THÈMES (x5) ET GENRES/KEYWORDS (x3) ---
    def create_soup(x):
        # On booste les thèmes au maximum (x5)
        themes = (x['all_themes'] + " ") * 5
        genres_key = (x['genres'] + " ") * 3 + (x['keywords'] + " ") * 3
        crew = x['director'] + " " + x['cast']
        return (themes + genres_key + crew).lower()

    df['soup'] = df.apply(create_soup, axis=1)
    return df

@st.cache_resource
def get_vectorizer(df):
    count = CountVectorizer(stop_words='english')
    return count.fit_transform(df['soup'])

# Initialisation
df = load_data()
count_matrix = get_vectorizer(df)
indices = pd.Series(df.index, index=df['name'].str.lower()).drop_duplicates()

# --- GESTION DE L'ÉTAT (Session State) ---
if 'offset' not in st.session_state:
    st.session_state.offset = 0
if 'last_selection' not in st.session_state:
    st.session_state.last_selection = ""

# --- LOGIQUE DE RECOMMANDATION ---
def get_recs(title):
    title_lower = title.lower()
    if title_lower not in indices: return None
    idx = indices[title_lower]
    if isinstance(idx, pd.Series): idx = idx.iloc[0]

    # Calcul de similarité à la volée (plus léger en RAM)
    cos_sim = cosine_similarity(count_matrix[idx], count_matrix)
    sim_scores = list(enumerate(cos_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # On exclut le film lui-même (index 0)
    movie_indices = [i[0] for i in sim_scores[1:101]]
    return df.iloc[movie_indices]

# --- INTERFACE ---
st.title("🎬 La Triade")

# Menu déroulant avec option "Vider" (en ajoutant un élément vide au début)
movie_list = [""] + sorted(df['name'].unique().tolist())
selected_movie = st.selectbox(
    "Quel film as-tu aimé ?", 
    movie_list, 
    index=0,
    help="Choisis un film pour obtenir ta Triade"
)

# Reset de l'offset si on change de film
if selected_movie != st.session_state.last_selection:
    st.session_state.offset = 0
    st.session_state.last_selection = selected_movie

if selected_movie != "":
    results = get_recs(selected_movie)
    
    if results is not None:
        st.write("---")
        col1, col2, col3 = st.columns(3)
        
        # Fonction d'affichage simplifiée
        def draw_movie(category, cat_filter, col):
            # On filtre par catégorie
            recs = results[results['category'].str.lower() == cat_filter.lower()]
            if len(recs) > st.session_state.offset:
                movie = recs.iloc[st.session_state.offset]
                with col:
                    st.subheader(category)
                    # Poster
                    if pd.notna(movie['poster_url']):
                        st.image(movie['poster_url'])
                    else:
                        st.image("https://via.placeholder.com/500x750?text=No+Poster")
                    
                    # Infos
                    st.markdown(f"#### {movie['name']}")
                    year = str(movie['year'])[:4] if 'year' in movie else "N/A"
                    rating = f"⭐ {movie['rating']}" if 'rating' in movie else ""
                    st.write(f"{year} | {rating}")
                    
                    # Lien Letterboxd
                    url = movie['film_url'] if 'film_url' in movie else "#"
                    st.markdown(f"[Voir sur Letterboxd ↗]({url})")
            else:
                col.warning(f"Plus de {category} disponible.")

        # Affichage des 3 colonnes
        draw_movie("La Valeur Sûre", "Blockbuster", col1)
        draw_movie("Le Choix Culte", "Culte", col2)
        draw_movie("La Pépite", "Pépite", col3)

        st.write("---")
        
        # Bouton Refresh
        _, center_col, _ = st.columns([1, 2, 1])
        if center_col.button("🔄 Pas convaincu ? Afficher d'autres résultats"):
            st.session_state.offset += 1
            st.rerun()
    else:
        st.error("Film non trouvé dans la base de données.")
else:
    st.info("Sélectionne un film ci-dessus pour commencer.")
