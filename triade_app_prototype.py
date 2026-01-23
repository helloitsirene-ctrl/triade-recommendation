import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION PAGE & DESIGN ---
st.set_page_config(page_title="La Triade", page_icon="🎬", layout="wide")

# CSS pour le fond, le texte et le centrage
st.markdown(f"""
<style>
    .stApp {{
        background-color: #445566;
        color: white;
    }}
    h1, h2, h3, h4, p, span, label {{
        color: white !important;
        text-align: center;
    }}
    .stMultiSelect div div div div {{
        color: black !important;
    }}
    img {{
        border-radius: 15px;
        display: block;
        margin-left: auto;
        margin-right: auto;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }}
    div[data-testid="column"] {{
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }}
    .stButton>button {{
        border-radius: 20px;
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 10px 24px;
    }}
</style>
""", unsafe_allow_html=True)

# --- CHARGEMENT ---
@st.cache_data
def load_data():
    df = pd.read_csv('Triade_ULTIMATE.csv')
    df.columns = [c.lower() for c in df.columns]
    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    
    cols = ['genres', 'keywords', 'director', 'cast', 'all_themes']
    for c in cols: df[c] = df[c].fillna('').astype(str)

    def create_soup(x):
        # Priorité maximale aux thèmes (x5)
        return ((x['all_themes'] + " ") * 5 + (x['genres'] + " ") * 3 + (x['keywords'] + " ") * 3 + x['director'] + " " + x['cast']).lower()

    df['soup'] = df.apply(create_soup, axis=1)
    return df

@st.cache_resource
def get_vectorizer(df):
    count = CountVectorizer(stop_words='english')
    return count.fit_transform(df['soup'])

df = load_data()
count_matrix = get_vectorizer(df)
indices = pd.Series(df.index, index=df['name'].str.lower()).drop_duplicates()

# --- ETAT DE LA SESSION ---
if 'offset' not in st.session_state: st.session_state.offset = 0

# --- MOTEUR MULTI-FILMS ---
def get_combined_recs(titles):
    all_sim_scores = None
    
    for title in titles:
        idx = indices[title.lower()]
        if isinstance(idx, pd.Series): idx = idx.iloc[0]
        
        cos_sim = cosine_similarity(count_matrix[idx], count_matrix)[0]
        if all_sim_scores is None:
            all_sim_scores = cos_sim
        else:
            all_sim_scores += cos_sim # On additionne les scores pour trouver les films proches de TOUS les choix
            
    sim_scores = list(enumerate(all_sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Exclure les films sélectionnés
    selected_indices = [indices[t.lower()] for t in titles]
    movie_indices = [i[0] for i in sim_scores if i[0] not in selected_indices]
    
    return df.iloc[movie_indices[0:150]]

# --- INTERFACE ---
st.markdown("<h1 style='font-size: 3rem;'>🎬 La Triade</h1>", unsafe_allow_html=True)

selected_movies = st.multiselect(
    "Choisis jusqu'à 4 films que tu as aimés :",
    options=sorted(df['name'].unique().tolist()),
    max_selections=4
)

if selected_movies:
    results = get_combined_recs(selected_movies)
    
    st.write("---")
    col1, col2, col3 = st.columns(3)
    
    def draw_movie(category, cat_filter, col):
        recs = results[results['category'].str.lower() == cat_filter.lower()]
        if len(recs) > st.session_state.offset:
            movie = recs.iloc[st.session_state.offset]
            with col:
                st.markdown(f"### {category}")
                if pd.notna(movie['poster_url']):
                    st.image(movie['poster_url'], width=250)
                else:
                    st.image("https://via.placeholder.com/500x750?text=No+Poster", width=250)
                
                st.markdown(f"#### {movie['name']}")
                year = str(movie['year'])[:4] if 'year' in movie else "N/A"
                rating = f"⭐ {movie['rating']}" if 'rating' in movie else ""
                st.write(f"{year} | {rating}")
                
                url = movie['film_url'] if 'film_url' in movie else "#"
                st.markdown(f"[Voir sur Letterboxd ↗]({url})")
        else:
            col.warning("Plus de résultats.")

    draw_movie("La Valeur Sûre", "Blockbuster", col1)
    draw_movie("Le Choix Culte", "Culte", col2)
    draw_movie("La Pépite", "Pépite", col3)

    st.write("---")
    _, center_col, _ = st.columns([1, 2, 1])
    if center_col.button("🔄 Pas convaincu ? Afficher d'autres résultats"):
        st.session_state.offset += 1
        st.rerun()
else:
    st.info("Ajoute au moins un film pour voir apparaître ta Triade.")
