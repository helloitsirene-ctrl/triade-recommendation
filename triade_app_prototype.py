import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION PAGE & DESIGN ---
st.set_page_config(page_title="La Triade", page_icon="🎬", layout="wide")

st.markdown(f"""
<style>
    .stApp {{
        background-color: #445566;
        color: white;
    }}
    h1, h2, h3, h4, h5, p, span, label {{
        color: white !important;
        text-align: center;
    }}
    [data-testid="column"] {{
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }}
    .movie-poster {{
        border-radius: 12px;
        width: 180px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        margin-bottom: 10px;
    }}
    .tag-style {{
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 2px 8px;
        margin: 2px;
        display: inline-block;
        font-size: 0.8rem;
        color: #ff4b4b !important;
    }}
    .description-text {{
        font-size: 0.9rem;
        font-style: italic;
        margin-top: 10px;
        max-width: 250px;
        text-align: justify;
    }}
</style>
""", unsafe_allow_html=True)

# --- CHARGEMENT ---
@st.cache_data
def load_data():
    df = pd.read_csv('Triade_ULTIMATE.csv')
    df.columns = [c.lower() for c in df.columns]
    
    # Réparation année et colonnes texte
    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    
    cols = ['genres', 'keywords', 'director', 'cast', 'all_themes', 'overview', 'name', 'category']
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna('').astype(str)

    def create_soup(x):
        return ((x['all_themes'] + " ") * 5 + (x['genres'] + " ") * 3 + (x['keywords'] + " ") * 3 + x['director'] + " " + x['cast']).lower()

    df['soup'] = df.apply(create_soup, axis=1)
    return df

@st.cache_resource
def get_vectorizer(df):
    count = CountVectorizer(stop_words='english')
    return count.fit_transform(df['soup'])

df = load_data()
count_matrix = get_vectorizer(df)
# Gestion propre des indices pour éviter l'erreur NumPy
indices = pd.Series(df.index, index=df['name'].str.lower()).drop_duplicates()

if 'offset' not in st.session_state: st.session_state.offset = 0

# --- MOTEUR CORRIGÉ ---
def get_combined_recs(titles):
    all_sim_scores = None
    input_soups = []
    
    selected_indices = []
    for title in titles:
        idx = indices[title.lower()]
        # Correction de l'erreur .iloc sur entier
        if hasattr(idx, 'iloc'): 
            actual_idx = idx.iloc[0]
        else:
            actual_idx = idx
            
        selected_indices.append(actual_idx)
        input_soups.append(df.iloc[actual_idx]['soup'])
        
        cos_sim = cosine_similarity(count_matrix[actual_idx], count_matrix)[0]
        if all_sim_scores is None:
            all_sim_scores = cos_sim
        else:
            all_sim_scores += cos_sim
            
    sim_scores = list(enumerate(all_sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Filtrage des films déjà sélectionnés
    movie_indices = [i[0] for i in sim_scores if i[0] not in selected_indices]
    
    return df.iloc[movie_indices[0:150]], " ".join(input_soups)

# --- INTERFACE ---
st.markdown("<h1>🎬 La Triade</h1>", unsafe_allow_html=True)

selected_movies = st.multiselect(
    "Choisis tes films favoris (max 4) :",
    options=sorted(df['name'].unique().tolist()),
    max_selections=4
)

if selected_movies:
    results, combined_input_soup = get_combined_recs(selected_movies)
    
    st.write("---")
    col1, col2, col3 = st.columns(3)
    
    def draw_movie(category, cat_filter, col):
        # Filtrage par catégorie
        recs = results[results['category'].str.lower() == cat_filter.lower()]
        
        if len(recs) > st.session_state.offset:
            movie = recs.iloc[st.session_state.offset]
            
            # Tags communs (hashtags)
            movie_words = set(movie['soup'].split())
            input_words = set(combined_input_soup.split())
            common_tags = [w for w in movie_words if w in input_words and len(w) > 3][:5]
            
            with col:
                st.markdown(f"### {category}")
                img = movie['poster_url'] if pd.notna(movie['poster_url']) else "https://via.placeholder.com/180x270"
                st.markdown(f'<img src="{img}" class="movie-poster">', unsafe_allow_html=True)
                
                st.markdown(f"#### {movie['name']}")
                st.write(f"{str(movie['year'])[:4]} | ⭐ {movie['rating']}")
                
                tags_html = "".join([f'<span class="tag-style">#{t}</span>' for t in common_tags])
                st.markdown(tags_html, unsafe_allow_html=True)
                
                if movie['overview']:
                    desc = movie['overview'][:150] + "..." if len(movie['overview']) > 150 else movie['overview']
                    st.markdown(f'<p class="description-text">{desc}</p>', unsafe_allow_html=True)
                
                url = movie['film_url'] if 'film_url' in movie else "#"
                st.markdown(f"[Voir sur Letterboxd ↗]({url})")
        else:
            col.warning("Plus de résultats.")

    draw_movie("La Valeur Sûre", "Blockbuster", col1)
    draw_movie("Le Choix Culte", "Culte", col2)
    draw_movie("La Pépite", "Pépite", col3)

    st.write("---")
    if st.button("🔄 Voir d'autres résultats"):
        st.session_state.offset += 1
        st.rerun()
else:
    st.info("Sélectionne des films pour générer ta Triade.")
