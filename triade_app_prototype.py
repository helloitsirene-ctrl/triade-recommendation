import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re  # Importation cruciale pour corriger le NameError

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
    /* Centrage des colonnes */
    [data-testid="column"] {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
    }}
    .movie-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
    }}
    .movie-poster {{
        border-radius: 12px;
        width: 180px;
        transition: transform 0.3s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }}
    .movie-poster:hover {{
        transform: scale(1.05);
    }}
    .tag-style {{
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 2px 10px;
        margin: 3px;
        display: inline-block;
        font-size: 0.8rem;
        color: #FFD700 !important;
        font-weight: bold;
    }}
    .description-text {{
        font-size: 0.85rem;
        font-style: italic;
        margin: 15px 0;
        max-width: 250px;
        text-align: center;
        line-height: 1.3;
    }}
    a {{ text-decoration: none !important; }}
</style>
""", unsafe_allow_html=True)

# --- CHARGEMENT ---
@st.cache_data
def load_data():
    df = pd.read_csv('Triade_ULTIMATE.csv')
    df.columns = [c.lower() for c in df.columns]
    
    # Nettoyage et sécurité des colonnes
    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4]
    
    # On s'assure que 'overview' existe pour éviter le KeyError
    expected = ['genres', 'keywords', 'director', 'cast', 'all_themes', 'overview', 'name', 'category']
    for c in expected:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna('').astype(str)

    # Création du label pour la recherche : Nom (Année)
    df['search_label'] = df['name'] + " (" + df['year'].astype(str).str.replace('.0', '', regex=False) + ")"
    
    # Soupe avec thèmes prioritaires
    df['soup'] = df.apply(lambda x: ((x['all_themes'] + " ") * 5 + (x['genres'] + " ") * 3 + (x['keywords'] + " ") * 3 + x['director'] + " " + x['cast']).lower(), axis=1)
    return df

@st.cache_resource
def get_vectorizer(df):
    return CountVectorizer(stop_words='english').fit_transform(df['soup'])

df = load_data()
count_matrix = get_vectorizer(df)
# On indexe par le nouveau label "Nom (Année)"
indices = pd.Series(df.index, index=df['search_label']).drop_duplicates()

if 'offset' not in st.session_state: st.session_state.offset = 0

def get_combined_recs(search_labels):
    all_sim_scores = None
    input_soups = []
    selected_indices = []
    for label in search_labels:
        idx = indices[label]
        actual_idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx
        selected_indices.append(actual_idx)
        input_soups.append(df.iloc[actual_idx]['soup'])
        cos_sim = cosine_similarity(count_matrix[actual_idx], count_matrix)[0]
        all_sim_scores = cos_sim if all_sim_scores is None else all_sim_scores + cos_sim
            
    sim_scores = sorted(list(enumerate(all_sim_scores)), key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores if i[0] not in selected_indices]
    return df.iloc[movie_indices[0:150]], " ".join(input_soups)

# --- INTERFACE ---
st.markdown("<h1>🎬 La Triade</h1>", unsafe_allow_html=True)

# Barre de recherche multi-sélection plus ergonomique
selected_labels = st.multiselect(
    "Tape le nom de tes films favoris :",
    options=df['search_label'].sort_values().tolist(),
    max_selections=4,
    placeholder="Ex: Inception (2010)"
)

if selected_labels:
    results, combined_input_soup = get_combined_recs(selected_labels)
    st.write("---")
    col1, col2, col3 = st.columns(3)
    
    def draw_movie(category, cat_filter, col):
        recs = results[results['category'].str.lower() == cat_filter.lower()]
        if len(recs) > st.session_state.offset:
            movie = recs.iloc[st.session_state.offset]
            
            # Nettoyage des tags
            movie_words = re.findall(r'\w+', movie['soup'])
            input_words = set(re.findall(r'\w+', combined_input_soup))
            common_tags = [w for w in movie_words if w in input_words and len(w) > 3][:5]
            
            url = movie['film_url'] if 'film_url' in movie else "#"
            img = movie['poster_url'] if movie['poster_url'] != "" else "https://via.placeholder.com/180x270"

            with col:
                st.markdown(f"### {category}")
                st.markdown(f'''
                    <div class="movie-container">
                        <a href="{url}" target="_blank">
                            <img src="{img}" class="movie-poster">
                            <h4 style="margin:10px 0 5px 0;">{movie['name']}</h4>
                        </a>
                    </div>
                ''', unsafe_allow_html=True)
                
                st.write(f"{str(movie['year'])[:4]} | ⭐ {movie['rating']}")
                
                # Tags centrés et propres
                tags_html = "".join([f'<span class="tag-style">#{t}</span>' for t in common_tags])
                st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)
                
                # Affichage forcé de la description
                if movie['overview']:
                    desc = movie['overview'][:150] + "..." if len(movie['overview']) > 150 else movie['overview']
                    st.markdown(f'<p class="description-text">{desc}</p>', unsafe_allow_html=True)
        else:
            col.warning("Désolé, plus de résultats.")

    draw_movie("La Valeur Sûre", "Blockbuster", col1)
    draw_movie("Le Choix Culte", "Culte", col2)
    draw_movie("La Pépite", "Pépite", col3)

    st.write("---")
    if st.button("🔄 Pas convaincu ? Voir d'autres résultats"):
        st.session_state.offset += 1
        st.rerun()
else:
    st.info("Utilise la barre de recherche pour ajouter tes films.")
