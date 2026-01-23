import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
st.set_page_config(page_title="LA TRIADE", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    a {text-decoration: none; color: inherit; font-weight: bold;}
    a:hover {color: #ff4b4b; text-decoration: underline;}
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 20px;}
</style>
""", unsafe_allow_html=True)

# --- CHARGEMENT (Avec nouveau nom pour casser le cache) ---
@st.cache_data
def load_data_FORCE_RESET():
    # 1. Chargement
    try:
        df = pd.read_csv('Triade_TAGGED_SOUP_V2.csv')
    except:
        df = pd.read_csv('Triade_TAGGED_SOUP.csv')
    
    # 2. Nettoyage colonnes
    df.columns = [c.lower() for c in df.columns]
    
    # 3. Réparation de l'année
    if 'year' not in df.columns and 'date' in df.columns:
        # On prend les 4 premiers caractères de la date (ex: 2023-01-01 -> 2023)
        df['year'] = df['date'].astype(str).str[:4]
    
    # Conversion en nombre pour être sûr
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

    df['soup'] = df['soup'].fillna('')
    return df

@st.cache_resource
def prepare_vectorizer(df):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])
    return count_matrix

try:
    # On appelle la nouvelle fonction
    df = load_data_FORCE_RESET()
    count_matrix = prepare_vectorizer(df)
    indices = pd.Series(df.index, index=df['name'].str.lower()).drop_duplicates()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# --- MOTEUR LÉGER ---
def get_recommendations(title):
    title = title.lower()
    if title not in indices: return None
    
    idx = indices[title]
    if isinstance(idx, pd.Series): idx = idx.iloc[0]

    cosine_sim = cosine_similarity(count_matrix[idx], count_matrix)
    
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:61]
    
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

# --- AFFICHAGE CARTE ---
def display_card(category_name, description, df_results, category_filter, emoji, metric_col, metric_label):
    st.markdown(f"### {emoji} {category_name}")
    st.caption(description)
    rec = df_results[df_results['category'] == category_filter].head(1) # Attention majuscule ici aussi
    
    # Fallback si 'Category' avec majuscule
    if rec.empty:
        rec = df_results[df_results['Category'] == category_filter].head(1)

    if not rec.empty:
        film = rec.iloc[0]
        url = film['film_url'] if 'film_url' in film else "#"
        st.markdown(f"#### [{film['name']}]({url})")
        
        rating = f"⭐ {film['rating']}" if 'rating' in film else ""
        st.write(f"{int(film['year'])} • {rating}")
        
        if metric_col == 'watches':
            st.metric(metric_label, f"{int(film['watches']):,}")
        elif metric_col == 'like_ratio':
            try:
                ratio = film['likes'] / film['watches'] if film['watches'] > 0 else 0
                st.metric(metric_label, f"{ratio:.1%}")
            except:
                st.metric(metric_label, "N/A")
        st.markdown(f"[Voir sur Letterboxd ↗]({url})")
    else:
        st.warning(f"Pas de {category_name} similaire trouvé.")

# --- INTERFACE ---
st.title("🎬 LA TRIADE (Reset)")

# DEBUG : Vérification visuelle pour toi
if 'year' in df.columns:
    st.success(f"✅ Colonne 'year' détectée ! (Exemple: {df['year'].iloc[0]})")
else:
    st.error("❌ La colonne 'year' manque encore !")

selected_movie = st.selectbox("Tu as aimé quel film ?", df['name'].sort_values().unique())

if st.button('Lancer la recherche'):
    with st.spinner('Calcul en cours...'):
        results = get_recommendations(selected_movie)
    
    if results is not None:
        st.write("---")
        st.subheader(f"Recommandations pour *{selected_movie}* :")
        col1, col2, col3 = st.columns(3)
        with col1: display_card("LA VALEUR SÛRE", "(Blockbuster)", results, 'Blockbuster', "🏛️", 'watches', "Vues")
        with col2: display_card("LE CHOIX CULTE", "(Cinéphiles)", results, 'Culte', "🎸", 'watches', "Vues")
        with col3: display_card("LA PÉPITE", "(Hidden Gem)", results, 'Pépite', "💎", 'like_ratio', "Like Ratio")
    else:
        st.error("Film introuvable.")
