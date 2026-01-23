# LA TRIADE

Je présente ici mon application de recommandation cinématographique personnalisée. L'objectif de ce projet est d'offrir une alternative aux algorithmes de recommandation classiques en proposant une sélection de trois films aux profils distincts, générée à partir d'une analyse sémantique profonde.

### [Lien vers l'application](https://triade-recommendation.streamlit.app/#la-triade)

---

## Méthodologie et Classification

Pour construire cet outil, j'ai utilisé une approche basée sur le traitement du langage naturel (NLP). La recommandation ne repose pas uniquement sur le genre du film, mais sur la création d'un profil textuel complet (soupe de données) intégrant :

* Les mots-clés (Keywords) et les thèmes récurrents.
* L'équipe technique : Réalisateur (Director), Scénariste (Writer) et l'équipe principale (Crew).
* La distribution (Cast).
* Le synopsis détaillé.

L'algorithme utilise la mesure de **similarité cosinus**. En transformant ces métadonnées en vecteurs mathématiques, je calcule la proximité entre les films sélectionnés par l'utilisateur et l'ensemble de ma base de données pour identifier les correspondances les plus pertinentes.

### Les trois catégories de la Triade

Une fois les films les plus proches identifiés, je les filtre selon trois catégories prédéfinies :

1. **La Valeur Sûre (Blockbuster)** : Une sélection de films à fort succès commercial et critique, garantissant une expérience de visionnage accessible et de haute qualité.
2. **Le Choix Culte (Culte)** : Des œuvres ayant acquis un statut historique ou une base de fans dévouée. Ce sont les piliers de la culture cinématographique.
3. **La Pépite (Pépite)** : Des films plus rares, indépendants ou moins connus du grand public, choisis pour leur originalité et leur qualité artistique.

---

## Stack Technique

* **Python** : Logique métier et manipulation de données.
* **Pandas** : Gestion et fusion des bases de données.
* **Scikit-Learn** : Utilisation de `CountVectorizer` pour la vectorisation et `cosine_similarity` pour le moteur de recommandation.
* **Streamlit** : Interface utilisateur avec intégration de CSS personnalisé pour respecter l'identité visuelle de Letterboxd
