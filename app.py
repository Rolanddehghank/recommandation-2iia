import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 🔹 Création du dataset avec des produits plus variés
data = pd.DataFrame({
    'Produit': [
        'Chaussures de Sport', 'Baskets Running', 'Sac de Sport', 'Montre Fitness', 'Bouteille Isotherme',
        'Casque Audio', 'PC Portable', 'Clavier Mécanique', 'Écran 27 pouces', 'Smartphone 5G',
        'Aspirateur Robot', 'Table Basse', 'Lampe LED', 'Jeu Vidéo', 'Écouteurs Sans Fil'
    ],
    'Description': [
        'Chaussures confortables pour le sport et la marche',
        'Baskets légères adaptées au running et au sport',
        'Sac pratique pour transporter ses affaires de sport',
        'Montre connectée pour le suivi d’activité sportive',
        'Bouteille isotherme idéale pour le sport et la randonnée',
        'Casque sans fil avec réduction de bruit pour la musique',
        'Ordinateur portable performant pour le travail et le gaming',
        'Clavier mécanique RGB pour une meilleure frappe et gaming',
        'Écran 27 pouces Full HD idéal pour travail et gaming',
        'Smartphone dernière génération avec connectivité 5G',
        'Aspirateur robot intelligent pour un nettoyage autonome',
        'Table basse design pour un salon moderne',
        'Lampe LED réglable pour un éclairage d’ambiance',
        'Jeu vidéo immersif pour console et PC',
        'Écouteurs sans fil avec réduction de bruit et autonomie longue'
    ],
    'Catégorie': [
        'Sport', 'Sport', 'Sport', 'Sport', 'Sport',
        'Tech', 'Tech', 'Tech', 'Tech', 'Tech',
        'Maison', 'Maison', 'Maison', 'Divertissement', 'Tech'
    ],
    'Image': [
        'https://via.placeholder.com/150', 'https://via.placeholder.com/150', 
        'https://via.placeholder.com/150', 'https://via.placeholder.com/150', 
        'https://via.placeholder.com/150', 'https://via.placeholder.com/150',
        'https://via.placeholder.com/150', 'https://via.placeholder.com/150', 
        'https://via.placeholder.com/150', 'https://via.placeholder.com/150',
        'https://via.placeholder.com/150', 'https://via.placeholder.com/150', 
        'https://via.placeholder.com/150', 'https://via.placeholder.com/150',
        'https://via.placeholder.com/150'
    ]
})

# 🔹 Transformer les descriptions en vecteurs numériques
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Description'])

# 🔹 Calculer la similarité cosinus entre les produits
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 🔹 Fonction de recommandation améliorée (meilleure diversité)
def recommander_produits(nom_produit, data, similarity_matrix, top_n=3):
    idx = data[data['Produit'] == nom_produit].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]  # Exclure le produit lui-même
    
    # Sélectionner les produits les plus pertinents avec des catégories variées
    recommandations = []
    categories_vues = set()
    
    for i, score in scores:
        produit_reco = data.iloc[i]
        if produit_reco["Catégorie"] not in categories_vues:
            recommandations.append(produit_reco)
            categories_vues.add(produit_reco["Catégorie"])
        
        if len(recommandations) >= top_n:
            break
    
    return recommandations

# 🔹 Interface utilisateur avec Streamlit
st.title("🛒 Recommandation de Produits Variés")
st.write("""
💡 **Comment ça marche ?**  
Sélectionnez un produit, et nous vous recommanderons **des articles variés mais cohérents**, selon vos préférences.
""")

# 🔹 Sélecteur de produit
produit_selectionne = st.selectbox("Choisissez un produit :", data['Produit'])

# 🔹 Afficher les recommandations
if produit_selectionne:
    recommandations = recommander_produits(produit_selectionne, data, similarity_matrix)
    
    if not recommandations:
        st.warning("❌ Aucun produit similaire trouvé.")
    else:
        st.success(f"📢 Si vous aimez **{produit_selectionne}**, vous pourriez aussi aimer :")

        # 🔹 Affichage des recommandations en colonnes
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for i, rec in enumerate(recommandations):
            with cols[i]:  # Affichage en colonnes
                st.image(rec['Image'], width=120)
                st.write(f"**{rec['Produit']}**")
