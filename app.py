import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 🔹 Création du dataset avec des images réelles
data = pd.DataFrame({
    'Produit': [
        'Chaussures de Sport', 'Baskets Running', 'Sac de Sport', 'Montre Fitness', 'Bouteille Isotherme',
        'Casque Audio', 'PC Portable', 'Clavier Mécanique', 'Écran 27 pouces', 'Smartphone 5G',
        'Aspirateur Robot', 'Table Basse', 'Lampe LED', 'Jeu Vidéo', 'Écouteurs Sans Fil'
    ],
    'Description': [
        'Chaussures légères et confortables pour la course',
        'Baskets running ultra légères pour performances optimales',
        'Sac de sport imperméable avec compartiments multiples',
        'Montre connectée avec suivi d’activité et GPS',
        'Bouteille isotherme pour boissons chaudes et froides',
        'Casque Bluetooth avec réduction de bruit active',
        'Ordinateur portable performant pour gaming et travail',
        'Clavier mécanique rétroéclairé avec switches rapides',
        'Écran Full HD 27 pouces pour une expérience visuelle immersive',
        'Smartphone dernière génération avec appareil photo avancé',
        'Aspirateur robot autonome avec programmation intelligente',
        'Table basse en bois massif avec espace de rangement',
        'Lampe LED avec intensité réglable et mode veilleuse',
        'Jeu vidéo immersif avec graphismes haute résolution',
        'Écouteurs sans fil avec autonomie prolongée'
    ],
    'Catégorie': [
        'Sport', 'Sport', 'Sport', 'Sport', 'Sport',
        'Tech', 'Tech', 'Tech', 'Tech', 'Tech',
        'Maison', 'Maison', 'Maison', 'Divertissement', 'Tech'
    ],
    'Image': [
        'https://example.com/images/chaussures_sport.jpg',
        'https://example.com/images/baskets_running.jpg',
        'https://example.com/images/sac_sport.jpg',
        'https://example.com/images/montre_fitness.jpg',
        'https://example.com/images/bouteille_isotherme.jpg',
        'https://example.com/images/casque_audio.jpg',
        'https://example.com/images/pc_portable.jpg',
        'https://example.com/images/clavier_mecanique.jpg',
        'https://example.com/images/ecran_27p.jpg',
        'https://example.com/images/smartphone_5g.jpg',
        'https://example.com/images/aspirateur_robot.jpg',
        'https://example.com/images/table_basse.jpg',
        'https://example.com/images/lampe_led.jpg',
        'https://example.com/images/jeu_video.jpg',
        'https://example.com/images/ecouteurs_sansfil.jpg'
    ]
})

# 🔹 Transformer les descriptions en vecteurs numériques
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Description'])

# 🔹 Calculer la similarité cosinus entre les produits
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 🔹 Fonction de recommandation améliorée
def recommander_produits(nom_produit, data, similarity_matrix, top_n=3):
    idx = data[data['Produit'] == nom_produit].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]  # Exclure le produit lui-même

    # 🔥 Filtrer pour recommander uniquement dans la même catégorie
    categorie = data.loc[idx, "Catégorie"]
    recommandations = [data.iloc[i[0]] for i in scores if data.iloc[i[0]]["Catégorie"] == categorie][:top_n]
    
    return recommandations

# 🔹 Interface utilisateur avec Streamlit
st.title("🛒 Recommandation de Produits par Catégorie")
st.write("""
💡 **Comment ça marche ?**  
Sélectionnez un produit, et nous vous recommanderons uniquement des articles de la même catégorie.
""")

# 🔹 Sélecteur de produit
produit_selectionne = st.selectbox("Choisissez un produit :", data['Produit'])

# 🔹 Afficher les recommandations
if produit_selectionne:
    recommandations = recommander_produits(produit_selectionne, data, similarity_matrix)
    
    if not recommandations:
        st.warning("❌ Aucun produit similaire trouvé.")
    else:
        st.success(f"Si vous aimez **{produit_selectionne}**, vous pourriez aussi aimer :")

        # 🔹 Affichage des recommandations en colonnes avec images
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for i, rec in enumerate(recommandations):
            with cols[i]:  # Affichage en colonnes
                st.image(rec['Image'], width=120)
                st.write(f"**{rec['Produit']}**")
