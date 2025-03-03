import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 🔹 Création du dataset avec produits, descriptions et images
data = pd.DataFrame({
    'Produit': ['Chaussures Sport', 'Baskets Running', 'Sac de Sport', 'Montre Fitness', 'Bouteille d’eau', 'Casque Audio'],
    'Description': [
        'Chaussures confortables pour le sport et la marche',
        'Baskets légères adaptées au running et au sport',
        'Sac pratique pour transporter ses affaires de sport',
        'Montre connectée pour le suivi d’activité sportive',
        'Bouteille isotherme idéale pour le sport et la randonnée',
        'Casque sans fil avec réduction de bruit pour la musique'
    ],
    'Image': [
        'https://via.placeholder.com/150',
        'https://via.placeholder.com/150',
        'https://via.placeholder.com/150',
        'https://via.placeholder.com/150',
        'https://via.placeholder.com/150',
        'https://via.placeholder.com/150'
    ]
})

# 🔹 Transformer les descriptions en vecteurs numériques (TF-IDF)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Description'])

# 🔹 Calculer la similarité cosinus entre les produits
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 🔹 Fonction de recommandation
def recommander_produits(nom_produit, data, similarity_matrix, top_n=3):
    idx = data[data['Produit'] == nom_produit].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recommandations = [data.iloc[i[0]] for i in scores]
    return recommandations

# 🔹 Interface utilisateur avec Streamlit
st.title("🛒 Recommandation de Produits")
st.write("""
💡 **Comment ça marche ?**  
Sélectionnez un produit, et nous vous recommanderons d’autres articles similaires basés sur leur description.
""")

# 🔹 Sélecteur de produit
produit_selectionne = st.selectbox("Choisissez un produit :", data['Produit'])

# 🔹 Afficher les recommandations
if produit_selectionne:
    recommandations = recommander_produits(produit_selectionne, data, similarity_matrix)
    
    st.success(f"📢 Si vous aimez **{produit_selectionne}**, vous pourriez aussi aimer :")

    # 🔹 Affichage des recommandations en colonnes
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for i, rec in enumerate(recommandations):
        with cols[i]:  # Affichage en colonnes
            st.image(rec['Image'], width=120)
            st.write(f"**{rec['Produit']}**")

# 🔹 Bouton pour relancer une nouvelle recommandation
if st.button("🔄 Relancer la recommandation"):
    st.experimental_rerun()
