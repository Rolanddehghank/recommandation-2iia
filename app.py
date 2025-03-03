import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 🔹 Création du dataset avec produits et descriptions
data = pd.DataFrame({
    'Produit': ['Chaussures Sport', 'Baskets Running', 'Sac de Sport', 'Montre Fitness', 'Bouteille d’eau', 'Casque Audio'],
    'Description': [
        'Chaussures confortables pour le sport et la marche',
        'Baskets légères adaptées au running et au sport',
        'Sac pratique pour transporter ses affaires de sport',
        'Montre connectée pour le suivi d’activité sportive',
        'Bouteille isotherme idéale pour le sport et la randonnée',
        'Casque sans fil avec réduction de bruit pour la musique'
    ]
})

# 🔹 Transformer les descriptions en vecteurs numériques
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Description'])

# 🔹 Calculer la similarité cosinus entre les produits
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 🔹 Fonction de recommandation
def recommander_produits(nom_produit, data, similarity_matrix, top_n=3):
    idx = data[data['Produit'] == nom_produit].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recommandations = [data['Produit'][i[0]] for i in scores]
    return recommandations

# 🔹 Interface utilisateur avec Streamlit
st.title("🛒 Améliorez vos ventes avec des recommandations intelligentes !")
st.write("Sélectionnez un produit, et découvrez ce que nous vous recommandons.")

# 🔹 Sélecteur de produit
produit_selectionne = st.selectbox("Choisissez un produit :", data['Produit'])

# 🔹 Afficher les recommandations
if produit_selectionne:
    recommandations = recommander_produits(produit_selectionne, data, similarity_matrix)
    st.success(f"Si vous aimez **{produit_selectionne}**, vous pourriez aussi aimer :")
    for rec in recommandations:
        st.write(f"- {rec}")
