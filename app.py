"""
==========================================
APPLICATION STREAMLIT - PRÉDICTION PRIX AIRBNB NYC
==========================================
Auteur : Votre Nom
Date : 2025
Description : Interface web pour prédire le prix des logements Airbnb à NYC
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================

st.set_page_config(
    page_title="Prédiction Prix Airbnb NYC",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# STYLE CSS PERSONNALISÉ
# ============================================

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #FF5A5F;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px 30px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF385C;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1 {
        color: #FF5A5F;
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    h2 {
        color: #484848;
        font-size: 28px;
    }
    h3 {
        color: #767676;
        font-size: 20px;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #767676;
        font-size: 14px;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# CHARGEMENT DES MODÈLES ET ENCODEURS
# ============================================

@st.cache_resource
def charger_modeles():
    """
    Charge tous les fichiers nécessaires pour faire les prédictions
    Returns:
        tuple: (modèle, scaler, encodeurs, informations)
    """
    try:
        model = joblib.load('airbnb_price_model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoder_group = joblib.load('encoder_neighbourhood_group.pkl')
        encoder_room = joblib.load('encoder_room_type.pkl')
        encoder_neighbourhood = joblib.load('encoder_neighbourhood.pkl')
        model_info = joblib.load('model_info.pkl')
        return model, scaler, encoder_group, encoder_room, encoder_neighbourhood, model_info
    except FileNotFoundError as e:
        st.error(f"❌ Erreur : Fichier manquant - {e}")
        st.stop()

# Chargement des modèles
model, scaler, encoder_group, encoder_room, encoder_neighbourhood, model_info = charger_modeles()

# ============================================
# EN-TÊTE DE L'APPLICATION
# ============================================

st.markdown("<h1>🏠 Prédiction des Prix Airbnb NYC</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #767676;'>Estimez le prix idéal pour votre logement à New York</p>", unsafe_allow_html=True)

# Séparateur visuel
st.markdown("---")

# ============================================
# AFFICHAGE DES PERFORMANCES DU MODÈLE
# ============================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="metric-card">
            <h3>📊 Modèle Utilisé</h3>
            <p style='font-size: 24px; font-weight: bold; color: #FF5A5F;'>Linear Regression</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Précision (R²)</h3>
            <p style='font-size: 24px; font-weight: bold; color: #00A699;'>{model_info['r2']*100:.2f}%</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <h3>📉 Erreur Moyenne (RMSE)</h3>
            <p style='font-size: 24px; font-weight: bold; color: #FC642D;'>${model_info['rmse']:.2f}</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# FORMULAIRE DE SAISIE
# ============================================

st.markdown("<h2>📝 Entrez les caractéristiques du logement</h2>", unsafe_allow_html=True)

# Organisation en colonnes pour un design moderne
col_gauche, col_droite = st.columns(2)

with col_gauche:
    st.markdown("### 📍 Localisation")
    
    # Quartier principal
    neighbourhood_group = st.selectbox(
        "Quartier principal",
        options=model_info['neighbourhood_groups'],
        help="Sélectionnez le quartier de New York"
    )
    
    # Quartier spécifique
    neighbourhood = st.selectbox(
        "Quartier spécifique",
        options=model_info['neighbourhoods'],
        help="Sélectionnez le quartier précis"
    )
    
    # Latitude
    latitude = st.slider(
        "Latitude",
        min_value=40.5,
        max_value=40.9,
        value=40.73,
        step=0.01,
        help="Position géographique (latitude)"
    )
    
    # Longitude
    longitude = st.slider(
        "Longitude",
        min_value=-74.3,
        max_value=-73.7,
        value=-73.95,
        step=0.01,
        help="Position géographique (longitude)"
    )
    
    st.markdown("### 🏠 Type de logement")
    
    # Type de chambre
    room_type = st.selectbox(
        "Type de chambre",
        options=model_info['room_types'],
        help="Sélectionnez le type de logement"
    )

with col_droite:
    st.markdown("### 📊 Caractéristiques")
    
    # Nuits minimum
    minimum_nights = st.number_input(
        "Nombre minimum de nuits",
        min_value=1,
        max_value=365,
        value=3,
        step=1,
        help="Durée minimale de séjour requise"
    )
    
    # Nombre de reviews
    number_of_reviews = st.number_input(
        "Nombre total de reviews",
        min_value=0,
        max_value=1000,
        value=10,
        step=1,
        help="Nombre d'avis reçus"
    )
    
    # Reviews par mois
    reviews_per_month = st.slider(
        "Reviews par mois",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Moyenne de reviews reçues par mois"
    )
    
    # Listings de l'hôte
    calculated_host_listings_count = st.number_input(
        "Nombre de logements de l'hôte",
        min_value=1,
        max_value=100,
        value=1,
        step=1,
        help="Nombre total de logements possédés par l'hôte"
    )
    
    # Disponibilité
    availability_365 = st.slider(
        "Disponibilité annuelle (jours)",
        min_value=0,
        max_value=365,
        value=180,
        step=1,
        help="Nombre de jours disponibles sur l'année"
    )

st.markdown("---")

# ============================================
# BOUTON DE PRÉDICTION
# ============================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("🔮 PRÉDIRE LE PRIX", use_container_width=True)

# ============================================
# TRAITEMENT DE LA PRÉDICTION
# ============================================

if predict_button:
    # Animation de chargement
    with st.spinner("🔄 Calcul en cours..."):
        
        # 1. Encodage des variables catégorielles
        neighbourhood_group_encoded = encoder_group.transform([neighbourhood_group])[0]
        room_type_encoded = encoder_room.transform([room_type])[0]
        neighbourhood_encoded = encoder_neighbourhood.transform([neighbourhood])[0]
        
        # 2. Création du vecteur de features
        # L'ordre DOIT être identique à celui utilisé lors de l'entraînement
        features_array = np.array([[
            neighbourhood_group_encoded,
            room_type_encoded,
            neighbourhood_encoded,
            latitude,
            longitude,
            minimum_nights,
            number_of_reviews,
            reviews_per_month,
            calculated_host_listings_count,
            availability_365
        ]])
        
        # 3. Normalisation avec le scaler
        features_scaled = scaler.transform(features_array)
        
        # 4. Prédiction
        prix_predit = model.predict(features_scaled)[0]
        
        # 5. Calcul des intervalles de confiance (basé sur RMSE)
        prix_min = max(0, prix_predit - model_info['rmse'])
        prix_max = prix_predit + model_info['rmse']
    
    # ============================================
    # AFFICHAGE DU RÉSULTAT
    # ============================================
    
    st.success("✅ Prédiction réussie !")
    
    # Box de prédiction avec design moderne
    st.markdown(f"""
        <div class="prediction-box">
            <h2 style='color: white; margin-bottom: 10px;'>💰 Prix Estimé</h2>
            <h1 style='color: white; font-size: 64px; margin: 20px 0;'>${prix_predit:.2f}</h1>
            <p style='font-size: 18px; opacity: 0.9;'>par nuit</p>
            <p style='font-size: 16px; opacity: 0.8; margin-top: 20px;'>
                Fourchette de prix : ${prix_min:.2f} - ${prix_max:.2f}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Détails de la prédiction
    st.markdown("### 📋 Détails de votre logement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **📍 Localisation**
        - Quartier : {neighbourhood_group}
        - Zone : {neighbourhood}
        - Coordonnées : ({latitude}, {longitude})
        
        **🏠 Type de logement**
        - {room_type}
        """)
    
    with col2:
        st.markdown(f"""
        **📊 Statistiques**
        - Minimum de nuits : {minimum_nights}
        - Reviews totales : {number_of_reviews}
        - Reviews/mois : {reviews_per_month}
        - Logements de l'hôte : {calculated_host_listings_count}
        - Disponibilité : {availability_365} jours/an
        """)
    
    # Conseils personnalisés
    st.markdown("### 💡 Recommandations")
    
    if prix_predit < 100:
        st.info("👍 Votre prix est compétitif ! Idéal pour attirer rapidement des voyageurs.")
    elif prix_predit < 200:
        st.success("⭐ Prix dans la moyenne pour ce type de logement. Bon équilibre qualité/prix.")
    else:
        st.warning("💎 Prix premium. Assurez-vous d'offrir des équipements et services exceptionnels.")
    
    # Facteurs influençant le prix
    with st.expander("📈 Facteurs qui influencent le prix"):
        st.markdown("""
        Les principaux facteurs qui déterminent le prix de votre logement :
        
        1. **Localisation** 🗺️
           - Manhattan commande généralement les prix les plus élevés
           - Proximité des attractions touristiques
        
        2. **Type de logement** 🏠
           - Logement entier > Chambre privée > Chambre partagée
        
        3. **Réputation** ⭐
           - Nombre de reviews positives
           - Historique de l'hôte
        
        4. **Disponibilité** 📅
           - Plus vous êtes disponible, plus vous attirez de clients
        
        5. **Flexibilité** 🔄
           - Durée minimale de séjour raisonnable
        """)

# ============================================
# SECTION INFORMATIVE
# ============================================

with st.expander("ℹ️ À propos de cette application"):
    st.markdown("""
    ### Comment fonctionne la prédiction ?
    
    Cette application utilise un modèle de **Machine Learning** (Régression Linéaire) 
    entraîné sur des milliers de listings Airbnb réels de New York City.
    
    **📊 Performance du modèle :**
    - Le modèle explique {:.2f}% de la variance des prix
    - Erreur moyenne de prédiction : ${:.2f}
    
    **🔧 Technologies utilisées :**
    - Python 🐍
    - Scikit-learn (Machine Learning)
    - Streamlit (Interface web)
    - Pandas & NumPy (Traitement des données)
    
    **📝 Note :** Les prédictions sont des estimations basées sur des données historiques. 
    Les prix réels peuvent varier en fonction de facteurs supplémentaires non pris en compte 
    par le modèle (saison, événements spéciaux, qualité des photos, etc.).
    """.format(model_info['r2']*100, model_info['rmse']))

# ============================================
# PIED DE PAGE
# ============================================

st.markdown("""
    <div class="footer">
        <p>🏠 Application de Prédiction des Prix Airbnb NYC | Développée avec ❤️ en Python</p>
        <p>© 2025 - Projet Machine Learning | Données : Airbnb Open Data</p>
    </div>
""", unsafe_allow_html=True)