# Portfolio Optimizer Pro 📈

Application Streamlit d'optimisation de portefeuille multi-objectifs.

## 🚀 Installation

```bash
# 1. Installer les dépendances
pip install streamlit plotly scipy pandas numpy yfinance tqdm

# 2. Télécharger les données réelles (optionnel)
python download.py

# OU générer des données de test
python generate_test_data.py

# 3. Lancer l'application
streamlit run portfolio_optimizer_pro.py
```

## 📁 Structure du Projet

```
Projet_Final/
├── .streamlit/
│   └── config.toml          # Configuration du thème
├── data/                     # Données de prix (générées ou téléchargées)
├── download.py               # Script de téléchargement Yahoo Finance
├── generate_test_data.py     # Script de génération de données test
├── portfolio_optimizer_pro.py # Application principale
├── tick.json                 # Liste des tickers par secteur
└── README.md
```

## 🎯 Fonctionnalités

### Onglet 1 : Frontière Efficiente (Markowitz)
- Optimisation bi-objectif classique (rendement vs risque)
- Visualisation de la frontière Pareto
- Sélection du portefeuille optimal respectant r_min

### Onglet 2 : Monte Carlo 3D
- Optimisation tri-objectif (rendement, risque, coûts)
- Contrainte de cardinalité (max K actifs)
- 5,000 simulations Monte Carlo
- Visualisation 3D interactive

### Onglet 3 : Analyse du Portefeuille
- Backtest historique
- Graphique de drawdown
- Répartition par actif et secteur
- Statistiques de performance

### Onglet 4 : Documentation
- Formules mathématiques
- Explication des contraintes
- Limites du modèle

## 📊 Format des Données

Les fichiers CSV doivent avoir le format suivant :

```csv
Date,TICKER
2020-01-02,195.52
2020-01-03,195.61
...
```

Ou avec métadonnées (format Yahoo Finance) :
```csv
Price,TICKER
Ticker,TICKER
Date,
2020-01-02,195.52
...
```

## 🔧 Paramètres

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| r_min | Rendement minimum requis | Moyenne des actifs |
| K | Nombre max d'actifs (cardinalité) | 5 |
| c | Coût de transaction | 0.5% |
| Initial | Capital initial | 1,000 € |

## 📐 Formules Mathématiques

### Objectifs
- **f₁(w) = -w'μ** : Rendement (à maximiser)
- **f₂(w) = w'Σw** : Risque (à minimiser)  
- **f₃(w) = c·Σ|wᵢ-wₜ,ᵢ|** : Coûts de transaction

### Contraintes
- **Σwᵢ = 1** : Investissement complet
- **wᵢ ≥ 0** : Pas de vente à découvert
- **Card(w) ≤ K** : Cardinalité maximale

## ⚠️ Corrections Apportées

Cette version corrige les problèmes suivants du code original :

1. ✅ Indentation cassée de l'onglet 3 (était imbriqué dans l'onglet 2)
2. ✅ Variable `df_det` utilisée hors de son bloc
3. ✅ Gestion robuste des formats CSV (métadonnées Yahoo Finance)
4. ✅ Vérification de synchronisation tickers/poids
5. ✅ Gestion des valeurs NaN et données manquantes
6. ✅ Protection contre les divisions par zéro

## 👥 Auteurs

Projet Final - Optimisation de Portefeuille  
ESAIP - 2025
