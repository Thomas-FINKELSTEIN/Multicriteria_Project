"""
Script pour créer des données de test pour le Portfolio Optimizer.
À utiliser si vous n'avez pas encore téléchargé les vraies données.
"""

import pandas as pd
import numpy as np
import os
import json

# Créer le dossier data
os.makedirs("data", exist_ok=True)

# Charger les tickers depuis tick.json
with open("tick.json", "r") as f:
    sectors = json.load(f)

# Créer un index de dates
dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="B")

print(f"Génération de données pour {sum(len(t) for t in sectors.values())} tickers...")

# Générer des prix synthétiques pour chaque ticker
for sector, tickers in sectors.items():
    print(f"Secteur: {sector}")
    
    for ticker in tickers:
        # Prix initial aléatoire entre 50 et 500
        initial_price = np.random.uniform(50, 500)
        
        # Rendements journaliers simulés (drift + volatilité)
        drift = np.random.uniform(-0.0001, 0.0003)  # Drift annualisé
        volatility = np.random.uniform(0.01, 0.03)  # Volatilité journalière
        
        returns = np.random.normal(drift, volatility, len(dates))
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Créer le DataFrame
        df = pd.DataFrame({
            "Date": dates,
            ticker: prices
        })
        df.set_index("Date", inplace=True)
        
        # Sauvegarder
        csv_path = os.path.join("data", f"{ticker}.csv")
        df.to_csv(csv_path)

print(f"\n✅ Données générées dans le dossier 'data/'")
print(f"   Total: {len(os.listdir('data'))} fichiers CSV")
