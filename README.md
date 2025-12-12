# Portfolio Optimizer Pro

Cet outil propose une solution interactive et robuste pour l'optimisation de portefeuilles financiers, s'appuyant sur des modèles quantitatifs avancés et une visualisation dynamique via Streamlit.

## Prérequis

* Python 3.8 ou supérieur
* pip (gestionnaire de paquets Python)

## Installation

La procédure suivante décrit comment configurer l'environnement virtuel et installer le projet en mode éditable via `pyproject.toml`.

### 1. Création de l'environnement virtuel

À la racine du projet, exécutez :

```bash
python -m venv venv
```

### 2. Activation de l'environnement

**Windows :**
```bash
venv\Scripts\activate
```

**macOS / Linux :**
```bash
source venv/bin/activate
```

### 3. Installation des dépendances

Une fois l'environnement activé, installez le projet et ses dépendances :

```bash
pip install -e .
```

## Lancement

Pour démarrer l'application Streamlit, utilisez la commande suivante à la racine du projet :

```bash
python -m streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut (généralement à l'adresse `http://localhost:8501`).

## Données

* Le dossier `data/` est destiné à accueillir les jeux de données financiers (CSV, Parquet, etc.).
* Le fichier `tick.json` contient les métadonnées ou les configurations spécifiques aux tickers boursiers.
* Le script `download.py` peut être utilisé pour récupérer les données de marché nécessaires.

## Arborescence du projet

```text
.
├── app.py
├── src/
│   └── portfolio_app/
│       ├── __init__.py
│       ├── app_main.py
│       ├── config.py
│       ├── styles.py
│       ├── ui_components.py
│       ├── data_loader.py
│       ├── core.py
│       └── charts.py
├── data/
├── tick.json
├── download.py
├── pyproject.toml
├── .gitignore
└── README.md
```