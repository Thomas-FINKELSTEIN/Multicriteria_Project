import os
import json
import pandas as pd

def load_market_data_impl(data_dir: str = "data", json_path: str = "tick.json"):
    # NOTE: Streamlit cache appliquée dans app_main.py (pour éviter st.cache_data avant set_page_config)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dossier de donnees '{data_dir}' introuvable. Lancez python download.py d'abord.")

    sector_map = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            sectors = json.load(f)
        sector_map = {t: s for s, tickers in sectors.items() for t in tickers}

    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("Aucun fichier CSV trouve dans le dossier data")

    dfs = []
    for f in files:
        path = os.path.join(data_dir, f)
        try:
            with open(path, "r", encoding="utf-8") as file_check:
                first_lines = [file_check.readline() for _ in range(8)]

            skip_rows = 0
            ticker_name = os.path.splitext(f)[0].replace("_", " ")

            for i, line in enumerate(first_lines):
                if line.startswith("Ticker,"):
                    parts = line.strip().split(",")
                    if len(parts) > 1 and parts[1]:
                        ticker_name = parts[1]
                if "Date" in line:
                    skip_rows = i
                    break

            df = pd.read_csv(path, skiprows=skip_rows, index_col=0, parse_dates=True)
            df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
            if df.empty:
                continue

            if df.shape[1] > 1:
                for col in df.columns:
                    if not df[col].isna().all():
                        temp_df = df[[col]].dropna()
                        temp_df.columns = [col]
                        dfs.append(temp_df)
            else:
                df.columns = [ticker_name]
                df = df.dropna()
                if not df.empty:
                    dfs.append(df)
        except Exception:
            continue

    if not dfs:
        raise ValueError("Aucune donnee valide n'a pu etre chargee.")

    prices = pd.concat(dfs, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.ffill().bfill().dropna()

    if prices.empty or prices.shape[1] < 2:
        raise ValueError("Pas assez de donnees valides apres nettoyage (besoin d'au moins 2 actifs).")

    returns = prices.pct_change().dropna()
    mu = returns.mean() * 252
    sigma = returns.cov() * 252

    return prices, returns, mu, sigma, sector_map
