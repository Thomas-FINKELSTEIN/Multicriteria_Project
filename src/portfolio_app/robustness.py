import numpy as np
import pandas as pd

class RobustnessAnalyzer:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns

    def bootstrap_sample(self):
        n_days = len(self.returns)
        idx = np.random.choice(n_days, size=n_days, replace=True)
        resampled = self.returns.iloc[idx]
        mu_boot = resampled.mean() * 252
        sigma_boot = resampled.cov() * 252
        return mu_boot, sigma_boot

    def run_bootstrap_analysis(self, base_weights: np.ndarray, n_bootstrap=100, confidence=0.95):
        rets, risks, sharpes = [], [], []

        for _ in range(n_bootstrap):
            mu_boot, sigma_boot = self.bootstrap_sample()
            w = np.asarray(base_weights, dtype=float)

            ret = np.dot(w, mu_boot.values)
            vol = np.sqrt(np.dot(w.T, np.dot(sigma_boot.values, w)))
            sharpe = ret / vol if vol > 0 else 0

            rets.append(ret)
            risks.append(vol)
            sharpes.append(sharpe)

        rets = np.asarray(rets)
        risks = np.asarray(risks)
        sharpes = np.asarray(sharpes)

        alpha = (1 - confidence) / 2

        return {
            "return_mean": rets.mean(),
            "return_std": rets.std(),
            "return_ci_low": np.percentile(rets, alpha * 100),
            "return_ci_high": np.percentile(rets, (1 - alpha) * 100),
            "return_worst": np.percentile(rets, 5),
            "risk_mean": risks.mean(),
            "risk_std": risks.std(),
            "risk_ci_low": np.percentile(risks, alpha * 100),
            "risk_ci_high": np.percentile(risks, (1 - alpha) * 100),
            "risk_worst": np.percentile(risks, 95),
            "sharpe_mean": sharpes.mean(),
            "sharpe_std": sharpes.std(),
            "sharpe_ci_low": np.percentile(sharpes, alpha * 100),
            "sharpe_ci_high": np.percentile(sharpes, (1 - alpha) * 100),
            "sharpe_worst": np.percentile(sharpes, 5),
            "returns_distribution": rets,
            "risks_distribution": risks,
            "sharpes_distribution": sharpes,
        }

    def compare_portfolios_robustness(self, portfolios_dict: dict, n_bootstrap=100):
        rows = []
        for name, weights in portfolios_dict.items():
            res = self.run_bootstrap_analysis(weights, n_bootstrap=n_bootstrap)
            rows.append(
                {
                    "Portfolio": name,
                    "Rendement Moyen (%)": res["return_mean"] * 100,
                    "Rendement Worst-Case (%)": res["return_worst"] * 100,
                    "Risque Moyen (%)": res["risk_mean"] * 100,
                    "Risque Worst-Case (%)": res["risk_worst"] * 100,
                    "Sharpe Moyen": res["sharpe_mean"],
                    "Sharpe Worst-Case": res["sharpe_worst"],
                    "Stabilite Rendement": 1 / (res["return_std"] + 0.001),
                }
            )
        return pd.DataFrame(rows)
