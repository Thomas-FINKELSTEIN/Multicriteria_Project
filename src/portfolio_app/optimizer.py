import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize

def compute_pareto_mask(df: pd.DataFrame, ui_context: bool = False) -> np.ndarray:
    if df is None or df.empty:
        return np.array([], dtype=bool)

    if not df.index.is_unique:
        if ui_context:
            st.warning("Attention : indices dupliques detectes, le calcul Pareto peut etre affecte.")
        return np.zeros(len(df), dtype=bool)

    cols = ["Return", "Risk", "Cost"]
    if not all(c in df.columns for c in cols):
        return np.zeros(len(df), dtype=bool)

    df_obj = df[cols].dropna()
    if df_obj.empty:
        return np.zeros(len(df), dtype=bool)

    vals = df_obj.values
    n = len(vals)
    is_pareto = np.ones(n, dtype=bool)

    objectives = np.column_stack([-vals[:, 0], vals[:, 1], vals[:, 2]])

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if (np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i])):
                is_pareto[i] = False
                break

    indexer = df.index.get_indexer(df_obj.index)
    if np.any(indexer < 0):
        return np.zeros(len(df), dtype=bool)

    mask_full = np.zeros(len(df), dtype=bool)
    mask_full[indexer] = is_pareto
    return mask_full

class PortfolioOptimizer:
    def __init__(self, mu: pd.Series, sigma: pd.DataFrame, current_weights=None, transaction_cost: float = 0.005):
        self.mu = mu.values
        self.sigma = sigma.values
        self.tickers = list(mu.index)
        self.n = len(mu)
        self.w_current = current_weights if current_weights is not None else np.zeros(self.n)
        self.c = transaction_cost

    def _apply_constraints(self, w, max_k, delta_tol):
        w = np.asarray(w, dtype=float).copy()

        if max_k < self.n:
            idx_zero = np.argsort(w)[:-max_k]
            w[idx_zero] = 0.0

        eff_tol = max(delta_tol, 1e-12)
        w[w < eff_tol] = 0.0

        s = w.sum()
        if s > 0:
            w /= s
        else:
            return np.zeros_like(w)

        return w

    def compute_performance(self, w):
        w = np.asarray(w, dtype=float)
        ret = np.dot(w, self.mu)
        vol = np.sqrt(np.dot(w.T, np.dot(self.sigma, w)))
        return ret, vol

    def compute_transaction_cost(self, w):
        w = np.asarray(w, dtype=float)
        return self.c * np.sum(np.abs(w - self.w_current))

    def compute_efficient_frontier(self, n_points=60):
        results = []
        bounds = tuple((0, 1) for _ in range(self.n))
        init = np.array([1.0 / self.n] * self.n)

        target_returns = np.linspace(self.mu.min(), self.mu.max(), n_points)

        for target in target_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: np.dot(w, self.mu) - t},
            ]
            try:
                sol = minimize(
                    lambda w: np.sqrt(np.dot(w.T, np.dot(self.sigma, w))),
                    init,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 1000},
                )
                if sol.success:
                    w = sol.x.copy()
                    w[w < 1e-6] = 0.0
                    if w.sum() <= 0:
                        continue
                    w /= w.sum()
                    ret, vol = self.compute_performance(w)
                    if vol > 0:
                        results.append({"Return": ret, "Risk": vol, "Sharpe": ret / vol, "Weights": w})
            except Exception:
                continue

        return pd.DataFrame(results)

    def run_monte_carlo(self, n_portfolios=5000, max_k=5, delta_tol=0.01):
        results = []
        eff_tol = max(delta_tol, 1e-12)

        for _ in range(n_portfolios):
            w_raw = np.random.random(self.n)
            w = self._apply_constraints(w_raw, max_k, delta_tol)
            if w.sum() == 0:
                continue

            ret, vol = self.compute_performance(w)
            cost = self.compute_transaction_cost(w)

            if vol > 0:
                results.append(
                    {
                        "Method": "Monte Carlo",
                        "Return": ret,
                        "Risk": vol,
                        "Cost": cost,
                        "Sharpe": ret / vol,
                        "Weights": w,
                        "N_Assets": int(np.count_nonzero(w > eff_tol)),
                    }
                )

        return pd.DataFrame(results)

    def optimize_scalarization(self, max_k=5, delta_tol=0.01):
        results = []
        bounds = tuple((0, 1) for _ in range(self.n))
        eff_tol = max(delta_tol, 1e-12)

        lambda_combinations = []
        steps = 5
        for l1 in np.linspace(0.1, 0.8, steps):
            for l2 in np.linspace(0.1, 0.8, steps):
                l3 = 1 - l1 - l2
                if l3 >= 0.05:
                    lambda_combinations.append((l1, l2, l3))

        for (l1, l2, l3) in lambda_combinations:

            def objective(w, l1=l1, l2=l2, l3=l3):
                ret = np.dot(w, self.mu)
                risk = np.sqrt(np.dot(w.T, np.dot(self.sigma, w)))
                cost = self.c * np.sum(np.abs(w - self.w_current))
                return -l1 * ret + l2 * risk + l3 * cost

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            init = np.array([1.0 / self.n] * self.n)

            try:
                sol = minimize(
                    objective,
                    init,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 1000},
                )
                if sol.success:
                    w_final = self._apply_constraints(sol.x.copy(), max_k, delta_tol)
                    if w_final.sum() == 0:
                        continue
                    ret, vol = self.compute_performance(w_final)
                    cost = self.compute_transaction_cost(w_final)
                    if vol > 0:
                        results.append(
                            {
                                "Return": ret,
                                "Risk": vol,
                                "Cost": cost,
                                "Sharpe": ret / vol,
                                "Weights": w_final,
                                "Lambda_Ret": l1,
                                "Lambda_Risk": l2,
                                "Lambda_Cost": l3,
                                "N_Assets": int(np.count_nonzero(w_final > eff_tol)),
                                "Method": "Scalarisation",
                            }
                        )
            except Exception:
                continue

        return pd.DataFrame(results)

    def optimize_nsga2_simple(self, n_generations=50, pop_size=100, max_k=5, delta_tol=0.01):
        eff_tol = max(delta_tol, 1e-12)

        def dominates(a, b):
            better_in_one = False
            for i in range(3):
                if a[i] > b[i]:
                    return False
                if a[i] < b[i]:
                    better_in_one = True
            return better_in_one

        def evaluate(w):
            ret, vol = self.compute_performance(w)
            cost = self.compute_transaction_cost(w)
            return (-ret, vol, cost)

        def create_individual():
            w = np.random.random(self.n)
            return self._apply_constraints(w, max_k, delta_tol)

        def crossover(p1, p2):
            alpha = np.random.random()
            child = alpha * p1 + (1 - alpha) * p2
            return self._apply_constraints(child, max_k, delta_tol)

        def mutate(w, rate=0.1):
            if np.random.random() < rate:
                w = w + np.random.normal(0, 0.05, self.n)
                w = np.clip(w, 0, 1)
                w = self._apply_constraints(w, max_k, delta_tol)
            return w

        population = [create_individual() for _ in range(pop_size)]
        population = [ind for ind in population if ind.sum() > 0]

        for _ in range(n_generations):
            evaluated = [(i, ind, evaluate(ind)) for i, ind in enumerate(population)]

            fronts = []
            remaining_ids = set(range(len(evaluated)))
            while remaining_ids:
                front_ids = []
                for i in list(remaining_ids):
                    obj_i = evaluated[i][2]
                    dominated_flag = False
                    for j in list(remaining_ids):
                        if i == j:
                            continue
                        obj_j = evaluated[j][2]
                        if dominates(obj_j, obj_i):
                            dominated_flag = True
                            break
                    if not dominated_flag:
                        front_ids.append(i)

                fronts.append(front_ids)
                remaining_ids -= set(front_ids)

            new_pop = []
            for front_ids in fronts:
                if len(new_pop) + len(front_ids) <= pop_size:
                    new_pop.extend([evaluated[i][1] for i in front_ids])
                else:
                    needed = pop_size - len(new_pop)
                    new_pop.extend([evaluated[i][1] for i in front_ids[:needed]])
                    break

            children = []
            while len(children) < pop_size // 2 and len(new_pop) >= 2:
                idx1, idx2 = np.random.choice(len(new_pop), 2, replace=False)
                child = crossover(new_pop[idx1].copy(), new_pop[idx2].copy())
                child = mutate(child)
                if child.sum() > 0:
                    children.append(child)

            population = (new_pop + children)[:pop_size]

        results = []
        for w in population:
            if w.sum() > 0:
                ret, vol = self.compute_performance(w)
                cost = self.compute_transaction_cost(w)
                if vol > 0:
                    results.append(
                        {
                            "Return": ret,
                            "Risk": vol,
                            "Cost": cost,
                            "Sharpe": ret / vol,
                            "Weights": w,
                            "N_Assets": int(np.count_nonzero(w > eff_tol)),
                            "Method": "NSGA-II",
                        }
                    )

        return pd.DataFrame(results)
