from __future__ import annotations

import numpy as np

from src.config import Mark0Config


class Mark0Model:
    """
    Mark 0 implementation following Appendix B of
    Gualdi et al. (2014), with the wage-update extension from Section 4
    activated only when gamma_w > 0.

    Conventions:
    - y : production / employment
    - p : price
    - w : wage
    - e : firm net deposits / equity
    - d : demand
    - profit : period profit
    - active : firm is active/alive
    - S : household accumulated savings
    """

    def __init__(self, config: Mark0Config) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        n = config.n_firms

        # Appendix B initialization
        self.w = np.ones(n, dtype=float)
        self.p = 1.0 + 0.2 * (self.rng.random(n) - 0.5)
        self.y = config.mu * (1.0 + 0.2 * (self.rng.random(n) - 0.5)) / 2.0
        self.e = self.w * self.y * 2.0 * self.rng.random(n)
        self.active = np.ones(n, dtype=bool)

        # Demand and profit are lagged state variables in Appendix B:
        # firms update using last round's D and P.
        self.d = np.zeros(n, dtype=float)
        self.profit = np.zeros(n, dtype=float)

        # Household savings from money-conservation initialization
        self.S = float(n - self.e.sum())

        self.history: dict[str, list[float]] = {
            "unemployment": [],
            "employment": [],
            "avg_price": [],
            "avg_wage": [],
            "total_output": [],
            "household_savings": [],
            "consumption_budget": [],
            "actual_consumption": [],
            "active_firms": [],
            "total_firm_equity": [],
            "dividends": [],
            "bankruptcies": [],
            "revivals": [],
        }

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def unemployment_rate(self) -> float:
        total_labor_force = self.config.mu * self.config.n_firms
        if total_labor_force <= 0:
            return 0.0
        employed = float(self.y[self.active].sum())
        u = 1.0 - employed / total_labor_force
        return float(np.clip(u, 0.0, 1.0))

    def employment_rate(self) -> float:
        return 1.0 - self.unemployment_rate()

    def average_price(self) -> float:
        active = self.active
        total_output = float(self.y[active].sum())
        if total_output <= 0.0:
            return 1.0
        return float(np.sum(self.p[active] * self.y[active]) / total_output)

    def average_wage(self) -> float:
        active = self.active
        total_output = float(self.y[active].sum())
        if total_output <= 0.0:
            return 1.0
        return float(np.sum(self.w[active] * self.y[active]) / total_output)

    def total_wage_bill(self) -> float:
        return float(np.sum(self.w[self.active] * self.y[self.active]))

    def consumption_budget(self) -> float:
        # Appendix B uses max(S, 0)
        return float(self.config.c * (max(self.S, 0.0) + self.total_wage_bill()))

    def _mu_u_tilde(self, unemployment: float, avg_wage: float) -> np.ndarray:
        """
        Firm-specific access to unemployed workers, Appendix B:
        u_tilde_i = exp(beta * W_i / wbar) / sum_active exp(...) * N_F * u

        Note: the pseudocode uses N_F * u, not mu * N_F * u, even though
        the prose uses mu * u_i. We follow Appendix B literally here.
        """
        n = self.config.n_firms
        out = np.zeros(n, dtype=float)

        active = self.active
        if not np.any(active):
            return out

        if avg_wage <= 0.0:
            avg_wage = 1.0

        logits = np.exp(self.config.beta * self.w[active] / avg_wage)
        denom = float(np.sum(logits))
        if denom <= 0.0:
            return out

        out[active] = logits / denom * self.config.mu * n * unemployment
        return out

    # ------------------------------------------------------------------
    # Core Mark 0 step blocks
    # ------------------------------------------------------------------

    def _update_firms(self, unemployment: float, employment: float, avg_price: float, avg_wage: float) -> None:
        """
        Firms update wages (optional extension), production, and prices
        using lagged demand and lagged profits, exactly in the order of
        Appendix B.
        """
        cfg = self.config
        mu_u_tilde = self._mu_u_tilde(unemployment, avg_wage)

        for i in np.where(self.active)[0]:
            y_i = self.y[i]
            d_i = self.d[i]
            p_i = self.p[i]

            # Optional wage-update extension from Section 4
            if cfg.gamma_w > 0.0 and y_i > 0.0:
                # If Y < D and P > 0: raise wage
                if y_i < d_i and self.profit[i] > 0.0:
                    candidate = self.w[i] * (1.0 + cfg.gamma_w * employment * self.rng.random())
                    # cap so profits would not turn negative ex post
                    wage_cap = p_i * min(d_i, y_i) / y_i
                    self.w[i] = min(candidate, wage_cap)

                # If Y > D and P < 0: cut wage
                elif y_i > d_i and self.profit[i] < 0.0:
                    self.w[i] = self.w[i] * (1.0 - cfg.gamma_w * unemployment * self.rng.random())
                    self.w[i] = max(self.w[i], 0.0)

            # Production and price update, Appendix B / Eq. (11)
            if y_i < d_i:
                self.y[i] = y_i + min(cfg.eta_plus * (d_i - y_i), mu_u_tilde[i])
                if p_i < avg_price:
                    self.p[i] = p_i * (1.0 + cfg.gamma_p * self.rng.random())

            elif y_i > d_i:
                self.y[i] = max(0.0, y_i - cfg.eta_minus * (y_i - d_i))
                if p_i > avg_price:
                    self.p[i] = p_i * (1.0 - cfg.gamma_p * self.rng.random())

    def _compute_demand(self) -> tuple[float, float]:
        """
        Households decide demand after firms updated prices/production.
        Returns (consumption_budget, avg_price_used_for_demand).
        """
        n = self.config.n_firms
        active = self.active

        self.d[:] = 0.0
        budget = self.consumption_budget()
        avg_price = self.average_price()

        if budget <= 0.0 or not np.any(active):
            return budget, avg_price

        exp_term = np.exp(-self.config.beta * self.p[active] / avg_price)
        denom = float(np.sum(exp_term))
        if denom <= 0.0:
            return budget, avg_price

        self.d[active] = (
            budget
            * exp_term
            / (self.p[active] * denom)
        )
        self.d[~active] = 0.0
        return budget, avg_price

    def _accounting(self) -> tuple[np.ndarray, float, float]:
        """
        Accounting block from Appendix B.
        Returns (healthy_idx, total_dividends, actual_consumption).
        """
        cfg = self.config
        active = self.active

        healthy = np.zeros(self.config.n_firms, dtype=bool)
        total_dividends = 0.0

        sold = np.minimum(self.y, self.d)
        actual_consumption = float(np.sum(self.p[active] * sold[active]))

        for i in np.where(active)[0]:
            self.profit[i] = self.p[i] * sold[i] - self.w[i] * self.y[i]

            # Household savings absorb minus profits first:
            # S <- S - P_i
            self.S -= self.profit[i]

            # Firm net deposits update:
            # E_i <- E_i + P_i
            self.e[i] += self.profit[i]

            # Dividends only if P_i > 0 and E_i > 0
            if self.profit[i] > 0.0 and self.e[i] > 0.0 and cfg.delta > 0.0:
                div = cfg.delta * self.profit[i]
                self.S += div
                self.e[i] -= div
                total_dividends += div

            # Healthy set H from Appendix B pseudocode
            if self.e[i] > cfg.theta * self.w[i] * self.y[i]:
                healthy[i] = True

        return healthy, total_dividends, actual_consumption

    def _defaults(self, healthy: np.ndarray) -> tuple[float, int]:
        """
        Default / bailout block from Appendix B.
        Returns (deficit, number_of_bankruptcies).

        Appendix B logic:
        - if E_i < -Theta * W_i * Y_i, firm is in distress
        - with prob 1-f and a healthy firm j with E_j > -E_i, bailout occurs
        - otherwise firm goes bankrupt:
            deficit += -E_i
            active[i] = 0
            y[i] = 0
            e[i] = 0
        """
        cfg = self.config
        deficit = 0.0
        bankruptcies = 0

        distressed = np.where(self.active & (self.e < -cfg.theta * self.w * self.y))[0]
        if distressed.size == 0:
            return deficit, bankruptcies

        for i in distressed:
            # Healthy set is dynamic; rebuild candidate list each time
            healthy_idx = np.where(healthy & self.active)[0]

            bailed_out = False
            if healthy_idx.size > 0 and self.rng.random() < (1.0 - cfg.f):
                j = int(self.rng.choice(healthy_idx))
                if self.e[j] > -self.e[i]:
                    # Bailout
                    self.e[j] += self.e[i]  # self.e[i] is negative
                    self.e[i] = 0.0
                    self.p[i] = self.p[j]
                    self.w[i] = self.w[j]
                    bailed_out = True

                    # Healthy status may change after transfer
                    healthy[j] = self.e[j] > cfg.theta * self.w[j] * self.y[j]
                    healthy[i] = self.e[i] > cfg.theta * self.w[i] * self.y[i]

            if not bailed_out:
                deficit += -self.e[i]
                self.active[i] = False
                self.y[i] = 0.0
                self.d[i] = 0.0
                self.profit[i] = 0.0
                self.e[i] = 0.0
                bankruptcies += 1
                healthy[i] = False

        return deficit, bankruptcies

    def _revivals(self, unemployment: float, avg_price: float, avg_wage: float,) -> tuple[float, int]:
        """
        Revival block from Appendix B.
        Returns (added_deficit, number_of_revivals).
        """

        """
        Careful: this is the pre-section 4 implementation. In section for they define an extension, 
        search for 'When a firm is revived from bankruptcy...' for passage
        """
        deficit = 0.0
        revivals = 0

        inactive_idx = np.where(~self.active)[0]
        if inactive_idx.size == 0:
            return deficit, revivals

        for i in inactive_idx:
            if self.rng.random() < self.config.phi:
                self.active[i] = True
                self.p[i] = avg_price

                # Optional: Section 4 extension: revived firms inherit the production-weighted
                # average wage of active firms.
                if self.config.gamma_w > 0.0:
                    self.w[i] = avg_wage

                # Optional : add noise to follow Appendix instead of main, default = False
                if self.config.random_revival:
                    self.y[i] = self.config.mu * unemployment * self.rng.random()
                else:
                    self.y[i] = self.config.mu * unemployment 
                self.e[i] = self.w[i] * self.y[i]
                self.d[i] = 0.0
                self.profit[i] = 0.0

                deficit += self.e[i]
                revivals += 1

        return deficit, revivals

    def _resolve_deficit(self, deficit: float) -> None:
        """
        Debt redistribution from Appendix B.

        If deficit > S:
            deficit -= S
            S = 0
            subtract residual deficit proportionally from positive-E firms
        else:
            S -= deficit
        """
        if deficit <= 0.0:
            return

        if deficit > self.S:
            residual = deficit - self.S
            self.S = 0.0

            pos_mask = self.active & (self.e > 0.0)
            e_plus = float(np.sum(self.e[pos_mask]))
            if e_plus > 0.0:
                self.e[pos_mask] -= (self.e[pos_mask] / e_plus) * residual
        else:
            self.S -= deficit

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record_state(
        self,
        consumption_budget: float = 0.0,
        actual_consumption: float = 0.0,
        total_dividends: float = 0.0,
        bankruptcies: int = 0,
        revivals: int = 0,
    ) -> None:
        self.history["unemployment"].append(self.unemployment_rate())
        self.history["employment"].append(self.employment_rate())
        self.history["avg_price"].append(self.average_price())
        self.history["avg_wage"].append(self.average_wage())
        self.history["total_output"].append(float(np.sum(self.y[self.active])))
        self.history["household_savings"].append(float(self.S))
        self.history["consumption_budget"].append(float(consumption_budget))
        self.history["actual_consumption"].append(float(actual_consumption))
        self.history["active_firms"].append(int(np.sum(self.active)))
        self.history["total_firm_equity"].append(float(np.sum(self.e[self.active])))
        self.history["dividends"].append(float(total_dividends))
        self.history["bankruptcies"].append(int(bankruptcies))
        self.history["revivals"].append(int(revivals))

    def step(self) -> None:
        # 1) Current aggregates
        unemployment = self.unemployment_rate()
        employment = 1.0 - unemployment
        avg_price = self.average_price()
        avg_wage = self.average_wage()

        # 2) Firms update prices, production, and wages (if enabled)
        self._update_firms(unemployment, employment, avg_price, avg_wage)

        # 3) Update aggregates again, as in Appendix B
        unemployment = self.unemployment_rate()
        avg_price = self.average_price()

        # 4) Households decide demand
        consumption_budget, _ = self._compute_demand()

        # 5) Accounting and healthy set
        healthy, total_dividends, actual_consumption = self._accounting()

        # 6) Defaults / bailouts
        deficit, bankruptcies = self._defaults(healthy)

        # 7) Revivals
        avg_wage = self.average_wage()
        revival_deficit, revivals = self._revivals(unemployment, avg_price, avg_wage)
        deficit += revival_deficit

        # 8) Debt redistribution
        self._resolve_deficit(deficit)

        # 9) Record
        self.record_state(
            consumption_budget=consumption_budget,
            actual_consumption=actual_consumption,
            total_dividends=total_dividends,
            bankruptcies=bankruptcies,
            revivals=revivals,
        )

    def run(self, n_steps: int) -> dict[str, list[float]]:
        for _ in range(n_steps):
            self.step()
        return self.history