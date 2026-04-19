import numpy as np

from src.config import Mark0Config


class Mark0Model:
    def __init__(self, config: Mark0Config) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        n = config.n_firms

        # Firm-level variables
        self.w = np.ones(n)  # wages
        self.p = 1.0 + 0.2 * (self.rng.random(n) - 0.5)  # prices
        self.y = config.mu * (1.0 + 0.2 * (self.rng.random(n) - 0.5)) / 2.0  # production/employment
        self.e = self.w * self.y * 2.0 * self.rng.random(n)  # deposits / equity

        self.active = np.ones(n, dtype=bool)  # active firms
        self.d = np.zeros(n)  # demand
        self.profit = np.zeros(n)

        # Aggregate household savings
        self.S = n - self.e.sum()

        # Time series storage
        self.history = {
            "unemployment": [],
            "avg_price": [],
            "total_output": [],
            "savings": [],
            "consumption_budget": [],
            "active_firms": [],
        }

    def unemployment_rate(self) -> float:
        employed = self.y.sum()
        total_labor_force = self.config.mu * self.config.n_firms
        u = 1.0 - employed / total_labor_force
        return float(np.clip(u, 0.0, 1.0))

    def average_price(self) -> float:
        total_output = self.y.sum()
        if total_output <= 0:
            return 1.0
        return float((self.p * self.y).sum() / total_output)

    def wage_income(self) -> float:
        return float(np.sum(self.w[self.active] * self.y[self.active]))

    def consumption_budget(self) -> float:
        return float(self.config.c * (max(self.S, 0.0) + self.wage_income()))

    def compute_demand(self) -> None:
        avg_price = self.average_price()
        budget = self.consumption_budget()

        active_idx = self.active
        self.d[:] = 0.0

        if not np.any(active_idx):
            return

        price_term = np.exp(-self.config.beta * self.p[active_idx] / avg_price)
        z = np.sum(price_term)

        if z <= 0:
            return

        self.d[active_idx] = (budget / self.p[active_idx]) * (price_term / z)

    def record_state(self) -> None:
        self.history["unemployment"].append(self.unemployment_rate())
        self.history["avg_price"].append(self.average_price())
        self.history["total_output"].append(float(self.y.sum()))
        self.history["savings"].append(float(self.S))
        self.history["consumption_budget"].append(self.consumption_budget())
        self.history["active_firms"].append(int(self.active.sum()))
    



    def update_production_and_prices(self) -> None:
        cfg = self.config

        avg_price = self.average_price()

        active = self.active

        for i in np.where(active)[0]:
            y_i = self.y[i]
            d_i = self.d[i]
            p_i = self.p[i]

            if d_i > y_i:
                # hire (increase production)
                self.y[i] += cfg.eta_plus * (d_i - y_i)

                # increase price if cheaper than average
                if p_i < avg_price:
                    price_change = self.rng.random() * cfg.gamma_p
                    self.p[i] *= (1 + price_change)

            elif d_i < y_i:
                # fire (reduce production)
                self.y[i] -= cfg.eta_minus * (y_i - d_i)

                # decrease price if more expensive than average
                if p_i > avg_price:
                    price_change = self.rng.random() * cfg.gamma_p
                    self.p[i] *= (1 - price_change)

            # safety: no negative production
            if self.y[i] < 0:
                self.y[i] = 0.0

    def update_accounting(self) -> None:
        # Firms sell what they can (min of demand and production)
        sold = np.minimum(self.y, self.d)

        revenue = self.p * sold
        wage_cost = self.w * self.y

        # Firm profits
        self.profit = revenue - wage_cost

        # Update firm deposits
        self.e += self.profit

        # Household side
        total_wages = np.sum(self.w[self.active] * self.y[self.active])
        total_consumption = np.sum(self.p * sold)

        # Update household savings
        self.S += total_wages - total_consumption



    def handle_bankruptcy(self) -> None:
        theta = self.config.theta

        # A firm goes bankrupt if its debt exceeds threshold
        bankrupt = self.e < -theta * self.w * self.y

        if not np.any(bankrupt):
            return

        # Deactivate bankrupt firms
        self.active[bankrupt] = False

        # Reset their variables (they are "dead" for now)
        self.y[bankrupt] = 0.0
        self.d[bankrupt] = 0.0




    def revive_firms(self) -> None:
        phi = self.config.phi
        inactive_idx = np.where(~self.active)[0]

        if inactive_idx.size == 0:
            return

        unemployment = self.unemployment_rate()
        avg_price = self.average_price()

        for i in inactive_idx:
            if self.rng.random() < phi:
                self.active[i] = True
                self.p[i] = avg_price
                self.y[i] = self.config.mu * unemployment * self.rng.random()
                self.e[i] = self.w[i] * self.y[i]
                self.d[i] = 0.0
                self.profit[i] = 0.0
                self.S -= self.e[i]



    def step(self) -> None:
        self.compute_demand()
        self.update_production_and_prices()
        self.update_accounting()
        self.handle_bankruptcy()
        self.revive_firms()
        self.record_state()


    def run(self, n_steps: int) -> dict[str, list[float]]:
        for _ in range(n_steps):
            self.step()
        return self.history