from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config import Mark1Config


@dataclass
class _LoanTerms:
    offered_rate: float
    contracted_credit: float


class Mark1Model:
    """
    Mark I+ implementation based on Appendix A of Gualdi et al. (2014).

    Design goal:
    - stay close to the paper's pseudocode
    - keep project style simple: one model file, numpy arrays + Python lists
    - preserve the money-conservation logic emphasized for Mark I+
    """

    def __init__(self, config: Mark1Config) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        nf = config.n_firms
        nh = config.n_households

        # -------------------------
        # Households
        # -------------------------
        # Working households H (the owner O is separate, as in Appendix A)
        self.household_wealth = np.zeros(nh, dtype=float)
        self.household_wage = np.zeros(nh, dtype=float)
        self.household_employer = np.full(nh, -1, dtype=int)

        # Owner O
        self.owner_wealth = 0.0

        # -------------------------
        # Firms
        # -------------------------
        self.firm_price = np.ones(nf, dtype=float)
        self.firm_production = np.ones(nf, dtype=float)
        self.firm_target_production = np.ones(nf, dtype=float)
        self.firm_demand = np.ones(nf, dtype=float)  # realized sales in last goods market
        self.firm_liquidity = np.full(nf, 50.0, dtype=float)
        self.firm_total_debt = np.zeros(nf, dtype=float)
        self.firm_interest_rate = np.zeros(nf, dtype=float)
        self.firm_interest_due = np.zeros(nf, dtype=float)
        self.firm_vacancies = np.zeros(nf, dtype=int)
        self.firm_age = np.zeros(nf, dtype=int)
        self.firm_labor_demand = np.ones(nf, dtype=int)

        # Employee roster per firm
        self.firm_employees: list[list[int]] = [[] for _ in range(nf)]

        # Aggregate / bookkeeping
        self.bank_liquidity = 0.0
        self.avg_price = 1.0

        # Make initial employment consistent with initial production.
        # With alpha = 1 and initial Y_i = 1, each firm should start with one worker.
        initial_workers_needed = int(round(1.0 / self.config.alpha))

        h = 0
        for i in range(nf):
            for _ in range(initial_workers_needed):
                if h >= nh:
                    break
                self.firm_employees[i].append(h)
                self.household_employer[h] = i
                self.household_wage[h] = self.config.wage
                h += 1

        self.history: dict[str, list[float]] = {
            "unemployment": [],
            "employment": [],
            "avg_price": [],
            "owner_wealth": [],
            "bank_liquidity": [],
            "total_household_wealth": [],
            "total_firm_liquidity": [],
            "total_firm_equity": [],
            "total_output": [],
            "total_sales": [],
            "bad_debts": [],
            "bankruptcies": [],
        }

        # Record true t=0 state before any dynamics
        self.initial_total_money = self.total_money()
        self._record_state(bad_debts=0.0, bankruptcies=0)
    # ------------------------------------------------------------------
    # Credit functions from the paper
    # ------------------------------------------------------------------

    @staticmethod
    def _g_fragility(leverage: float) -> float:
        # Mark I+ choice in the paper
        return 1.0 + np.log1p(max(leverage, 0.0))

    @staticmethod
    def _credit_contraction(rho: float) -> float:
        """
        Continuous contraction function from Eq. (4), with rho in decimals.
        5% -> 0.05, 10% -> 0.10.
        """
        if rho < 0.05:
            return 1.0
        if rho < 0.10:
            return 1.0 - (rho - 0.05) / 0.05
        return 0.0

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def employment_rate(self) -> float:
        employed = float(np.sum(self.household_employer >= 0))
        if self.config.n_households <= 0:
            return 0.0
        return employed / self.config.n_households

    def unemployment_rate(self) -> float:
        return 1.0 - self.employment_rate()

    def firm_equity(self, i: int) -> float:
        return float(self.firm_liquidity[i] - self.firm_total_debt[i])

    def total_money(self) -> float:

            return float(

                self.bank_liquidity

                + np.sum(self.firm_liquidity)

                + np.sum(self.household_wealth)

                + self.owner_wealth

            )
            
    def average_price(self, firm_idx: Optional[np.ndarray] = None) -> float:
        if firm_idx is None:
            firm_idx = np.arange(self.config.n_firms)

        sales = self.firm_demand[firm_idx]
        denom = float(np.sum(sales))
        if denom <= 0.0:
            return 1.0

        return float(np.sum(self.firm_price[firm_idx] * sales) / denom)

    def total_household_wealth(self) -> float:
        return float(np.sum(self.household_wealth)) #don't include owner wealth

    def total_firm_liquidity(self) -> float:
        return float(np.sum(self.firm_liquidity))

    def total_firm_equity(self) -> float:
        return float(np.sum(self.firm_liquidity - self.firm_total_debt))

    def total_output(self) -> float:
        return float(np.sum(self.firm_production))

    def total_sales(self) -> float:
        return float(np.sum(self.firm_demand))

    # ------------------------------------------------------------------
    # Firm methods
    # ------------------------------------------------------------------

    def _set_new_strategy(self, i: int, avg_price: float) -> None:
        self.firm_age[i] += 1

        y = self.firm_production[i]
        d = self.firm_demand[i]
        p = self.firm_price[i]

        # Appendix A / Eq. (1) logic.
        # Realized demand in the pseudocode is realized sales, so D <= Y typically.
        if np.isclose(y, d) and p >= avg_price:
            self.firm_target_production[i] = y * (1.0 + self.config.gamma_y * self.rng.random())
        elif np.isclose(y, d) and p < avg_price:
            self.firm_price[i] = p * (1.0 + self.config.gamma_p * self.rng.random())
        elif y > d and p >= avg_price:
            self.firm_price[i] = p * (1.0 - self.config.gamma_p * self.rng.random())
        elif y > d and p < avg_price:
            self.firm_target_production[i] = y * (1.0 - self.config.gamma_y * self.rng.random())

        self.firm_target_production[i] = max(self.firm_target_production[i], self.config.alpha)
        self.firm_labor_demand[i] = int(np.ceil(self.firm_target_production[i] / self.config.alpha))

    def _get_loans(self, i: int) -> None:
        wage = self.config.wage
        labor_needed = self.firm_labor_demand[i]
        financial_need = wage * labor_needed - self.firm_liquidity[i]

        if financial_need <= 0.0:
            return

        leverage = (self.firm_total_debt[i] + financial_need) / (self.firm_liquidity[i] + 1e-3)
        offered_rate = self.config.rho0 * self._g_fragility(leverage)
        contracted_credit = financial_need * self._credit_contraction(offered_rate)

        if contracted_credit > 0.0:
            self.firm_interest_rate[i] = offered_rate
            self.firm_total_debt[i] += contracted_credit
            self.firm_liquidity[i] += contracted_credit
            self.bank_liquidity -= contracted_credit

    def _compute_interests(self, i: int) -> None:
        self.firm_interest_due[i] = self.firm_interest_rate[i] * self.firm_total_debt[i]

    def _define_labor_demand(self, i: int) -> None:
        wage = self.config.wage
        affordable_workers = int(np.floor(self.firm_liquidity[i] / wage))
        affordable_workers = max(affordable_workers, 0)
        self.firm_labor_demand[i] = min(self.firm_labor_demand[i], affordable_workers)
        self.firm_vacancies[i] = self.firm_labor_demand[i] - len(self.firm_employees[i])

    def _fire_worker(self, i: int, h: int) -> None:
        self.household_employer[h] = -1
        self.household_wage[h] = 0.0
        self.firm_employees[i].remove(h)
        self.firm_vacancies[i] += 1

    def _fire_random_worker(self, i: int) -> None:
        if not self.firm_employees[i]:
            return
        pos = int(self.rng.integers(0, len(self.firm_employees[i])))
        h = self.firm_employees[i][pos]
        self._fire_worker(i, h)

    def _hire(self, i: int, h: int) -> None:
        self.firm_employees[i].append(h)
        self.household_employer[h] = i
        self.household_wage[h] = self.config.wage
        self.firm_vacancies[i] -= 1

    def _produce(self, i: int) -> None:
        self.firm_production[i] = min(
            self.firm_target_production[i],
            self.config.alpha * len(self.firm_employees[i]),
        )
        self.firm_demand[i] = 0.0

    def _pay_workers(self, i: int) -> None:
        if not self.firm_employees[i]:
            return

        total_wages = self.config.wage * len(self.firm_employees[i])
        for h in self.firm_employees[i]:
            self.household_wealth[h] += self.config.wage
        self.firm_liquidity[i] -= total_wages

    def _markup_rule(self, i: int) -> None:
        if self.firm_production[i] <= 0.0:
            return

        markup_price = (1.0 + self.config.markup_mu) * (
            self.config.wage * len(self.firm_employees[i]) + self.firm_interest_due[i]
        ) / self.firm_production[i]
        self.firm_price[i] = max(self.firm_price[i], markup_price)

    def _sell(self, i: int, quantity: float) -> None:
        if quantity <= 0.0:
            return
        self.firm_demand[i] += quantity
        self.firm_liquidity[i] += self.firm_price[i] * quantity

    def _accounting(self, i: int) -> None:
        # Interest + partial debt repayment
        self.firm_liquidity[i] -= self.firm_interest_due[i] + self.config.debt_repayment_tau * self.firm_total_debt[i]
        self.bank_liquidity += self.firm_interest_due[i] + self.config.debt_repayment_tau * self.firm_total_debt[i]
        self.firm_total_debt[i] *= (1.0 - self.config.debt_repayment_tau)

        profit = (
            self.firm_price[i] * self.firm_demand[i]
            - self.config.wage * len(self.firm_employees[i])
            - self.firm_interest_due[i]
        )

        if profit > 0.0:
            div = self.config.dividend_delta * profit
            self.owner_wealth += div
            self.firm_liquidity[i] -= div

    def _reinit_firm(self, i: int, avg_price: float, avg_target_prod: float, avg_prod: float) -> None:
        self.firm_price[i] = avg_price
        self.firm_production[i] = avg_prod
        self.firm_target_production[i] = avg_target_prod
        self.firm_demand[i] = 0.0

        restart_cash = min(self.owner_wealth, avg_prod / self.config.alpha)
        self.firm_liquidity[i] = restart_cash
        self.owner_wealth -= restart_cash

        self.firm_vacancies[i] = 0
        self.firm_total_debt[i] = 0.0
        self.firm_interest_rate[i] = 0.0
        self.firm_interest_due[i] = 0.0
        self.firm_age[i] = 0
        self.firm_labor_demand[i] = int(np.ceil(avg_target_prod / self.config.alpha))

        # Fire all current workers
        for h in list(self.firm_employees[i]):
            self.household_employer[h] = -1
            self.household_wage[h] = 0.0
        self.firm_employees[i].clear()

    # ------------------------------------------------------------------
    # Household goods market
    # ------------------------------------------------------------------

    def _consume_household(self, h: int) -> None:
        budget = self.config.consumption_propensity * self.household_wealth[h]
        if budget <= 0.0:
            return

        m = min(self.config.m_sampled_firms, self.config.n_firms)
        chosen = self.rng.choice(self.config.n_firms, size=m, replace=False)
        chosen = chosen[np.argsort(self.firm_price[chosen])]

        spent = 0.0
        for i in chosen:
            if spent >= budget:
                break

            stock = self.firm_production[i] - self.firm_demand[i]
            if stock <= 0.0:
                continue

            max_qty = (budget - spent) / self.firm_price[i]
            if stock > max_qty:
                qty = max_qty
            else:
                qty = stock

            self._sell(i, qty)
            spent += qty * self.firm_price[i]

        self.household_wealth[h] -= spent

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self) -> bool:
        nf = self.config.n_firms

        # 1) Firms decide strategies, loans, interests, labor demand
        demanding_firms: list[int] = []
        excess_firms: list[int] = []

        current_avg_price = self.avg_price

        for i in range(nf):
            self._set_new_strategy(i, current_avg_price)
            self._get_loans(i)
            self._compute_interests(i)
            self._define_labor_demand(i)

            if self.firm_vacancies[i] > 0:
                demanding_firms.append(i)
            elif self.firm_vacancies[i] < 0:
                excess_firms.append(i)

        # 2) Labor market
        for i in excess_firms:
            while self.firm_vacancies[i] < 0:
                self._fire_random_worker(i)

        unemployed = [h for h in range(self.config.n_households) if self.household_employer[h] < 0]

        while unemployed and demanding_firms:
            uh_pos = int(self.rng.integers(0, len(unemployed)))
            ff_pos = int(self.rng.integers(0, len(demanding_firms)))

            h = unemployed.pop(uh_pos)
            i = demanding_firms[ff_pos]

            self._hire(i, h)
            if self.firm_vacancies[i] == 0:
                demanding_firms.pop(ff_pos)

        # 3) Production and wage payments
        for i in range(nf):
            self._produce(i)
            self._pay_workers(i)
            if self.firm_age[i] < 100:
                self._markup_rule(i)

        # 4) Goods market
        household_order = self.rng.permutation(self.config.n_households)
        for h in household_order:
            self._consume_household(int(h))

        # 5) Accounting and bankruptcies
        bad_debts = 0.0
        bankrupt_firms: list[int] = []
        healthy_firms: list[int] = []

        for i in range(nf):
            self._accounting(i)
            if self.firm_liquidity[i] < 0.0:
                bad_debts += self.firm_liquidity[i]  # negative
                bankrupt_firms.append(i)
            else:
                healthy_firms.append(i)

        # If all firms are bankrupt, stop the run
        if len(healthy_firms) == 0:
            self._record_state(bad_debts=bad_debts, bankruptcies=len(bankrupt_firms))
            return False

        # 6) Reinitialize bankrupt firms using healthy-firm averages
        healthy_idx = np.array(healthy_firms, dtype=int)
        avg_price_healthy = float(np.mean(self.firm_price[healthy_idx]))
        avg_target_prod_healthy = float(np.mean(self.firm_target_production[healthy_idx]))
        avg_prod_healthy = float(np.mean(self.firm_production[healthy_idx]))

        for i in bankrupt_firms:
            self._reinit_firm(i, avg_price_healthy, avg_target_prod_healthy, avg_prod_healthy)

        # 7) Spread bad debt proportionally over wealth/equity to conserve money
        # Include the owner as part of household wealth for conservation.
        firm_equities = np.array([max(self.firm_equity(i), 0.0) for i in range(nf)], dtype=float)
        hh_wealth = np.maximum(self.household_wealth, 0.0)
        owner_wealth = max(self.owner_wealth, 0.0)

        total_liquidity = float(np.sum(firm_equities) + np.sum(hh_wealth) + owner_wealth)

        if total_liquidity > 0.0 and bad_debts < 0.0:
            # bad_debts is negative, so this subtracts wealth/equity
            firm_shock = bad_debts * firm_equities / total_liquidity
            hh_shock = bad_debts * hh_wealth / total_liquidity
            owner_shock = bad_debts * owner_wealth / total_liquidity

            self.firm_liquidity += firm_shock
            self.household_wealth += hh_shock
            self.owner_wealth += float(owner_shock)

        # 8) Update average price using realized sales
        realized_sales = np.sum(self.firm_demand)
        if realized_sales > 0.0:
            self.avg_price = float(np.sum(self.firm_price * self.firm_demand) / realized_sales)
        else:
            self.avg_price = 1.0

        self._record_state(bad_debts=bad_debts, bankruptcies=len(bankrupt_firms))
        return True

    # ------------------------------------------------------------------
    # History / run
    # ------------------------------------------------------------------

    def _record_state(self, bad_debts: float, bankruptcies: int) -> None:
        current_total_money = self.total_money()
        if not np.isclose(current_total_money, self.initial_total_money, atol=1e-8):
            raise RuntimeError(
                f"Money conservation failed: current_total_money={current_total_money:.12f}, "
                f"initial_total_money={self.initial_total_money:.12f}"
            )
            
        self.history["unemployment"].append(self.unemployment_rate())
        self.history["employment"].append(self.employment_rate())
        self.history["avg_price"].append(self.avg_price)
        self.history["owner_wealth"].append(float(self.owner_wealth))
        self.history["bank_liquidity"].append(float(self.bank_liquidity))
        self.history["total_household_wealth"].append(self.total_household_wealth())
        self.history["total_firm_liquidity"].append(self.total_firm_liquidity())
        self.history["total_firm_equity"].append(self.total_firm_equity())
        self.history["total_output"].append(self.total_output())
        self.history["total_sales"].append(self.total_sales())
        self.history["bad_debts"].append(float(bad_debts))
        self.history["bankruptcies"].append(int(bankruptcies))

    def run(self, n_steps: int) -> dict[str, list[float]]:
        for _ in range(n_steps):
            alive = self.step()
            if not alive:
                break
        return self.history