from __future__ import annotations

from src.config import Mark1Config
from src.models.mark1 import Mark1Model


def main() -> None:
    config = Mark1Config(
        n_firms=100,
        n_households=1000,
        wage=1.0,
        alpha=1.0,
        gamma_p=0.10,
        gamma_y=0.10,
        markup_mu=0.0,
        dividend_delta=0.20,
        debt_repayment_tau=0.05,
        m_sampled_firms=3,
        consumption_propensity=0.80,
        rho0=0.02,   # 2%
        seed=42,
    )

    model = Mark1Model(config)
    history = model.run(1000)

    print("Mark I+ run completed.")
    print()
    print("Configuration:")
    print(config)
    print()

    if len(history["unemployment"]) == 0:
        print("No history recorded.")
        return

    print("Initial conditions:")
    print(f"  unemployment           = {history['unemployment'][0]:.6f}")
    print(f"  employment             = {history['employment'][0]:.6f}")
    print(f"  avg_price              = {history['avg_price'][0]:.6f}")
    print(f"  owner_wealth           = {history['owner_wealth'][0]:.6f}")
    print(f"  bank_liquidity         = {history['bank_liquidity'][0]:.6f}")
    print(f"  working_household_wealth      = {history['total_household_wealth'][0]:.6f}")
    print(
    f"  total_money                   = "
    f"{history['bank_liquidity'][0] + history['total_firm_liquidity'][0] + history['total_household_wealth'][0] + history['owner_wealth'][0]:.6f}")
    print(f"  total_firm_liquidity   = {history['total_firm_liquidity'][0]:.6f}")
    print(f"  total_firm_equity      = {history['total_firm_equity'][0]:.6f}")
    print()

    print("Final conditions:")
    print(f"  unemployment           = {history['unemployment'][-1]:.6f}")
    print(f"  employment             = {history['employment'][-1]:.6f}")
    print(f"  avg_price              = {history['avg_price'][-1]:.6f}")
    print(f"  owner_wealth           = {history['owner_wealth'][-1]:.6f}")
    print(f"  bank_liquidity         = {history['bank_liquidity'][-1]:.6f}")
    print(f"  working_household_wealth      = {history['total_household_wealth'][-1]:.6f}")
    print(
    f"  total_money                   = "
    f"{history['bank_liquidity'][-1] + history['total_firm_liquidity'][-1] + history['total_household_wealth'][-1] + history['owner_wealth'][-1]:.6f}")
    print(f"  total_firm_liquidity   = {history['total_firm_liquidity'][-1]:.6f}")
    print(f"  total_firm_equity      = {history['total_firm_equity'][-1]:.6f}")
    print(f"  total_output           = {history['total_output'][-1]:.6f}")
    print(f"  total_sales            = {history['total_sales'][-1]:.6f}")
    print()

    print("Event totals:")
    print(f"  total_bankruptcies     = {sum(history['bankruptcies'])}")
    print(f"  cumulative_bad_debts   = {sum(history['bad_debts']):.6f}")
    print()

    print("Sanity checks:")
    print(f"  unemployment in [0,1]  = {all(0.0 <= u <= 1.0 for u in history['unemployment'])}")
    print(f"  avg_price > 0          = {all(p > 0.0 for p in history['avg_price'])}")
    print()

    print("Last 5 observations:")
    kmax = min(5, len(history["unemployment"]))
    for k in range(kmax, 0, -1):
        idx = -k
        print(
            f"  t={len(history['unemployment']) - k:4d} | "
            f"u={history['unemployment'][idx]:.6f} | "
            f"p={history['avg_price'][idx]:.6f} | "
            f"Y={history['total_output'][idx]:.6f} | "
            f"sales={history['total_sales'][idx]:.6f} | "
            f"bk={history['bankruptcies'][idx]}"
        )


if __name__ == "__main__":
    main()