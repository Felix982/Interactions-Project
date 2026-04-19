from __future__ import annotations

from src.config import Mark0Config
from src.models.mark0 import Mark0Model


def main() -> None:
    # Basic Mark 0 from the paper:
    # fixed wages unless gamma_w > 0
    config = Mark0Config(
        n_firms=1000,
        mu=1.0,
        c=0.5,
        beta=2.0,
        gamma_p=0.05,
        eta_plus=0.1,
        eta_minus=0.1,
        delta=0.02,
        theta=2.0,
        phi=0.1,
        f=1.0,
        gamma_w=0.0,  # keep 0.0 for the basic model
        seed=42,
    )

    model = Mark0Model(config)
    history = model.run(2500)

    print("Mark 0 run completed.")
    print()
    print("Configuration:")
    print(config)
    print()

    print("Initial conditions:")
    print(f"  unemployment       = {history['unemployment'][0]:.6f}")
    print(f"  employment         = {history['employment'][0]:.6f}")
    print(f"  avg_price          = {history['avg_price'][0]:.6f}")
    print(f"  avg_wage           = {history['avg_wage'][0]:.6f}")
    print(f"  total_output       = {history['total_output'][0]:.6f}")
    print(f"  household_savings  = {history['household_savings'][0]:.6f}")
    print(f"  active_firms       = {history['active_firms'][0]}")
    print()

    print("Final conditions:")
    print(f"  unemployment       = {history['unemployment'][-1]:.6f}")
    print(f"  employment         = {history['employment'][-1]:.6f}")
    print(f"  avg_price          = {history['avg_price'][-1]:.6f}")
    print(f"  avg_wage           = {history['avg_wage'][-1]:.6f}")
    print(f"  total_output       = {history['total_output'][-1]:.6f}")
    print(f"  household_savings  = {history['household_savings'][-1]:.6f}")
    print(f"  active_firms       = {history['active_firms'][-1]}")
    print()

    print("Event totals:")
    print(f"  total_bankruptcies = {sum(history['bankruptcies'])}")
    print(f"  total_revivals     = {sum(history['revivals'])}")
    print()

    print("Sanity checks:")
    print(f"  unemployment in [0, 1]: {all(0.0 <= u <= 1.0 for u in history['unemployment'])}")
    print(f"  active firms >= 0       : {all(a >= 0 for a in history['active_firms'])}")
    print(f"  prices > 0              : {bool((model.p > 0).all())}")
    print(f"  wages >= 0              : {bool((model.w >= 0).all())}")
    print()

    print("Last 5 observations:")
    for k in range(5, 0, -1):
        idx = -k
        print(
            f"  t={len(history['unemployment']) - k:4d} | "
            f"u={history['unemployment'][idx]:.6f} | "
            f"p={history['avg_price'][idx]:.6f} | "
            f"Y={history['total_output'][idx]:.6f} | "
            f"S={history['household_savings'][idx]:.6f} | "
            f"active={history['active_firms'][idx]}"
        )


if __name__ == "__main__":
    main()