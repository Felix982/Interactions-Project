from dataclasses import dataclass


@dataclass
class Mark0Config:
    n_firms: int = 1000
    mu: float = 1.0
    c: float = 0.5
    beta: float = 2.0
    gamma_p: float = 0.05
    eta_plus: float = 0.1
    eta_minus: float = 0.1
    delta: float = 0.02
    theta: float = 2.0
    phi: float = 0.1
    f: float = 1.0
    gamma_w: float = 0.0  # 0.0 = fixed-wage basic Mark 0; >0 enables wage update extension
    seed: int = 42
    random_revival: bool = False


@dataclass
class Mark1Config:
    n_firms: int = 100
    n_households: int = 1000

    # Firm / market parameters
    wage: float = 1.0
    alpha: float = 1.0
    gamma_p: float = 0.10
    gamma_y: float = 0.10
    markup_mu: float = 0.0
    dividend_delta: float = 0.20
    debt_repayment_tau: float = 0.05

    # Household parameters
    m_sampled_firms: int = 3
    consumption_propensity: float = 0.80

    # Bank / credit parameters
    rho0: float = 0.02  # 2%

    # Simulation
    seed: int = 42
    add_rate_noise : bool = False