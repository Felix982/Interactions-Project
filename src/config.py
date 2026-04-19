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