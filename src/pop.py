import numpy as np
from scipy.linalg import expm
from .params import _nonnegative_finite ,TripletRates, InitialPopulations

def RateMatrix(rates: TripletRates) -> np.ndarray:
    kx, ky, kz = rates.kx, rates.ky, rates.kz
    wxy, wyz, wxz = rates.wxy, rates.wyz, rates.wxz

    return np.array([
        [-(kx + wxy + wxz), wxy, wxz],
        [wxy, -(ky + wxy + wyz), wyz],
        [wxz, wyz, -(kz + wxz + wyz)],
    ], dtype=float)

def _from_vector(values:np.ndarray) -> InitialPopulations:
    if np.min(values) < -1e-10 or abs(np.sum(values) - 1) > 1e-10:
        raise ArithmeticError("unphysical propagated populations")
    
    values = np.maximum(values, 0)
    values = values / np.sum(values)

    return InitialPopulations(*map(float, values))

def PropagatePopulations(initial: InitialPopulations,
                         rates: TripletRates,
                         duration: float) -> InitialPopulations:
    _nonnegative_finite("duration", duration)
    R = RateMatrix(rates)
    P = expm(R * duration)
    final = P @ np.array([initial.Px, initial.Py, initial.Pz], dtype=float)
    return _from_vector(np.r_[1 - final.sum(), final])

def SwapMatrix(transition: str, efficiency: float) -> np.ndarray:
    _nonnegative_finite("efficiency", efficiency)
    if efficiency > 1:
        raise ValueError(f"efficiency must be between 0 and 1, got {efficiency}")
    
    if transition == "xy":
        return np.array([[1 - efficiency, efficiency, 0],
                         [efficiency, 1 - efficiency, 0],
                         [0, 0, 1]], dtype=float)
    elif transition == "yz":
        return np.array([[1, 0, 0],
                         [0, 1 - efficiency, efficiency],
                         [0, efficiency, 1 - efficiency]], dtype=float)
    elif transition == "xz":
        return np.array([[1 - efficiency, 0, efficiency],
                         [0, 1, 0],
                         [efficiency, 0, 1 - efficiency]], dtype=float)
    else:
        raise ValueError(f"transition must be one of 'xy', 'yz', or 'xz', got {transition}")


def PostSwapPopulations(initial: InitialPopulations,
                        transition: str,
                        efficiency: float) -> InitialPopulations:
    S = SwapMatrix(transition, efficiency)
    final = S @ np.array([initial.Px, initial.Py, initial.Pz], dtype=float)
    return _from_vector(np.r_[initial.P0, final])
