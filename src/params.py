import math
from dataclasses import dataclass


def _nonnegative_finite(name: str, value: float) -> None:
    if value < 0 or math.isinf(value) or math.isnan(value):
        raise ValueError(f"{name} must be non-negative and finite, got {value}")


@dataclass(frozen=True)
class TripletRates:
    kx: float = 4442
    ky: float = 715
    kz: float = 13130

    wxy: float = 960
    wyz: float = 0
    wxz: float = 0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            _nonnegative_finite(name, value)

@dataclass(frozen=True)
class InitialPopulations:
    P0: float = 0.9646 
    Px: float = 0.0112
    Py: float = 0.0095
    Pz: float = 0.0147

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            _nonnegative_finite(name, value)

@dataclass(frozen=True)
class LindbladDephasing:
    transition: str
    gamma_phi: float

    def __post_init__(self) -> None:
        if self.transition not in {"xy", "yz", "xz"}:
            raise ValueError(f"transition must be one of 'xy', 'yz', or 'xz', got {self.transition}")
        _nonnegative_finite("gamma_phi", self.gamma_phi)

@dataclass(frozen=True)
class SequenceTiming:
    pi: float = 25e-9
    tr: float = 150e-6
    tw: float = 180e-6
    invert : bool = True
    invert_time: float = 100e-6

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name != "invert":
                _nonnegative_finite(name, value)

        if self.invert and self.invert_time > self.tr:
            raise ValueError("Inversion has to occur within the relaxation time.")


@dataclass(frozen=True)
class PulseEfficiency:
    eta_xy : float = 0.594
    eta_yz : float = 0.776
    eta_xz : float = 0.630

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            _nonnegative_finite(name, value)
            if value > 1:
                raise ValueError(f"{name} must be less than or equal to 1, got {value}") 

@dataclass(frozen=True)
class Transitions:
    xy: tuple = (0, 1)
    yz: tuple = (1, 2)
    xz: tuple = (0, 2)

    