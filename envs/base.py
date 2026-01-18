from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Any, Dict


class BaseEnv(ABC):
    
    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def action_shape(self) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        pass

        # state --> np.darray 
        # reward -> float
        # done -> bool
        # info -> dict