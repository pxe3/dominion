from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict

class BaseAlgo(ABC):

    @abstractmethod
    def select_action(self, obs) -> Tuple[Any, ...]:
        pass

    def update(self, buffer) -> Dict[str, float]:
        pass

    
