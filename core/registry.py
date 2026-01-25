import importlib
import pkgutil

class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name: str):
        def decorator(cls):
            self._registry[name] = cls
            return cls
        return decorator
    
    def make(self, name: str, **kwargs):
        if name not in self._registry:
            raise KeyError(f"{name} not found. Available: {list(self._registry.keys())}")
        cls = self._registry[name]
        return cls(**kwargs)

def auto_register(package_name):
    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package_name}.{module_name}")

ENV_REGISTRY = Registry()
ALGO_REGISTRY = Registry()