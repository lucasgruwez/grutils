# from importlib import import_module
import importlib
import pkgutil

TEX_HANDLERS = []
ENV_HANDLERS = {}

# auto-import all handler modules
for loader, module_name, _ in pkgutil.iter_modules(__path__):
    mod = importlib.import_module(f"{__name__}.{module_name}")

    # Handlers for generic tex conversions
    if hasattr(mod, "TEX_HANDLERS"):
        TEX_HANDLERS.extend(mod.TEX_HANDLERS)

    # Environment-specific handlers
    if hasattr(mod, "ENV_HANDLERS"):
        ENV_HANDLERS.update(mod.ENV_HANDLERS)
