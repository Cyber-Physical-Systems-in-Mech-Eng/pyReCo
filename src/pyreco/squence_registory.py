
sequence_registry = {}


def register_sequence(name):
    def decorator(func):
        if name in sequence_registry:
            raise ValueError(f"Sequence '{name}' already registered")

        sequence_registry[name] = func
        return func
    return decorator
