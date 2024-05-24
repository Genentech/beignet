class _DispatchByState:
    def __init__(self, fn):
        self._fn = fn

        self._registry = {}

    def __call__(self, state, *args, **kwargs):
        if type(state.positions) in self._registry:
            return self._registry[type(state.positions)](state, *args, **kwargs)

        return self._fn(state, *args, **kwargs)

    def register(self, oftype):
        def register_fn(fn):
            self._registry[oftype] = fn

        return register_fn
