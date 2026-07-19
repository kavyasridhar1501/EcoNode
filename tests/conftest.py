"""
Stub the heavy/network-dependent dependencies (Prophet's compiled Stan
backend, the Supabase client) so pipeline.py's pure functions can be
imported and unit tested without installing them or reaching the network.
Only the pure data-transformation functions are under test here — nothing
that would need a real Prophet model or a real database connection.
"""

import sys
import types

if "prophet" not in sys.modules:
    prophet_stub = types.ModuleType("prophet")

    class _StubProphet:
        def __init__(self, *args, **kwargs):
            pass

        def add_seasonality(self, *args, **kwargs):
            pass

        def add_regressor(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return self

        def make_future_dataframe(self, *args, **kwargs):
            raise NotImplementedError("Prophet is stubbed out in tests")

        def predict(self, *args, **kwargs):
            raise NotImplementedError("Prophet is stubbed out in tests")

    prophet_stub.Prophet = _StubProphet
    sys.modules["prophet"] = prophet_stub

if "supabase" not in sys.modules:
    supabase_stub = types.ModuleType("supabase")

    class _StubClient:
        pass

    def _stub_create_client(*args, **kwargs):
        raise NotImplementedError("Supabase is stubbed out in tests")

    supabase_stub.Client = _StubClient
    supabase_stub.create_client = _stub_create_client
    sys.modules["supabase"] = supabase_stub
