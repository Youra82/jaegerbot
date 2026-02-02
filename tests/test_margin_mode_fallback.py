# tests/test_margin_mode_fallback.py
import types

from jaegerbot.utils.exchange import Exchange


class FakeInner:
    def __init__(self):
        self.calls = []

    def set_margin_mode(self, *args, **kwargs):
        # Simulate a failure when called with 3 args (mode, symbol, params)
        self.calls.append(('set_margin_mode', args, kwargs))
        if len(args) == 3:
            raise Exception("unsupported signature")
        return True


def test_set_margin_mode_fallback():
    # Create Exchange instance without running __init__ (avoid ccxt network calls)
    ex = Exchange.__new__(Exchange)
    ex.exchange = FakeInner()
    ex.markets = True

    # Call set_margin_mode - should try with params first, then fallback to simple signature
    success = Exchange.set_margin_mode(ex, 'BTC/USDT:USDT', 'isolated')

    assert success is True
    # Two attempts should have been recorded: first with params (failed), then fallback success
    assert len(ex.exchange.calls) >= 2
    first_call = ex.exchange.calls[0]
    second_call = ex.exchange.calls[1]

    assert first_call[0] == 'set_margin_mode'
    # first call should be invoked with 3 args (mode, symbol, params)
    assert len(first_call[1]) == 3
    assert second_call[0] == 'set_margin_mode'
    # fallback call should be invoked with 2 args (mode, symbol)
    assert len(second_call[1]) == 2
