"""Unit tests for L.O.R.A Emergency Kill Switch (src/lora/safety/killswitch.py)."""


from lora.safety.killswitch import KillSwitch


def test_killswitch_initial_state():
    ks = KillSwitch()
    ks.reset()
    assert ks.is_halted() is False
    assert ks.halt_reason == ""


def test_killswitch_trigger_and_reset():
    ks = KillSwitch()
    ks.trigger("Emergency user stop")
    assert ks.is_halted() is True
    assert ks.halt_reason == "Emergency user stop"

    ks.reset()
    assert ks.is_halted() is False
    assert ks.halt_reason == ""


def test_killswitch_callbacks():
    ks = KillSwitch()
    ks.reset()

    callback_fired = []

    def on_halt(reason: str):
        callback_fired.append(reason)

    ks.register_callback(on_halt)
    ks.trigger("Hotkey pressed")

    assert ks.is_halted() is True
    assert len(callback_fired) == 1
    assert callback_fired[0] == "Hotkey pressed"
