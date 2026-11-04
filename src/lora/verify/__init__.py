"""Compatibility exports for lora.experimental.verify.

Keep this package path intact so historical child imports resolve through their
forwarding modules and cannot construct duplicate model classes.
"""

__all__ = []


def __getattr__(name):
    from importlib import import_module

    return getattr(import_module("lora.experimental.verify"), name)
