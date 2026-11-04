"""Compatibility import for lora.experimental.verify.claims.

The implementation is experimental and is loaded only on explicit import.
"""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("lora.experimental.verify.claims")
