"""Compatibility import for lora.experimental.supervisor.graph.

The implementation is experimental and is loaded only on explicit import.
"""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("lora.experimental.supervisor.graph")
