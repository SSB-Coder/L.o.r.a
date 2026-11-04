# Experimental and historical components

The supported desktop uses `AgentRuntime`, `agent/research.py`,
`voice/desktop_input.py`, and `voice/desktop_output.py`. Normal research performs
cited synthesis and structural reference checks. It does not import this package
or run model-based verification, a supervisor, or prompt evolution.

| Directory | Purpose | Activation |
|---|---|---|
| `agent/` | Deep-review helpers, claim verification, recursive refinement | Explicit API call |
| `verify/` | Standalone claim decomposition, entailment, corroboration and repair | Explicit API call and appropriate local assets |
| `supervisor/` | Experimental execution graph, reviews, checkpointing and round table | Explicit construction |
| `outer_loop/` | Prompt genomes, optimization and offline evaluation | Explicit construction |
| `legacy/` | Older voice/control/research coordinators retained for compatibility | Explicit import/construction |

There is one implementation of each retained component. Old module paths such
as `lora.verify.nli`, `lora.voice.controller`, `lora.core.controller`, and
`lora.grounding.pipeline` forward to the corresponding module object here.
Class identity and module-level patches therefore remain consistent across old
and new imports. Prefer the `lora.experimental` paths in new experimental code.

The old `AgentRuntime.run_rsi_outer_loop` entry point still delegates explicitly
to `agent/refinement.py`. None of the normal research modes invokes it.
The evidence-card conversion used by normal notebooks belongs in
`lora.grounding.cards`; it does not require a verification pipeline.

Install the default runtime and development tools to run the retained controlled
tests. The `experimental` extra supplies optional neural/transformer compatibility
dependencies; it does not run an experiment or fetch weights:

```powershell
uv sync --locked --extra dev --extra experimental
uv run python -m pytest tests/unit/test_outer_loop_loop.py tests/unit/test_supervisor_graph.py tests/unit/test_research_pipeline.py -q
```

Live model evaluations remain explicit (`tests/evals --run-model-evals`) and
require the intended local models. Passing controlled tests is not evidence
that these experiments improve answer quality or fit a particular GPU budget.
Keep their activation and resource budgets separate from the everyday desktop.

GPU Qwen TTS is retired, rather than moved here. Desktop speech and its historical
worker-name compatibility wrapper use CPU Kokoro only. Microphone listening,
deferred STT, speech interruption, sleep/wake, and mobile boundaries are unchanged.
