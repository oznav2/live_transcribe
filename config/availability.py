"""Placeholder for provider availability and model config assembly.

To be implemented in Phase 1c. This module should:
- Detect SDK/library availability using lightweight checks (find_spec, try/except)
- Assemble MODEL_CONFIGS and expose MODEL_SIZE decisions
- Provide Deepgram event enums with defensive fallbacks

Avoid importing heavy ML libraries at module import time.
"""