# Modular Refactor Implementation Plan

Branch: Modular
Objective: Safely refactor `app.py` into modules with identical behavior.

Constraints:
- Do not change function names, parameters, or return types
- Zero functional differences; preserve WebSocket semantics and caches
- Use `start.sh` for server operations and verify via `.logs/`
- Line numbers below refer to `app.py` (from `docs/plans/APP_FUNCTION_INVENTORY.md`)

## Expected Result

### Original Structure 
  - app.py: 3,618 lines (monolithic) 
  - All code in single file 
  - Embedded HTML and CSS 

### New Modular Structure 
  - app.py: ~40-50 lines (entry point only) 
  - 15 module files organized by functionality 
  - 2 static files (HTML, CSS) 
  - Total reduction: ~98.6% from original app.py 

### Files Created 

 Configuration (2 files): 
  - config/settings.py 
  - config/constants.py 

 Core (2 files): 
  - core/lifespan.py 
  - core/state.py 

 Models (1 file): 
  - models/loader.py 

 Services (4 files): 
  - services/audio_processor.py 
  - services/transcription.py 
  - services/diarization.py 
  - services/video_metadata.py 

 API (2 files): 
  - api/routes.py 
  - api/websocket.py 

 Utils (4 files): 
  - utils/cache.py 
  - utils/validators.py 
  - utils/helpers.py 
  - utils/websocket_helpers.py 

 Static (2 files): 
  - static/index.html 
  - static/css/styles.css 

 Docker (2 files): 
  - .dockerignore 
  - DOCKER_BUILD_READY.md 

 Total: 19 new files + 1 minimal app.py

Refactor Guardrails (Mandatory)

- NEVER MODIFY MAIN BRANCH: All work happens on the `Modular` branch
- ZERO BREAKING CHANGES: Application must work identically after refactoring
- PRESERVE EXACT FUNCTIONALITY: Every function must behave the same way
- MAINTAIN EXACT SIGNATURES: Do not change names, parameters, or return types

## 🔒 SAFETY GUARANTEES

By following this plan:
- ✅ Zero breaking changes (same functionality)
- ✅ Zero data loss (caching preserved)
- ✅ Zero performance degradation
- ✅ Zero security regressions
- ✅ 100% backward compatibility
- ✅ Rollback capability at each phase

---

## 🔎 Navigation Aid (app_py_map.md)

- Use `app_py_map.md` to locate `app.py` function signatures and line anchors
  quickly. The map lists each function with its current line numbers to avoid
  scanning the entire file (~3619 lines).
- Workflow: open `app_py_map.md`, find the function, then jump to the indicated
  lines in `app.py` before editing, extracting, or reviewing.
- Keep `app_py_map.md` in sync with `docs/plans/APP_FUNCTION_INVENTORY.md`.
  If line numbers drift, update the map first, then proceed with refactoring.
- Prefer using the map over full-file searches to save tokens and reduce
  navigation overhead.

## 🚫 No Summaries Directive

- Do not add phase-by-phase or section summaries.
- Keep transitions tight and action-focused; avoid verbose recaps.
- Use concise, imperative commit messages without long summaries.
- When updating docs, include only changes and instructions—skip meta summaries.

## 🧠 Token-Saving Instructions for Smooth Implementation

- Read planning docs once at start: `ARCHITECTURE_DESIGN.md`, `IMPLEMENTATION_PLAN.md`,
  `APP_FUNCTION_INVENTORY.md`, `app_py_map.md`.
- Execute Phase 0 (setup) before any extraction.

### 1. Batch File Operations Per Phase (Not Per Function)

Instead of: Creating one file at a time, committing, updating imports, committing again for each function.

Do this:

- Extract ALL functions for a phase into their target files in one session
- Make ALL import changes to `app.py` at once
- Single commit per phase completion

Example for Phase 2 (Utils):

```
# Create all utils files in one go
cat > utils/validators.py << 'EOF'
[all validator functions copied at once]
EOF

cat > utils/helpers.py << 'EOF'
[all helper functions copied at once]
EOF

cat > utils/websocket_helpers.py << 'EOF'
[all websocket helper functions copied at once]
EOF

# Update app.py imports once
# Remove all extracted function definitions once
# Single commit
git add utils/ app.py
git commit -m "PHASE 2 COMPLETE: Extract all utils modules"
```

Token Savings: ~60% fewer commands, single context for related functions.

### 2. Use Function Line Numbers from APP_FUNCTION_INVENTORY.md (Not Grep/Search)

Instead of: Searching for functions in `Old_Files/app.py.original` each time.

Do this:

- Reference `APP_FUNCTION_INVENTORY.md` which lists exact line numbers
- Extract by line range directly

Example:

```
# From inventory: "def is_youtube_url(url) @ line 2129"
sed -n '2129,2141p' Old_Files/app.py.original > temp_function.txt

# Copy to target file
cat temp_function.txt >> utils/validators.py
```

Token Savings: No need to search, no need to describe "find this function", just use line numbers.

### 3. Consolidate Multi-Step Instructions into Single Phase Prompts

Instead of: Asking LLM to complete Step 5.1, then 5.2, then 5.3 separately.

Do this: Give the entire phase at once with clear structure.

Prompt Template:

```
PHASE 5: Extract Audio Processing (Complete this entire phase in one response)

Create: services/audio_processor.py

Extract these from Old_Files/app.py.original (use line numbers from APP_FUNCTION_INVENTORY.md):
- AudioStreamProcessor class (lines 2220-2333) - COMPLETE CLASS AS UNIT
- download_audio_with_ffmpeg + nested helpers (lines 866-1158) - ATOMIC MOVE
- download_with_fallback (lines 1159-1224)
- download_audio_with_ytdlp_async (lines 1225-1459)
- get_audio_duration_seconds (lines 1460-1482)
- calculate_progress_metrics (lines 1483-1530)
- split_audio_for_incremental (lines 1531-1577)
- split_audio_into_chunks (lines 1578-1625)

Required imports at top:
```python
import asyncio
import logging
import os
import subprocess
import tempfile
import shutil
import re
import time
import queue
from pathlib import Path
from typing import Optional, List, Tuple
from fastapi import WebSocket

from config.constants import (
	SAMPLE_RATE,
	CHANNELS,
	CHUNK_DURATION,
	CHUNK_OVERLAP,
	AUDIO_QUEUE_SIZE,
)

from utils.cache import get_cached_download, save_download_to_cache
# Optional: use helpers for WS-safe sends if adopted in code
from utils.websocket_helpers import safe_ws_send
```

Update app.py:
- Remove lines 866-1625 (all extracted functions)
- Add import: from services.audio_processor import [list all function names]

Git commit: "PHASE 5 COMPLETE: Extract audio_processor.py"

Provide the complete services/audio_processor.py file.
```

Token Savings: Single comprehensive instruction vs. 8 separate back-and-forth exchanges.

### 4. Provide Pre-Built Import Statements (Don't Make LLM Figure Out)

Instead of: Letting LLM determine what to import in each module.

Do this: Pre-calculate imports for each module in your plan.

Add to IMPLEMENTATION_PLAN.md for each phase:

#### Phase 6: services/transcription.py

Exact imports to add at top of file:

```python
import asyncio
import logging
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor
from fastapi import WebSocket

from models.loader import load_model
from services.audio_processor import split_audio_for_incremental, get_audio_duration_seconds
from services.diarization import transcribe_with_diarization
from utils.websocket_helpers import safe_ws_send, send_transcription_progress, send_transcription_chunk
from core.state import executor, whisper_models, faster_whisper_models
from config.settings import DEEPGRAM_API_KEY
```

Exact imports to add to app.py (replacing removed code):

```python
from services.transcription import (
    transcribe_with_incremental_output,
    transcribe_chunk,
    transcribe_audio_stream,
    transcribe_vod_with_deepgram,
    transcribe_with_deepgram,
)
```

Token Savings: No "figure out the imports" back-and-forth, copy-paste ready.

#### Phase 5: services/audio_processor.py

Exact imports to add at top of file:

```python
import asyncio
import logging
import os
import subprocess
import tempfile
import shutil
import re
import time
import queue
from pathlib import Path
from typing import Optional, List, Tuple
from fastapi import WebSocket

from config.constants import (
	SAMPLE_RATE,
	CHANNELS,
	CHUNK_DURATION,
	CHUNK_OVERLAP,
	AUDIO_QUEUE_SIZE,
)

from utils.cache import get_cached_download, save_download_to_cache
# Optional: use helpers for WS-safe sends if adopted in code
from utils.websocket_helpers import safe_ws_send
```

Exact imports to add to app.py (replacing removed code):

```python
from services.audio_processor import (
	AudioStreamProcessor,
	download_audio_with_ffmpeg,
	download_with_fallback,
	download_audio_with_ytdlp_async,
	get_audio_duration_seconds,
	calculate_progress_metrics,
	split_audio_for_incremental,
	split_audio_into_chunks,
)
```

#### Phase 8: api/routes.py

Exact imports to add at top of file:

```python
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
import logging
import os
from pathlib import Path
import torch

from utils.validators import is_youtube_url
from utils.helpers import format_duration, format_view_count
from services.video_metadata import get_youtube_metadata
from core.state import (
	cached_index_html,
	current_model,
	current_model_name,
	MODEL_SIZE,
)
from config.constants import CACHE_DIR, DOWNLOAD_CACHE_DIR
```

Prefer router inclusion over direct function imports:

```python
# In api/routes.py
router = APIRouter()

# Example usage in routes module
@router.get('/', response_class=HTMLResponse)
async def get_home():
	return HTMLResponse(content=cached_index_html)

# ... define other endpoints on `router`

# In app.py (after FastAPI app creation)
from api.routes import router as http_router
app.include_router(http_router)
```

If not using routers, alternatively import the functions directly into `app.py`:

```python
from api.routes import (
	get_home,
	health_check,
	get_video_info,
	gpu_diagnostics,
	cache_stats,
	clear_cache,
	download_cache_stats,
	clear_download_cache,
)
```

#### Phase 9: api/websocket.py

Exact imports to add at top of file:

```python
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketState
import asyncio
import logging
import os
import re
import threading
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import torch

from utils.validators import should_use_ytdlp
from utils.websocket_helpers import safe_ws_send
from models.loader import load_model
from services.audio_processor import (
	AudioStreamProcessor,
	download_with_fallback,
	split_audio_into_chunks,
)
from services.transcription import (
	transcribe_audio_stream,
	transcribe_with_incremental_output,
	transcribe_chunk,
	transcribe_vod_with_deepgram,
	transcribe_with_deepgram,
)
from services.diarization import transcribe_with_diarization
from core.state import (
	MODEL_CONFIGS,
	CAPTURE_DIR,
	CAPTURES,
	DOWNLOAD_CACHE_DIR,
)
from config.settings import DEEPGRAM_API_KEY
try:
	from deepgram import DeepgramClient
except Exception:
	DeepgramClient = None  # preserve guardrails if SDK unavailable
```

Prefer router inclusion for the WebSocket endpoint:

```python
# In api/websocket.py
router = APIRouter()

@router.websocket('/ws/transcribe')
async def websocket_transcribe(websocket: WebSocket):
	...

# In app.py
from api.websocket import router as ws_router
app.include_router(ws_router)
```

#### Phase X: services/video_metadata.py

Exact imports to add at top of file:

```python
import asyncio
import logging
import subprocess
import json
from typing import Optional, Dict
```

### 5. Use Diff-Style Instructions for app.py Modifications (Not Full File Rewrites)

Instead of: Regenerating entire `app.py` after each extraction.

Do this: Provide surgical edit instructions.

Example:

```
In app.py, make these changes:

REMOVE (lines 866-1625):
  [8 function definitions that were extracted]

REPLACE (around line 145, in import section):
  # Before:
  import asyncio

  # After:
  import asyncio
  from services.audio_processor import (
      download_audio_with_ffmpeg,
      download_with_fallback,
      [... other functions]
  )

KEEP UNCHANGED:
  - Everything before line 866
  - Everything after line 1625
  - All other imports
  - lifespan function
  - endpoint definitions
```

Token Savings:

- Don't regenerate 3,618 lines of code
- Only show the 10-20 lines that change
- LLM processes ~95% fewer tokens per app.py update

### Phase Completion Checklist (Single End-of-Phase Verification)

- [ ] New module file created with all target functions
- [ ] Functions removed from `app.py` (verify line count decreased)
- [ ] Imports added to `app.py` and new module
- [ ] Single git commit with phase name
- [ ] Ready for next phase

Commands for self-check:

```
wc -l app.py  # Should be smaller
python -c "import services.module_name"  # Should not error
git log -1 --oneline  # Should show phase commit
```

Create Directory Structure

- Create these directories and `__init__.py` files (packages):
  - `config/`, `core/`, `models/`, `services/`, `api/`, `utils/`, `static/css/`, `cache/audio/`, `cache/downloads/`, `cache/captures/`, `logs/`
  - Note: `cache/*` and `logs/` are runtime directories; ensure startup creates them if missing (handled in `core/lifespan.py`)

### Step 0.5: Create Reference Copy of Original Files

PURPOSE: Preserve exact copies of original files in the Modular branch for
reference during migration, eliminating need to switch branches.

CRITICAL: This step copies files FROM main TO Modular branch. Main branch
remains completely untouched.

Commands:

```
# Ensure you are on Modular branch
git branch --show-current
# Expected output: Modular

# Create Old_Files directory
mkdir -p Old_Files

# Copy original files from current state (which is from main)
cp app.py Old_Files/app.py.original
cp Dockerfile.ivrit Old_Files/Dockerfile.ivrit.original
cp docker-compose.ivrit.yml Old_Files/docker-compose.ivrit.yml.original
cp requirements.txt Old_Files/requirements.txt.original
cp requirements.ivrit.txt Old_Files/requirements.ivrit.txt.original

# Note: index.html is embedded in app.py, so it's already captured in app.py.original

# Commit the reference files
git add Old_Files/
git commit -m "REFERENCE: Create Old_Files directory with original source files"

# Verify files are in place
ls -la Old_Files/
```

Expected Output:

```
Old_Files/
├── app.py.original
├── Dockerfile.ivrit.original
├── docker-compose.ivrit.yml.original
├── requirements.txt.original
└── requirements.ivrit.txt.original
```

Usage Instructions for All Subsequent Phases:

CRITICAL RULES:

- NEVER MODIFY MAIN BRANCH
- All implementation happens ONLY on Modular branch
- Never run `git checkout main` during implementation phases
- If you accidentally switch to main, immediately return: `git checkout Modular`

USE Old_Files/ AS REFERENCE SOURCE

- When extracting functions, classes, or code blocks, read from
  `Old_Files/app.py.original`
- When copying exact syntax, method signatures, or logic, reference
  `Old_Files/app.py.original`
- When updating Docker config, compare with `Old_Files/Dockerfile.ivrit.original`
- DO NOT read from `app.py` directly as it will change during refactoring

REFERENCE PATTERN FOR EACH EXTRACTION:

```
# Example: Extracting a function to new module

# Step 1: Read function from Old_Files reference
grep -A 50 "def function_name" Old_Files/app.py.original

# Step 2: Copy EXACT code to new module file
# (copy the function definition exactly as it appears)

# Step 3: Remove from working app.py
# (edit the working app.py, not Old_Files)
```

VERIFICATION PATTERN:

```
# After each extraction, verify syntax matches original:
diff <(grep -A 20 "def function_name" Old_Files/app.py.original) \
     <(grep -A 20 "def function_name" new_module/file.py)

# Should show minimal differences (only imports/indentation changes)
```

HTML/CSS EXTRACTION:

- In Phase 10, extract HTML from `Old_Files/app.py.original`
- Search for the HTML string variable (likely starts with `html_content = """`
  or similar)
- Extract inline `<style>` tags to create `static/css/stylesheet.css`

Quick Reference Commands:

```
# To view function in original file
grep -A 30 "def function_name" Old_Files/app.py.original

# To search for specific code pattern
grep -n "pattern" Old_Files/app.py.original

# To extract line range
sed -n '100,150p' Old_Files/app.py.original

# To count remaining functions in working app.py
grep -c "^def " app.py
grep -c "^async def " app.py

# To verify you're on Modular branch
git branch --show-current
```

---

### CHECKPOINT 0.5

- [ ] On Modular branch (verify with `git branch --show-current`)
- [ ] `Old_Files/` directory created with 5 files
- [ ] All files have `.original` extension
- [ ] Files committed to Modular branch
- [ ] Main branch still pristine (verify with `git log main --oneline -1`)
- [ ] Understand: ALL code reading happens from `Old_Files/`, ALL code writing
  happens to working files

If any checkpoint fails, STOP and fix before proceeding to Phase 1.

---

## Reminder for ALL Phases

At the start of EVERY phase, remind yourself:

```
✅ Working on: Modular branch
✅ Reading from: Old_Files/*.original
✅ Writing to: Working files (app.py, new modules)
❌ NEVER touch: Main branch
❌ NEVER modify: Old_Files/*.original (read-only reference)
```

- Function-to-Module Assignment (with line numbers):
  - 158 `async lifespan(app)` → `core/lifespan.py`
  - 241 `async safe_ws_send(websocket, data)` → `utils/websocket_helpers.py`
  - 359 `def init_capture_dir()` → `utils/cache.py`
  - 366 `def init_download_cache_dir()` → `utils/cache.py`
  - 387 `def get_url_hash(url)` → `utils/cache.py`
  - 392 `def get_cached_download(url)` → `utils/cache.py`
  - 417 `def save_download_to_cache(url, audio_file)` → `utils/cache.py`
  - 452 `def load_model(model_name)` → `models/loader.py` (see also `models/whisper_manager.py` for future management)
  - 553 `def get_diarization_pipeline()` → `models/loader.py`
  - 591 `async transcribe_with_diarization(model, model_config, audio_file, language, websocket, model_name)` → `services/diarization.py`
  - 833 `def should_use_ytdlp(url)` → `utils/validators.py`
  - 866 `async download_audio_with_ffmpeg(url, format, duration, websocket, use_cache)` → `services/audio_processor.py`
  - 1159 `async download_with_fallback(url, language, format, websocket, use_cache)` → `services/audio_processor.py`
  - 1225 `async download_audio_with_ytdlp_async(url, language, format, websocket, use_cache)` → `services/audio_processor.py`
  - 1460 `def get_audio_duration_seconds(audio_path)` → `services/audio_processor.py`
  - 1483 `def calculate_progress_metrics(audio_duration, elapsed_time, processed_chunks, total_chunks)` → `services/audio_processor.py`
  - 1531 `def split_audio_for_incremental(audio_path, chunk_seconds, overlap_seconds)` → `services/audio_processor.py`
  - 1578 `def split_audio_into_chunks(audio_path, chunk_seconds, overlap_seconds)` → `services/audio_processor.py`
  - 1626 `async transcribe_with_incremental_output(model, model_config, audio_file, language, websocket, model_name, chunk_seconds)` → `services/transcription.py`
  - 1977 `def transcribe_chunk(model_config, model, chunk_path, language)` → `services/transcription.py`
  - 2033 `def init_cache_dir()` → `utils/cache.py`
  - 2057 `def generate_cache_key(audio_data, sample_rate, channels)` → `utils/cache.py`
  - 2065 `def get_cached_audio(cache_key)` → `utils/cache.py`
  - 2077 `def save_to_cache(cache_key, audio_path)` → `utils/cache.py`
  - 2099 `def format_duration(seconds)` → `utils/helpers.py`
  - 2114 `def format_view_count(count)` → `utils/helpers.py`
  - 2129 `def is_youtube_url(url)` → `utils/validators.py`
  - 2144 `async get_youtube_metadata(url)` → `services/video_metadata.py`
  - 2335 `async transcribe_audio_stream(websocket, processor)` → `services/transcription.py`
  - 2502 `async transcribe_vod_with_deepgram(websocket, url, language)` → `services/transcription.py`
  - 2781 `async transcribe_with_deepgram(websocket, url, language)` → `services/transcription.py`
  - 3068 `async get_home()` [GET `/`] → `api/routes.py`
  - 3074 `async websocket_transcribe(websocket)` [WebSocket `/ws/transcribe`] → `api/websocket.py`
  - 3419 `async health_check()` [GET `/health`] → `api/routes.py`
  - 3429 `async get_video_info(request)` [POST `/api/video-info`] → `api/routes.py`
  - 3485 `async gpu_diagnostics()` [GET `/gpu`] → `api/routes.py`
  - 3518 `async cache_stats()` [GET `/api/cache/stats`] → `api/routes.py`
  - 3538 `async clear_cache()` [POST `/api/cache/clear`] → `api/routes.py`
  - 3555 `async download_cache_stats()` [GET `/api/download-cache/stats`] → `api/routes.py`
  - 3584 `async clear_download_cache()` [POST `/api/download-cache/clear`] → `api/routes.py`

## 📚 Module Assignment Matrix

Generate and maintain this table for every function moved to a module. Use the risk levels LOW (pure/utility), MEDIUM (I/O, cache, minor side-effects), HIGH (concurrency, WebSocket/session lifecycle, heavy libs). Dependencies should list key modules/libs and any global state or locks.

| Original Function | Target Module | Risk Level | Dependencies |
|-------------------|---------------|------------|--------------|
| lifespan(app) | core/lifespan.py | MEDIUM | asyncio, pathlib, logging, config.settings, core.state |
| safe_ws_send(websocket, data) | utils/websocket_helpers.py | HIGH | websockets, logging |
| send_status_update(websocket, message) | utils/websocket_helpers.py | HIGH | websockets, logging |
| send_transcription_progress(websocket, progress, total_duration) | utils/websocket_helpers.py | HIGH | websockets, logging |
| send_transcription_chunk(websocket, text, start, end, speaker?) | utils/websocket_helpers.py | HIGH | websockets, logging |
| send_transcription_status(websocket, status) | utils/websocket_helpers.py | HIGH | websockets, logging |
| init_capture_dir() | utils/cache.py | MEDIUM | os, pathlib |
| init_download_cache_dir() | utils/cache.py | MEDIUM | os, pathlib |
| get_url_hash(url) | utils/cache.py | LOW | hashlib |
| get_cached_download(url) | utils/cache.py | MEDIUM | os, pathlib |
| save_download_to_cache(url, audio_file) | utils/cache.py | MEDIUM | os, pathlib |
| load_model(model_name) | models/loader.py | HIGH | threading, config.settings, core.state |
| get_diarization_pipeline() | models/loader.py | HIGH | threading, core.state |
| transcribe_with_diarization(...) | services/diarization.py | HIGH | models.loader, core.state, asyncio |
| should_use_ytdlp(url) | utils/validators.py | LOW | re (if used) |
| download_audio_with_ffmpeg(...) | services/audio_processor.py | HIGH | ffmpeg/subprocess, asyncio, websockets |
| download_with_fallback(...) | services/audio_processor.py | HIGH | ytdlp, ffmpeg, asyncio, websockets |
| download_audio_with_ytdlp_async(...) | services/audio_processor.py | HIGH | ytdlp, asyncio, websockets |
| get_audio_duration_seconds(audio_path) | services/audio_processor.py | MEDIUM | ffprobe/mediainfo, pathlib |
| calculate_progress_metrics(...) | services/audio_processor.py | LOW | math |
| split_audio_for_incremental(...) | services/audio_processor.py | MEDIUM | audio libs, pathlib |
| split_audio_into_chunks(...) | services/audio_processor.py | MEDIUM | audio libs, pathlib |
| transcribe_with_incremental_output(...) | services/transcription.py | HIGH | models.loader, asyncio, websockets |
| transcribe_chunk(...) | services/transcription.py | MEDIUM | models.loader |
| init_cache_dir() | utils/cache.py | MEDIUM | os, pathlib |
| generate_cache_key(...) | utils/cache.py | LOW | hashlib |
| get_cached_audio(cache_key) | utils/cache.py | MEDIUM | os, pathlib |
| save_to_cache(cache_key, audio_path) | utils/cache.py | MEDIUM | os, pathlib |
| format_duration(seconds) | utils/helpers.py | LOW | datetime |
| format_view_count(count) | utils/helpers.py | LOW | math |
| is_youtube_url(url) | utils/validators.py | LOW | re |
| get_youtube_metadata(url) | services/video_metadata.py | MEDIUM | http client, asyncio |
| transcribe_audio_stream(websocket, processor) | services/transcription.py | HIGH | services.audio_processor, asyncio, websockets |
| transcribe_vod_with_deepgram(websocket, url, language) | services/transcription.py | HIGH | deepgram sdk, asyncio, websockets |
| transcribe_with_deepgram(websocket, url, language) | services/transcription.py | HIGH | deepgram sdk, asyncio, websockets |
| get_home() | api/routes.py | LOW | core.state (cached HTML), static files |
| websocket_transcribe(websocket) | api/websocket.py | HIGH | services.audio_processor, services.transcription, core.state |
| health_check() | api/routes.py | LOW | config.settings, core.state |
| get_video_info(request) | api/routes.py | MEDIUM | services.video_metadata, utils.validators |
| gpu_diagnostics() | api/routes.py | MEDIUM | torch/cuda libs (if any), logging |
| cache_stats() | api/routes.py | LOW | utils.cache |
| clear_cache() | api/routes.py | MEDIUM | utils.cache |
| download_cache_stats() | api/routes.py | LOW | utils.cache |
| clear_download_cache() | api/routes.py | MEDIUM | utils.cache |

Continue for all 48 functions using `docs/plans/APP_FUNCTION_INVENTORY.md` and the mapping above. Update risk and dependencies based on each phase’s “Dependencies Required”.

## 🗺️ Import Dependency Map

Keep this tree updated as modules are extracted. It should reflect actual imports, not intended design. Validate via static analysis and runtime logs.

```
app.py
  ├── config.settings
  ├── config.constants
  ├── config.availability
  ├── core.lifespan
  ├── core.state
  ├── api.routes
  │   ├── utils.cache
  │   ├── utils.validators
  │   ├── services.video_metadata
  │   └── core.state
  ├── api.websocket
  │   ├── services.audio_processor
  │   ├── services.transcription
  │   ├── utils.websocket_helpers
  │   ├── models.loader
  │   ├── utils.cache
  │   ├── utils.validators
  │   ├── config.availability
  │   └── core.state
  ├── models.loader
  ├── utils.helpers
  └── static (via api.routes.get_home and core.state cache)

services.transcription
  ├── models.loader
  ├── services.diarization
  ├── utils.cache
  ├── utils.validators
  ├── core.state
  └── utils.websocket_helpers

services.audio_processor
  ├── utils.cache
  ├── utils.validators
  ├── core.state
  └── utils.websocket_helpers

models.loader
  ├── config.settings
  ├── config.availability
  └── core.state

core.lifespan
  ├── config.settings
  ├── config.availability
  ├── core.state
  └── utils.cache (directory creation)

api.routes
  ├── utils.cache
  ├── utils.validators
  └── services.video_metadata

api.websocket
  ├── services.audio_processor
  ├── services.transcription
  └── utils.websocket_helpers
  ├── config.availability

[complete tree maintained during refactor]
```

## ⚠️ Critical Considerations

1. Thread Safety: Locking required in `models/loader.py` (model cache), `core/state.py` (global singletons), and diarization pipeline access. Maintain atomic updates and use existing locks.
2. Global State: Centralize in `core/state.py`. Preserve singletons (`current_model`, `diarization_pipeline`, cached HTML, executor) and access patterns. Do not duplicate or reinitialize in submodules.
3. Async Operations: Async-heavy modules include `api.websocket`, `api.routes` (handlers), `services.audio_processor`, `services.transcription`, and `services.diarization`. Preserve async boundaries and avoid blocking calls in the event loop.
4. WebSocket State: Enforce connected state before sends via `safe_ws_send`. Maintain message sequencing, progress updates, and disconnect flows exactly.
5. Docker Compatibility: Ensure `Dockerfile*` COPY updated directories (`config`, `core`, `models`, `services`, `api`, `utils`, `static`). Confirm runtime directories (`cache/*`, `logs/`) exist at startup via `core/lifespan.py`.

6. Import Order: `config.settings` → `config.constants` → `config.availability` → `core.state` → `core.lifespan` → `models.loader` → `services/*` → `api/*`. This order avoids `NameError`, ensures environment availability, and prevents circular imports.
7. Availability & Events: Consolidate provider flags and Deepgram event probing in `config/availability.py` with defensive checks against SDK variations.

---

## 🔒 High-Risk Breakpoints Mitigation

- Import Order & Env Loading
  - Always import `config.settings` first and call `load_dotenv()` before any env reads
  - Import sequence: `config.settings` → `config.constants` → `config.availability` → `core.state` → `core.lifespan` → `models.loader` → `services/*` → `api/*`
  - Never import heavy ML libs inside `config/availability.py`; use `find_spec` and guarded try/except

- Circular Import Prevention
  - `core/*` must be leaf-only; do not import `services/*` or `api/*` from `core`
  - `models/loader.py` imports only `config.settings`, `config.availability`, `core.state`
  - `core/lifespan.py` reads flags/configs from `config.availability` only; it must not import `models.loader`

- Shared State & Cache Ownership
  - All globals (`current_model`, `diarization_pipeline`, `cached_index_html`, `CAPTURES`, `URL_DOWNLOADS`) live in `core.state`
  - Prohibit global `ThreadPoolExecutor`; create executors per-use and keep local
  - `get_home()` reads `core.state.cached_index_html` at runtime; do not capture module-local snapshots

- Deepgram Event Probing & Streaming
  - Compute `DG_EVENT_*` symbols in `config/availability` with attribute checks and fallbacks
  - Preserve `asyncio.run_coroutine_threadsafe` bridging in Deepgram callbacks; maintain thread-boundary semantics
  - Enforce generous finish waits and timeout handling with finalization signals

- Router Conversion Semantics
  - Use `APIRouter`; call `app.include_router(...)` only after `app = FastAPI(...)`
  - Route handlers must reference shared state from `core.state` and avoid capturing defaults at import time

- Audio Processing & Class Extraction
  - Extract `AudioStreamProcessor` atomically; preserve `ffmpeg_process`, `audio_queue`, `is_running`, and `stop()` lifecycle
  - Maintain queue backpressure, overlap behavior, and identical shutdown semantics

---

- Dockerfiles: Update `Dockerfile` and `Dockerfile.ivrit` to COPY new directories (`config`, `core`, `models`, `services`, `api`, `utils`, `static`, and runtime `cache` creation if needed).

## 📁 EXPECTED FINAL STRUCTURE

```
/home/user/webapp/
├── app.py                          # ✅ NEW: ~150 lines (main entry)
├── config/
│   ├── __init__.py
│   ├── settings.py                 # ✅ Environment variables
│   ├── constants.py                # ✅ Application constants
│   └── availability.py             # ✅ Provider flags, events, MODEL_CONFIGS
├── core/
│   ├── __init__.py
│   ├── lifespan.py                 # ✅ Startup/shutdown logic
│   └── state.py                    # ✅ Global state management
├── models/
│   ├── __init__.py
│   ├── loader.py                   # ✅ Thread-safe model loading
│   └── whisper_manager.py          # ✅ Model management
├── services/
│   ├── __init__.py
│   ├── audio_processor.py          # ✅ Download, split, cache
│   ├── transcription.py            # ✅ Core transcription logic
│   ├── diarization.py              # ✅ Speaker diarization
│   └── video_metadata.py           # ✅ YouTube metadata
├── api/
│   ├── __init__.py
│   ├── routes.py                   # ✅ HTTP endpoints
│   └── websocket.py                # ✅ WebSocket endpoint
├── utils/
│   ├── __init__.py
│   ├── cache.py                    # ✅ Caching utilities
│   ├── validators.py               # ✅ Input validation
│   ├── helpers.py                  # ✅ General utilities
│   └── websocket_helpers.py        # ✅ WebSocket helper utilities
├── static/
│   ├── index.html                  # ✅ EXTRACTED from app.py
│   └── css/
│       └── stylesheet.css          # ✅ EXTRACTED from app.py
├── requirements.txt                # ✅ Unchanged
├── requirements.ivrit.txt          # ✅ Unchanged
├── Dockerfile                      # ⚠️ UPDATED (COPY commands)
├── Dockerfile.ivrit                # ⚠️ UPDATED (COPY commands)
├── cache/                          # Runtime directories
│   ├── audio/
│   ├── downloads/
│   └── captures/
└── logs/                           # Runtime logs
```

---

 ## 🎯 SUCCESS CRITERIA

 The refactoring is successful when:

 1. ✅ Application starts without any errors
 2. ✅ All endpoints return correct responses
 3. ✅ WebSocket transcription works end-to-end
 4. ✅ Docker builds succeed (both Dockerfiles)
 5. ✅ All tests pass (functional and technical)
 6. ✅ Performance is equal or better than original
 7. ✅ Code is more maintainable and organized
 8. ✅ No functionality is lost or changed
 9. ✅ Documentation is updated
 10. ✅ Team can understand and modify code easily

 ---

 ## 🚀 EXECUTION STRATEGY

 1. **Phase-by-Phase**: Complete one phase before starting next
 2. **Commit After Each Phase**: Git commit for rollback capability
 3. **Reflect After Each Phase**: Use reflection template
 4. **Validate Imports**: Test imports in isolation before integration
 5. **Maintain Backup**: Keep `app.py.backup` until fully complete

## 🛠️ REFACTORING GUIDELINES

 1. Read First, Code Second: Understand entire `app.py` before extracting
 2. Copy Exactly: Don't "improve" code during refactoring
 3. Test Continuously: Run application after each change
 4. Check WebSocket State: Always verify WebSocket is connected before sending
 5. Preserve Async: Never convert async to sync or vice versa
 6. Import at Top: Place all imports at module top (PEP 8)
 7. Use Type Hints: Add type hints for clarity (do not change signatures)
 8. Handle Errors: Preserve all try/except blocks exactly
 9. Test in Docker: Ensure Docker containers support functionality before finalizing
 10. Ask for Help: If unsure, ask for clarification before proceeding

11. Atomic Moves for Nested Closures and Classes
  - Move parent function together with all nested helper functions in one commit
  - Do not split nested closures across phases; preserve closure variables and shared state
  - Extract classes as complete units, including all methods and constructor
  - Validate that all references and imports remain consistent after move

---

## 📐 UI Contract Preservation (Mandatory)

The UI in `static/index.html` relies on specific function names, payload shapes, DOM IDs, and WebSocket message types. During refactor, preserve this contract exactly.

- Connection Path: Keep WebSocket endpoint at `/ws/transcribe`
- Client Request Shape: First message must be JSON with keys:
  - `url` (string), `model` (string), optional `language` (string), optional `captureMode` (string), optional `diarization` (boolean)
- Server Message Types: Preserve `type` and payload keys for all messages:
  - `status` → `{ type: 'status', message }`
  - `download_progress` → `{ type: 'download_progress', downloaded, total, percent }`
  - `cached_file` → `{ type: 'cached_file', message }`
  - `transcription_progress` → `{ type: 'transcription_progress', progress, total_duration }`
  - `transcription_chunk` → `{ type: 'transcription_chunk', text, start?, end?, speaker? }`
  - `transcription` → `{ type: 'transcription', text, language }` (stream mode)
  - `capture_ready` → `{ type: 'capture_ready', filename }`
  - `complete` → `{ type: 'complete', message }`
  - `error` → `{ type: 'error', error }`
- Sequencing & State:
  - Accept WebSocket, then expect a single JSON request before sending any progress
  - Guard sends with `websocket.client_state === CONNECTED` via `safe_ws_send`
  - Maintain heartbeats: send `{ type: 'status', message: 'Waiting for audio...' }` when idle
  - Always send a final `{ type: 'complete', message }` unless an `error` was sent
- HTML/JS Integration:
  - Do not rename UI-facing functions or IDs used by JS handlers
  - Keep cached HTML behavior: `get_home()` returns `cached_index_html` exactly
  - When extracting static assets, ensure paths remain `static/index.html` and `static/css/stylesheet.css`
- Validation Protocol:
  - After each phase, connect from browser and verify onmessage switch handles all `type` values above
  - Compare payload keys against `APP_PY_ANALYSIS.md` examples; no additions/removals
  - Use `start.sh` and check `.logs/server-output.log` for sequencing/log statements
  - Verify `cached_index_html` lifecycle: set in lifespan, read via `core.state` inside `get_home()`
  - Confirm environment loading early: `load_dotenv()` executed before any env reads; `DEEPGRAM_API_KEY` present
  - Ensure availability flags and `MODEL_CONFIGS` originate from a single module (`config.availability`) and match pre-refactor values
  - Check `AudioStreamProcessor` fields remain (`audio_queue`, `ffmpeg_process`, `is_running`) and `stop()` behavior unchanged

These rules ensure the JavaScript-driven UI continues to operate without modification.

PHASE 0: Scaffolding & Prep (LOW RISK)

- Functions to Extract:
  - None

- Target Module:
  - Create: `config/`, `core/`, `models/`, `services/`, `api/`, `utils/`, `static/css/`

- Dependencies Required:
  - Imports: None
  - Imported by: N/A

- Code to Copy:
  - Add `__init__.py` in each new directory
  - Prepare placeholder files (empty module shells)
  - Ensure no behavioral change

- Import Changes in app.py:
  - None

- Git Commit Message:
  - `CHORE: Scaffold modular directories and placeholders`

- Critical Notes:
  - Keep runtime identical; do not alter app initialization
  - Do not mount static yet; UI behavior must remain unchanged

PHASE 0.1: Environment Bootstrap (CRITICAL)

- Functions to Extract:
  - None

- Target Module:
  - `config/settings.py`

- Dependencies Required:
  - Imports: `os`, `typing`, `dotenv`
  - Imported by: `core/state.py`, `config/constants.py`, `config/availability.py`, `models/loader.py`, `api/*`, `services/*`

- Code to Copy:
  - Add `from dotenv import load_dotenv` and call `load_dotenv()` at the very top of `config/settings.py`
  - Define all `os.getenv(...)` after `load_dotenv()` to ensure local `.env` keys are available

- Import Changes in app.py:
  - Ensure `config.settings` is imported before any module that reads env-derived settings

- Critical Notes:
  - `load_dotenv()` MUST execute before importing `config/constants.py`, `config/availability.py`, or any module that reads env variables
  - This ordering prevents missing keys (e.g., `DEEPGRAM_API_KEY`) and wrong device selections

---

PHASE 1a: Extract Configuration (LOW RISK)

- Functions to Extract:
  - None (environment variable assignments)

- Target Module:
  - `config/settings.py`

- Dependencies Required:
  - Imports: `os`, `typing`
  - Imported by: `core/state.py`, `services/*`, `api/*`, `models/loader.py`

- Code to Copy:
  - All `os.getenv(...)` assignments (e.g., `DEEPGRAM_API_KEY`, model names, device settings)
  - Preserve default values and types
  - Include derived flags (e.g., booleans parsed from strings)

- Import Changes in app.py:
  - Remove: inline `os.getenv(...)` definitions
  - Add: `from config.settings import (DEEPGRAM_API_KEY, WHISPER_MODEL, IVRIT_MODEL_NAME, IVRIT_DEVICE, IVRIT_COMPUTE_TYPE, ...)`
  - Example: `from config.settings import DEEPGRAM_API_KEY`

- Git Commit Message:
  - `EXTRACT: Environment configuration to config/settings.py`

- Critical Notes:
  - Only move definitions; keep usage unchanged
  - Validate via `/health` and `/gpu` after changes using `start.sh`
  - Cross-check: Ensure `config.settings` is imported before `config.availability` and any module reading `MODEL_CONFIGS`; confirm no NameError during lifespan

---

PHASE 1b: Core Lifespan & State (MEDIUM RISK)

- Functions to Extract:
  - `lifespan(app)` @ line 158 (async)
  - Globals to centralize in `core/state.py`: `whisper_models`, `current_model`, `current_model_name`, `model_lock`, `diarization_pipeline`, `diarization_pipeline_lock`, `cached_index_html`, `CAPTURES`, any progress debounce variables

- Target Module:
  - `core/lifespan.py`
  - `core/state.py`

- Dependencies Required:
  - Imports (lifespan): `asyncio`, `concurrent.futures`, `os`, `pathlib.Path`, `logging`, `config.settings`
  - Imports (state): `threading`, `typing`, `config.settings`
  - Imported by: `app.py`, `api/*`, `services/*`, `models/loader.py`

- Code to Copy:
  - Lifespan startup/shutdown, directory creation, HTML cache bootstrap
  - Global state declarations and thread locks in one place
  - Ensure functions and modules reference `core.state` globals
  - This must happen BEFORE Phase 4 (models), Phase 5 (audio), Phase 6 (transcription)
  - `core/state.py` (globals):
```
whisper_models = {}
faster_whisper_models = {}
diarization_pipeline = None
model_lock = threading.Lock()
faster_whisper_lock = threading.Lock()
diarization_lock = threading.Lock()
cached_index_html = None
CAPTURES = {}
URL_DOWNLOADS = {}
```

- Import Changes in app.py:
  - Remove: inline `lifespan` function and global variable declarations
  - Add: `from core.lifespan import lifespan`; `from core import state`
  - Example: `app = FastAPI(..., lifespan=lifespan)`

- Git Commit Message:
  - `EXTRACT: Lifespan and global state into core/`

- Critical Notes:
  - Preserve lock semantics and singletons
  - Avoid circular imports by keeping `core/*` leaf-only
  - Cross-check: Verify `core/lifespan.py` imports `config.availability` after `config.settings`; confirm `MODEL_CONFIGS` availability without cycles

---

Global Variable Migration Checklist (After Phase 1b)

- [ ] Remove duplicated globals from `app.py`
- [ ] Migrate all references to `core.state`
- [ ] Verify locks guard writes and reads (`model_lock`, `faster_whisper_lock`, `diarization_lock`)
- [ ] Do not introduce a global executor; keep per-use `ThreadPoolExecutor` local
- [ ] Confirm `cached_index_html` populated in lifespan and used by `get_home`
- [ ] Search import graph for accidental reinit; fix via `core.state` import
- [ ] Validate no circular imports between `core.state` and submodules
- [ ] Ensure `URL_DOWNLOADS` and `CAPTURES` are only defined in `core.state` and referenced everywhere

---
PHASE 1c: Provider Availability & Model Config Assembly (HIGH RISK)

- Functions to Extract:
  - Availability detection: `FASTER_WHISPER_AVAILABLE`, `OPENAI_WHISPER_AVAILABLE`, `DEEPGRAM_AVAILABLE`, optional `IVRIT_AVAILABLE`, `PYANNOTE_AVAILABLE`
  - Deepgram event probing: compute `DG_EVENT_OPEN`, `DG_EVENT_CLOSE`, `DG_EVENT_ERROR`, `DG_EVENT_DATA` safely
  - `MODEL_CONFIGS` and `MODEL_SIZE` assembly based on env and availability

- Target Module:
  - `config/availability.py`

- Dependencies Required:
  - Imports: lightweight checks only; use `importlib.util.find_spec` and guarded try/except around optional SDKs
  - Imported by: `core/lifespan.py`, `core.state.py`, `api/websocket.py`, `models/loader.py`, `services/transcription.py`

- Code to Copy:
  - Move top-level availability detection and Deepgram event probing from `app.py` into `config/availability.py`
  - Assemble `MODEL_CONFIGS` using `config.settings` and availability flags; expose `MODEL_SIZE`
  - Keep probing guards resilient to SDK changes (attribute checks with fallbacks)

- Import Changes in app.py:
  - Add: `from config.availability import (FASTER_WHISPER_AVAILABLE, OPENAI_WHISPER_AVAILABLE, DEEPGRAM_AVAILABLE, DG_EVENT_OPEN, DG_EVENT_CLOSE, DG_EVENT_ERROR, DG_EVENT_DATA, MODEL_CONFIGS, MODEL_SIZE)`
  - Ensure `config.settings` imports precede `config.availability`

- Critical Notes:
  - Do NOT import heavy ML libs in `config/availability.py`; perform capability checks via lightweight imports or `find_spec`
  - `core/lifespan.py` should read flags and configs from `config.availability` only; it must not import `models.loader` to avoid cycles
  - All modules must use `MODEL_CONFIGS` from a single source (`config.availability`)
  - Cross-check: Validate import order `config.settings` → `config.availability`; confirm `MODEL_CONFIGS` resolves in `core.lifespan`, `models.loader`, `services/transcription` without circular imports

PHASE 2: Utils (Validators, Helpers, WebSocket Helpers) (LOW RISK)

- Functions to Extract:
  - Helpers (`utils/helpers.py`):
    - `format_duration(seconds)` @ 2099
    - `format_view_count(count)` @ 2114
  - Validators (`utils/validators.py`):
    - `is_youtube_url(url)` @ 2129
    - `should_use_ytdlp(url)` @ 833
  - WebSocket Helpers (`utils/websocket_helpers.py`):
    - `safe_ws_send(websocket, data)` @ 241 (async)
    - `send_status_update(websocket, message)` (async)
    - `send_transcription_progress(websocket, progress)` (async)
    - `send_transcription_chunk(websocket, chunk, meta)` (async)
    - `send_transcription_status(websocket, status)` (async)

- Target Module:
  - `utils/helpers.py`, `utils/validators.py`, `utils/websocket_helpers.py`

- Dependencies Required:
  - Imports: `logging`, `re`, `datetime`, `math`, `typing`
  - Imported by: `services/*`, `api/websocket.py`, `api/routes.py`

- Code to Copy:
  - Pure helper and validator functions unchanged
  - Create new `utils/websocket_helpers.py` and move all five WebSocket helper functions here
  - Do not alter message shapes or keys

- Import Changes in app.py:
  - Remove: inline helpers/validators/WebSocket helper definitions
  - Add: `from utils.helpers import format_duration, format_view_count`
  - Add: `from utils.validators import is_youtube_url, should_use_ytdlp`
  - Add: `from utils.websocket_helpers import (\
      safe_ws_send,\
      send_status_update,\
      send_transcription_progress,\
      send_transcription_chunk,\
      send_transcription_status,\
    )`

- Git Commit Message:
  - `EXTRACT: Helpers, validators, and WebSocket helpers to utils/`

- Critical Notes:
  - This extraction happens BEFORE services (audio/transcription) to eliminate circular deps
  - `services/*` and `api/websocket.py` both import from `utils.websocket_helpers`
  - Preserve async boundaries and error handling behavior
  - Cross-check: Confirm utils do not import `config.availability`; avoid back-edges that may reorder `MODEL_CONFIGS` assembly

---

PHASE 3: Utils & Cache (MEDIUM RISK)

- Functions to Extract:
  - Cache (`utils/cache.py`):
    - `init_capture_dir()` @ 359
    - `init_download_cache_dir()` @ 366
    - `get_url_hash(url)` @ 387
    - `get_cached_download(url)` @ 392
    - `save_download_to_cache(url, audio_file)` @ 417
    - `init_cache_dir()` @ 2033
    - `generate_cache_key(audio_data, sample_rate, channels)` @ 2057
    - `get_cached_audio(cache_key)` @ 2065
    - `save_to_cache(cache_key, audio_path)` @ 2077
  - Note: Validators and helpers are extracted earlier in Phase 2

- Target Module:
  - `utils/cache.py`

- Dependencies Required:
  - Imports: `hashlib`, `os`, `pathlib.Path`, `typing`
  - Imported by: `services/audio_processor.py`, `api/routes.py`

- Code to Copy:
  - All cache directory and on-disk operations, hash utilities, stats functions
  - Pure helpers and validation predicates
  - No behavioral change; preserve file layout and semantics

- Import Changes in app.py:
  - Remove: inline cache/helper/validator function definitions
  - Add: `from utils.cache import get_url_hash, ...`
  - Example: `from utils.validators import is_youtube_url, should_use_ytdlp`

- Git Commit Message:
  - `EXTRACT: Cache utilities to utils/cache.py`

- Critical Notes:
  - Keep SHA256 strategy and directory paths identical
  - Ensure functions are imported where used (audio, api, websocket)

---

PHASE 4: Audio Services (HIGH RISK)

- Functions to Extract:
  - `download_audio_with_ffmpeg(url, format, duration, websocket, use_cache)` @ 866 (async)
  - `download_with_fallback(url, language, format, websocket, use_cache)` @ 1159 (async)
  - `download_audio_with_ytdlp_async(url, language, format, websocket, use_cache)` @ 1225 (async)
  - `get_audio_duration_seconds(audio_path)` @ 1460
  - `calculate_progress_metrics(audio_duration, elapsed_time, processed_chunks, total_chunks)` @ 1483
  - `split_audio_for_incremental(audio_path, chunk_seconds, overlap_seconds)` @ 1531
  - `split_audio_into_chunks(audio_path, chunk_seconds, overlap_seconds)` @ 1578
  - Class: `AudioStreamProcessor.__init__` @ 2220; `start_ffmpeg_stream` @ 2229; `read_audio_chunks` @ 2258; `stop` @ 2321
  - Nested helpers: `monitor_progress` @ 944; `read_progress_file` @ 960

- ATOMIC MOVE:
  - Move `download_audio_with_ffmpeg` together with its nested `monitor_progress` and `read_progress_file` in a single commit

- Class Extraction:
  - **CRITICAL ATOMIC MOVE**: Move entire `AudioStreamProcessor` class (lines 2220–2333) in one commit with ALL its internal state unchanged:
    - `ffmpeg_process` (subprocess handle)
    - `audio_queue` (asyncio.Queue with backpressure)
    - `is_running` (lifecycle flag)
    - `stop()` method semantics
  - **HIGH RISK**: Do not introduce new fields or change behavior; preserve subprocess lifecycle, async queue flow, and shutdown exactly

- Target Module:
  - `services/audio_processor.py`

- Dependencies Required:
  - Imports: `asyncio`, `subprocess`, `re`, `time`, `pathlib.Path`, `utils.cache`, `utils.validators`, `config.settings`, `logging`
  - Imported by: `api/websocket.py`, `services/transcription.py`

- Code to Copy:
  - FFmpeg and yt-dlp orchestration, progress reporting, chunking, stream processing class
  - Preserve WebSocket progress semantics and debouncing
  - Maintain cache-aware download behavior and fallback logic

- Import Changes in app.py:
  - Remove: audio download and processing functions/classes
  - Add: `from services.audio_processor import AudioStreamProcessor, download_with_fallback, ...`
  - Example: `from services.audio_processor import split_audio_for_incremental`

- Git Commit Message:
  - `EXTRACT: Audio download, chunking, and stream processing to services/audio_processor.py`
  - `ATOMIC MOVE: AudioStreamProcessor class to services/audio_processor.py`

- Critical Notes:
  - Keep non-blocking file reads via executor in progress monitor
  - Ensure paths and temp files cleanup behavior remain identical
  - Cross-check: Audio services must not assemble `MODEL_CONFIGS`; consume from `config.availability` only with prior `config.settings` import

---

PHASE 5: Model Loader & Diarization Pipeline (HIGH RISK)

- Functions to Extract:
  - `load_model(model_name)` @ 452
  - `get_diarization_pipeline()` @ 553

- Target Module:
  - `models/loader.py` (model loader and diarization pipeline getter)

- Dependencies Required:
  - Imports: `threading`, any model-specific libraries, `config.settings`, `core.state`
  - Imported by: `services/transcription.py`, `services/diarization.py`, `api/websocket.py`

- Code to Copy:
  - Double-check locking for model caching and pipeline singleton
  - GPU/CPU detection logic and compute type selection
  - Read `MODEL_CONFIGS` and `MODEL_SIZE` from `config.availability`; do not assemble configs here

- Import Changes in app.py:
  - Remove: inline model/pipeline loader functions
  - Add: `from models.loader import load_model, get_diarization_pipeline`
  - Example: `from models.loader import load_model`

- Git Commit Message:
  - `EXTRACT: Model loader and diarization pipeline to models/loader.py`

- Critical Notes:
  - Retain global caches and lock semantics via `core.state`
  - Avoid importing heavy libs at module import time if lazily loaded
  - Do not import `core.lifespan` or any API modules; only `config.settings`, `config.availability`, and `core.state`
  - Cross-check: Confirm `MODEL_CONFIGS` is imported from `config.availability` after `config.settings`; no cycles with `core.state`

---

PHASE 6: Diarization Service (HIGH RISK)

- Functions to Extract:
  - `transcribe_with_diarization(model, model_config, audio_file, language, websocket, model_name)` @ 591 (async)
  - Internal helpers: speaker alignment within diarization

- Target Module:
  - `services/diarization.py`

- Dependencies Required:
  - Imports: `models.loader.get_diarization_pipeline`, `config.settings`, `logging`
  - Imported by: `api/websocket.py`, `services/transcription.py`

- Code to Copy:
  - Orchestration that aligns transcription segments to diarized speakers
  - Preserve return structure and WebSocket chunk formatting

- Import Changes in app.py:
  - Remove: diarization function and helpers
  - Add: `from services.diarization import transcribe_with_diarization`

- Git Commit Message:
  - `EXTRACT: Diarization orchestration to services/diarization.py`

- Critical Notes:
  - Keep HuggingFace token handling and lazy pipeline init intact
  - Cross-check: Verify diarization reads `MODEL_CONFIGS` indirectly via `models.loader` only; avoid direct assembly

---

PHASE 7: Transcription Orchestration & Deepgram (HIGH RISK)

- Functions to Extract:
  - `transcribe_with_incremental_output(model, model_config, audio_file, language, websocket, model_name, chunk_seconds)` @ 1626 (async)
  - Nested: `run_fw_transcription` @ 1671; `run_transcription` @ 1728; `transcribe_fw_chunk` @ 1831; `transcribe_openai_chunk` @ 1866
  - `transcribe_chunk(model_config, model, chunk_path, language)` @ 1977
  - `transcribe_audio_stream(websocket, processor)` @ 2335 (async)
  - `transcribe_vod_with_deepgram(websocket, url, language)` @ 2502 (async)
  - `transcribe_with_deepgram(websocket, url, language)` @ 2781 (async)
  - Nested: `extract_deepgram_transcript` @ 2794; `on_message` @ 2845; `on_close` @ 2874; `on_error` @ 2887; `read_audio_file` @ 2610

- Target Module:
  - `services/transcription.py`

- Dependencies Required:
  - Imports: `models.loader`, `services.audio_processor`, `services.diarization`, `config.settings`, `logging`, Deepgram SDK
  - Imported by: `api/websocket.py`

- Code to Copy:
  - Full incremental orchestration, streaming handlers, Deepgram VOD/live integrations
  - Preserve progress, chunking, and message payload shapes

- Import Changes in app.py:
  - Remove: transcription engine functions
  - Add: `from services.transcription import transcribe_with_incremental_output, ...`

- Git Commit Message:
  - `EXTRACT: Transcription orchestration and Deepgram integrations to services/transcription.py`
  - `ATOMIC MOVE: transcribe_with_incremental_output + nested helpers to services/transcription.py`

- Critical Notes:
  - **CRITICAL THREADING WARNING**: `transcribe_with_deepgram` uses nested event handlers (`on_message`, `on_close`, `on_error`) that capture closure variables and run in Deepgram SDK threads. These callbacks use `asyncio.run_coroutine_threadsafe` to bridge back to the main event loop. Ensure atomic extraction preserves this threading model.
  - Deepgram callbacks cross threads; capture loop safely and use thread-safe scheduling
  - Maintain cache checks and short-circuit behavior
  - Cross-check: Ensure module imports `config.availability` (for flags/events) after `config.settings`; no local `MODEL_CONFIGS` assembly

---

PHASE 8: HTTP API Routes (MEDIUM RISK)

- Functions to Extract:
  - `get_home()` @ 3068 (async) [GET `/`]
  - `health_check()` @ 3419 (async) [GET `/health`]
  - `get_video_info(request)` @ 3429 (async) [POST `/api/video-info`]
  - `gpu_diagnostics()` @ 3485 (async) [GET `/gpu`]
  - `cache_stats()` @ 3518 (async) [GET `/api/cache/stats`]
  - `clear_cache()` @ 3538 (async) [POST `/api/cache/clear`]
  - `download_cache_stats()` @ 3555 (async) [GET `/api/download-cache/stats`]
  - `clear_download_cache()` @ 3584 (async) [POST `/api/download-cache/clear`]

- Target Module:
  - `api/routes.py`

- Dependencies Required:
  - Imports: `utils.cache`, `services.video_metadata`, `core.state`
  - Imported by: `app.py`

- Code to Copy:
  - Existing route handlers unchanged
  - Read static HTML from cache (or disk post Phase 10)

- Import Changes in app.py:
  - Remove: route function definitions
  - Add: `from api.routes import get_home, health_check, ...`

- Router Conversion Semantics:
  - Convert `@app.get` and `@app.websocket` endpoints to `APIRouter` routes
  - Ensure `app.include_router(...)` happens after `app = FastAPI(...)` is created
  - All route handlers must read shared state from `core.state` (e.g., `cached_index_html`, `CAPTURES`) and not capture module-local snapshots
  - `get_home()` must read `core.state.cached_index_html` at runtime; do not bind default values at import

- Git Commit Message:
  - `EXTRACT: HTTP API endpoints to api/routes.py`

- Critical Notes:
  - Keep response types and status codes identical
  - Ensure static file serving transition is coordinated with Phase 10
  - Cross-check: `api/routes.py` must not import `config.availability` unless necessary; if used, ensure `config.settings` precedes and avoid cycles via `core.state`

---

PHASE 9: WebSocket Endpoint & Helpers (HIGH RISK)

- Functions to Extract:
  - `websocket_transcribe(websocket)` @ 3074 (async) [WS `/ws/transcribe`]
  - Nested: `read_capture_file` @ 3238

- Target Module:
  - `api/websocket.py`

- Dependencies Required:
  - Imports: `services.audio_processor`, `services.transcription`, `utils.cache`, `utils.validators`, `utils.websocket_helpers`, `models.loader`, `core.state`, `logging`
  - Imported by: `app.py`

- Code to Copy:
  - Entire WebSocket session lifecycle, message types, progress and error handling
  - Maintain `WebSocketState` checks and disconnection flows

- Import Changes in app.py:
  - Remove: WebSocket functions
  - Add: `from api.websocket import websocket_transcribe`
  - WebSocket helpers now imported from: `from utils.websocket_helpers import safe_ws_send`

- Git Commit Message:
  - `EXTRACT: WebSocket endpoint to api/websocket.py`

- Critical Notes:
  - Preserve message payloads and sequencing strictly
  - Ensure thread and async boundaries are respected
  - Cross-check: WebSocket module imports `config.availability` (flags/events) only after `config.settings`; never assemble `MODEL_CONFIGS` locally

---

PHASE 10: Static UI Extraction (MEDIUM RISK)

- Functions to Extract:
  - None

- Target Module:
  - `static/index.html`, `static/css/stylesheet.css`

- Dependencies Required:
  - Imports: None
  - Imported by: `api/routes.get_home` will read from disk or cache

- Code to Copy:
  - Move inline HTML/CSS from `app.py` to static files
  - Implement in-memory cache for HTML identical to current behavior

  CSS Extraction Safety (stylesheet.css)
  - Identify inline `<style>` block (in `app.py` HTML string or `static/index.html`)
  - Copy CSS EXACTLY as-is (no edits, no reformatting) into `static/css/stylesheet.css`
  - Do not change selector names, class names, IDs, or order; UI JS relies on them
  - Update `static/index.html` `<head>` to include:
    - `<link rel="stylesheet" href="/static/css/stylesheet.css">`
  - Remove the inline `<style>` block once the link tag is added
  - In `app.py`, ensure static files are mounted:
    - `app.mount('/static', StaticFiles(directory='static'), name='static')`
  - Dockerfiles: ensure `COPY static/` includes `static/css/stylesheet.css`
  - Validation: open `/` in browser, confirm `/static/css/stylesheet.css` loads (HTTP 200), UI renders identically, and no console errors

- Import Changes in app.py:
  - Remove: embedded HTML string
  - Add: logic to load cached HTML via `core/state`

- Git Commit Message:
  - `EXTRACT: Inline UI to static/index.html and static/css/styles.css`

- Critical Notes:
  - Validate UI via `open_preview` and WebSocket connectivity
  - Keep paths and caching semantics unchanged

---

PHASE 11: Final Consistency & Safety Checks (LOW RISK)

- Functions to Extract:
  - None

- Target Module:
  - N/A

- Dependencies Required:
  - Imports: N/A
  - Imported by: N/A

- Code to Copy:
  - None

- Import Changes in app.py:
  - Ensure all imports reference new modules
  - Remove any dead code

- Git Commit Message:
  - `CHORE: Final consistency pass, imports aligned, dead code removed`

- Critical Notes:
  - Run with `start.sh`, monitor `.logs/server-output.log` and `.logs/server-runtime-errors.log`
  - Verify endpoints and WebSocket flows match pre-refactor behavior

---

Validation Checklist

- [ ] Endpoints respond exactly as before
- [ ] WebSocket messages identical (order, shape, timing within tolerance)
- [ ] Cache stats and clear endpoints unchanged
- [ ] Diarization results consistent
- [ ] Deepgram paths working with/without diarization
- [ ] No circular dependencies
- [ ] No signature changes across functions
- [ ] **AudioStreamProcessor class extracted atomically** with all state management intact (ffmpeg_process, audio_queue, overlap_buffer, stop_event)
 - [ ] **AudioStreamProcessor class extracted atomically** with all state management intact (`ffmpeg_process`, `audio_queue`, `is_running`, `stop()` semantics)
- [ ] **Deepgram WebSocket threading preserved** - nested event handlers and asyncio.run_coroutine_threadsafe working correctly
- ATOMIC MOVE:
  - Move `transcribe_with_incremental_output` together with nested `run_fw_transcription`, `run_transcription`, `transcribe_fw_chunk`, and `transcribe_openai_chunk` in one commit
  - Move `AudioStreamProcessor` class atomically with ALL its state management components
