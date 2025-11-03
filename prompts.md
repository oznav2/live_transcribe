================================================================================
PHASE 0: INITIAL SETUP AND PROGRESS TRACKER
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Read from Old_Files/*.original for all code extraction
- Never modify main branch
- No Docker builds until Phase 13 complete and user approve
- Batch operations per phase
- Use line numbers from APP_FUNCTION_INVENTORY.md

PHASE 0: Initial Setup (Complete all steps in one response)

STEP 0: Read Planning Docs (single pass)
  head -n 80 ARCHITECTURE_DESIGN.md
  head -n 120 IMPLEMENTATION_PLAN.md
  head -n 40 docs/plans/APP_FUNCTION_INVENTORY.md
  head -n 40 app_py_map.md

Confirm module boundaries, import dependency map, and function line anchors.

OBJECTIVE: Create branch structure, reference files, and progress tracker

STEP 1: Verify Git Status
  git status
  git log --oneline -5

STEP 2: Create Modular Branch
  git checkout main
  git pull origin main
  git checkout -b Modular
  git branch --show-current

Expected output: "Modular"

STEP 3: Create Directory Structure
  mkdir -p config core models services api utils static/css .backups Old_Files .logs
  touch config/__init__.py
  touch core/__init__.py
  touch models/__init__.py
  touch services/__init__.py
  touch api/__init__.py
  touch utils/__init__.py

STEP 4: Create Reference Files
  cp app.py Old_Files/app.py.original
  cp Dockerfile.ivrit Old_Files/Dockerfile.ivrit.original
  cp docker-compose.ivrit.yml Old_Files/docker-compose.ivrit.yml.original
  cp requirements.txt Old_Files/requirements.txt.original
  cp requirements.ivrit.txt Old_Files/requirements.ivrit.txt.original

STEP 5: Create Backups
  cp app.py .backups/app.py.backup
  cp Dockerfile.ivrit .backups/Dockerfile.ivrit.backup
  cp docker-compose.ivrit.yml .backups/docker-compose.ivrit.yml.backup

copy from main branch to modular branch root dir: IMPLEMENTATION_PLAN.md, prompts.md, app_py_map.md, architecture_design.md

Content:
---START FILE---

STEP 6: Create PROGRESS.md

Create file: PROGRESS.md

inside progress.md create: 
# Refactoring Progress Tracker

Last Updated: [CURRENT_DATE]
Branch: Modular
Original app.py: 3,618 lines
Runtime Logs: .logs/

## Phase Status Overview

Phase 0: Setup ⏳ IN PROGRESS
Phase 1A: Config ⏳ NOT STARTED
Phase 1B: State ⏳ NOT STARTED
Phase 2: Utils ⏳ NOT STARTED
Phase 3: Cache ⏳ NOT STARTED
Phase 4: Models ⏳ NOT STARTED
Phase 5: Audio ⏳ NOT STARTED
Phase 6: Transcription ⏳ NOT STARTED
Phase 7: Diarization ⏳ NOT STARTED
Phase 8: Video Metadata ⏳ NOT STARTED
Phase 9: API Routes ⏳ NOT STARTED
Phase 10: WebSocket ⏳ NOT STARTED
Phase 11: Lifespan ⏳ NOT STARTED
Phase 12: Static Files ⏳ NOT STARTED
Phase 13: Docker Config ⏳ NOT STARTED

---

## Phase 0: Setup ⏳ IN PROGRESS

Target:
  - Create directory structure
  - Create Old_Files reference copies
  - Create progress tracker

Status: In progress
Files Created: None yet
Commits: 0

---

## Phase 1A: Extract Configuration ⏳ NOT STARTED

Target Files:
  - config/settings.py
  - config/constants.py

Functions to Extract:
  - All os.getenv() calls -> settings.py
  - All UPPERCASE constants -> constants.py

Status: Not started
Files Created: None
Commits: 0

---

## Phase 1B: Create Global State ⏳ NOT STARTED

Target Files:
  - core/state.py

Globals to Extract:
  - whisper_models, faster_whisper_models, diarization_pipeline
  - model_lock, faster_whisper_lock, diarization_lock
  - executor, cached_index_html, CAPTURES

Status: Not started
Files Created: None
Commits: 0

---

## Phase 2: Extract Utilities ⏳ NOT STARTED

Target Files:
  - utils/validators.py
  - utils/helpers.py
  - utils/websocket_helpers.py

Functions to Extract:
  - is_youtube_url, should_use_ytdlp -> validators.py
  - format_duration, format_view_count -> helpers.py
  - safe_ws_send -> websocket_helpers.py

Status: Not started
Files Created: None
Commits: 0

---

## Phase 3: Extract Cache Management ⏳ NOT STARTED

Target Files:
  - utils/cache.py

Functions to Extract (from APP_FUNCTION_INVENTORY.md):
  - init_capture_dir (line 359)
  - init_download_cache_dir (line 366)
  - init_cache_dir (line 2033)
  - get_url_hash (line 387)
  - get_cached_download (line 392)
  - save_download_to_cache (line 417)
  - generate_cache_key (line 2057)
  - get_cached_audio (line 2065)
  - save_to_cache (line 2077)
  - All other cache functions

Status: Not started
Files Created: None
Commits: 0

---

## Phase 4: Extract Model Management ⏳ NOT STARTED

Target Files:
  - models/loader.py

Functions to Extract (HIGH RISK - Thread Safety):
  - load_model (line 452) - PRESERVE locking
  - get_diarization_pipeline (line 553) - PRESERVE locking

Status: Not started
Files Created: None
Commits: 0

---

## Phase 5: Extract Audio Processing ⏳ NOT STARTED

Target Files:
  - services/audio_processor.py

Functions to Extract (ATOMIC MOVES):
  - download_audio_with_ffmpeg (line 866) + nested helpers (lines 944, 960)
  - download_with_fallback (line 1159)
  - download_audio_with_ytdlp_async (line 1225)
  - get_audio_duration_seconds (line 1460)
  - calculate_progress_metrics (line 1483)
  - split_audio_for_incremental (line 1531)
  - split_audio_into_chunks (line 1578)
  - AudioStreamProcessor class (lines 2220-2333) - COMPLETE CLASS

Status: Not started
Files Created: None
Commits: 0

---

## Phase 6: Extract Transcription Services ⏳ NOT STARTED

Target Files:
  - services/transcription.py

Functions to Extract (ATOMIC MOVES - Keep nested helpers):
  - transcribe_with_incremental_output (line 1626) + 4 nested helpers
  - transcribe_chunk (line 1977)
  - transcribe_audio_stream (line 2335)
  - transcribe_vod_with_deepgram (line 2502) + nested helper
  - transcribe_with_deepgram (line 2781) + 4 nested helpers

Status: Not started
Files Created: None
Commits: 0

---

## Phase 7: Extract Diarization ⏳ NOT STARTED

Target Files:
  - services/diarization.py

Functions to Extract:
  - transcribe_with_diarization (line 591)

Status: Not started
Files Created: None
Commits: 0

---

## Phase 8: Extract Video Metadata ⏳ NOT STARTED

Target Files:
  - services/video_metadata.py

Functions to Extract:
  - get_youtube_metadata (line 2144)

Status: Not started
Files Created: None
Commits: 0

---

## Phase 9: Extract API Routes ⏳ NOT STARTED

Target Files:
  - api/routes.py

Endpoints to Extract (Convert @app to @router):
  - get_home (line 3068) [GET /]
  - health_check (line 3419) [GET /health]
  - get_video_info (line 3429) [POST /api/video-info]
  - gpu_diagnostics (line 3485) [GET /gpu]
  - cache_stats (line 3518) [GET /api/cache/stats]
  - clear_cache (line 3538) [POST /api/cache/clear]
  - download_cache_stats (line 3555) [GET /api/download-cache/stats]
  - clear_download_cache (line 3584) [POST /api/download-cache/clear]

Status: Not started
Files Created: None
Commits: 0

---

## Phase 10: Extract WebSocket Endpoint ⏳ NOT STARTED

Target Files:
  - api/websocket.py

Endpoint to Extract (CRITICAL - Complete workflow):
  - websocket_transcribe (line 3074) + nested helper (line 3238)

Status: Not started
Files Created: None
Commits: 0

---

## Phase 11: Extract Lifespan ⏳ NOT STARTED

Target Files:
  - core/lifespan.py
  - app.py (finalize to minimal entry point)

Functions to Extract:
  - lifespan (line 158)

Status: Not started
Files Created: None
Commits: 0

---

## Phase 12: Extract Static Files ⏳ NOT STARTED

Target Files:
  - static/index.html
  - static/css/styles.css

Content to Extract:
  - HTML string from Old_Files/app.py.original
  - CSS from within <style> tags

Status: Not started
Files Created: None
Commits: 0

---

## Phase 13: Update Docker Configuration ⏳ NOT STARTED

Target Files:
  - .dockerignore
  - Dockerfile
  - Dockerfile.ivrit
  - docker-compose.ivrit.yml (review)
  - DOCKER_BUILD_READY.md

Changes:
  - Update COPY commands for modular structure
  - Create .dockerignore
  - Document build readiness
  - DO NOT BUILD CONTAINERS

Status: Not started
Files Created: None
Commits: 0

---

## Summary Statistics

Total Phases: 14 (0-13)
Completed: 0
In Progress: 1
Not Started: 13

Total Files to Create: ~17 modules + 2 static files
Estimated Final app.py Size: ~40-50 lines
Original app.py Size: 3,618 lines

---
---END FILE---

STEP 2.1: AST Enumeration (lifespan)
  Enumerate the lifespan function in Old_Files/app.py.original using Python AST.
  Focus: function name, async flag, @asynccontextmanager decorator, parameters,
  and startup/shutdown orchestration calls and assignments.

  Enumeration outputs (write files):
    - .backups/lifespan_phase11_funcs.txt           # function names and signatures
    - .backups/lifespan_phase11_async.json          # async flags, decorator presence
    - .backups/lifespan_phase11_imports.json        # imports used by lifespan
    - .backups/lifespan_phase11_startup_shutdown.json # startup/shutdown calls & assignments

  Run:
    python - << 'PY'
    import ast, json, sys, re
    from pathlib import Path

    src = Path('Old_Files/app.py.original').read_text(encoding='utf-8')
    tree = ast.parse(src)

    data_funcs = []
    async_info = {}
    imports_info = set()
    orchestration = {
      'startup_calls': [],
      'startup_assignments': [],
      'shutdown_calls': [],
      'shutdown_assignments': [],
    }

    # collect imports used in module
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        for n in node.names:
          imports_info.add(n.name)
      elif isinstance(node, ast.ImportFrom):
        mod = node.module or ''
        for n in node.names:
          imports_info.add(f"{mod}.{n.name}")

    def is_lifespan_func (f: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
      return isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == 'lifespan'

    lifespan_node = None
    for node in tree.body:
      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'lifespan':
        lifespan_node = node
        break

    if lifespan_node is None:
      print('lifespan not found', file=sys.stderr)
      sys.exit(1)

    # function signature & decorators
    decos = []
    for d in lifespan_node.decorator_list:
      if isinstance(d, ast.Name):
        decos.append(d.id)
      elif isinstance(d, ast.Attribute):
        decos.append(f"{getattr(d.value, 'id', '')}.{d.attr}")
      elif isinstance(d, ast.Call):
        # e.g., @something(...)
        func = d.func
        if isinstance(func, ast.Name):
          decos.append(func.id)
        elif isinstance(func, ast.Attribute):
          decos.append(f"{getattr(func.value, 'id', '')}.{func.attr}")

    params = [arg.arg for arg in lifespan_node.args.args]
    is_async = isinstance(lifespan_node, ast.AsyncFunctionDef)
    data_funcs.append({
      'name': lifespan_node.name,
      'params': params,
      'decorators': decos,
      'is_async': is_async,
    })
    async_info['lifespan'] = {
      'is_async': is_async,
      'decorators': decos,
    }

    # scan body for startup/shutdown sections heuristically: before/after first ast.Yield
    yield_idx = None
    for idx, stmt in enumerate(lifespan_node.body):
      if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.YieldFrom):
        yield_idx = idx
        break
      if isinstance(stmt, ast.Yield):
        yield_idx = idx
        break

    def collect_calls_and_assigns (stmts, dest_calls, dest_assigns):
      for s in stmts:
        for n in ast.walk(s):
          if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
              dest_calls.append(n.func.id)
            elif isinstance(n.func, ast.Attribute):
              base = getattr(n.func.value, 'id', '')
              dest_calls.append(f"{base}.{n.func.attr}")
          elif isinstance(n, ast.Assign):
            # record target names and value call names
            targets = []
            for t in n.targets:
              if isinstance(t, ast.Name):
                targets.append(t.id)
              elif isinstance(t, ast.Attribute):
                base = getattr(t.value, 'id', '')
                targets.append(f"{base}.{t.attr}")
            val = None
            if isinstance(n.value, ast.Call):
              if isinstance(n.value.func, ast.Name):
                val = n.value.func.id
              elif isinstance(n.value.func, ast.Attribute):
                val = f"{getattr(n.value.func.value, 'id', '')}.{n.value.func.attr}"
            dest_assigns.append({'targets': targets, 'value_call': val})

    if yield_idx is None:
      # treat entire body as startup
      collect_calls_and_assigns(lifespan_node.body, orchestration['startup_calls'], orchestration['startup_assignments'])
    else:
      collect_calls_and_assigns(lifespan_node.body[:yield_idx], orchestration['startup_calls'], orchestration['startup_assignments'])
      collect_calls_and_assigns(lifespan_node.body[yield_idx+1:], orchestration['shutdown_calls'], orchestration['shutdown_assignments'])

    Path('.backups').mkdir(exist_ok=True)
    Path('.backups/lifespan_phase11_funcs.txt').write_text('\n'.join(
      f"{d['name']}({', '.join(d['params'])}) decorators={d['decorators']} async={d['is_async']}" for d in data_funcs
    ), encoding='utf-8')
    Path('.backups/lifespan_phase11_async.json').write_text(json.dumps(async_info, indent=2), encoding='utf-8')
    Path('.backups/lifespan_phase11_imports.json').write_text(json.dumps(sorted(imports_info), indent=2), encoding='utf-8')
    Path('.backups/lifespan_phase11_startup_shutdown.json').write_text(json.dumps(orchestration, indent=2), encoding='utf-8')
    print('AST enumeration for lifespan complete')
    PY

STEP 2.2: Strict Guardrails (startup/shutdown orchestration)
  Preserve exact behavior and semantics when extracting lifespan:
  - Keep @asynccontextmanager decorator intact and imported from contextlib
  - Preserve 'async def lifespan(app: FastAPI)' signature and parameter typing
  - Maintain startup tasks:
    - Initialize ThreadPoolExecutor and assign to core.state.executor
    - Initialize cache directories: init_cache_dir, init_download_cache_dir, init_capture_dir
    - Preload default model via models.loader.load_model and assign to state
    - Cache index.html into core.state.cached_index_html
    - Configure logging consistently with original app.py
  - Maintain shutdown tasks:
    - Properly shutdown/cleanup ThreadPoolExecutor and related resources
  - Preserve imports, constants, and configuration usage from config.settings
  - Preserve references to core.state module-level variables
  - Do not change error handling or logging semantics
  - app.py changes must be additive-only: import lifespan and include routers
  - Do not alter HTTP or WebSocket endpoint behavior while modifying app.py

STEP 2.3: Dependent Coverage Cross-Check
  Cross-check modules interacting with lifespan startup/shutdown artifacts:
  - core/state.py (executor, cached_index_html, default model references)
  - utils/cache.py (init_cache_dir, init_download_cache_dir, init_capture_dir)
  - models/loader.py (load_model and related defaults)
  - api/routes.py (uses cached_index_html in get_home)
  - api/websocket.py (may rely on state and default model)
  - config/settings.py (settings consumed by lifespan)
  - services/audio_processor.py, services/transcription.py, services/diarization.py
  - utils/validators.py, utils/websocket_helpers.py (if referenced)

  Record findings (imports, symbol usage) to:
    - .backups/lifespan_phase11_dependents.txt


STEP 7: Commit All Setup Files
  git add .
  git commit -m "PHASE 0: Initial setup - directory structure, reference files, progress tracker"

STEP 8: Update PROGRESS.md - Mark Phase 0 Complete

Edit PROGRESS.md:

Change Phase 0 section from:
  ## Phase 0: Setup ⏳ IN PROGRESS

To:
  ## Phase 0: Setup ✅ COMPLETE
  
  Completed: [CURRENT_TIMESTAMP]
  
  Files Created:
    - Old_Files/app.py.original
    - Old_Files/Dockerfile.ivrit.original
    - Old_Files/docker-compose.ivrit.yml.original
    - Old_Files/requirements.txt.original
    - Old_Files/requirements.ivrit.txt.original
    - .backups/app.py.backup
    - .backups/Dockerfile.ivrit.backup
    - .backups/docker-compose.ivrit.yml.backup
    - .logs/server-output.log
    - .logs/server-runtime-errors.log
    - config/__init__.py
    - core/__init__.py
    - models/__init__.py
    - services/__init__.py
    - api/__init__.py
    - utils/__init__.py
    - PROGRESS.md
    - start.sh (executable)
  
  Commits: 1
  Commit Hash: [git log -1 --format=%H]
  
  Notes:
    - Directory structure created
    - Reference files ready in Old_Files/
    - Progress tracker initialized
    - Ready for Phase 1A

Update Phase Status Overview:
  Phase 0: Setup ✅ COMPLETE

STEP 9: Commit Progress Update
  git add PROGRESS.md
  git commit -m "PHASE 0: Update progress tracker - marked complete"

STEP 10: Verification
  git log --oneline -2
  ls -la Old_Files/
  ls -la .logs/
  test -x start.sh && echo "start.sh executable" || echo "start.sh missing or not executable"
  cat PROGRESS.md | head -30

STEP 11: Report Completion

Provide this report:

PHASE 0 COMPLETE

Summary:
  - Branch: Modular
  - Directories created: 9
  - Reference files created: 5
  - Logs initialized: .logs/server-output.log, .logs/server-runtime-errors.log
  - start.sh: executable
  - Progress tracker: Initialized
  - Commits: 2

Next Phase: Phase 1A (Extract Configuration)

Ready for next instruction.


================================================================================
PHASE 1A: EXTRACT CONFIGURATION
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Read from Old_Files/app.py.original for all code extraction
- Never modify main branch
- No Docker builds until Phase 13 complete
- Batch operations per phase
- Use line numbers from APP_FUNCTION_INVENTORY.md

PHASE 1A: Extract Configuration (Complete this entire phase in one response)

STEP 1: Read PROGRESS.md
  cat PROGRESS.md | grep "Phase 1A" -A 10

Confirm Phase 1A status is: NOT STARTED
Confirm Phase 0 status is: COMPLETE

STEP 2: Create config/settings.py

Extract ALL environment variable assignments from Old_Files/app.py.original
Search for pattern: os.getenv("VARIABLE_NAME", "default")

Create file: config/settings.py

STEP 2.1: Enumerate environment variables to extract
  # Generate a unique list of env keys referenced in Old_Files/app.py.original
  grep -n "os.getenv(" Old_Files/app.py.original \
    | sed -E "s/.*os.getenv\(\"([A-Z0-9_]+)\".*/\1/" \
    | sort -u > .backups/env_keys_phase1a.txt
  wc -l .backups/env_keys_phase1a.txt

Extraction Rules (strict):
  - Preserve variable names EXACTLY as in Old_Files/app.py.original
  - Preserve default values; apply correct type conversions:
    - int values: wrap with int(...)
    - float values: wrap with float(...)
    - bool flags: implement safe parsing ("true"/"1"/"yes" → True)
  - Use Optional[str] for keys that can be empty strings
  - For path-like values, expose as strings here; constants module handles Paths
  - Do NOT change semantics or defaults; no behavioral change

Content structure:
---START FILE---
"""Application configuration from environment variables."""
import os
from typing import Optional

# Helper for strict boolean parsing
def parse_bool (value: str) -> bool:
    return str(value).strip().lower() in { '1', 'true', 't', 'yes', 'y' }

# Deepgram API
DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")

# Model Configuration
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "whisper-v3-turbo")
IVRIT_MODEL_NAME: str = os.getenv("IVRIT_MODEL_NAME", "ivrit-ai/whisper-large-v3-turbo-ct2")
IVRIT_DEVICE: str = os.getenv("IVRIT_DEVICE", "cuda")
IVRIT_COMPUTE_TYPE: str = os.getenv("IVRIT_COMPUTE_TYPE", "float16")
IVRIT_BEAM_SIZE: int = int(os.getenv("IVRIT_BEAM_SIZE", "5"))

# [Extract ALL other os.getenv() calls from Old_Files/app.py.original]
# Search entire file and copy EXACTLY with same variable names and defaults

# Audio Processing Configuration
# Caching Configuration
# Performance Configuration
# Path Configuration
# [etc. - extract all]
---END FILE---

STEP 3: Create config/constants.py

Extract ALL constant definitions (UPPERCASE variables with fixed values)
Search Old_Files/app.py.original for pattern: CONSTANT_NAME = value

Create file: config/constants.py

STEP 3.1: Enumerate constants to extract
  # List UPPERCASE assignments that are pure constants (exclude os.getenv-derived)
  grep -n "^[A-Z_][A-Z0-9_]*\s*=\s*" Old_Files/app.py.original \
    | grep -v "os.getenv" > .backups/constants_phase1a.txt
  wc -l .backups/constants_phase1a.txt

Extraction Rules (strict):
  - Include ONLY fixed values (numbers, strings, tuples, dicts) defined at module level
  - Represent filesystem paths using pathlib.Path
  - Keep names and values EXACTLY; no renaming, no unit changes
  - Do NOT move dynamic or computed values into constants

Content structure:
---START FILE---
"""Application constants."""
from pathlib import Path

# Cache Directories
CACHE_DIR = Path("cache/audio")
DOWNLOAD_CACHE_DIR = Path("cache/downloads")
CAPTURE_DIR = Path("cache/captures")

# Audio Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 60
CHUNK_OVERLAP = 5
AUDIO_QUEUE_SIZE = 10

# [Extract ALL other constants from Old_Files/app.py.original]
# Copy EXACTLY - preserve names and values
---END FILE---

STEP 4: Add Imports to app.py

In working app.py (NOT Old_Files version), add these imports after existing imports:

---ADD TO app.py---
from config.settings import (
    DEEPGRAM_API_KEY,
    WHISPER_MODEL,
    IVRIT_MODEL_NAME,
    IVRIT_DEVICE,
    IVRIT_COMPUTE_TYPE,
    IVRIT_BEAM_SIZE,
    # [list ALL variables extracted to settings.py]
)

from config.constants import (
    CACHE_DIR,
    DOWNLOAD_CACHE_DIR,
    CAPTURE_DIR,
    SAMPLE_RATE,
    CHANNELS,
    CHUNK_DURATION,
    CHUNK_OVERLAP,
    AUDIO_QUEUE_SIZE,
    # [list ALL constants extracted to constants.py]
)
---END ADDITIONS---

NOTE: Do NOT remove any code from app.py yet. Only ADD imports.

STEP 4.1: Cross-check import coverage
  # Confirm settings/constants imports present in app.py
  grep -n "from config.settings" app.py
  grep -n "from config.constants" app.py
  # Compare counts with enumerated keys/constants
  echo "Env keys enumerated: $(wc -l < .backups/env_keys_phase1a.txt)"
  echo "Constants enumerated: $(wc -l < .backups/constants_phase1a.txt)"

STEP 5: Git Commit
  git add config/settings.py config/constants.py app.py
  git commit -m "PHASE 1A: Extract config/settings.py and config/constants.py, add imports to app.py"

STEP 6: Verify
  # Baseline import test and file line counts
  python -c "from config import settings, constants; print('Config OK')"
  wc -l config/settings.py config/constants.py

  # Enumerated coverage check (env keys/consts discovered vs. defined)
  python - << 'PY'
import importlib, json
s = importlib.import_module('config.settings')
c = importlib.import_module('config.constants')
settings_exports = [k for k in dir(s) if k.isupper()]
constants_exports = [k for k in dir(c) if k.isupper()]
print('Settings exports:', len(settings_exports))
print('Constants exports:', len(constants_exports))
PY

  # app import sanity (no runtime execution)
  python -c "import app; print('app import OK')"

  # Optional runtime check via start.sh and logs (recommended)
  ./start.sh || true
  sleep 2
  tail -n +1 .logs/server-runtime-errors.log
  tail -n 50 .logs/server-output.log
  curl -sSf http://localhost:8009/health | head -c 200 || true

STEP 7: Update PROGRESS.md

Edit PROGRESS.md:

Change Phase 1A section to:
---UPDATE---
## Phase 1A: Extract Configuration ✅ COMPLETE

Completed: [CURRENT_TIMESTAMP]

Files Created:
  - config/settings.py ([X] lines)
  - config/constants.py ([Y] lines)

Imports Added to app.py:
  - from config.settings import [list variables]
  - from config.constants import [list constants]

Commits: 1
Commit Hash: [git log -1 --format=%H]

Notes:
  - All environment variables extracted to settings.py
  - All constants extracted to constants.py
  - Imports added to app.py (additive only)
  - Original app.py code still intact
  - Config modules verified with import test

Next Phase: Phase 1B (Create Global State)
---END UPDATE---

Update Phase Status Overview:
  Phase 1A: Config ✅ COMPLETE
  Phase 1B: State ⏳ NOW READY

STEP 8: Commit Progress Update
  git add PROGRESS.md
  git commit -m "PHASE 1A: Update progress tracker - marked complete"

STEP 9: Report Completion

Provide this report:

PHASE 1A COMPLETE

Files Created:
  - config/settings.py ([X] lines)
  - config/constants.py ([Y] lines)

Imports Added: [number] imports to app.py

Verification: Import test passed

Commits: 2

Next Phase: Phase 1B (Create Global State Management)

Ready for next instruction.


================================================================================
PHASE 1B: CREATE GLOBAL STATE MANAGEMENT
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Read from Old_Files/app.py.original for all code extraction
- Never modify main branch
- No Docker builds until Phase 13 complete
- Batch operations per phase
- Use line numbers from APP_FUNCTION_INVENTORY.md

PHASE 1B: Create Global State Management (Complete this entire phase in one response)

STEP 1: Read PROGRESS.md
  cat PROGRESS.md | grep "Phase 1B" -A 10

Confirm Phase 1B status is: NOT STARTED or NOW READY
Confirm Phase 1A status is: COMPLETE

STEP 2: Create core/state.py

This module centralizes ALL global variables to prevent duplication.

Extract from Old_Files/app.py.original:
  - Search for all global variable declarations (not inside functions)
  - Look for: model dictionaries, locks, executor, cache variables
  - Common patterns: "= {}", "= None", "= threading.Lock()", "= ThreadPoolExecutor"

STEP 2.1: Enumerate globals to extract (strict)
  # Use Python AST to list module-level assignments (exclude functions/classes)
  python - << 'PY'
import ast, sys, json
src = open('Old_Files/app.py.original', 'r', encoding='utf-8').read()
tree = ast.parse(src)
globals_ = []
for node in tree.body:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        # collect simple names only
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        if target:
            globals_.append(target)
open('.backups/globals_phase1b.txt', 'w', encoding='utf-8').write('\n'.join(sorted(set(globals_))))
print('Discovered globals:', len(set(globals_)))
PY
  wc -l .backups/globals_phase1b.txt

Extraction Rules (strict):
  - Centralize ONLY module-level runtime state:
    - whisper_models, faster_whisper_models, diarization_pipeline
    - model_lock, faster_whisper_lock, diarization_lock
    - executor (ThreadPoolExecutor)
    - cached_index_html, CAPTURES, last_progress_update
    - [include any other top-level runtime flags/state]
  - Preserve names, types, and default values EXACTLY
  - Locks must be instances of threading.Lock()
  - Dictionaries must be initialized as empty dicts unless otherwise set
  - Use Optional[...] for nullable globals
  - Do NOT move pure constants here (they belong in config/constants.py)
  - Do NOT change semantics or initialization order

Create file: core/state.py

Content structure:
---START FILE---
"""Global state management for the application.

This module centralizes all global variables to ensure single source of truth
and prevent duplication across modules.
"""
import threading
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Model caches - store loaded models to avoid reloading
whisper_models: Dict[str, Any] = {}
faster_whisper_models: Dict[str, Any] = {}
diarization_pipeline: Optional[Any] = None

# Thread safety locks for model loading
model_lock: threading.Lock = threading.Lock()
faster_whisper_lock: threading.Lock = threading.Lock()
diarization_lock: threading.Lock = threading.Lock()

# Thread pool executor for CPU-bound tasks
executor: Optional[ThreadPoolExecutor] = None

# Runtime application state
cached_index_html: Optional[str] = None
CAPTURES: Dict[str, Any] = {}
last_progress_update: float = 0.0

# [Extract ANY other global variables from Old_Files/app.py.original]
# Look for variables defined at module level (not in functions)
---END FILE---

STEP 3: Add Imports to app.py

In working app.py, add this import after config imports:

---ADD TO app.py---
from core.state import (
    whisper_models,
    faster_whisper_models,
    diarization_pipeline,
    model_lock,
    faster_whisper_lock,
    diarization_lock,
    executor,
    cached_index_html,
    CAPTURES,
    last_progress_update,
    # [list ANY other globals extracted to core/state.py]
)
---END ADDITIONS---

NOTE: Do NOT remove global variable definitions from app.py yet. Only ADD import.

STEP 3.1: Cross-check import coverage and duplication safety
  # Confirm core.state import exists in app.py
  grep -n "from core.state import" app.py
  # Match imported names against enumerated globals
  echo "Globals enumerated: $(wc -l < .backups/globals_phase1b.txt)"
  # Guardrail: ensure app.py does not reassign same names AFTER imports
  # (report-only; removal happens in Phase 11)
  grep -nE "^(whisper_models|faster_whisper_models|diarization_pipeline|model_lock|faster_whisper_lock|diarization_lock|executor|cached_index_html|CAPTURES|last_progress_update)\s*=" app.py || true

STEP 4: Git Commit
  git add core/state.py app.py
  git commit -m "PHASE 1B: Create core/state.py with global state, add imports to app.py"

STEP 5: Verify
  # Baseline import tests and file line count
  python -c "from core import state; print('State OK')"
  python -c "from core.state import whisper_models, executor; print('Globals OK')"
  wc -l core/state.py

  # Enumerated coverage check (globals discovered vs. defined)
  python - << 'PY'
import importlib
import pathlib
state = importlib.import_module('core.state')
exports = [k for k in dir(state) if not k.startswith('_')]
print('State exports:', len(exports))
required = {
    'whisper_models', 'faster_whisper_models', 'diarization_pipeline',
    'model_lock', 'faster_whisper_lock', 'diarization_lock',
    'executor', 'cached_index_html', 'CAPTURES', 'last_progress_update',
}
missing = sorted(list(required - set(exports)))
print('Missing required globals:', missing)
PY

  # Type sanity checks (report-only)
  python - << 'PY'
from core.state import (
    whisper_models, faster_whisper_models, diarization_pipeline,
    model_lock, faster_whisper_lock, diarization_lock,
    executor, cached_index_html, CAPTURES, last_progress_update,
)
import threading
from concurrent.futures import ThreadPoolExecutor
assert isinstance(whisper_models, dict)
assert isinstance(faster_whisper_models, dict)
assert (diarization_pipeline is None) or True
assert isinstance(model_lock, threading.Lock.__class__)
assert isinstance(faster_whisper_lock, threading.Lock.__class__)
assert isinstance(diarization_lock, threading.Lock.__class__)
assert (executor is None) or isinstance(executor, ThreadPoolExecutor)
assert (cached_index_html is None) or isinstance(cached_index_html, str)
assert isinstance(CAPTURES, dict)
assert isinstance(last_progress_update, float)
print('Type checks OK')
PY

  # app import sanity (no runtime execution)
  python -c "import app; print('app import OK')"

  # Optional runtime check via start.sh and logs (recommended)
  ./start.sh || true
  sleep 2
  tail -n +1 .logs/server-runtime-errors.log
  tail -n 50 .logs/server-output.log
  curl -sSf http://localhost:8009/health | head -c 200 || true

STEP 6: Update PROGRESS.md

Edit PROGRESS.md:

Change Phase 1B section to:
---UPDATE---
## Phase 1B: Create Global State ✅ COMPLETE

Completed: [CURRENT_TIMESTAMP]

Files Created:
  - core/state.py ([X] lines)

Global Variables Centralized:
  - whisper_models, faster_whisper_models, diarization_pipeline
  - model_lock, faster_whisper_lock, diarization_lock
  - executor (ThreadPoolExecutor)
  - cached_index_html, CAPTURES, last_progress_update
  - [list any others]

Imports Added to app.py:
  - from core.state import [list all globals]

Commits: 1
Commit Hash: [git log -1 --format=%H]

Notes:
  - Global state centralized for single source of truth
  - Thread safety locks preserved
  - Imports added to app.py (additive only)
  - Original app.py code still intact
  - State module verified with import test

Next Phase: Phase 2 (Extract Utilities)
---END UPDATE---

Update Phase Status Overview:
  Phase 1B: State ✅ COMPLETE
  Phase 2: Utils ⏳ NOW READY

STEP 7: Commit Progress Update
  git add PROGRESS.md
  git commit -m "PHASE 1B: Update progress tracker - marked complete"

STEP 8: Report Completion

Provide this report:

PHASE 1B COMPLETE

Files Created:
  - core/state.py ([X] lines)

Global Variables: [number] variables centralized

Verification: Import test passed

Commits: 2

Next Phase: Phase 2 (Extract Utilities)

Ready for next instruction.


================================================================================
PHASE 2: EXTRACT UTILITIES
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Read from Old_Files/app.py.original for all code extraction
- Never modify main branch
- No Docker builds until Phase 13 complete
- Batch operations per phase
- Use line numbers from APP_FUNCTION_INVENTORY.md

PHASE 2: Extract Utilities (Complete this entire phase in one response)

STEP 1: Read PROGRESS.md
  cat PROGRESS.md | grep "Phase 2" -A 10

Confirm Phase 2 status is: NOT STARTED or NOW READY
Confirm Phase 1B status is: COMPLETE

STEP 2: Create utils/validators.py

Extract validation functions from Old_Files/app.py.original using line numbers:
  - is_youtube_url (line 2129)
  - should_use_ytdlp (line 833)

Create file: utils/validators.py

Content structure:
---START FILE---
"""URL and input validation utilities."""
import re
from typing import Optional

# Copy is_youtube_url function from line 2129 EXACTLY
def is_youtube_url(url: str) -> bool:
    # [Copy complete function body]
    pass

# Copy should_use_ytdlp function from line 833 EXACTLY
def should_use_ytdlp(url: str) -> bool:
    # [Copy complete function body]
    pass
---END FILE---

STEP 2.1: Enumerate utility functions to extract (strict)
  # List candidate utility functions by name from Old_Files/app.py.original
  # Prefer exact anchors from APP_FUNCTION_INVENTORY.md when available
  python - << 'PY'
import ast
src = open('Old_Files/app.py.original', 'r', encoding='utf-8').read()
tree = ast.parse(src)
funcs = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        funcs.append(node.name)
open('.backups/utils_phase2_funcs.txt', 'w', encoding='utf-8').write('\n'.join(sorted(funcs)))
print('Discovered top-level functions:', len(funcs))
PY
  wc -l .backups/utils_phase2_funcs.txt

Extraction Rules (strict):
  - Move ONLY pure helpers and validators (no side effects, no global writes)
  - Keep names, signatures, and return types EXACTLY
  - Preserve regex patterns and edge-case handling
  - Do NOT move cache functions (that is Phase 3)
  - Do NOT move nested helpers unless specified in inventory

STEP 3: Create utils/helpers.py

Extract helper functions from Old_Files/app.py.original using line numbers:
  - format_duration (line 2099)
  - format_view_count (line 2114)

Create file: utils/helpers.py

Content structure:
---START FILE---
"""General helper utilities."""
from datetime import timedelta
import math
from typing import Any

# Copy format_duration function from line 2099 EXACTLY
def format_duration(seconds: float) -> str:
    # [Copy complete function body]
    pass

# Copy format_view_count function from line 2114 EXACTLY
def format_view_count(count: int) -> str:
    # [Copy complete function body]
    pass
---END FILE---

STEP 3.1: Enumerate helper anchors (strict)
  # Narrow list to known helpers using name filters for verification
  grep -nE "def (format_duration|format_view_count)\(" Old_Files/app.py.original \
    | tee .backups/helpers_phase2_anchors.txt
  wc -l .backups/helpers_phase2_anchors.txt

STEP 4: Create utils/websocket_helpers.py

Extract WebSocket helper functions from Old_Files/app.py.original using line numbers:
  - safe_ws_send (line 241)

Create file: utils/websocket_helpers.py

Content structure:
---START FILE---
"""WebSocket helper utilities for safe message sending."""
import json
import logging
from typing import Optional, Dict, Any
from fastapi import WebSocket
from fastapi.websockets import WebSocketState

# Copy safe_ws_send function from line 241 EXACTLY
async def safe_ws_send(websocket: WebSocket, data: dict) -> None:
    # [Copy complete function body including WebSocketState check]
    pass

# If there are any send_status_update, send_transcription_progress functions, copy them too
---END FILE---

STEP 4.1: Enumerate WebSocket helper references
  # Identify WS helper call sites for import coverage later
  grep -nE "safe_ws_send|send_(status|progress|chunk)" Old_Files/app.py.original \
    > .backups/ws_helpers_phase2_refs.txt || true
  wc -l .backups/ws_helpers_phase2_refs.txt || true

STEP 5: Add Imports to app.py

In working app.py, add these imports:

---ADD TO app.py---
from utils.validators import is_youtube_url, should_use_ytdlp
from utils.helpers import format_duration, format_view_count
from utils.websocket_helpers import safe_ws_send
---END ADDITIONS---

STEP 5.1: Cross-check import coverage in dependent modules
  # app.py imports
  grep -n "from utils.validators" app.py || true
  grep -n "from utils.helpers" app.py || true
  grep -n "from utils.websocket_helpers" app.py || true
  # api/routes.py may import validators/helpers
  test -f api/routes.py && grep -nE "utils\.validators|utils\.helpers" api/routes.py || true
  # services/transcription.py may import websocket helpers
  test -f services/transcription.py && grep -n "utils.websocket_helpers" services/transcription.py || true
  # services/audio_processor.py may import websocket helpers
  test -f services/audio_processor.py && grep -n "utils.websocket_helpers" services/audio_processor.py || true

NOTE: Do NOT remove function definitions from app.py yet. Only ADD imports.

STEP 6: Git Commit
  git add utils/validators.py utils/helpers.py utils/websocket_helpers.py app.py
  git commit -m "PHASE 2: Extract utils modules (validators, helpers, websocket_helpers), add imports to app.py"

STEP 7: Verify
  # Baseline import tests and file line counts
  python -c "from utils.validators import is_youtube_url; print('Validators OK')"
  python -c "from utils.helpers import format_duration; print('Helpers OK')"
  python -c "from utils.websocket_helpers import safe_ws_send; print('WebSocket helpers OK')"
  wc -l utils/validators.py utils/helpers.py utils/websocket_helpers.py

  # Function export checks
  python - << 'PY'
import importlib
v = importlib.import_module('utils.validators')
h = importlib.import_module('utils.helpers')
ws = importlib.import_module('utils.websocket_helpers')
exports_v = [k for k in dir(v) if not k.startswith('_')]
exports_h = [k for k in dir(h) if not k.startswith('_')]
exports_ws = [k for k in dir(ws) if not k.startswith('_')]
print('validators exports:', exports_v)
print('helpers exports:', exports_h)
print('websocket_helpers exports:', exports_ws)
PY

  # app import sanity (no runtime execution)
  python -c "import app; print('app import OK')"

  # Optional runtime check via start.sh and logs (recommended)
  ./start.sh || true
  sleep 2
  tail -n +1 .logs/server-runtime-errors.log
  tail -n 50 .logs/server-output.log
  curl -sSf http://localhost:8009/health | head -c 200 || true
  python -c "from utils.validators import is_youtube_url; print('Validators OK')"
  python -c "from utils.helpers import format_duration; print('Helpers OK')"
  python -c "from utils.websocket_helpers import safe_ws_send; print('WebSocket helpers OK')"
  wc -l utils/validators.py utils/helpers.py utils/websocket_helpers.py

STEP 8: Update PROGRESS.md

Edit PROGRESS.md:

Change Phase 2 section to:
---UPDATE---
## Phase 2: Extract Utilities ✅ COMPLETE

Completed: [CURRENT_TIMESTAMP]

Files Created:
  - utils/validators.py ([X] lines)
  - utils/helpers.py ([Y] lines)
  - utils/websocket_helpers.py ([Z] lines)

Functions Extracted:
  validators.py:
    - is_youtube_url (line 2129)
    - should_use_ytdlp (line 833)
  helpers.py:
    - format_duration (line 2099)
    - format_view_count (line 2114)
  websocket_helpers.py:
    - safe_ws_send (line 241)

Imports Added to app.py:
  - from utils.validators import is_youtube_url, should_use_ytdlp
  - from utils.helpers import format_duration, format_view_count
  - from utils.websocket_helpers import safe_ws_send

Commits: 1
Commit Hash: [git log -1 --format=%H]

Notes:
  - All utility functions extracted
  - Imports added to app.py (additive only)
  - Original app.py code still intact
  - All utils modules verified with import tests

Next Phase: Phase 3 (Extract Cache Management)
---END UPDATE---

Update Phase Status Overview:
  Phase 2: Utils ✅ COMPLETE
  Phase 3: Cache ⏳ NOW READY

STEP 9: Commit Progress Update
  git add PROGRESS.md
  git commit -m "PHASE 2: Update progress tracker - marked complete"

STEP 10: Report Completion

Provide this report:

PHASE 2 COMPLETE

Files Created:
  - utils/validators.py ([X] lines)
  - utils/helpers.py ([Y] lines)
  - utils/websocket_helpers.py ([Z] lines)

Functions Extracted: 5 functions

Verification: All import tests passed

Commits: 2

Next Phase: Phase 3 (Extract Cache Management)

Ready for next instruction.


================================================================================
PHASE 3: EXTRACT CACHE MANAGEMENT
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Read from Old_Files/app.py.original for all code extraction
- Never modify main branch
- No Docker builds until Phase 13 complete
- Batch operations per phase
- Use line numbers from APP_FUNCTION_INVENTORY.md

PHASE 3: Extract Cache Management (Complete this entire phase in one response)

STEP 1: Read PROGRESS.md
  cat PROGRESS.md | grep "Phase 3" -A 10

Confirm Phase 3 status is: NOT STARTED or NOW READY
Confirm Phase 2 status is: COMPLETE

STEP 2: Create utils/cache.py

Extract ALL cache-related functions from Old_Files/app.py.original using line numbers from APP_FUNCTION_INVENTORY.md:
  - init_capture_dir (line 359)
  - init_download_cache_dir (line 366)
  - init_cache_dir (line 2033)
  - get_url_hash (line 387)
  - get_cached_download (line 392)
  - save_download_to_cache (line 417)
  - generate_cache_key (line 2057)
  - get_cached_audio (line 2065)
  - save_to_cache (line 2077)
  - Any calculate_file_hash, get_cached_transcription, save_transcription_to_cache if present
  - Any get_cache_stats, clear_cache functions if present
  - Any get_download_cache_stats, clear_download_cache functions if present

Create file: utils/cache.py

Content structure:
---START FILE---
"""Cache management utilities for audio, downloads, and transcriptions."""
import os
import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from config.constants import CACHE_DIR, DOWNLOAD_CACHE_DIR, CAPTURE_DIR

# Copy init_capture_dir from line 359 EXACTLY
def init_capture_dir() -> None:
    # [Copy complete function body]
    pass

# Copy init_download_cache_dir from line 366 EXACTLY
def init_download_cache_dir() -> None:
    # [Copy complete function body]
    pass

# Copy init_cache_dir from line 2033 EXACTLY
def init_cache_dir() -> None:
    # [Copy complete function body]
    pass

# Copy get_url_hash from line 387 EXACTLY
def get_url_hash(url: str) -> str:
    # [Copy complete function body]
    pass

# Copy get_cached_download from line 392 EXACTLY
def get_cached_download(url: str) -> Optional[str]:
    # [Copy complete function body]
    pass

# Copy save_download_to_cache from line 417 EXACTLY
def save_download_to_cache(url: str, audio_file: str) -> None:
    # [Copy complete function body]
    pass

# Copy generate_cache_key from line 2057 EXACTLY
def generate_cache_key(audio_data, sample_rate, channels) -> str:
    # [Copy complete function body]
    pass

# Copy get_cached_audio from line 2065 EXACTLY
def get_cached_audio(cache_key: str) -> Optional[str]:
    # [Copy complete function body]
    pass

# Copy save_to_cache from line 2077 EXACTLY
def save_to_cache(cache_key: str, audio_path: str) -> None:
    # [Copy complete function body]
    pass

# [Copy ALL other cache-related functions from Old_Files/app.py.original]
# Search for functions with "cache" in their name
# Preserve async keyword if function is async
---END FILE---

STEP 2.1: Enumerate cache-related functions to extract (strict)
  # Discover cache functions and anchors from Old_Files/app.py.original
  # Prefer APP_FUNCTION_INVENTORY.md line numbers where available
  python - << 'PY'
import ast, re
src = open('Old_Files/app.py.original', 'r', encoding='utf-8').read()
tree = ast.parse(src)
funcs = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        name = node.name
        if (
            'cache' in name
            or name in {
                'init_capture_dir', 'init_download_cache_dir', 'init_cache_dir',
                'get_url_hash', 'get_cached_download', 'save_download_to_cache',
                'generate_cache_key', 'get_cached_audio', 'save_to_cache',
            }
        ):
            funcs.append(name)
open('.backups/cache_phase3_funcs.txt', 'w', encoding='utf-8').write('\n'.join(sorted(set(funcs))))
print('Discovered cache-related functions:', len(set(funcs)))
PY
  wc -l .backups/cache_phase3_funcs.txt

Extraction Rules (strict):
  - Move ONLY cache utilities and helpers (file I/O, hashing, read/write)
  - Keep names, signatures, and return types EXACTLY
  - Preserve path handling and `Path` usage; avoid side-effect changes
  - Do NOT move state/global locks (handled in Phase 1B)
  - Preserve async functions as async


STEP 3: Add Imports to app.py

In working app.py, add this import:

---ADD TO app.py---
from utils.cache import (
    init_capture_dir,
    init_download_cache_dir,
    init_cache_dir,
    get_url_hash,
    get_cached_download,
    save_download_to_cache,
    generate_cache_key,
    get_cached_audio,
    save_to_cache,
    # [list ALL other cache functions extracted]
)
---END ADDITIONS---

STEP 3.1: Cross-check import coverage in dependent modules
  # app.py import presence
  grep -n "from utils.cache" app.py || true
  # services depending on cache
  test -f services/transcription.py && grep -nE "utils\.cache|generate_cache_key|get_cached_audio" services/transcription.py || true
  test -f services/audio_processor.py && grep -nE "utils\.cache|get_url_hash|get_cached_download|save_download_to_cache" services/audio_processor.py || true
  # api routes might use cache helpers
  test -f api/routes.py && grep -nE "utils\.cache|init_cache_dir" api/routes.py || true

NOTE: Do NOT remove function definitions from app.py yet. Only ADD imports.

STEP 4: Git Commit
  git add utils/cache.py app.py
  git commit -m "PHASE 3: Extract utils/cache.py with all cache management functions, add imports to app.py"

STEP 5: Verify
  # Baseline import tests and compile
  python -c "from utils.cache import get_url_hash, init_cache_dir; print('Cache OK')"
  python -m py_compile utils/cache.py
  wc -l utils/cache.py

  # Function export checks
  python - << 'PY'
import importlib
c = importlib.import_module('utils.cache')
exports = [k for k in dir(c) if not k.startswith('_')]
required = {
    'init_capture_dir', 'init_download_cache_dir', 'init_cache_dir',
    'get_url_hash', 'get_cached_download', 'save_download_to_cache',
    'generate_cache_key', 'get_cached_audio', 'save_to_cache',
}
print('cache exports:', exports)
missing = sorted(list(required - set(exports)))
print('Missing required cache funcs:', missing)
PY

  # app import sanity (no runtime execution)
  python -c "import app; print('app import OK')"

  # Optional runtime check via start.sh and logs (recommended)
  ./start.sh || true
  sleep 2
  tail -n +1 .logs/server-runtime-errors.log
  tail -n 50 .logs/server-output.log
  curl -sSf http://localhost:8009/health | head -c 200 || true

STEP 6: Update PROGRESS.md

Edit PROGRESS.md:

Change Phase 3 section to:
---UPDATE---
## Phase 3: Extract Cache Management ✅ COMPLETE

Completed: [CURRENT_TIMESTAMP]

Files Created:
  - utils/cache.py ([X] lines)

Functions Extracted:
  - init_capture_dir (line 359)
  - init_download_cache_dir (line 366)
  - init_cache_dir (line 2033)
  - get_url_hash (line 387)
  - get_cached_download (line 392)
  - save_download_to_cache (line 417)
  - generate_cache_key (line 2057)
  - get_cached_audio (line 2065)
  - save_to_cache (line 2077)
  - [list any others extracted]

Imports Added to app.py:
  - from utils.cache import [list all cache functions]

Commits: 1
Commit Hash: [git log -1 --format=%H]

Notes:
  - All cache management functions extracted
  - Async functions preserved where present
  - File I/O patterns maintained
  - Imports added to app.py (additive only)
  - Original app.py code still intact
  - Cache module verified with import test

Next Phase: Phase 4 (Extract Model Management)
---END UPDATE---

Update Phase Status Overview:
  Phase 3: Cache ✅ COMPLETE
  Phase 4: Models ⏳ NOW READY

STEP 7: Commit Progress Update
  git add PROGRESS.md
  git commit -m "PHASE 3: Update progress tracker - marked complete"

STEP 8: Report Completion

Provide this report:

PHASE 3 COMPLETE

Files Created:
  - utils/cache.py ([X] lines)

Functions Extracted: [number] cache functions

Verification: Import test and compile check passed

Commits: 2

Next Phase: Phase 4 (Extract Model Management - HIGH RISK)

Ready for next instruction.


================================================================================
PHASE 4: EXTRACT MODEL MANAGEMENT
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Read from Old_Files/app.py.original for all code extraction
- Never modify main branch
- No Docker builds until Phase 13 complete
- Batch operations per phase
- Use line numbers from APP_FUNCTION_INVENTORY.md

WARNING: HIGH RISK PHASE - Thread safety and global state critical

PHASE 4: Extract Model Management (Complete this entire phase in one response)

STEP 1: Read PROGRESS.md
  cat PROGRESS.md | grep "Phase 4" -A 10

Confirm Phase 4 status is: NOT STARTED or NOW READY
Confirm Phase 3 status is: COMPLETE

STEP 2: Create models/loader.py

Extract model loading functions from Old_Files/app.py.original using line numbers:
  - load_model (line 452) - CRITICAL: Preserve double-check locking pattern
  - get_diarization_pipeline (line 553) - CRITICAL: Preserve thread safety

Create file: models/loader.py

Content structure:
---START FILE---
"""Model loading with thread-safe singleton pattern.

CRITICAL: This module handles global state and thread-safe model loading.
The double-check locking pattern MUST be preserved exactly.
"""
import threading
import logging
from typing import Any, Optional
import torch

from config.settings import (
    WHISPER_MODEL,
    IVRIT_MODEL_NAME,
    IVRIT_DEVICE,
    IVRIT_COMPUTE_TYPE,
    IVRIT_BEAM_SIZE,
)
from core.state import (
    whisper_models,
    faster_whisper_models,
    diarization_pipeline,
    model_lock,
    faster_whisper_lock,
    diarization_lock,
)

# Copy load_model function from line 452 EXACTLY
# CRITICAL: Must use 'global' keyword for whisper_models, faster_whisper_models
# CRITICAL: Must preserve double-check locking pattern:
#   1. Check if model in cache (outside lock)
#   2. Acquire lock
#   3. Check again if model in cache (inside lock)
#   4. Load model if not in cache
# CRITICAL: Must preserve with model_lock / with faster_whisper_lock blocks
def load_model(model_name: str) -> Any:
    global whisper_models, faster_whisper_models
    # [Copy EXACT function body from line 452]
    # Do NOT modify locking logic
    # Do NOT modify model selection logic
    # Do NOT modify error handling
    pass

# Copy get_diarization_pipeline function from line 553 EXACTLY
# CRITICAL: Must use 'global' keyword for diarization_pipeline
# CRITICAL: Must preserve with diarization_lock block
def get_diarization_pipeline() -> Any:
    global diarization_pipeline
    # [Copy EXACT function body from line 553]
    # Do NOT modify locking logic
    # Do NOT modify pipeline initialization
    pass
---END FILE---

STEP 2.1: Enumerate model-management functions and locks (strict)
  # Enumerate functions and check for lock usage and globals
  python - << 'PY'
import ast
src = open('Old_Files/app.py.original', 'r', encoding='utf-8').read()
tree = ast.parse(src)
funcs = []
globals_used = {}
locks_used = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        name = node.name
        if name in {'load_model', 'get_diarization_pipeline'}:
            funcs.append(name)
            # Inspect body for 'global' statements and lock usage
            gdecl = []
            lusage = []
            for n in ast.walk(node):
                if isinstance(n, ast.Global):
                    gdecl.extend(n.names)
                if isinstance(n, ast.With):
                    # Capture lock names in with statements
                    for item in n.items:
                        ctx = item.context_expr
                        if isinstance(ctx, ast.Name):
                            lusage.append(ctx.id)
                        elif isinstance(ctx, ast.Attribute):
                            lusage.append(ctx.attr)
            globals_used[name] = sorted(set(gdecl))
            locks_used[name] = sorted(set(lusage))
open('.backups/models_phase4_funcs.txt', 'w', encoding='utf-8').write('\n'.join(funcs))
open('.backups/models_phase4_globals.json', 'w', encoding='utf-8').write(str(globals_used))
open('.backups/models_phase4_locks.json', 'w', encoding='utf-8').write(str(locks_used))
print('Discovered model funcs:', funcs)
print('Globals per func:', globals_used)
print('Locks per func:', locks_used)
PY
  wc -l .backups/models_phase4_funcs.txt

Thread-Safety Guardrails (strict):
  - Preserve double-check locking in `load_model` (outside check → acquire lock → inside check → load)
  - Use `global` for `whisper_models`, `faster_whisper_models`, `diarization_pipeline`
  - Preserve `with model_lock`, `with faster_whisper_lock`, `with diarization_lock` blocks
  - Do NOT alter device selection, error handling, or cache lookup logic
  - Keep function signatures and returns EXACTLY


STEP 3: Add Imports to app.py

In working app.py, add this import:

---ADD TO app.py---
from models.loader import load_model, get_diarization_pipeline
---END ADDITIONS---

STEP 3.1: Cross-check import coverage in dependent modules
  # app.py import presence
  grep -n "from models.loader" app.py || true
  # services that might call model loader
  test -f services/transcription.py && grep -nE "load_model|get_diarization_pipeline" services/transcription.py || true
  test -f services/audio_processor.py && grep -nE "load_model|get_diarization_pipeline" services/audio_processor.py || true
  # api routes may initialize models
  test -f api/routes.py && grep -nE "load_model|get_diarization_pipeline" api/routes.py || true

NOTE: Do NOT remove function definitions from app.py yet. Only ADD imports.

STEP 4: Git Commit
  git add models/loader.py app.py
  git commit -m "PHASE 4: Extract models/loader.py with thread-safe model loading, add imports to app.py"

STEP 5: Verify
  # Baseline import tests and compile
  python -c "from models.loader import load_model, get_diarization_pipeline; print('Model loader OK')"
  python -m py_compile models/loader.py
  wc -l models/loader.py

  # Lock and globals presence checks (report-only)
  python - << 'PY'
import inspect
from models import loader
lm = loader.load_model
dp = loader.get_diarization_pipeline
src_lm = inspect.getsource(lm)
src_dp = inspect.getsource(dp)
def contains_all(s, toks):
    return all(t in s for t in toks)
print('load_model has globals:', contains_all(src_lm, ['global whisper_models', 'global faster_whisper_models']))
print('load_model uses locks:', contains_all(src_lm, ['with model_lock', 'with faster_whisper_lock']))
print('get_diarization_pipeline has global:', 'global diarization_pipeline' in src_dp)
print('get_diarization_pipeline uses lock:', 'with diarization_lock' in src_dp)
PY

  # app import sanity (no runtime execution)
  python -c "import app; print('app import OK')"

  # Optional runtime check via start.sh and logs (recommended)
  ./start.sh || true
  sleep 2
  tail -n +1 .logs/server-runtime-errors.log
  tail -n 50 .logs/server-output.log
  curl -sSf http://localhost:8009/health | head -c 200 || true

STEP 6: Update PROGRESS.md

Edit PROGRESS.md:

Change Phase 4 section to:
---UPDATE---
## Phase 4: Extract Model Management ✅ COMPLETE

Completed: [CURRENT_TIMESTAMP]

Risk Level: HIGH (Thread safety, global state)

Files Created:
  - models/loader.py ([X] lines)

Functions Extracted:
  - load_model (line 452) - Thread-safe with double-check locking
  - get_diarization_pipeline (line 553) - Thread-safe with lock

Critical Patterns Preserved:
  - Double-check locking for model caching
  - global keyword for state modifications
  - with lock: blocks for thread safety
  - Error handling intact
  - GPU/CPU device selection logic

Imports Added to app.py:
  - from models.loader import load_model, get_diarization_pipeline

Commits: 1
Commit Hash: [git log -1 --format=%H]

Notes:
  - Thread safety patterns verified intact
  - Global state modifications use 'global' keyword
  - Locking patterns preserved exactly
  - Imports added to app.py (additive only)
  - Original app.py code still intact
  - Model loader module verified with import test

CRITICAL CHECKPOINT: Model loading is core functionality with thread safety.
Verify thread-safe patterns before proceeding.

Next Phase: Phase 5 (Extract Audio Processing - HIGH RISK)
---END UPDATE---

Update Phase Status Overview:
  Phase 4: Models ✅ COMPLETE
  Phase 5: Audio ⏳ NOW READY

STEP 7: Commit Progress Update
  git add PROGRESS.md
  git commit -m "PHASE 4: Update progress tracker - marked complete"

STEP 8: Report Completion

Provide this report:

PHASE 4 COMPLETE

Risk Level: HIGH (Thread safety maintained)

Files Created:
  - models/loader.py ([X] lines)

Functions Extracted: 2 thread-safe functions

Critical Patterns: Double-check locking preserved

Verification: Import test and compile check passed

Commits: 2

Next Phase: Phase 5 (Extract Audio Processing - HIGH RISK, Async + Subprocess)

Ready for next instruction.


================================================================================
PHASE 5: EXTRACT AUDIO PROCESSING
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Read from Old_Files/app.py.original for all code extraction
- Never modify main branch
- No Docker builds until Phase 13 complete
- Batch operations per phase
- Use line numbers from APP_FUNCTION_INVENTORY.md

WARNING: HIGH RISK PHASE - Async functions, nested helpers, subprocess management

PHASE 5: Extract Audio Processing (Complete this entire phase in one response)

STEP 1: Read PROGRESS.md
  cat PROGRESS.md | grep "Phase 5" -A 10

Confirm Phase 5 status is: NOT STARTED or NOW READY
Confirm Phase 4 status is: COMPLETE

STEP 2: Create services/audio_processor.py

Extract audio processing functions from Old_Files/app.py.original:

CRITICAL - ATOMIC MOVES (keep nested functions with parents):
  - download_audio_with_ffmpeg (line 866) + nested monitor_progress (line 944) + nested read_progress_file (line 960)
  - download_with_fallback (line 1159)
  - download_audio_with_ytdlp_async (line 1225)
  - get_audio_duration_seconds (line 1460)
  - calculate_progress_metrics (line 1483)
  - split_audio_for_incremental (line 1531)
  - split_audio_into_chunks (line 1578)
  - AudioStreamProcessor class (lines 2220-2333) - COMPLETE CLASS AS UNIT

STEP 2.1: Enumerate audio-processing functions and nested helpers (strict)
  # Use Python AST to list top-level functions/classes and detect async
  # Also enumerate nested helpers inside parent functions (maintain closure relationships)
  python - << 'PY'
import ast, json, pathlib, re
p = pathlib.Path('Old_Files/app.py.original')
src = p.read_text(encoding='utf-8')
tree = ast.parse(src)
targets = {
    'download_audio_with_ffmpeg',
    'download_with_fallback',
    'download_audio_with_ytdlp_async',
    'get_audio_duration_seconds',
    'calculate_progress_metrics',
    'split_audio_for_incremental',
    'split_audio_into_chunks',
    'AudioStreamProcessor',
}

items = []
nested = {}
async_map = {}

def collect_nested(body):
    return [
        {'name': n.name, 'kind': 'async' if isinstance(n, ast.AsyncFunctionDef) else 'def', 'lineno': n.lineno}
        for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in targets:
        items.append({'name': node.name, 'kind': 'async' if isinstance(node, ast.AsyncFunctionDef) else 'def', 'lineno': node.lineno})
        nested[node.name] = collect_nested(node.body)
        async_map[node.name] = isinstance(node, ast.AsyncFunctionDef)
    elif isinstance(node, ast.ClassDef) and node.name in targets:
        items.append({'name': node.name, 'kind': 'class', 'lineno': node.lineno})

outdir = pathlib.Path('.backups')
outdir.mkdir(exist_ok=True)
pathlib.Path(outdir, 'audio_phase5_funcs.txt').write_text('\n'.join([f"{i['kind']} {i['name']} @ {i['lineno']}" for i in items]), encoding='utf-8')
pathlib.Path(outdir, 'audio_phase5_nested.json').write_text(json.dumps(nested, indent=2), encoding='utf-8')
pathlib.Path(outdir, 'audio_phase5_async.json').write_text(json.dumps(async_map, indent=2), encoding='utf-8')

# Report-only: enumerate subprocess and asyncio usage within relevant ranges
subproc_hits = []
for m in re.finditer(r"^(.*)(subprocess\.|asyncio\.create_subprocess_exec)", src, re.M):
    line_no = src.count('\n', 0, m.start()) + 1
    subproc_hits.append({'lineno': line_no, 'line': m.group(0).strip()})
pathlib.Path(outdir, 'audio_phase5_subprocess.json').write_text(json.dumps(subproc_hits[:200], indent=2), encoding='utf-8')

# Report-only: list WebSocket references and safe send helper usage
ws_hits = []
for m in re.finditer(r"WebSocket|safe_ws_send", src):
    line_no = src.count('\n', 0, m.start()) + 1
    ws_hits.append({'lineno': line_no, 'token': m.group(0)})
pathlib.Path(outdir, 'audio_phase5_websocket.json').write_text(json.dumps(ws_hits[:200], indent=2), encoding='utf-8')
print('Phase 5 enumeration complete')
PY

STEP 2.2: Strict extraction guardrails (Audio)
  - Keep nested helpers INSIDE parent functions (closures must remain intact)
  - Preserve all `async` keywords and exact function signatures/param order
  - Preserve `asyncio.create_subprocess_exec`, `subprocess.Popen/run`, and IO patterns
  - Preserve WebSocket sends and checks; prefer `safe_ws_send` if present
  - Maintain cache interactions: `get_cached_download`, `save_download_to_cache`
  - Keep constants usage (`SAMPLE_RATE`, `CHANNELS`, etc.) via `config.constants`
  - Maintain temp file and directory lifecycle (`tempfile`, `shutil`, `Path`)
  - Keep queue/threading patterns inside `AudioStreamProcessor` (stream safety)
  - Do not alter error handling or logging semantics
  - Do not change return types; keep type hints where present

STEP 2.3: Cross-check dependent import coverage (Audio)
  - Confirm imports added to `app.py` for all 8 exports
  - Scan dependent modules and note references (report-only):
    - `api/websocket.py`: usage of download helpers or `AudioStreamProcessor`
    - `api/routes.py`: any audio-related utilities, if present
    - `services/transcription.py`: streaming/audio interactions
    - `utils/websocket_helpers.py`: ensure `safe_ws_send` exists and matches usage
  - Record results to `.backups/audio_phase5_dependents.txt` for traceability

  python - << 'PY'
import pathlib, re, json
targets = [
    'download_audio_with_ffmpeg',
    'download_with_fallback',
    'download_audio_with_ytdlp_async',
    'get_audio_duration_seconds',
    'calculate_progress_metrics',
    'split_audio_for_incremental',
    'split_audio_into_chunks',
    'AudioStreamProcessor',
    'safe_ws_send',
]
files = [
    pathlib.Path('api/websocket.py'),
    pathlib.Path('api/routes.py'),
    pathlib.Path('services/transcription.py'),
    pathlib.Path('utils/websocket_helpers.py'),
]
results = {}
for f in files:
    if not f.exists():
        results[str(f)] = {'exists': False, 'hits': []}
        continue
    text = f.read_text(encoding='utf-8')
    hits = []
    for t in targets:
        for m in re.finditer(rf"\b{re.escape(t)}\b", text):
            line_no = text.count('\n', 0, m.start()) + 1
            hits.append({'symbol': t, 'lineno': line_no})
    results[str(f)] = {'exists': True, 'hits': hits}
outdir = pathlib.Path('.backups')
outdir.mkdir(exist_ok=True)
pathlib.Path(outdir, 'audio_phase5_dependents.txt').write_text(json.dumps(results, indent=2), encoding='utf-8')
print('Dependent coverage scan complete')
PY

Create file: services/audio_processor.py

Content structure:
---START FILE---
"""Audio download and processing services.

CRITICAL: Contains async functions, nested helpers, and subprocess management.
Nested functions MUST stay with their parent functions (use parent variables via closure).
"""
import asyncio
import logging
import os
import subprocess
import tempfile
import shutil
import re
import time
import queue
import threading
from pathlib import Path
from typing import Optional, List, Tuple
from fastapi import WebSocket

from config.settings import (
    # [import relevant settings]
)
from config.constants import (
    SAMPLE_RATE,
    CHANNELS,
    CHUNK_DURATION,
    CHUNK_OVERLAP,
    AUDIO_QUEUE_SIZE,
    CACHE_DIR,
    DOWNLOAD_CACHE_DIR,
)
from utils.cache import get_cached_download, save_download_to_cache
from utils.websocket_helpers import safe_ws_send
from utils.validators import should_use_ytdlp

# Copy download_audio_with_ffmpeg from line 866 EXACTLY
# CRITICAL: Keep nested monitor_progress (line 944) INSIDE this function
# CRITICAL: Keep nested read_progress_file (line 960) INSIDE monitor_progress
# Do NOT separate nested functions - they access parent function variables
async def download_audio_with_ffmpeg(url, format, duration, websocket, use_cache):
    # [Copy EXACT function body starting from line 866]
    # Must include complete nested function monitor_progress at line 944
    # Must include complete nested function read_progress_file at line 960
    # Preserve ALL async/await patterns
    # Preserve subprocess.create_subprocess_exec patterns
    pass

# Copy download_with_fallback from line 1159 EXACTLY
async def download_with_fallback(url, language, format, websocket, use_cache):
    # [Copy EXACT function body]
    # Preserve async keyword
    # Preserve error handling
    pass

# Copy download_audio_with_ytdlp_async from line 1225 EXACTLY
async def download_audio_with_ytdlp_async(url, language, format, websocket, use_cache):
    # [Copy EXACT function body]
    # Preserve async keyword
    # Preserve yt-dlp integration
    pass

# Copy get_audio_duration_seconds from line 1460 EXACTLY
def get_audio_duration_seconds(audio_path: str) -> float:
    # [Copy EXACT function body]
    # Preserve ffprobe subprocess call
    pass

# Copy calculate_progress_metrics from line 1483 EXACTLY
def calculate_progress_metrics(audio_duration, elapsed_time, processed_chunks, total_chunks) -> dict:
    # [Copy EXACT function body]
    pass

# Copy split_audio_for_incremental from line 1531 EXACTLY
def split_audio_for_incremental(audio_path, chunk_seconds, overlap_seconds) -> Tuple[str, List[str]]:
    # [Copy EXACT function body]
    pass

# Copy split_audio_into_chunks from line 1578 EXACTLY
def split_audio_into_chunks(audio_path, chunk_seconds, overlap_seconds) -> List[str]:
    # [Copy EXACT function body]
    pass

# Copy AudioStreamProcessor class from lines 2220-2333 EXACTLY
# CRITICAL: Copy COMPLETE CLASS as single unit
# Include all 4 methods: __init__, start_ffmpeg_stream, read_audio_chunks, stop
class AudioStreamProcessor:
    # [Copy EXACT class definition from line 2220]
    # Include __init__ (line 2220)
    # Include start_ffmpeg_stream (line 2229)
    # Include read_audio_chunks (line 2258)
    # Include stop (line 2321)
    # Preserve all instance variables
    # Preserve subprocess management
    # Preserve queue handling
    pass
---END FILE---

STEP 3: Add Imports to app.py

In working app.py, add this import:

---ADD TO app.py---
from services.audio_processor import (
    download_audio_with_ffmpeg,
    download_with_fallback,
    download_audio_with_ytdlp_async,
    get_audio_duration_seconds,
    calculate_progress_metrics,
    split_audio_for_incremental,
    split_audio_into_chunks,
    AudioStreamProcessor,
)
---END ADDITIONS---

NOTE: Do NOT remove function definitions from app.py yet. Only ADD imports.

STEP 4: Git Commit
  git add services/audio_processor.py app.py
  git commit -m "PHASE 5: Extract services/audio_processor.py with async functions and AudioStreamProcessor class, add imports to app.py"

STEP 5: Verify
  python -c "from services.audio_processor import AudioStreamProcessor; print('Audio processor OK')"
  python -m py_compile services/audio_processor.py
  wc -l services/audio_processor.py

STEP 5.1: Enhanced Verification (Audio)
  # Baseline import tests for all exports and async indicators
  python - << 'PY'
import inspect
from services.audio_processor import (
    download_audio_with_ffmpeg,
    download_with_fallback,
    download_audio_with_ytdlp_async,
    get_audio_duration_seconds,
    calculate_progress_metrics,
    split_audio_for_incremental,
    split_audio_into_chunks,
    AudioStreamProcessor,
)
assert inspect.iscoroutinefunction(download_audio_with_ffmpeg)
assert inspect.iscoroutinefunction(download_with_fallback)
assert inspect.iscoroutinefunction(download_audio_with_ytdlp_async)
assert inspect.isfunction(get_audio_duration_seconds)
assert inspect.isfunction(calculate_progress_metrics)
assert inspect.isfunction(split_audio_for_incremental)
assert inspect.isfunction(split_audio_into_chunks)
assert inspect.isclass(AudioStreamProcessor)
print('Baseline import tests OK')
PY

  # Export listing (report-only)
  python - << 'PY'
import pkgutil, importlib
m = importlib.import_module('services.audio_processor')
exports = sorted([n for n in dir(m) if not n.startswith('_')])
print('\n'.join(exports))
PY

  # Verify nested helpers preserved in extracted file (AST scan)
  python - << 'PY'
import ast, pathlib, json
fp = pathlib.Path('services/audio_processor.py')
src = fp.read_text(encoding='utf-8')
tree = ast.parse(src)
def find_func(name):
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None
ff = find_func('download_audio_with_ffmpeg')
assert ff is not None, 'download_audio_with_ffmpeg missing'
inner = [n.name for n in ff.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
assert 'monitor_progress' in inner, 'monitor_progress not nested in ffmpeg'
# monitor_progress should contain read_progress_file nested
mp = None
for n in ff.body:
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'monitor_progress':
        mp = n
        break
assert mp is not None, 'monitor_progress missing'
mp_inner = [n.name for n in mp.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
assert 'read_progress_file' in mp_inner, 'read_progress_file not nested properly'
print('Nested helper verification OK')
PY

  # App import sanity (ensures additive imports do not break app module)
  python - << 'PY'
import app
print('app import OK')
PY

  # Optional runtime check (report-only): start server and inspect logs
  # bash ./start.sh & sleep 4; tail -n 100 .logs/server-runtime-errors.log; tail -n 100 .logs/server-output.log

STEP 6: Update PROGRESS.md

Edit PROGRESS.md:

Change Phase 5 section to:
---UPDATE---
## Phase 5: Extract Audio Processing ✅ COMPLETE

Completed: [CURRENT_TIMESTAMP]

Risk Level: HIGH (Async, subprocess, nested functions)

Files Created:
  - services/audio_processor.py ([X] lines)

Functions Extracted:
  - download_audio_with_ffmpeg (line 866) + nested helpers (lines 944, 960) [ATOMIC]
  - download_with_fallback (line 1159)
  - download_audio_with_ytdlp_async (line 1225)
  - get_audio_duration_seconds (line 1460)
  - calculate_progress_metrics (line 1483)
  - split_audio_for_incremental (line 1531)
  - split_audio_into_chunks (line 1578)

Classes Extracted:
  - AudioStreamProcessor (lines 2220-2333) [COMPLETE CLASS]
    - __init__, start_ffmpeg_stream, read_audio_chunks, stop

Critical Patterns Preserved:
  - Nested functions kept with parents (closures intact)
  - All async keywords preserved
  - asyncio.create_subprocess_exec patterns intact
  - WebSocket communication preserved
  - Error handling intact

Imports Added to app.py:
  - from services.audio_processor import [8 items]

Commits: 1
Commit Hash: [git log -1 --format=%H]

Notes:
  - Nested functions NOT separated (closures preserved)
  - AudioStreamProcessor extracted as complete unit
  - All async patterns verified intact
  - Subprocess management preserved
  - Imports added to app.py (additive only)
  - Original app.py code still intact
  - Audio processor module verified

Next Phase: Phase 6 (Extract Transcription Services - HIGH RISK)
---END UPDATE---

Update Phase Status Overview:
  Phase 5: Audio ✅ COMPLETE
  Phase 6: Transcription ⏳ NOW READY

STEP 7: Commit Progress Update
  git add PROGRESS.md
  git commit -m "PHASE 5: Update progress tracker - marked complete"

STEP 8: Report Completion

Provide this report:

PHASE 5 COMPLETE

Risk Level: HIGH (Async and subprocess maintained)

Files Created:
  - services/audio_processor.py ([X] lines)

Functions Extracted: 7 functions + 1 class (4 methods)

Critical Patterns: Nested functions, async, subprocess all preserved

Verification: Import test and compile check passed

Commits: 2

Next Phase: Phase 6 (Extract Transcription Services - HIGH RISK, Multiple nested helpers)

Ready for next instruction.


================================================================================
PHASE 6: EXTRACT TRANSCRIPTION SERVICES
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Read from Old_Files/app.py.original for all code extraction
- Never modify main branch
- No Docker builds until Phase 13 complete
- Batch operations per phase
- Use line numbers from APP_FUNCTION_INVENTORY.md

WARNING: HIGH RISK PHASE - Multiple async functions with nested helpers, Deepgram integration

PHASE 6: Extract Transcription Services (Complete this entire phase in one response)

STEP 1: Read PROGRESS.md
  cat PROGRESS.md | grep "Phase 6" -A 10

Confirm Phase 6 status is: NOT STARTED or NOW READY
Confirm Phase 5 status is: COMPLETE

STEP 2: Create services/transcription.py

Extract transcription functions from Old_Files/app.py.original:

CRITICAL - ATOMIC MOVES (keep nested functions with parents):
  - transcribe_with_incremental_output (line 1626) + 4 nested helpers (lines 1671, 1728, 1831, 1866)
  - transcribe_chunk (line 1977)
  - transcribe_audio_stream (line 2335)
  - transcribe_vod_with_deepgram (line 2502) + nested read_audio_file (line 2610)
  - transcribe_with_deepgram (line 2781) + 4 nested helpers (lines 2794, 2845, 2874, 2887)

Create file: services/transcription.py

STEP 2.1: Enumerate transcription functions and nested helpers (strict)
  # Use Python AST to list target functions, nested helpers, and async flags
  # Targets include Deepgram-related helpers and streaming functions
  python - << 'PY'
import ast, json, pathlib, re
p = pathlib.Path('Old_Files/app.py.original')
src = p.read_text(encoding='utf-8')
tree = ast.parse(src)
targets = {
    'transcribe_with_incremental_output',
    'transcribe_chunk',
    'transcribe_audio_stream',
    'transcribe_vod_with_deepgram',
    'transcribe_with_deepgram',
}

items = []
nested = {}
async_map = {}

def collect_nested (body):
    return [
        {'name': n.name, 'kind': 'async' if isinstance(n, ast.AsyncFunctionDef) else 'def', 'lineno': n.lineno}
        for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in targets:
        items.append({'name': node.name, 'kind': 'async' if isinstance(node, ast.AsyncFunctionDef) else 'def', 'lineno': node.lineno})
        nested[node.name] = collect_nested(node.body)
        async_map[node.name] = isinstance(node, ast.AsyncFunctionDef)

outdir = pathlib.Path('.backups')
outdir.mkdir(exist_ok=True)
pathlib.Path(outdir, 'transcribe_phase6_funcs.txt').write_text('\n'.join([f"{i['kind']} {i['name']} @ {i['lineno']}" for i in items]), encoding='utf-8')
pathlib.Path(outdir, 'transcribe_phase6_nested.json').write_text(json.dumps(nested, indent=2), encoding='utf-8')
pathlib.Path(outdir, 'transcribe_phase6_async.json').write_text(json.dumps(async_map, indent=2), encoding='utf-8')

# Report-only: enumerate Deepgram/WebSocket/ThreadPool usage within relevant ranges
hits = []
patterns = [r"Deepgram", r"safe_ws_send", r"WebSocket", r"ThreadPoolExecutor", r"deepgram"]
for pat in patterns:
    for m in re.finditer(pat, src):
        line_no = src.count('\n', 0, m.start()) + 1
        hits.append({'pattern': pat, 'lineno': line_no})
pathlib.Path(outdir, 'transcribe_phase6_integrations.json').write_text(json.dumps(hits[:300], indent=2), encoding='utf-8')
print('Phase 6 enumeration complete')
PY

STEP 2.2: Strict extraction guardrails (Transcription)
  - Keep nested helpers INSIDE parent functions (closures must remain intact)
  - Preserve all `async` keywords and exact function signatures/param order
  - Maintain ThreadPoolExecutor usage and concurrency patterns
  - Preserve Deepgram SDK and WebSocket handler structure (`on_message`, `on_close`, `on_error`)
  - Keep `safe_ws_send` usage and WebSocket state checks where present
  - Do not alter error handling, logging messages, or return types
  - Maintain constants and configuration imports (e.g., `DEEPGRAM_API_KEY`)
  - Keep streaming loop semantics and backpressure logic
  - Preserve model access via `models.loader.load_model` and state via `core.state`

STEP 2.3: Cross-check dependent import coverage (Transcription)
  - Confirm imports added to `app.py` for all 5 exports
  - Scan dependent modules and note references (report-only):
    - `api/websocket.py`: orchestration calls and streaming use
    - `api/routes.py`: VOD endpoints and transcription triggers
    - `services/audio_processor.py`: interactions with `AudioStreamProcessor`
    - `utils/websocket_helpers.py`: ensure `safe_ws_send` exists and matches usage
  - Record results to `.backups/transcribe_phase6_dependents.txt` for traceability

  python - << 'PY'
import pathlib, re, json
targets = [
    'transcribe_with_incremental_output',
    'transcribe_chunk',
    'transcribe_audio_stream',
    'transcribe_vod_with_deepgram',
    'transcribe_with_deepgram',
    'safe_ws_send',
]
files = [
    pathlib.Path('api/websocket.py'),
    pathlib.Path('api/routes.py'),
    pathlib.Path('services/audio_processor.py'),
    pathlib.Path('utils/websocket_helpers.py'),
]
results = {}
for f in files:
    if not f.exists():
        results[str(f)] = {'exists': False, 'hits': []}
        continue
    text = f.read_text(encoding='utf-8')
    hits = []
    for t in targets:
        for m in re.finditer(rf"\b{re.escape(t)}\b", text):
            line_no = text.count('\n', 0, m.start()) + 1
            hits.append({'symbol': t, 'lineno': line_no})
    results[str(f)] = {'exists': True, 'hits': hits}
outdir = pathlib.Path('.backups')
outdir.mkdir(exist_ok=True)
pathlib.Path(outdir, 'transcribe_phase6_dependents.txt').write_text(json.dumps(results, indent=2), encoding='utf-8')
print('Dependent coverage scan complete')
PY

Content structure:
---START FILE---
"""Transcription services for audio processing.

CRITICAL: Contains async functions with nested helpers and Deepgram integration.
Nested functions MUST stay with their parent functions (use parent variables via closure).
"""
import asyncio
import logging
import io
from typing import Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
from fastapi import WebSocket

from models.loader import load_model
from services.audio_processor import (
    split_audio_for_incremental,
    get_audio_duration_seconds,
    AudioStreamProcessor,
)
from utils.websocket_helpers import safe_ws_send
from core.state import executor, whisper_models, faster_whisper_models
from config.settings import DEEPGRAM_API_KEY

# Copy transcribe_with_incremental_output from line 1626 EXACTLY
# CRITICAL: Keep ALL 4 nested helpers INSIDE this function
# - run_fw_transcription (line 1671)
# - run_transcription (line 1728)
# - transcribe_fw_chunk (line 1831)
# - transcribe_openai_chunk (line 1866)
# Do NOT separate nested functions - they access parent function variables
async def transcribe_with_incremental_output(model, model_config, audio_file, language, websocket, model_name, chunk_seconds):
    # [Copy EXACT function body starting from line 1626]
    # Must include complete nested function run_fw_transcription at line 1671
    # Must include complete nested function run_transcription at line 1728
    # Must include complete nested function transcribe_fw_chunk at line 1831
    # Must include complete nested function transcribe_openai_chunk at line 1866
    # Preserve ThreadPoolExecutor usage
    # Preserve ALL async/await patterns
    # Preserve WebSocket updates
    pass

# Copy transcribe_chunk from line 1977 EXACTLY
def transcribe_chunk(model_config, model, chunk_path, language) -> Dict:
    # [Copy EXACT function body]
    pass

# Copy transcribe_audio_stream from line 2335 EXACTLY
async def transcribe_audio_stream(websocket: WebSocket, processor) -> None:
    # [Copy EXACT function body]
    # Preserve async keyword
    # Preserve WebSocket communication
    pass

# Copy transcribe_vod_with_deepgram from line 2502 EXACTLY
# CRITICAL: Keep nested read_audio_file (line 2610) INSIDE this function
async def transcribe_vod_with_deepgram(websocket, url, language):
    # [Copy EXACT function body starting from line 2502]
    # Must include complete nested function read_audio_file at line 2610
    # Preserve Deepgram SDK usage
    # Preserve async patterns
    pass

# Copy transcribe_with_deepgram from line 2781 EXACTLY
# CRITICAL: Keep ALL 4 nested helpers INSIDE this function
# - extract_deepgram_transcript (line 2794)
# - on_message (line 2845)
# - on_close (line 2874)
# - on_error (line 2887)
async def transcribe_with_deepgram(websocket, url, language):
    # [Copy EXACT function body starting from line 2781]
    # Must include complete nested function extract_deepgram_transcript at line 2794
    # Must include complete nested function on_message at line 2845
    # Must include complete nested function on_close at line 2874
    # Must include complete nested function on_error at line 2887
    # Preserve Deepgram WebSocket handlers
    # Preserve error handling
    pass
---END FILE---

STEP 3: Add Imports to app.py

In working app.py, add this import:

---ADD TO app.py---
from services.transcription import (
    transcribe_with_incremental_output,
    transcribe_chunk,
    transcribe_audio_stream,
    transcribe_vod_with_deepgram,
    transcribe_with_deepgram,
)
---END ADDITIONS---

NOTE: Do NOT remove function definitions from app.py yet. Only ADD imports.

STEP 4: Git Commit
  git add services/transcription.py app.py
  git commit -m "PHASE 6: Extract services/transcription.py with async functions and nested helpers, add imports to app.py"

STEP 5: Verify
  python -c "from services.transcription import transcribe_chunk; print('Transcription OK')"
  python -m py_compile services/transcription.py
  wc -l services/transcription.py

STEP 5.1: Enhanced Verification (Transcription)
  # Baseline import tests for all exports and async indicators
  python - << 'PY'
import inspect
from services.transcription import (
    transcribe_with_incremental_output,
    transcribe_chunk,
    transcribe_audio_stream,
    transcribe_vod_with_deepgram,
    transcribe_with_deepgram,
)
assert inspect.iscoroutinefunction(transcribe_with_incremental_output)
assert inspect.isfunction(transcribe_chunk)
assert inspect.iscoroutinefunction(transcribe_audio_stream)
assert inspect.iscoroutinefunction(transcribe_vod_with_deepgram)
assert inspect.iscoroutinefunction(transcribe_with_deepgram)
print('Baseline import tests OK')
PY

  # Export listing (report-only)
  python - << 'PY'
import importlib
m = importlib.import_module('services.transcription')
exports = sorted([n for n in dir(m) if not n.startswith('_')])
print('\n'.join(exports))
PY

  # Verify nested helpers preserved in extracted file (AST scan)
  python - << 'PY'
import ast, pathlib, json
fp = pathlib.Path('services/transcription.py')
src = fp.read_text(encoding='utf-8')
tree = ast.parse(src)
def find_func (name):
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None

inc = find_func('transcribe_with_incremental_output')
assert inc is not None, 'transcribe_with_incremental_output missing'
inc_inner = [n.name for n in inc.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
for required in ['run_fw_transcription', 'run_transcription', 'transcribe_fw_chunk', 'transcribe_openai_chunk']:
    assert required in inc_inner, f'{required} not nested in incremental function'

dg_vod = find_func('transcribe_vod_with_deepgram')
assert dg_vod is not None, 'transcribe_vod_with_deepgram missing'
vod_inner = [n.name for n in dg_vod.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
assert 'read_audio_file' in vod_inner, 'read_audio_file not nested in Deepgram VOD'

dg = find_func('transcribe_with_deepgram')
assert dg is not None, 'transcribe_with_deepgram missing'
dg_inner = [n.name for n in dg.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
for required in ['extract_deepgram_transcript', 'on_message', 'on_close', 'on_error']:
    assert required in dg_inner, f'{required} not nested in Deepgram live function'
print('Nested helper verification OK')
PY

  # App import sanity (ensures additive imports do not break app module)
  python - << 'PY'
import app
print('app import OK')
PY

  # Optional runtime check (report-only): start server and inspect logs
  # bash ./start.sh & sleep 4; tail -n 100 .logs/server-runtime-errors.log; tail -n 100 .logs/server-output.log

STEP 6: Update PROGRESS.md

Edit PROGRESS.md:

Change Phase 6 section to:
---UPDATE---
## Phase 6: Extract Transcription Services ✅ COMPLETE

Completed: [CURRENT_TIMESTAMP]

Risk Level: HIGH (Async, nested helpers, Deepgram integration)

Files Created:
  - services/transcription.py ([X] lines)

Functions Extracted:
  - transcribe_with_incremental_output (line 1626) + 4 nested helpers [ATOMIC]
    - run_fw_transcription (line 1671)
    - run_transcription (line 1728)
    - transcribe_fw_chunk (line 1831)
    - transcribe_openai_chunk (line 1866)
  - transcribe_chunk (line 1977)
  - transcribe_audio_stream (line 2335)
  - transcribe_vod_with_deepgram (line 2502) + nested helper [ATOMIC]
    - read_audio_file (line 2610)
  - transcribe_with_deepgram (line 2781) + 4 nested helpers [ATOMIC]
    - extract_deepgram_transcript (line 2794)
    - on_message (line 2845)
    - on_close (line 2874)
    - on_error (line 2887)

Critical Patterns Preserved:
  - Nested functions kept with parents (closures intact)
  - All async keywords preserved
  - ThreadPoolExecutor usage intact
  - Deepgram SDK integration preserved
  - WebSocket handlers intact
  - Error handling preserved

Imports Added to app.py:
  - from services.transcription import [5 functions]

Commits: 1
Commit Hash: [git log -1 --format=%H]

Notes:
  - 9 nested helpers kept with parent functions
  - All async patterns verified intact
  - Deepgram integration preserved
  - ThreadPoolExecutor usage maintained
  - Imports added to app.py (additive only)
  - Original app.py code still intact
  - Transcription module verified

Next Phase: Phase 7 (Extract Diarization)
---END UPDATE---

Update Phase Status Overview:
  Phase 6: Transcription ✅ COMPLETE
  Phase 7: Diarization ⏳ NOW READY

STEP 7: Commit Progress Update
  git add PROGRESS.md
  git commit -m "PHASE 6: Update progress tracker - marked complete"

STEP 8: Report Completion

Provide this report:

PHASE 6 COMPLETE

Risk Level: HIGH (Complex async with nested helpers)

Files Created:
  - services/transcription.py ([X] lines)

Functions Extracted: 5 functions + 9 nested helpers

Critical Patterns: Nested functions, async, ThreadPoolExecutor, Deepgram all preserved

Verification: Import test and compile check passed

Commits: 2

Next Phase: Phase 7 (Extract Diarization)

Ready for next instruction.


================================================================================
PHASE 7: EXTRACT DIARIZATION
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Read from Old_Files/app.py.original for all code extraction
- Never modify main branch
- No Docker builds until Phase 13 complete
- Batch operations per phase
- Use line numbers from APP_FUNCTION_INVENTORY.md

PHASE 7: Extract Diarization (Complete this entire phase in one response)

STEP 1: Read PROGRESS.md
  cat PROGRESS.md | grep "Phase 7" -A 10

Confirm Phase 7 status is: NOT STARTED or NOW READY
Confirm Phase 6 status is: COMPLETE

STEP 2: Create services/diarization.py

Extract diarization function from Old_Files/app.py.original:
  - transcribe_with_diarization (line 591)

Create file: services/diarization.py

STEP 2.1: Enumerate diarization function and nested helpers (strict)
  # Use Python AST to list target function, nested helpers, and async flag
  python - << 'PY'
import ast, json, pathlib, re
p = pathlib.Path('Old_Files/app.py.original')
src = p.read_text(encoding='utf-8')
tree = ast.parse(src)

target = 'transcribe_with_diarization'
items = []
nested = {}
async_map = {}

def collect_nested (body):
    return [
        {'name': n.name, 'kind': 'async' if isinstance(n, ast.AsyncFunctionDef) else 'def', 'lineno': n.lineno}
        for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target:
        items.append({'name': node.name, 'kind': 'async' if isinstance(node, ast.AsyncFunctionDef) else 'def', 'lineno': node.lineno})
        nested[node.name] = collect_nested(node.body)
        async_map[node.name] = isinstance(node, ast.AsyncFunctionDef)

outdir = pathlib.Path('.backups')
outdir.mkdir(exist_ok=True)
pathlib.Path(outdir, 'diarization_phase7_funcs.txt').write_text('\n'.join([f"{i['kind']} {i['name']} @ {i['lineno']}" for i in items]), encoding='utf-8')
pathlib.Path(outdir, 'diarization_phase7_nested.json').write_text(json.dumps(nested, indent=2), encoding='utf-8')
pathlib.Path(outdir, 'diarization_phase7_async.json').write_text(json.dumps(async_map, indent=2), encoding='utf-8')

# Report-only: enumerate integrations and helpers usage hints
hits = []
patterns = [r"WebSocket", r"safe_ws_send", r"diarization", r"get_diarization_pipeline"]
for pat in patterns:
    for m in re.finditer(pat, src):
        line_no = src.count('\n', 0, m.start()) + 1
        hits.append({'pattern': pat, 'lineno': line_no})
pathlib.Path(outdir, 'diarization_phase7_integrations.json').write_text(json.dumps(hits[:300], indent=2), encoding='utf-8')
print('Phase 7 enumeration complete')
PY

STEP 2.2: Strict extraction guardrails (Diarization)
  - Keep any nested helpers INSIDE parent function (closures must remain intact)
  - Preserve `async` keyword and exact function signature/param order
  - Preserve diarization pipeline usage and acquisition (`get_diarization_pipeline`)
  - Maintain speaker assignment logic and label generation (Hebrew/English)
  - Keep `safe_ws_send` usage and WebSocket communication semantics
  - Do not alter error handling, logging messages, or return types
  - Maintain constants and configuration imports where applicable
  - Preserve model access via `models.loader.get_diarization_pipeline` and state via `core.state`

STEP 2.3: Cross-check dependent import coverage (Diarization)
  - Confirm import added to `app.py` for diarization export
  - Scan dependent modules and note references (report-only):
    - `api/websocket.py`: any diarization triggers or flow references
    - `api/routes.py`: endpoints calling diarization
    - `utils/websocket_helpers.py`: ensure `safe_ws_send` exists and matches usage
    - `models/loader.py`: `get_diarization_pipeline` interactions
  - Record results to `.backups/diarization_phase7_dependents.txt` for traceability

  python - << 'PY'
import pathlib, re, json
targets = [
    'transcribe_with_diarization',
    'safe_ws_send',
    'get_diarization_pipeline',
]
files = [
    pathlib.Path('api/websocket.py'),
    pathlib.Path('api/routes.py'),
    pathlib.Path('utils/websocket_helpers.py'),
    pathlib.Path('models/loader.py'),
]
results = {}
for f in files:
    if not f.exists():
        results[str(f)] = {'exists': False, 'hits': []}
        continue
    text = f.read_text(encoding='utf-8')
    hits = []
    for t in targets:
        for m in re.finditer(rf"\b{re.escape(t)}\b", text):
            line_no = text.count('\n', 0, m.start()) + 1
            hits.append({'symbol': t, 'lineno': line_no})
    results[str(f)] = {'exists': True, 'hits': hits}
outdir = pathlib.Path('.backups')
outdir.mkdir(exist_ok=True)
pathlib.Path(outdir, 'diarization_phase7_dependents.txt').write_text(json.dumps(results, indent=2), encoding='utf-8')
print('Dependent coverage scan complete')
PY

Content structure:
---START FILE---
"""Speaker diarization services."""
import asyncio
import logging
from typing import Optional, Dict, Any
from fastapi import WebSocket

from models.loader import load_model, get_diarization_pipeline
from core.state import diarization_pipeline
from utils.websocket_helpers import safe_ws_send

# Copy transcribe_with_diarization from line 591 EXACTLY
# Preserve async keyword
# Preserve diarization pipeline usage
# Preserve speaker assignment logic
# May contain inline helper functions for speaker assignment
async def transcribe_with_diarization(model, model_config, audio_file, language, websocket, model_name):
    # [Copy EXACT function body from line 591]
    # Preserve all error handling
    # Preserve WebSocket updates
    # Preserve speaker label generation (Hebrew/English)
    pass
---END FILE---

STEP 3: Add Imports to app.py

In working app.py, add this import:

---ADD TO app.py---
from services.diarization import transcribe_with_diarization
---END ADDITIONS---

NOTE: Do NOT remove function definition from app.py yet. Only ADD import.

STEP 4: Git Commit
  git add services/diarization.py app.py
  git commit -m "PHASE 7: Extract services/diarization.py, add import to app.py"

STEP 5: Verify
  python -c "from services.diarization import transcribe_with_diarization; print('Diarization OK')"
  python -m py_compile services/diarization.py
  wc -l services/diarization.py

STEP 5.1: Enhanced Verification (Diarization)
  # Baseline import test and async indicator
  python - << 'PY'
import inspect
from services.diarization import transcribe_with_diarization
assert inspect.iscoroutinefunction(transcribe_with_diarization)
print('Baseline import test OK')
PY

  # Export listing (report-only)
  python - << 'PY'
import importlib
m = importlib.import_module('services.diarization')
exports = sorted([n for n in dir(m) if not n.startswith('_')])
print('\n'.join(exports))
PY

  # Verify nested helpers preserved in extracted file (AST scan - report-only if none)
  python - << 'PY'
import ast, pathlib
fp = pathlib.Path('services/diarization.py')
src = fp.read_text(encoding='utf-8')
tree = ast.parse(src)
def find_func (name):
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None

fn = find_func('transcribe_with_diarization')
assert fn is not None, 'transcribe_with_diarization missing'
inner = [n.name for n in fn.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
print('Nested helpers:', inner)
print('Nested helper verification OK')
PY

  # App import sanity (ensures additive imports do not break app module)
  python - << 'PY'
import app
print('app import OK')
PY

  # Optional runtime check (report-only): start server and inspect logs
  # bash ./start.sh & sleep 4; tail -n 100 .logs/server-runtime-errors.log; tail -n 100 .logs/server-output.log

STEP 6: Update PROGRESS.md

Edit PROGRESS.md:

Change Phase 7 section to:
---UPDATE---
## Phase 7: Extract Diarization ✅ COMPLETE

Completed: [CURRENT_TIMESTAMP]

Files Created:
  - services/diarization.py ([X] lines)

Functions Extracted:
  - transcribe_with_diarization (line 591)

Critical Patterns Preserved:
  - Async keyword preserved
  - Diarization pipeline usage intact
  - Speaker assignment logic maintained
  - Hebrew/English speaker labels preserved
  - WebSocket communication intact

Imports Added to app.py:
  - from services.diarization import transcribe_with_diarization

Commits: 1
Commit Hash: [git log -1 --format=%H]

Notes:
  - Diarization function extracted
  - Speaker detection logic preserved
  - Imports added to app.py (additive only)
  - Original app.py code still intact
  - Diarization module verified

Next Phase: Phase 8 (Extract Video Metadata)
---END UPDATE---

Update Phase Status Overview:
  Phase 7: Diarization ✅ COMPLETE
  Phase 8: Video Metadata ⏳ NOW READY

STEP 7: Commit Progress Update
  git add PROGRESS.md
  git commit -m "PHASE 7: Update progress tracker - marked complete"

STEP 8: Report Completion

Provide this report:

PHASE 7 COMPLETE

Files Created:
  - services/diarization.py ([X] lines)

Functions Extracted: 1 function

Verification: Import test and compile check passed

Commits: 2

Next Phase: Phase 8 (Extract Video Metadata)

Ready for next instruction.


================================================================================
PHASE 8: EXTRACT VIDEO METADATA
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Read from Old_Files/app.py.original for all code extraction
- Never modify main branch
- No Docker builds until Phase 13 complete
- Batch operations per phase
- Use line numbers from APP_FUNCTION_INVENTORY.md

PHASE 8: Extract Video Metadata (Complete this entire phase in one response)

STEP 1: Read PROGRESS.md
  cat PROGRESS.md | grep "Phase 8" -A 10

Confirm Phase 8 status is: NOT STARTED or NOW READY
Confirm Phase 7 status is: COMPLETE

STEP 2: Create services/video_metadata.py

Extract video metadata function from Old_Files/app.py.original:
  - get_youtube_metadata (line 2144)

Create file: services/video_metadata.py

STEP 2.1: Enumerate video metadata function and helpers (strict)
  # Use Python AST to list target function and async flag; scan integrations
  python - << 'PY'
import ast, json, pathlib, re
p = pathlib.Path('Old_Files/app.py.original')
src = p.read_text(encoding='utf-8')
tree = ast.parse(src)

target = 'get_youtube_metadata'
items = []
async_map = {}

for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target:
        items.append({'name': node.name, 'kind': 'async' if isinstance(node, ast.AsyncFunctionDef) else 'def', 'lineno': node.lineno})
        async_map[node.name] = isinstance(node, ast.AsyncFunctionDef)

outdir = pathlib.Path('.backups')
outdir.mkdir(exist_ok=True)
pathlib.Path(outdir, 'video_metadata_phase8_funcs.txt').write_text('\n'.join([f"{i['kind']} {i['name']} @ {i['lineno']}" for i in items]), encoding='utf-8')
pathlib.Path(outdir, 'video_metadata_phase8_async.json').write_text(json.dumps(async_map, indent=2), encoding='utf-8')

# Report-only: enumerate integrations and usage hints
hits = []
patterns = [r"yt_dlp", r"YoutubeDL", r"is_youtube_url", r"logging", r"async"]
for pat in patterns:
    for m in re.finditer(pat, src):
        line_no = src.count('\n', 0, m.start()) + 1
        hits.append({'pattern': pat, 'lineno': line_no})
pathlib.Path(outdir, 'video_metadata_phase8_integrations.json').write_text(json.dumps(hits[:300], indent=2), encoding='utf-8')
print('Phase 8 enumeration complete')
PY

STEP 2.2: Strict extraction guardrails (Video Metadata)
  - Preserve `async` keyword and exact function signature/param order
  - Keep `yt_dlp.YoutubeDL` configuration and option dict exactly
  - Maintain URL validation flow if present (`is_youtube_url`)
  - Do not alter error handling, logging messages, or return types
  - Preserve constants and configuration imports where applicable
  - Maintain any caching or memoization patterns used by metadata retrieval

STEP 2.3: Cross-check dependent import coverage (Video Metadata)
  - Confirm import added to `app.py` for video metadata export
  - Scan dependent modules and note references (report-only):
    - `api/routes.py`: endpoints calling `get_youtube_metadata`
    - `utils/validators.py`: ensure `is_youtube_url` exists and matches usage
    - `utils/cache.py`: any caching helpers used by metadata flow
  - Record results to `.backups/video_metadata_phase8_dependents.txt` for traceability

  python - << 'PY'
import pathlib, re, json
targets = [
    'get_youtube_metadata',
    'is_youtube_url',
]
files = [
    pathlib.Path('api/routes.py'),
    pathlib.Path('utils/validators.py'),
    pathlib.Path('utils/cache.py'),
]
results = {}
for f in files:
    if not f.exists():
        results[str(f)] = {'exists': False, 'hits': []}
        continue
    text = f.read_text(encoding='utf-8')
    hits = []
    for t in targets:
        for m in re.finditer(rf"\b{re.escape(t)}\b", text):
            line_no = text.count('\n', 0, m.start()) + 1
            hits.append({'symbol': t, 'lineno': line_no})
    results[str(f)] = {'exists': True, 'hits': hits}
outdir = pathlib.Path('.backups')
outdir.mkdir(exist_ok=True)
pathlib.Path(outdir, 'video_metadata_phase8_dependents.txt').write_text(json.dumps(results, indent=2), encoding='utf-8')
print('Dependent coverage scan complete')
PY

Content structure:
---START FILE---
"""YouTube video metadata extraction services."""
import logging
from typing import Optional, Dict, Any
import yt_dlp

# Copy get_youtube_metadata from line 2144 EXACTLY
# Preserve async keyword
# Preserve yt_dlp configuration
# Preserve error handling
async def get_youtube_metadata(url: str) -> Dict:
    # [Copy EXACT function body from line 2144]
    # Preserve yt_dlp.YoutubeDL configuration
    # Preserve metadata extraction logic
    # Preserve error handling
    pass
---END FILE---

STEP 3: Add Imports to app.py

In working app.py, add this import:

---ADD TO app.py---
from services.video_metadata import get_youtube_metadata
---END ADDITIONS---

NOTE: Do NOT remove function definition from app.py yet. Only ADD import.

STEP 4: Git Commit
  git add services/video_metadata.py app.py
  git commit -m "PHASE 8: Extract services/video_metadata.py, add import to app.py"

STEP 5: Verify
  python -c "from services.video_metadata import get_youtube_metadata; print('Video metadata OK')"
  python -m py_compile services/video_metadata.py
  wc -l services/video_metadata.py

STEP 5.1: Enhanced Verification (Video Metadata)
  # Baseline import test and async indicator
  python - << 'PY'
import inspect
from services.video_metadata import get_youtube_metadata
assert inspect.iscoroutinefunction(get_youtube_metadata)
print('Baseline import test OK')
PY

  # Export listing (report-only)
  python - << 'PY'
import importlib
m = importlib.import_module('services.video_metadata')
exports = sorted([n for n in dir(m) if not n.startswith('_')])
print('\n'.join(exports))
PY

  # Verify yt_dlp configuration usage (AST scan - report-only)
  python - << 'PY'
import ast, pathlib
fp = pathlib.Path('services/video_metadata.py')
src = fp.read_text(encoding='utf-8')
tree = ast.parse(src)
def find_func (name):
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None

fn = find_func('get_youtube_metadata')
assert fn is not None, 'get_youtube_metadata missing'
names = set()
class NameVisitor(ast.NodeVisitor):
    def visit_Name (self, node):
        names.add(node.id)
NameVisitor().visit(fn)
print('Identifiers in function:', sorted(names))
print('yt_dlp presence:', 'yt_dlp' in names)
print('Enhanced verification OK')
PY

  # App import sanity (ensures additive imports do not break app module)
  python - << 'PY'
import app
print('app import OK')
PY

STEP 6: Update PROGRESS.md

Edit PROGRESS.md:

Change Phase 8 section to:
---UPDATE---
## Phase 8: Extract Video Metadata ✅ COMPLETE

Completed: [CURRENT_TIMESTAMP]

Files Created:
  - services/video_metadata.py ([X] lines)

Functions Extracted:
  - get_youtube_metadata (line 2144)

Critical Patterns Preserved:
  - Async keyword preserved
  - yt_dlp configuration intact
  - Metadata extraction logic maintained
  - Error handling preserved

Imports Added to app.py:
  - from services.video_metadata import get_youtube_metadata

Commits: 1
Commit Hash: [git log -1 --format=%H]

Notes:
  - YouTube metadata function extracted
  - yt_dlp integration preserved
  - Imports added to app.py (additive only)
  - Original app.py code still intact
  - Video metadata module verified

Next Phase: Phase 9 (Extract API Routes)
---END UPDATE---

Update Phase Status Overview:
  Phase 8: Video Metadata ✅ COMPLETE
  Phase 9: API Routes ⏳ NOW READY

STEP 7: Commit Progress Update
  git add PROGRESS.md
  git commit -m "PHASE 8: Update progress tracker - marked complete"

STEP 8: Report Completion

Provide this report:

PHASE 8 COMPLETE

Files Created:
  - services/video_metadata.py ([X] lines)

Functions Extracted: 1 function

Verification: Import test and compile check passed

Commits: 2

Next Phase: Phase 9 (Extract API Routes)

Ready for next instruction.


================================================================================
PHASE 9: EXTRACT API ROUTES
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Read from Old_Files/app.py.original for all code extraction
- Never modify main branch
- No Docker builds until Phase 13 complete
- Batch operations per phase
- Use line numbers from APP_FUNCTION_INVENTORY.md

PHASE 9: Extract API Routes (Complete this entire phase in one response)

STEP 1: Read PROGRESS.md
  cat PROGRESS.md | grep "Phase 9" -A 10

Confirm Phase 9 status is: NOT STARTED or NOW READY
Confirm Phase 8 status is: COMPLETE

STEP 2: Create api/routes.py

Extract HTTP endpoint functions from Old_Files/app.py.original:
  - get_home (line 3068) [GET /]
  - health_check (line 3419) [GET /health]
  - get_video_info (line 3429) [POST /api/video-info]
  - gpu_diagnostics (line 3485) [GET /gpu]
  - cache_stats (line 3518) [GET /api/cache/stats]
  - clear_cache (line 3538) [POST /api/cache/clear]
  - download_cache_stats (line 3555) [GET /api/download-cache/stats]
  - clear_download_cache (line 3584) [POST /api/download-cache/clear]

Also extract Pydantic models if present:
  - VideoInfoRequest
  - Any other request/response models

Create file: api/routes.py

STEP 2.1: Enumerate API route endpoints (strict)
  # Use Python AST to list endpoint functions, async flag, and decorators
  python - << 'PY'
import ast, json, pathlib, re
p = pathlib.Path('Old_Files/app.py.original')
src = p.read_text(encoding='utf-8')
tree = ast.parse(src)

targets = [
    ('get_home', '/'),
    ('health_check', '/health'),
    ('get_video_info', '/api/video-info'),
    ('gpu_diagnostics', '/gpu'),
    ('cache_stats', '/api/cache/stats'),
    ('clear_cache', '/api/cache/clear'),
    ('download_cache_stats', '/api/download-cache/stats'),
    ('clear_download_cache', '/api/download-cache/clear'),
]

items = []
decorators = {}
async_map = {}

def extract_decorator_info (fn):
    info = []
    for d in getattr(fn, 'decorator_list', []):
        if isinstance(d, ast.Call):
            callee = d.func
            if isinstance(callee, ast.Attribute):
                name = f"{callee.value.id}.{callee.attr}" if isinstance(callee.value, ast.Name) else callee.attr
            elif isinstance(callee, ast.Name):
                name = callee.id
            else:
                name = 'unknown'
            path = None
            if d.args and isinstance(d.args[0], ast.Constant) and isinstance(d.args[0].value, str):
                path = d.args[0].value
            info.append({'decorator': name, 'path': path})
        elif isinstance(d, ast.Attribute):
            name = f"{d.value.id}.{d.attr}" if isinstance(d.value, ast.Name) else d.attr
            info.append({'decorator': name, 'path': None})
        elif isinstance(d, ast.Name):
            info.append({'decorator': d.id, 'path': None})
    return info

for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        name = node.name
        if any(t[0] == name for t in targets):
            items.append({'name': name, 'kind': 'async' if isinstance(node, ast.AsyncFunctionDef) else 'def', 'lineno': node.lineno})
            async_map[name] = isinstance(node, ast.AsyncFunctionDef)
            decorators[name] = extract_decorator_info(node)

outdir = pathlib.Path('.backups')
outdir.mkdir(exist_ok=True)
pathlib.Path(outdir, 'api_routes_phase9_funcs.txt').write_text('\n'.join([f"{i['kind']} {i['name']} @ {i['lineno']}" for i in items]), encoding='utf-8')
pathlib.Path(outdir, 'api_routes_phase9_async.json').write_text(json.dumps(async_map, indent=2), encoding='utf-8')
pathlib.Path(outdir, 'api_routes_phase9_decorators.json').write_text(json.dumps(decorators, indent=2), encoding='utf-8')

# Report-only: enumerate integrations and usage hints
hits = []
patterns = [r"cached_index_html", r"JSONResponse", r"HTMLResponse", r"is_youtube_url", r"get_youtube_metadata", r"utils.cache"]
for pat in patterns:
    for m in re.finditer(pat, src):
        line_no = src.count('\n', 0, m.start()) + 1
        hits.append({'pattern': pat, 'lineno': line_no})
pathlib.Path(outdir, 'api_routes_phase9_integrations.json').write_text(json.dumps(hits[:300], indent=2), encoding='utf-8')
print('Phase 9 enumeration complete')
PY

STEP 2.2: Strict extraction guardrails (API Routes)
  - Convert decorators from `@app.get/post/...` to `@router.get/post/...` without losing args
  - Preserve `response_class=HTMLResponse/JSONResponse` and any `status_code`, `tags`, etc.
  - Keep Pydantic models unchanged and colocated in the module if present
  - Maintain imports: `get_youtube_metadata`, `is_youtube_url`, cache helpers, `cached_index_html`, settings
  - Do not alter logging messages, error handling, or return types
  - Preserve constants and configuration imports where applicable
  - Only ADD router inclusion to `app.py`; do not remove original endpoints yet

STEP 2.3: Cross-check dependent import coverage (API Routes)
  - Confirm router inclusion added to `app.py`
  - Scan dependent modules and note references (report-only):
    - `core/state.py`: ensure `cached_index_html` exists
    - `utils/validators.py`: ensure `is_youtube_url` exists and matches usage
    - `services/video_metadata.py`: ensure `get_youtube_metadata` exists
    - `utils/cache.py`: check cache helper presence
  - Record results to `.backups/api_routes_phase9_dependents.txt` for traceability

  python - << 'PY'
import pathlib, re, json
targets = [
    'cached_index_html',
    'is_youtube_url',
    'get_youtube_metadata',
]
files = [
    pathlib.Path('core/state.py'),
    pathlib.Path('utils/validators.py'),
    pathlib.Path('services/video_metadata.py'),
    pathlib.Path('utils/cache.py'),
]
results = {}
for f in files:
    if not f.exists():
        results[str(f)] = {'exists': False, 'hits': []}
        continue
    text = f.read_text(encoding='utf-8')
    hits = []
    for t in targets:
        for m in re.finditer(rf"\b{re.escape(t)}\b", text):
            line_no = text.count('\n', 0, m.start()) + 1
            hits.append({'symbol': t, 'lineno': line_no})
    results[str(f)] = {'exists': True, 'hits': hits}
outdir = pathlib.Path('.backups')
outdir.mkdir(exist_ok=True)
pathlib.Path(outdir, 'api_routes_phase9_dependents.txt').write_text(json.dumps(results, indent=2), encoding='utf-8')
print('Dependent coverage scan complete')
PY

Content structure:
---START FILE---
"""HTTP API route handlers."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import logging
from typing import Optional

from services.video_metadata import get_youtube_metadata
from utils.cache import (
    # [import cache functions needed for endpoints]
)
from utils.validators import is_youtube_url
from core.state import cached_index_html
from config.settings import (
    # [import relevant settings]
)

router = APIRouter()

# Extract Pydantic models if present in Old_Files/app.py.original
class VideoInfoRequest(BaseModel):
    # [Copy exact model definition if present]
    pass

# Copy get_home from line 3068 EXACTLY
# CHANGE: @app.get("/") to @router.get("/")
@router.get("/", response_class=HTMLResponse)
async def get_home():
    # [Copy EXACT function body from line 3068]
    # Preserve cached_index_html usage
    pass

# Copy health_check from line 3419 EXACTLY
# CHANGE: @app.get("/health") to @router.get("/health")
@router.get("/health")
async def health_check():
    # [Copy EXACT function body]
    pass

# Copy get_video_info from line 3429 EXACTLY
# CHANGE: @app.post("/api/video-info") to @router.post("/api/video-info")
@router.post("/api/video-info")
async def get_video_info(request: VideoInfoRequest):
    # [Copy EXACT function body]
    pass

# Copy gpu_diagnostics from line 3485 EXACTLY
# CHANGE: @app.get("/gpu") to @router.get("/gpu")
@router.get("/gpu")
async def gpu_diagnostics():
    # [Copy EXACT function body]
    pass

# Copy cache_stats from line 3518 EXACTLY
# CHANGE: @app.get("/api/cache/stats") to @router.get("/api/cache/stats")
@router.get("/api/cache/stats")
async def cache_stats():
    # [Copy EXACT function body]
    pass

# Copy clear_cache from line 3538 EXACTLY
# CHANGE: @app.post("/api/cache/clear") to @router.post("/api/cache/clear")
@router.post("/api/cache/clear")
async def clear_cache():
    # [Copy EXACT function body]
    pass

# Copy download_cache_stats from line 3555 EXACTLY
# CHANGE: @app.get("/api/download-cache/stats") to @router.get("/api/download-cache/stats")
@router.get("/api/download-cache/stats")
async def download_cache_stats():
    # [Copy EXACT function body]
    pass

# Copy clear_download_cache from line 3584 EXACTLY
# CHANGE: @app.post("/api/download-cache/clear") to @router.post("/api/download-cache/clear")
@router.post("/api/download-cache/clear")
async def clear_download_cache():
    # [Copy EXACT function body]
pass
---END FILE---

STEP 2.1: Enumerate WebSocket endpoint and nested helper (strict)
  mkdir -p .backups
  python - << 'PY'
  import ast, json, re, sys

  src_path = 'Old_Files/app.py.original'
  try:
  	src = open(src_path, 'r', encoding='utf-8').read()
  except FileNotFoundError:
  	print(f'ERROR: missing {src_path}', file=sys.stderr)
  	sys.exit(1)

  tree = ast.parse(src)

  TARGET_FUNC = 'websocket_transcribe'
  EXPECTED_NESTED = {'read_capture_file'}

  func_info = {}
  decorators_info = {}
  nested_info = {'parent': TARGET_FUNC, 'nested': []}
  integrations = {
  	'found': set(),
  	'expected': {
  		'safe_ws_send', 'WebSocketState', 'WebSocketDisconnect', 'CAPTURES',
  		'download_with_fallback', 'get_audio_duration_seconds', 'load_model',
  		'transcribe_with_incremental_output', 'transcribe_audio_stream',
  		'transcribe_with_deepgram', 'transcribe_with_diarization',
  		'is_youtube_url', 'logging', 'asyncio',
  	},
  }

  def walk_names(node):
  	for n in ast.walk(node):
  		if isinstance(n, ast.Name):
  			yield n.id
  		elif isinstance(n, ast.Attribute):
  			# capture attribute tail like WebSocketState.OPEN
  			yield n.attr

  def find_nested_functions(node):
  	results = []
  	for n in ast.walk(node):
  		if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
  			# exclude the parent itself
  			if n is not node and n.name:
  				results.append({'name': n.name, 'lineno': n.lineno, 'end_lineno': getattr(n, 'end_lineno', None), 'async': isinstance(n, ast.AsyncFunctionDef)})
  	return results

  for n in ast.walk(tree):
  	if isinstance(n, ast.AsyncFunctionDef) and n.name == TARGET_FUNC:
  		func_info = {
  			'name': n.name,
  			'async': True,
  			'lineno': n.lineno,
  			'end_lineno': getattr(n, 'end_lineno', None),
  		}

  		# decorators
  		decs = []
  		for d in n.decorator_list:
  			kind = None
  			path = None
  			if isinstance(d, ast.Call):
  				if isinstance(d.func, ast.Attribute) and isinstance(d.func.value, ast.Name):
  					kind = f"{d.func.value.id}.{d.func.attr}"
  					# first positional arg if constant string
  					if d.args:
  						arg0 = d.args[0]
  						if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
  							path = arg0.value
  			elif isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name):
  				kind = f"{d.value.id}.{d.attr}"
  			decs.append({'kind': kind, 'path': path})
  		decorators_info = {'function': TARGET_FUNC, 'decorators': decs}

  		# nested functions
  		nested = find_nested_functions(n)
  		nested_info['nested'] = nested

  		# integrations found
  		integrations['found'] = set(walk_names(n))
  		break

  # write backups
  with open('.backups/websocket_phase10_funcs.txt', 'w', encoding='utf-8') as f:
  	f.write(f"{func_info.get('name')} [async={func_info.get('async')}] "
  			f"lines: {func_info.get('lineno')}..{func_info.get('end_lineno')}\n")
  	for nf in nested_info['nested']:
  		f.write(f"  nested: {nf['name']} [async={nf['async']}] "
  			f"lines: {nf['lineno']}..{nf['end_lineno']}\n")

  with open('.backups/websocket_phase10_async.json', 'w', encoding='utf-8') as f:
  	json.dump({'websocket_transcribe': bool(func_info.get('async'))}, f, indent=2)

  with open('.backups/websocket_phase10_decorators.json', 'w', encoding='utf-8') as f:
  	json.dump(decorators_info, f, indent=2)

  with open('.backups/websocket_phase10_nested.json', 'w', encoding='utf-8') as f:
  	json.dump(nested_info, f, indent=2)

  # convert set to sorted list for JSON
  integrations['found'] = sorted(list(integrations['found']))
  with open('.backups/websocket_phase10_integrations.json', 'w', encoding='utf-8') as f:
  	json.dump(integrations, f, indent=2)

  # lightweight text scan for critical WS patterns
  ws_hits = []
  for pat in (r'WebSocketDisconnect', r'WebSocketState', r'safe_ws_send', r'@app\.websocket\('):
  	for m in re.finditer(pat, src):
  		ws_hits.append({'pattern': pat, 'pos': m.start()})
  with open('.backups/websocket_phase10_ws_patterns.txt', 'w', encoding='utf-8') as f:
  	for h in ws_hits:
  		f.write(f"{h['pattern']} @ {h['pos']}\n")
  PY
  
  # Quick view
  cat .backups/websocket_phase10_funcs.txt | sed -n '1,8p'
  cat .backups/websocket_phase10_decorators.json | head -60
  cat .backups/websocket_phase10_nested.json | head -40
  cat .backups/websocket_phase10_integrations.json | head -60

STEP 2.2: Strict guardrails (WebSocket orchestration + closure helpers)
  Guardrails to preserve during extraction:
  - Keep nested helper inside parent: do NOT lift out of closure
  - Preserve 'async def websocket_transcribe' signature and await semantics
  - Convert '@app.websocket("/ws/transcribe")' to '@router.websocket("/ws/transcribe")' ONLY
  - Preserve WebSocket handshake, state checks, and disconnect handling:
    - 'await websocket.accept()' if present
    - check 'websocket.client_state' against 'WebSocketState.OPEN' before sends
    - catch 'WebSocketDisconnect' and perform cleanup
  - Maintain use of 'safe_ws_send' for resilient message sends
  - Preserve complete orchestration flow:
    - audio download ('download_with_fallback'), duration, and validation
    - model loading ('load_model') + any locks/coordination already extracted
    - transcription paths (incremental, streaming, Deepgram)
    - diarization integration ('transcribe_with_diarization') when enabled
    - progress updates and payload schema in outgoing frames
    - error handling, logging, and resource cleanup
  - Keep references to 'CAPTURES', cache utilities, validators, and settings usage
  - Do NOT remove original endpoint from app.py yet; router inclusion must be additive
  - In app.py, only add:
    - 'from api.websocket import router as ws_router'
    - 'app.include_router(ws_router)'
  - Avoid modifying unrelated modules or changing message formats

STEP 2.3: Dependent coverage cross-check (imports and integrations)
  # Verify imported dependencies resolve and are present
  python - << 'PY'
  import importlib, sys
  mods = [
  	('services.audio_processor', ['download_with_fallback','get_audio_duration_seconds']),
  	('services.transcription', ['transcribe_with_incremental_output','transcribe_audio_stream','transcribe_with_deepgram']),
  	('services.diarization', ['transcribe_with_diarization']),
  	('models.loader', ['load_model']),
  	('utils.cache', []),
  	('utils.validators', ['is_youtube_url']),
  	('utils.websocket_helpers', ['safe_ws_send']),
  	('core.state', ['CAPTURES']),
  	('config.settings', []),
  ]
  ok = True
  for mod, names in mods:
  	try:
  		m = importlib.import_module(mod)
  		for n in names:
  			getattr(m, n)
  	except Exception as e:
  		ok = False
  		print(f'WARN: {mod} import check failed: {e}', file=sys.stderr)
  print('DEPENDENT IMPORT COVERAGE:', 'OK' if ok else 'ISSUES')
  PY
  
  # Scan working app.py for router inclusion (additive only)
  grep -n "from api.websocket import router as ws_router" app.py || true
  grep -n "app.include_router(ws_router)" app.py || true

STEP 3: Add Router to app.py

In working app.py, add this import and router inclusion:

---ADD TO app.py---
from api.routes import router as http_router

# After app = FastAPI(...), add:
app.include_router(http_router)
---END ADDITIONS---

NOTE: Do NOT remove endpoint definitions from app.py yet. Only ADD router.

STEP 4: Git Commit
  git add api/routes.py app.py
  git commit -m "PHASE 9: Extract api/routes.py with HTTP endpoints, include router in app.py"

STEP 5: Verify
  python -c "from api.routes import router; print('API routes OK')"
  python -m py_compile api/routes.py
  wc -l api/routes.py

STEP 5.1: Enhanced Verification (API Routes)
  # Baseline import test and router indicator
  python - << 'PY'
import importlib
m = importlib.import_module('api.routes')
router = getattr(m, 'router', None)
assert router is not None, 'router missing'
print('Baseline import test OK')
PY

  # Export listing (report-only)
  python - << 'PY'
import importlib
m = importlib.import_module('api.routes')
exports = sorted([n for n in dir(m) if not n.startswith('_')])
print('\n'.join(exports))
PY

  # Verify decorator conversions and endpoint async functions (AST/text scan)
  python - << 'PY'
import ast, pathlib, inspect, importlib
fp = pathlib.Path('api/routes.py')
src = fp.read_text(encoding='utf-8')
assert '@app.' not in src, 'Found @app decorators; should be @router'
assert '@router.' in src, 'No @router decorators found'

m = importlib.import_module('api.routes')
names = [
    'get_home',
    'health_check',
    'get_video_info',
    'gpu_diagnostics',
    'cache_stats',
    'clear_cache',
    'download_cache_stats',
    'clear_download_cache',
]
missing = [n for n in names if not hasattr(m, n)]
assert not missing, f'Missing endpoints: {missing}'
async_flags = {n: inspect.iscoroutinefunction(getattr(m, n)) for n in names}
print('Endpoint async flags:', async_flags)
print('Enhanced verification OK')
PY

  # App import sanity (ensures additive imports do not break app module)
  python - << 'PY'
import app
print('app import OK')
PY

STEP 6: Update PROGRESS.md

Edit PROGRESS.md:

Change Phase 9 section to:
---UPDATE---
## Phase 9: Extract API Routes ✅ COMPLETE

Completed: [CURRENT_TIMESTAMP]

Files Created:
  - api/routes.py ([X] lines)

Endpoints Extracted:
  - GET / (get_home, line 3068)
  - GET /health (health_check, line 3419)
  - POST /api/video-info (get_video_info, line 3429)
  - GET /gpu (gpu_diagnostics, line 3485)
  - GET /api/cache/stats (cache_stats, line 3518)
  - POST /api/cache/clear (clear_cache, line 3538)
  - GET /api/download-cache/stats (download_cache_stats, line 3555)
  - POST /api/download-cache/clear (clear_download_cache, line 3584)

Models Extracted:
  - VideoInfoRequest (if present)
  - [list any other models]

Decorator Changes:
  - @app.get(...) converted to @router.get(...)
  - @app.post(...) converted to @router.post(...)

Router Added to app.py:
  - app.include_router(http_router)

Commits: 1
Commit Hash: [git log -1 --format=%H]

Notes:
  - All HTTP endpoints extracted
  - Router pattern implemented
  - Imports added to app.py (additive only)
  - Original app.py code still intact
  - API routes module verified

Next Phase: Phase 10 (Extract WebSocket Endpoint - CRITICAL)
---END UPDATE---

Update Phase Status Overview:
  Phase 9: API Routes ✅ COMPLETE
  Phase 10: WebSocket ⏳ NOW READY

STEP 7: Commit Progress Update
  git add PROGRESS.md
  git commit -m "PHASE 9: Update progress tracker - marked complete"

STEP 8: Report Completion

Provide this report:

PHASE 9 COMPLETE

Files Created:
  - api/routes.py ([X] lines)

Endpoints Extracted: 8 HTTP endpoints

Verification: Import test and compile check passed

Commits: 2

Next Phase: Phase 10 (Extract WebSocket Endpoint - CRITICAL, Core transcription flow)

Ready for next instruction.


================================================================================
PHASE 10: EXTRACT WEBSOCKET ENDPOINT
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Read from Old_Files/app.py.original for all code extraction
- Never modify main branch
- No Docker builds until Phase 13 complete
- Batch operations per phase
- Use line numbers from APP_FUNCTION_INVENTORY.md

WARNING: CRITICAL PHASE - Core transcription workflow, complete WebSocket orchestration

PHASE 10: Extract WebSocket Endpoint (Complete this entire phase in one response)

STEP 1: Read PROGRESS.md
  cat PROGRESS.md | grep "Phase 10" -A 10

Confirm Phase 10 status is: NOT STARTED or NOW READY
Confirm Phase 9 status is: COMPLETE

STEP 2: Create api/websocket.py

Extract WebSocket endpoint from Old_Files/app.py.original:
  - websocket_transcribe (line 3074) + nested read_capture_file (line 3238) - ATOMIC MOVE

This is the COMPLETE transcription workflow orchestration.

Create file: api/websocket.py

Content structure:
---START FILE---
"""WebSocket endpoint for real-time transcription.

CRITICAL: This is the core transcription workflow orchestration.
Handles complete end-to-end transcription process.
Nested helper MUST stay with parent function (uses parent variables via closure).
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import asyncio
import logging
import json
from typing import Optional

from services.audio_processor import (
    download_with_fallback,
    get_audio_duration_seconds,
)
from services.transcription import (
    transcribe_with_incremental_output,
    transcribe_audio_stream,
    transcribe_with_deepgram,
)
from services.diarization import transcribe_with_diarization
from models.loader import load_model
from utils.cache import (
    # [import cache functions needed]
)
from utils.validators import is_youtube_url
from utils.websocket_helpers import safe_ws_send
from core.state import CAPTURES
from config.settings import (
    # [import relevant settings]
)

router = APIRouter()

# Copy websocket_transcribe from line 3074 EXACTLY
# CRITICAL: Keep nested read_capture_file (line 3238) INSIDE this function
# CHANGE: @app.websocket("/ws/transcribe") to @router.websocket("/ws/transcribe")
@router.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    # [Copy EXACT function body starting from line 3074]
    # Must include complete nested function read_capture_file at line 3238
    # This orchestrates the ENTIRE transcription workflow:
    #   - WebSocket connection handling
    #   - Audio download
    #   - Model loading
    #   - Transcription (with/without diarization)
    #   - Progress updates
    #   - Error handling
    #   - Cleanup
    # Preserve ALL WebSocket state checks
    # Preserve ALL error handling
    # Preserve disconnect handling
    # Preserve complete workflow logic
    pass
---END FILE---

STEP 3: Add Router to app.py

In working app.py, add this import and router inclusion:

---ADD TO app.py---
from api.websocket import router as ws_router

# After app.include_router(http_router), add:
app.include_router(ws_router)
---END ADDITIONS---

NOTE: Do NOT remove WebSocket endpoint from app.py yet. Only ADD router.

STEP 4: Git Commit
  git add api/websocket.py app.py
  git commit -m "PHASE 10: Extract api/websocket.py with core transcription workflow, include router in app.py"

STEP 5: Verify
  python -c "from api.websocket import router; print('WebSocket OK')"
  python -m py_compile api/websocket.py
  wc -l api/websocket.py

STEP 5.1: Enhanced verification (router, async, decorator, nested helper)
  # Assert router exists and endpoint is async with correct decorator path
  python - << 'PY'
  import ast, json, sys
  p = 'api/websocket.py'
  try:
  	src = open(p, 'r', encoding='utf-8').read()
  except FileNotFoundError:
  	print('ERROR: websocket module missing', file=sys.stderr)
  	sys.exit(1)
  t = ast.parse(src)
  router_ok = False
  func_ok = False
  dec_ok = False
  nested_ok = False
  ws_usage = {'WebSocketDisconnect': False, 'WebSocketState': False, 'safe_ws_send': False}

  def walk_names(node):
  	for n in ast.walk(node):
  		if isinstance(n, ast.Name):
  			yield n.id
  		elif isinstance(n, ast.Attribute):
  			yield n.attr

  # router existence
  for n in ast.walk(t):
  	if isinstance(n, ast.Assign):
  		if any(isinstance(tg, ast.Name) and tg.id == 'router' for tg in n.targets):
  			if isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'APIRouter':
  				router_ok = True

  # function and decorator
  for n in ast.walk(t):
  	if isinstance(n, ast.AsyncFunctionDef) and n.name == 'websocket_transcribe':
  		func_ok = True
  		# decorator check
  		for d in n.decorator_list:
  			if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute):
  				if isinstance(d.func.value, ast.Name) and d.func.value.id == 'router' and d.func.attr == 'websocket':
  					if d.args and isinstance(d.args[0], ast.Constant) and d.args[0].value == '/ws/transcribe':
  						dec_ok = True
  		# nested helper present inside parent
  		for sub in ast.walk(n):
  			if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub is not n and sub.name == 'read_capture_file':
  				nested_ok = True
  		# usage checks within endpoint
  		names = set(walk_names(n))
  		for k in ws_usage:
  			ws_usage[k] = ws_usage[k] or (k in names)
  		break

  data = {
  	'router_present': router_ok,
  	'endpoint_async': func_ok,
  	'decorator_router_websocket_path': dec_ok,
  	'nested_read_capture_file_present': nested_ok,
  	'ws_usage': ws_usage,
  }
  print(json.dumps(data, indent=2))
  open('.backups/websocket_phase10_enhanced_verify.json','w').write(json.dumps(data, indent=2))
  PY

  # Export listing and decorator presence
  python - << 'PY'
  import inspect
  from api import websocket as ws
  print('exports:', [n for n in dir(ws) if not n.startswith('_')])
  print('endpoint_is_async:', inspect.iscoroutinefunction(ws.websocket_transcribe))
  PY

  # Text checks for '@router.websocket' and nested helper symbol
  grep -n "@router.websocket(\"/ws/transcribe\")" api/websocket.py || true
  grep -n "def read_capture_file" api/websocket.py || true

  # App import sanity: verify additive router inclusion
  grep -n "from api.websocket import router as ws_router" app.py || true
  grep -n "app.include_router(ws_router)" app.py || true

STEP 6: Update PROGRESS.md

Edit PROGRESS.md:

Change Phase 10 section to:
---UPDATE---
## Phase 10: Extract WebSocket Endpoint ✅ COMPLETE

Completed: [CURRENT_TIMESTAMP]

Risk Level: CRITICAL (Core transcription workflow)

Files Created:
  - api/websocket.py ([X] lines)

Endpoint Extracted:
  - WS /ws/transcribe (websocket_transcribe, line 3074) + nested helper [ATOMIC]
    - read_capture_file (line 3238)

Workflow Components:
  - WebSocket connection management
  - Audio download orchestration
  - Model loading coordination
  - Transcription execution (incremental/streaming/Deepgram)
  - Diarization integration
  - Progress tracking and updates
  - Error handling and recovery
  - Resource cleanup

Critical Patterns Preserved:
  - Nested function kept with parent (closure intact)
  - WebSocket state checks before all sends
  - WebSocketDisconnect handling
  - Complete workflow orchestration
  - All async patterns preserved
  - All error handling intact

Decorator Changes:
  - @app.websocket(...) converted to @router.websocket(...)

Router Added to app.py:
  - app.include_router(ws_router)

Commits: 1
Commit Hash: [git log -1 --format=%H]

Notes:
  - MOST CRITICAL component extracted successfully
  - Complete transcription workflow preserved
  - Nested helper function kept with parent
  - WebSocket patterns verified intact
  - All orchestration logic maintained
  - Imports added to app.py (additive only)
  - Original app.py code still intact
  - WebSocket module verified

CRITICAL CHECKPOINT: Core functionality extracted.
All transcription paths (incremental, streaming, Deepgram, diarization) preserved.

Next Phase: Phase 11 (Extract Lifespan and Finalize app.py)
---END UPDATE---

Update Phase Status Overview:
  Phase 10: WebSocket ✅ COMPLETE
  Phase 11: Lifespan ⏳ NOW READY

STEP 7: Commit Progress Update
  git add PROGRESS.md
  git commit -m "PHASE 10: Update progress tracker - marked complete"

STEP 8: Report Completion

Provide this report:

PHASE 10 COMPLETE

Risk Level: CRITICAL (Core workflow maintained)

Files Created:
  - api/websocket.py ([X] lines)

Endpoint Extracted: 1 WebSocket endpoint (complete transcription orchestration)

Critical Patterns: WebSocket state, nested function, complete workflow all preserved

Verification: Import test and compile check passed

Commits: 2

Next Phase: Phase 11 (Extract Lifespan and Finalize app.py to minimal entry point)

Ready for next instruction.


================================================================================
PHASE 11: EXTRACT LIFESPAN AND FINALIZE APP.PY
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Read from Old_Files/app.py.original for all code extraction
- Never modify main branch
- No Docker builds until Phase 13 complete
- Batch operations per phase
- Use line numbers from APP_FUNCTION_INVENTORY.md

PHASE 11: Extract Lifespan and Finalize app.py (Complete this entire phase in one response)

STEP 1: Read PROGRESS.md
  cat PROGRESS.md | grep "Phase 11" -A 10

Confirm Phase 11 status is: NOT STARTED or NOW READY
Confirm Phase 10 status is: COMPLETE

STEP 2: Create core/lifespan.py

Extract lifespan function from Old_Files/app.py.original:
  - lifespan (line 158)

Create file: core/lifespan.py

Content structure:
---START FILE---
"""Application lifespan management for startup and shutdown tasks."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from config.settings import (
    # [import relevant settings]
)
from core import state
from utils.cache import init_cache_dir, init_download_cache_dir, init_capture_dir
from models.loader import load_model

# Copy lifespan from line 158 EXACTLY
# Preserve @asynccontextmanager decorator
# Preserve startup tasks:
#   - ThreadPoolExecutor initialization
#   - Cache directory initialization
#   - Default model loading
#   - HTML caching
#   - Logging setup
# Preserve shutdown tasks:
#   - Executor cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [Copy EXACT function body from line 158]
    # Startup section:
    #   - Initialize executor in core.state
    #   - Initialize cache directories
    #   - Load default model
    #   - Cache index.html
    #   - Setup logging
    
    yield
    
    # Shutdown section:
    #   - Cleanup executor
    #   - Any other cleanup
    pass
---END FILE---

STEP 3: Create NEW MINIMAL app.py

CRITICAL: Now we replace app.py with minimal entry point.

Create NEW content for app.py:

---REPLACE ENTIRE app.py WITH---
"""Main FastAPI application entry point.

This is the minimal entry point that imports all modular components.
All functionality has been extracted to appropriate modules.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import core functionality
from core.lifespan import lifespan

# Import routers
from api.routes import router as http_router
from api.websocket import router as ws_router

# Create FastAPI application
app = FastAPI(
    title="Live Transcription Service",
    version="2.0",
    lifespan=lifespan
)

# Mount static files for CSS
app.mount("/css", StaticFiles(directory="static/css"), name="css")

# Include HTTP routes
app.include_router(http_router)

# Include WebSocket routes
app.include_router(ws_router)

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8009,
        log_level="info"
    )
---END REPLACEMENT---

STEP 4: Git Commit
  git add core/lifespan.py app.py
  git commit -m "PHASE 11: Extract core/lifespan.py and create minimal app.py entry point"

STEP 5: Verify
  python -c "from core.lifespan import lifespan; print('Lifespan OK')"
  python -m py_compile core/lifespan.py
  wc -l app.py
  wc -l core/lifespan.py

Expected app.py size: ~40-50 lines

STEP 5.1: Enhanced Verification (AST + app.py linkage)
  Verify the extracted lifespan and minimal app.py with a stricter script.

  Run:
    python - << 'PY'
    import ast, sys, json
    from pathlib import Path

    # Verify core/lifespan.py
    lp = Path('core/lifespan.py')
    src = lp.read_text(encoding='utf-8')
    tree = ast.parse(src)

    lifespan = None
    for node in tree.body:
      if isinstance(node, ast.AsyncFunctionDef) and node.name == 'lifespan':
        lifespan = node
        break
    assert lifespan is not None, 'lifespan async function not found'

    decorators = []
    for d in lifespan.decorator_list:
      if isinstance(d, ast.Name): decorators.append(d.id)
      elif isinstance(d, ast.Attribute): decorators.append(d.attr)
      elif isinstance(d, ast.Call):
        f = d.func
        if isinstance(f, ast.Name): decorators.append(f.id)
        elif isinstance(f, ast.Attribute): decorators.append(f.attr)
    assert 'asynccontextmanager' in decorators, '@asynccontextmanager missing on lifespan'

    # ensure yield present
    assert any(isinstance(s, ast.Yield) for s in lifespan.body), 'yield missing in lifespan body'

    # check for orchestration primitives
    text = src
    must_have = [
      'ThreadPoolExecutor',
      'init_cache_dir', 'init_download_cache_dir', 'init_capture_dir',
      'load_model',
      'cached_index_html',
      'logging',
    ]
    for key in must_have:
      assert key in text, f'missing expected pattern in lifespan: {key}'

    # Verify app.py inclusion is additive and wired to lifespan
    ap = Path('app.py').read_text(encoding='utf-8')
    assert 'from core.lifespan import lifespan' in ap, 'app.py missing lifespan import'
    assert 'FastAPI(' in ap and 'lifespan=lifespan' in ap, 'app.py missing FastAPI lifespan linkage'
    assert 'app.include_router(http_router)' in ap, 'http_router inclusion missing'
    assert 'app.include_router(ws_router)' in ap, 'ws_router inclusion missing'
    assert 'StaticFiles' in ap and 'app.mount("/css"' in ap, 'StaticFiles mount missing'
    assert 'uvicorn.run' in ap, 'uvicorn entry missing'

    print('Enhanced Phase 11 verification OK')
    PY

STEP 6: Update PROGRESS.md

Edit PROGRESS.md:

Change Phase 11 section to:
---UPDATE---
## Phase 11: Extract Lifespan and Finalize app.py ✅ COMPLETE

Completed: [CURRENT_TIMESTAMP]

Files Created:
  - core/lifespan.py ([X] lines)

Files Modified:
  - app.py (REPLACED with minimal entry point, ~40-50 lines)

Lifespan Function Extracted:
  - lifespan (line 158)
    - Startup: executor init, cache dirs, model loading, HTML cache
    - Shutdown: executor cleanup

New app.py Structure:
  - Imports from core.lifespan
  - Imports from api.routes, api.websocket
  - FastAPI app creation with lifespan
  - Static file mounting (/css)
  - Router inclusion (http_router, ws_router)
  - Uvicorn entry point

Original app.py: 3,618 lines
New app.py: ~40-50 lines
Reduction: ~98.6%

Critical Patterns Preserved:
  - @asynccontextmanager decorator
  - Startup/shutdown tasks intact
  - ThreadPoolExecutor management
  - Cache initialization
  - Model preloading

Commits: 1
Commit Hash: [git log -1 --format=%H]

Notes:
  - Lifespan management extracted
  - app.py now minimal entry point
  - All functionality imported from modules
  - Application structure complete
  - Verification tests passed

Next Phase: Phase 12 (Extract Static Files)
---END UPDATE---

Update Phase Status Overview:
  Phase 11: Lifespan ✅ COMPLETE
  Phase 12: Static Files ⏳ NOW READY

STEP 7: Commit Progress Update
  git add PROGRESS.md
  git commit -m "PHASE 11: Update progress tracker - marked complete"

STEP 8: Report Completion

Provide this report:

PHASE 11 COMPLETE

Files Created:
  - core/lifespan.py ([X] lines)

Files Replaced:
  - app.py (now ~40-50 lines, was 3,618 lines)

Reduction: ~98.6% of original app.py

Verification: Import test and compile check passed

Commits: 2

Next Phase: Phase 12 (Extract Static Files - HTML and CSS)

Ready for next instruction.


================================================================================
PHASE 12: EXTRACT STATIC FILES
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Read from Old_Files/app.py.original for all code extraction
- Never modify main branch
- No Docker builds until Phase 13 complete
- Batch operations per phase
- Use line numbers from APP_FUNCTION_INVENTORY.md

PHASE 12: Extract Static Files (Complete this entire phase in one response)

STEP 1: Read PROGRESS.md
  cat PROGRESS.md | grep "Phase 12" -A 10

Confirm Phase 12 status is: NOT STARTED or NOW READY
Confirm Phase 11 status is: COMPLETE

STEP 2: Extract HTML from Old_Files/app.py.original

Search Old_Files/app.py.original for HTML content:
  - Look for large string variable containing HTML
  - Usually starts with <!DOCTYPE html> or <html>
  - May be in variable like: html_content = or HTML_TEMPLATE = or similar

Create file: static/index.html

Copy ENTIRE HTML content EXACTLY as found in Old_Files/app.py.original

STEP 2.1: AST/Text Enumeration (HTML source)
  Enumerate the embedded HTML and CSS origins from Old_Files/app.py.original.
  Identify the variable or assignment holding full HTML (e.g., HTML_TEMPLATE,
  html_content) and locate inline <style> blocks. Use Python AST where possible
  and fallback to text search to capture large string assignments.

  Enumeration outputs:
    - .backups/static_phase12_html_vars.txt        # candidate var names, line spans
    - .backups/static_phase12_html_extract.txt     # extracted full HTML text (preview)
    - .backups/static_phase12_style_blocks.txt     # extracted style block contents
    - .backups/static_phase12_summary.json         # summary with counts and anchors

  Run:
    python - << 'PY'
    import ast, re, json
    from pathlib import Path

    p = Path('Old_Files/app.py.original')
    s = p.read_text(encoding='utf-8')
    t = ast.parse(s)

    html_candidates = []
    for node in ast.walk(t):
      # Look for assignments to names with large Constant strings containing <html>
      if isinstance(node, ast.Assign):
        for tg in node.targets:
          if isinstance(tg, ast.Name):
            name = tg.id
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
              val = node.value.value
              if '<html' in val or '<!DOCTYPE html' in val:
                html_candidates.append({'name': name, 'preview': val[:200]})
      elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        name = node.target.id
        v = node.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
          val = v.value
          if '<html' in val or '<!DOCTYPE html' in val:
            html_candidates.append({'name': name, 'preview': val[:200]})

    # Fallback: text regex for large triple-quoted or assigned HTML content
    re_blocks = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([\"\']{3}.*?[\"\']{3}|\".*?\"|\'.*?\')",
                           s, flags=re.DOTALL)
    text_html_candidates = []
    for name, val in re_blocks:
      if '<html' in val or '<!DOCTYPE html' in val:
        text_html_candidates.append({'name': name, 'preview': val[:200]})

    # Extract style blocks from candidates and full source
    style_blocks = re.findall(r"<style[^>]*>(.*?)</style>", s, flags=re.DOTALL)

    Path('.backups').mkdir(exist_ok=True)
    Path('.backups/static_phase12_html_vars.txt').write_text('\n'.join(
      f"{c['name']}: {c['preview'].replace('\n',' ')[:180]}" for c in html_candidates
    ), encoding='utf-8')
    # For preview, pick first candidate from either list
    preview = ''
    if html_candidates:
      preview = re.findall(r"<html.*</html>", html_candidates[0]['preview'] + ' </html>', flags=re.DOTALL)
      preview = html_candidates[0]['preview']
    elif text_html_candidates:
      preview = text_html_candidates[0]['preview']
    Path('.backups/static_phase12_html_extract.txt').write_text(preview, encoding='utf-8')
    Path('.backups/static_phase12_style_blocks.txt').write_text('\n\n---\n\n'.join(style_blocks[:3]), encoding='utf-8')
    Path('.backups/static_phase12_summary.json').write_text(json.dumps({
      'ast_html_candidates': html_candidates,
      'text_html_candidates': text_html_candidates,
      'style_block_count': len(style_blocks),
    }, indent=2), encoding='utf-8')
    print('Static Phase 12 enumeration complete')
    PY

STEP 3: Extract CSS from static/index.html

Open the newly created static/index.html
Find the <style> tags
Extract ALL content between <style> and </style>

Create file: static/css/styles.css

Copy all CSS content (without the <style> tags themselves)

STEP 2.2: Strict Guardrails (HTML/CSS extraction)
  Preserve the exact UI and styles:
  - Copy HTML EXACTLY from Old_Files/app.py.original (no alterations)
  - Extract ALL CSS from inline <style> blocks into static/css/styles.css
  - Update index.html to link external CSS with:
    <link rel="stylesheet" href="/css/styles.css">
  - Do NOT modify text content, structure, or attributes beyond removing <style> tags
  - Preserve bilingual and responsive design attributes
  - Maintain caching behavior: do not change lifespan startup caching of index.html
  - app.py remains minimal with StaticFiles mount at /css (Phase 11)
  - Ensure get_home reads from file only when cache is None; do not mutate cache

STEP 2.3: Dependent Coverage Cross-Check
  Cross-check modules using static files and related cache variables:
  - core/lifespan.py (preload cached_index_html at startup)
  - core/state.py (cached_index_html variable)
  - api/routes.py (get_home reads from cache or file)
  - config/settings.py (any static path or template references)
  - utils/cache.py (directory setup for static, if applicable)
  - app.py (StaticFiles mount at /css)

  Record findings to:
    - .backups/static_phase12_dependents.txt

STEP 4: Update static/index.html to link external CSS

Edit static/index.html:

REMOVE:
  <style>
  [all CSS content]
  </style>

REPLACE WITH:
  <link rel="stylesheet" href="/css/styles.css">

STEP 5: Update api/routes.py get_home() function

Edit api/routes.py:

Find the get_home() function
Update to read from file if not cached:

---MODIFY get_home() FUNCTION---
@router.get("/", response_class=HTMLResponse)
async def get_home():
    from core.state import cached_index_html
    
    # Read from file if not cached
    if cached_index_html is None:
        try:
            with open("static/index.html", "r", encoding="utf-8") as f:
                content = f.read()
            # Note: Can't modify cached_index_html here (it's a module-level import)
            # Caching happens in lifespan at startup
            return content
        except FileNotFoundError:
            return "<html><body><h1>Error: index.html not found</h1></body></html>"
    
    return cached_index_html
---END MODIFICATION---

STEP 6: Git Commit
  git add static/index.html static/css/styles.css api/routes.py
  git commit -m "PHASE 12: Extract HTML and CSS to static files, update routes to serve from files"

STEP 7: Verify
  ls -la static/
  ls -la static/css/
  head -20 static/index.html
  head -20 static/css/styles.css
  grep -c "<link rel=\"stylesheet\"" static/index.html

Expected: 1 (should have the CSS link)

STEP 5.1: Enhanced Verification (static files + route behavior)
  Validate that static files exist and get_home behaves correctly.

  Run:
    # Check files exist and contain expected markers
    test -f static/index.html && echo "index.html exists"
    test -f static/css/styles.css && echo "styles.css exists"
    grep -n "<link rel=\"stylesheet\" href=\"/css/styles.css\"" static/index.html
    grep -n "</style>" static/index.html && echo "ERROR: inline style not removed" || true

    # AST/text verify get_home implementation
    python - << 'PY'
    import ast, sys
    import inspect
    from pathlib import Path
    p = Path('api/routes.py')
    src = p.read_text(encoding='utf-8')
    t = ast.parse(src)
    fn = None
    for n in ast.walk(t):
      if isinstance(n, ast.AsyncFunctionDef) and n.name == 'get_home':
        fn = n
        break
    assert fn is not None, 'get_home not found'
    txt = src
    assert '@router.get("/", response_class=HTMLResponse)' in txt, 'decorator mismatch for get_home'
    assert 'cached_index_html' in txt, 'cached_index_html usage missing'
    assert 'open("static/index.html"' in txt, 'file read path missing'
    assert 'encoding="utf-8"' in txt, 'encoding missing'
    print('Static Phase 12 enhanced verification OK')
    PY

STEP 8: Update PROGRESS.md

Edit PROGRESS.md:

Change Phase 12 section to:
---UPDATE---
## Phase 12: Extract Static Files ✅ COMPLETE

Completed: [CURRENT_TIMESTAMP]

Files Created:
  - static/index.html ([X] lines)
  - static/css/styles.css ([Y] lines)

Files Modified:
  - api/routes.py (updated get_home() to read from file)

Content Extracted:
  - Complete HTML UI extracted from Old_Files/app.py.original
  - All CSS styles extracted from <style> tags
  - HTML updated to link external CSS
  - Routes updated to serve from files

File Structure:
  static/
    index.html (full UI markup)
    css/
      styles.css (all styles)

Static File Serving:
  - CSS mounted at /css via FastAPI StaticFiles (done in Phase 11)
  - HTML served via get_home() endpoint
  - Caching preserved (loaded at startup in lifespan)

Commits: 1
Commit Hash: [git log -1 --format=%H]

Notes:
  - HTML and CSS successfully separated
  - UI structure preserved
  - Bilingual support intact
  - Responsive design maintained
  - Static file serving configured
  - All styles extracted

Next Phase: Phase 13 (Update Docker Configuration)
---END UPDATE---

Update Phase Status Overview:
  Phase 12: Static Files ✅ COMPLETE
  Phase 13: Docker Config ⏳ NOW READY

STEP 9: Commit Progress Update
  git add PROGRESS.md
  git commit -m "PHASE 12: Update progress tracker - marked complete"

STEP 10: Report Completion

Provide this report:

PHASE 12 COMPLETE

Files Created:
  - static/index.html ([X] lines)
  - static/css/styles.css ([Y] lines)

Files Modified:
  - api/routes.py (get_home() function)

Verification: HTML and CSS files exist, CSS link present

Commits: 2

Next Phase: Phase 13 (Update Docker Configuration - FINAL PHASE)

Ready for next instruction.


================================================================================
PHASE 13: UPDATE DOCKER CONFIGURATION
================================================================================

CRITICAL REMINDERS:
- Work ONLY on Modular branch
- Never modify main branch
- DO NOT BUILD DOCKER IMAGES - Only update configuration
- User will build containers after Phase 13 complete

PHASE 13: Update Docker Configuration (Complete this entire phase in one response)

STEP 1: Read PROGRESS.md
  cat PROGRESS.md | grep "Phase 13" -A 10

Confirm Phase 13 status is: NOT STARTED or NOW READY
Confirm Phase 12 status is: COMPLETE

STEP 2: Create .dockerignore

Create file: .dockerignore

Content:
---START FILE---
.git
.gitignore
.backups
Old_Files
__pycache__
*.pyc
*.pyo
*.pyd
.Python
.env
test_*.py
*.log
.DS_Store
*.md
docs/
.vscode/
.idea/
PROGRESS.md
---END FILE---

STEP 2.1: AST/Text Enumeration (Docker config)
  Enumerate current Dockerfile and Dockerfile.ivrit structure to inform COPY
  replacements. Capture existing COPY/ADD commands, base image, workdir,
  exposed ports, entrypoint/cmd, and any environment settings.

  Enumeration outputs:
    - .backups/docker_phase13_copy_dockerfile.txt         # COPY/ADD lines
    - .backups/docker_phase13_copy_dockerfile_ivrit.txt   # COPY/ADD lines
    - .backups/docker_phase13_summary.json                # structural summary

  Run:
    python - << 'PY'
    import re, json
    from pathlib import Path

    def parse_dockerfile (path: str):
        p = Path(path)
        if not p.exists():
            return {
                'path': path,
                'exists': False,
                'base_image': None,
                'workdir': None,
                'expose': [],
                'cmd': None,
                'entrypoint': None,
                'env': [],
                'copy': [],
                'add': [],
            }
        s = p.read_text(encoding='utf-8')
        lines = s.splitlines()
        def m (pat):
            for ln in lines:
                m = re.match(pat, ln.strip(), flags=re.IGNORECASE)
                if m:
                    return m
            return None
        base = m(r'^FROM\s+(.+)$')
        workdir = m(r'^WORKDIR\s+(.+)$')
        cmd = m(r'^CMD\s+(.+)$')
        entry = m(r'^ENTRYPOINT\s+(.+)$')
        expose = [re.match(r'^EXPOSE\s+(.+)$', ln.strip(), flags=re.IGNORECASE).group(1)
                  for ln in lines if re.match(r'^EXPOSE\s+(.+)$', ln.strip(), flags=re.IGNORECASE)]
        env = [ln.strip() for ln in lines if re.match(r'^ENV\s+', ln.strip(), flags=re.IGNORECASE)]
        copy = [ln.strip() for ln in lines if re.match(r'^(COPY|ADD)\s+', ln.strip(), flags=re.IGNORECASE)]
        add = [ln.strip() for ln in lines if ln.strip().upper().startswith('ADD ')]
        return {
            'path': path,
            'exists': True,
            'base_image': base.group(1) if base else None,
            'workdir': workdir.group(1) if workdir else None,
            'expose': expose,
            'cmd': cmd.group(1) if cmd else None,
            'entrypoint': entry.group(1) if entry else None,
            'env': env,
            'copy': copy,
            'add': add,
        }

    d1 = parse_dockerfile('Dockerfile')
    d2 = parse_dockerfile('Dockerfile.ivrit')

    Path('.backups').mkdir(exist_ok=True)
    Path('.backups/docker_phase13_copy_dockerfile.txt').write_text('\n'.join(d1['copy']), encoding='utf-8')
    Path('.backups/docker_phase13_copy_dockerfile_ivrit.txt').write_text('\n'.join(d2['copy']), encoding='utf-8')
    Path('.backups/docker_phase13_summary.json').write_text(json.dumps({'Dockerfile': d1, 'Dockerfile.ivrit': d2}, indent=2), encoding='utf-8')
    print('Docker Phase 13 enumeration complete')
    PY

STEP 2.2: Strict Guardrails (Docker changes)
  Preserve configuration outside COPY updates and .dockerignore creation:
  - Do NOT change base image, Python version, workdir, entrypoint, or env
  - Only replace COPY/ADD block to include modular directories
  - Keep .dockerignore exclusions exact; do not exclude runtime-needed files
  - DO NOT run docker build or compose; configuration updates only
  - Ensure static assets are included (static/, including css/)
  - Ensure requirements.txt is copied; pip installs unchanged
  - Maintain server start via start.sh (user rule) and avoid dev-only commands

STEP 2.3: Dependent Coverage Cross-Check
  Confirm directories/files referenced in Docker COPY exist in repo:
  - config/, core/, models/, services/, api/, utils/, static/
  - requirements.txt present
  - start.sh present (execution via user workflow)
  - docker-compose.ivrit.yml paths consistent with modular structure

  Record findings to:
    - .backups/docker_phase13_dependents.txt

  Run:
    python - << 'PY'
    from pathlib import Path
    items = [
      'config', 'core', 'models', 'services', 'api', 'utils', 'static',
      'requirements.txt', 'start.sh', 'docker-compose.ivrit.yml',
    ]
    lines = []
    for item in items:
      p = Path(item)
      exists = p.exists()
      typ = 'dir' if p.is_dir() else ('file' if p.is_file() else 'missing')
      lines.append(f"{item}: {exists} ({typ})")
    Path('.backups').mkdir(exist_ok=True)
    Path('.backups/docker_phase13_dependents.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('Docker Phase 13 dependents check complete')
    PY

STEP 3: Update Dockerfile

Edit Dockerfile:

FIND the COPY commands section (search for "COPY app.py"):

REPLACE the COPY section with:
---REPLACE COPY SECTION---
# Copy modular application structure
COPY app.py /app/
COPY config/ /app/config/
COPY core/ /app/core/
COPY models/ /app/models/
COPY services/ /app/services/
COPY api/ /app/api/
COPY utils/ /app/utils/
COPY static/ /app/static/
COPY requirements.txt /app/
---END REPLACEMENT---

DO NOT modify any other sections of Dockerfile.

STEP 4: Update Dockerfile.ivrit

Edit Dockerfile.ivrit - make the SAME changes as Step 3:

FIND the COPY commands section
REPLACE with the same modular structure as above

STEP 5: Review docker-compose.ivrit.yml

Review docker-compose.ivrit.yml for any hardcoded path references.

If no changes needed: Note in commit message "No changes required"
If changes needed: Update paths and document changes

STEP 6: Create Docker Build Documentation

Create file: DOCKER_BUILD_READY.md

Content:
---START FILE---
# Docker Build Readiness

Refactoring Complete: [CURRENT_TIMESTAMP]
Branch: Modular

## Modular Structure

All code has been refactored from monolithic app.py (3,618 lines) to modular architecture:

config/
  - settings.py (environment variables)
  - constants.py (application constants)

core/
  - lifespan.py (startup/shutdown)
  - state.py (global state management)

models/
  - loader.py (thread-safe model loading)

services/
  - audio_processor.py (download, processing, AudioStreamProcessor)
  - transcription.py (whisper, faster-whisper, Deepgram)
  - diarization.py (speaker detection)
  - video_metadata.py (YouTube metadata)

api/
  - routes.py (HTTP endpoints)
  - websocket.py (WebSocket transcription)

utils/
  - cache.py (cache management)
  - validators.py (URL validation)
  - helpers.py (utility functions)
  - websocket_helpers.py (WebSocket utilities)

static/
  - index.html (UI)
  - css/styles.css (styles)

app.py (~40 lines - entry point)

## Docker Build Commands

Standard image:
  docker build -f Dockerfile -t transcription-modular:latest .

Ivrit-optimized image:
  docker build -f Dockerfile.ivrit -t transcription-ivrit-modular:latest .

With docker-compose:
  docker-compose -f docker-compose.ivrit.yml build

## Test Commands

Run container:
  docker run --rm -p 8009:8009 transcription-ivrit-modular:latest

Test health:
  curl http://localhost:8009/health

Open UI:
  http://localhost:8009

## Next Steps

1. Build Docker images
2. Test container startup
3. Verify all endpoints
4. Test transcription workflow
5. Verify caching works
6. Test diarization
7. Merge to main if all tests pass

## Notes

- All functionality preserved
- Thread safety maintained
- Async patterns intact
- WebSocket orchestration preserved
- Cache management working
- Static files served correctly

Ready for production build and deployment.
---END FILE---

STEP 7: Git Commit
  git add .dockerignore Dockerfile Dockerfile.ivrit docker-compose.ivrit.yml DOCKER_BUILD_READY.md
  git commit -m "PHASE 13: Update Docker configuration for modular structure, create build documentation"

STEP 8: Verify Configuration
  cat Dockerfile | grep "COPY config/"
  cat Dockerfile.ivrit | grep "COPY config/"
  cat .dockerignore | head -10
  ls -la DOCKER_BUILD_READY.md

CRITICAL: DO NOT RUN:
  - docker build
  - docker-compose build
  - docker run

STEP 8.1: Enhanced Verification (Docker config)
  Validate Docker configuration changes are correct and limited to COPY updates.

  Run:
    # Quick grep checks for expected COPY lines
    for f in Dockerfile Dockerfile.ivrit; do
      echo "Checking $f"
      grep -q "COPY app.py /app/" $f && echo "OK: app.py copy" || echo "MISSING: app.py copy"
      for d in config core models services api utils static; do
        grep -q "COPY $d/ /app/$d/" $f && echo "OK: $d/ copy" || echo "MISSING: $d/ copy"
      done
      grep -q "COPY requirements.txt /app/" $f && echo "OK: requirements.txt copy" || echo "MISSING: requirements.txt copy"
    done

    # Python structural verification comparing pre-enumeration summary when available
    python - << 'PY'
    import json, re
    from pathlib import Path

    def parse(path: str):
        p = Path(path)
        s = p.read_text(encoding='utf-8')
        lines = s.splitlines()
        def first(pat):
            for ln in lines:
                m = re.match(pat, ln.strip(), flags=re.IGNORECASE)
                if m:
                    return m.group(1)
            return None
        expose = [re.match(r'^EXPOSE\s+(.+)$', ln.strip(), flags=re.IGNORECASE).group(1)
                  for ln in lines if re.match(r'^EXPOSE\s+(.+)$', ln.strip(), flags=re.IGNORECASE)]
        copy = [ln.strip() for ln in lines if re.match(r'^(COPY|ADD)\s+', ln.strip(), flags=re.IGNORECASE)]
        return {
            'base_image': first(r'^FROM\s+(.+)$'),
            'workdir': first(r'^WORKDIR\s+(.+)$'),
            'cmd': first(r'^CMD\s+(.+)$'),
            'entrypoint': first(r'^ENTRYPOINT\s+(.+)$'),
            'expose': expose,
            'copy': copy,
        }

    cur_d = parse('Dockerfile')
    cur_i = parse('Dockerfile.ivrit')

    # Expected COPY entries
    expected = [
      'COPY app.py /app/',
      'COPY config/ /app/config/',
      'COPY core/ /app/core/',
      'COPY models/ /app/models/',
      'COPY services/ /app/services/',
      'COPY api/ /app/api/',
      'COPY utils/ /app/utils/',
      'COPY static/ /app/static/',
      'COPY requirements.txt /app/',
    ]

    for name, cur in [('Dockerfile', cur_d), ('Dockerfile.ivrit', cur_i)]:
        for e in expected:
            assert any(e in c for c in cur['copy']), f"{name}: missing '{e}'"

    # If enumeration summary exists, confirm sensitive fields unchanged
    summ_p = Path('.backups/docker_phase13_summary.json')
    if summ_p.exists():
        prev = json.loads(summ_p.read_text(encoding='utf-8'))
        for name, cur in [('Dockerfile', cur_d), ('Dockerfile.ivrit', cur_i)]:
            old = prev.get(name, {})
            for key in ['base_image', 'workdir', 'cmd', 'entrypoint']:
                if old.get(key) is not None:
                    assert old.get(key) == cur.get(key), f"{name}: {key} changed"
    else:
        print('Warning: .backups/docker_phase13_summary.json not found; skipping immutability check')

    # .dockerignore must contain key exclusions
    di = Path('.dockerignore').read_text(encoding='utf-8')
    for token in ['.backups', 'Old_Files', 'PROGRESS.md']:
        assert token in di, f".dockerignore missing {token}"

    # DOCKER_BUILD_READY.md must contain headers and Modular Structure section
    db = Path('DOCKER_BUILD_READY.md').read_text(encoding='utf-8')
    assert 'Docker Build Readiness' in db, 'DOCKER_BUILD_READY.md missing header'
    assert 'Modular Structure' in db, 'DOCKER_BUILD_READY.md missing Modular Structure'
    for token in ['config/', 'core/', 'api/', 'utils/', 'static/']:
        assert token in db, f"DOCKER_BUILD_READY.md missing {token} entry"

    print('Docker Phase 13 enhanced verification OK')
    PY

STEP 9: Update PROGRESS.md

Edit PROGRESS.md:

Change Phase 13 section AND add Final Summary:
---UPDATE---
## Phase 13: Update Docker Configuration ✅ COMPLETE

Completed: [CURRENT_TIMESTAMP]

Files Created:
  - .dockerignore
  - DOCKER_BUILD_READY.md

Files Modified:
  - Dockerfile (updated COPY commands)
  - Dockerfile.ivrit (updated COPY commands)
  - docker-compose.ivrit.yml (reviewed, [changes made/no changes required])

Docker Configuration Changes:
  - COPY commands updated to include all module directories
  - .dockerignore created to exclude dev files
  - Build documentation created
  - Ready for container build

COPY Structure:
  COPY app.py /app/
  COPY config/ /app/config/
  COPY core/ /app/core/
  COPY models/ /app/models/
  COPY services/ /app/services/
  COPY api/ /app/api/
  COPY utils/ /app/utils/
  COPY static/ /app/static/
  COPY requirements.txt /app/

Commits: 1
Commit Hash: [git log -1 --format=%H]

Notes:
  - Docker configuration updated for modular structure
  - All module directories will be copied to container
  - .dockerignore excludes dev/temp files
  - Build documentation ready
  - DO NOT BUILD - User will build containers

CRITICAL: Docker build NOT performed (as instructed)
User will build and test containers after reviewing all changes.

---

## 🎉 REFACTORING COMPLETE - FINAL SUMMARY

Completion Date: [CURRENT_TIMESTAMP]
Branch: Modular
Total Phases: 14 (0-13)
All Phases: ✅ COMPLETE

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

### Commits Summary
  - Phase 0: 2 commits (setup, progress tracker)
  - Phase 1A: 2 commits (config extraction)
  - Phase 1B: 2 commits (state management)
  - Phase 2: 2 commits (utilities)
  - Phase 3: 2 commits (cache)
  - Phase 4: 2 commits (models)
  - Phase 5: 2 commits (audio processing)
  - Phase 6: 2 commits (transcription)
  - Phase 7: 2 commits (diarization)
  - Phase 8: 2 commits (video metadata)
  - Phase 9: 2 commits (API routes)
  - Phase 10: 2 commits (WebSocket)
  - Phase 11: 2 commits (lifespan, minimal app.py)
  - Phase 12: 2 commits (static files)
  - Phase 13: 2 commits (Docker config)
  
  Total: 30 commits

### Critical Patterns Preserved
  ✅ Thread-safe model loading (double-check locking)
  ✅ Global state management (centralized in core/state.py)
  ✅ Async/await patterns throughout
  ✅ WebSocket state checks and error handling
  ✅ Nested functions kept with parents (closures intact)
  ✅ Subprocess management (FFmpeg, yt-dlp)
  ✅ ThreadPoolExecutor for CPU-bound tasks
  ✅ Cache deduplication (SHA256 hashing)
  ✅ Diarization integration
  ✅ Deepgram SDK integration
  ✅ Complete transcription workflow orchestration

### Functionality Verification
  ✅ All 48 functions extracted and preserved
  ✅ All 3 classes extracted (AudioStreamProcessor + 2 others if present)
  ✅ All imports updated correctly
  ✅ All modules compile without errors
  ✅ Import tests passed for all modules
  ✅ Static files extracted and linked

### Next Steps for User

1. Review Changes:
   git log --oneline --graph
   git diff main..Modular --stat

2. Test Locally:
   python app.py
   # Should start without errors
   # Test endpoints: http://localhost:8009/health
   # Test UI: http://localhost:8009

3. Build Docker Images:
   docker build -f Dockerfile.ivrit -t transcription-modular:test .

4. Test Container:
   docker run --rm -p 8009:8009 transcription-modular:test

5. Full Testing:
   - Test YouTube transcription
   - Test live stream
   - Test diarization
   - Test caching
   - Test all endpoints

6. Merge to Main:
   git checkout main
   git merge Modular
   git push origin main

### Files for Review
  - PROGRESS.md (complete phase-by-phase record)
  - DOCKER_BUILD_READY.md (build instructions)
  - All module files in their directories

### Success Criteria Met
  ✅ Zero breaking changes
  ✅ All functionality preserved
  ✅ Thread safety maintained
  ✅ Async patterns intact
  ✅ Modular organization achieved
  ✅ Docker ready for build
  ✅ Main branch untouched

REFACTORING 100% COMPLETE
Ready for user build and deployment testing.
---END UPDATE---

Update Phase Status Overview (mark ALL phases complete):
  Phase 0: Setup ✅ COMPLETE
  Phase 1A: Config ✅ COMPLETE
  Phase 1B: State ✅ COMPLETE
  Phase 2: Utils ✅ COMPLETE
  Phase 3: Cache ✅ COMPLETE
  Phase 4: Models ✅ COMPLETE
  Phase 5: Audio ✅ COMPLETE
  Phase 6: Transcription ✅ COMPLETE
  Phase 7: Diarization ✅ COMPLETE
  Phase 8: Video Metadata ✅ COMPLETE
  Phase 9: API Routes ✅ COMPLETE
  Phase 10: WebSocket ✅ COMPLETE
  Phase 11: Lifespan ✅ COMPLETE
  Phase 12: Static Files ✅ COMPLETE
  Phase 13: Docker Config ✅ COMPLETE

STEP 10: Final Commit of Progress
  git add PROGRESS.md
  git commit -m "PHASE 13: Update progress tracker - ALL PHASES COMPLETE"

STEP 11: Final Report

Provide this comprehensive report:

================================================================================
🎉 REFACTORING COMPLETE - ALL PHASES DONE
================================================================================

Branch: Modular
Phases Completed: 14 (0-13)
Total Commits: 30

FILES CREATED:
  - Configuration: 2 files
  - Core: 2 files
  - Models: 1 file
  - Services: 4 files
  - API: 2 files
  - Utils: 4 files
  - Static: 2 files
  - Docker: 2 files
  - Total: 19 new files

APP.PY TRANSFORMATION:
  - Original: 3,618 lines
  - New: ~40-50 lines
  - Reduction: 98.6%

CRITICAL PATTERNS PRESERVED:
  ✅ Thread safety
  ✅ Global state
  ✅ Async patterns
  ✅ WebSocket orchestration
  ✅ Nested functions (closures)
  ✅ Subprocess management

VERIFICATION STATUS:
  ✅ All modules compile
  ✅ All imports work
  ✅ Static files extracted
  ✅ Docker config updated

READY FOR:
  1. Local testing (python app.py)
  2. Docker build
  3. Container testing
  4. Merge to main

DOCUMENTATION:
  - PROGRESS.md (complete phase record)
  - DOCKER_BUILD_READY.md (build instructions)

USER ACTION REQUIRED:
  1. Review all changes
  2. Test application locally
  3. Build Docker images
  4. Test containers
  5. Merge to main if all tests pass

Main branch: Untouched (as required)
Rollback available: git checkout main && git branch -D Modular

================================================================================
ALL REFACTORING PHASES SUCCESSFULLY COMPLETED
================================================================================

Ready for user build and deployment.


================================================================================
END OF PROMPTS FILE
================================================================================