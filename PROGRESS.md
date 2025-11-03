# Refactoring Progress Tracker

Last Updated: 2025-11-03
Branch: Modular
Original app.py: 3,618 lines
Runtime Logs: .logs/

## Phase Status Overview

Phase 0: Setup ✅ COMPLETE
Phase 1A: Config ✅ COMPLETE
Phase 1B: State ✅ COMPLETE
Phase 2: Utils ✅ COMPLETE
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

## Phase 0: Setup ✅ COMPLETE

Completed: 2025-11-03

Files Created:
  - Old_Files/app.py.original
  - Old_Files/Dockerfile.ivrit.original
  - Old_Files/docker-compose.ivrit.yml.original
  - Old_Files/requirements.txt.original
  - Old_Files/requirements.ivrit.txt.original
  - .backups/app.py.backup
  - .backups/Dockerfile.ivrit.backup
  - .backups/docker-compose.ivrit.yml.backup
  - PROGRESS.md

Commits: 1
Commit Hash: 35a1b76

Notes:
  - Directory structure created
  - Reference files ready in Old_Files/
  - Progress tracker initialized
  - Ready for Phase 1A

---

## Phase 1A: Extract Configuration ✅ COMPLETE

Completed: 2025-11-03

Files Created:
  - config/settings.py (already existed - 50+ lines)
  - config/constants.py (17 lines)

Environment Variables Extracted:
  - DEEPGRAM_API_KEY, WHISPER_MODEL, IVRIT_MODEL_NAME
  - USE_PARALLEL_TRANSCRIPTION, PARALLEL_WORKERS
  - YTDLP_CHUNK_SECONDS, YTDLP_CHUNK_OVERLAP
  - AUDIO_CACHE_ENABLED, and others

Constants Extracted:
  - CACHE_DIR, DOWNLOAD_CACHE_DIR, CAPTURE_DIR
  - SAMPLE_RATE, CHANNELS, CHUNK_DURATION
  - CHUNK_OVERLAP, AUDIO_QUEUE_SIZE

Commits: 1
Notes:
  - config/settings.py was pre-existing with all env vars
  - Created config/constants.py with application constants

---

## Phase 1B: Create Global State ✅ COMPLETE

Completed: 2025-11-03

Files Created:
  - core/state.py (30 lines)

Global Variables Centralized:
  - whisper_models, diarization_pipeline
  - model_lock, diarization_pipeline_lock
  - executor (ThreadPoolExecutor - initialized at runtime)
  - cached_index_html
  - CAPTURES, URL_DOWNLOADS

Commits: 1
Notes:
  - Global state centralized for single source of truth
  - Thread safety locks preserved
  - No separate faster_whisper_models found in original

---

## Phase 2: Extract Utilities ✅ COMPLETE

Completed: 2025-11-03

Files Created:
  - utils/validators.py (47 lines)
  - utils/helpers.py (33 lines)
  - utils/websocket_helpers.py (29 lines)

Functions Extracted:
  validators.py:
    - is_youtube_url (line 2141)
    - should_use_ytdlp (line 845)
  helpers.py:
    - format_duration (line 2111)
    - format_view_count (line 2126)
  websocket_helpers.py:
    - safe_ws_send (line 256)

Commits: 1
Notes:
  - All utility functions extracted
  - Pure functions preserved exactly

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