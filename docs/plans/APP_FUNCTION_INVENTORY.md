# app.py Function Inventory (Expanded)

Total functions (incl. nested & methods): 56
- Top-level: 40
- Class methods: 4
- Nested functions: 12

## Top-level
- async lifespan(app) @ line 158
- async safe_ws_send(websocket, data) @ line 241
- def init_capture_dir() @ line 359
- def init_download_cache_dir() @ line 366
- def get_url_hash(url) @ line 387
- def get_cached_download(url) @ line 392
- def save_download_to_cache(url, audio_file) @ line 417
- def load_model(model_name) @ line 452
- def get_diarization_pipeline() @ line 553
- async transcribe_with_diarization(model, model_config, audio_file, language, websocket, model_name) @ line 591
- def should_use_ytdlp(url) @ line 833
- async download_audio_with_ffmpeg(url, format, duration, websocket, use_cache) @ line 866
- async download_with_fallback(url, language, format, websocket, use_cache) @ line 1159
- async download_audio_with_ytdlp_async(url, language, format, websocket, use_cache) @ line 1225
- def get_audio_duration_seconds(audio_path) @ line 1460
- def calculate_progress_metrics(audio_duration, elapsed_time, processed_chunks, total_chunks) @ line 1483
- def split_audio_for_incremental(audio_path, chunk_seconds, overlap_seconds) @ line 1531
- def split_audio_into_chunks(audio_path, chunk_seconds, overlap_seconds) @ line 1578
- async transcribe_with_incremental_output(model, model_config, audio_file, language, websocket, model_name, chunk_seconds) @ line 1626
- def transcribe_chunk(model_config, model, chunk_path, language) @ line 1977
- def init_cache_dir() @ line 2033
- def generate_cache_key(audio_data, sample_rate, channels) @ line 2057
- def get_cached_audio(cache_key) @ line 2065
- def save_to_cache(cache_key, audio_path) @ line 2077
- def format_duration(seconds) @ line 2099
- def format_view_count(count) @ line 2114
- def is_youtube_url(url) @ line 2129
- async get_youtube_metadata(url) @ line 2144
- async transcribe_audio_stream(websocket, processor) @ line 2335
- async transcribe_vod_with_deepgram(websocket, url, language) @ line 2502
- async transcribe_with_deepgram(websocket, url, language) @ line 2781
- async get_home() @ line 3068 [get /]
- async websocket_transcribe(websocket) @ line 3074 [websocket /ws/transcribe]
- async health_check() @ line 3419 [get /health]
- async get_video_info(request) @ line 3429 [post /api/video-info]
- async gpu_diagnostics() @ line 3485 [get /gpu]
- async cache_stats() @ line 3518 [get /api/cache/stats]
- async clear_cache() @ line 3538 [post /api/cache/clear]
- async download_cache_stats() @ line 3555 [get /api/download-cache/stats]
- async clear_download_cache() @ line 3584 [post /api/download-cache/clear]

## Class methods
- def AudioStreamProcessor.__init__(self, url, language, model_name) @ line 2220
- def AudioStreamProcessor.start_ffmpeg_stream(self) @ line 2229
- def AudioStreamProcessor.read_audio_chunks(self) @ line 2258
- def AudioStreamProcessor.stop(self) @ line 2321

## Nested functions
- async download_audio_with_ffmpeg.monitor_progress() @ line 944
- def download_audio_with_ffmpeg.monitor_progress.read_progress_file() @ line 960
- def transcribe_with_incremental_output.run_fw_transcription() @ line 1671
- def transcribe_with_incremental_output.run_transcription() @ line 1728
- def transcribe_with_incremental_output.transcribe_fw_chunk() @ line 1831
- def transcribe_with_incremental_output.transcribe_openai_chunk() @ line 1866
- def transcribe_vod_with_deepgram.read_audio_file() @ line 2610
- def transcribe_with_deepgram.extract_deepgram_transcript(message) @ line 2794
- def transcribe_with_deepgram.on_message() @ line 2845
- def transcribe_with_deepgram.on_close() @ line 2874
- def transcribe_with_deepgram.on_error() @ line 2887
- def websocket_transcribe.read_capture_file() @ line 3238

---

## 📚 Module Assignment Matrix (Complete)

Top-level functions

| Original Function | Async | Line | Target Module | Risk Level | Dependencies |
|-------------------|-------|------|---------------|------------|--------------|
| lifespan(app) | yes | 158 | core/lifespan.py | MEDIUM | asyncio, concurrent.futures, os, pathlib.Path, logging, config.settings, core.state |
| safe_ws_send(websocket, data) | yes | 241 | utils/websocket_helpers.py | HIGH | fastapi.websockets, json, logging |
| send_status_update(websocket, message) | yes | — | utils/websocket_helpers.py | MEDIUM | fastapi.websockets, json, logging |
| send_transcription_progress(websocket, progress) | yes | — | utils/websocket_helpers.py | MEDIUM | fastapi.websockets, json, logging |
| send_transcription_chunk(websocket, chunk) | yes | — | utils/websocket_helpers.py | MEDIUM | fastapi.websockets, json, logging |
| send_transcription_status(websocket, status) | yes | — | utils/websocket_helpers.py | MEDIUM | fastapi.websockets, json, logging |
| init_capture_dir() | no | 359 | utils/cache.py | MEDIUM | os, pathlib.Path |
| init_download_cache_dir() | no | 366 | utils/cache.py | MEDIUM | os, pathlib.Path |
| get_url_hash(url) | no | 387 | utils/cache.py | LOW | hashlib |
| get_cached_download(url) | no | 392 | utils/cache.py | MEDIUM | os, pathlib.Path |
| save_download_to_cache(url, audio_file) | no | 417 | utils/cache.py | MEDIUM | os, pathlib.Path |
| load_model(model_name) | no | 452 | models/loader.py | HIGH | threading, whisper libs, config.settings, core.state |
| get_diarization_pipeline() | no | 553 | models/loader.py | HIGH | threading, diarization libs, core.state |
| transcribe_with_diarization(...) | yes | 591 | services/diarization.py | HIGH | models.loader, core.state, asyncio, websockets |
| should_use_ytdlp(url) | no | 833 | utils/validators.py | LOW | re |
| download_audio_with_ffmpeg(...) | yes | 866 | services/audio_processor.py | HIGH | subprocess/ffmpeg, asyncio, websockets, pathlib.Path |
| download_with_fallback(...) | yes | 1159 | services/audio_processor.py | HIGH | yt-dlp, ffmpeg, asyncio, websockets |
| download_audio_with_ytdlp_async(...) | yes | 1225 | services/audio_processor.py | HIGH | yt-dlp, asyncio, websockets |
| get_audio_duration_seconds(audio_path) | no | 1460 | services/audio_processor.py | MEDIUM | ffprobe/subprocess, pathlib.Path |
| calculate_progress_metrics(...) | no | 1483 | services/audio_processor.py | LOW | math, time |
| split_audio_for_incremental(...) | no | 1531 | services/audio_processor.py | MEDIUM | audio libs, pathlib.Path |
| split_audio_into_chunks(...) | no | 1578 | services/audio_processor.py | MEDIUM | audio libs, pathlib.Path |
| transcribe_with_incremental_output(...) | yes | 1626 | services/transcription.py | HIGH | models.loader, asyncio, websockets, services.audio_processor |
| transcribe_chunk(model_config, model, chunk_path, language) | no | 1977 | services/transcription.py | MEDIUM | whisper libs, io |
| init_cache_dir() | no | 2033 | utils/cache.py | MEDIUM | os, pathlib.Path |
| generate_cache_key(audio_data, sample_rate, channels) | no | 2057 | utils/cache.py | LOW | hashlib |
| get_cached_audio(cache_key) | no | 2065 | utils/cache.py | MEDIUM | os, pathlib.Path |
| save_to_cache(cache_key, audio_path) | no | 2077 | utils/cache.py | MEDIUM | os, pathlib.Path |
| format_duration(seconds) | no | 2099 | utils/helpers.py | LOW | datetime |
| format_view_count(count) | no | 2114 | utils/helpers.py | LOW | math |
| is_youtube_url(url) | no | 2129 | utils/validators.py | LOW | re |
| get_youtube_metadata(url) | yes | 2144 | services/video_metadata.py | MEDIUM | yt-dlp/http client, asyncio |
| transcribe_audio_stream(websocket, processor) | yes | 2335 | services/transcription.py | HIGH | services.audio_processor, asyncio, websockets |
| transcribe_vod_with_deepgram(websocket, url, language) | yes | 2502 | services/transcription.py | HIGH | deepgram sdk, asyncio, websockets |
| transcribe_with_deepgram(websocket, url, language) | yes | 2781 | services/transcription.py | HIGH | deepgram sdk, asyncio, websockets |
| get_home() | yes | 3068 | api/routes.py | LOW | core.state (cached HTML), static files |
| websocket_transcribe(websocket) | yes | 3074 | api/websocket.py | HIGH | services.audio_processor, services.transcription, utils.validators, utils.websocket_helpers, core.state |
| health_check() | yes | 3419 | api/routes.py | LOW | config.settings, core.state |
| get_video_info(request) | yes | 3429 | api/routes.py | MEDIUM | services.video_metadata, utils.validators |
| gpu_diagnostics() | yes | 3485 | api/routes.py | MEDIUM | torch/cuda libs, logging |
| cache_stats() | yes | 3518 | api/routes.py | LOW | utils.cache |
| clear_cache() | yes | 3538 | api/routes.py | MEDIUM | utils.cache |
| download_cache_stats() | yes | 3555 | api/routes.py | LOW | utils.cache |
| clear_download_cache() | yes | 3584 | api/routes.py | MEDIUM | utils.cache |

Class methods

| Original Method | Async | Line | Target Module | Risk Level | Dependencies |
|-----------------|-------|------|---------------|------------|--------------|
| AudioStreamProcessor.__init__(self, url, language, model_name) | no | 2220 | services/audio_processor.py | MEDIUM | pathlib.Path, config, cache |
| AudioStreamProcessor.start_ffmpeg_stream(self) | no | 2229 | services/audio_processor.py | HIGH | subprocess/ffmpeg, threading |
| AudioStreamProcessor.read_audio_chunks(self) | no | 2258 | services/audio_processor.py | HIGH | streaming IO, queues |
| AudioStreamProcessor.stop(self) | no | 2321 | services/audio_processor.py | MEDIUM | cleanup, IO |

Nested helpers

| Original Helper | Async | Line | Target Module | Risk Level | Dependencies |
|------------------|-------|------|---------------|------------|--------------|
| download_audio_with_ffmpeg.monitor_progress() | yes | 944 | services/audio_processor.py | HIGH | asyncio, file system |
| monitor_progress.read_progress_file() | no | 960 | services/audio_processor.py | HIGH | file IO |
| transcribe_with_incremental_output.run_fw_transcription() | no | 1671 | services/transcription.py | HIGH | model inference |
| transcribe_with_incremental_output.run_transcription() | no | 1728 | services/transcription.py | HIGH | model inference |
| transcribe_with_incremental_output.transcribe_fw_chunk() | no | 1831 | services/transcription.py | HIGH | model inference |
| transcribe_with_incremental_output.transcribe_openai_chunk() | no | 1866 | services/transcription.py | HIGH | external API |
| transcribe_vod_with_deepgram.read_audio_file() | no | 2610 | services/transcription.py | MEDIUM | file IO |
| transcribe_with_deepgram.extract_deepgram_transcript(message) | no | 2794 | services/transcription.py | MEDIUM | json parsing |
| transcribe_with_deepgram.on_message() | no | 2845 | services/transcription.py | HIGH | websocket message handling |
| transcribe_with_deepgram.on_close() | no | 2874 | services/transcription.py | MEDIUM | ws close management |
| transcribe_with_deepgram.on_error() | no | 2887 | services/transcription.py | MEDIUM | error handling |
| websocket_transcribe.read_capture_file() | no | 3238 | api/websocket.py | MEDIUM | file IO |

Maintain this matrix in lockstep with the refactor phases. Use `docs/plans/APP_FUNCTION_INVENTORY.md` line anchors and `IMPLEMENTATION_PLAN.md` mappings to update Risk Level and Dependencies when moving functions.

---

## 🗺️ Import Dependency Map (live)

Keep this map in sync with actual imports after each phase. Validate via static analysis (e.g., grep/ripgrep) and runtime logs.

```
app.py
  ├── config.settings
  ├── config.constants
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
  │   ├── models.loader
  │   ├── utils.cache
  │   ├── utils.validators
  │   ├── utils.websocket_helpers
  │   └── core.state
  ├── models.loader
  ├── utils.helpers
  └── static (via api.routes.get_home and core.state cache)

services.transcription
  ├── models.loader
  ├── services.diarization
  ├── utils.cache
  ├── utils.validators
  ├── utils.websocket_helpers
  └── core.state

services.audio_processor
  ├── utils.cache
  ├── utils.validators
  ├── utils.websocket_helpers
  └── core.state

models.loader
  ├── config.settings
  └── core.state

core.lifespan
  ├── config.settings
  ├── core.state
  └── utils.cache (directory creation)

api.routes
  ├── utils.cache
  ├── utils.validators
  └── services.video_metadata

api.websocket
  ├── services.audio_processor
  ├── utils.websocket_helpers
  └── services.transcription

[Maintain and update after each phase]
```

---

## ⚠️ Critical Considerations Cross-Check

- Thread Safety: Locking is required in `models/loader.py` (model cache), `core/state.py` (global singletons), and diarization pipeline access. Preserve atomic updates and use existing locks.
- Global State: Centralize singletons in `core/state.py` (`current_model`, `current_model_name`, `diarization_pipeline`, `cached_index_html`, `executor`, `CAPTURES`). Do not duplicate or reinitialize in submodules.
- Async Operations: Async-heavy modules include `api.websocket`, `api.routes` (handlers), `services.audio_processor`, `services.transcription`, `services.diarization`. Preserve async boundaries; avoid blocking calls on the event loop.
- WebSocket State: Enforce connected state before sends via `safe_ws_send` (centralized in `utils/websocket_helpers.py`). Maintain message sequencing, progress updates, and disconnect flows exactly.
- UI Contract: Ensure stylesheet reference uses `static/css/stylesheet.css` consistently across routes and docs.
- Docker Compatibility: Update `Dockerfile*` COPY instructions to include new module directories. Ensure runtime directories (`cache/*`, `logs/`) exist at startup via `core/lifespan.py`.

---

## 🔄 Update Protocol

1. After completing each refactor phase, update this matrix with the new Target Module, Risk Level, and Dependencies for the moved functions.
2. Rebuild the Import Dependency Map from actual imports (grep/ripgrep) and verify with runtime logs.
3. Cross-check Critical Considerations before committing to avoid regressions.
4. Keep line numbers synced by referencing `docs/plans/APP_FUNCTION_INVENTORY.md` and re-running a quick code scan if lines shift.