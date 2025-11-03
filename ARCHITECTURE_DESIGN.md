# Modular Refactor Architecture Design

Last Updated: 2025-11-02

Purpose: Define the target module layout and precise function assignments to refactor `app.py` into a safe, modular architecture with zero functional changes.

---

## Goals

- Preserve exact functionality, signatures, and behavior
- Avoid breaking changes and keep the main branch pristine
- Eliminate circular dependencies through clear module boundaries
- Keep WebSocket behavior and error handling identical
- Maintain global caches, locks, and state semantics

---

## Proposed Directory Structure

Estimated lines and risk per file are approximate; risk reflects migration complexity and concurrency sensitivity.

```
config/
	__init__.py
	settings.py              (~80 LOC, LOW)    # env, paths, flags
	constants.py             (~80 LOC, LOW)    # static constants

core/
	__init__.py
	lifespan.py              (~120 LOC, MEDIUM) # startup/shutdown; thread pool; dirs
	state.py                 (~160 LOC, HIGH)   # global models, locks, executors

models/
	__init__.py
	loader.py                (~220 LOC, HIGH)   # load_model, get_diarization_pipeline

services/
	__init__.py
	audio_processor.py       (~500 LOC, HIGH)   # download_*; split_*; duration; ffmpeg; ytdlp
	transcription.py         (~750 LOC, HIGH)   # whisper/faster-whisper/deepgram; incremental
	diarization.py           (~280 LOC, HIGH)   # diarization + alignment
	video_metadata.py        (~140 LOC, LOW)    # get_youtube_metadata

api/
	__init__.py
	routes.py                (~260 LOC, MEDIUM) # GET /, /health, /gpu, cache endpoints, video-info
	websocket.py             (~400 LOC, HIGH)   # /ws/transcribe and ws helpers

utils/
	__init__.py
	cache.py                 (~260 LOC, MEDIUM) # file/url hashes; cache get/save; stats; clear
	validators.py            (~140 LOC, LOW)    # is_youtube_url, should_use_ytdlp, alt url
	helpers.py               (~120 LOC, LOW)    # format_timestamp, misc small helpers
	websocket_helpers.py     (~140 LOC, HIGH)   # WebSocket send helpers (safe_ws_send, status/progress/chunk)

static/
	index.html               (extracted, MEDIUM)
	css/
		stylesheet.css       (extracted, MEDIUM)
```

Notes:
- `core/state.py` centralizes globals and locks to avoid duplication and ensure consistent import graph.
- `utils/websocket_helpers.py` centralizes WebSocket helpers; `api/websocket.py` imports and uses them to keep socket semantics consistent without circular deps.
- `services/transcription.py` contains all engines and incremental orchestration; Deepgram can be kept here to avoid an extra module.

---

## Module Assignment Matrix (48 functions)

Map each current `app.py` function to its target module. Names and signatures must remain unchanged.

WebSocket Helpers → `utils/websocket_helpers.py`
- `safe_ws_send(websocket: WebSocket, message: dict) -> None`
- `send_status_update(websocket: WebSocket, message: str, progress: Optional[int] = None) -> None`
- `send_transcription_progress(websocket: WebSocket, progress: int, total_duration: float) -> None`
- `send_transcription_chunk(websocket: WebSocket, text: str, start: float, end: float, speaker: Optional[str] = None) -> None`
- `send_transcription_status(websocket: WebSocket, status: str) -> None`

Lifecycle & State → `core/lifespan.py` and `core/state.py`
- `lifespan(app: FastAPI)` → `core/lifespan.py`

Cache (Transcription & Download) → `utils/cache.py`
- `calculate_file_hash(file_path: str) -> str`
- `get_cached_transcription(file_path: str) -> Optional[Dict]`
- `save_transcription_to_cache(file_path: str, result: Dict) -> None`
- `get_url_hash(url: str) -> str`
- `get_cached_download(url: str) -> Optional[str]`
- `save_download_to_cache(url: str, audio_path: str) -> None`
- `get_cache_stats() -> Dict`
- `clear_cache() -> Dict`
- `get_download_cache_stats() -> Dict`
- `clear_download_cache() -> Dict`
- `init_capture_dir() -> None`
- `init_download_cache_dir() -> None`
- `init_cache_dir() -> None`
- `generate_cache_key(audio_data, sample_rate, channels) -> str`
- `get_cached_audio(cache_key: str) -> Optional[str]`
- `save_to_cache(cache_key: str, audio_path: str) -> None`

Validators & Small Utilities → `utils/validators.py` and `utils/helpers.py`
- `is_youtube_url(url: str) -> bool` → validators
- `should_use_ytdlp(url: str) -> bool` → validators
- `format_timestamp(seconds: float) -> str` → helpers
- `format_duration(seconds: float) -> str` → helpers
- `format_view_count(count: int) -> str` → helpers
- `split_audio_into_chunks(...)` (helper variant, if present) → audio_processor (see below)

Audio Processing & Download → `services/audio_processor.py`
- `download_audio(url: str, websocket: Optional[WebSocket] = None) -> str`
- `download_with_fallback(url: str, websocket: Optional[WebSocket] = None) -> str`
- `download_audio_with_ffmpeg(url: str, format: str = 'wav', duration: int = 60, websocket = None, use_cache: bool = True) -> Optional[str]`
- `download_audio_with_ytdlp_async(url: str, format: str = 'wav', websocket: Optional[WebSocket] = None) -> Optional[str]`
- `get_audio_duration_seconds(audio_path: str) -> float`
- `split_audio_for_incremental(audio_path: str, chunk_seconds: int = 60, overlap_seconds: int = 5) -> Tuple[str, List[str>]`
- `split_audio_into_chunks(audio_path: str, ...) -> List[str]` (if present)
- `calculate_progress_metrics(audio_duration: float, elapsed_time: float, processed_chunks: int = 0, total_chunks: int = 0) -> dict`
- `AudioStreamProcessor` class → `services/audio_processor.py` (or `audio_stream.py` if separated later)

Transcription Engines & Orchestration → `services/transcription.py`
- `transcribe_with_incremental_output(model, model_config, audio_file, language, websocket, model_name, chunk_seconds) -> Dict`
- `transcribe_chunk(model_config, model, chunk_path, language) -> Dict`
- `transcribe_audio_stream(websocket: WebSocket, processor) -> None`
- `transcribe_vod_with_deepgram(websocket: WebSocket, url: str, language: Optional[str]) -> None`
- `transcribe_with_deepgram(websocket: WebSocket, url: str, language: Optional[str]) -> None`

Diarization → `services/diarization.py`
- `transcribe_with_diarization(model, model_config, audio_file, language, websocket, model_name) -> Dict`
- `get_diarization_pipeline() -> Pipeline` (extracted helper around cached pipeline)
- `assign_speaker(segment_start: float, segment_end: float, diarization_result) -> str` (helper inside module)

Model Loading → `models/loader.py`
- `load_model(model_name: str) -> Any` (lock-protected)
- `get_diarization_pipeline() -> Pipeline` (if centralized here instead of services)

YouTube Integration → `services/video_metadata.py`
- `get_youtube_metadata(url: str) -> Dict`

API Endpoints → `api/routes.py`
- `get_home()` → GET `/`
- `health_check()` → GET `/health`
- `get_gpu_status()` → GET `/gpu`
- `cache_stats()` → GET `/api/cache/stats`
- `clear_transcription_cache()` → POST `/api/cache/clear`
- `download_cache_stats()` → GET `/api/download-cache/stats`
- `clear_download_cache_api()` → POST `/api/download-cache/clear`
- `get_video_info(request: VideoInfoRequest)` → POST `/api/video-info`

API WebSocket → `api/websocket.py`
- `websocket_transcribe(websocket: WebSocket)` → WS `/ws/transcribe`

Data Models → `api/models.py` (alternatively co-locate in `api/__init__.py` or `api/types.py`)
- `VideoInfoRequest`
- `TranscriptionRequest`
- `TranscriptionSegment` (TypedDict)
- `TranscriptionResult` (TypedDict)

---

## Import Dependency Map

Top-level application (`app.py`) will reduce to slim bootstrap:
- `from core.lifespan import lifespan`
- `from api.routes import router as http_router` (if using FastAPI Router pattern)
- `from api.websocket import register_websocket_routes` (or expose app to add `@app.websocket`)

Module imports:
- `api/routes.py` imports: `services.video_metadata`, `utils.cache`, `core.state`
- `api/websocket.py` imports: `services.audio_processor`, `services.transcription`, `models.loader`, `utils.cache`, `utils.validators`, `utils.websocket_helpers`, `core.state`
- `services/audio_processor.py` imports: `config.settings`, `utils.cache`, `utils.validators`, `utils.websocket_helpers`
- `services/transcription.py` imports: `models.loader`, `services.audio_processor`, `services.diarization`, `config.settings`, `core.state`, `utils.websocket_helpers`
- `services/diarization.py` imports: `models.loader` (or direct pyannote), `core.state`
- `models/loader.py` imports: `config.settings`, `core.state`
- `utils/cache.py` imports: `config.constants`, `config.settings`

No circular dependencies:
- `api/*` depends on `services/*` and `utils/*`, but not vice-versa.
- `services/*` depends on `models/*` and `utils/*`; `models/*` depends only on `config/*` and `core/*`.
- `core/*` and `config/*` are leaf dependencies.

---

## Function Inventory Reconciliation

- AST-derived inventory reports 40 top-level functions, 4 class methods, and 12 nested helpers (total 56).
- The analysis document summarizes 48 functions, likely counting selected nested helpers as functions while excluding some internal ones.
- This matrix now includes all top-level functions and class methods; nested helpers are mapped implicitly within their parent modules (e.g., `monitor_progress` inside `download_audio_with_ffmpeg`, chunk transcription helpers inside `transcribe_with_incremental_output`).

---

## Critical Preservation Requirements

Global Variables (keep names and semantics):
- `whisper_models`, `faster_whisper_models`, `diarization_pipeline`
- `model_lock`, `faster_whisper_lock`, `diarization_lock`
- `executor` (ThreadPoolExecutor)
- `last_progress_update`
- Cache directories and flags: `CACHE_DIR`, `DOWNLOAD_CACHE_DIR`, `CACHE_ENABLED`, `CAPTURE_DIR`, `CAPTURES`

Async Functions:
- Preserve async status for endpoints, websocket handlers, download/transcription functions
- Maintain `asyncio.create_subprocess_exec` usage where present
- Any function receiving `WebSocket` must retain identical send semantics and state checks

Thread Safety:
- Retain double-check locking for model loading
- Maintain lock acquisition/release order and scope
- Keep model caches as singletons

WebSocket State:
- Always check `WebSocketState.CONNECTED` before sending
- Preserve error handling and `WebSocketDisconnect` flows

Caching:
- Keep SHA256 content/URL hash strategy unchanged
- Preserve on-disk file layout and metadata schemas

---

## HTML/CSS Extraction Plan

Current:
- Embedded HTML string with inline `<style>` in `app.py`

Target:
- `static/index.html` containing the full UI markup
- `static/css/stylesheet.css` extracted from inline styles
- Update GET `/` route (`get_home`) to read and cache `index.html` from disk (cache in memory as before)
- Mount static files for CSS at `/static/css/stylesheet.css`

Serving Behavior:
- Preserve cached HTML serving (in-memory cache remains)
- Fallback to disk read if cache missing
- No changes to endpoint path or response type

Risk & Mitigation:
- Risk: minor path issues → add existence checks and logs
- Validate by manual UI open and WebSocket connect test

---

## Notes on Testing & Verification

- Use existing API and WebSocket flows to verify equivalence
- Compare responses (status, progress, chunks) across pre- and post-refactor
- Validate cache hits/misses, diarization on/off, Deepgram fallback
- Exercise `first60` capture mode and download fallbacks (yt-dlp → ffmpeg)

---

## Conclusion

This modular architecture isolates concerns, reduces coupling, and preserves runtime behavior. It sets up Phase B implementation with clear boundaries, minimized risk, and reversibility.