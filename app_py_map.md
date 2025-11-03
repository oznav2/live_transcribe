# app.py Function & Method Map (Signatures + Line Anchors)

Purpose: Help LLMs locate exact function/method syntax in `app.py` (3619 lines)
without re-reading the file. Use this map for quick navigation and grep.

How to use
- Find the entry by name; jump to `app.py:<line>`.
- When lines shift, re-run a quick scan and update anchors in this file.

Top-level functions
- 158 async lifespan(app)
- 241 async safe_ws_send(websocket, data)
- 359 def init_capture_dir()
- 366 def init_download_cache_dir()
- 387 def get_url_hash(url)
- 392 def get_cached_download(url)
- 417 def save_download_to_cache(url, audio_file)
- 452 def load_model(model_name)
- 553 def get_diarization_pipeline()
- 591 async transcribe_with_diarization(model, model_config, audio_file, language, websocket, model_name)
- 833 def should_use_ytdlp(url)
- 866 async download_audio_with_ffmpeg(url, format, duration, websocket, use_cache)
- 1159 async download_with_fallback(url, language, format, websocket, use_cache)
- 1225 async download_audio_with_ytdlp_async(url, language, format, websocket, use_cache)
- 1460 def get_audio_duration_seconds(audio_path)
- 1483 def calculate_progress_metrics(audio_duration, elapsed_time, processed_chunks, total_chunks)
- 1531 def split_audio_for_incremental(audio_path, chunk_seconds, overlap_seconds)
- 1578 def split_audio_into_chunks(audio_path, chunk_seconds, overlap_seconds)
- 1626 async transcribe_with_incremental_output(model, model_config, audio_file, language, websocket, model_name, chunk_seconds)
- 1977 def transcribe_chunk(model_config, model, chunk_path, language)
- 2033 def init_cache_dir()
- 2057 def generate_cache_key(audio_data, sample_rate, channels)
- 2065 def get_cached_audio(cache_key)
- 2077 def save_to_cache(cache_key, audio_path)
- 2099 def format_duration(seconds)
- 2114 def format_view_count(count)
- 2129 def is_youtube_url(url)
- 2144 async get_youtube_metadata(url)
- 2335 async transcribe_audio_stream(websocket, processor)
- 2502 async transcribe_vod_with_deepgram(websocket, url, language)
- 2781 async transcribe_with_deepgram(websocket, url, language)
- 3068 async get_home()  # get /
- 3074 async websocket_transcribe(websocket)  # websocket /ws/transcribe
- 3419 async health_check()  # get /health
- 3429 async get_video_info(request)  # post /api/video-info
- 3485 async gpu_diagnostics()  # get /gpu
- 3518 async cache_stats()  # get /api/cache/stats
- 3538 async clear_cache()  # post /api/cache/clear
- 3555 async download_cache_stats()  # get /api/download-cache/stats
- 3584 async clear_download_cache()  # post /api/download-cache/clear

Class: AudioStreamProcessor
- 2220 def __init__(self, url, language, model_name)
- 2229 def start_ffmpeg_stream(self)
- 2258 def read_audio_chunks(self)
- 2321 def stop(self)

Nested helpers (selected)
- 944 async download_audio_with_ffmpeg.monitor_progress()
- 960 def download_audio_with_ffmpeg.monitor_progress.read_progress_file()
- 1671 def transcribe_with_incremental_output.run_fw_transcription()
- 1728 def transcribe_with_incremental_output.run_transcription()
- 1831 def transcribe_with_incremental_output.transcribe_fw_chunk()
- 1866 def transcribe_with_incremental_output.transcribe_openai_chunk()
- 2610 def transcribe_vod_with_deepgram.read_audio_file()
- 2794 def transcribe_with_deepgram.extract_deepgram_transcript(message)
- 2845 def transcribe_with_deepgram.on_message()
- 2874 def transcribe_with_deepgram.on_close()
- 2887 def transcribe_with_deepgram.on_error()
- 3238 def websocket_transcribe.read_capture_file()

Notes
- WebSocket helpers will be centralized in `utils/websocket_helpers.py` in refactor.
- If helpers are extracted, this map still preserves original `app.py` anchors.