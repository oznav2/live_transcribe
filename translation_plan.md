Translation Feature Implementation Plan
Overview
Add a translation button that appears after transcription completes, allowing users to translate the transcribed text between Hebrew and English using OpenAI GPT-4/5.
Architecture
New service: services/translation.py - OpenAI translation logic
New cache utilities: utils/translation_cache.py - Save transcriptions and translations as text files
REST endpoint: /api/translate - Handle translation requests
WebSocket enhancement: Include detected_language in complete messages
Frontend: Translate button with dynamic text + translation display
Implementation Steps
Phase 1: Backend Translation Service
Create services/translation.py:
OpenAI client initialization with API key sanitization
translate_text() function with context-aware prompt
Language detection logic (Hebrew↔English)
Dynamic target language selection
Error handling for missing API keys
Phase 2: Translation Caching
Create utils/translation_cache.py:
save_transcription_to_file() - Save original text as .txt
save_translation_to_file() - Save translated text as .txt
Use SHA256 hash of URL for cache folder naming
Store in DOWNLOAD_CACHE_DIR with timestamp
Phase 3: Configuration
Modify config/settings.py:
Add OPENAI_API_KEY with sanitize_token()
Add OPENAI_BASE_URL (default: "https://api.openai.com/v1")
Update .env:
Add OPENAI_API_KEY=your-key-here
Add OPENAI_BASE_URL=https://api.openai.com/v1
Phase 4: REST API Endpoint
Modify api/routes.py:
Add TranslationRequest Pydantic model (text, language, url, video_title)
Create /api/translate POST endpoint
Call translation service
Save both texts to cache
Return translated text with detected languages
Phase 5: WebSocket Enhancement
Modify api/websocket.py:
Track detected_language throughout transcription
Include detected_language in all complete messages (lines 246, 312, 352)
Store video URL for cache key generation
Phase 6: Frontend - Translate Button
Modify static/index.html (lines 166-177):
Add translate button after download button
Icon: <i class="fas fa-language"></i> (multilingual icon)
Dynamic button text: "תרגם לעברית" or "Translate to English"
Initially hidden, show only on complete message
RTL support for Hebrew text
Phase 7: Frontend - Translation Display
Modify static/index.html:
Add translation result display area below transcription
Toggle visibility when translation completes
Show source and target languages
Show translation progress during API call
Phase 8: Frontend - JavaScript Logic
Modify static/index.html WebSocket handler:
Capture detected_language from complete messages (line 829)
Show/hide translate button based on language
Set button text dynamically
Handle translate button click → POST to /api/translate
Display translation result with proper RTL formatting
Handle errors gracefully (missing API key, network errors)
Phase 9: Dependencies
Update requirements.txt:
Add openai>=1.50.0
Key Technical Decisions
Cache Strategy: Use same SHA256 URL hashing as existing download cache
File Naming:
Transcription: {url_hash}_transcription_{timestamp}.txt
Translation: {url_hash}_translation_{timestamp}.txt
Language Detection: Use detected_language from transcription (already available)
Button Visibility: Only show when transcription completes and connection closes
Translation Direction:
English → Hebrew: Button shows "תרגם לעברית"
Hebrew/Other → English: Button shows "Translate to English"
Error Handling: Graceful degradation if OpenAI key not configured
Prompt Engineering: Context-aware prompt includes video title if available
Files to Create
services/translation.py (new)
utils/translation_cache.py (new)
Files to Modify
config/settings.py (add OPENAI settings)
api/routes.py (add /api/translate endpoint)
api/websocket.py (include detected_language in complete messages)
static/index.html (UI + JavaScript logic)
.env (add OPENAI variables)
requirements.txt (add openai package)
Safety Guarantees
Translation is opt-in (user clicks button)
Existing transcription functionality unchanged
No breaking changes to WebSocket protocol
Backward compatible (works without OpenAI key configured)