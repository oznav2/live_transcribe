# Implementation Plan: YouTube Video Metadata Display

## 📋 Requirements

### User Request:
1. Fetch video metadata when URL is YouTube
2. Display: Title, Channel Name, Video Length, View Count
3. Create new UI section under URL input
4. Bilingual support:
   - **Hebrew (RTL)** if Ivrit CT2 model selected OR Hebrew language detected
   - **English (LTR)** for all other cases

---

## 🔍 Analysis of Current Application

### Existing Infrastructure:
✅ **yt-dlp already installed** - Used for audio extraction  
✅ **Async pattern established** - All downloads are async  
✅ **WebSocket communication** - Real-time updates to UI  
✅ **Model detection** - Can detect Ivrit models  
✅ **Language detection** - Can detect Hebrew (he)  
✅ **RTL support** - Already exists in CSS  

### Key Files:
- **app.py** - Backend logic, yt-dlp integration
- **static/index.html** - Frontend UI, form structure

---

## 🎯 Implementation Strategy

### Phase 1: Backend - Metadata Extraction API

#### 1.1 Create Async Function to Extract Metadata
```python
async def get_youtube_metadata(url: str) -> Optional[dict]:
    """
    Extract metadata from YouTube video using yt-dlp
    
    Returns:
        dict with keys: title, channel, duration, view_count, thumbnail
        None if not YouTube or extraction fails
    """
```

**Implementation:**
- Use `yt-dlp --dump-json` to extract metadata (non-blocking)
- Run in executor to avoid blocking event loop
- Parse JSON output
- Handle errors gracefully (return None on failure)
- Cache metadata to avoid repeated extraction

**yt-dlp Command:**
```bash
yt-dlp --dump-json --no-playlist --skip-download <URL>
```

**Output JSON Fields:**
```json
{
  "title": "Video Title",
  "uploader": "Channel Name", 
  "duration": 1234,  // seconds
  "view_count": 123456,
  "thumbnail": "https://..."
}
```

#### 1.2 Create API Endpoint
```python
@app.post("/api/video-info")
async def get_video_info(url: HttpUrl):
    """
    REST endpoint to fetch video metadata
    Returns JSON with video info or error
    """
```

**Response Format:**
```json
{
  "success": true,
  "data": {
    "title": "Video Title",
    "channel": "Channel Name",
    "duration_seconds": 1234,
    "duration_formatted": "20:34",
    "view_count": 123456,
    "view_count_formatted": "123K",
    "thumbnail": "https://...",
    "is_youtube": true
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Not a YouTube URL" | "Failed to fetch metadata"
}
```

---

### Phase 2: Frontend - UI Section

#### 2.1 HTML Structure
Insert after URL input (line 629):

```html
<!-- Video Information Section -->
<div id="videoInfoSection" class="video-info-section" style="display: none;">
    <div class="video-info-header">
        <i class="fab fa-youtube"></i>
        <span id="videoInfoTitle">Video Information</span>
    </div>
    
    <div class="video-info-content" id="videoInfoContent">
        <!-- Thumbnail -->
        <div class="video-thumbnail">
            <img id="videoThumbnail" src="" alt="Video thumbnail">
        </div>
        
        <!-- Video Details -->
        <div class="video-details">
            <div class="video-title" id="videoTitle">Loading...</div>
            
            <div class="video-meta">
                <div class="meta-item">
                    <i class="fas fa-user"></i>
                    <span id="videoChannel">Channel</span>
                </div>
                <div class="meta-item">
                    <i class="fas fa-clock"></i>
                    <span id="videoDuration">Duration</span>
                </div>
                <div class="meta-item">
                    <i class="fas fa-eye"></i>
                    <span id="videoViews">Views</span>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Loading State -->
    <div id="videoInfoLoading" class="video-info-loading" style="display: none;">
        <i class="fas fa-spinner fa-spin"></i>
        <span id="videoInfoLoadingText">Fetching video information...</span>
    </div>
    
    <!-- Error State -->
    <div id="videoInfoError" class="video-info-error" style="display: none;">
        <i class="fas fa-exclamation-triangle"></i>
        <span id="videoInfoErrorText">Could not fetch video information</span>
    </div>
</div>
```

#### 2.2 CSS Styles
```css
.video-info-section {
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 16px;
    margin-top: 16px;
    animation: slideIn 0.3s ease-out;
}

.video-info-section.rtl {
    direction: rtl;
    text-align: right;
}

.video-info-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    color: #0066cc;
    font-weight: 600;
}

.video-info-content {
    display: flex;
    gap: 16px;
}

.video-thumbnail {
    flex-shrink: 0;
    width: 160px;
    height: 90px;
    border-radius: 8px;
    overflow: hidden;
    background: #2a2a2a;
}

.video-thumbnail img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.video-details {
    flex: 1;
    min-width: 0;
}

.video-title {
    font-size: 15px;
    font-weight: 600;
    color: #fff;
    margin-bottom: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.video-meta {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.meta-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: #999;
}

.meta-item i {
    width: 16px;
    color: #0066cc;
}

.video-info-loading,
.video-info-error {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px;
    border-radius: 8px;
    font-size: 13px;
}

.video-info-loading {
    background: #1a3a5a;
    color: #0099ff;
}

.video-info-error {
    background: #3a1a1a;
    color: #ff6666;
}
```

#### 2.3 JavaScript Logic
```javascript
// Debounce timer for URL input
let urlDebounceTimer = null;

// Listen to URL input changes
document.getElementById('urlInput').addEventListener('input', (e) => {
    const url = e.target.value.trim();
    
    // Clear previous timer
    clearTimeout(urlDebounceTimer);
    
    // Hide video info if URL is empty
    if (!url) {
        hideVideoInfo();
        return;
    }
    
    // Check if URL is YouTube
    if (!isYouTubeUrl(url)) {
        hideVideoInfo();
        return;
    }
    
    // Debounce: Wait 500ms after user stops typing
    urlDebounceTimer = setTimeout(() => {
        fetchVideoInfo(url);
    }, 500);
});

// Listen to model selection changes for language preference
document.getElementById('modelSelect').addEventListener('change', () => {
    // Re-render video info with new language if already loaded
    if (currentVideoInfo) {
        renderVideoInfo(currentVideoInfo);
    }
});

function isYouTubeUrl(url) {
    const youtubePatterns = [
        /youtube\.com\/watch\?v=/,
        /youtu\.be\//,
        /youtube\.com\/embed\//,
        /youtube\.com\/v\//,
        /m\.youtube\.com/
    ];
    return youtubePatterns.some(pattern => pattern.test(url));
}

async function fetchVideoInfo(url) {
    const section = document.getElementById('videoInfoSection');
    const loading = document.getElementById('videoInfoLoading');
    const content = document.getElementById('videoInfoContent');
    const error = document.getElementById('videoInfoError');
    
    // Show section and loading state
    section.style.display = 'block';
    loading.style.display = 'flex';
    content.style.display = 'none';
    error.style.display = 'none';
    
    try {
        const response = await fetch('/api/video-info', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url })
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentVideoInfo = result.data;
            renderVideoInfo(result.data);
        } else {
            showVideoInfoError(result.error);
        }
    } catch (err) {
        showVideoInfoError('Network error');
    }
}

function renderVideoInfo(data) {
    const section = document.getElementById('videoInfoSection');
    const loading = document.getElementById('videoInfoLoading');
    const content = document.getElementById('videoInfoContent');
    const error = document.getElementById('videoInfoError');
    
    // Determine language preference
    const selectedModel = document.getElementById('modelSelect').value;
    const isHebrewMode = selectedModel.includes('ivrit') || 
                         selectedModel.includes('hebrew');
    
    // Apply RTL if Hebrew
    if (isHebrewMode) {
        section.classList.add('rtl');
        updateHebrewLabels();
    } else {
        section.classList.remove('rtl');
        updateEnglishLabels();
    }
    
    // Update content
    document.getElementById('videoThumbnail').src = data.thumbnail;
    document.getElementById('videoTitle').textContent = data.title;
    document.getElementById('videoChannel').textContent = data.channel;
    document.getElementById('videoDuration').textContent = data.duration_formatted;
    document.getElementById('videoViews').textContent = data.view_count_formatted;
    
    // Show content
    loading.style.display = 'none';
    error.style.display = 'none';
    content.style.display = 'flex';
}

function updateHebrewLabels() {
    document.getElementById('videoInfoTitle').textContent = 'מידע על הסרטון';
    document.getElementById('videoInfoLoadingText').textContent = 'טוען מידע...';
    // Icons stay the same (universal)
}

function updateEnglishLabels() {
    document.getElementById('videoInfoTitle').textContent = 'Video Information';
    document.getElementById('videoInfoLoadingText').textContent = 'Fetching video information...';
}

function showVideoInfoError(message) {
    const section = document.getElementById('videoInfoSection');
    const loading = document.getElementById('videoInfoLoading');
    const content = document.getElementById('videoInfoContent');
    const error = document.getElementById('videoInfoError');
    
    document.getElementById('videoInfoErrorText').textContent = message;
    
    loading.style.display = 'none';
    content.style.display = 'none';
    error.style.display = 'flex';
}

function hideVideoInfo() {
    document.getElementById('videoInfoSection').style.display = 'none';
    currentVideoInfo = null;
}
```

---

## 🛡️ Error Handling & Edge Cases

### Potential Issues & Solutions:

#### 1. **Non-YouTube URLs**
- **Issue:** User enters non-YouTube URL
- **Solution:** Check URL pattern, don't show video info section

#### 2. **yt-dlp Extraction Fails**
- **Issue:** Network error, video unavailable, private video
- **Solution:** 
  - Catch exception in async function
  - Return None
  - Show error state in UI: "Could not fetch video information"
  - Don't block transcription process

#### 3. **Slow Metadata Extraction**
- **Issue:** yt-dlp takes time to extract metadata
- **Solution:**
  - Show loading spinner immediately
  - Run in executor (non-blocking)
  - Timeout after 10 seconds
  - Allow transcription to start even if metadata pending

#### 4. **Missing Fields**
- **Issue:** Some videos missing view_count or uploader
- **Solution:**
  - Use `.get()` with defaults
  - Display "N/A" or hide field if missing

#### 5. **Language Switching Mid-Session**
- **Issue:** User changes model after metadata loaded
- **Solution:**
  - Listen to model selection change
  - Re-render labels in new language
  - Keep data cached

#### 6. **URL Changes Rapidly**
- **Issue:** User types/pastes, triggers many requests
- **Solution:**
  - Debounce input (500ms delay)
  - Cancel previous request if new one starts
  - Only fetch when user stops typing

#### 7. **Blocking Event Loop**
- **Issue:** yt-dlp subprocess blocks
- **Solution:**
  - Use `asyncio.create_subprocess_exec`
  - Set timeout (10 seconds)
  - Run in executor if needed

#### 8. **Cache Invalidation**
- **Issue:** Same URL entered multiple times
- **Solution:**
  - Cache metadata by URL hash
  - TTL: 1 hour
  - Reuse cached data

---

## 🧪 Testing Strategy

### Test Cases:

#### 1. YouTube URL Detection
```
✅ https://www.youtube.com/watch?v=dQw4w9WgXcQ
✅ https://youtu.be/dQw4w9WgXcQ
✅ https://m.youtube.com/watch?v=dQw4w9WgXcQ
❌ https://vimeo.com/123456 (should not show video info)
❌ https://example.com/video.mp4 (should not show video info)
```

#### 2. Metadata Extraction
```
Test video: Public YouTube video
Expected: Title, channel, duration, views all displayed
```

#### 3. Hebrew Mode
```
Select model: "ivrit-ct2"
Expected: 
- RTL layout
- Hebrew labels: "מידע על הסרטון"
- Hebrew loading text
```

#### 4. English Mode
```
Select model: "whisper-v3-turbo"
Expected:
- LTR layout
- English labels: "Video Information"
```

#### 5. Error Handling
```
Test: Private YouTube video
Expected: Error message, transcription still works
```

#### 6. Empty URL
```
Clear URL input
Expected: Video info section hidden
```

---

## 🔒 Security Considerations

### 1. **URL Validation**
- Validate URL format on backend
- Only allow HTTP/HTTPS
- Prevent command injection in yt-dlp

### 2. **Rate Limiting**
- Limit metadata requests per IP
- Cache aggressively

### 3. **Timeout**
- Maximum 10 seconds for metadata extraction
- Don't let it block transcription

### 4. **Error Messages**
- Don't expose internal errors to user
- Generic error messages

---

## 📦 Implementation Checklist

### Backend (app.py):
- [ ] Add `get_youtube_metadata()` async function
- [ ] Add `/api/video-info` POST endpoint
- [ ] Add URL validation helper
- [ ] Add view count formatting helper
- [ ] Add duration formatting helper (seconds → MM:SS)
- [ ] Add error handling
- [ ] Add timeout (10s)
- [ ] Add metadata caching

### Frontend (index.html):
- [ ] Add video info HTML section
- [ ] Add CSS styles (LTR + RTL)
- [ ] Add JavaScript event listeners
- [ ] Add `isYouTubeUrl()` function
- [ ] Add `fetchVideoInfo()` function
- [ ] Add `renderVideoInfo()` function
- [ ] Add Hebrew/English label switching
- [ ] Add error state handling
- [ ] Add loading state
- [ ] Add debounce logic

### Testing:
- [ ] Test with YouTube URLs
- [ ] Test with non-YouTube URLs
- [ ] Test Hebrew mode (Ivrit model)
- [ ] Test English mode
- [ ] Test error cases (private video, network error)
- [ ] Test rapid URL changes (debounce)
- [ ] Test empty URL (hide section)
- [ ] Test model switching (language change)

---

## 🎯 Success Criteria

1. ✅ Video metadata appears within 2 seconds of URL input
2. ✅ Metadata displays: Title, Channel, Duration, Views, Thumbnail
3. ✅ Hebrew mode: RTL layout + Hebrew labels
4. ✅ English mode: LTR layout + English labels
5. ✅ Graceful error handling (doesn't block transcription)
6. ✅ No blocking of async event loop
7. ✅ Responsive design (works on mobile)
8. ✅ Smooth animations (slideIn)

---

## 🚀 Deployment Steps

1. Implement backend function
2. Implement backend API endpoint
3. Test API with curl/Postman
4. Implement frontend HTML
5. Implement frontend CSS
6. Implement frontend JavaScript
7. Test integration
8. Test error cases
9. Commit and push
10. Create PR with documentation

---

## 📊 Estimated Impact

- **User Experience:** ⬆️⬆️⬆️ (High improvement)
- **Code Complexity:** ⬆️ (Medium increase, manageable)
- **Performance:** ↔️ (Neutral - async, cached)
- **Maintenance:** ⬆️ (Slight increase - new feature)
- **Breaking Changes:** None

---

**Status:** Planning Complete ✅  
**Next Step:** Implementation  
**Risk Level:** Low (isolated feature, graceful degradation)
