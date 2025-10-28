#!/usr/bin/env python3
"""
Example Python client for Live Audio Stream Transcription
Demonstrates how to use the WebSocket API programmatically
"""
import asyncio
import websockets
import json
import sys

async def transcribe_stream(url: str, language: str = None):
    """
    Transcribe an audio/video stream URL
    
    Args:
        url: Audio or video stream URL (m3u8, mp4, mp3, etc.)
        language: Optional language code (e.g., 'en', 'es', 'fr')
    """
    # WebSocket endpoint
    ws_url = "ws://localhost:8000/ws/transcribe"
    
    print(f"Connecting to {ws_url}...")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✓ Connected to transcription service")
            
            # Send transcription request
            request = {
                "url": url,
                "language": language
            }
            
            print(f"\nStarting transcription for: {url}")
            if language:
                print(f"Language: {language}")
            print("\n" + "="*60)
            
            await websocket.send(json.dumps(request))
            
            # Receive and process transcription results
            full_transcription = []
            
            async for message in websocket:
                data = json.loads(message)
                
                if data.get("type") == "status":
                    print(f"Status: {data['message']}")
                    
                elif data.get("type") == "transcription":
                    text = data['text']
                    detected_lang = data.get('language', 'unknown')
                    
                    print(f"\n[{detected_lang}] {text}")
                    full_transcription.append(text)
                    
                elif data.get("type") == "complete":
                    print("\n" + "="*60)
                    print("✓ Transcription complete!")
                    break
                    
                elif data.get("error"):
                    print(f"\n❌ Error: {data['error']}")
                    break
            
            # Print full transcription
            if full_transcription:
                print("\n" + "="*60)
                print("FULL TRANSCRIPTION:")
                print("="*60)
                print(" ".join(full_transcription))
                print("="*60)
                print(f"\nTotal segments: {len(full_transcription)}")
                print(f"Total words: {len(' '.join(full_transcription).split())}")
            
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused. Is the server running on port 8000?")
        print("   Start the server with: docker-compose up")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


async def main():
    """Main function"""
    print("="*60)
    print("Live Audio Stream Transcription - Python Client")
    print("="*60)
    print()
    
    # Example URLs - replace with your own
    example_urls = {
        "1": {
            "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
            "name": "Direct MP3 Audio",
            "language": None
        },
        "2": {
            "url": "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
            "name": "HLS Stream (m3u8)",
            "language": None
        },
    }
    
    # Interactive selection or command-line argument
    if len(sys.argv) > 1:
        # URL provided as command-line argument
        url = sys.argv[1]
        language = sys.argv[2] if len(sys.argv) > 2 else None
        await transcribe_stream(url, language)
    else:
        # Interactive mode
        print("Select an option:")
        print("1. Test with example MP3 audio")
        print("2. Test with example m3u8 stream")
        print("3. Enter custom URL")
        print()
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice in example_urls:
            example = example_urls[choice]
            print(f"\nUsing: {example['name']}")
            await transcribe_stream(example['url'], example['language'])
        elif choice == "3":
            url = input("\nEnter audio/video URL: ").strip()
            language = input("Enter language code (optional, press Enter to skip): ").strip() or None
            await transcribe_stream(url, language)
        else:
            print("Invalid choice")
            sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
