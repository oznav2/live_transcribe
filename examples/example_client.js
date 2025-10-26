#!/usr/bin/env node
/**
 * Example Node.js/JavaScript client for Live Audio Stream Transcription
 * Demonstrates how to use the WebSocket API programmatically
 * 
 * Installation:
 *   npm install ws
 * 
 * Usage:
 *   node example_client.js <url> [language]
 *   node example_client.js https://example.com/audio.mp3
 *   node example_client.js https://example.com/video.m3u8 en
 */

const WebSocket = require('ws');

// Configuration
const WS_URL = 'ws://localhost:8000/ws/transcribe';

/**
 * Transcribe an audio/video stream URL
 */
async function transcribeStream(url, language = null) {
    return new Promise((resolve, reject) => {
        console.log(`\nConnecting to ${WS_URL}...`);
        
        const ws = new WebSocket(WS_URL);
        const fullTranscription = [];
        
        // Connection opened
        ws.on('open', () => {
            console.log('✓ Connected to transcription service\n');
            
            // Send transcription request
            const request = {
                url: url,
                language: language
            };
            
            console.log(`Starting transcription for: ${url}`);
            if (language) {
                console.log(`Language: ${language}`);
            }
            console.log('='.repeat(60));
            
            ws.send(JSON.stringify(request));
        });
        
        // Receive messages
        ws.on('message', (data) => {
            const message = JSON.parse(data.toString());
            
            if (message.type === 'status') {
                console.log(`Status: ${message.message}`);
            }
            else if (message.type === 'transcription') {
                const text = message.text;
                const detectedLang = message.language || 'unknown';
                
                console.log(`\n[${detectedLang}] ${text}`);
                fullTranscription.push(text);
            }
            else if (message.type === 'complete') {
                console.log('\n' + '='.repeat(60));
                console.log('✓ Transcription complete!');
                
                if (fullTranscription.length > 0) {
                    console.log('\n' + '='.repeat(60));
                    console.log('FULL TRANSCRIPTION:');
                    console.log('='.repeat(60));
                    console.log(fullTranscription.join(' '));
                    console.log('='.repeat(60));
                    console.log(`\nTotal segments: ${fullTranscription.length}`);
                    console.log(`Total words: ${fullTranscription.join(' ').split(/\s+/).length}`);
                }
                
                ws.close();
                resolve(fullTranscription);
            }
            else if (message.error) {
                console.error(`\n❌ Error: ${message.error}`);
                ws.close();
                reject(new Error(message.error));
            }
        });
        
        // Handle errors
        ws.on('error', (error) => {
            console.error(`❌ WebSocket error: ${error.message}`);
            if (error.code === 'ECONNREFUSED') {
                console.error('Connection refused. Is the server running on port 8000?');
                console.error('Start the server with: docker-compose up');
            }
            reject(error);
        });
        
        // Handle close
        ws.on('close', () => {
            console.log('\nConnection closed');
        });
    });
}

/**
 * Main function
 */
async function main() {
    console.log('='.repeat(60));
    console.log('Live Audio Stream Transcription - JavaScript Client');
    console.log('='.repeat(60));
    
    // Parse command line arguments
    const args = process.argv.slice(2);
    
    if (args.length === 0) {
        console.log('\nUsage:');
        console.log('  node example_client.js <url> [language]');
        console.log('\nExamples:');
        console.log('  node example_client.js https://example.com/audio.mp3');
        console.log('  node example_client.js https://example.com/stream.m3u8 en');
        console.log('  node example_client.js https://example.com/video.mp4 es');
        console.log('\nSupported languages:');
        console.log('  en (English), es (Spanish), fr (French), de (German),');
        console.log('  it (Italian), pt (Portuguese), ru (Russian), ja (Japanese),');
        console.log('  ko (Korean), zh (Chinese), ar (Arabic), hi (Hindi), etc.');
        console.log('\nNote: Language is optional. Auto-detection will be used if not specified.\n');
        process.exit(1);
    }
    
    const url = args[0];
    const language = args[1] || null;
    
    try {
        await transcribeStream(url, language);
        console.log('\n✓ Done!\n');
        process.exit(0);
    } catch (error) {
        console.error(`\n❌ Failed: ${error.message}\n`);
        process.exit(1);
    }
}

// Run if executed directly
if (require.main === module) {
    main().catch((error) => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
}

// Export for use as module
module.exports = { transcribeStream };
