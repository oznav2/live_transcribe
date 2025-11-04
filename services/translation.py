"""Translation service using OpenAI GPT-4/5 for context-aware translations."""
import logging
import os
import re
from typing import Optional, Tuple, List, AsyncGenerator

from utils.validators import sanitize_token

logger = logging.getLogger(__name__)

# OpenAI client - import conditionally
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    AsyncOpenAI = None


def get_openai_client() -> Optional[OpenAI]:
    """Initialize synchronous OpenAI client with API key from environment."""
    if not OPENAI_AVAILABLE:
        logger.error("OpenAI package not installed. Install with: pip install openai>=1.50.0")
        return None

    # Get and sanitize API key
    api_key = sanitize_token(os.getenv('OPENAI_API_KEY', ''))
    if not api_key:
        logger.warning("OPENAI_API_KEY not configured in .env file")
        return None

    # Get base URL (optional, defaults to OpenAI)
    base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return None


async def get_async_openai_client() -> Optional[AsyncOpenAI]:
    """Initialize asynchronous OpenAI client with API key from environment."""
    if not OPENAI_AVAILABLE:
        logger.error("OpenAI package not installed. Install with: pip install openai>=1.50.0")
        return None

    # Get and sanitize API key
    api_key = sanitize_token(os.getenv('OPENAI_API_KEY', ''))
    if not api_key:
        logger.warning("OPENAI_API_KEY not configured in .env file")
        return None

    # Get base URL (optional, defaults to OpenAI)
    base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

    try:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize async OpenAI client: {e}")
        return None


def detect_translation_direction(source_language: str) -> Tuple[str, str, str]:
    """
    Determine translation direction based on source language.

    Args:
        source_language: Detected language code (e.g., 'en', 'he', 'en-US', 'iw')

    Returns:
        Tuple of (target_language_code, target_language_name, button_text)
    """
    # Normalize language code
    lang_lower = source_language.lower()

    # Check if Hebrew (he, iw, hebrew, עברית)
    if any(code in lang_lower for code in ['he', 'iw', 'hebrew', 'עברית']):
        return ('en', 'English', 'Translate to English')

    # For English or other languages, translate to Hebrew
    return ('he', 'Hebrew', 'תרגם לעברית')


def build_translation_prompt(
    text: str,
    target_language: str,
    video_title: Optional[str] = None,
    source_language: Optional[str] = None
) -> str:
    """
    Build a context-aware translation prompt for OpenAI.

    Args:
        text: Text to translate
        target_language: Target language name (e.g., 'Hebrew', 'English')
        video_title: Optional video title for context
        source_language: Optional source language for context

    Returns:
        Formatted prompt string
    """
    # Base instruction
    prompt_parts = [
        f"You are a professional translator specializing in accurate, context-aware translations.",
        f"",
        f"Task: Translate the following text to {target_language}.",
        f""
    ]

    # Add context if available
    if video_title:
        prompt_parts.append(f"Context: This is a transcription from a video titled '{video_title}'.")

    if source_language:
        prompt_parts.append(f"Source language: {source_language}")

    # Add translation requirements
    prompt_parts.extend([
        "",
        "Requirements:",
        "1. Preserve the exact meaning and intent of the original text",
        "2. Maintain the tone and style (formal/informal/technical)",
        "3. Keep proper nouns and technical terms when appropriate",
        "4. Ensure the translation reads naturally in the target language",
        "5. Do not add explanations or notes - provide only the translation",
        "",
        "Text to translate:",
        "---",
        text,
        "---",
        "",
        f"Provide only the {target_language} translation below:"
    ])

    return "\n".join(prompt_parts)


async def translate_text(
    text: str,
    source_language: str,
    video_title: Optional[str] = None,
    model: str = "gpt-4"
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Translate text using OpenAI GPT model.

    Args:
        text: Text to translate
        source_language: Detected source language code
        video_title: Optional video title for context
        model: OpenAI model to use (default: gpt-4)

    Returns:
        Tuple of (translated_text, target_language_code, target_language_name)
        Returns (None, None, None) on failure
    """
    # Get OpenAI client
    client = get_openai_client()
    if not client:
        logger.error("OpenAI client not available - check API key configuration")
        return (None, None, None)

    # Determine translation direction
    target_lang_code, target_lang_name, _ = detect_translation_direction(source_language)

    # Build prompt
    prompt = build_translation_prompt(
        text=text,
        target_language=target_lang_name,
        video_title=video_title,
        source_language=source_language
    )

    try:
        logger.info(f"Starting translation: {source_language} → {target_lang_name} using {model}")

        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. Translate text to {target_lang_name} accurately while preserving meaning and context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more consistent translations
            max_tokens=4096   # Allow for long translations
        )

        # Extract translated text
        translated_text = response.choices[0].message.content.strip()

        logger.info(f"Translation completed successfully ({len(translated_text)} chars)")
        return (translated_text, target_lang_code, target_lang_name)

    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        return (None, None, None)


def get_translation_button_text(source_language: str) -> str:
    """
    Get the appropriate button text based on source language.

    Args:
        source_language: Detected language code

    Returns:
        Button text in appropriate language
    """
    _, _, button_text = detect_translation_direction(source_language)
    return button_text


def split_text_into_chunks(text: str, max_words: int = 100) -> List[str]:
    """
    Split text into chunks of approximately max_words each.
    Tries to split at sentence boundaries when possible.

    Args:
        text: Text to split
        max_words: Maximum words per chunk (default: 100)

    Returns:
        List of text chunks
    """
    # Split text into sentences (basic sentence detection)
    sentence_endings = r'[.!?]\s+'
    sentences = re.split(sentence_endings, text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Count words in this sentence
        words = sentence.split()
        word_count = len(words)

        # If adding this sentence would exceed max_words, start a new chunk
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count

    # Add remaining text as final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


async def translate_text_chunked(
    text: str,
    source_language: str,
    video_title: Optional[str] = None,
    model: str = "gpt-4",
    chunk_size: int = 100
) -> AsyncGenerator[Tuple[str, int, int, str, str], None]:
    """
    Translate text in chunks, yielding progress updates.

    Args:
        text: Text to translate
        source_language: Detected source language code
        video_title: Optional video title for context
        model: OpenAI model to use (default: gpt-4)
        chunk_size: Words per chunk (default: 100)

    Yields:
        Tuple of (translated_chunk, chunk_index, total_chunks, target_lang_code, target_lang_name)
    """
    # Get async OpenAI client
    client = await get_async_openai_client()
    if not client:
        logger.error("OpenAI client not available - check API key configuration")
        return

    # Determine translation direction
    target_lang_code, target_lang_name, _ = detect_translation_direction(source_language)

    # Split text into chunks
    chunks = split_text_into_chunks(text, max_words=chunk_size)
    total_chunks = len(chunks)

    logger.info(f"Translating {len(text)} chars in {total_chunks} chunks ({chunk_size} words each)")

    # Translate each chunk
    for i, chunk in enumerate(chunks):
        try:
            # Build prompt for this chunk
            prompt = build_translation_prompt(
                text=chunk,
                target_language=target_lang_name,
                video_title=video_title if i == 0 else None,  # Only include title in first chunk
                source_language=source_language
            )

            # Call OpenAI API asynchronously (allows event loop to send WebSocket messages)
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional translator. Translate text to {target_lang_name} accurately while preserving meaning and context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2048
            )

            # Extract translated chunk
            translated_chunk = response.choices[0].message.content.strip()

            # Yield progress update
            yield (translated_chunk, i, total_chunks, target_lang_code, target_lang_name)

            logger.debug(f"Translated chunk {i+1}/{total_chunks}")

        except Exception as e:
            logger.error(f"Failed to translate chunk {i+1}/{total_chunks}: {e}")
            # Yield empty string to indicate error but continue
            yield ("", i, total_chunks, target_lang_code, target_lang_name)
