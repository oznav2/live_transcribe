"""Hebrew summarization service using OpenAI GPT-4o-mini."""
import logging
import os
from typing import Optional, AsyncGenerator, Tuple

from utils.validators import sanitize_token

logger = logging.getLogger(__name__)

# OpenAI client - import conditionally
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


def get_openai_client() -> Optional[OpenAI]:
    """Initialize OpenAI client with API key from environment."""
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


def build_summary_prompt(
    text: str,
    video_title: Optional[str] = None,
    source_language: Optional[str] = None
) -> str:
    """
    Build a context-aware Hebrew summarization prompt for OpenAI.

    Args:
        text: Text to summarize
        video_title: Optional video title for context
        source_language: Optional source language for context

    Returns:
        Formatted prompt string
    """
    # Base instruction - emphasize Hebrew output and single paragraph
    prompt_parts = [
        "You are a professional content summarizer specializing in creating concise, context-aware summaries in Hebrew.",
        "",
        "Task: Summarize the following text into a single paragraph in Hebrew (עברית).",
        ""
    ]

    # Add context if available
    if video_title:
        prompt_parts.append(f"Context: This is a transcription from a video titled '{video_title}'.")

    if source_language:
        prompt_parts.append(f"Source language: {source_language}")

    # Add summarization requirements
    prompt_parts.extend([
        "",
        "CRITICAL Requirements:",
        "1. Output MUST be in Hebrew (עברית) only",
        "2. Output MUST be a single paragraph (no multiple paragraphs or bullet points)",
        "3. DO NOT include the video title in your response",
        "4. DO NOT include any prefixes like 'בוידאו', 'מעבד את הטקסט', or any introductory phrases",
        "5. DO NOT include colons (:) at the beginning",
        "6. Start DIRECTLY with the summary content",
        "7. Capture the main ideas, key points, and overall context",
        "8. Preserve important names, terms, and concepts in their original language",
        "9. Keep the summary concise but comprehensive (approximately 3-5 sentences)",
        "10. Maintain natural Hebrew language flow and proper grammar",
        "",
        "Text to summarize:",
        "---",
        text,
        "---",
        "",
        "Provide ONLY the Hebrew summary paragraph (no titles, no prefixes, no metadata):"
    ])

    return "\n".join(prompt_parts)


async def generate_hebrew_summary(
    text: str,
    source_language: str,
    video_title: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> Optional[str]:
    """
    Generate a Hebrew summary of the text using OpenAI GPT model.

    Args:
        text: Text to summarize
        source_language: Detected source language code
        video_title: Optional video title for context
        model: OpenAI model to use (default: gpt-4o-mini)

    Returns:
        Hebrew summary as single paragraph, or None on failure
    """
    # Get OpenAI client
    client = get_openai_client()
    if not client:
        logger.error("OpenAI client not available - check API key configuration")
        return None

    # Build prompt
    prompt = build_summary_prompt(
        text=text,
        video_title=video_title,
        source_language=source_language
    )

    try:
        logger.info(f"Starting Hebrew summarization using {model}")

        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional summarizer. Create concise Hebrew summaries that capture the essence of the content in a single paragraph."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more consistent summaries
            max_tokens=2048   # Allow for comprehensive summary
        )

        # Extract summary
        summary = response.choices[0].message.content.strip()

        logger.info(f"Hebrew summary generated successfully ({len(summary)} chars)")
        return summary

    except Exception as e:
        logger.error(f"Hebrew summarization failed: {e}", exc_info=True)
        return None


async def generate_hebrew_summary_chunked(
    text: str,
    source_language: str,
    video_title: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    Generate Hebrew summary with progress updates.

    For summaries, we don't actually chunk the output (summary is always single paragraph),
    but we provide progress updates during the API call for better UX.

    Args:
        text: Text to summarize
        source_language: Detected source language code
        video_title: Optional video title for context
        model: OpenAI model to use (default: gpt-4o-mini)

    Yields:
        Tuple of (summary_text, chunk_index, total_chunks)
        - First yield (0, 2): "Processing..." (50% progress)
        - Final yield (1, 2): actual summary (100% progress)
    """
    # Get OpenAI client
    client = get_openai_client()
    if not client:
        logger.error("OpenAI client not available - check API key configuration")
        return

    # Build prompt
    prompt = build_summary_prompt(
        text=text,
        video_title=video_title,
        source_language=source_language
    )

    try:
        logger.info(f"Starting Hebrew summarization ({len(text)} chars) using {model}")

        # Send initial progress update (50%)
        yield ("מעבד את הטקסט...", 0, 2)  # "Processing the text..." in Hebrew

        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional summarizer. Create concise Hebrew summaries that capture the essence of the content in a single paragraph."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2048
        )

        # Extract summary
        summary = response.choices[0].message.content.strip()

        logger.info(f"Hebrew summary generated successfully ({len(summary)} chars)")

        # Send final summary (100%)
        yield (summary, 1, 2)

    except Exception as e:
        logger.error(f"Hebrew summarization failed: {e}", exc_info=True)
        # Yield error message in Hebrew
        yield ("שגיאה ביצירת הסיכום", 1, 2)  # "Error creating summary" in Hebrew
