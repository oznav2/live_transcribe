"""Text deduplication utilities for removing duplicate sentences and word sequences."""
import logging
import re
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling multiple languages.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    # Simple sentence splitting that works for most languages
    # Splits on common sentence terminators followed by space/newline
    sentences = re.split(r'([.!?]+[\s\n]+)', text)

    # Recombine sentences with their terminators
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i]
        terminator = sentences[i + 1] if i + 1 < len(sentences) else ''
        combined = sentence + terminator
        if combined.strip():
            result.append(combined)

    # Handle last sentence if it doesn't have terminator
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1])

    return result


def remove_consecutive_duplicate_sentences(sentences: List[str]) -> List[str]:
    """
    Remove consecutive duplicate sentences.

    Args:
        sentences: List of sentences

    Returns:
        List with consecutive duplicates removed
    """
    if not sentences:
        return []

    result = [sentences[0]]
    duplicates_removed = 0

    for i in range(1, len(sentences)):
        # Normalize for comparison (strip, lowercase)
        current_normalized = sentences[i].strip().lower()
        previous_normalized = sentences[i - 1].strip().lower()

        if current_normalized != previous_normalized:
            result.append(sentences[i])
        else:
            duplicates_removed += 1

    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} consecutive duplicate sentences")

    return result


def remove_repeated_word_sequences(text: str, min_sequence_length: int = 5) -> str:
    """
    Remove repeated word sequences using sliding window approach.

    This handles cases like:
    "I'll use the drop-in replacement for Docker. And I'll use the drop-in replacement for Docker."

    Args:
        text: Text to clean
        min_sequence_length: Minimum number of words to consider a sequence (default: 5)

    Returns:
        Text with repeated sequences removed
    """
    words = text.split()

    if len(words) < min_sequence_length * 2:
        return text  # Too short to have meaningful duplicates

    # Track which positions to keep
    keep = [True] * len(words)
    sequences_removed = 0

    # Use sliding window to find repeated sequences
    for seq_len in range(min_sequence_length, len(words) // 2 + 1):
        i = 0
        while i < len(words) - seq_len:
            # Skip if this position is already marked for removal
            if not keep[i]:
                i += 1
                continue

            # Get current sequence
            sequence = words[i:i + seq_len]

            # Check if the same sequence appears immediately after
            next_start = i + seq_len
            if next_start + seq_len <= len(words):
                next_sequence = words[next_start:next_start + seq_len]

                # Normalize for comparison (case-insensitive)
                if [w.lower() for w in sequence] == [w.lower() for w in next_sequence]:
                    # Mark the duplicate sequence for removal
                    for j in range(next_start, next_start + seq_len):
                        keep[j] = False
                    sequences_removed += 1

                    # Move past the duplicate
                    i = next_start + seq_len
                    continue

            i += 1

    if sequences_removed > 0:
        logger.info(f"Removed {sequences_removed} repeated word sequences")

    # Reconstruct text from kept words
    result_words = [word for i, word in enumerate(words) if keep[i]]
    return ' '.join(result_words)


def remove_duplicate_text(text: str, aggressive: bool = False) -> Tuple[str, dict]:
    """
    Remove duplicate sentences and word sequences from text.

    This is the main entry point for text deduplication. It handles:
    1. Consecutive duplicate sentences
    2. Repeated word sequences within sentences

    Args:
        text: Text to clean
        aggressive: If True, use more aggressive deduplication (smaller sequence length)

    Returns:
        Tuple of (cleaned_text, stats_dict)
        stats_dict contains: original_length, cleaned_length, reduction_percent
    """
    if not text or not text.strip():
        return text, {"original_length": 0, "cleaned_length": 0, "reduction_percent": 0}

    original_length = len(text)

    try:
        logger.info(f"Starting text deduplication (original length: {original_length} chars)")

        # Step 1: Remove repeated word sequences
        # Use smaller sequence length if aggressive mode enabled
        min_seq_length = 3 if aggressive else 5
        cleaned = remove_repeated_word_sequences(text, min_sequence_length=min_seq_length)

        # Step 2: Split into sentences and remove consecutive duplicates
        sentences = split_into_sentences(cleaned)
        sentences = remove_consecutive_duplicate_sentences(sentences)
        cleaned = ''.join(sentences)

        # Calculate statistics
        cleaned_length = len(cleaned)
        reduction_percent = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0

        stats = {
            "original_length": original_length,
            "cleaned_length": cleaned_length,
            "reduction_percent": round(reduction_percent, 2)
        }

        if reduction_percent > 1:  # Only log if significant reduction
            logger.info(f"Deduplication complete: {original_length} → {cleaned_length} chars ({reduction_percent:.1f}% reduction)")
        else:
            logger.info("Deduplication complete: no significant duplicates found")

        return cleaned, stats

    except Exception as e:
        logger.error(f"Text deduplication failed: {e}", exc_info=True)
        # Return original text if cleaning fails (fail-safe)
        return text, {"original_length": original_length, "cleaned_length": original_length, "reduction_percent": 0, "error": str(e)}


class IncrementalDeduplicator:
    """
    Stateful deduplication for incremental chunk processing.
    
    Maintains context between chunks to detect duplicates that span chunk boundaries.
    Designed to be fast (O(n)), robust (falls back to original on error), and language-agnostic.
    """
    
    def __init__(self, context_window: int = 20, min_sequence_length: int = 5):
        """
        Initialize the incremental deduplicator.
        
        Args:
            context_window: Number of words to keep from previous chunk for overlap detection
            min_sequence_length: Minimum word sequence length to consider duplicate
        """
        self.context_window = context_window
        self.min_sequence_length = min_sequence_length
        self.previous_words: List[str] = []
        self.total_duplicates_removed = 0
        
    def process_chunk(self, chunk: str) -> str:
        """
        Process a single chunk with deduplication, maintaining state for next chunk.
        
        Args:
            chunk: Text chunk to process
            
        Returns:
            Deduplicated chunk text
        """
        if not chunk or not chunk.strip():
            return chunk
            
        try:
            # Split current chunk into words
            current_words = chunk.split()
            
            if not current_words:
                return chunk
            
            # Find overlap with previous chunk
            skip_count = 0
            if self.previous_words:
                skip_count = self._find_overlap(self.previous_words, current_words)
                
            # Remove overlapping words from current chunk
            if skip_count > 0:
                self.total_duplicates_removed += skip_count
                current_words = current_words[skip_count:]
                logger.info(f"Removed {skip_count} duplicate words at chunk boundary")
            
            # Apply intra-chunk deduplication
            if len(current_words) >= self.min_sequence_length * 2:
                current_words = self._remove_internal_duplicates(current_words)
            
            # Update context for next chunk (keep last N words)
            self.previous_words = current_words[-self.context_window:] if len(current_words) > self.context_window else current_words[:]
            
            # Reconstruct text
            result = ' '.join(current_words)
            return result
            
        except Exception as e:
            logger.error(f"Incremental deduplication failed: {e}", exc_info=True)
            # Fail-safe: return original chunk
            return chunk
    
    def _find_overlap(self, prev_words: List[str], curr_words: List[str]) -> int:
        """
        Find how many words at the start of curr_words overlap with end of prev_words.
        
        Returns:
            Number of words to skip from curr_words
        """
        max_overlap = min(len(prev_words), len(curr_words), self.context_window)
        
        # Try to find longest matching sequence
        for overlap_len in range(max_overlap, self.min_sequence_length - 1, -1):
            prev_segment = [w.lower() for w in prev_words[-overlap_len:]]
            curr_segment = [w.lower() for w in curr_words[:overlap_len]]
            
            if prev_segment == curr_segment:
                return overlap_len
        
        return 0
    
    def _remove_internal_duplicates(self, words: List[str]) -> List[str]:
        """
        Remove consecutive duplicate word sequences within the chunk.
        
        Args:
            words: List of words
            
        Returns:
            List with internal duplicates removed
        """
        keep = [True] * len(words)
        
        # Check for repeated sequences
        for seq_len in range(self.min_sequence_length, len(words) // 2 + 1):
            i = 0
            while i < len(words) - seq_len:
                if not keep[i]:
                    i += 1
                    continue
                
                sequence = [w.lower() for w in words[i:i + seq_len]]
                next_start = i + seq_len
                
                if next_start + seq_len <= len(words):
                    next_sequence = [w.lower() for w in words[next_start:next_start + seq_len]]
                    
                    if sequence == next_sequence:
                        # Mark duplicate for removal
                        for j in range(next_start, next_start + seq_len):
                            keep[j] = False
                        self.total_duplicates_removed += seq_len
                        i = next_start + seq_len
                        continue
                
                i += 1
        
        return [w for i, w in enumerate(words) if keep[i]]
    
    def reset(self):
        """Reset the deduplicator state."""
        self.previous_words = []
        self.total_duplicates_removed = 0
    
    def get_stats(self) -> dict:
        """Get deduplication statistics."""
        return {
            "total_duplicates_removed": self.total_duplicates_removed,
            "context_size": len(self.previous_words)
        }


def clean_transcription_text(text: str) -> str:
    """
    Convenience function for cleaning transcription text.

    This is a simple wrapper around remove_duplicate_text() that returns
    only the cleaned text (not stats).

    Args:
        text: Transcription text to clean

    Returns:
        Cleaned text
    """
    cleaned, stats = remove_duplicate_text(text, aggressive=False)
    return cleaned
