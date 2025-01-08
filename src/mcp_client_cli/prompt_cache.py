"""Utilities for prompt caching and token estimation."""

from typing import List, Dict, Any, Tuple
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.
    
    This is a simple estimation based on character count.
    For more accurate results, consider using a proper tokenizer.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated number of tokens
    """
    # A simple estimation: roughly 4 characters per token
    return len(str(text)) // 4

def process_messages_for_caching(
    messages: List[BaseMessage], 
    max_cache_blocks: int = 4
) -> List[Dict[str, Any]]:
    """Process messages to add caching for system prompt and largest messages.
    
    Args:
        messages: List of messages to process
        max_cache_blocks: Maximum number of cache blocks to use (default: 4)
        
    Returns:
        List of processed messages with cache control added
    """
    processed_messages = []
    cache_count = 0
    
    # Get message sizes for all non-system messages
    message_sizes: List[Tuple[int, int, BaseMessage]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, SystemMessage):
            token_count = estimate_tokens(str(msg.content))
            message_sizes.append((token_count, i, msg))
    
    # Sort by size descending
    message_sizes.sort(reverse=True)
    
    # Get indices of largest messages to cache
    to_cache = set()
    remaining_cache_blocks = max_cache_blocks
    
    # Process messages
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            # Always try to cache system message if it's large enough
            token_count = estimate_tokens(str(msg.content))
            if token_count > 1024 and remaining_cache_blocks > 0:
                content = [{
                    "type": "text",
                    "text": str(msg.content),
                    "cache_control": {"type": "ephemeral"}
                }]
                remaining_cache_blocks -= 1
            else:
                content = [{"type": "text", "text": str(msg.content)}]
                
            processed_messages.append({
                "role": "system",
                "content": content
            })
            continue
            
        # For non-system messages, cache the largest ones
        if remaining_cache_blocks > 0:
            for _, idx, _ in message_sizes:
                if idx == i:
                    to_cache.add(i)
                    remaining_cache_blocks -= 1
                    break
                if remaining_cache_blocks == 0:
                    break
        
        # Process message content
        if i in to_cache:
            content = [{
                "type": "text",
                "text": str(msg.content),
                "cache_control": {"type": "ephemeral"}
            }]
        else:
            content = [{"type": "text", "text": str(msg.content)}]
            
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        processed_messages.append({
            "role": role,
            "content": content
        })
        
    return processed_messages
