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
) -> List[Dict[str, Any]]:
    """Process messages to add caching for system prompt and last 3 user messages.
    
    Args:
        messages: List of messages to process
        
    Returns:
        List of processed messages with cache control added
    """
    processed_messages = []
    
    # Find indices of last three user messages
    lastThreeUserMsgIndices = [
        i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)
    ][-3:]
    
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            # Always cache system message
            content = [{
                "type": "text",
                "text": str(msg.content),
                "cache_control": {"type": "ephemeral"}
            }]
            processed_messages.append({
                "role": "system",
                "content": content
            })
        elif isinstance(msg, HumanMessage) and i in lastThreeUserMsgIndices:
            # Cache last three user messages
            content = [{
                "type": "text",
                "text": str(msg.content),
                "cache_control": {"type": "ephemeral"}
            }]
            processed_messages.append({
                "role": "user",
                "content": content
            })
        else:
            # Do not cache other messages
            content = [{"type": "text", "text": str(msg.content)}]
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            processed_messages.append({
                "role": role,
                "content": content
            })
            
    return processed_messages
