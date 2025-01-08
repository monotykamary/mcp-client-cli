from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.patch_stdout import patch_stdout

async def get_user_input(prompt_text: str = "") -> str:
    """
    Get user input with proper handling of special keys.
    
    Args:
        prompt_text: Text to show as prompt
        
    Returns:
        The user's input as a string
    """
    # Create a session with custom style
    style = Style.from_dict({
        'prompt': 'bold green',
    })
    
    session = PromptSession(style=style)
    
    # Get input with proper key handling
    try:
        # Use patch_stdout to prevent output issues in async context
        with patch_stdout():
            # Add a small delay to ensure the console output is complete
            import asyncio
            await asyncio.sleep(0.1)
            result = await session.prompt_async(prompt_text)
        return result.strip()
    except KeyboardInterrupt:
        return '/exit'
    except EOFError:
        return '/exit'
