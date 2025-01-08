from datetime import datetime, timedelta
from typing import Optional, List
from mcp import StdioServerParameters, types
import json
import aiosqlite
import uuid
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .const import *

def get_cached_tools(server_param: StdioServerParameters) -> Optional[List[types.Tool]]:
    """Retrieve cached tools if available and not expired.
    
    Args:
        server_param (StdioServerParameters): The server parameters to identify the cache.
    
    Returns:
        Optional[List[types.Tool]]: A list of tools if cache is available and not expired, otherwise None.
    """
    from .tool import generate_server_id  # Import here to avoid circular dependency
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = generate_server_id(server_param)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    if not cache_file.exists():
        return None
        
    cache_data = json.loads(cache_file.read_text())
    cached_time = datetime.fromisoformat(cache_data["cached_at"])
    
    if datetime.now() - cached_time > timedelta(hours=CACHE_EXPIRY_HOURS):
        return None
            
    return [types.Tool(**tool) for tool in cache_data["tools"]]


def save_tools_cache(server_param: StdioServerParameters, tools: List[types.Tool]) -> None:
    """Save tools to cache.
    
    Args:
        server_param (StdioServerParameters): The server parameters to identify the cache.
        tools (List[types.Tool]): The list of tools to be cached.
    """
    from .tool import generate_server_id  # Import here to avoid circular dependency
    
    cache_key = generate_server_id(server_param)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    cache_data = {
        "cached_at": datetime.now().isoformat(),
        "tools": [tool.model_dump() for tool in tools]
    }
    cache_file.write_text(json.dumps(cache_data))


class ConversationManager:
    """Manages conversation persistence in SQLite database."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def _init_db(self, db) -> None:
        """Initialize database schema.
        
        Args:
            db: The database connection object.
        """
        await db.execute("""
            CREATE TABLE IF NOT EXISTS last_conversation (
                id INTEGER PRIMARY KEY,
                thread_id TEXT NOT NULL
            )
        """)
        await db.commit()
    
    async def get_last_id(self) -> str:
        """Get the thread ID of the last conversation.
        
        Returns:
            str: The thread ID of the last conversation, or a new UUID if no conversation exists.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await self._init_db(db)
            async with db.execute("SELECT thread_id FROM last_conversation LIMIT 1") as cursor:
                row = await cursor.fetchone()
            return row[0] if row else uuid.uuid4().hex
    
    async def save_id(self, thread_id: str, db = None) -> None:
        """Save thread ID as the last conversation.
        
        Args:
            thread_id (str): The thread ID to save.
            db: The database connection object (optional).
        """
        if db is None:
            async with aiosqlite.connect(self.db_path) as db:
                await self._save_id(db, thread_id)
        else:
            await self._save_id(db, thread_id)
    
    async def _save_id(self, db, thread_id: str) -> None:
        """Internal method to save thread ID.
        
        Args:
            db: The database connection object.
            thread_id (str): The thread ID to save.
        """
        async with db.cursor() as cursor:
            await self._init_db(db)
            await cursor.execute("DELETE FROM last_conversation")
            await cursor.execute(
                "INSERT INTO last_conversation (thread_id) VALUES (?)", 
                (thread_id,)
            )
            await db.commit()
            
    async def get_history(self, thread_id: str, db = None) -> List[BaseMessage]:
        """Get conversation history for a thread.
        
        Args:
            thread_id (str): The thread ID to get history for.
            db: The database connection object (optional).
            
        Returns:
            List[BaseMessage]: List of messages in the conversation history.
        """
        if db is None:
            async with aiosqlite.connect(self.db_path) as db:
                return await self._get_history(db, thread_id)
        else:
            return await self._get_history(db, thread_id)
            
    async def _get_history(self, db, thread_id: str) -> List[BaseMessage]:
        """Internal method to get conversation history.
        
        Args:
            db: The database connection object.
            thread_id (str): The thread ID to get history for.
            
        Returns:
            List[BaseMessage]: List of messages in the conversation history.
        """
        await self._init_db(db)
        # Create messages table if it doesn't exist
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                thread_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
        
        # Get messages for thread
        async with db.execute(
            "SELECT role, content FROM messages WHERE thread_id = ? ORDER BY created_at",
            (thread_id,)
        ) as cursor:
            messages = []
            async for row in cursor:
                role, content = row
                if role == "human":
                    messages.append(HumanMessage(content=content))
                elif role == "ai":
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))
            return messages
            
    async def save_message(self, thread_id: str, message: BaseMessage, db = None) -> None:
        """Save a message to the conversation history.
        
        Args:
            thread_id (str): The thread ID to save the message under.
            message (BaseMessage): The message to save.
            db: The database connection object (optional).
        """
        if db is None:
            async with aiosqlite.connect(self.db_path) as db:
                await self._save_message(db, thread_id, message)
        else:
            await self._save_message(db, thread_id, message)
            
    async def _save_message(self, db, thread_id: str, message: BaseMessage) -> None:
        """Internal method to save a message.
        
        Args:
            db: The database connection object.
            thread_id (str): The thread ID to save the message under.
            message (BaseMessage): The message to save.
        """
        # Create messages table if it doesn't exist
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                thread_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Determine message role
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "unknown"
            
        # Save message
        await db.execute(
            "INSERT INTO messages (thread_id, session_id, role, content) VALUES (?, ?, ?, ?)",
            (thread_id, thread_id, role, str(message.content))
        )
        await db.commit()
