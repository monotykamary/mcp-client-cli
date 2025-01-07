"""Configuration management for the MCP client CLI."""

from dataclasses import dataclass
from pathlib import Path
import os
import commentjson
from typing import Dict, List, Optional

from .const import CONFIG_FILE, CONFIG_DIR, LLMRULES_FILE, LOCAL_LLMRULES_FILE

@dataclass
class LLMConfig:
    """Configuration for the LLM model."""
    model: str = "gpt-4o"
    provider: str = "openai"
    api_key: Optional[str] = None
    temperature: float = 0
    base_url: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict) -> "LLMConfig":
        """Create LLMConfig from dictionary."""
        return cls(
            model=config.get("model", cls.model),
            provider=config.get("provider", cls.provider),
            api_key=config.get("api_key", os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))),
            temperature=config.get("temperature", cls.temperature),
            base_url=config.get("base_url"),
        )

@dataclass
class ServerConfig:
    """Configuration for an MCP server."""
    command: str
    args: List[str] = None
    env: Dict[str, str] = None
    enabled: bool = True
    exclude_tools: List[str] = None
    requires_confirmation: List[str] = None

    @staticmethod
    def _substitute_path_variables(value: str) -> str:
        """Replace path variables with their actual values."""
        if not value:
            return value
        
        cwd = Path.cwd()
        # Replace {pwd} with current working directory
        if "{pwd}" in value:
            value = value.replace("{pwd}", str(cwd))
        # Replace {basename_pwd} with the name of the current directory
        if "{basename_pwd}" in value:
            value = value.replace("{basename_pwd}", cwd.name)
        return value

    @classmethod
    def from_dict(cls, config: dict) -> "ServerConfig":
        """Create ServerConfig from dictionary."""
        # Process command and args for path substitution
        command = cls._substitute_path_variables(config["command"])
        args = [cls._substitute_path_variables(arg) for arg in config.get("args", [])]
        
        return cls(
            command=command,
            args=args,
            env=config.get("env", {}),
            enabled=config.get("enabled", True),
            exclude_tools=config.get("exclude_tools", []),
            requires_confirmation=config.get("requires_confirmation", [])
        )

@dataclass
class AppConfig:
    """Main application configuration."""
    llm: LLMConfig
    system_prompt: str
    mcp_servers: Dict[str, ServerConfig]
    tools_requires_confirmation: List[str]

    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration from file."""
        # Load main config
        config_paths = [CONFIG_FILE, CONFIG_DIR / "config.json"]
        chosen_path = next((path for path in config_paths if os.path.exists(path)), None)
        
        if chosen_path is None:
            raise FileNotFoundError(f"Could not find config file in any of: {', '.join(map(str, config_paths))}")

        with open(chosen_path, 'r') as f:
            config = commentjson.load(f)

        # Load system prompt from .llmrules files, defaulting to empty string
        system_prompt = ""
        # Check local .llmrules first
        if os.path.exists(LOCAL_LLMRULES_FILE):
            with open(LOCAL_LLMRULES_FILE, 'r') as f:
                system_prompt = f.read().strip()
        # Fall back to global .llmrules if local doesn't exist
        elif os.path.exists(LLMRULES_FILE):
            with open(LLMRULES_FILE, 'r') as f:
                system_prompt = f.read().strip()

        # Extract tools requiring confirmation
        tools_requires_confirmation = []
        for server_config in config["mcpServers"].values():
            tools_requires_confirmation.extend(server_config.get("requires_confirmation", []))

        return cls(
            llm=LLMConfig.from_dict(config.get("llm", {})),
            system_prompt=system_prompt,
            mcp_servers={
                name: ServerConfig.from_dict(server_config)
                for name, server_config in config["mcpServers"].items()
            },
            tools_requires_confirmation=tools_requires_confirmation
        )

    def get_enabled_servers(self) -> Dict[str, ServerConfig]:
        """Get only enabled server configurations."""
        return {
            name: config 
            for name, config in self.mcp_servers.items() 
            if config.enabled
        }
