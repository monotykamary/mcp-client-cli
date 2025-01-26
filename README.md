# MCP CLI client

A fork of [adhikasp/mcp-client-cli](https://github.com/adhikasp/mcp-client-cli) that adds interactive mode, dynamic command configuration, and more features.

This is a CLI program that implements the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) client, allowing you to run LLM prompts from your terminal.

You can use any [MCP-compatible servers](https://github.com/punkpeye/awesome-mcp-servers) through this client. It serves as an alternative to Claude Desktop, and supports various LLM providers including OpenAI, Groq, or local LLM models via [llama](https://github.com/ggerganov/llama.cpp).

![C4 Diagram](c4_diagram.png)

## Usage

### Basic Usage

```bash
$ llm What is the capital city of North Sumatra?
The capital city of North Sumatra is Medan.
```

You can omit the quotes, but be careful with bash special characters like `&`, `|`, `;` that might be interpreted by your shell.

### Interactive Mode

You can start an interactive session using the `-i` or `--interactive` flag:

```bash
$ llm -i
──────────────────
User: What is the capital city of North Sumatra?

Assistant:
The capital city of North Sumatra is Medan.
──────────────────
User: Tell me more about Medan

Assistant:
[LLM will provide information about Medan]
──────────────────
User: exit
```

In interactive mode:
- Each message is clearly separated with a divider
- Your input is labeled as "User:" and responses are prefixed with "A:"
- The conversation context is maintained throughout the session
- Type `exit` or press Ctrl+C to end the session

You can also pipe input from other commands or files:

```bash
$ echo "What is the capital city of North Sumatra?" | llm
The capital city of North Sumatra is Medan.

$ echo "Given a location, tell me its capital city." > instructions.txt
$ cat instruction.txt | llm "West Java"
The capital city of West Java is Bandung.
```

### Image Input

You can pipe image files to analyze them with multimodal LLMs:

```bash
$ cat image.jpg | llm "What do you see in this image?"
[LLM will analyze and describe the image]

$ cat screenshot.png | llm "Is there any error in this screenshot?"
[LLM will analyze the screenshot and point out any errors]
```

### Using Prompt Templates

You can use predefined prompt templates by using the `p` prefix followed by the template name and its arguments:

```bash
# List available prompt templates
$ llm --list-prompts

# Use a template
$ llm p review  # Review git changes
$ llm p commit  # Generate commit message
$ llm p yt url=https://youtube.com/...  # Summarize YouTube video
```

### Triggering a tool

```bash
$ llm What is the top article on hackernews today?

================================== Ai Message ==================================
Tool Calls:
  brave_web_search (call_eXmFQizLUp8TKBgPtgFo71et)
 Call ID: call_eXmFQizLUp8TKBgPtgFo71et
  Args:
    query: site:news.ycombinator.com
    count: 1
Brave Search MCP Server running on stdio

# If the tool requires confirmation, you'll be prompted:
Confirm tool call? [y/n]: y

================================== Ai Message ==================================
Tool Calls:
  fetch (call_xH32S0QKqMfudgN1ZGV6vH1P)
 Call ID: call_xH32S0QKqMfudgN1ZGV6vH1P
  Args:
    url: https://news.ycombinator.com/
================================= Tool Message =================================
Name: fetch

[TextContent(type='text', text='Contents [REDACTED]]
================================== Ai Message ==================================

The top article on Hacker News today is:

### [Why pipes sometimes get "stuck": buffering](https://jvns.ca)
- **Points:** 31
- **Posted by:** tanelpoder
- **Posted:** 1 hour ago

You can view the full list of articles on [Hacker News](https://news.ycombinator.com/)
```

To bypass tool confirmation requirements, use the `--no-confirmations` flag:

```bash
$ llm --no-confirmations "What is the top article on hackernews today?"
```

To use in bash scripts, add the --no-intermediates, so it doesn't print intermediate messages, only the concluding end message.
```bash
$ llm --no-intermediates "What is the time in Tokyo right now?"
```

```bash
$ llm --no-confirmations "What is the top article on hackernews today?"
```

### Continuation

Add a `c ` prefix to your message to continue the last conversation.

```bash
$ llm asldkfjasdfkl
It seems like your message might have been a typo or an error. Could you please clarify or provide more details about what you need help with?
$ llm c what did i say previously?
You previously typed "asldkfjasdfkl," which appears to be a random string of characters. If you meant to ask something specific or if you have a question, please let me know!
```

### Clipboard Support

You can use content from your clipboard using the `cb` command:

```bash
# After copying text to clipboard
$ llm cb
[LLM will process the clipboard text]

$ llm cb "What language is this code written in?"
[LLM will analyze the clipboard text with your question]

# After copying an image to clipboard
$ llm cb "What do you see in this image?"
[LLM will analyze the clipboard image]

# You can combine it with continuation
$ llm cb c "Tell me more about what you see"
[LLM will continue the conversation about the clipboard content]
```

The clipboard feature works in:
- Native Windows/macOS/Linux environments
  - Windows: Uses PowerShell
  - macOS: Uses `pbpaste` for text, `pngpaste` for images (optional)
  - Linux: Uses `xclip` (required for clipboard support)
- Windows Subsystem for Linux (WSL)
  - Accesses the Windows clipboard through PowerShell
  - Works with both text and images
  - Make sure you have access to `powershell.exe` from WSL

Required tools for clipboard support:
- Windows: PowerShell (built-in)
- macOS: 
  - `pbpaste` (built-in) for text
  - `pngpaste` (optional) for images: `brew install pngpaste`
- Linux: 
  - `xclip`: `sudo apt install xclip` or equivalent

The CLI automatically detects if the clipboard content is text or image and handles it appropriately.

### Additional Options

```bash
$ llm --list-tools                # List all available tools
$ llm --list-prompts              # List available prompt templates
$ llm --no-tools                  # Run without any tools
$ llm --force-refresh             # Force refresh tool capabilities
$ llm --text-only                 # Output raw text without markdown formatting
$ llm --show-memories             # Show user memories
$ llm --clear-memories            # Clear all user memories
$ llm -i, --interactive           # Start an interactive chat session
```

## Setup

1. Clone the repository:
   ```bash
   pip install git+https://github.com/monotykamary/mcp-client-cli.git
   # OR
   pipx install git+https://github.com/monotykamary/mcp-client-cli.git
   ```

2. Create the following configuration files:

   `.llmrules` (optional, in your project directory) or `~/.llm/.llmrules` (optional, global):
   ```text
   You are an AI assistant helping a software engineer...
   ```
   Note: 
   - The local .llmrules file in your project directory takes precedence over the global one in ~/.llm/
   - If no .llmrules file is found, the system will start with an empty system prompt

   `~/.llm/config.json`:
   ```json
   {
     "llm": {
       "provider": "openai",
       "model": "gpt-4",
       "api_key": "your-openai-api-key",
       "temperature": 0.7,
       "base_url": "https://api.openai.com/v1"  // Optional, for OpenRouter or other providers
     },
     "mcpServers": {
       "fetch": {
         "command": "uvx",
         "args": ["mcp-server-fetch"],
         "requires_confirmation": ["fetch"],
         "enabled": true,  // Optional, defaults to true
         "exclude_tools": []  // Optional, list of tool names to exclude
       },
       "local-server": {
         "command": "python",
         "args": ["{pwd}/my_server.py"],  // {pwd} will be replaced with the current working directory
         "requires_confirmation": ["local_tool"]
       },
       "project-server": {
         "command": "node",
         "args": ["server.js", "--name", "{basename_pwd}"],  // {basename_pwd} will be replaced with the current directory name
         "requires_confirmation": ["project_tool"]
       },
       "brave-search": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-brave-search"],
         "env": {
           "BRAVE_API_KEY": "your-brave-api-key"
         },
         "requires_confirmation": ["brave_web_search"]
       },
       "youtube": {
         "command": "uvx",
         "args": ["--from", "git+https://github.com/adhikasp/mcp-youtube", "mcp-youtube"]
       }
     }
   }
   ```

   Note: 
   - Use `requires_confirmation` to specify which tools need user confirmation before execution
   - The LLM API key can also be set via environment variables `LLM_API_KEY` or `OPENAI_API_KEY`
   - The config file can be placed in either `~/.llm/config.json` or `$PWD/.llm/config.json`
   - You can comment the JSON config file with `//` if you like to switch around the configuration
   - Use `{pwd}` in command or args to reference the current working directory where the command is being executed (useful for local server paths)
   - Use `{basename_pwd}` to reference just the name of the current directory (e.g., if pwd is "/path/to/project", basename_pwd will be "project")

3. Run the CLI:
   ```bash
   llm "What is the capital city of North Sumatra?"
   ```

## Contributing

Feel free to submit issues and pull requests for improvements or bug fixes.
