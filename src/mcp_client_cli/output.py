from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, ToolMessage
from rich.console import Console, ConsoleDimensions
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Confirm

class OutputHandler:
    def __init__(self, text_only: bool = False, interactive: bool = False):
        self.console = Console()
        self.text_only = text_only
        self.interactive = interactive
        if self.text_only:
            self.md = ""
        else:
            self.md = "Thinking...\n"
        self._live = None
        self.full_content = ""  # Store complete content for later display

    def start(self):
        if not self.text_only:
            # Only show Assistant prefix in interactive mode
            if self.interactive:
                self.console.print("[bold cyan]Assistant:[/bold cyan]")
            self._live = Live(
                Markdown(self.md),
                vertical_overflow="visible",  # Allow scrolling during streaming
                console=self.console,
                refresh_per_second=30,
                auto_refresh=False,  # We'll manually refresh to ensure immediate updates
            )
            self._live.start()

    def update(self, chunk: any):
        new_content = self._parse_chunk(chunk, self.md)
        # Update full content for later display
        if not self.text_only:
            self.full_content = new_content
        
        if self.text_only:
            self.console.print(self._parse_chunk(chunk), end="")
        else:
            if new_content.startswith("Thinking...") and not new_content.strip("Thinking...").isspace():
                new_content = new_content.strip("Thinking...").strip()
            # Update display content while streaming
            self.md = new_content
            # Show all content during streaming with visible overflow
            self._live.update(Markdown(self.md), refresh=True)

    def update_error(self, error: Exception):
        import traceback
        self.md += f"Error: {error}\n\nStack trace:\n```\n{traceback.format_exc()}```"
        if self.text_only:
            self.console.print(self.md)
        else:
            self._live.update(Markdown(self.md), refresh=True)

    def stop(self):
        if not self.text_only and self._live:
            self._live.stop()

    def confirm_tool_call(self, config: dict, chunk: any) -> bool:
        if not self._is_tool_call_requested(chunk, config):
            return True

        self.stop()
        is_confirmed = self._ask_tool_call_confirmation()
        if not is_confirmed:
            self.md += "# Tool call denied"
            return False
            
        if not self.text_only:
            self.start()
        return True

    def finish(self):
        self.stop()

    def _parse_chunk(self, chunk: any, md: str = "") -> str:
        """
        Parse the chunk of agent response.
        It will stream the response as it is received.
        """
        # If this is a message chunk
        if isinstance(chunk, tuple) and chunk[0] == "messages":
            message_chunk = chunk[1][0]  # Get the message content
            if isinstance(message_chunk, AIMessageChunk):
                content = message_chunk.content
                if isinstance(content, str):
                    md += content
                elif isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict) and "text" in content[0]:
                    md += content[0]["text"]
        # If this is a final value
        elif isinstance(chunk, dict) and "messages" in chunk:
            # Get the last message content and stream it
            last_message = chunk["messages"][-1]
            if isinstance(last_message, AIMessage):
                content = last_message.content
                if isinstance(content, str):
                    md += content
                elif isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict) and "text" in content[0]:
                    md += content[0]["text"]
            md += "\n"
        elif isinstance(chunk, tuple) and chunk[0] == "values":
            message: BaseMessage = chunk[1]['messages'][-1]
            if isinstance(message, AIMessage) and message.tool_calls:
                # Ensure there's a newline before Tool Calls if not already present
                if not md.endswith('\n'):
                    md += '\n\n'
                md += "**Tool Calls:**"
                for tc in message.tool_calls:
                    lines = [
                        f"  {tc.get('name', 'Tool')}",
                    ]

                    args = tc.get("args")
                    if args:  # Only add code block if there are arguments
                        lines.append("```")
                        if isinstance(args, str):
                            lines.append(f"{args}")
                        elif isinstance(args, dict):
                            for arg, value in args.items():
                                lines.append(f"{arg}: {value}")
                        lines.append("```")
                    lines.append("")  # Add empty line for spacing
                    md += "\n".join(lines)
            elif isinstance(message, ToolMessage) and message.status != "success":
                # Stream each part of the error message
                if not md.endswith('\n'):
                    md += '\n'
                # Stream the header
                md += "Failed call with error:\n```\n"
                # Stream the error content character by character
                if isinstance(message.content, str):
                    md += message.content
                elif isinstance(message.content, dict):
                    md += str(message.content)
                md += "\n```\n"
            md += "\n"
        return md

    def _is_tool_call_requested(self, chunk: any, config: dict) -> bool:
        """
        Check if the chunk contains a tool call request and requires confirmation.
        """
        if isinstance(chunk, tuple) and chunk[0] == "values":
            if len(chunk) > 1 and isinstance(chunk[1], dict) and "messages" in chunk[1]:
                message = chunk[1]['messages'][-1]
                if isinstance(message, AIMessage) and message.tool_calls:
                    for tc in message.tool_calls:
                        if tc.get("name") in config["tools_requires_confirmation"]:
                            return True
        return False

    def _ask_tool_call_confirmation(self) -> bool:
        """
        Ask the user for confirmation to run a tool call.
        """
        self.console.print("\n")
        self.console.print(Markdown(self.md))
        self.console.print("\n")
        is_tool_call_confirmed = Confirm.ask(f"Confirm tool call?", console=self.console)
        if not is_tool_call_confirmed:
            return False
        return True
