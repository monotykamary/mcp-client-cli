from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Confirm

class OutputHandler:
    def __init__(self, text_only: bool = False, interactive: bool = False):
        self.console = Console()
        self.text_only = text_only
        self.interactive = interactive
        self.md = ""

    def start(self):
        if self.interactive:
            self.console.print("[bold cyan]Assistant:[/bold cyan]")

    def update(self, chunk: any):
        new_content = self._parse_chunk(chunk, self.md)

        # Get just the new content
        if len(new_content) > len(self.md):
            diff = new_content[len(self.md):]
            # Print new content directly
            if self.text_only:
                print(diff, end="", flush=True)
            else:
                self.console.print(diff, end="", soft_wrap=True)

        # Update stored content after handling output
        self.md = new_content

    def update_error(self, error: Exception):
        import traceback
        error_msg = f"Error: {error}\n\nStack trace:\n```\n{traceback.format_exc()}```"
        self.md += error_msg
        if self.text_only:
            print(error_msg)
        else:
            self.console.print(error_msg)

    def stop(self):
        if not self.text_only:
            self.console.print()

    def confirm_tool_call(self, config: dict, chunk: any) -> bool:
        if not self._is_tool_call_requested(chunk, config):
            return True

        # For tool confirmation, render the complete markdown
        self.console.print("\n")
        self.console.print(Markdown(self.md))
        self.console.print("\n")

        # Get confirmation
        is_confirmed = self._ask_tool_call_confirmation()

        if not is_confirmed:
            denial_msg = "\n# Tool call denied"
            self.md += denial_msg
            if not self.text_only:
                self.console.print(denial_msg)
            return False

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
                    # Escape any existing Rich markup in the content
                    content = content.replace("[", "\\[").replace("]", "\\]")
                    md += content
                elif isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict) and "text" in content[0]:
                    # Escape any existing Rich markup in the text content
                    text = content[0]["text"].replace("[", "\\[").replace("]", "\\]")
                    md += text
        # If this is a final value
        elif isinstance(chunk, dict) and "messages" in chunk:
            last_message = chunk["messages"][-1]
            if isinstance(last_message, AIMessage):
                content = last_message.content
                if isinstance(content, str):
                    # Escape any existing Rich markup in the content
                    content = content.replace("[", "\\[").replace("]", "\\]")
                    md += content
                elif isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict) and "text" in content[0]:
                    # Escape any existing Rich markup in the text content
                    text = content[0]["text"].replace("[", "\\[").replace("]", "\\]")
                    md += text
            md += "\n"
        elif isinstance(chunk, tuple) and chunk[0] == "values":
            message: BaseMessage = chunk[1]['messages'][-1]
            if isinstance(message, AIMessage) and message.tool_calls:
                for tc in message.tool_calls:
                    # Escape any Rich markup in tool call output
                    tool_name = str(tc.get('name', 'Tool')).replace("[", "\\[").replace("]", "\\]")
                    args = str(tc.get("args", {})).replace("[", "\\[").replace("]", "\\]")
                    md += f"\n\n{tool_name}: {args}"
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
        is_tool_call_confirmed = Confirm.ask(f"Confirm tool call?", console=self.console)
        return is_tool_call_confirmed
