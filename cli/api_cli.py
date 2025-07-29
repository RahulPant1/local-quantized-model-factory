#!/usr/bin/env python3
"""
LQMF API Command Line Interface

A comprehensive CLI for managing quantized models and API server functionality.
Provides an interactive interface for:
- Loading and unloading quantized models
- Starting and stopping the API server
- Interactive chat sessions with loaded models
- API endpoint testing and validation
- Real-time server status monitoring
- Model performance benchmarking

Author: LQMF Development Team
Version: 1.0.0
Compatibility: Python 3.8+

Usage:
    python cli/api_cli.py
    
Example Commands:
    load microsoft/DialoGPT-small    # Load a quantized model
    chat                             # Start interactive chat
    start 8080                       # Start server on custom port
    test model_name Hello world      # Test API endpoint
    status                           # Show server and model status
"""

import sys
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.api_server_agent import APIServerAgent

# Configuration constants
class CLIConfig:
    """Configuration constants for the API CLI."""
    DEFAULT_PORT = 8000
    DEFAULT_HOST = "127.0.0.1"
    MAX_CHAT_TOKENS = 150
    DEFAULT_TEMPERATURE = 0.7
    REQUEST_TIMEOUT = 30
    HEALTH_CHECK_TIMEOUT = 5
    
class CommandType(Enum):
    """Enumeration of available CLI commands."""
    LOAD = "load"
    UNLOAD = "unload"
    START = "start"
    STOP = "stop"
    STATUS = "status"
    CHAT = "chat"
    TEST = "test"
    LIST = "list"
    LOADED = "loaded"
    CONFIG = "config"
    HELP = "help"
    EXIT = "exit"
    CLEAR = "clear"

@dataclass
class ChatSession:
    """Data class representing an active chat session."""
    model_name: str
    message_count: int = 0
    total_tokens: int = 0
    start_time: float = 0.0

# Initialize console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False)]
)
logger = logging.getLogger("lqmf.api_cli")

class APICommandLineInterface:
    """
    Advanced API Command Line Interface for LQMF (Local Quantized Model Factory).
    
    This class provides a comprehensive interactive CLI for managing quantized models,
    API server operations, and real-time chat sessions. It serves as the primary
    interface for users to interact with the LQMF system.
    
    Features:
    - Automatic model loading and server management
    - Interactive chat sessions with loaded models
    - Real-time API testing and validation
    - Comprehensive status monitoring
    - Error handling and recovery
    - Rich console output with progress indicators
    
    Attributes:
        api_server_agent (APIServerAgent): Core server management agent
        current_chat_session (Optional[ChatSession]): Active chat session data
        command_history (List[str]): History of executed commands
    
    Example:
        >>> cli = APICommandLineInterface()
        >>> cli.run_interactive_session()
    """
    
    def __init__(self) -> None:
        """
        Initialize the API Command Line Interface.
        
        Sets up the API server agent, initializes console output,
        and displays the welcome interface with initial status.
        
        Raises:
            ImportError: If required dependencies are not available
            RuntimeError: If initialization fails
        """
        try:
            self.api_server_agent = APIServerAgent()
            self.current_chat_session: Optional[ChatSession] = None
            self.command_history: List[str] = []
            
            logger.info("API CLI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API CLI: {e}")
            raise RuntimeError(f"Initialization failed: {e}") from e
        
        console.print(Panel.fit(
            "🚀 LQMF Simple API CLI - Model Serving Interface",
            style="bold blue"
        ))
        
        self._show_initial_status()
    
    def _show_initial_status(self) -> None:
        """
        Display initial system status including available models and hardware info.
        
        Shows:
        - Number of quantized models available
        - Hardware capabilities (GPU availability)
        - Basic usage instructions
        """
        try:
            available_models = self.api_server_agent.discover_quantized_models()
        except Exception as e:
            logger.error(f"Failed to discover models: {e}")
            available_models = []
        
        if available_models:
            console.print(f"[green]Found {len(available_models)} quantized models available[/green]")
        else:
            console.print("[yellow]No quantized models found. Quantize models first using main CLI.[/yellow]")
        
        console.print("[dim]Type 'help' for commands[/dim]")
    
    def run_interactive_session(self) -> None:
        """
        Run the main interactive CLI session loop.
        
        Provides a continuous command-line interface that accepts user input,
        processes commands, and executes appropriate actions. The session continues
        until the user explicitly exits.
        
        Handles:
        - Command parsing and routing
        - Error recovery and user feedback
        - Command history tracking
        - Graceful shutdown procedures
        
        Raises:
            KeyboardInterrupt: Handled gracefully for user exit
            Exception: Logged and displayed to user with recovery options
        """
        console.print("\n[cyan]🚀 LQMF API CLI - Interactive Session Started[/cyan]")
        console.print("[dim]Type 'help' for commands, 'exit' to quit[/dim]")
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]LQMF>[/bold blue]", default="").strip()
                
                if not user_input:
                    continue
                    
                # Track command history
                self.command_history.append(user_input)
                user_input_lower = user_input.lower()
                
                # Enhanced command routing with validation
                if user_input_lower in ['exit', 'quit', 'q']:
                    self._handle_exit()
                    break
                elif user_input_lower in ['help', 'h', '?']:
                    self._show_help()
                elif user_input_lower.startswith('load'):
                    self._handle_load_model(user_input)
                elif user_input_lower.startswith('unload'):
                    self._handle_unload_model(user_input)
                elif user_input_lower in ['status', 'show status', 'api status', 'info']:
                    self._handle_show_status()
                elif user_input_lower.startswith('start'):
                    self._handle_start_server(user_input)
                elif user_input_lower.startswith('stop'):
                    self._handle_stop_server()
                elif user_input_lower.startswith('test'):
                    self._handle_test_api(user_input)
                elif user_input_lower in ['list', 'models', 'show models', 'list models', 'ls']:
                    self._handle_list_models()
                elif user_input_lower in ['loaded', 'show loaded', 'loaded models', 'active']:
                    self._handle_show_loaded_models()
                elif user_input_lower in ['clear', 'cls']:
                    self._handle_clear_screen()
                elif user_input_lower in ['config', 'show config', 'settings']:
                    self._handle_show_config()
                elif user_input_lower.startswith('chat'):
                    self._handle_chat_mode(user_input)
                else:
                    self._handle_unknown_command(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit properly.[/yellow]")
            except Exception as e:
                logger.error(f"Unexpected error in interactive session: {e}")
                console.print(f"[red]Unexpected error: {e}[/red]")
                console.print("[dim]Type 'help' for available commands or 'exit' to quit[/dim]")
    
    def _handle_unknown_command(self, command: str) -> None:
        """
        Handle unknown commands with helpful suggestions.
        
        Args:
            command: The unrecognized command entered by the user
        """
        console.print(f"[red]Unknown command: '{command}'[/red]")
        
        # Provide command suggestions based on similarity
        suggestions = self._get_command_suggestions(command.lower())
        if suggestions:
            console.print(f"[yellow]Did you mean: {', '.join(suggestions)}?[/yellow]")
        
        console.print("[dim]Type 'help' for available commands[/dim]")
    
    def _get_command_suggestions(self, command: str) -> List[str]:
        """
        Generate command suggestions based on input similarity.
        
        Args:
            command: The partial or misspelled command
            
        Returns:
            List of suggested commands
        """
        all_commands = [
            'load', 'unload', 'start', 'stop', 'status', 'chat', 
            'test', 'list', 'loaded', 'config', 'help', 'exit', 'clear'
        ]
        
        suggestions = []
        for cmd in all_commands:
            if command in cmd or cmd.startswith(command[:3]):
                suggestions.append(cmd)
        
        return suggestions[:3]  # Limit to top 3 suggestions"
    
    def _show_help(self) -> None:
        """
        Display comprehensive help information for all available commands.
        
        Shows categorized command listings with descriptions, usage examples,
        and practical tips for effective CLI usage.
        """
        help_text = """
[bold cyan]🚀 LQMF API CLI Commands:[/bold cyan]

[bold yellow]🖥️  Server Management:[/bold yellow]
  • [bold]start[/bold] [port] - Start API server (default port 8000)
  • [bold]stop[/bold] - Stop API server and cleanup resources
  • [bold]status[/bold] - Show comprehensive server and model status

[bold yellow]🤖 Model Management:[/bold yellow]  
  • [bold]load[/bold] <model_name> - Load quantized model (auto-starts server)
  • [bold]unload[/bold] <model_name> - Safely unload model from memory
  • [bold]list[/bold] / [bold]models[/bold] - Show all available quantized models
  • [bold]loaded[/bold] - Show currently active models with stats

[bold yellow]💬 Interactive Features:[/bold yellow]
  • [bold]chat[/bold] [model_name] - Start interactive chat session
  • [bold]test[/bold] <model_name> [message] - Test API with custom message
  • [bold]test health[/bold] - Verify server health and connectivity

[bold yellow]🔧 Utilities:[/bold yellow]
  • [bold]config[/bold] - Show system configuration and capabilities
  • [bold]clear[/bold] - Clear screen and reset display
  • [bold]help[/bold] - Show this comprehensive help guide
  • [bold]exit[/bold] - Gracefully shutdown CLI and server

[dim]💡 Quick Start Examples:[/dim]
[dim]  load meta-llama/Llama-2-7b-chat-hf    # Load model (auto-starts server)[/dim]
[dim]  chat                                   # Start chat with loaded model[/dim]
[dim]  test my_model "Explain quantum physics" # Test with custom prompt[/dim]
[dim]  start 8080                             # Start server on custom port[/dim]

[dim]💡 Pro Tips:[/dim]
[dim]  • Models auto-load when you use 'load' command[/dim]
[dim]  • Server auto-starts when loading first model[/dim]
[dim]  • Use Ctrl+C to exit chat sessions gracefully[/dim]
[dim]  • Check 'status' regularly for system health[/dim]
        """
        console.print(Panel(help_text, title="Help", expand=False))
    
    def _handle_load_model(self, command: str):
        """Handle load model command"""
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            # Show available models to choose from
            available_models = self.api_server_agent.discover_quantized_models()
            if not available_models:
                console.print("[red]No quantized models available[/red]")
                return
            
            console.print("\n[cyan]Available models:[/cyan]")
            for i, model in enumerate(available_models, 1):
                console.print(f"  {i}. {model}")
            
            choice = Prompt.ask("\nEnter model name or number")
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_models):
                    model_name = available_models[idx]
                else:
                    console.print("[red]Invalid choice[/red]")
                    return
            else:
                model_name = choice
        else:
            model_name = parts[1]
        
        # Validate model name
        if not self._validate_model_name(model_name):
            return
            
        # Check if model is already loaded
        if model_name in self.api_server_agent.loaded_models:
            console.print(f"[yellow]⚠️  Model '{model_name}' is already loaded[/yellow]")
            console.print("[dim]Use 'unload' first if you want to reload it[/dim]")
            return
        
        console.print(f"[blue]📥 Loading model: {model_name}[/blue]")
        
        try:
            success = self.api_server_agent.load_model(model_name)
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            console.print(f"[red]❌ Failed to load model: {e}[/red]")
            return
        
        if success:
            console.print(f"[green]✅ Model '{model_name}' loaded successfully![/green]")
            
            # Display model information
            self._display_model_info(model_name)
            
            # Auto-start server if not running
            self._auto_start_server_if_needed()
            
            # Offer next steps
            self._show_post_load_options(model_name)
        else:
            console.print(f"[red]❌ Failed to load model '{model_name}'[/red]")
            console.print("[dim]Check model availability with 'list' command[/dim]")
    
    def _handle_unload_model(self, command: str):
        """Handle unload model command"""
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            loaded_models = list(self.api_server_agent.loaded_models.keys())
            if not loaded_models:
                console.print("[yellow]No models currently loaded[/yellow]")
                return
            
            console.print("\n[cyan]Loaded models:[/cyan]")
            for i, model in enumerate(loaded_models, 1):
                console.print(f"  {i}. {model}")
            
            choice = Prompt.ask("\nEnter model name or number")
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(loaded_models):
                    model_name = loaded_models[idx]
                else:
                    console.print("[red]Invalid choice[/red]")
                    return
            else:
                model_name = choice
        else:
            model_name = parts[1]
        
        console.print(f"[blue]Unloading model: {model_name}[/blue]")
        success = self.api_server_agent.unload_model(model_name)
        
        if success:
            console.print(f"[green]✅ Model {model_name} unloaded successfully[/green]")
        else:
            console.print(f"[red]❌ Failed to unload model {model_name}[/red]")
    
    def _handle_show_status(self):
        """Show server and model status"""
        # Server status
        if hasattr(self.api_server_agent, 'server_thread') and self.api_server_agent.server_thread:
            console.print(f"[green]🟢 Server: Running on {self.api_server_agent.server_host}:{self.api_server_agent.server_port}[/green]")
            console.print(f"[dim]   API docs: http://{self.api_server_agent.server_host}:{self.api_server_agent.server_port}/docs[/dim]")
        else:
            console.print("[red]🔴 Server: Not running[/red]")
        
        # Loaded models
        loaded_models = self.api_server_agent.loaded_models
        if loaded_models:
            console.print(f"\n[cyan]Loaded Models ({len(loaded_models)}):[/cyan]")
            
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Model", style="white")
            table.add_column("Quantization", style="yellow")
            table.add_column("Bit Width", style="green")
            
            for model_name, model_info in loaded_models.items():
                table.add_row(
                    model_name,
                    model_info.quantization_method,
                    f"{model_info.bit_width}-bit"
                )
            
            console.print(table)
        else:
            console.print("\n[yellow]No models loaded[/yellow]")
        
        # Available models
        available_models = self.api_server_agent.discover_quantized_models()
        console.print(f"\n[dim]Available quantized models: {len(available_models)}[/dim]")
    
    def _handle_start_server(self, command: str):
        """Start the API server"""
        # Extract port from command if provided
        parts = command.split()
        port = 8000  # default port
        
        if len(parts) > 1 and parts[1].isdigit():
            port = int(parts[1])
        
        console.print(f"[blue]Starting API server on port {port}...[/blue]")
        success = self.api_server_agent.start_server_background(port)
        
        if success:
            console.print(f"[green]✅ Server started on http://localhost:{port}[/green]")
            console.print(f"[dim]API Documentation: http://localhost:{port}/docs[/dim]")
        else:
            console.print("[red]❌ Failed to start server[/red]")
    
    def _handle_stop_server(self):
        """Stop the API server"""
        console.print("[blue]Stopping API server...[/blue]")
        success = self.api_server_agent.stop_server()
        
        if success:
            console.print("[green]✅ Server stopped[/green]")
        else:
            console.print("[red]❌ Failed to stop server or server was not running[/red]")
    
    def _handle_test_api(self, command: str):
        """Test API endpoints"""
        parts = command.split(maxsplit=2)
        
        if len(parts) < 2:
            console.print("[red]Usage: test <model_name> [message] or test health[/red]")
            return
        
        if parts[1] == 'health':
            self._test_health_endpoint()
            return
        
        model_name = parts[1]
        message = parts[2] if len(parts) > 2 else "Hello, how are you?"
        
        # Check if model is loaded
        if model_name not in self.api_server_agent.loaded_models:
            console.print(f"[red]Model {model_name} is not loaded. Load it first with: load {model_name}[/red]")
            return
        
        # Check if server is running
        if not (hasattr(self.api_server_agent, 'server_thread') and self.api_server_agent.server_thread):
            console.print("[red]Server is not running. Start it first with: start[/red]")
            return
        
        self._test_model_endpoint(model_name, message)
    
    def _test_health_endpoint(self) -> None:
        """
        Test the health endpoint with comprehensive error handling.
        """
        url = f"http://{self.api_server_agent.server_host}:{self.api_server_agent.server_port}/health"
        response = self._safe_request(url, timeout=CLIConfig.HEALTH_CHECK_TIMEOUT)
        
        if response:
            if response.status_code == 200:
                console.print("[green]✅ Health check passed[/green]")
                try:
                    health_data = response.json()
                    console.print(f"[dim]Response: {health_data}[/dim]")
                except ValueError:
                    console.print("[dim]Server responded but with invalid JSON[/dim]")
            else:
                console.print(f"[red]❌ Health check failed: {response.status_code}[/red]")
    
    def _test_model_endpoint(self, model_name: str, message: str) -> None:
        """
        Test a model endpoint with comprehensive error handling.
        
        Args:
            model_name: Name of the model to test
            message: Test message to send
        """
        url = f"http://{self.api_server_agent.server_host}:{self.api_server_agent.server_port}/models/{model_name}/chat"
        
        payload = {
            "message": message,
            "max_tokens": 100,
            "temperature": CLIConfig.DEFAULT_TEMPERATURE
        }
        
        console.print(f"[blue]🧪 Testing {model_name} with: '{message}'[/blue]")
        
        response = self._safe_request(url, "POST", json=payload)
        
        if response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    console.print("[green]✅ API test successful[/green]")
                    console.print(f"[cyan]Response:[/cyan] {result.get('response', 'No response')}")
                    
                    if 'stats' in result:
                        stats = result['stats']
                        console.print(f"[dim]📊 Tokens: {stats.get('total_tokens', 'N/A')}, "
                                    f"Time: {stats.get('response_time_ms', 'N/A')}ms[/dim]")
                except ValueError as e:
                    console.print(f"[red]❌ Invalid response format: {e}[/red]")
            else:
                console.print(f"[red]❌ API test failed: {response.status_code}[/red]")
                if response.text:
                    console.print(f"[dim]Error: {response.text}[/dim]")
    
    def _handle_list_models(self):
        """List available quantized models"""
        available_models = self.api_server_agent.discover_quantized_models()
        
        if not available_models:
            console.print("[yellow]No quantized models found[/yellow]")
            console.print("[dim]Run quantization using the main CLI first[/dim]")
            return
        
        console.print(f"\n[cyan]Available Quantized Models ({len(available_models)}):[/cyan]")
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Model", style="white")
        table.add_column("Method", style="yellow")
        table.add_column("Bits", style="green")
        table.add_column("Status", style="blue")
        
        for model_info in available_models:
            model_name = model_info.get('display_name', model_info.get('model_name', 'Unknown'))
            quantization_method = model_info.get('quantization_method', 'Unknown')
            bit_width = model_info.get('bit_width', 'Unknown')
            
            # Check if loaded
            is_loaded = model_name in self.api_server_agent.loaded_models
            status = "🟢 Loaded" if is_loaded else "⚫ Available"
            
            table.add_row(
                model_name,
                quantization_method,
                f"{bit_width}-bit" if bit_width != 'Unknown' else 'Unknown',
                status
            )
        
        console.print(table)
    
    def _handle_show_loaded_models(self):
        """Show currently loaded models"""
        loaded_models = self.api_server_agent.loaded_models
        
        if not loaded_models:
            console.print("[yellow]No models currently loaded[/yellow]")
            console.print("[dim]Use 'load <model_name>' to load a model[/dim]")
            return
        
        console.print(f"\n[cyan]Currently Loaded Models ({len(loaded_models)}):[/cyan]")
        
        table = Table(show_header=True, header_style="bold green")
        table.add_column("Model", style="white")
        table.add_column("Quantization", style="yellow")
        table.add_column("Bit Width", style="green")
        table.add_column("Status", style="blue")
        
        for model_name, model_info in loaded_models.items():
            table.add_row(
                model_name,
                model_info.quantization_method,
                f"{model_info.bit_width}-bit",
                f"🟢 {model_info.status.value.title()}"
            )
        
        console.print(table)
    
    def _handle_clear_screen(self):
        """Clear the screen"""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
        console.print(Panel.fit(
            "🚀 LQMF Simple API CLI - Model Serving Interface",
            style="bold blue"
        ))
    
    def _handle_chat_mode(self, command: str):
        """Handle interactive chat mode with a loaded model"""
        parts = command.split(maxsplit=1)
        model_name = None
        
        if len(parts) > 1:
            model_name = parts[1]
        
        # If no model specified, ask user to choose from loaded models
        if not model_name:
            loaded_models = list(self.api_server_agent.loaded_models.keys())
            if not loaded_models:
                console.print("[red]No models currently loaded. Load a model first with 'load <model_name>'[/red]")
                return
            
            if len(loaded_models) == 1:
                model_name = loaded_models[0]
                console.print(f"[blue]Using loaded model: {model_name}[/blue]")
            else:
                console.print("\n[cyan]Select a model for chat:[/cyan]")
                for i, model in enumerate(loaded_models, 1):
                    console.print(f"  {i}. {model}")
                
                choice = Prompt.ask("\nEnter model name or number")
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(loaded_models):
                        model_name = loaded_models[idx]
                    else:
                        console.print("[red]Invalid choice[/red]")
                        return
                else:
                    model_name = choice
        
        # Verify model is loaded
        if model_name not in self.api_server_agent.loaded_models:
            console.print(f"[red]Model {model_name} is not loaded. Load it first with 'load {model_name}'[/red]")
            return
        
        # Verify server is running
        if not (hasattr(self.api_server_agent, 'server_thread') and self.api_server_agent.server_thread):
            console.print("[red]Server is not running. Starting server...[/red]")
            success = self.api_server_agent.start_server_background(8000)
            if not success:
                console.print("[red]Failed to start server. Use 'start' command manually[/red]")
                return
        
        # Initialize chat session
        import time
        self.current_chat_session = ChatSession(
            model_name=model_name,
            start_time=time.time()
        )
        
        # Start chat session
        console.print(f"\n[green]🤖 Starting chat with {model_name}[/green]")
        console.print("[dim]Type 'exit', 'quit', or press Ctrl+C to end chat[/dim]")
        console.print("[dim]" + "─" * 60 + "[/dim]")
        
        while True:
            try:
                user_message = Prompt.ask(f"\n[bold blue]You[/bold blue]", default="").strip()
                
                if not user_message:
                    continue
                
                if user_message.lower() in ['exit', 'quit', 'end', 'stop']:
                    console.print("[yellow]Ending chat session[/yellow]")
                    break
                
                # Send to API with improved error handling
                console.print("[dim]🤔 Thinking...[/dim]")
                
                url = f"http://{self.api_server_agent.server_host}:{self.api_server_agent.server_port}/models/{model_name}/chat"
                
                payload = {
                    "message": user_message,
                    "max_tokens": CLIConfig.MAX_CHAT_TOKENS,
                    "temperature": CLIConfig.DEFAULT_TEMPERATURE
                }
                
                response = self._safe_request(url, "POST", json=payload)
                
                if response:
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            model_response = result.get('response', 'No response received')
                            
                            console.print(f"[bold green]{model_name}[/bold green]: {model_response}")
                            
                            # Update chat session stats
                            if self.current_chat_session:
                                self.current_chat_session.message_count += 1
                                if 'stats' in result:
                                    self.current_chat_session.total_tokens += result['stats'].get('total_tokens', 0)
                            
                            # Show stats if available
                            if 'stats' in result:
                                stats = result['stats']
                                tokens = stats.get('total_tokens', 'N/A')
                                time_ms = stats.get('response_time_ms', 'N/A')
                                console.print(f"[dim]({tokens} tokens, {time_ms}ms)[/dim]")
                        except ValueError as e:
                            console.print(f"[red]❌ Invalid response format: {e}[/red]")
                    else:
                        console.print(f"[red]❌ API Error: {response.status_code}[/red]")
                        if response.text:
                            console.print(f"[dim]{response.text}[/dim]")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Chat session ended[/yellow]")
                break
            except Exception as e:
                logger.error(f"Chat session error: {e}")
                console.print(f"[red]Unexpected error: {e}[/red]")
        
        # Show chat session summary
        self._show_chat_summary()
    
    def _handle_show_config(self):
        """Show configuration information"""
        console.print("\n[cyan]📋 API CLI Configuration:[/cyan]")
        
        config_info = f"""
[bold]Server Configuration:[/bold]
  Host: {self.api_server_agent.server_host}
  Default Port: 8000
  Quantized Models Directory: {self.api_server_agent.quantized_models_dir}

[bold]Current Status:[/bold]
  Server Running: {'🟢 Yes' if hasattr(self.api_server_agent, 'server_thread') and self.api_server_agent.server_thread else '🔴 No'}
  Loaded Models: {len(self.api_server_agent.loaded_models)}
  Available Models: {len(self.api_server_agent.discover_quantized_models())}

[bold]Hardware:[/bold]
  GPU Available: {'🟢 Yes' if TORCH_AVAILABLE and torch.cuda.is_available() else '🔴 No'}
  GPU Count: {torch.cuda.device_count() if TORCH_AVAILABLE and torch.cuda.is_available() else 0}
        """
        
        console.print(Panel(config_info.strip(), title="Configuration", style="cyan"))
    
    def _handle_exit(self):
        """Handle exit command"""
        if hasattr(self.api_server_agent, 'server_thread') and self.api_server_agent.server_thread:
            if Confirm.ask("Stop the API server before exiting?", default=True):
                self.api_server_agent.stop_server()
        
        console.print("\n[green]Thanks for using LQMF API CLI! Goodbye! 👋[/green]")
    
    # Helper methods for improved functionality
    def _validate_model_name(self, model_name: str) -> bool:
        """
        Validate model name format and availability.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            True if model name is valid, False otherwise
        """
        if not model_name or not model_name.strip():
            console.print("[red]❌ Model name cannot be empty[/red]")
            return False
            
        # Check for obviously invalid characters
        invalid_chars = ['<', '>', '|', '?', '*']
        if any(char in model_name for char in invalid_chars):
            console.print(f"[red]❌ Invalid characters in model name: {model_name}[/red]")
            return False
            
        return True
    
    def _display_model_info(self, model_name: str) -> None:
        """
        Display detailed information about a loaded model.
        
        Args:
            model_name: Name of the loaded model
        """
        try:
            if model_name in self.api_server_agent.loaded_models:
                model_info = self.api_server_agent.loaded_models[model_name]
                console.print(f"[dim]📊 Quantization: {model_info.quantization_method}, "
                            f"Bit width: {model_info.bit_width}-bit[/dim]")
        except Exception as e:
            logger.warning(f"Could not display model info: {e}")
    
    def _auto_start_server_if_needed(self) -> None:
        """Automatically start the API server if it's not already running."""
        if not (hasattr(self.api_server_agent, 'server_thread') and self.api_server_agent.server_thread):
            console.print("[blue]🚀 Auto-starting API server...[/blue]")
            try:
                server_success = self.api_server_agent.start_server_background(CLIConfig.DEFAULT_PORT)
                if server_success:
                    console.print(f"[green]✅ API server started on http://localhost:{CLIConfig.DEFAULT_PORT}[/green]")
                    console.print(f"[dim]📚 API Documentation: http://localhost:{CLIConfig.DEFAULT_PORT}/docs[/dim]")
                else:
                    console.print("[yellow]⚠️  Server auto-start failed, use 'start' command manually[/yellow]")
            except Exception as e:
                logger.error(f"Server auto-start failed: {e}")
                console.print("[yellow]⚠️  Server auto-start failed, use 'start' command manually[/yellow]")
    
    def _show_post_load_options(self, model_name: str) -> None:
        """
        Show available options after successful model loading.
        
        Args:
            model_name: Name of the loaded model
        """
        console.print("\n[cyan]💬 Model loaded! Available options:[/cyan]")
        console.print(f"[dim]  • Type [bold]'chat'[/bold] to start interactive chat with {model_name}[/dim]")
        console.print(f"[dim]  • Use [bold]'test {model_name} <message>'[/bold] for single queries[/dim]")
        console.print(f"[dim]  • Check [bold]'status'[/bold] for system overview[/dim]")
        console.print(f"[dim]  • Visit [bold]http://localhost:{CLIConfig.DEFAULT_PORT}/docs[/bold] for API docs[/dim]")
        
    def _safe_request(self, url: str, method: str = "GET", **kwargs) -> Optional[requests.Response]:
        """
        Make a safe HTTP request with error handling.
        
        Args:
            url: The URL to request
            method: HTTP method (GET, POST, etc.)
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object if successful, None otherwise
        """
        try:
            if method.upper() == "POST":
                response = requests.post(url, timeout=CLIConfig.REQUEST_TIMEOUT, **kwargs)
            else:
                response = requests.get(url, timeout=CLIConfig.REQUEST_TIMEOUT, **kwargs)
            return response
        except requests.exceptions.ConnectionError:
            console.print(f"[red]❌ Connection failed - is the server running?[/red]")
            return None
        except requests.exceptions.Timeout:
            console.print(f"[red]❌ Request timeout - server may be overloaded[/red]")
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            console.print(f"[red]❌ Request failed: {e}[/red]")
            return None
    
    def _show_chat_summary(self) -> None:
        """Show summary of the completed chat session."""
        if self.current_chat_session:
            import time
            session_duration = time.time() - self.current_chat_session.start_time
            
            console.print(f"\n[cyan]📊 Chat Session Summary:[/cyan]")
            console.print(f"[dim]Model: {self.current_chat_session.model_name}[/dim]")
            console.print(f"[dim]Messages: {self.current_chat_session.message_count}[/dim]")
            console.print(f"[dim]Total tokens: {self.current_chat_session.total_tokens}[/dim]")
            console.print(f"[dim]Duration: {session_duration:.1f}s[/dim]")
            
            self.current_chat_session = None

def main() -> None:
    """
    Main entry point for the LQMF API Command Line Interface.
    
    Initializes the CLI application, handles startup errors gracefully,
    and provides user-friendly error messages for common issues.
    
    Raises:
        SystemExit: On critical initialization failures
    """
    try:
        console.print("[dim]Initializing LQMF API CLI...[/dim]")
        cli = APICommandLineInterface()
        cli.run_interactive_session()
    except KeyboardInterrupt:
        console.print("\n[yellow]Startup interrupted by user[/yellow]")
        sys.exit(0)
    except ImportError as e:
        console.print(f"[red]❌ Missing dependencies: {e}[/red]")
        console.print("[dim]Please install required packages and try again[/dim]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error during startup: {e}")
        console.print(f"[red]❌ Critical error: {e}[/red]")
        console.print("[dim]Please check your installation and try again[/dim]")
        sys.exit(1)

if __name__ == "__main__":
    main()