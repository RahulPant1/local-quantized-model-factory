#!/usr/bin/env python3
"""
Local Quantized Model Factory (LQMF) - Main CLI Interface
Interactive chat-based interface for model quantization
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
import click
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live
from rich.spinner import Spinner
from rich.layout import Layout
from rich.text import Text
from rich.rule import Rule
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.planner_agent import PlannerAgent, QuantizationPlan
from agents.executor_agent import ExecutorAgent, ExecutionResult
from agents.memory_agent import MemoryAgent, StorageBackend, SimilarModel, QuantizationSuggestion, RecoveryAction, ModelSuggestion, ExperimentPlan, ComparisonReport, LearningPattern, PerformanceBenchmark, HuggingFaceModelRecommendation
from agents.feedback_agent import FeedbackAgent
from agents.enhanced_decision_agent import EnhancedDecisionAgent as DecisionAgent, UserIntent
from utils.gemini_helper import GeminiHelper

console = Console()

class LQMF:
    """Local Quantized Model Factory - Main Application"""
    
    def __init__(self):
        # Load config first
        self.config = self._load_config()
        
        # Initialize agents with config
        self.decision = DecisionAgent()  # Decision agent for intent understanding
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent(
            models_dir=self.config.get("models_dir", "models"),
            quantized_models_dir=self.config.get("quantized_models_dir", "quantized-models"),
            logs_dir=self.config.get("logs_dir", "logs")
        )
        self.memory = MemoryAgent(storage_dir=self.config.get("mcp_dir", "mcp"))
        self.feedback = FeedbackAgent(
            quantized_models_dir=self.config.get("quantized_models_dir", "quantized-models")
        )
        self.current_session = None
        
        # Display welcome message
        self._display_welcome()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        config_path = Path("config.json")
        default_config = {
            "storage_backend": "sqlite",
            "models_dir": "models",
            "quantized_models_dir": "quantized-models",
            "configs_dir": "configs",
            "logs_dir": "logs",
            "mcp_dir": "mcp",
            "gemini_api_key": os.getenv("GEMINI_API_KEY"),
            "auto_save_plans": True,
            "max_retries": 3
        }
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
        else:
            config = default_config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        return config
    
    def _display_welcome(self):
        """Display welcome message and system info"""
        welcome_text = """
# 🏭 Local Quantized Model Factory (LQMF)

**Interactive Agent-Powered Model Quantization System**

Welcome to your local quantization factory! This tool helps you:
- Select and download Hugging Face models
- Plan optimal quantization strategies
- Execute quantization with multiple backends
- Track experiments and results
- Run quantized models locally

**Available Commands:**
- `quantize <model-name>` - Start quantization process
- `test <model-name>` - Test a quantized model
- `test list` - List available quantized models
- `search <query>` - Search for models on HuggingFace
- `hf login/status/logout` - HuggingFace authentication
- `list` - Show previous experiments
- `stats` - Display statistics
- `help` - Show help information
- `exit` - Exit the application

**HuggingFace Commands:**
- `hf login` - Interactive HuggingFace authentication
- `hf status` - Check authentication status
- `hf logout` - Logout from HuggingFace

**Example Usage:**
```
> quantize mistralai/Mistral-7B-Instruct
> quantize llama2 to 4-bit GGUF for CPU
> test microsoft_DialoGPT-small_bnb_4bit
> test list
> list successful experiments
> stats
```
        """
        
        console.print(Panel(
            Markdown(welcome_text),
            title="🏭 LQMF - Local Quantized Model Factory",
            style="bold blue"
        ))
        
        # Show system capabilities
        self._show_system_info()
    
    def _show_system_info(self):
        """Display system capabilities"""
        import torch
        import psutil
        
        table = Table(title="System Information")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        # CUDA
        cuda_status = "✅ Available" if torch.cuda.is_available() else "❌ Not Available"
        cuda_details = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU Only"
        table.add_row("CUDA", cuda_status, cuda_details)
        
        # Memory
        gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB" if torch.cuda.is_available() else "N/A"
        cpu_mem = f"{psutil.virtual_memory().total / (1024**3):.1f}GB"
        table.add_row("GPU Memory", gpu_mem, "Available for quantization")
        table.add_row("System RAM", cpu_mem, "Available for CPU fallback")
        
        # Storage
        storage_backend = self.config.get("storage_backend", "sqlite").upper()
        table.add_row("Storage Backend", storage_backend, "For experiment tracking")
        
        # HuggingFace Authentication
        hf_status = self.executor.get_hf_auth_status()
        if hf_status['authenticated']:
            hf_status_text = "✅ Authenticated"
            hf_details = f"User: {hf_status['user_info'].username}"
        else:
            hf_status_text = "❌ Not Authenticated"
            hf_details = "Use 'hf login' to authenticate"
        table.add_row("HuggingFace", hf_status_text, hf_details)
        
        # Gemini API
        gemini_working = self.planner.model is not None
        gemini_status = "✅ Configured" if gemini_working else "❌ Not Configured"
        gemini_details = "Planning assistance enabled" if gemini_working else "Offline mode only"
        table.add_row("Gemini API", gemini_status, gemini_details)
        
        console.print(table)
        console.print("")
    
    def run(self):
        """Main interactive loop with intelligent command routing"""
        try:
            while True:
                # Get user input
                user_input = Prompt.ask(
                    "[bold blue]LQMF>[/bold blue]",
                    default="help"
                ).strip()
                
                if not user_input:
                    continue
                
                # Use Decision Agent to understand user intent
                intent_result = self.decision.analyze_user_intent(user_input)
                
                # Handle conversational responses first
                if not self.decision.should_route_to_agent(intent_result):
                    response = self.decision.generate_conversational_response(user_input, intent_result)
                    if response:
                        console.print(f"[blue]{response}[/blue]")
                        self.decision.add_to_conversation_history(user_input, response)
                        continue
                
                # Route to appropriate handlers based on intent
                if intent_result.intent == UserIntent.EXIT:
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    break
                elif intent_result.intent == UserIntent.SHOW_HELP:
                    self._show_help()
                elif intent_result.intent == UserIntent.SHOW_STATISTICS:
                    self._show_statistics()
                elif intent_result.intent == UserIntent.LIST_EXPERIMENTS:
                    self._list_experiments(user_input)
                elif intent_result.intent == UserIntent.QUANTIZE_MODEL:
                    self._handle_quantization(user_input)
                elif intent_result.intent == UserIntent.DOWNLOAD_MODEL:
                    self._handle_download(user_input, intent_result)
                elif intent_result.intent == UserIntent.TEST_MODEL:
                    self._handle_testing(user_input)
                elif intent_result.intent == UserIntent.HF_AUTHENTICATION:
                    self._handle_huggingface_intent(user_input, intent_result)
                elif intent_result.intent == UserIntent.SEARCH_MODELS:
                    self._handle_model_search(user_input)
                elif intent_result.intent == UserIntent.DISCOVER_MODELS:
                    self._handle_model_discovery(user_input)
                elif intent_result.intent == UserIntent.SHOW_CONFIG:
                    self._show_config()
                elif user_input.lower().startswith('insights'):
                    self._show_learning_insights()
                elif user_input.lower().startswith('recommend'):
                    self._handle_model_recommendations(user_input)
                elif user_input.lower().startswith('compare'):
                    self._handle_experiment_comparison(user_input)
                elif user_input.lower().startswith('suggest'):
                    self._handle_experiment_suggestions(user_input)
                elif user_input.lower().startswith('discover'):
                    self._handle_model_discovery(user_input)
                elif user_input.lower().startswith('find-models'):
                    self._handle_model_discovery(user_input)
                elif user_input.lower() == 'clear':
                    console.clear()
                    self._display_welcome()
                else:
                    # Fallback - show clarification message
                    response = self.decision.generate_conversational_response(user_input, intent_result)
                    console.print(f"[yellow]{response}[/yellow]")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user. Goodbye! 👋[/yellow]")
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            console.print("[yellow]Please check your configuration and try again.[/yellow]")
    
    def _show_help(self):
        """Display help information"""
        help_text = """
# 🏭 LQMF Help

## Commands:
- **quantize <model-name>** - Start quantization process
- **test <model-name>** - Test a quantized model
- **test list** - List available quantized models
- **test discover** - Discover all quantized models
- **list [filter]** - Show experiments (all/successful/failed)
- **stats** - Display statistics and metrics
- **insights** - Show learning insights and patterns
- **recommend <goals>** - Get model recommendations based on goals
- **compare <ids>** - Compare experiments by ID
- **suggest** - Get next experiment suggestions
- **discover <goals>** - Discover HuggingFace models for specific goals
- **find-models <goals>** - Find and recommend HuggingFace models
- **config** - Show current configuration
- **clear** - Clear screen and show welcome
- **help** - Show this help message
- **exit** - Exit the application

## Quantization Examples:
```
quantize mistralai/Mistral-7B-Instruct
quantize llama2 to 4-bit GGUF for CPU
quantize microsoft/DialoGPT-medium with GPTQ 8-bit
quantize facebook/opt-1.3b using BitsAndBytes
```

## Model Discovery Examples:
```
discover chat models for production
find-models lightweight models for mobile
discover code generation models
find-models fast inference models
```

## Testing Examples:
```
test microsoft_DialoGPT-small_bnb_4bit
test list
test discover
test DialoGPT (partial match)
```

## Natural Language Support:
You can use natural language to describe your quantization needs:
- "I want to quantize Mistral 7B for CPU inference"
- "Convert llama2 to 4-bit GGUF format"
- "Quantize GPT model with GPTQ for 8GB GPU"

## Storage Backends:
- **SQLite** (default) - Full-featured with indexing
- **JSON** - Simple file-based storage

## Quantization Methods:
- **GPTQ** - Fast GPU inference, good compression
- **GGUF** - CPU-friendly, llama.cpp compatible
- **BitsAndBytes** - Easy to use, experimentation-friendly
        """
        
        console.print(Panel(
            Markdown(help_text),
            title="Help",
            style="cyan"
        ))
    
    def _show_statistics(self):
        """Display quantization statistics"""
        console.print(Panel.fit("📊 Generating Statistics...", style="blue"))
        
        stats = self.memory.get_statistics()
        
        # Create main stats table
        table = Table(title="Quantization Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Experiments", str(stats["total_records"]))
        table.add_row("Successful", str(stats["successful_quantizations"]))
        table.add_row("Failed", str(stats["failed_quantizations"]))
        
        if stats["total_records"] > 0:
            success_rate = (stats["successful_quantizations"] / stats["total_records"]) * 100
            table.add_row("Success Rate", f"{success_rate:.1f}%")
        
        table.add_row("Avg Execution Time", f"{stats['avg_execution_time']:.2f}s")
        table.add_row("Storage Saved", f"{stats['total_storage_saved_mb']:.1f}MB")
        
        console.print(table)
        
        # Show quantization types distribution
        if stats["quantization_types"]:
            type_table = Table(title="Quantization Methods Used")
            type_table.add_column("Method", style="cyan")
            type_table.add_column("Count", style="green")
            type_table.add_column("Percentage", style="yellow")
            
            total = sum(stats["quantization_types"].values())
            for method, count in stats["quantization_types"].items():
                percentage = (count / total) * 100
                type_table.add_row(method.upper(), str(count), f"{percentage:.1f}%")
            
            console.print(type_table)
    
    def _list_experiments(self, command: str):
        """List previous experiments"""
        # Parse filter from command
        filter_success = None
        if "successful" in command.lower():
            filter_success = True
        elif "failed" in command.lower():
            filter_success = False
        
        records = self.memory.get_model_records(success_only=filter_success)
        
        if not records:
            console.print("[yellow]No experiments found.[/yellow]")
            return
        
        table = Table(title="Experiment History")
        table.add_column("Model", style="cyan")
        table.add_column("Method", style="green")
        table.add_column("Bits", style="yellow")
        table.add_column("Status", style="red")
        table.add_column("Size (MB)", style="blue")
        table.add_column("Time (s)", style="magenta")
        table.add_column("Date", style="white")
        
        for record in records[:20]:  # Show latest 20
            status = "✅ Success" if record.success else "❌ Failed"
            size = f"{record.quantized_size_mb:.1f}" if record.quantized_size_mb else "N/A"
            exec_time = f"{record.execution_time:.2f}" if record.execution_time else "N/A"
            date = record.created_at.split('T')[0]  # Just date part
            
            table.add_row(
                record.model_name,
                record.quantization_type.upper(),
                f"{record.bit_width}-bit",
                status,
                size,
                exec_time,
                date
            )
        
        console.print(table)
        
        if len(records) > 20:
            console.print(f"[dim]Showing latest 20 of {len(records)} experiments[/dim]")
    
    def _handle_quantization(self, user_input: str):
        """Handle quantization request with enhanced MCP intelligence"""
        console.print(Panel.fit("🧠 Starting Intelligent Quantization Process", style="bold blue"))
        
        try:
            # Extract model name for MCP analysis
            model_name = self._extract_model_name_from_input(user_input)
            
            # Show MCP-powered insights before planning
            if model_name:
                self._show_mcp_insights(model_name)
            
            # Plan quantization
            plan = self.planner.interactive_planning(user_input)
            
            if not plan:
                console.print("[yellow]Quantization cancelled.[/yellow]")
                return
            
            # Save plan if configured
            if self.config.get("auto_save_plans", True):
                plan_path = Path(self.config["configs_dir"]) / f"{plan.model_name.replace('/', '_')}_plan.json"
                plan_path.parent.mkdir(exist_ok=True)
                self.planner.save_plan(plan, str(plan_path))
            
            # Execute quantization
            console.print(Panel.fit("⚙️ Executing Quantization", style="bold green"))
            
            # Show progress
            with Live(Spinner("dots", text="Quantizing model..."), refresh_per_second=10):
                result = self.executor.execute_quantization(plan)
            
            # Store results in memory
            record_id = self.memory.store_quantization_attempt(plan, result)
            
            # Log completion
            self.memory.log_experiment_step(
                record_id,
                "quantization_complete",
                "Quantization process completed" if result.success else "Quantization failed",
                "INFO" if result.success else "ERROR"
            )
            
            # Show final results
            if result.success:
                console.print(Panel.fit("✅ Quantization Successful!", style="bold green"))
                console.print(f"[green]Model saved to: {result.output_path}[/green]")
                console.print(f"[green]Size: {result.model_size_mb:.1f}MB[/green]")
                console.print(f"[green]Time: {result.execution_time:.2f}s[/green]")
                
                # Ask about next steps
                if Confirm.ask("Would you like to run a test inference?"):
                    self._test_inference(result.output_path, plan)
            else:
                console.print(Panel.fit("❌ Quantization Failed", style="bold red"))
                console.print(f"[red]Error: {result.error_message}[/red]")
                
                # Enhanced MCP-powered remediation
                self._suggest_mcp_remediation(plan, result)
        
        except Exception as e:
            console.print(f"[red]Error during quantization: {e}[/red]")
            console.print("[yellow]Please check your configuration and try again.[/yellow]")
    
    def _test_inference(self, model_path: str, plan: QuantizationPlan):
        """Test inference with quantized model"""
        console.print(Panel.fit("🧪 Testing Inference", style="blue"))
        
        try:
            # This is a simplified test - in practice, you'd load the model
            # and run actual inference
            test_prompt = Prompt.ask(
                "Enter a test prompt",
                default="Hello, how are you?"
            )
            
            console.print(f"[blue]Testing with prompt: {test_prompt}[/blue]")
            
            # Simulate inference (replace with actual inference code)
            with Live(Spinner("dots", text="Running inference..."), refresh_per_second=10):
                time.sleep(2)  # Simulate inference time
            
            console.print("[green]✅ Inference test completed successfully![/green]")
            console.print("[dim]Note: This is a simulation. Implement actual inference loading for production use.[/dim]")
            
        except Exception as e:
            console.print(f"[red]Inference test failed: {e}[/red]")
    
    def _handle_testing(self, user_input: str):
        """Handle model testing commands"""
        parts = user_input.split()
        
        if len(parts) < 2:
            console.print("[red]Usage: test <model-name|list|discover>[/red]")
            return
        
        command = parts[1].lower()
        
        if command == "list" or command == "discover":
            self._list_quantized_models()
        else:
            # Test specific model
            model_name = " ".join(parts[1:])  # Handle models with spaces
            self._test_quantized_model(model_name)
    
    def _list_quantized_models(self):
        """List all available quantized models"""
        console.print(Panel.fit("🔍 Discovering Quantized Models", style="bold blue"))
        
        try:
            models = self.feedback.discover_quantized_models()
            
            if not models:
                console.print("[yellow]No quantized models found in the quantized-models directory.[/yellow]")
                console.print("[dim]Run quantization first to create models for testing.[/dim]")
                return
            
            table = Table(title="Available Quantized Models")
            table.add_column("Model Name", style="cyan")
            table.add_column("Original Model", style="green")
            table.add_column("Method", style="yellow")
            table.add_column("Bits", style="magenta")
            table.add_column("Architecture", style="blue")
            table.add_column("Size (MB)", style="red")
            table.add_column("Files", style="white")
            
            for model in models:
                table.add_row(
                    model["name"],
                    model["original_model"],
                    model["quantization_method"].upper(),
                    f"{model['bit_width']}-bit" if model["bit_width"] != "unknown" else "unknown",
                    model["architecture"],
                    f"{model['size_mb']:.1f}",
                    str(model["files_count"])
                )
            
            console.print(table)
            console.print(f"[green]Found {len(models)} quantized models[/green]")
            
            # Ask if user wants to test one
            if Confirm.ask("Would you like to test one of these models?"):
                model_name = Prompt.ask("Enter model name to test")
                if model_name:
                    self._test_quantized_model(model_name)
        
        except Exception as e:
            console.print(f"[red]Error discovering models: {e}[/red]")
    
    def _test_quantized_model(self, model_name: str):
        """Test a specific quantized model"""
        console.print(Panel.fit(f"🧪 Testing Model: {model_name}", style="bold blue"))
        
        try:
            # Ask for test mode
            comprehensive = Confirm.ask("Run comprehensive test?", default=True)
            
            # Run the test
            console.print("[blue]Starting model test...[/blue]")
            result = self.feedback.test_quantized_model(model_name, comprehensive=comprehensive)
            
            if result.success:
                console.print(Panel.fit("✅ Model Test Completed Successfully!", style="bold green"))
                
                # Store test result in memory if we have integration
                self._store_test_result(model_name, result)
                
                # Ask about next steps
                if Confirm.ask("Would you like to run another test?"):
                    self._handle_testing("test list")
            else:
                console.print(Panel.fit("❌ Model Test Failed", style="bold red"))
                console.print(f"[red]Error: {result.error_message}[/red]")
                
                # Suggest remediation
                self._suggest_test_remediation(model_name, result)
        
        except Exception as e:
            console.print(f"[red]Error testing model: {e}[/red]")
            console.print("[yellow]Please check that the model exists and is accessible.[/yellow]")
    
    def _store_test_result(self, model_name: str, result):
        """Store test result in memory system"""
        try:
            # Create a simple log entry for the test
            self.memory.log_experiment_step(
                model_name,  # Use model name as pseudo record ID
                "model_test",
                f"Model test {'successful' if result.success else 'failed'}",
                "INFO" if result.success else "ERROR"
            )
            console.print("[dim]Test result stored in memory database[/dim]")
        except Exception as e:
            console.print(f"[yellow]Could not store test result: {e}[/yellow]")
    
    def _suggest_test_remediation(self, model_name: str, result):
        """Suggest remediation for failed model tests"""
        console.print(Panel.fit("💡 Test Remediation Suggestions", style="yellow"))
        
        suggestions = []
        
        if "not found" in result.error_message.lower():
            suggestions.append("• Check that the model name is correct")
            suggestions.append("• Use 'test list' to see available models")
            suggestions.append("• Try partial name matching")
        
        if "load" in result.error_message.lower():
            suggestions.append("• Check that the model files are not corrupted")
            suggestions.append("• Verify sufficient memory is available")
            suggestions.append("• Try restarting the application")
        
        if "memory" in result.error_message.lower():
            suggestions.append("• Close other applications to free memory")
            suggestions.append("• Try testing with a smaller model")
            suggestions.append("• Enable CPU fallback if available")
        
        if not suggestions:
            suggestions.append("• Check the error details above")
            suggestions.append("• Try testing a different model")
            suggestions.append("• Check model compatibility")
        
        for suggestion in suggestions:
            console.print(suggestion)
        
        if Confirm.ask("Would you like to try testing a different model?"):
            self._handle_testing("test list")
    
    def _handle_download(self, user_input: str, intent_result):
        """Handle model download requests"""
        model_name = intent_result.extracted_info.get('model_name')
        
        if not model_name:
            model_name = Prompt.ask("Which model would you like to download?")
        
        console.print(Panel.fit(f"📥 Downloading Model: {model_name}", style="bold blue"))
        
        try:
            # Download the model
            success, result = self.executor.download_model(model_name)
            
            if success:
                console.print(Panel.fit("✅ Download Successful!", style="bold green"))
                console.print(f"[green]Model downloaded to: {result}[/green]")
                
                # Ask if user wants to quantize it now
                if Confirm.ask("Would you like to quantize this model now?"):
                    self._handle_quantization(f"quantize {model_name}")
            else:
                console.print(Panel.fit("❌ Download Failed", style="bold red"))
                console.print(f"[red]Error: {result}[/red]")
        
        except Exception as e:
            console.print(f"[red]Error during download: {e}[/red]")
    
    def _handle_model_search(self, user_input: str):
        """Handle model search requests"""
        # Extract search query from user input
        parts = user_input.split()
        if len(parts) < 2:
            query = Prompt.ask("What model are you looking for?", default="llama")
        else:
            # Remove "search" and use the rest as query
            if parts[0].lower() == "search":
                query = " ".join(parts[1:])
            else:
                # Extract keywords that might be model names
                query = " ".join([word for word in parts if word.lower() not in ['find', 'look', 'for', 'show', 'browse']])
        
        console.print(Panel.fit(f"🔍 Searching for models: {query}", style="bold blue"))
        
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            token = self.executor.hf_auth.get_current_token()
            
            # Search for models
            console.print("[blue]Searching HuggingFace Hub...[/blue]")
            models = list(api.list_models(
                search=query, 
                limit=10,
                token=token,
                sort="downloads",
                direction=-1
            ))
            
            if not models:
                console.print(f"[yellow]No models found for '{query}'[/yellow]")
                console.print("[dim]Try different keywords or check the exact model name[/dim]")
                return
            
            # Display results
            table = Table(title=f"Search Results for '{query}'")
            table.add_column("Model Name", style="cyan")
            table.add_column("Downloads", style="green")
            table.add_column("Tags", style="yellow")
            table.add_column("Updated", style="blue")
            
            for model in models:
                tags = ", ".join(model.tags[:3]) if model.tags else "N/A"
                downloads = f"{model.downloads:,}" if model.downloads else "N/A"
                updated = model.lastModified.strftime("%Y-%m-%d") if model.lastModified else "N/A"
                
                table.add_row(
                    model.modelId,
                    downloads,
                    tags,
                    updated
                )
            
            console.print(table)
            
            # Ask if user wants to quantize one of these models
            if Confirm.ask("Would you like to quantize one of these models?"):
                model_name = Prompt.ask("Enter the model name to quantize")
                if model_name:
                    self._handle_quantization(f"quantize {model_name}")
        
        except Exception as e:
            console.print(f"[red]Error searching models: {e}[/red]")
            console.print("[yellow]You can also browse models at https://huggingface.co/models[/yellow]")
    
    def _handle_huggingface_intent(self, user_input: str, intent_result):
        """Handle HuggingFace authentication with intelligent routing"""
        # Check if this is a structured command (like "hf login")
        if user_input.lower().startswith('hf '):
            self._handle_huggingface_commands(user_input)
            return
        
        # Handle natural language HF requests
        extracted_info = intent_result.extracted_info
        action = extracted_info.get('action', 'status')  # Default to status
        
        # Provide conversational feedback
        if 'login' in user_input.lower():
            console.print("[blue]I'll help you login to HuggingFace![/blue]")
            self._hf_login()
        elif 'status' in user_input.lower() or 'check' in user_input.lower():
            console.print("[blue]Let me check your HuggingFace authentication status.[/blue]")
            self._hf_status()
        elif 'logout' in user_input.lower():
            console.print("[blue]I'll help you logout from HuggingFace.[/blue]")
            self._hf_logout()
        else:
            # Default behavior - show status and ask what they want to do
            console.print("[blue]I can help you with HuggingFace authentication.[/blue]")
            self._hf_status()
            
            if not self.executor.get_hf_auth_status()['authenticated']:
                if Confirm.ask("Would you like to login to HuggingFace now?"):
                    self._hf_login()
    
    def _handle_huggingface_commands(self, user_input: str):
        """Handle HuggingFace authentication commands"""
        parts = user_input.split()
        
        if len(parts) < 2:
            console.print("[red]Usage: hf <command>[/red]")
            console.print("[dim]Available commands: login, status, logout[/dim]")
            return
        
        command = parts[1].lower()
        
        if command == "login":
            self._hf_login()
        elif command == "status":
            self._hf_status()
        elif command == "logout":
            self._hf_logout()
        else:
            console.print(f"[red]Unknown HuggingFace command: {command}[/red]")
            console.print("[dim]Available commands: login, status, logout[/dim]")
    
    def _hf_login(self):
        """Handle HuggingFace login"""
        console.print(Panel.fit("🤗 HuggingFace Login", style="bold blue"))
        
        # Check if already authenticated
        status = self.executor.get_hf_auth_status()
        if status['authenticated']:
            console.print(f"[green]Already authenticated as {status['user_info'].username}[/green]")
            if not Confirm.ask("Would you like to login with a different account?"):
                return
        
        # Perform login
        success = self.executor.login_to_huggingface()
        if success:
            console.print("[green]🎉 Login successful! You can now access private models.[/green]")
        else:
            console.print("[yellow]Login cancelled or failed.[/yellow]")
    
    def _hf_status(self):
        """Display HuggingFace authentication status"""
        console.print(Panel.fit("🤗 HuggingFace Status", style="bold blue"))
        
        status = self.executor.get_hf_auth_status()
        
        status_table = Table(title="HuggingFace Authentication Status")
        status_table.add_column("Property", style="cyan")
        status_table.add_column("Value", style="green")
        
        status_table.add_row("Status", "✅ Authenticated" if status['authenticated'] else "❌ Not Authenticated")
        
        if status['authenticated']:
            user_info = status['user_info']
            status_table.add_row("Username", user_info.username)
            status_table.add_row("Full Name", user_info.fullname or "N/A")
            status_table.add_row("Plan", user_info.plan or "free")
            status_table.add_row("Token Source", status['token_source'])
            
            if user_info.orgs:
                status_table.add_row("Organizations", ", ".join(user_info.orgs))
        else:
            status_table.add_row("Token Source", "None")
            status_table.add_row("Access", "Public models only")
            status_table.add_row("Suggestion", "Use 'hf login' to authenticate")
        
        console.print(status_table)
        
        # Test access
        if status['authenticated']:
            console.print("\n[blue]Testing HuggingFace access...[/blue]")
            if self.executor.hf_auth.test_access():
                console.print("[green]✅ HuggingFace access working correctly![/green]")
            else:
                console.print("[red]❌ HuggingFace access test failed.[/red]")
    
    def _hf_logout(self):
        """Handle HuggingFace logout"""
        console.print(Panel.fit("🤗 HuggingFace Logout", style="bold blue"))
        
        status = self.executor.get_hf_auth_status()
        if not status['authenticated']:
            console.print("[yellow]Not currently authenticated with HuggingFace.[/yellow]")
            return
        
        if Confirm.ask(f"Are you sure you want to logout from HuggingFace ({status['user_info'].username})?"):
            self.executor.logout_from_huggingface()
            console.print("[green]🎉 Logout successful![/green]")
        else:
            console.print("[yellow]Logout cancelled.[/yellow]")
    
    def _suggest_remediation(self, plan: QuantizationPlan, result: ExecutionResult):
        """Suggest remediation steps for failed quantization"""
        console.print(Panel.fit("💡 Suggested Remediation", style="yellow"))
        
        suggestions = []
        
        if "memory" in result.error_message.lower():
            suggestions.append("• Try enabling CPU fallback")
            suggestions.append("• Reduce batch size or sequence length")
            suggestions.append("• Use a different quantization method")
        
        if "compatibility" in result.error_message.lower():
            suggestions.append("• Check model architecture compatibility")
            suggestions.append("• Try a different quantization backend")
        
        if "download" in result.error_message.lower() or "404" in result.error_message:
            suggestions.append("• Check internet connection")
            suggestions.append("• Verify model name is correct (use exact HuggingFace repo name)")
            suggestions.append("• Try using a Hugging Face token for private models")
            
            # Check if this looks like a model name issue
            if "404" in result.error_message or "not found" in result.error_message.lower():
                suggestions.append("• Model name might be incorrect - see suggestions above")
                suggestions.append("• Check HuggingFace Model Hub for exact model names")
                suggestions.append("• Some models require access approval (gated models)")
        
        if not suggestions:
            suggestions.append("• Check the error log for more details")
            suggestions.append("• Try a different model or quantization method")
        
        for suggestion in suggestions:
            console.print(suggestion)
        
        if Confirm.ask("Would you like to try again with different settings?"):
            self._handle_quantization(f"quantize {plan.model_name}")
    
    def _show_config(self):
        """Display current configuration"""
        console.print(Panel.fit("⚙️ Current Configuration", style="cyan"))
        
        table = Table()
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in self.config.items():
            if key == "gemini_api_key":
                display_value = "***configured***" if value else "not configured"
            else:
                display_value = str(value)
            
            table.add_row(key, display_value)
        
        console.print(table)
    
    # === ENHANCED MCP FEATURES INTEGRATION ===
    
    def _extract_model_name_from_input(self, user_input: str) -> Optional[str]:
        """Extract model name from user input"""
        # Remove common command words
        words = user_input.split()
        filtered_words = [w for w in words if w.lower() not in ['quantize', 'to', 'with', 'using', 'for', 'cpu', 'gpu']]
        
        # Look for model patterns (org/model or just model)
        for word in filtered_words:
            if '/' in word or any(family in word.lower() for family in ['llama', 'mistral', 'gpt', 'opt', 'bloom']):
                return word
        
        # If no clear pattern, return the first meaningful word
        return filtered_words[0] if filtered_words else None
    
    def _show_mcp_insights(self, model_name: str):
        """Show MCP-powered insights before quantization"""
        console.print(Panel.fit("🔍 Analyzing Historical Data", style="cyan"))
        
        try:
            # Get contextual insights
            insights = self.memory.get_contextual_insights(model_name)
            
            # Show similar models count
            similar_count = insights["similar_models_count"]
            if similar_count > 0:
                console.print(f"[blue]📊 Found {similar_count} similar models in history[/blue]")
            else:
                console.print(f"[yellow]📊 No similar models found - this will be a new exploration[/yellow]")
            
            # Show confidence and recommendations
            confidence = insights["confidence_score"]
            strategy = insights["recommended_strategy"]
            
            if confidence > 0.6:
                confidence_color = "green"
                confidence_text = "High"
            elif confidence > 0.4:
                confidence_color = "yellow"
                confidence_text = "Medium"
            else:
                confidence_color = "red"
                confidence_text = "Low"
            
            console.print(f"[{confidence_color}]🎯 Confidence Level: {confidence_text} ({confidence:.1%})[/{confidence_color}]")
            
            # Show recommendations
            if strategy:
                console.print(Panel(
                    f"**Recommended Strategy:**\n"
                    f"• Method: {strategy['method'].upper()}\n"
                    f"• Bit Width: {strategy['bit_width']}-bit\n"
                    f"• Format: {strategy['format']}\n"
                    f"• CPU Fallback: {'Yes' if strategy['cpu_fallback'] else 'No'}",
                    title="🤖 MCP Suggestion",
                    style="blue"
                ))
            
            # Show success indicators
            if insights["success_indicators"]:
                console.print("[green]✅ Success Indicators:[/green]")
                for indicator in insights["success_indicators"]:
                    console.print(f"  • {indicator}")
            
            # Show risk factors
            if insights["risk_factors"]:
                console.print("[red]⚠️  Risk Factors:[/red]")
                for risk in insights["risk_factors"]:
                    console.print(f"  • {risk}")
            
            console.print("")  # Add spacing
            
        except Exception as e:
            console.print(f"[yellow]Could not analyze historical data: {e}[/yellow]")
    
    def _suggest_mcp_remediation(self, plan: QuantizationPlan, result: ExecutionResult):
        """Enhanced MCP-powered remediation suggestions"""
        console.print(Panel.fit("🔧 MCP-Powered Recovery Suggestions", style="yellow"))
        
        try:
            # Get recovery suggestions from MCP
            recovery_actions = self.memory.suggest_error_recovery(result.error_message, plan)
            
            if not recovery_actions:
                console.print("[yellow]No specific recovery suggestions available.[/yellow]")
                self._suggest_remediation(plan, result)  # Fallback to original method
                return
            
            console.print("[blue]Based on analysis of similar failures:[/blue]\n")
            
            # Show recovery actions
            for i, action in enumerate(recovery_actions, 1):
                probability_color = "green" if action.success_probability > 0.6 else "yellow" if action.success_probability > 0.3 else "red"
                
                console.print(f"[bold]{i}. {action.description}[/bold]")
                console.print(f"   Success Probability: [{probability_color}]{action.success_probability:.1%}[/{probability_color}]")
                console.print(f"   Reasoning: {action.reasoning}")
                
                # Show parameters if available
                if action.parameters:
                    param_str = ", ".join([f"{k}={v}" for k, v in action.parameters.items()])
                    console.print(f"   Parameters: [dim]{param_str}[/dim]")
                console.print("")
            
            # Ask user if they want to try the top suggestion
            if recovery_actions and Confirm.ask(f"Would you like to try the top suggestion ({recovery_actions[0].description})?"):
                self._apply_recovery_action(plan, recovery_actions[0])
            
        except Exception as e:
            console.print(f"[yellow]Could not generate MCP recovery suggestions: {e}[/yellow]")
            self._suggest_remediation(plan, result)  # Fallback to original method
    
    def _apply_recovery_action(self, original_plan: QuantizationPlan, action: RecoveryAction):
        """Apply a recovery action to retry quantization"""
        console.print(Panel.fit(f"🔄 Applying Recovery: {action.description}", style="blue"))
        
        try:
            # Create a modified plan based on the recovery action
            modified_plan = self._modify_plan_with_recovery(original_plan, action)
            
            if modified_plan:
                console.print("[blue]Retrying quantization with modified parameters...[/blue]")
                
                # Execute with modified plan
                with Live(Spinner("dots", text="Retrying quantization..."), refresh_per_second=10):
                    result = self.executor.execute_quantization(modified_plan)
                
                # Store results
                record_id = self.memory.store_quantization_attempt(modified_plan, result)
                self.memory.log_experiment_step(
                    record_id,
                    "recovery_attempt",
                    f"Recovery action applied: {action.action_type}",
                    "INFO"
                )
                
                # Show results
                if result.success:
                    console.print(Panel.fit("✅ Recovery Successful!", style="bold green"))
                    console.print(f"[green]Model saved to: {result.output_path}[/green]")
                    console.print(f"[green]Size: {result.model_size_mb:.1f}MB[/green]")
                    console.print(f"[green]Time: {result.execution_time:.2f}s[/green]")
                else:
                    console.print(Panel.fit("❌ Recovery Failed", style="bold red"))
                    console.print(f"[red]Error: {result.error_message}[/red]")
            else:
                console.print("[yellow]Could not apply recovery action automatically.[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error applying recovery action: {e}[/red]")
    
    def _modify_plan_with_recovery(self, original_plan: QuantizationPlan, action: RecoveryAction) -> Optional[QuantizationPlan]:
        """Modify quantization plan based on recovery action"""
        try:
            # Create a copy of the original plan
            from copy import deepcopy
            modified_plan = deepcopy(original_plan)
            
            # Apply modifications based on action type
            if action.action_type == "enable_cpu_fallback":
                modified_plan.cpu_fallback = True
                
            elif action.action_type == "increase_bit_width":
                new_bit_width = action.parameters.get("bit_width", original_plan.bit_width + 2)
                modified_plan.bit_width = new_bit_width
                
            elif action.action_type == "change_method":
                new_method = action.parameters.get("quantization_type")
                if new_method:
                    # Import the enum to set the new method
                    from agents.planner_agent import QuantizationType
                    
                    # Map method names to enum values
                    method_mapping = {
                        'bnb': QuantizationType.BITSANDBYTES,
                        'bitsandbytes': QuantizationType.BITSANDBYTES,
                        'gptq': QuantizationType.GPTQ,
                        'gguf': QuantizationType.GGUF
                    }
                    
                    if new_method.lower() in method_mapping:
                        modified_plan.quantization_type = method_mapping[new_method.lower()]
                        
                        # Also update the target_format field to match the quantization type
                        from agents.planner_agent import TargetFormat
                        if new_method.lower() == 'bnb':
                            modified_plan.target_format = TargetFormat.SAFETENSORS
                        elif new_method.lower() == 'gptq':
                            modified_plan.target_format = TargetFormat.SAFETENSORS
                        elif new_method.lower() == 'gguf':
                            modified_plan.target_format = TargetFormat.GGUF
                        
                        console.print(f"[blue]Changed quantization method to {new_method.upper()}[/blue]")
                    
            elif action.action_type == "verify_model_name":
                # This would require user interaction to select a corrected name
                suggested_names = action.parameters.get("suggested_names", [])
                if suggested_names:
                    console.print("[blue]Suggested model names:[/blue]")
                    for i, name in enumerate(suggested_names, 1):
                        console.print(f"  {i}. {name}")
                    
                    choice = Prompt.ask("Select a model name (number) or enter manually", default="manual")
                    if choice.isdigit() and 1 <= int(choice) <= len(suggested_names):
                        modified_plan.model_name = suggested_names[int(choice) - 1]
                    else:
                        new_name = Prompt.ask("Enter the correct model name")
                        if new_name:
                            modified_plan.model_name = new_name
                        else:
                            return None
                else:
                    return None
            
            return modified_plan
            
        except Exception as e:
            console.print(f"[yellow]Could not modify plan: {e}[/yellow]")
            return None
    
    # === ENHANCED MCP FEATURES - PHASE 2 CLI INTEGRATION ===
    
    def _show_learning_insights(self):
        """Show comprehensive learning insights and patterns"""
        console.print(Panel.fit("🧠 Learning Insights & Patterns", style="bold blue"))
        
        try:
            insights = self.memory.get_learning_insights_summary()
            patterns = self.memory.discover_learning_patterns()
            compatibility_matrix = self.memory.get_model_compatibility_matrix()
            
            # Learning Progress
            progress = insights["learning_progress"]
            progress_color = {
                "Getting Started": "yellow",
                "Beginner": "yellow", 
                "Intermediate": "blue",
                "Advanced": "green"
            }.get(progress, "white")
            
            console.print(f"[{progress_color}]📈 Learning Progress: {progress}[/{progress_color}]")
            console.print(f"🎯 Overall Success Rate: {insights['overall_success_rate']:.1%}")
            console.print(f"📊 Total Experiments: {insights['total_experiments']}")
            console.print("")
            
            # Discovered Patterns
            if patterns:
                console.print("[bold green]🔍 Discovered Patterns:[/bold green]")
                for i, pattern in enumerate(patterns, 1):
                    confidence_color = "green" if pattern.confidence > 0.7 else "yellow" if pattern.confidence > 0.4 else "red"
                    console.print(f"  {i}. {pattern.description}")
                    console.print(f"     Confidence: [{confidence_color}]{pattern.confidence:.1%}[/{confidence_color}]")
                    if pattern.recommendations:
                        console.print(f"     💡 {pattern.recommendations[0]}")
                console.print("")
            
            # Compatibility Matrix
            if compatibility_matrix:
                console.print("[bold blue]🧬 Model Compatibility Matrix:[/bold blue]")
                
                table = Table(title="Success Rates by Model Family & Method")
                table.add_column("Family", style="cyan")
                
                # Get all methods
                all_methods = set()
                for methods in compatibility_matrix.values():
                    all_methods.update(methods.keys())
                
                for method in sorted(all_methods):
                    table.add_column(method.upper(), style="green")
                
                for family, methods in compatibility_matrix.items():
                    row = [family.title()]
                    for method in sorted(all_methods):
                        rate = methods.get(method, 0.0)
                        if rate > 0:
                            color = "green" if rate > 0.7 else "yellow" if rate > 0.4 else "red"
                            row.append(f"[{color}]{rate:.1%}[/{color}]")
                        else:
                            row.append("[dim]N/A[/dim]")
                    table.add_row(*row)
                
                console.print(table)
                console.print("")
            
            # Recommendations
            if insights["recommendations"]:
                console.print("[bold yellow]💡 Learning Recommendations:[/bold yellow]")
                for rec in insights["recommendations"]:
                    console.print(f"  • {rec}")
                console.print("")
            
            # Compatibility Insights
            if insights["compatibility_insights"]:
                console.print("[bold green]✅ Key Compatibility Insights:[/bold green]")
                for insight in insights["compatibility_insights"]:
                    console.print(f"  • {insight}")
                    
        except Exception as e:
            console.print(f"[red]Error generating insights: {e}[/red]")
    
    def _handle_model_recommendations(self, user_input: str):
        """Handle model recommendation requests"""
        # Extract goals from command
        parts = user_input.split(maxsplit=1)
        if len(parts) > 1:
            goals = parts[1]
        else:
            goals = Prompt.ask("What are your goals for model quantization?", 
                             default="fast and reliable quantization")
        
        console.print(Panel.fit(f"🎯 Model Recommendations for: '{goals}'", style="bold blue"))
        
        try:
            recommendations = self.memory.get_model_recommendations(goals, limit=5)
            
            if not recommendations:
                console.print("[yellow]No recommendations available yet. Try quantizing some models first![/yellow]")
                return
            
            table = Table(title="Recommended Models")
            table.add_column("Model", style="cyan")
            table.add_column("Confidence", style="green") 
            table.add_column("Success Rate", style="yellow")
            table.add_column("Avg Size", style="blue")
            table.add_column("Avg Time", style="magenta")
            table.add_column("Top Reason", style="white")
            
            for rec in recommendations:
                confidence_color = "green" if rec.confidence_score > 0.7 else "yellow" if rec.confidence_score > 0.4 else "red"
                confidence_text = f"[{confidence_color}]{rec.confidence_score:.1%}[/{confidence_color}]"
                
                table.add_row(
                    rec.model_name,
                    confidence_text,
                    f"{rec.success_rate:.1%}",
                    f"{rec.avg_size_mb:.0f}MB" if rec.avg_size_mb else "N/A",
                    f"{rec.avg_execution_time:.0f}s" if rec.avg_execution_time else "N/A",
                    rec.reasons[0] if rec.reasons else "N/A"
                )
            
            console.print(table)
            
            # Show detailed reasons for top recommendation
            if recommendations:
                top_rec = recommendations[0]
                console.print(f"\n[bold]🏆 Top Choice: {top_rec.model_name}[/bold]")
                console.print("[green]Reasons:[/green]")
                for reason in top_rec.reasons:
                    console.print(f"  • {reason}")
                console.print(f"[blue]Compatible Methods: {', '.join(top_rec.compatible_methods)}[/blue]")
                
                if Confirm.ask("Would you like to quantize the top recommendation?"):
                    model_name = top_rec.model_name.split("(e.g., ")[-1].rstrip(")")
                    self._handle_quantization(f"quantize {model_name}")
                    
        except Exception as e:
            console.print(f"[red]Error generating recommendations: {e}[/red]")
    
    def _handle_experiment_comparison(self, user_input: str):
        """Handle experiment comparison requests"""
        # Extract experiment IDs from command
        parts = user_input.split()[1:]  # Remove 'compare'
        
        if not parts:
            # Show recent experiments for selection
            recent_records = self.memory.get_model_records(limit=10)
            if not recent_records:
                console.print("[yellow]No experiments to compare.[/yellow]")
                return
            
            console.print("[blue]Recent Experiments:[/blue]")
            for i, record in enumerate(recent_records, 1):
                status = "✅" if record.success else "❌"
                console.print(f"  {i}. {status} {record.model_name} ({record.quantization_type} {record.bit_width}-bit) - ID: {record.id[:8]}")
            
            ids_input = Prompt.ask("Enter experiment IDs to compare (comma-separated)", default="")
            if not ids_input:
                return
            
            # Find full IDs from partial IDs
            partial_ids = [id.strip() for id in ids_input.split(",")]
            experiment_ids = []
            for partial_id in partial_ids:
                for record in recent_records:
                    if record.id.startswith(partial_id):
                        experiment_ids.append(record.id)
                        break
        else:
            experiment_ids = parts
        
        if len(experiment_ids) < 2:
            console.print("[red]Need at least 2 experiments to compare.[/red]")
            return
        
        console.print(Panel.fit(f"📊 Comparing {len(experiment_ids)} Experiments", style="bold blue"))
        
        try:
            comparison = self.memory.compare_experiments(experiment_ids)
            
            # Performance Metrics Table
            table = Table(title="Performance Comparison")
            table.add_column("Model", style="cyan")
            table.add_column("Method", style="green")
            table.add_column("Bits", style="yellow")
            table.add_column("Success", style="red")
            table.add_column("Size (MB)", style="blue")
            table.add_column("Time (s)", style="magenta")
            table.add_column("Compression", style="white")
            
            for exp_id, metrics in comparison.performance_metrics.items():
                status = "✅" if metrics['success'] else "❌"
                size_text = f"{metrics['size_mb']:.1f}" if metrics['size_mb'] else "N/A"
                time_text = f"{metrics['execution_time']:.1f}" if metrics['execution_time'] else "N/A"
                compression_text = f"{metrics['compression_ratio']:.1f}x" if metrics['compression_ratio'] else "N/A"
                
                table.add_row(
                    metrics['model_name'],
                    metrics['method'].upper(),
                    f"{metrics['bit_width']}-bit",
                    status,
                    size_text,
                    time_text,
                    compression_text
                )
            
            console.print(table)
            
            # Success Analysis
            analysis = comparison.success_analysis
            console.print(f"\n[bold blue]📈 Success Analysis:[/bold blue]")
            console.print(f"  Total Experiments: {analysis['total_experiments']}")
            console.print(f"  Successful: {analysis['successful_experiments']}")
            console.print(f"  Success Rate: {analysis['success_rate']:.1%}")
            if analysis['avg_execution_time'] > 0:
                console.print(f"  Avg Execution Time: {analysis['avg_execution_time']:.1f}s")
            if analysis['best_compression']:
                console.print(f"  Best Compression: {analysis['best_compression']}")
            if comparison.best_performer:
                console.print(f"  Best Performer: {comparison.best_performer}")
            
            # Optimization Suggestions
            if comparison.optimization_suggestions:
                console.print(f"\n[bold yellow]💡 Optimization Suggestions:[/bold yellow]")
                for suggestion in comparison.optimization_suggestions:
                    console.print(f"  • {suggestion}")
                    
        except Exception as e:
            console.print(f"[red]Error comparing experiments: {e}[/red]")
    
    def _handle_experiment_suggestions(self, user_input: str):
        """Handle next experiment suggestions"""
        # Extract goals from command
        parts = user_input.split(maxsplit=1)
        if len(parts) > 1:
            goals = parts[1]
        else:
            goals = Prompt.ask("What are your learning goals?", 
                             default="explore new quantization methods")
        
        console.print(Panel.fit(f"🔬 Experiment Suggestions for: '{goals}'", style="bold blue"))
        
        try:
            suggestions = self.memory.suggest_next_experiments(goals, limit=5)
            
            if not suggestions:
                console.print("[yellow]No new experiment suggestions. You've covered a lot of ground![/yellow]")
                return
            
            table = Table(title="Suggested Experiments")
            table.add_column("Model", style="cyan")
            table.add_column("Method", style="green")
            table.add_column("Bits", style="yellow")
            table.add_column("Priority", style="red")
            table.add_column("Learning Value", style="blue")
            table.add_column("Est. Memory", style="magenta")
            table.add_column("Est. Time", style="white")
            
            for suggestion in suggestions:
                priority_color = "green" if suggestion.priority_score > 0.7 else "yellow" if suggestion.priority_score > 0.5 else "red"
                priority_text = f"[{priority_color}]{suggestion.priority_score:.1%}[/{priority_color}]"
                
                reqs = suggestion.resource_requirements
                memory_text = f"{reqs['estimated_memory_mb']}MB"
                time_text = f"{reqs['estimated_time_minutes']}min"
                
                table.add_row(
                    suggestion.model_name,
                    suggestion.quantization_method.upper(),
                    f"{suggestion.bit_width}-bit",
                    priority_text,
                    suggestion.learning_value,
                    memory_text,
                    time_text
                )
            
            console.print(table)
            
            # Show detailed insights for top suggestion
            if suggestions:
                top_suggestion = suggestions[0]
                console.print(f"\n[bold]🎯 Top Priority: {top_suggestion.model_name}[/bold]")
                console.print(f"[blue]Method: {top_suggestion.quantization_method.upper()} {top_suggestion.bit_width}-bit[/blue]")
                console.print(f"[green]Learning Value: {top_suggestion.learning_value}[/green]")
                console.print("[yellow]Expected Insights:[/yellow]")
                for insight in top_suggestion.expected_insights:
                    console.print(f"  • {insight}")
                
                if Confirm.ask("Would you like to run the top priority experiment?"):
                    command = f"quantize {top_suggestion.model_name} with {top_suggestion.quantization_method} {top_suggestion.bit_width}-bit"
                    self._handle_quantization(command)
                    
        except Exception as e:
            console.print(f"[red]Error generating experiment suggestions: {e}[/red]")
    
    def _handle_model_discovery(self, user_input: str):
        """Handle HuggingFace model discovery and recommendations"""
        # Extract goals from command
        parts = user_input.split(maxsplit=1)
        if len(parts) > 1:
            goals = parts[1]
        else:
            goals = Prompt.ask("What kind of models are you looking for?", 
                             default="chat models for production use")
        
        console.print(Panel.fit(f"🔍 Discovering HuggingFace Models for: '{goals}'", style="bold blue"))
        
        try:
            # Show spinner while discovering models
            with Live(Spinner("dots", text="Analyzing HuggingFace models..."), refresh_per_second=10):
                recommendations = self.memory.discover_huggingface_models(goals, limit=8)
            
            if not recommendations:
                console.print("[yellow]No model recommendations available. This might be due to:[/yellow]")
                console.print("  • HuggingFace API access issues")
                console.print("  • Very specific or uncommon goals")
                console.print("  • Network connectivity problems")
                return
            
            # Display recommendations table
            table = Table(title="HuggingFace Model Recommendations")
            table.add_column("Model", style="cyan")
            table.add_column("Category", style="green")
            table.add_column("Popularity", style="yellow")
            table.add_column("Compatibility", style="blue")
            table.add_column("Learning Value", style="magenta")
            table.add_column("Resource Fit", style="red")
            table.add_column("Overall Score", style="white")
            
            for rec in recommendations:
                # Color code the scores
                pop_color = "green" if rec.popularity_score > 0.7 else "yellow" if rec.popularity_score > 0.4 else "red"
                comp_color = "green" if rec.compatibility_score > 0.7 else "yellow" if rec.compatibility_score > 0.4 else "red"
                learn_color = "green" if rec.learning_value_score > 0.7 else "yellow" if rec.learning_value_score > 0.4 else "red"
                res_color = "green" if rec.resource_fit_score > 0.7 else "yellow" if rec.resource_fit_score > 0.4 else "red"
                overall_color = "green" if rec.overall_score > 0.7 else "yellow" if rec.overall_score > 0.5 else "red"
                
                table.add_row(
                    rec.model_name,
                    rec.category,
                    f"[{pop_color}]{rec.popularity_score:.1%}[/{pop_color}]",
                    f"[{comp_color}]{rec.compatibility_score:.1%}[/{comp_color}]",
                    f"[{learn_color}]{rec.learning_value_score:.1%}[/{learn_color}]",
                    f"[{res_color}]{rec.resource_fit_score:.1%}[/{res_color}]",
                    f"[{overall_color}]{rec.overall_score:.1%}[/{overall_color}]"
                )
            
            console.print(table)
            
            # Show detailed reasoning for top recommendation
            if recommendations:
                top_rec = recommendations[0]
                console.print(f"\n[bold]🏆 Top Recommendation: {top_rec.model_name}[/bold]")
                console.print(f"[blue]Category: {top_rec.category}[/blue]")
                console.print(f"[green]Overall Score: {top_rec.overall_score:.1%}[/green]")
                
                console.print("\n[bold yellow]🤖 Why This Model:[/bold yellow]")
                for reason in top_rec.reasoning:
                    console.print(f"  • {reason}")
                
                console.print("\n[bold blue]🔧 Quantization Predictions:[/bold blue]")
                for method, prediction in top_rec.quantization_predictions.items():
                    success_color = "green" if prediction['success_probability'] > 0.7 else "yellow" if prediction['success_probability'] > 0.4 else "red"
                    console.print(f"  • {method.upper()}: [{success_color}]{prediction['success_probability']:.1%}[/{success_color}] success probability")
                    console.print(f"    └─ {prediction['reasoning']}")
                
                console.print("\n[bold cyan]📊 Resource Requirements:[/bold cyan]")
                reqs = top_rec.resource_requirements
                console.print(f"  • Download Size: ~{reqs['estimated_download_size_gb']:.1f}GB")
                console.print(f"  • Memory Needed: ~{reqs['estimated_quantization_memory_gb']:.1f}GB")
                console.print(f"  • Est. Quant Time: ~{reqs['estimated_quantization_time_minutes']:.0f} minutes")
                
                # Ask if user wants to quantize the top recommendation
                if Confirm.ask("Would you like to quantize the top recommendation?"):
                    suggested_method = max(top_rec.quantization_predictions.items(), 
                                         key=lambda x: x[1]['success_probability'])[0]
                    console.print(f"[blue]Suggested method: {suggested_method.upper()}[/blue]")
                    
                    use_suggestion = Confirm.ask(f"Use suggested method ({suggested_method.upper()})?")
                    if use_suggestion:
                        command = f"quantize {top_rec.model_name} with {suggested_method}"
                    else:
                        command = f"quantize {top_rec.model_name}"
                    
                    self._handle_quantization(command)
                
                # Ask if they want to see alternative recommendations
                elif len(recommendations) > 1 and Confirm.ask("Show alternative recommendations?"):
                    console.print("\n[bold]🔄 Alternative Recommendations:[/bold]")
                    for i, rec in enumerate(recommendations[1:4], 2):  # Show next 3
                        console.print(f"\n[bold]{i}. {rec.model_name}[/bold] (Score: {rec.overall_score:.1%})")
                        console.print(f"   Category: {rec.category}")
                        console.print(f"   Top Reason: {rec.reasoning[0] if rec.reasoning else 'No specific reason'}")
                    
                    choice = Prompt.ask("Select a model to quantize (2-4) or press Enter to skip", default="")
                    if choice.isdigit() and 2 <= int(choice) <= min(4, len(recommendations)):
                        selected_rec = recommendations[int(choice) - 1]
                        self._handle_quantization(f"quantize {selected_rec.model_name}")
                        
        except Exception as e:
            console.print(f"[red]Error discovering models: {e}[/red]")
            console.print("[yellow]This might be due to HuggingFace API access issues or network problems.[/yellow]")
            
            # Offer fallback suggestions
            if Confirm.ask("Would you like to see popular model suggestions instead?"):
                console.print("\n[bold blue]💡 Popular Model Suggestions:[/bold blue]")
                fallback_models = [
                    "microsoft/DialoGPT-small - Lightweight conversational model",
                    "microsoft/DialoGPT-medium - Balanced conversational model", 
                    "huggingface/CodeBERTa-small-v1 - Code understanding model",
                    "sentence-transformers/all-MiniLM-L6-v2 - Sentence embedding model",
                    "distilbert-base-uncased - Lightweight BERT variant"
                ]
                
                for i, model in enumerate(fallback_models, 1):
                    console.print(f"  {i}. {model}")
                
                choice = Prompt.ask("Select a model to quantize (1-5) or press Enter to skip", default="")
                if choice.isdigit() and 1 <= int(choice) <= 5:
                    model_name = fallback_models[int(choice) - 1].split(" - ")[0]
                    self._handle_quantization(f"quantize {model_name}")

@click.command()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--gemini-key', help='Gemini API key')
@click.option('--storage-backend', type=click.Choice(['sqlite', 'json']), default='sqlite', help='Storage backend')
def main(config, gemini_key, storage_backend):
    """Local Quantized Model Factory - Interactive CLI"""
    
    # Set environment variables if provided
    if gemini_key:
        os.environ['GEMINI_API_KEY'] = gemini_key
    
    # Initialize and run application
    app = LQMF()
    
    # Override config if provided
    if storage_backend:
        app.config['storage_backend'] = storage_backend
        app.memory = MemoryAgent(backend=StorageBackend(storage_backend))
    
    app.run()

if __name__ == "__main__":
    main()