#!/usr/bin/env python3
"""
Fine-Tuning CLI for LQMF - Command-line interface for LoRA/QLoRA fine-tuning

This CLI provides comprehensive fine-tuning capabilities including:
- Starting fine-tuning jobs with various configurations
- Managing adapters (list, load, unload, export)
- Benchmarking fine-tuned models
- Monitoring training progress

Author: LQMF Development Team
Version: 1.0.0
Compatibility: Python 3.8+

Usage:
    python cli/finetuning_cli.py
    
Commands:
    finetune <model> <dataset> <task>    # Start fine-tuning job
    list adapters                        # List all adapters
    load adapter <name>                  # Load adapter for inference
    unload adapter                       # Unload current adapter
    export adapter <name> <path>         # Export adapter
    benchmark <adapter> <prompts>        # Benchmark adapter
    
Examples:
    finetune mistral-7b data.csv chat
    load adapter finance-style
    benchmark finance-style test_prompts.txt
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.finetuning_agent import (
    FineTuningAgent, FineTuningConfig, TaskType, AdapterInfo, AdapterStatus
)
from agents.api_server_agent import APIServerAgent
from agents.adapter_manager import AdapterManager

console = Console()

class FineTuningCLI:
    """
    Command-line interface for fine-tuning operations.
    
    Provides an interactive interface for managing fine-tuning jobs,
    adapters, and model inference with adapters.
    """
    
    def __init__(self):
        """Initialize the Fine-Tuning CLI."""
        try:
            self.finetuning_agent = FineTuningAgent()
            self.api_server_agent = APIServerAgent()
            self.adapter_manager = self.api_server_agent.adapter_manager
            
            console.print("[green]✅ Fine-Tuning CLI initialized successfully[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to initialize CLI: {e}[/red]")
            sys.exit(1)
    
    def run_interactive_session(self):
        """Run the main interactive CLI session."""
        console.print(Panel.fit(
            "🧠 LQMF Fine-Tuning CLI - LoRA/QLoRA Training Interface",
            style="bold blue"
        ))
        
        self._show_initial_status()
        
        console.print("\n[cyan]🚀 Interactive Fine-Tuning CLI Started[/cyan]")
        console.print("[dim]Type 'help' for commands, 'exit' to quit[/dim]")
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]FT>[/bold blue]", default="").strip()
                
                if not user_input:
                    continue
                
                args = user_input.split()
                command = args[0].lower()
                
                if command in ['exit', 'quit', 'q']:
                    self._handle_exit()
                    break
                elif command in ['help', 'h', '?']:
                    self._show_help()
                elif command == 'finetune':
                    self._handle_finetune_command(args[1:])
                elif command == 'list':
                    self._handle_list_command(args[1:])
                elif command == 'load':
                    self._handle_load_command(args[1:])
                elif command == 'unload':
                    self._handle_unload_command(args[1:])
                elif command == 'switch':
                    self._handle_switch_command(args[1:])
                elif command == 'export':
                    self._handle_export_command(args[1:])
                elif command == 'benchmark':
                    self._handle_benchmark_command(args[1:])
                elif command == 'status':
                    self._handle_status_command()
                elif command == 'config':
                    self._handle_config_command(args[1:])
                elif command == 'clear':
                    self._handle_clear_screen()
                # Phase 1: AI-Powered Intelligence - New commands
                elif command == 'analyze':
                    self._handle_analyze_command(args[1:])
                elif command == 'suggest':
                    self._handle_suggest_command(args[1:])
                elif command == 'predict':
                    self._handle_predict_command(args[1:])
                elif command == 'optimize':
                    self._handle_optimize_command(args[1:])
                else:
                    console.print(f"[red]Unknown command: {command}[/red]")
                    console.print("[dim]Type 'help' for available commands[/dim]")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit properly.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    def _show_initial_status(self):
        """Show initial system status."""
        # Check GPU availability
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            console.print(f"[green]🔧 GPU Available: {gpu_memory:.1f}GB[/green]")
        else:
            console.print("[yellow]⚠️  No GPU detected - training will be slow[/yellow]")
        
        # Show available models and adapters
        adapters = self.finetuning_agent.list_adapters()
        available_models = self.api_server_agent.discover_quantized_models()
        
        console.print(f"[blue]📊 Available models: {len(available_models)}, Trained adapters: {len(adapters)}[/blue]")
    
    def _show_help(self):
        """Show comprehensive help information."""
        help_text = """
[bold cyan]🧠 LQMF Fine-Tuning CLI Commands:[/bold cyan]

[bold yellow]📚 Fine-Tuning Operations:[/bold yellow]
  • [bold]finetune[/bold] <model> <dataset> <task> - Start a fine-tuning job
  • [bold]config[/bold] create <name> - Create fine-tuning configuration
  • [bold]config[/bold] edit <name> - Edit existing configuration

[bold yellow]🔧 Adapter Management:[/bold yellow]
  • [bold]list[/bold] adapters - Show all trained adapters
  • [bold]list[/bold] models - Show available base models
  • [bold]load[/bold] adapter <name> [model] - Load adapter for inference
  • [bold]unload[/bold] adapter [model] - Unload current adapter
  • [bold]switch[/bold] adapter <new_name> [model] - Switch to different adapter

[bold yellow]📤 Export & Sharing:[/bold yellow]
  • [bold]export[/bold] adapter <name> <path> - Export adapter to file
  • [bold]import[/bold] adapter <path> <name> - Import adapter from file

[bold yellow]📊 Benchmarking & Testing:[/bold yellow]
  • [bold]benchmark[/bold] <adapter> [prompts_file] - Benchmark adapter performance
  • [bold]test[/bold] <adapter> "<prompt>" - Test adapter with single prompt

[bold yellow]🧠 AI-Powered Intelligence:[/bold yellow]
  • [bold]analyze[/bold] dataset <path> - AI-powered dataset analysis
  • [bold]suggest[/bold] config <model> <task> <dataset_size> - Get optimal LoRA config
  • [bold]predict[/bold] training <config_file> - Predict training time and memory
  • [bold]optimize[/bold] memory <config_file> <gpu_gb> - Memory optimization suggestions

[bold yellow]ℹ️  Information:[/bold yellow]
  • [bold]status[/bold] - Show system status and loaded models
  • [bold]clear[/bold] - Clear screen
  • [bold]help[/bold] - Show this help
  • [bold]exit[/bold] - Exit CLI

[dim]💡 Quick Examples:[/dim]
[dim]  finetune mistral-7b ./data/chat.csv chat[/dim]
[dim]  load adapter finance-style[/dim]
[dim]  benchmark finance-style ./test_prompts.txt[/dim]
[dim]  export adapter finance-style ./exports/[/dim]

[dim]📝 Supported Task Types:[/dim]
[dim]  • chat - Conversational AI fine-tuning[/dim]
[dim]  • classification - Text classification tasks[/dim]
[dim]  • summarization - Text summarization tasks[/dim]
[dim]  • instruction_following - General instruction following[/dim]

[dim]📁 Supported Dataset Formats:[/dim]
[dim]  • CSV files with appropriate columns[/dim]
[dim]  • JSONL files with structured data[/dim]
[dim]  • Text directories with training files[/dim]
        """
        console.print(Panel(help_text, title="Help", expand=False))
    
    def _handle_finetune_command(self, args: List[str]):
        """Handle fine-tuning command."""
        if len(args) < 3:
            console.print("[red]Usage: finetune <model> <dataset> <task>[/red]")
            console.print("[dim]Example: finetune mistral-7b data.csv chat[/dim]")
            return
        
        base_model = args[0]
        dataset_path = args[1]
        task_type_str = args[2]
        
        # Validate task type
        try:
            task_type = TaskType(task_type_str.lower())
        except ValueError:
            console.print(f"[red]Invalid task type: {task_type_str}[/red]")
            console.print(f"[dim]Valid types: {[t.value for t in TaskType]}[/dim]")
            return
        
        # Check if dataset exists
        if not Path(dataset_path).exists():
            console.print(f"[red]Dataset not found: {dataset_path}[/red]")
            return
        
        # Interactive configuration
        console.print(f"\n[cyan]🔧 Configuring fine-tuning job for {base_model}[/cyan]")
        
        job_name = Prompt.ask("Job name", default=f"{base_model.split('/')[-1]}-{task_type_str}-{int(time.time())}")
        
        # Advanced configuration
        if Confirm.ask("Configure advanced settings?", default=False):
            config = self._create_advanced_config(job_name, base_model, dataset_path, task_type)
        else:
            config = self._create_default_config(job_name, base_model, dataset_path, task_type)
        
        # Show configuration summary
        self._show_config_summary(config)
        
        if not Confirm.ask("Start fine-tuning with this configuration?", default=True):
            console.print("[yellow]Fine-tuning cancelled[/yellow]")
            return
        
        # Start fine-tuning
        console.print(f"\n[blue]🚀 Starting fine-tuning job: {job_name}[/blue]")
        success = self.finetuning_agent.start_fine_tuning(config)
        
        if success:
            console.print(f"[green]✅ Fine-tuning completed successfully![/green]")
            console.print(f"[dim]Adapter saved as: {job_name}[/dim]")
        else:
            console.print(f"[red]❌ Fine-tuning failed[/red]")
    
    def _create_default_config(self, job_name: str, base_model: str, dataset_path: str, task_type: TaskType) -> FineTuningConfig:
        """Create default fine-tuning configuration."""
        output_dir = f"./adapters/{job_name}/training"
        
        return FineTuningConfig(
            job_name=job_name,
            base_model=base_model,
            task_type=task_type,
            dataset_path=dataset_path,
            output_dir=output_dir,
            # Optimized defaults for 8GB GPU
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            learning_rate=2e-4,
            num_train_epochs=3,
            per_device_train_batch_size=2,  # Small batch size for memory
            gradient_accumulation_steps=8,  # Higher accumulation
            use_gradient_checkpointing=True,
            use_4bit_quantization=True,
            max_seq_length=512
        )
    
    def _create_advanced_config(self, job_name: str, base_model: str, dataset_path: str, task_type: TaskType) -> FineTuningConfig:
        """Create advanced fine-tuning configuration with user input."""
        output_dir = f"./adapters/{job_name}/training"
        
        console.print("\n[cyan]📝 Advanced Configuration[/cyan]")
        
        # LoRA parameters
        lora_r = IntPrompt.ask("LoRA rank (r)", default=16)
        lora_alpha = IntPrompt.ask("LoRA alpha", default=32)
        lora_dropout = FloatPrompt.ask("LoRA dropout", default=0.1)
        
        # Training parameters
        learning_rate = FloatPrompt.ask("Learning rate", default=2e-4)
        num_epochs = IntPrompt.ask("Number of epochs", default=3)
        batch_size = IntPrompt.ask("Batch size", default=2)
        grad_accumulation = IntPrompt.ask("Gradient accumulation steps", default=8)
        max_seq_length = IntPrompt.ask("Max sequence length", default=512)
        
        # Memory optimization
        use_4bit = Confirm.ask("Use 4-bit quantization?", default=True)
        use_gradient_checkpointing = Confirm.ask("Use gradient checkpointing?", default=True)
        
        return FineTuningConfig(
            job_name=job_name,
            base_model=base_model,
            task_type=task_type,
            dataset_path=dataset_path,
            output_dir=output_dir,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accumulation,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_4bit_quantization=use_4bit,
            max_seq_length=max_seq_length
        )
    
    def _show_config_summary(self, config: FineTuningConfig):
        """Show configuration summary."""
        console.print("\n[cyan]📋 Configuration Summary:[/cyan]")
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Parameter", style="white")
        table.add_column("Value", style="yellow")
        
        table.add_row("Job Name", config.job_name)
        table.add_row("Base Model", config.base_model)
        table.add_row("Task Type", config.task_type.value)
        table.add_row("Dataset", config.dataset_path)
        table.add_row("LoRA Rank", str(config.lora_r))
        table.add_row("LoRA Alpha", str(config.lora_alpha))
        table.add_row("Learning Rate", str(config.learning_rate))
        table.add_row("Epochs", str(config.num_train_epochs))
        table.add_row("Batch Size", str(config.per_device_train_batch_size))
        table.add_row("4-bit Quantization", "✅" if config.use_4bit_quantization else "❌")
        table.add_row("Gradient Checkpointing", "✅" if config.use_gradient_checkpointing else "❌")
        
        console.print(table)
    
    def _handle_list_command(self, args: List[str]):
        """Handle list command."""
        if not args:
            console.print("[red]Usage: list <adapters|models>[/red]")
            return
        
        list_type = args[0].lower()
        
        if list_type == 'adapters':
            self._list_adapters()
        elif list_type == 'models':
            self._list_models()
        else:
            console.print(f"[red]Unknown list type: {list_type}[/red]")
            console.print("[dim]Use: list adapters or list models[/dim]")
    
    def _list_adapters(self):
        """List all available adapters."""
        adapters = self.finetuning_agent.list_adapters()
        
        if not adapters:
            console.print("[yellow]No adapters found[/yellow]")
            console.print("[dim]Use 'finetune' command to create adapters[/dim]")
            return
        
        console.print(f"\n[cyan]📚 Available Adapters ({len(adapters)}):[/cyan]")
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name", style="white")
        table.add_column("Base Model", style="blue")
        table.add_column("Task", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Status", style="magenta")
        table.add_column("Created", style="dim")
        
        for adapter in adapters:
            status_icon = "🟢" if adapter.status == AdapterStatus.COMPLETED else "🔴"
            table.add_row(
                adapter.adapter_name,
                adapter.base_model,
                adapter.task_type.value,
                f"{adapter.size_mb:.1f}MB",
                f"{status_icon} {adapter.status.value}",
                adapter.created_at[:10]  # Just the date
            )
        
        console.print(table)
    
    def _list_models(self):
        """List all available base models."""
        models = self.api_server_agent.discover_quantized_models()
        
        if not models:
            console.print("[yellow]No quantized models found[/yellow]")
            console.print("[dim]Use the main CLI to quantize models first[/dim]")
            return
        
        console.print(f"\n[cyan]🤖 Available Models ({len(models)}):[/cyan]")
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Model", style="white")
        table.add_column("Method", style="blue")
        table.add_column("Bits", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Status", style="magenta")
        
        for model in models:
            model_name = model.get('display_name', model.get('model_name', 'Unknown'))
            is_loaded = model_name in self.api_server_agent.loaded_models
            status = "🟢 Loaded" if is_loaded else "⚫ Available"
            
            table.add_row(
                model_name,
                model.get('quantization_method', 'Unknown'),
                f"{model.get('bit_width', 'Unknown')}-bit",
                f"{model.get('model_size_mb', 0):.1f}MB",
                status
            )
        
        console.print(table)
    
    def _handle_load_command(self, args: List[str]):
        """Handle load adapter command."""
        if len(args) < 2 or args[0] != 'adapter':
            console.print("[red]Usage: load adapter <name> [model][/red]")
            return
        
        adapter_name = args[1]
        model_name = args[2] if len(args) > 2 else None
        
        console.print(f"[blue]🔄 Loading adapter: {adapter_name}[/blue]")
        success = self.api_server_agent.load_adapter(adapter_name, model_name)
        
        if success:
            console.print(f"[green]✅ Adapter '{adapter_name}' loaded successfully[/green]")
        else:
            console.print(f"[red]❌ Failed to load adapter '{adapter_name}'[/red]")
    
    def _handle_unload_command(self, args: List[str]):
        """Handle unload adapter command."""
        if args and args[0] != 'adapter':
            console.print("[red]Usage: unload adapter [model][/red]")
            return
        
        model_name = args[1] if len(args) > 1 else None
        
        console.print("[blue]🔄 Unloading adapter...[/blue]")
        success = self.api_server_agent.unload_adapter(model_name)
        
        if success:
            console.print("[green]✅ Adapter unloaded successfully[/green]")
        else:
            console.print("[red]❌ Failed to unload adapter[/red]")
    
    def _handle_switch_command(self, args: List[str]):
        """Handle switch adapter command."""
        if len(args) < 2 or args[0] != 'adapter':
            console.print("[red]Usage: switch adapter <new_name> [model][/red]")
            return
        
        new_adapter = args[1]
        model_name = args[2] if len(args) > 2 else None
        
        console.print(f"[blue]🔄 Switching to adapter: {new_adapter}[/blue]")
        success = self.api_server_agent.switch_adapter(new_adapter, model_name)
        
        if success:
            console.print(f"[green]✅ Switched to adapter '{new_adapter}'[/green]")
        else:
            console.print(f"[red]❌ Failed to switch to adapter '{new_adapter}'[/red]")
    
    def _handle_export_command(self, args: List[str]):
        """Handle export adapter command."""
        if len(args) < 3 or args[0] != 'adapter':
            console.print("[red]Usage: export adapter <name> <path>[/red]")
            return
        
        adapter_name = args[1]
        export_path = args[2]
        
        console.print(f"[blue]📤 Exporting adapter: {adapter_name}[/blue]")
        success = self.finetuning_agent.export_adapter(adapter_name, export_path)
        
        if success:
            console.print(f"[green]✅ Adapter exported to: {export_path}[/green]")
        else:
            console.print(f"[red]❌ Failed to export adapter[/red]")
    
    def _handle_benchmark_command(self, args: List[str]):
        """Handle benchmark command."""
        if not args:
            console.print("[red]Usage: benchmark <adapter> [prompts_file][/red]")
            return
        
        adapter_name = args[0]
        prompts_file = args[1] if len(args) > 1 else None
        
        # Get test prompts
        if prompts_file and Path(prompts_file).exists():
            with open(prompts_file, 'r') as f:
                test_prompts = [line.strip() for line in f if line.strip()]
        else:
            # Default test prompts
            test_prompts = [
                "What is artificial intelligence?",
                "Explain the concept of machine learning.",
                "How does natural language processing work?",
                "What are the benefits of renewable energy?",
                "Describe the water cycle."
            ]
        
        console.print(f"[blue]📊 Benchmarking adapter: {adapter_name}[/blue]")
        results = self.api_server_agent.benchmark_adapter(adapter_name, test_prompts)
        
        if "error" in results:
            console.print(f"[red]❌ Benchmark failed: {results['error']}[/red]")
            return
        
        # Show results
        console.print(f"\n[cyan]📊 Benchmark Results for {adapter_name}:[/cyan]")
        
        stats_table = Table(show_header=True, header_style="bold cyan")
        stats_table.add_column("Metric", style="white")
        stats_table.add_column("Value", style="yellow")
        
        stats_table.add_row("Test Prompts", str(results['test_count']))
        stats_table.add_row("Avg Response Time", f"{results['avg_response_time']:.2f}s")
        stats_table.add_row("Total Tokens", str(results['total_tokens']))
        stats_table.add_row("Tokens/Second", f"{results.get('tokens_per_second', 0):.1f}")
        
        console.print(stats_table)
        
        # Show sample responses
        if Confirm.ask("Show sample responses?", default=False):
            console.print("\n[cyan]📝 Sample Responses:[/cyan]")
            for i, response in enumerate(results['responses'][:3]):  # Show first 3
                console.print(f"\n[bold]Prompt {i+1}:[/bold] {response['prompt']}")
                console.print(f"[bold]Response:[/bold] {response.get('response', 'Error')}")
                console.print(f"[dim]Time: {response.get('response_time', 0):.2f}s, Tokens: {response.get('tokens', 0)}[/dim]")
    
    def _handle_status_command(self):
        """Handle status command."""
        console.print("\n[cyan]📊 System Status:[/cyan]")
        
        # GPU status
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated() / 1024**3
            console.print(f"[green]🔧 GPU: {allocated:.1f}GB / {gpu_memory:.1f}GB used[/green]")
        else:
            console.print("[yellow]⚠️  No GPU available[/yellow]")
        
        # Loaded models
        loaded_models = self.api_server_agent.loaded_models
        console.print(f"[blue]🤖 Loaded Models: {len(loaded_models)}[/blue]")
        
        for model_name, model_info in loaded_models.items():
            adapter_text = f" (Adapter: {model_info.active_adapter})" if model_info.active_adapter else ""
            console.print(f"  • {model_name}{adapter_text}")
        
        # Available adapters
        adapters = self.finetuning_agent.list_adapters()
        console.print(f"[green]📚 Available Adapters: {len(adapters)}[/green]")
        
        # Memory usage
        memory_info = self.adapter_manager.get_memory_usage()
        console.print(f"[blue]💾 Loaded Adapters: {memory_info['loaded_adapters_count']}/{memory_info['max_adapters']}[/blue]")
    
    def _handle_config_command(self, args: List[str]):
        """Handle configuration commands."""
        if not args:
            console.print("[red]Usage: config <create|edit> <name>[/red]")
            return
        
        action = args[0].lower()
        
        if action == 'create' and len(args) > 1:
            self._create_config_template(args[1])
        elif action == 'edit' and len(args) > 1:
            self._edit_config(args[1])
        else:
            console.print("[red]Usage: config <create|edit> <name>[/red]")
    
    def _create_config_template(self, name: str):
        """Create a configuration template file."""
        config_path = Path(f"configs/{name}_config.json")
        config_path.parent.mkdir(exist_ok=True)
        
        template_config = {
            "job_name": name,
            "base_model": "mistral-7b",
            "task_type": "chat",
            "dataset_path": "./data/training.csv",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "learning_rate": 2e-4,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "use_gradient_checkpointing": True,
            "use_4bit_quantization": True,
            "max_seq_length": 512
        }
        
        with open(config_path, 'w') as f:
            json.dump(template_config, f, indent=2)
        
        console.print(f"[green]✅ Configuration template created: {config_path}[/green]")
        console.print("[dim]Edit the file and use 'config edit' to load it[/dim]")
    
    def _edit_config(self, name: str):
        """Edit an existing configuration."""
        config_path = Path(f"configs/{name}_config.json")
        
        if not config_path.exists():
            console.print(f"[red]Configuration not found: {config_path}[/red]")
            console.print(f"[dim]Use 'config create {name}' to create it first[/dim]")
            return
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            console.print(f"[green]📝 Configuration loaded: {config_path}[/green]")
            
            # Show current config
            syntax = Syntax(json.dumps(config_data, indent=2), "json", theme="monokai")
            console.print(syntax)
            
            if Confirm.ask("Use this configuration for fine-tuning?", default=True):
                # Convert to FineTuningConfig
                config = FineTuningConfig(**config_data)
                
                console.print(f"\n[blue]🚀 Starting fine-tuning with config: {name}[/blue]")
                success = self.finetuning_agent.start_fine_tuning(config)
                
                if success:
                    console.print(f"[green]✅ Fine-tuning completed successfully![/green]")
                else:
                    console.print(f"[red]❌ Fine-tuning failed[/red]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load configuration: {e}[/red]")
    
    def _handle_clear_screen(self):
        """Clear the screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
        console.print(Panel.fit(
            "🧠 LQMF Fine-Tuning CLI - LoRA/QLoRA Training Interface",
            style="bold blue"
        ))
    
    def _handle_exit(self):
        """Handle exit command."""
        console.print("\n[green]Thanks for using LQMF Fine-Tuning CLI! Goodbye! 👋[/green]")

    
    # Phase 1: AI-Powered Intelligence - New command handlers
    def _handle_analyze_command(self, args: List[str]):
        """Handle AI-powered dataset analysis command."""
        if len(args) < 2 or args[0] != 'dataset':
            console.print("[red]Usage: analyze dataset <path> [task_type][/red]")
            console.print("[dim]Example: analyze dataset ./data/training.csv chat[/dim]")
            return
        
        dataset_path = args[1]
        task_type = args[2] if len(args) > 2 else None
        
        if not Path(dataset_path).exists():
            console.print(f"[red]Dataset not found: {dataset_path}[/red]")
            return
        
        try:
            # Import and use DatasetIntelligenceAgent
            from agents.dataset_intelligence_agent import DatasetIntelligenceAgent
            
            agent = DatasetIntelligenceAgent()
            analysis = agent.analyze_dataset(dataset_path, task_type)
            
            # Display analysis results
            console.print(f"\n[cyan]🔍 Dataset Analysis Results for {dataset_path}:[/cyan]")
            
            # Basic statistics
            stats_table = Table(show_header=True, header_style="bold cyan")
            stats_table.add_column("Metric", style="white")
            stats_table.add_column("Value", style="yellow")
            
            stats_table.add_row("Total Samples", str(analysis.stats.total_samples))
            stats_table.add_row("Format", analysis.stats.format.value)
            stats_table.add_row("File Size", f"{analysis.stats.file_size_mb:.1f}MB")
            stats_table.add_row("Task Type", analysis.task_type.value)
            stats_table.add_row("Completeness Score", f"{analysis.quality.completeness_score:.2f}")
            stats_table.add_row("Balance Score", f"{analysis.quality.balance_score:.2f}")
            stats_table.add_row("AI Confidence", f"{analysis.confidence:.2f}")
            
            console.print(stats_table)
            
            # Quality issues
            if analysis.quality.quality_issues:
                console.print(f"\n[yellow]⚠️ Quality Issues:[/yellow]")
                for issue in analysis.quality.quality_issues:
                    console.print(f"  • {issue}")
            
            # Recommendations
            if analysis.recommendations:
                console.print(f"\n[green]💡 AI Recommendations:[/green]")
                for rec in analysis.recommendations:
                    console.print(f"  • {rec}")
            
            # Preprocessing steps
            if analysis.preprocessing_steps:
                console.print(f"\n[blue]🔧 Preprocessing Steps:[/blue]")
                for step in analysis.preprocessing_steps:
                    console.print(f"  • {step}")
            
            # Suggested augmentations
            if analysis.augmentation_suggestions:
                console.print(f"\n[magenta]🔄 Augmentation Suggestions:[/blue]")
                for aug in analysis.augmentation_suggestions:
                    console.print(f"  • {aug}")
            
            # Export option
            if Confirm.ask("Export analysis to file?", default=False):
                export_path = f"./reports/dataset_analysis_{int(time.time())}.json"
                Path(export_path).parent.mkdir(exist_ok=True)
                agent.export_analysis(analysis, export_path)
                
        except Exception as e:
            console.print(f"[red]❌ Analysis failed: {e}[/red]")
    
    def _handle_suggest_command(self, args: List[str]):
        """Handle AI-powered configuration suggestions."""
        if len(args) < 4 or args[0] != 'config':
            console.print("[red]Usage: suggest config <model> <task> <dataset_size> [gpu_gb][/red]")
            console.print("[dim]Example: suggest config mistral-7b chat 1000 8[/dim]")
            return
        
        model_name = args[1]
        task_type = args[2]
        try:
            dataset_size = int(args[3])
        except ValueError:
            console.print("[red]Dataset size must be a number[/red]")
            return
        
        gpu_memory_gb = int(args[4]) if len(args) > 4 else 8
        
        try:
            # Import and use PlannerAgent
            from agents.planner_agent import PlannerAgent
            
            planner = PlannerAgent()
            config = planner.suggest_optimal_lora_config(model_name, task_type, dataset_size, gpu_memory_gb)
            
            # Display suggested configuration
            console.print(f"\n[cyan]🧠 AI-Suggested LoRA Configuration:[/cyan]")
            
            config_table = Table(show_header=True, header_style="bold cyan")
            config_table.add_column("Parameter", style="white")
            config_table.add_column("Suggested Value", style="yellow")
            config_table.add_column("Reasoning", style="dim")
            
            config_table.add_row("LoRA Rank (r)", str(config.get('lora_r', 16)), "Balances adaptation capability vs memory")
            config_table.add_row("LoRA Alpha", str(config.get('lora_alpha', 32)), "Scaling factor for LoRA weights")
            config_table.add_row("LoRA Dropout", str(config.get('lora_dropout', 0.1)), "Regularization to prevent overfitting")
            config_table.add_row("Learning Rate", str(config.get('learning_rate', 2e-4)), "Optimal for LoRA fine-tuning")
            config_table.add_row("Batch Size", str(config.get('per_device_train_batch_size', 2)), f"Optimized for {gpu_memory_gb}GB GPU")
            config_table.add_row("Epochs", str(config.get('num_train_epochs', 3)), "Based on dataset size")
            
            console.print(config_table)
            
            # Performance predictions
            if 'estimated_memory_gb' in config:
                console.print(f"\n[blue]📊 Performance Estimates:[/blue]")
                console.print(f"  • Estimated GPU Memory: {config['estimated_memory_gb']:.1f}GB")
                console.print(f"  • Estimated Training Time: {config.get('estimated_training_hours', 2.0):.1f} hours")
            
            # Reasoning
            if 'reasoning' in config:
                console.print(f"\n[green]💭 AI Reasoning:[/green]")
                console.print(f"  {config['reasoning']}")
            
            # Save config option
            if Confirm.ask("Save this configuration?", default=True):
                config_name = Prompt.ask("Configuration name", default=f"{model_name.split('/')[-1]}-{task_type}-optimized")
                config_path = Path(f"configs/{config_name}_config.json")
                config_path.parent.mkdir(exist_ok=True)
                
                # Convert to standard config format
                save_config = {
                    "job_name": config_name,
                    "base_model": model_name,
                    "task_type": task_type,
                    "dataset_path": "./data/training.csv",  # Placeholder
                    "lora_r": config.get('lora_r', 16),
                    "lora_alpha": config.get('lora_alpha', 32),
                    "lora_dropout": config.get('lora_dropout', 0.1),
                    "learning_rate": config.get('learning_rate', 2e-4),
                    "num_train_epochs": config.get('num_train_epochs', 3),
                    "per_device_train_batch_size": config.get('per_device_train_batch_size', 2),
                    "gradient_accumulation_steps": config.get('gradient_accumulation_steps', 8),
                    "use_gradient_checkpointing": True,
                    "use_4bit_quantization": True,
                    "max_seq_length": 512
                }
                
                with open(config_path, 'w') as f:
                    json.dump(save_config, f, indent=2)
                
                console.print(f"[green]✅ Configuration saved to: {config_path}[/green]")
                
        except Exception as e:
            console.print(f"[red]❌ Configuration suggestion failed: {e}[/red]")
    
    def _handle_predict_command(self, args: List[str]):
        """Handle training prediction command."""
        if len(args) < 2 or args[0] != 'training':
            console.print("[red]Usage: predict training <config_file>[/red]")
            console.print("[dim]Example: predict training ./configs/my_config.json[/dim]")
            return
        
        config_file = args[1]
        
        if not Path(config_file).exists():
            console.print(f"[red]Configuration file not found: {config_file}[/red]")
            return
        
        try:
            # Load configuration
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Import and use PlannerAgent
            from agents.planner_agent import PlannerAgent
            
            planner = PlannerAgent()
            predictions = planner.predict_training_time(config)
            
            # Display predictions
            console.print(f"\n[cyan]⏱️ Training Predictions for {Path(config_file).name}:[/cyan]")
            
            pred_table = Table(show_header=True, header_style="bold cyan")
            pred_table.add_column("Metric", style="white")
            pred_table.add_column("Prediction", style="yellow")
            pred_table.add_column("Confidence", style="green")
            
            pred_table.add_row("GPU Memory Usage", f"{predictions.get('estimated_memory_gb', 6.0):.1f}GB", "High")
            pred_table.add_row("Training Time per Epoch", f"{predictions.get('training_time_per_epoch_minutes', 30):.0f} minutes", "Medium")
            pred_table.add_row("Total Training Time", f"{predictions.get('total_training_hours', 2.0):.1f} hours", "Medium")
            pred_table.add_row("CPU Memory", f"{predictions.get('cpu_memory_gb', 8.0):.1f}GB", "High")
            pred_table.add_row("Storage Requirements", f"{predictions.get('storage_gb', 5.0):.1f}GB", "High")
            
            console.print(pred_table)
            
            # Recommendations
            if 'recommendations' in predictions:
                console.print(f"\n[green]💡 Optimization Recommendations:[/green]")
                for rec in predictions['recommendations']:
                    console.print(f"  • {rec}")
            
            # Check for potential issues
            estimated_memory = predictions.get('estimated_memory_gb', 6.0)
            if estimated_memory > 7.5:
                console.print(f"\n[red]⚠️ Warning: Predicted memory usage ({estimated_memory:.1f}GB) is close to 8GB limit![/red]")
                console.print("[yellow]Consider using memory optimization suggestions.[/yellow]")
            
        except Exception as e:
            console.print(f"[red]❌ Prediction failed: {e}[/red]")
    
    def _handle_optimize_command(self, args: List[str]):
        """Handle memory optimization command."""
        if len(args) < 3 or args[0] != 'memory':
            console.print("[red]Usage: optimize memory <config_file> <gpu_gb>[/red]")
            console.print("[dim]Example: optimize memory ./configs/my_config.json 8[/dim]")
            return
        
        config_file = args[1]
        try:
            gpu_limit = int(args[2])
        except ValueError:
            console.print("[red]GPU memory limit must be a number[/red]")
            return
        
        if not Path(config_file).exists():
            console.print(f"[red]Configuration file not found: {config_file}[/red]")
            return
        
        try:
            # Load configuration
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Import and use MemoryAgent
            from agents.memory_agent import MemoryAgent
            
            memory_agent = MemoryAgent()
            optimizations = memory_agent.suggest_memory_optimizations(gpu_limit, config)
            
            if not optimizations:
                console.print(f"[green]✅ Configuration is already optimized for {gpu_limit}GB GPU![/green]")
                return
            
            # Display optimization suggestions
            console.print(f"\n[cyan]🔧 Memory Optimization Suggestions for {gpu_limit}GB GPU:[/cyan]")
            
            opt_table = Table(show_header=True, header_style="bold cyan")
            opt_table.add_column("Strategy", style="white")
            opt_table.add_column("Action", style="blue")
            opt_table.add_column("Memory Savings", style="green")
            opt_table.add_column("Impact", style="yellow")
            
            for opt in optimizations:
                savings = f"{opt['expected_savings_gb']:.1f}GB" if opt['expected_savings_gb'] > 0 else "Performance"
                opt_table.add_row(
                    opt['strategy'],
                    opt['action'],
                    savings,
                    opt['impact']
                )
            
            console.print(opt_table)
            
            # Apply optimizations option
            if Confirm.ask("Apply these optimizations to the configuration?", default=True):
                # Apply optimizations
                optimized_config = config.copy()
                
                for opt in optimizations:
                    if 'batch_size' in opt['implementation']:
                        optimized_config['per_device_train_batch_size'] = 1
                    elif 'gradient_checkpointing' in opt['implementation']:
                        optimized_config['gradient_checkpointing'] = True
                    elif 'load_in_4bit' in opt['implementation']:
                        optimized_config['load_in_4bit'] = True
                    elif 'lora_r' in opt['implementation']:
                        current_r = optimized_config.get('lora_r', 16)
                        optimized_config['lora_r'] = current_r // 2
                    elif 'gradient_accumulation_steps' in opt['implementation']:
                        optimized_config['gradient_accumulation_steps'] = 8
                
                # Save optimized configuration
                optimized_path = config_file.replace('.json', '_optimized.json')
                with open(optimized_path, 'w') as f:
                    json.dump(optimized_config, f, indent=2)
                
                console.print(f"[green]✅ Optimized configuration saved to: {optimized_path}[/green]")
                
                # Show memory prediction for optimized config
                from agents.planner_agent import PlannerAgent
                planner = PlannerAgent()
                new_predictions = planner.predict_training_time(optimized_config)
                
                new_memory = new_predictions.get('estimated_memory_gb', 6.0)
                console.print(f"[blue]📊 New predicted memory usage: {new_memory:.1f}GB[/blue]")
                
                if new_memory <= gpu_limit:
                    console.print(f"[green]✅ Configuration now fits within {gpu_limit}GB limit![/green]")
                else:
                    console.print(f"[yellow]⚠️ May still exceed {gpu_limit}GB limit. Consider manual adjustments.[/yellow]")
            
        except Exception as e:
            console.print(f"[red]❌ Optimization failed: {e}[/red]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LQMF Fine-Tuning CLI")
    parser.add_argument('--non-interactive', action='store_true', 
                       help='Run in non-interactive mode')
    
    args = parser.parse_args()
    
    try:
        cli = FineTuningCLI()
        
        if args.non_interactive:
            # TODO: Add non-interactive mode support
            console.print("[yellow]Non-interactive mode not yet implemented[/yellow]")
        else:
            cli.run_interactive_session()
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()