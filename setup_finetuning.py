#!/usr/bin/env python3
"""
LQMF Fine-Tuning Setup Verification

This script verifies that the fine-tuning system is properly set up
and provides guidance for getting started.
"""

import sys
import torch
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def check_system_requirements():
    """Check system requirements for fine-tuning."""
    console.print("[cyan]🔍 Checking System Requirements...[/cyan]")
    
    requirements_table = Table(show_header=True, header_style="bold cyan")
    requirements_table.add_column("Component", style="white")
    requirements_table.add_column("Status", style="green")
    requirements_table.add_column("Details", style="dim")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_ok = sys.version_info >= (3, 8)
    requirements_table.add_row(
        "Python",
        "✅ OK" if python_ok else "❌ FAIL",
        f"v{python_version} {'(3.8+ required)' if not python_ok else ''}"
    )
    
    # Check PyTorch
    try:
        torch_version = torch.__version__
        torch_ok = True
        requirements_table.add_row("PyTorch", "✅ OK", f"v{torch_version}")
    except Exception as e:
        torch_ok = False
        requirements_table.add_row("PyTorch", "❌ FAIL", str(e))
    
    # Check GPU
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_status = f"{gpu_name} ({gpu_memory:.1f}GB)"
        gpu_ok = gpu_memory >= 6.0  # Minimum 6GB for fine-tuning
        requirements_table.add_row(
            "GPU",
            "✅ OK" if gpu_ok else "⚠️  LIMITED",
            gpu_status + (" (6GB+ recommended)" if not gpu_ok else "")
        )
    else:
        requirements_table.add_row("GPU", "❌ NONE", "CPU-only mode (slow training)")
    
    # Check PEFT
    try:
        import peft
        requirements_table.add_row("PEFT", "✅ OK", f"v{peft.__version__}")
        peft_ok = True
    except ImportError:
        requirements_table.add_row("PEFT", "❌ MISSING", "Run: pip install peft")
        peft_ok = False
    
    # Check BitsAndBytes
    try:
        import bitsandbytes
        requirements_table.add_row("BitsAndBytes", "✅ OK", "Quantization support")
        bnb_ok = True
    except ImportError:
        requirements_table.add_row("BitsAndBytes", "❌ MISSING", "Run: pip install bitsandbytes")
        bnb_ok = False
    
    console.print(requirements_table)
    
    return python_ok and torch_ok and peft_ok

def check_example_datasets():
    """Check if example datasets are available."""
    console.print("\n[cyan]📁 Checking Example Datasets...[/cyan]")
    
    datasets_dir = Path("examples/datasets")
    example_files = [
        "chat_training.csv",
        "instruction_following.jsonl", 
        "classification_training.csv"
    ]
    
    datasets_table = Table(show_header=True, header_style="bold cyan")
    datasets_table.add_column("Dataset", style="white")
    datasets_table.add_column("Status", style="green")
    datasets_table.add_column("Use Case", style="dim")
    
    for filename in example_files:
        filepath = datasets_dir / filename
        if filepath.exists():
            status = "✅ Available"
            file_size = filepath.stat().st_size
            use_case = {
                "chat_training.csv": f"Conversational AI ({file_size} bytes)",
                "instruction_following.jsonl": f"Task completion ({file_size} bytes)",
                "classification_training.csv": f"Text classification ({file_size} bytes)"
            }[filename]
        else:
            status = "❌ Missing"
            use_case = "Example dataset not found"
        
        datasets_table.add_row(filename, status, use_case)
    
    console.print(datasets_table)

def show_quick_start_guide():
    """Show a quick start guide."""
    guide_text = """
[bold cyan]🚀 Quick Start Guide:[/bold cyan]

[bold yellow]1. Start the Fine-Tuning CLI:[/bold yellow]
   python cli/finetuning_cli.py

[bold yellow]2. Try Your First Fine-Tuning:[/bold yellow]
   FT> list models                              # See available models
   FT> finetune mistral-7b examples/datasets/chat_training.csv chat

[bold yellow]3. Load and Test Your Adapter:[/bold yellow]
   FT> list adapters                            # See trained adapters
   FT> load adapter [adapter_name]              # Load for inference
   FT> benchmark [adapter_name]                 # Test performance

[bold yellow]4. Hot-Swap Adapters (Advanced):[/bold yellow]
   # In API CLI:
   python cli/api_cli.py
   API> load [model_name]
   
   # In Fine-Tuning CLI:
   python cli/finetuning_cli.py
   FT> load adapter [adapter_name]
   FT> switch adapter [different_adapter]       # Instant switch!

[bold yellow]5. Create Custom Datasets:[/bold yellow]
   • Chat: CSV with 'input,output' columns
   • Classification: CSV with 'text,label' columns  
   • Instructions: JSONL with 'instruction,input,output' fields

[dim]💡 Tip: Start with the example datasets to familiarize yourself with the system.[/dim]
    """
    
    console.print(Panel(guide_text.strip(), title="Quick Start", expand=False))

def show_memory_recommendations():
    """Show memory optimization recommendations."""
    if not torch.cuda.is_available():
        return
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    console.print(f"\n[cyan]💾 Memory Recommendations for {gpu_memory:.1f}GB GPU:[/cyan]")
    
    if gpu_memory >= 12:
        console.print("[green]🎯 High-end GPU detected![/green]")
        console.print("• Batch size: 4-8")
        console.print("• Sequence length: 1024+")
        console.print("• Can train larger models efficiently")
    elif gpu_memory >= 8:
        console.print("[green]🎯 Great for fine-tuning![/green]")
        console.print("• Batch size: 2-4")
        console.print("• Sequence length: 512-1024")
        console.print("• Optimal for most models")
    elif gpu_memory >= 6:
        console.print("[yellow]⚠️  Limited but workable[/yellow]")
        console.print("• Batch size: 1-2")
        console.print("• Sequence length: 256-512")
        console.print("• Use gradient accumulation")
    else:
        console.print("[red]⚠️  Very limited GPU memory[/red]")
        console.print("• Batch size: 1")
        console.print("• Sequence length: 256")
        console.print("• Consider CPU training for small models")

def main():
    """Main setup verification."""
    console.print(Panel.fit(
        "🧠 LQMF Fine-Tuning Setup Verification",
        style="bold blue"
    ))
    
    # Check system requirements
    system_ready = check_system_requirements()
    
    # Check example datasets
    check_example_datasets()
    
    # Show memory recommendations
    show_memory_recommendations()
    
    # Overall status
    console.print("\n" + "="*60)
    
    if system_ready:
        console.print("[bold green]🎉 System is ready for fine-tuning![/bold green]")
        show_quick_start_guide()
    else:
        console.print("[bold red]❌ System setup incomplete[/bold red]")
        console.print("Please install missing dependencies:")
        console.print("  python install_finetuning_deps.py")
    
    console.print("\n[dim]📚 For detailed documentation, see: FINETUNING_README.md[/dim]")

if __name__ == "__main__":
    main()