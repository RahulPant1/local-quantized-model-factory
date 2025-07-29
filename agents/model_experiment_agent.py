#!/usr/bin/env python3
"""
Model Experiment Agent - Interface for testing and experimenting with loaded models
Provides interactive testing, batch testing, and API experimentation capabilities
"""

import json
import time
import requests
import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import related agents
import sys
sys.path.append(str(Path(__file__).parent.parent))
from agents.api_server_agent import APIServerAgent
from agents.memory_agent import MemoryAgent

console = Console()

class ExperimentType(Enum):
    """Types of experiments"""
    SINGLE_PROMPT = "single_prompt"
    BATCH_PROMPTS = "batch_prompts"
    CONVERSATION = "conversation"
    BENCHMARK = "benchmark"
    COMPARISON = "comparison"

@dataclass
class ExperimentResult:
    """Result of a model experiment"""
    experiment_id: str
    model_name: str
    experiment_type: ExperimentType
    prompt: str
    response: str
    tokens_generated: int
    time_taken: float
    timestamp: datetime
    metadata: Dict[str, Any]
    quality_score: Optional[float] = None
    error_message: Optional[str] = None

class ModelExperimentAgent:
    """Agent for testing and experimenting with loaded models"""
    
    def __init__(self, api_agent: APIServerAgent = None):
        self.api_agent = api_agent or APIServerAgent()
        self.memory_agent = MemoryAgent()
        self.experiment_results: List[ExperimentResult] = []
        
        # Default test prompts for quick testing
        self.default_test_prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about nature.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "Tell me about the history of computers.",
            "What is machine learning?",
            "Explain the concept of recursion.",
            "Write a short story about a robot.",
            "What are the challenges of space exploration?"
        ]
        
        # Conversation starters
        self.conversation_starters = [
            "Hello! How are you today?",
            "Can you help me with a programming problem?",
            "I'm learning about AI. Can you explain neural networks?",
            "What do you think about the future of technology?",
            "Can you write a creative story for me?"
        ]
    
    def discover_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models for experimentation"""
        return self.api_agent.discover_quantized_models()
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self.api_agent.loaded_models.keys())
    
    def ensure_model_loaded(self, model_name: str) -> bool:
        """Ensure a model is loaded for experimentation"""
        if model_name in self.api_agent.loaded_models:
            return True
        
        console.print(f"[yellow]Model {model_name} not loaded. Loading now...[/yellow]")
        return self.api_agent.load_model(model_name)
    
    def single_prompt_experiment(self, model_name: str, prompt: str, 
                                max_tokens: int = 100, temperature: float = 0.7) -> ExperimentResult:
        """Run a single prompt experiment"""
        if not self.ensure_model_loaded(model_name):
            return ExperimentResult(
                experiment_id=f"exp_{int(time.time())}",
                model_name=model_name,
                experiment_type=ExperimentType.SINGLE_PROMPT,
                prompt=prompt,
                response="",
                tokens_generated=0,
                time_taken=0,
                timestamp=datetime.now(),
                metadata={},
                error_message="Failed to load model"
            )
        
        try:
            # Test via API if server is running
            if hasattr(self.api_agent, 'server_thread') and self.api_agent.server_thread:
                result = self._test_via_api(model_name, prompt, max_tokens, temperature)
            else:
                # Test directly
                result = self._test_directly(model_name, prompt, max_tokens, temperature)
            
            # Store result
            self.experiment_results.append(result)
            
            # Log to memory agent
            self.memory_agent.log_experiment(
                model_name=model_name,
                quantization_method=self.api_agent.loaded_models[model_name].quantization_method,
                bit_width=self.api_agent.loaded_models[model_name].bit_width,
                status="experiment_completed",
                metadata={
                    "experiment_type": "single_prompt",
                    "tokens_generated": result.tokens_generated,
                    "time_taken": result.time_taken,
                    "quality_score": result.quality_score
                }
            )
            
            return result
            
        except Exception as e:
            error_result = ExperimentResult(
                experiment_id=f"exp_{int(time.time())}",
                model_name=model_name,
                experiment_type=ExperimentType.SINGLE_PROMPT,
                prompt=prompt,
                response="",
                tokens_generated=0,
                time_taken=0,
                timestamp=datetime.now(),
                metadata={"max_tokens": max_tokens, "temperature": temperature},
                error_message=str(e)
            )
            self.experiment_results.append(error_result)
            return error_result
    
    def _test_via_api(self, model_name: str, prompt: str, max_tokens: int, temperature: float) -> ExperimentResult:
        """Test model via API endpoint"""
        url = f"http://{self.api_agent.server_host}:{self.api_agent.server_port}/models/{model_name}/chat"
        
        payload = {
            "message": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        start_time = time.time()
        response = requests.post(url, json=payload)
        time_taken = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return ExperimentResult(
                experiment_id=f"exp_{int(time.time())}",
                model_name=model_name,
                experiment_type=ExperimentType.SINGLE_PROMPT,
                prompt=prompt,
                response=data["response"],
                tokens_generated=data["tokens_generated"],
                time_taken=data["time_taken"],
                timestamp=datetime.now(),
                metadata={"max_tokens": max_tokens, "temperature": temperature, "via_api": True},
                quality_score=self._calculate_quality_score(prompt, data["response"])
            )
        else:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
    
    def _test_directly(self, model_name: str, prompt: str, max_tokens: int, temperature: float) -> ExperimentResult:
        """Test model directly without API"""
        model_info = self.api_agent.loaded_models[model_name]
        
        start_time = time.time()
        response = self.api_agent._generate_response(model_info, prompt, max_tokens, temperature)
        time_taken = time.time() - start_time
        
        return ExperimentResult(
            experiment_id=f"exp_{int(time.time())}",
            model_name=model_name,
            experiment_type=ExperimentType.SINGLE_PROMPT,
            prompt=prompt,
            response=response,
            tokens_generated=len(response.split()),
            time_taken=time_taken,
            timestamp=datetime.now(),
            metadata={"max_tokens": max_tokens, "temperature": temperature, "via_api": False},
            quality_score=self._calculate_quality_score(prompt, response)
        )
    
    def batch_prompt_experiment(self, model_name: str, prompts: List[str], 
                               max_tokens: int = 100, temperature: float = 0.7) -> List[ExperimentResult]:
        """Run batch prompt experiments"""
        if not self.ensure_model_loaded(model_name):
            return []
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Testing {model_name}", total=len(prompts))
            
            for i, prompt in enumerate(prompts):
                result = self.single_prompt_experiment(model_name, prompt, max_tokens, temperature)
                results.append(result)
                progress.update(task, advance=1)
        
        return results
    
    def conversation_experiment(self, model_name: str, max_turns: int = 5) -> List[ExperimentResult]:
        """Run a conversation experiment"""
        if not self.ensure_model_loaded(model_name):
            return []
        
        results = []
        conversation_history = []
        
        console.print(f"[cyan]Starting conversation with {model_name}[/cyan]")
        console.print("[dim]Type 'quit' to end the conversation[/dim]")
        
        # Start with a greeting
        starter = self.conversation_starters[0]
        result = self.single_prompt_experiment(model_name, starter)
        results.append(result)
        conversation_history.append(f"Human: {starter}")
        conversation_history.append(f"Assistant: {result.response}")
        
        console.print(f"[green]Assistant:[/green] {result.response}")
        
        for turn in range(max_turns - 1):
            try:
                user_input = input(f"\n[You]: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                # Add context from conversation history
                context = "\n".join(conversation_history[-4:])  # Last 2 turns
                full_prompt = f"{context}\nHuman: {user_input}\nAssistant:"
                
                result = self.single_prompt_experiment(model_name, full_prompt)
                results.append(result)
                
                conversation_history.append(f"Human: {user_input}")
                conversation_history.append(f"Assistant: {result.response}")
                
                console.print(f"[green]Assistant:[/green] {result.response}")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Conversation ended[/yellow]")
                break
        
        return results
    
    def benchmark_experiment(self, model_name: str) -> Dict[str, Any]:
        """Run comprehensive benchmark experiment"""
        if not self.ensure_model_loaded(model_name):
            return {}
        
        console.print(f"[cyan]Running benchmark for {model_name}[/cyan]")
        
        # Run batch test with default prompts
        results = self.batch_prompt_experiment(model_name, self.default_test_prompts)
        
        # Calculate metrics
        total_tokens = sum(r.tokens_generated for r in results)
        total_time = sum(r.time_taken for r in results)
        avg_quality = sum(r.quality_score for r in results if r.quality_score) / len(results)
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        benchmark_data = {
            "model_name": model_name,
            "total_prompts": len(results),
            "total_tokens": total_tokens,
            "total_time": total_time,
            "avg_quality_score": avg_quality,
            "tokens_per_second": tokens_per_second,
            "successful_responses": len([r for r in results if not r.error_message]),
            "failed_responses": len([r for r in results if r.error_message]),
            "timestamp": datetime.now().isoformat()
        }
        
        # Display results
        self._display_benchmark_results(benchmark_data)
        
        return benchmark_data
    
    def comparison_experiment(self, model_names: List[str], test_prompts: List[str] = None) -> Dict[str, Any]:
        """Compare multiple models on the same prompts"""
        if not test_prompts:
            test_prompts = self.default_test_prompts[:5]  # Use first 5 for comparison
        
        comparison_data = {
            "models": model_names,
            "prompts": test_prompts,
            "results": {},
            "summary": {}
        }
        
        # Test each model
        for model_name in model_names:
            console.print(f"[cyan]Testing {model_name}[/cyan]")
            
            if not self.ensure_model_loaded(model_name):
                comparison_data["results"][model_name] = {"error": "Failed to load model"}
                continue
            
            model_results = self.batch_prompt_experiment(model_name, test_prompts)
            
            # Calculate metrics for this model
            total_tokens = sum(r.tokens_generated for r in model_results)
            total_time = sum(r.time_taken for r in model_results)
            avg_quality = sum(r.quality_score for r in model_results if r.quality_score) / len(model_results)
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            
            comparison_data["results"][model_name] = {
                "total_tokens": total_tokens,
                "total_time": total_time,
                "avg_quality_score": avg_quality,
                "tokens_per_second": tokens_per_second,
                "results": model_results
            }
        
        # Generate comparison summary
        self._display_comparison_results(comparison_data)
        
        return comparison_data
    
    def _calculate_quality_score(self, prompt: str, response: str) -> float:
        """Calculate a simple quality score for the response"""
        # Simple heuristic-based quality scoring
        score = 0.0
        
        # Length check (not too short, not too long)
        if 10 <= len(response) <= 500:
            score += 0.3
        
        # Relevance check (contains words from prompt)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words.intersection(response_words))
        if overlap > 0:
            score += min(0.3, overlap * 0.1)
        
        # Coherence check (not too repetitive)
        response_words_list = response.lower().split()
        if len(response_words_list) > 0:
            unique_ratio = len(set(response_words_list)) / len(response_words_list)
            score += unique_ratio * 0.4
        
        return min(1.0, score)
    
    def _display_benchmark_results(self, benchmark_data: Dict[str, Any]):
        """Display benchmark results in a nice format"""
        table = Table(title=f"Benchmark Results - {benchmark_data['model_name']}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Prompts", str(benchmark_data["total_prompts"]))
        table.add_row("Total Tokens", str(benchmark_data["total_tokens"]))
        table.add_row("Total Time (s)", f"{benchmark_data['total_time']:.2f}")
        table.add_row("Avg Quality Score", f"{benchmark_data['avg_quality_score']:.2f}")
        table.add_row("Tokens/Second", f"{benchmark_data['tokens_per_second']:.2f}")
        table.add_row("Successful", str(benchmark_data["successful_responses"]))
        table.add_row("Failed", str(benchmark_data["failed_responses"]))
        
        console.print(table)
    
    def _display_comparison_results(self, comparison_data: Dict[str, Any]):
        """Display model comparison results"""
        table = Table(title="Model Comparison Results")
        table.add_column("Model", style="cyan")
        table.add_column("Avg Quality", style="green")
        table.add_column("Tokens/Sec", style="yellow")
        table.add_column("Total Time", style="red")
        table.add_column("Total Tokens", style="magenta")
        
        for model_name, results in comparison_data["results"].items():
            if "error" in results:
                table.add_row(model_name, "ERROR", "N/A", "N/A", "N/A")
            else:
                table.add_row(
                    model_name,
                    f"{results['avg_quality_score']:.2f}",
                    f"{results['tokens_per_second']:.2f}",
                    f"{results['total_time']:.2f}s",
                    str(results['total_tokens'])
                )
        
        console.print(table)
    
    def interactive_experiment_menu(self):
        """Interactive menu for running experiments"""
        while True:
            console.print("\n[bold cyan]Model Experiment Menu[/bold cyan]")
            console.print("1. Single Prompt Test")
            console.print("2. Batch Prompt Test")
            console.print("3. Conversation Test")
            console.print("4. Benchmark Test")
            console.print("5. Model Comparison")
            console.print("6. Show Available Models")
            console.print("7. Show Loaded Models")
            console.print("8. Load Model")
            console.print("9. Show Experiment History")
            console.print("0. Exit")
            
            try:
                choice = input("\nEnter your choice (0-9): ").strip()
                
                if choice == "0":
                    break
                elif choice == "1":
                    self._interactive_single_prompt()
                elif choice == "2":
                    self._interactive_batch_prompt()
                elif choice == "3":
                    self._interactive_conversation()
                elif choice == "4":
                    self._interactive_benchmark()
                elif choice == "5":
                    self._interactive_comparison()
                elif choice == "6":
                    self.api_agent.show_available_models()
                elif choice == "7":
                    self.api_agent.show_loaded_models()
                elif choice == "8":
                    self._interactive_load_model()
                elif choice == "9":
                    self._show_experiment_history()
                else:
                    console.print("[red]Invalid choice. Please try again.[/red]")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Exiting...[/yellow]")
                break
    
    def _interactive_single_prompt(self):
        """Interactive single prompt testing"""
        loaded_models = self.get_loaded_models()
        if not loaded_models:
            console.print("[yellow]No models loaded. Please load a model first.[/yellow]")
            return
        
        # Select model
        if len(loaded_models) == 1:
            model_name = loaded_models[0]
        else:
            console.print("Available models:")
            for i, model in enumerate(loaded_models):
                console.print(f"{i+1}. {model}")
            
            try:
                choice = int(input("Select model (number): ")) - 1
                model_name = loaded_models[choice]
            except (ValueError, IndexError):
                console.print("[red]Invalid selection[/red]")
                return
        
        # Get prompt
        prompt = input("Enter your prompt: ").strip()
        if not prompt:
            console.print("[red]Empty prompt[/red]")
            return
        
        # Get parameters
        max_tokens = int(input("Max tokens (default 100): ") or "100")
        temperature = float(input("Temperature (default 0.7): ") or "0.7")
        
        # Run experiment
        console.print(f"[cyan]Testing with {model_name}...[/cyan]")
        result = self.single_prompt_experiment(model_name, prompt, max_tokens, temperature)
        
        # Display result
        console.print(f"[green]Response:[/green] {result.response}")
        console.print(f"[dim]Tokens: {result.tokens_generated}, Time: {result.time_taken:.2f}s, Quality: {result.quality_score:.2f}[/dim]")
    
    def _interactive_batch_prompt(self):
        """Interactive batch prompt testing"""
        loaded_models = self.get_loaded_models()
        if not loaded_models:
            console.print("[yellow]No models loaded. Please load a model first.[/yellow]")
            return
        
        # Select model
        if len(loaded_models) == 1:
            model_name = loaded_models[0]
        else:
            console.print("Available models:")
            for i, model in enumerate(loaded_models):
                console.print(f"{i+1}. {model}")
            
            try:
                choice = int(input("Select model (number): ")) - 1
                model_name = loaded_models[choice]
            except (ValueError, IndexError):
                console.print("[red]Invalid selection[/red]")
                return
        
        # Choose prompts
        use_default = Confirm.ask("Use default test prompts?")
        if use_default:
            prompts = self.default_test_prompts
        else:
            prompts = []
            console.print("Enter prompts (empty line to finish):")
            while True:
                prompt = input(f"Prompt {len(prompts)+1}: ").strip()
                if not prompt:
                    break
                prompts.append(prompt)
        
        if not prompts:
            console.print("[red]No prompts provided[/red]")
            return
        
        # Run experiments
        results = self.batch_prompt_experiment(model_name, prompts)
        
        # Display summary
        successful = len([r for r in results if not r.error_message])
        total_tokens = sum(r.tokens_generated for r in results)
        total_time = sum(r.time_taken for r in results)
        avg_quality = sum(r.quality_score for r in results if r.quality_score) / len(results)
        
        console.print(f"[green]Batch test completed![/green]")
        console.print(f"Successful: {successful}/{len(results)}")
        console.print(f"Total tokens: {total_tokens}")
        console.print(f"Total time: {total_time:.2f}s")
        console.print(f"Average quality: {avg_quality:.2f}")
    
    def _interactive_conversation(self):
        """Interactive conversation testing"""
        loaded_models = self.get_loaded_models()
        if not loaded_models:
            console.print("[yellow]No models loaded. Please load a model first.[/yellow]")
            return
        
        # Select model
        if len(loaded_models) == 1:
            model_name = loaded_models[0]
        else:
            console.print("Available models:")
            for i, model in enumerate(loaded_models):
                console.print(f"{i+1}. {model}")
            
            try:
                choice = int(input("Select model (number): ")) - 1
                model_name = loaded_models[choice]
            except (ValueError, IndexError):
                console.print("[red]Invalid selection[/red]")
                return
        
        # Get max turns
        max_turns = int(input("Max conversation turns (default 5): ") or "5")
        
        # Run conversation
        results = self.conversation_experiment(model_name, max_turns)
        
        console.print(f"[green]Conversation completed with {len(results)} turns![/green]")
    
    def _interactive_benchmark(self):
        """Interactive benchmark testing"""
        loaded_models = self.get_loaded_models()
        if not loaded_models:
            console.print("[yellow]No models loaded. Please load a model first.[/yellow]")
            return
        
        # Select model
        if len(loaded_models) == 1:
            model_name = loaded_models[0]
        else:
            console.print("Available models:")
            for i, model in enumerate(loaded_models):
                console.print(f"{i+1}. {model}")
            
            try:
                choice = int(input("Select model (number): ")) - 1
                model_name = loaded_models[choice]
            except (ValueError, IndexError):
                console.print("[red]Invalid selection[/red]")
                return
        
        # Run benchmark
        benchmark_data = self.benchmark_experiment(model_name)
    
    def _interactive_comparison(self):
        """Interactive model comparison"""
        loaded_models = self.get_loaded_models()
        if len(loaded_models) < 2:
            console.print("[yellow]Need at least 2 loaded models for comparison.[/yellow]")
            return
        
        # Select models for comparison
        console.print("Available models:")
        for i, model in enumerate(loaded_models):
            console.print(f"{i+1}. {model}")
        
        try:
            choices = input("Select models for comparison (comma-separated numbers): ").split(',')
            selected_models = [loaded_models[int(c.strip())-1] for c in choices]
        except (ValueError, IndexError):
            console.print("[red]Invalid selection[/red]")
            return
        
        # Run comparison
        comparison_data = self.comparison_experiment(selected_models)
    
    def _interactive_load_model(self):
        """Interactive model loading"""
        model_name = self.api_agent.interactive_model_selection()
        if model_name:
            success = self.api_agent.load_model(model_name)
            if success:
                console.print(f"[green]Model {model_name} loaded successfully![/green]")
            else:
                console.print(f"[red]Failed to load model {model_name}[/red]")
    
    def _show_experiment_history(self):
        """Show experiment history"""
        if not self.experiment_results:
            console.print("[yellow]No experiments run yet[/yellow]")
            return
        
        table = Table(title="Experiment History")
        table.add_column("ID", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Tokens", style="magenta")
        table.add_column("Time (s)", style="red")
        table.add_column("Quality", style="blue")
        
        for result in self.experiment_results[-10:]:  # Show last 10
            table.add_row(
                result.experiment_id[-8:],  # Show last 8 chars
                result.model_name,
                result.experiment_type.value,
                str(result.tokens_generated),
                f"{result.time_taken:.2f}",
                f"{result.quality_score:.2f}" if result.quality_score else "N/A"
            )
        
        console.print(table)


def main():
    """Test the Model Experiment Agent"""
    console.print(Panel.fit("🧪 Testing Model Experiment Agent", style="bold blue"))
    
    # Initialize agents
    api_agent = APIServerAgent()
    experiment_agent = ModelExperimentAgent(api_agent)
    
    # Show available models
    console.print("\n[cyan]Available Models:[/cyan]")
    experiment_agent.api_agent.show_available_models()
    
    # Run interactive menu
    experiment_agent.interactive_experiment_menu()


if __name__ == "__main__":
    main()