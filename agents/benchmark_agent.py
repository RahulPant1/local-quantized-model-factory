#!/usr/bin/env python3
"""
Benchmark Agent for LQMF - Comprehensive Model and Adapter Performance Testing

This agent provides sophisticated benchmarking capabilities for both base models
and fine-tuned adapters, including performance metrics, quality assessment,
and comparison tools.

Author: LQMF Development Team
Version: 1.0.0
Compatibility: Python 3.8+

Features:
- Performance benchmarking (speed, throughput, memory)
- Quality assessment (BLEU, ROUGE, perplexity)
- Comparative analysis between models/adapters
- Automated test suite generation
- Memory efficiency profiling
- Real-time monitoring during inference
"""

import os
import json
import time
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import psutil

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

# Import LQMF components
import sys
sys.path.append(str(Path(__file__).parent.parent))

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking jobs"""
    benchmark_name: str
    model_name: str
    adapter_name: Optional[str] = None
    test_prompts: List[str] = None
    max_tokens: int = 100
    temperature: float = 0.7
    num_runs: int = 5
    warmup_runs: int = 1
    batch_size: int = 1
    include_quality_metrics: bool = True
    include_memory_profiling: bool = True
    
@dataclass
class PerformanceMetrics:
    """Performance metrics for a benchmark run"""
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    std_response_time: float
    tokens_per_second: float
    total_tokens: int
    memory_usage_mb: float
    gpu_memory_usage_mb: Optional[float] = None
    cpu_usage_percent: float = 0.0

@dataclass
class QualityMetrics:
    """Quality metrics for generated text"""
    bleu_score: Optional[float] = None
    rouge_l_score: Optional[float] = None
    perplexity: Optional[float] = None
    avg_sequence_length: float = 0.0
    repetition_ratio: float = 0.0
    
@dataclass
class BenchmarkResult:
    """Complete benchmark result"""
    benchmark_name: str
    model_name: str
    adapter_name: Optional[str]
    config: BenchmarkConfig
    performance: PerformanceMetrics
    quality: Optional[QualityMetrics]
    responses: List[Dict[str, Any]]
    timestamp: str
    duration: float

class BenchmarkAgent:
    """
    Comprehensive benchmarking agent for models and adapters.
    
    Provides detailed performance and quality analysis with support for
    comparative benchmarking and automated report generation.
    """
    
    def __init__(self, 
                 benchmarks_dir: str = "benchmarks",
                 reports_dir: str = "reports"):
        """
        Initialize the Benchmark Agent.
        
        Args:
            benchmarks_dir: Directory to store benchmark results
            reports_dir: Directory to store generated reports
        """
        self.benchmarks_dir = Path(benchmarks_dir)
        self.reports_dir = Path(reports_dir)
        
        # Create directories
        self.benchmarks_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Check dependencies
        self._check_dependencies()
        
        # Load default test prompts
        self.default_prompts = self._load_default_prompts()
        
        console.print("[green]✅ Benchmark Agent initialized[/green]")
    
    def _check_dependencies(self):
        """Check optional dependencies for quality metrics."""
        if not NLTK_AVAILABLE:
            console.print("[yellow]⚠️  NLTK not available. BLEU scores disabled.[/yellow]")
        else:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                console.print("[blue]📥 Downloading NLTK data...[/blue]")
                nltk.download('punkt', quiet=True)
        
        if not ROUGE_AVAILABLE:
            console.print("[yellow]⚠️  rouge-score not available. ROUGE scores disabled.[/yellow]")
    
    def _load_default_prompts(self) -> List[str]:
        """Load default test prompts."""
        return [
            "What is artificial intelligence and how does it work?",
            "Explain the concept of machine learning in simple terms.",
            "Describe the benefits and risks of renewable energy.",
            "How does natural language processing enable computers to understand human language?",
            "What are the main differences between supervised and unsupervised learning?",
            "Explain the water cycle and its importance to life on Earth.",
            "What role does data play in modern business decision making?",
            "Describe the basic principles of quantum computing.",
            "How do neural networks learn from data?",
            "What are the ethical considerations in AI development?"
        ]
    
    def run_benchmark(self, 
                     model_instance: nn.Module,
                     tokenizer: Any,
                     config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run a comprehensive benchmark on a model.
        
        Args:
            model_instance: The model to benchmark
            tokenizer: Associated tokenizer
            config: Benchmark configuration
            
        Returns:
            Complete benchmark results
        """
        console.print(f"[blue]📊 Starting benchmark: {config.benchmark_name}[/blue]")
        
        start_time = time.time()
        test_prompts = config.test_prompts or self.default_prompts
        
        # Warmup runs
        if config.warmup_runs > 0:
            console.print("[dim]🔥 Running warmup...[/dim]")
            self._run_warmup(model_instance, tokenizer, config, test_prompts[:2])
        
        # Main benchmark runs
        all_responses = []
        performance_data = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True
        ) as progress:
            
            total_runs = config.num_runs * len(test_prompts)
            task = progress.add_task(f"Benchmarking {config.model_name}...", total=total_runs)
            
            for run_idx in range(config.num_runs):
                for prompt_idx, prompt in enumerate(test_prompts):
                    
                    # Run single inference
                    response_data = self._run_single_inference(
                        model_instance, tokenizer, prompt, config
                    )
                    
                    all_responses.append({
                        "run": run_idx + 1,
                        "prompt_idx": prompt_idx,
                        "prompt": prompt,
                        **response_data
                    })
                    
                    performance_data.append(response_data)
                    progress.update(task, advance=1)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(performance_data)
        
        # Calculate quality metrics
        quality = None
        if config.include_quality_metrics:
            quality = self._calculate_quality_metrics(all_responses)
        
        # Create benchmark result
        result = BenchmarkResult(
            benchmark_name=config.benchmark_name,
            model_name=config.model_name,
            adapter_name=config.adapter_name,
            config=config,
            performance=performance,
            quality=quality,
            responses=all_responses,
            timestamp=datetime.now().isoformat(),
            duration=time.time() - start_time
        )
        
        # Save results
        self._save_benchmark_result(result)
        
        console.print(f"[green]✅ Benchmark completed in {result.duration:.2f}s[/green]")
        return result
    
    def _run_warmup(self, 
                   model_instance: nn.Module,
                   tokenizer: Any,
                   config: BenchmarkConfig,
                   warmup_prompts: List[str]):
        """Run warmup inferences."""
        for _ in range(config.warmup_runs):
            for prompt in warmup_prompts:
                self._run_single_inference(model_instance, tokenizer, prompt, config)
    
    def _run_single_inference(self, 
                             model_instance: nn.Module,
                             tokenizer: Any,
                             prompt: str,
                             config: BenchmarkConfig) -> Dict[str, Any]:
        """Run a single inference and collect metrics."""
        
        # Memory before inference
        memory_before = psutil.Process().memory_info().rss / 1024**2
        gpu_memory_before = None
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**2
        
        # CPU usage monitoring
        cpu_percent_before = psutil.cpu_percent(interval=None)
        
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_length = len(inputs['input_ids'][0])
            
            # Generate response
            with torch.no_grad():
                outputs = model_instance.generate(
                    **inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=inputs.get('attention_mask')
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            
            end_time = time.time()
            
            # Calculate metrics
            response_time = end_time - start_time
            output_length = len(outputs[0]) - input_length
            tokens_per_second = output_length / response_time if response_time > 0 else 0
            
            # Memory after inference
            memory_after = psutil.Process().memory_info().rss / 1024**2
            memory_used = memory_after - memory_before
            
            gpu_memory_used = None
            if torch.cuda.is_available() and gpu_memory_before is not None:
                gpu_memory_after = torch.cuda.memory_allocated() / 1024**2
                gpu_memory_used = gpu_memory_after - gpu_memory_before
            
            cpu_percent_after = psutil.cpu_percent(interval=None)
            cpu_usage = (cpu_percent_after + cpu_percent_before) / 2
            
            return {
                "response": response,
                "response_time": response_time,
                "input_tokens": input_length,
                "output_tokens": output_length,
                "total_tokens": input_length + output_length,
                "tokens_per_second": tokens_per_second,
                "memory_used_mb": memory_used,
                "gpu_memory_used_mb": gpu_memory_used,
                "cpu_usage_percent": cpu_usage,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                "response": "",
                "response_time": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "tokens_per_second": 0,
                "memory_used_mb": 0,
                "gpu_memory_used_mb": None,
                "cpu_usage_percent": 0,
                "success": False,
                "error": str(e)
            }
    
    def _calculate_performance_metrics(self, performance_data: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Calculate aggregated performance metrics."""
        successful_runs = [r for r in performance_data if r.get('success', True)]
        
        if not successful_runs:
            raise ValueError("No successful inference runs")
        
        response_times = [r['response_time'] for r in successful_runs]
        tokens_per_second = [r['tokens_per_second'] for r in successful_runs]
        memory_usage = [r['memory_used_mb'] for r in successful_runs]
        cpu_usage = [r['cpu_usage_percent'] for r in successful_runs]
        
        # GPU memory (if available)
        gpu_memory_usage = None
        gpu_memory_values = [r['gpu_memory_used_mb'] for r in successful_runs if r['gpu_memory_used_mb'] is not None]
        if gpu_memory_values:
            gpu_memory_usage = statistics.mean(gpu_memory_values)
        
        return PerformanceMetrics(
            avg_response_time=statistics.mean(response_times),
            min_response_time=min(response_times),
            max_response_time=max(response_times),
            std_response_time=statistics.stdev(response_times) if len(response_times) > 1 else 0,
            tokens_per_second=statistics.mean(tokens_per_second),
            total_tokens=sum(r['total_tokens'] for r in successful_runs),
            memory_usage_mb=statistics.mean(memory_usage),
            gpu_memory_usage_mb=gpu_memory_usage,
            cpu_usage_percent=statistics.mean(cpu_usage)
        )
    
    def _calculate_quality_metrics(self, responses: List[Dict[str, Any]]) -> QualityMetrics:
        """Calculate quality metrics for generated responses."""
        successful_responses = [r for r in responses if r.get('success', True) and r.get('response')]
        
        if not successful_responses:
            return QualityMetrics()
        
        responses_text = [r['response'] for r in successful_responses]
        
        # Calculate basic metrics
        avg_length = statistics.mean([len(r.split()) for r in responses_text])
        
        # Calculate repetition ratio
        repetition_ratios = []
        for response in responses_text:
            words = response.split()
            if len(words) > 1:
                unique_words = len(set(words))
                repetition_ratio = 1 - (unique_words / len(words))
                repetition_ratios.append(repetition_ratio)
        
        avg_repetition = statistics.mean(repetition_ratios) if repetition_ratios else 0
        
        # BLEU score (if reference available and NLTK available)
        bleu_score = None
        if NLTK_AVAILABLE:
            # For auto-evaluation, we'll skip BLEU as it requires references
            pass
        
        # ROUGE score (if available)
        rouge_score = None
        if ROUGE_AVAILABLE:
            # For auto-evaluation, we'll skip ROUGE as it requires references
            pass
        
        return QualityMetrics(
            bleu_score=bleu_score,
            rouge_l_score=rouge_score,
            avg_sequence_length=avg_length,
            repetition_ratio=avg_repetition
        )
    
    def _save_benchmark_result(self, result: BenchmarkResult):
        """Save benchmark result to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.benchmark_name}_{timestamp}.json"
        filepath = self.benchmarks_dir / filename
        
        # Convert to dict for JSON serialization
        result_dict = asdict(result)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            console.print(f"[green]💾 Results saved: {filepath}[/green]")
        except Exception as e:
            logger.error(f"Failed to save benchmark result: {e}")
            console.print(f"[red]❌ Failed to save results: {e}[/red]")
    
    def compare_models(self, 
                      results: List[BenchmarkResult],
                      metric: str = "avg_response_time") -> Table:
        """
        Create a comparison table for multiple benchmark results.
        
        Args:
            results: List of benchmark results to compare
            metric: Primary metric for comparison
            
        Returns:
            Rich table with comparison data
        """
        table = Table(show_header=True, header_style="bold cyan", title="Model Comparison")
        
        table.add_column("Model", style="white")
        table.add_column("Adapter", style="blue")
        table.add_column("Avg Time (s)", style="yellow")
        table.add_column("Tokens/sec", style="green")
        table.add_column("Memory (MB)", style="magenta")
        table.add_column("Quality Score", style="cyan")
        
        for result in results:
            adapter_name = result.adapter_name or "-"
            
            # Calculate quality score (simple average of available metrics)
            quality_score = "N/A"
            if result.quality:
                scores = []
                if result.quality.bleu_score is not None:
                    scores.append(result.quality.bleu_score)
                if result.quality.rouge_l_score is not None:
                    scores.append(result.quality.rouge_l_score)
                if scores:
                    quality_score = f"{statistics.mean(scores):.3f}"
            
            table.add_row(
                result.model_name,
                adapter_name,
                f"{result.performance.avg_response_time:.3f}",
                f"{result.performance.tokens_per_second:.1f}",
                f"{result.performance.memory_usage_mb:.1f}",
                quality_score
            )
        
        return table
    
    def generate_report(self, 
                       result: BenchmarkResult,
                       include_responses: bool = False) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            result: Benchmark result to report on
            include_responses: Whether to include sample responses
            
        Returns:
            Formatted report as string
        """
        report_lines = [
            f"# Benchmark Report: {result.benchmark_name}",
            f"**Generated:** {result.timestamp}",
            f"**Duration:** {result.duration:.2f}s",
            f"**Model:** {result.model_name}",
            f"**Adapter:** {result.adapter_name or 'None'}",
            "",
            "## Performance Metrics",
            f"- **Average Response Time:** {result.performance.avg_response_time:.3f}s",
            f"- **Min/Max Response Time:** {result.performance.min_response_time:.3f}s / {result.performance.max_response_time:.3f}s",
            f"- **Standard Deviation:** {result.performance.std_response_time:.3f}s",
            f"- **Tokens per Second:** {result.performance.tokens_per_second:.1f}",
            f"- **Total Tokens Generated:** {result.performance.total_tokens}",
            f"- **Memory Usage:** {result.performance.memory_usage_mb:.1f}MB",
        ]
        
        if result.performance.gpu_memory_usage_mb is not None:
            report_lines.append(f"- **GPU Memory Usage:** {result.performance.gpu_memory_usage_mb:.1f}MB")
        
        report_lines.append(f"- **CPU Usage:** {result.performance.cpu_usage_percent:.1f}%")
        
        if result.quality:
            report_lines.extend([
                "",
                "## Quality Metrics",
                f"- **Average Sequence Length:** {result.quality.avg_sequence_length:.1f} words",
                f"- **Repetition Ratio:** {result.quality.repetition_ratio:.3f}",
            ])
            
            if result.quality.bleu_score is not None:
                report_lines.append(f"- **BLEU Score:** {result.quality.bleu_score:.3f}")
            
            if result.quality.rouge_l_score is not None:
                report_lines.append(f"- **ROUGE-L Score:** {result.quality.rouge_l_score:.3f}")
        
        # Configuration
        report_lines.extend([
            "",
            "## Configuration",
            f"- **Test Prompts:** {len(result.config.test_prompts or self.default_prompts)}",
            f"- **Runs per Prompt:** {result.config.num_runs}",
            f"- **Max Tokens:** {result.config.max_tokens}",
            f"- **Temperature:** {result.config.temperature}",
            f"- **Batch Size:** {result.config.batch_size}",
        ])
        
        if include_responses:
            report_lines.extend([
                "",
                "## Sample Responses",
            ])
            
            # Show first few responses
            sample_responses = result.responses[:3]
            for i, response in enumerate(sample_responses, 1):
                report_lines.extend([
                    f"### Sample {i}",
                    f"**Prompt:** {response['prompt']}",
                    f"**Response:** {response['response']}",
                    f"**Time:** {response['response_time']:.3f}s",
                    ""
                ])
        
        return "\n".join(report_lines)
    
    def save_report(self, 
                   result: BenchmarkResult,
                   report_format: str = "markdown") -> Path:
        """
        Save benchmark report to file.
        
        Args:
            result: Benchmark result
            report_format: Format for the report (markdown, json)
            
        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if report_format.lower() == "markdown":
            filename = f"{result.benchmark_name}_report_{timestamp}.md"
            filepath = self.reports_dir / filename
            
            report_content = self.generate_report(result, include_responses=True)
            
            with open(filepath, 'w') as f:
                f.write(report_content)
        
        elif report_format.lower() == "json":
            filename = f"{result.benchmark_name}_report_{timestamp}.json"
            filepath = self.reports_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported report format: {report_format}")
        
        console.print(f"[green]📄 Report saved: {filepath}[/green]")
        return filepath
    
    def load_benchmark_results(self, pattern: str = "*") -> List[BenchmarkResult]:
        """
        Load benchmark results from disk.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of loaded benchmark results
        """
        results = []
        
        for filepath in self.benchmarks_dir.glob(f"{pattern}.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Convert back to dataclass (simplified)
                result = BenchmarkResult(**data)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to load benchmark result {filepath}: {e}")
        
        return results