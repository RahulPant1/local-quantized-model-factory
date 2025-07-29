"""
FeedbackAgent - Optional benchmarking and performance analysis
Provides model performance testing and quality assessment
"""

import os
import time
import json
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

console = Console()

@dataclass
class PerformanceMetrics:
    tokens_per_second: float
    memory_usage_mb: float
    peak_memory_mb: float
    inference_time_ms: float
    model_size_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None

@dataclass
class QualityMetrics:
    perplexity: Optional[float] = None
    response_quality_score: Optional[float] = None
    coherence_score: Optional[float] = None
    factual_accuracy: Optional[float] = None
    response_length_avg: Optional[float] = None

@dataclass
class BenchmarkResult:
    model_path: str
    quantization_config: Dict[str, Any]
    performance_metrics: PerformanceMetrics
    quality_metrics: QualityMetrics
    test_prompts: List[str]
    generated_responses: List[str]
    benchmark_timestamp: str
    benchmark_duration_seconds: float
    success: bool
    error_message: Optional[str] = None

class SystemMonitor:
    """Monitor system resources during inference"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_readings = []
        self.memory_readings = []
        self.gpu_readings = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.cpu_readings = []
        self.memory_readings = []
        self.gpu_readings = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return average metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        metrics = {
            "avg_cpu_percent": sum(self.cpu_readings) / len(self.cpu_readings) if self.cpu_readings else 0,
            "avg_memory_mb": sum(self.memory_readings) / len(self.memory_readings) if self.memory_readings else 0,
            "peak_memory_mb": max(self.memory_readings) if self.memory_readings else 0,
        }
        
        if self.gpu_readings and torch.cuda.is_available():
            gpu_usage = [r["usage"] for r in self.gpu_readings]
            gpu_memory = [r["memory"] for r in self.gpu_readings]
            metrics["avg_gpu_percent"] = sum(gpu_usage) / len(gpu_usage)
            metrics["avg_gpu_memory_mb"] = sum(gpu_memory) / len(gpu_memory)
            metrics["peak_gpu_memory_mb"] = max(gpu_memory)
        
        return metrics
    
    def _monitor_loop(self):
        """Monitoring loop running in separate thread"""
        while self.monitoring:
            # CPU and RAM monitoring
            self.cpu_readings.append(psutil.cpu_percent(interval=0.1))
            self.memory_readings.append(psutil.virtual_memory().used / (1024 * 1024))
            
            # GPU monitoring if available
            if torch.cuda.is_available():
                try:
                    gpu_usage = torch.cuda.utilization()
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    self.gpu_readings.append({"usage": gpu_usage, "memory": gpu_memory})
                except:
                    pass
            
            time.sleep(0.5)

class FeedbackAgent:
    """Agent for benchmarking and performance analysis of quantized models"""
    
    def __init__(self, benchmark_dir: str = "benchmarks", quantized_models_dir: str = "quantized-models"):
        self.benchmark_dir = Path(benchmark_dir)
        self.quantized_models_dir = Path(quantized_models_dir)
        self.benchmark_dir.mkdir(exist_ok=True)
        self.quantized_models_dir.mkdir(exist_ok=True)
        
        # Default test prompts for evaluation
        self.default_test_prompts = [
            "Explain the concept of machine learning in simple terms.",
            "Write a short story about a robot learning to paint.",
            "What are the benefits and drawbacks of renewable energy?",
            "Describe the process of photosynthesis.",
            "How do neural networks work?",
            "What is the capital of France and what is it famous for?",
            "Explain quantum computing to a 10-year-old.",
            "Write a recipe for chocolate chip cookies.",
            "What causes the seasons to change?",
            "Describe the importance of biodiversity."
        ]
    
    def benchmark_model(self, model_path: str, quantization_config: Dict[str, Any], 
                       test_prompts: Optional[List[str]] = None, 
                       max_new_tokens: int = 100) -> BenchmarkResult:
        """Run comprehensive benchmark on a quantized model"""
        console.print(Panel.fit("🧪 Starting Model Benchmark", style="bold blue"))
        
        start_time = time.time()
        prompts = test_prompts or self.default_test_prompts
        
        try:
            # Load model and tokenizer
            console.print("[blue]Loading model and tokenizer...[/blue]")
            model, tokenizer = self._load_model(model_path, quantization_config)
            
            if not model or not tokenizer:
                return BenchmarkResult(
                    model_path=model_path,
                    quantization_config=quantization_config,
                    performance_metrics=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                    quality_metrics=QualityMetrics(),
                    test_prompts=prompts,
                    generated_responses=[],
                    benchmark_timestamp=datetime.now().isoformat(),
                    benchmark_duration_seconds=0,
                    success=False,
                    error_message="Failed to load model"
                )
            
            # Get model size
            model_size_mb = self._get_model_size(model_path)
            
            # Run performance benchmarks
            console.print("[blue]Running performance benchmarks...[/blue]")
            performance_metrics = self._benchmark_performance(model, tokenizer, prompts, max_new_tokens)
            performance_metrics.model_size_mb = model_size_mb
            
            # Generate responses for quality evaluation
            console.print("[blue]Generating test responses...[/blue]")
            responses = self._generate_responses(model, tokenizer, prompts, max_new_tokens)
            
            # Evaluate response quality
            console.print("[blue]Evaluating response quality...[/blue]")
            quality_metrics = self._evaluate_quality(prompts, responses)
            
            duration = time.time() - start_time
            
            result = BenchmarkResult(
                model_path=model_path,
                quantization_config=quantization_config,
                performance_metrics=performance_metrics,
                quality_metrics=quality_metrics,
                test_prompts=prompts,
                generated_responses=responses,
                benchmark_timestamp=datetime.now().isoformat(),
                benchmark_duration_seconds=duration,
                success=True
            )
            
            # Save benchmark results
            self._save_benchmark_result(result)
            
            # Display results
            self._display_benchmark_results(result)
            
            return result
            
        except Exception as e:
            console.print(f"[red]Benchmark failed: {e}[/red]")
            return BenchmarkResult(
                model_path=model_path,
                quantization_config=quantization_config,
                performance_metrics=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                quality_metrics=QualityMetrics(),
                test_prompts=prompts,
                generated_responses=[],
                benchmark_timestamp=datetime.now().isoformat(),
                benchmark_duration_seconds=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _load_model(self, model_path: str, config: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
        """Load model and tokenizer based on quantization configuration"""
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model based on quantization type
            if config.get("quantization_type") == "bitsandbytes":
                from transformers import BitsAndBytesConfig
                
                bit_width = config.get("bit_width", 4)
                if bit_width == 4:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                elif bit_width == 8:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            elif config.get("quantization_type") == "gguf":
                # For GGUF models, we'd typically use llama-cpp-python
                try:
                    from llama_cpp import Llama
                    model = Llama(model_path=model_path, n_ctx=2048, verbose=False)
                    return model, tokenizer
                except ImportError:
                    console.print("[yellow]llama-cpp-python not available for GGUF inference[/yellow]")
                    return None, None
            
            else:
                # Standard loading for GPTQ or other formats
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            return model, tokenizer
            
        except Exception as e:
            console.print(f"[red]Failed to load model: {e}[/red]")
            return None, None
    
    def _benchmark_performance(self, model: Any, tokenizer: Any, 
                             prompts: List[str], max_new_tokens: int) -> PerformanceMetrics:
        """Benchmark model performance metrics"""
        monitor = SystemMonitor()
        
        total_tokens = 0
        total_time = 0
        
        monitor.start_monitoring()
        
        for prompt in prompts[:3]:  # Use subset for performance testing
            # Encode prompt
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response and measure time
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generation_time = time.time() - start_time
            
            # Count tokens generated
            input_length = inputs["input_ids"].shape[1]
            output_length = outputs.shape[1]
            tokens_generated = output_length - input_length
            
            total_tokens += tokens_generated
            total_time += generation_time
        
        system_metrics = monitor.stop_monitoring()
        
        # Calculate metrics
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        avg_inference_time = (total_time / len(prompts[:3])) * 1000  # Convert to ms
        
        return PerformanceMetrics(
            tokens_per_second=tokens_per_second,
            memory_usage_mb=system_metrics["avg_memory_mb"],
            peak_memory_mb=system_metrics["peak_memory_mb"],
            inference_time_ms=avg_inference_time,
            model_size_mb=0,  # Will be set by caller
            cpu_usage_percent=system_metrics["avg_cpu_percent"],
            gpu_usage_percent=system_metrics.get("avg_gpu_percent"),
            gpu_memory_mb=system_metrics.get("avg_gpu_memory_mb")
        )
    
    def _generate_responses(self, model: Any, tokenizer: Any, 
                          prompts: List[str], max_new_tokens: int) -> List[str]:
        """Generate responses for all test prompts"""
        responses = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating responses...", total=len(prompts))
            
            for prompt in prompts:
                try:
                    # Check if this is a GGUF model (llama-cpp-python)
                    if hasattr(model, '__call__'):  # GGUF model
                        response = model(prompt, max_tokens=max_new_tokens, stop=["Human:", "\n\n"])
                        response_text = response['choices'][0]['text'].strip()
                    else:
                        # Standard transformers model
                        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                        if torch.cuda.is_available():
                            inputs = {k: v.cuda() for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=0.7,
                                pad_token_id=tokenizer.eos_token_id
                            )
                        
                        # Decode only the generated portion
                        input_length = inputs["input_ids"].shape[1]
                        generated_tokens = outputs[0][input_length:]
                        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    
                    responses.append(response_text)
                    
                except Exception as e:
                    console.print(f"[yellow]Failed to generate response for prompt: {e}[/yellow]")
                    responses.append(f"[Error: {str(e)}]")
                
                progress.advance(task)
        
        return responses
    
    def _evaluate_quality(self, prompts: List[str], responses: List[str]) -> QualityMetrics:
        """Evaluate response quality using simple heuristics"""
        if not responses:
            return QualityMetrics()
        
        # Calculate basic metrics
        response_lengths = [len(r.split()) for r in responses if not r.startswith("[Error:")]
        avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        
        # Simple quality heuristics
        quality_scores = []
        for prompt, response in zip(prompts, responses):
            if response.startswith("[Error:"):
                quality_scores.append(0.0)
                continue
            
            score = 0.0
            
            # Length appropriateness (not too short, not too long)
            response_length = len(response.split())
            if 10 <= response_length <= 200:
                score += 0.3
            elif response_length > 5:
                score += 0.1
            
            # Basic coherence (contains some words from prompt context)
            prompt_words = set(prompt.lower().split())
            response_words = set(response.lower().split())
            
            # Check for relevant content
            if len(response) > 20:  # Minimum length
                score += 0.2
            
            # Check for complete sentences
            if response.endswith(('.', '!', '?')):
                score += 0.1
            
            # Check for reasonable structure
            sentences = response.split('.')
            if len(sentences) > 1:
                score += 0.2
            
            # Penalize very repetitive responses
            unique_words = len(response_words)
            total_words = len(response.split())
            if total_words > 0 and (unique_words / total_words) > 0.3:
                score += 0.2
            
            quality_scores.append(min(score, 1.0))
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return QualityMetrics(
            response_quality_score=avg_quality,
            response_length_avg=avg_length,
            coherence_score=avg_quality * 0.8,  # Simplified coherence estimate
        )
    
    def _get_model_size(self, model_path: str) -> float:
        """Get model size in MB"""
        total_size = 0
        model_path = Path(model_path)
        
        if model_path.is_file():
            return model_path.stat().st_size / (1024 * 1024)
        
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)
    
    def _save_benchmark_result(self, result: BenchmarkResult):
        """Save benchmark result to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(result.model_path).name
        filename = f"benchmark_{model_name}_{timestamp}.json"
        file_path = self.benchmark_dir / filename
        
        # Convert dataclass to dict for JSON serialization
        result_dict = asdict(result)
        
        with open(file_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        console.print(f"[green]Benchmark results saved to {file_path}[/green]")
    
    def _display_benchmark_results(self, result: BenchmarkResult):
        """Display benchmark results in a formatted table"""
        if not result.success:
            console.print(Panel(f"❌ Benchmark Failed\n\n{result.error_message}", style="red"))
            return
        
        # Performance metrics table
        perf_table = Table(title="Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        pm = result.performance_metrics
        perf_table.add_row("Tokens/Second", f"{pm.tokens_per_second:.2f}")
        perf_table.add_row("Avg Inference Time", f"{pm.inference_time_ms:.2f} ms")
        perf_table.add_row("Model Size", f"{pm.model_size_mb:.1f} MB")
        perf_table.add_row("Peak Memory", f"{pm.peak_memory_mb:.1f} MB")
        perf_table.add_row("CPU Usage", f"{pm.cpu_usage_percent:.1f}%")
        
        if pm.gpu_usage_percent is not None:
            perf_table.add_row("GPU Usage", f"{pm.gpu_usage_percent:.1f}%")
            perf_table.add_row("GPU Memory", f"{pm.gpu_memory_mb:.1f} MB")
        
        console.print(perf_table)
        
        # Quality metrics table
        quality_table = Table(title="Quality Metrics")
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Score", style="yellow")
        
        qm = result.quality_metrics
        if qm.response_quality_score is not None:
            quality_table.add_row("Overall Quality", f"{qm.response_quality_score:.2f}/1.0")
        if qm.coherence_score is not None:
            quality_table.add_row("Coherence", f"{qm.coherence_score:.2f}/1.0")
        if qm.response_length_avg is not None:
            quality_table.add_row("Avg Response Length", f"{qm.response_length_avg:.1f} words")
        
        console.print(quality_table)
        
        # Summary
        console.print(Panel.fit(
            f"✅ Benchmark completed in {result.benchmark_duration_seconds:.2f} seconds",
            style="bold green"
        ))
    
    def compare_models(self, benchmark_files: List[str]) -> None:
        """Compare multiple benchmark results"""
        console.print(Panel.fit("📊 Model Comparison", style="bold blue"))
        
        results = []
        for file_path in benchmark_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                console.print(f"[red]Failed to load {file_path}: {e}[/red]")
        
        if len(results) < 2:
            console.print("[yellow]Need at least 2 benchmark results to compare[/yellow]")
            return
        
        # Create comparison table
        table = Table(title="Model Performance Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("Tokens/Sec", style="green")
        table.add_column("Size (MB)", style="yellow")
        table.add_column("Quality", style="magenta")
        table.add_column("Memory (MB)", style="blue")
        
        for result in results:
            model_name = Path(result["model_path"]).name
            perf = result["performance_metrics"]
            quality = result["quality_metrics"]
            
            table.add_row(
                model_name,
                f"{perf['tokens_per_second']:.2f}",
                f"{perf['model_size_mb']:.1f}",
                f"{quality.get('response_quality_score', 0):.2f}",
                f"{perf['peak_memory_mb']:.1f}"
            )
        
        console.print(table)
    
    def get_benchmark_history(self, model_path: Optional[str] = None) -> List[Dict]:
        """Get benchmark history for a specific model or all models"""
        history = []
        
        for file_path in self.benchmark_dir.glob("benchmark_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    if model_path is None or data["model_path"] == model_path:
                        history.append(data)
            except Exception as e:
                console.print(f"[yellow]Failed to read {file_path}: {e}[/yellow]")
        
        # Sort by timestamp
        history.sort(key=lambda x: x["benchmark_timestamp"], reverse=True)
        return history
    
    def discover_quantized_models(self) -> List[Dict[str, Any]]:
        """Discover all quantized models in the quantized-models directory"""
        models = []
        
        for model_dir in self.quantized_models_dir.iterdir():
            if model_dir.is_dir():
                config_path = model_dir / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        
                        # Calculate model size
                        model_files = list(model_dir.glob("*.safetensors"))
                        if not model_files:
                            model_files = list(model_dir.glob("*.bin"))
                        
                        total_size = sum(f.stat().st_size for f in model_files)
                        size_mb = total_size / (1024 * 1024)
                        
                        # Extract quantization info from directory name
                        dir_name = model_dir.name
                        quant_info = self._parse_quantization_info(dir_name)
                        
                        model_info = {
                            "name": model_dir.name,
                            "path": str(model_dir),
                            "original_model": quant_info.get("base_model", "unknown"),
                            "quantization_method": quant_info.get("method", "unknown"),
                            "bit_width": quant_info.get("bits", "unknown"),
                            "model_type": config.get("model_type", "unknown"),
                            "architecture": config.get("architectures", ["unknown"])[0] if config.get("architectures") else "unknown",
                            "vocab_size": config.get("vocab_size", 0),
                            "hidden_size": config.get("n_embd", config.get("hidden_size", 0)),
                            "num_layers": config.get("n_layer", config.get("num_hidden_layers", 0)),
                            "context_length": config.get("n_ctx", config.get("max_position_embeddings", 0)),
                            "size_mb": size_mb,
                            "files_count": len(list(model_dir.glob("*")))
                        }
                        models.append(model_info)
                        
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not read model info for {model_dir.name}: {e}[/yellow]")
        
        return sorted(models, key=lambda x: x["name"])
    
    def _parse_quantization_info(self, dir_name: str) -> Dict[str, str]:
        """Parse quantization information from directory name"""
        # Expected format: originalmodel_method_bits
        # e.g., microsoft_DialoGPT-small_bnb_4bit
        parts = dir_name.split('_')
        
        info = {}
        
        # Try to extract base model name (first parts before method)
        if len(parts) >= 3:
            # Look for method indicators
            method_indicators = ['bnb', 'gptq', 'gguf', 'awq']
            method_idx = None
            
            for i, part in enumerate(parts):
                if any(indicator in part.lower() for indicator in method_indicators):
                    method_idx = i
                    break
            
            if method_idx is not None:
                info["base_model"] = "_".join(parts[:method_idx])
                info["method"] = parts[method_idx]
                
                # Look for bit width in remaining parts
                for part in parts[method_idx:]:
                    if 'bit' in part.lower():
                        info["bits"] = part.replace('bit', '').replace('bits', '')
                        break
        
        return info
    
    def test_quantized_model(self, model_name_or_path: str, comprehensive: bool = True) -> BenchmarkResult:
        """Test a specific quantized model by name or path"""
        console.print(Panel.fit(f"🧪 Testing Quantized Model: {model_name_or_path}", style="bold blue"))
        
        # Determine model path
        model_path = self._resolve_model_path(model_name_or_path)
        if not model_path:
            return BenchmarkResult(
                model_path=model_name_or_path,
                quantization_config={},
                performance_metrics=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                quality_metrics=QualityMetrics(),
                test_prompts=[],
                generated_responses=[],
                benchmark_timestamp=datetime.now().isoformat(),
                benchmark_duration_seconds=0,
                success=False,
                error_message=f"Model not found: {model_name_or_path}"
            )
        
        start_time = time.time()
        
        try:
            # Load model info
            model_info = self._get_model_info(model_path)
            console.print(f"[green]Model found: {model_info['architecture']} ({model_info['size_mb']:.1f}MB)[/green]")
            
            # Load model and tokenizer
            console.print("[blue]Loading model and tokenizer...[/blue]")
            model, tokenizer = self._load_quantized_model(model_path, model_info)
            
            if not model or not tokenizer:
                return BenchmarkResult(
                    model_path=str(model_path),
                    quantization_config=model_info,
                    performance_metrics=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                    quality_metrics=QualityMetrics(),
                    test_prompts=[],
                    generated_responses=[],
                    benchmark_timestamp=datetime.now().isoformat(),
                    benchmark_duration_seconds=time.time() - start_time,
                    success=False,
                    error_message="Failed to load model"
                )
            
            console.print("[green]✅ Model loaded successfully![/green]")
            
            # Run tests based on comprehensiveness
            if comprehensive:
                test_prompts = self._get_comprehensive_test_prompts()
            else:
                test_prompts = self.default_test_prompts[:5]  # Quick test
            
            # Run performance benchmark
            performance_metrics = self._benchmark_loaded_model(model, tokenizer, test_prompts)
            performance_metrics.model_size_mb = model_info['size_mb']
            
            # Generate responses for quality evaluation
            responses = self._generate_test_responses(model, tokenizer, test_prompts)
            
            # Evaluate quality
            quality_metrics = self._evaluate_response_quality(test_prompts, responses)
            
            duration = time.time() - start_time
            
            # Create benchmark result
            result = BenchmarkResult(
                model_path=str(model_path),
                quantization_config=model_info,
                performance_metrics=performance_metrics,
                quality_metrics=quality_metrics,
                test_prompts=test_prompts,
                generated_responses=responses,
                benchmark_timestamp=datetime.now().isoformat(),
                benchmark_duration_seconds=duration,
                success=True
            )
            
            # Display results
            self._display_test_results(result)
            
            # Save benchmark
            self._save_benchmark_result(result)
            
            return result
            
        except Exception as e:
            console.print(f"[red]Test failed: {e}[/red]")
            return BenchmarkResult(
                model_path=str(model_path) if model_path else model_name_or_path,
                quantization_config={},
                performance_metrics=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                quality_metrics=QualityMetrics(),
                test_prompts=[],
                generated_responses=[],
                benchmark_timestamp=datetime.now().isoformat(),
                benchmark_duration_seconds=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _resolve_model_path(self, model_name_or_path: str) -> Optional[Path]:
        """Resolve model name to full path"""
        # If it's already a path and exists, use it
        if Path(model_name_or_path).exists():
            return Path(model_name_or_path)
        
        # Look in quantized-models directory
        candidate_path = self.quantized_models_dir / model_name_or_path
        if candidate_path.exists():
            return candidate_path
        
        # Try partial matching
        for model_dir in self.quantized_models_dir.iterdir():
            if model_dir.is_dir() and model_name_or_path.lower() in model_dir.name.lower():
                return model_dir
        
        return None
    
    def _get_model_info(self, model_path: Path) -> Dict[str, Any]:
        """Get model information from config"""
        config_path = model_path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Calculate size
        model_files = list(model_path.glob("*.safetensors"))
        if not model_files:
            model_files = list(model_path.glob("*.bin"))
        
        total_size = sum(f.stat().st_size for f in model_files)
        size_mb = total_size / (1024 * 1024)
        
        return {
            "architecture": config.get("architectures", ["unknown"])[0] if config.get("architectures") else "unknown",
            "model_type": config.get("model_type", "unknown"),
            "vocab_size": config.get("vocab_size", 0),
            "hidden_size": config.get("n_embd", config.get("hidden_size", 0)),
            "num_layers": config.get("n_layer", config.get("num_hidden_layers", 0)),
            "size_mb": size_mb
        }
    
    def _load_quantized_model(self, model_path: Path, model_info: Dict[str, Any]):
        """Load a quantized model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Determine quantization config based on directory name
            dir_name = model_path.name
            if 'bnb' in dir_name or 'bitsandbytes' in dir_name:
                if '4bit' in dir_name:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                elif '8bit' in dir_name:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            else:
                # Try loading without quantization config
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            return model, tokenizer
            
        except Exception as e:
            console.print(f"[red]Failed to load model: {e}[/red]")
            return None, None
    
    def _get_comprehensive_test_prompts(self) -> List[str]:
        """Get comprehensive test prompts"""
        return [
            # Conversational
            "Hello, how are you today?",
            "What's your favorite color?",
            "Tell me about yourself.",
            # Knowledge
            "What is the capital of France?",
            "Explain machine learning in simple terms.",
            "What is photosynthesis?",
            # Creative
            "Tell me a joke.",
            "Write a short poem about cats.",
            "Create a story about a robot.",
            # Reasoning
            "If I have 5 apples and give away 2, how many do I have left?",
            "What comes after Monday?",
            # Open-ended
            "What would you do if you could fly?",
            "Describe a perfect day.",
            "What is the meaning of life?",
        ]
    
    def _benchmark_loaded_model(self, model, tokenizer, test_prompts: List[str]) -> PerformanceMetrics:
        """Benchmark a loaded model"""
        monitor = SystemMonitor()
        
        total_tokens = 0
        total_time = 0
        
        monitor.start_monitoring()
        
        # Use first 3 prompts for performance testing
        for prompt in test_prompts[:3]:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generation_time = time.time() - start_time
            
            input_length = inputs["input_ids"].shape[1]
            output_length = outputs.shape[1]
            tokens_generated = output_length - input_length
            
            total_tokens += tokens_generated
            total_time += generation_time
        
        system_metrics = monitor.stop_monitoring()
        
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        avg_inference_time = (total_time / 3) * 1000  # Convert to ms
        
        return PerformanceMetrics(
            tokens_per_second=tokens_per_second,
            memory_usage_mb=system_metrics["avg_memory_mb"],
            peak_memory_mb=system_metrics["peak_memory_mb"],
            inference_time_ms=avg_inference_time,
            model_size_mb=0,  # Will be set elsewhere
            cpu_usage_percent=system_metrics["avg_cpu_percent"],
            gpu_usage_percent=system_metrics.get("avg_gpu_percent"),
            gpu_memory_mb=system_metrics.get("avg_gpu_memory_mb")
        )
    
    def _generate_test_responses(self, model, tokenizer, test_prompts: List[str]) -> List[str]:
        """Generate responses for test prompts"""
        responses = []
        
        console.print("[cyan]Generating test responses...[/cyan]")
        
        for prompt in test_prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=25,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                responses.append(response)
                
            except Exception as e:
                responses.append(f"[Error: {str(e)}]")
        
        return responses
    
    def _evaluate_response_quality(self, prompts: List[str], responses: List[str]) -> QualityMetrics:
        """Evaluate response quality"""
        return self._evaluate_quality(prompts, responses)  # Use existing method
    
    def _display_test_results(self, result: BenchmarkResult):
        """Display comprehensive test results"""
        if not result.success:
            console.print(Panel(f"❌ Test Failed\n\n{result.error_message}", style="red"))
            return
        
        # Model info table
        info_table = Table(title="Model Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        
        config = result.quantization_config
        info_table.add_row("Model Path", result.model_path)
        info_table.add_row("Architecture", config.get("architecture", "Unknown"))
        info_table.add_row("Model Type", config.get("model_type", "Unknown"))
        info_table.add_row("Model Size", f"{config.get('size_mb', 0):.1f} MB")
        
        console.print(info_table)
        
        # Performance metrics
        perf_table = Table(title="Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="yellow")
        
        pm = result.performance_metrics
        perf_table.add_row("Tokens/Second", f"{pm.tokens_per_second:.2f}")
        perf_table.add_row("Avg Inference Time", f"{pm.inference_time_ms:.2f} ms")
        perf_table.add_row("Peak Memory", f"{pm.peak_memory_mb:.1f} MB")
        perf_table.add_row("CPU Usage", f"{pm.cpu_usage_percent:.1f}%")
        
        if pm.gpu_usage_percent is not None:
            perf_table.add_row("GPU Usage", f"{pm.gpu_usage_percent:.1f}%")
            perf_table.add_row("GPU Memory", f"{pm.gpu_memory_mb:.1f} MB")
        
        console.print(perf_table)
        
        # Quality metrics
        qm = result.quality_metrics
        if qm.response_quality_score is not None:
            quality_table = Table(title="Quality Metrics")
            quality_table.add_column("Metric", style="cyan")
            quality_table.add_column("Score", style="magenta")
            
            quality_table.add_row("Overall Quality", f"{qm.response_quality_score:.2f}/1.0")
            if qm.coherence_score is not None:
                quality_table.add_row("Coherence", f"{qm.coherence_score:.2f}/1.0")
            if qm.response_length_avg is not None:
                quality_table.add_row("Avg Response Length", f"{qm.response_length_avg:.1f} words")
            
            console.print(quality_table)
        
        # Sample responses
        if result.generated_responses:
            console.print("\n[bold cyan]Sample Responses:[/bold cyan]")
            for i, (prompt, response) in enumerate(zip(result.test_prompts[:5], result.generated_responses[:5])):
                console.print(f"[green]Q{i+1}:[/green] {prompt}")
                console.print(f"[yellow]A{i+1}:[/yellow] {response[:100]}{'...' if len(response) > 100 else ''}")
                console.print()
        
        console.print(Panel.fit(
            f"✅ Test completed in {result.benchmark_duration_seconds:.2f} seconds",
            style="bold green"
        ))