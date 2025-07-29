#!/usr/bin/env python3
"""
Adapter Manager for LQMF - Dynamic LoRA/QLoRA Adapter Loading and Management

This module provides hot-swapping capabilities for LoRA adapters, allowing
users to dynamically load and unload adapters into running models without
restarting the inference server.

Author: LQMF Development Team
Version: 1.0.0
Compatibility: Python 3.8+, PyTorch 2.0+

Features:
- Hot-swapping of LoRA adapters
- Multiple adapter management per model
- Adapter versioning and metadata tracking
- Memory-efficient adapter loading/unloading
- Integration with API server for real-time switching
- Adapter performance benchmarking
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel, PeftConfig, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import LQMF components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from agents.finetuning_agent import AdapterInfo, AdapterStatus

console = Console()
logger = logging.getLogger(__name__)

class LoadedAdapter:
    """Information about a currently loaded adapter"""
    
    def __init__(self, 
                 adapter_name: str, 
                 adapter_info: AdapterInfo,
                 model: nn.Module,
                 load_time: float):
        self.adapter_name = adapter_name
        self.adapter_info = adapter_info
        self.model = model
        self.load_time = load_time
        self.usage_count = 0
        self.last_used = datetime.now()
    
    def mark_used(self):
        """Mark adapter as recently used"""
        self.usage_count += 1
        self.last_used = datetime.now()

class AdapterManager:
    """
    Manager for dynamic LoRA adapter loading and hot-swapping.
    
    Provides comprehensive adapter management capabilities including:
    - Dynamic loading/unloading of adapters
    - Memory management for multiple adapters
    - Integration with inference APIs
    - Performance monitoring and optimization
    """
    
    def __init__(self, 
                 adapters_dir: str = "adapters",
                 max_loaded_adapters: int = 3):
        """
        Initialize the Adapter Manager.
        
        Args:
            adapters_dir: Directory containing trained adapters
            max_loaded_adapters: Maximum number of adapters to keep in memory
        """
        self.adapters_dir = Path(adapters_dir)
        self.max_loaded_adapters = max_loaded_adapters
        
        # Loaded adapters registry
        self.loaded_adapters: Dict[str, Dict[str, LoadedAdapter]] = {}
        self.base_models: Dict[str, Tuple[nn.Module, Any]] = {}  # base_model -> (model, tokenizer)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load adapters registry
        self.registry_path = self.adapters_dir / "registry.json"
        self.available_adapters = self._load_available_adapters()
        
        console.print("[green]✅ Adapter Manager initialized[/green]")
    
    def _load_available_adapters(self) -> Dict[str, AdapterInfo]:
        """Load available adapters from registry"""
        if not self.registry_path.exists():
            return {}
        
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                adapters = {}
                for name, info in data.items():
                    adapters[name] = AdapterInfo(**info)
                return adapters
        except Exception as e:
            logger.error(f"Failed to load adapters registry: {e}")
            return {}
    
    def _load_base_model(self, model_name: str) -> Tuple[nn.Module, Any]:
        """
        Load base model if not already loaded.
        
        Args:
            model_name: Name of the base model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_name in self.base_models:
            return self.base_models[model_name]
        
        console.print(f"[blue]📥 Loading base model: {model_name}[/blue]")
        
        try:
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.base_models[model_name] = (model, tokenizer)
            console.print(f"[green]✅ Base model loaded: {model_name}[/green]")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load base model {model_name}: {e}")
            raise
    
    def load_adapter(self, 
                    adapter_name: str, 
                    base_model_name: Optional[str] = None) -> bool:
        """
        Load an adapter into memory for inference.
        
        Args:
            adapter_name: Name of the adapter to load
            base_model_name: Base model name (if different from adapter's base model)
            
        Returns:
            True if adapter loaded successfully, False otherwise
        """
        with self._lock:
            # Check if adapter exists
            if adapter_name not in self.available_adapters:
                console.print(f"[red]❌ Adapter not found: {adapter_name}[/red]")
                return False
            
            adapter_info = self.available_adapters[adapter_name]
            target_base_model = base_model_name or adapter_info.base_model
            
            # Check if adapter is already loaded
            if (target_base_model in self.loaded_adapters and 
                adapter_name in self.loaded_adapters[target_base_model]):
                console.print(f"[yellow]⚠️  Adapter already loaded: {adapter_name}[/yellow]")
                return True
            
            try:
                # Load base model if needed
                base_model, tokenizer = self._load_base_model(target_base_model)
                
                console.print(f"[blue]🔄 Loading adapter: {adapter_name}[/blue]")
                start_time = time.time()
                
                # Load adapter
                adapter_path = Path(adapter_info.adapter_path)
                adapted_model = PeftModel.from_pretrained(
                    base_model,
                    adapter_path,
                    torch_dtype=torch.float16
                )
                
                load_time = time.time() - start_time
                
                # Create loaded adapter entry
                loaded_adapter = LoadedAdapter(
                    adapter_name=adapter_name,
                    adapter_info=adapter_info,
                    model=adapted_model,
                    load_time=load_time
                )
                
                # Add to loaded adapters
                if target_base_model not in self.loaded_adapters:
                    self.loaded_adapters[target_base_model] = {}
                
                self.loaded_adapters[target_base_model][adapter_name] = loaded_adapter
                
                # Update adapter status
                adapter_info.status = AdapterStatus.LOADED
                
                # Check memory limits
                self._manage_memory()
                
                console.print(f"[green]✅ Adapter loaded: {adapter_name} ({load_time:.2f}s)[/green]")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load adapter {adapter_name}: {e}")
                console.print(f"[red]❌ Failed to load adapter: {e}[/red]")
                return False
    
    def unload_adapter(self, 
                      adapter_name: str, 
                      base_model_name: Optional[str] = None) -> bool:
        """
        Unload an adapter from memory.
        
        Args:
            adapter_name: Name of the adapter to unload
            base_model_name: Base model name (if specified)
            
        Returns:
            True if adapter unloaded successfully, False otherwise
        """
        with self._lock:
            target_base_model = base_model_name
            
            # Find the adapter if base model not specified
            if target_base_model is None:
                for model_name, adapters in self.loaded_adapters.items():
                    if adapter_name in adapters:
                        target_base_model = model_name
                        break
            
            if (target_base_model is None or 
                target_base_model not in self.loaded_adapters or
                adapter_name not in self.loaded_adapters[target_base_model]):
                console.print(f"[red]❌ Adapter not loaded: {adapter_name}[/red]")
                return False
            
            try:
                # Remove from loaded adapters
                loaded_adapter = self.loaded_adapters[target_base_model][adapter_name]
                del self.loaded_adapters[target_base_model][adapter_name]
                
                # Update status
                if adapter_name in self.available_adapters:
                    self.available_adapters[adapter_name].status = AdapterStatus.UNLOADED
                
                # Clean up empty model entries
                if not self.loaded_adapters[target_base_model]:
                    del self.loaded_adapters[target_base_model]
                
                # Force garbage collection
                del loaded_adapter
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                console.print(f"[green]✅ Adapter unloaded: {adapter_name}[/green]")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload adapter {adapter_name}: {e}")
                console.print(f"[red]❌ Failed to unload adapter: {e}[/red]")
                return False
    
    def switch_adapter(self, 
                      old_adapter: str, 
                      new_adapter: str,
                      base_model_name: Optional[str] = None) -> bool:
        """
        Hot-swap one adapter for another.
        
        Args:
            old_adapter: Name of adapter to unload
            new_adapter: Name of adapter to load
            base_model_name: Base model name
            
        Returns:
            True if switch successful, False otherwise
        """
        console.print(f"[blue]🔄 Switching adapter: {old_adapter} → {new_adapter}[/blue]")
        
        # Load new adapter first
        if not self.load_adapter(new_adapter, base_model_name):
            return False
        
        # Unload old adapter
        if not self.unload_adapter(old_adapter, base_model_name):
            console.print("[yellow]⚠️  Failed to unload old adapter, but new adapter is loaded[/yellow]")
        
        console.print(f"[green]✅ Adapter switched successfully[/green]")
        return True
    
    def get_loaded_adapters(self, base_model_name: Optional[str] = None) -> Dict[str, LoadedAdapter]:
        """
        Get information about currently loaded adapters.
        
        Args:
            base_model_name: Filter by base model (optional)
            
        Returns:
            Dictionary of loaded adapters
        """
        if base_model_name:
            return self.loaded_adapters.get(base_model_name, {})
        
        # Return all loaded adapters
        all_adapters = {}
        for model_adapters in self.loaded_adapters.values():
            all_adapters.update(model_adapters)
        return all_adapters
    
    def get_adapter_model(self, 
                         adapter_name: str, 
                         base_model_name: Optional[str] = None) -> Optional[nn.Module]:
        """
        Get the model instance for a loaded adapter.
        
        Args:
            adapter_name: Name of the adapter
            base_model_name: Base model name (optional)
            
        Returns:
            Model instance if found, None otherwise
        """
        loaded_adapters = self.get_loaded_adapters(base_model_name)
        
        if adapter_name in loaded_adapters:
            loaded_adapter = loaded_adapters[adapter_name]
            loaded_adapter.mark_used()
            return loaded_adapter.model
        
        return None
    
    def _manage_memory(self) -> None:
        """Manage memory by unloading least recently used adapters if needed."""
        total_loaded = sum(len(adapters) for adapters in self.loaded_adapters.values())
        
        if total_loaded <= self.max_loaded_adapters:
            return
        
        # Collect all loaded adapters with usage info
        all_adapters = []
        for model_name, adapters in self.loaded_adapters.items():
            for adapter_name, loaded_adapter in adapters.items():
                all_adapters.append((model_name, adapter_name, loaded_adapter))
        
        # Sort by last used time (oldest first)
        all_adapters.sort(key=lambda x: x[2].last_used)
        
        # Unload oldest adapters until we're under the limit
        adapters_to_unload = total_loaded - self.max_loaded_adapters
        for i in range(adapters_to_unload):
            model_name, adapter_name, _ = all_adapters[i]
            console.print(f"[yellow]🧹 Auto-unloading LRU adapter: {adapter_name}[/yellow]")
            self.unload_adapter(adapter_name, model_name)
    
    def benchmark_adapter(self, 
                         adapter_name: str,
                         test_prompts: List[str],
                         base_model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Benchmark an adapter's performance.
        
        Args:
            adapter_name: Name of the adapter to benchmark
            test_prompts: List of test prompts
            base_model_name: Base model name (optional)
            
        Returns:
            Benchmark results
        """
        console.print(f"[blue]📊 Benchmarking adapter: {adapter_name}[/blue]")
        
        # Ensure adapter is loaded
        if not self.load_adapter(adapter_name, base_model_name):
            return {"error": "Failed to load adapter"}
        
        # Get model and tokenizer
        model = self.get_adapter_model(adapter_name, base_model_name)
        if model is None:
            return {"error": "Adapter model not found"}
        
        # Find base model for tokenizer
        target_base_model = base_model_name
        if target_base_model is None:
            adapter_info = self.available_adapters.get(adapter_name)
            if adapter_info:
                target_base_model = adapter_info.base_model
        
        if target_base_model not in self.base_models:
            return {"error": "Base model not loaded"}
        
        _, tokenizer = self.base_models[target_base_model]
        
        results = {
            "adapter_name": adapter_name,
            "base_model": target_base_model,
            "test_count": len(test_prompts),
            "responses": [],
            "avg_response_time": 0,
            "total_tokens": 0
        }
        
        total_time = 0
        total_tokens = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"Benchmarking {adapter_name}...", total=len(test_prompts))
            
            for i, prompt in enumerate(test_prompts):
                try:
                    start_time = time.time()
                    
                    # Tokenize input
                    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=100,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # Decode response
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response[len(prompt):].strip()
                    
                    response_time = time.time() - start_time
                    tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
                    
                    results["responses"].append({
                        "prompt": prompt,
                        "response": response,
                        "response_time": response_time,
                        "tokens": tokens_generated
                    })
                    
                    total_time += response_time
                    total_tokens += tokens_generated
                    
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    logger.error(f"Benchmark error for prompt {i}: {e}")
                    results["responses"].append({
                        "prompt": prompt,
                        "error": str(e)
                    })
        
        results["avg_response_time"] = total_time / len(test_prompts) if test_prompts else 0
        results["total_tokens"] = total_tokens
        results["tokens_per_second"] = total_tokens / total_time if total_time > 0 else 0
        
        console.print(f"[green]✅ Benchmark completed: {results['avg_response_time']:.2f}s avg, "
                     f"{results['tokens_per_second']:.1f} tokens/s[/green]")
        
        return results
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        memory_info = {
            "loaded_adapters_count": sum(len(adapters) for adapters in self.loaded_adapters.values()),
            "loaded_base_models": len(self.base_models),
            "max_adapters": self.max_loaded_adapters
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
        
        return memory_info
    
    def cleanup(self) -> None:
        """Clean up all loaded models and adapters."""
        console.print("[blue]🧹 Cleaning up adapter manager...[/blue]")
        
        with self._lock:
            # Unload all adapters
            for model_name in list(self.loaded_adapters.keys()):
                for adapter_name in list(self.loaded_adapters[model_name].keys()):
                    self.unload_adapter(adapter_name, model_name)
            
            # Clear base models
            self.base_models.clear()
            
            # Force garbage collection
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        console.print("[green]✅ Cleanup completed[/green]")