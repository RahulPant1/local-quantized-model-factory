#!/usr/bin/env python3
"""
Memory Optimizer for LQMF - 8GB GPU Memory Management

This module provides comprehensive memory optimization utilities specifically
designed for 8GB GPU environments, ensuring efficient fine-tuning and inference
operations within resource constraints.

Author: LQMF Development Team
Version: 1.0.0
Compatibility: Python 3.8+, PyTorch 2.0+

Features:
- Dynamic batch size optimization
- Memory usage monitoring and alerts
- Automatic gradient checkpointing
- Model sharding for large models
- Memory-efficient data loading
- Garbage collection optimization
- VRAM leak detection and cleanup
"""

import gc
import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import psutil

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_total_mb: float
    cpu_memory_mb: float
    cpu_percent: float
    timestamp: float

class MemoryOptimizer:
    """
    Comprehensive memory optimization for 8GB GPU environments.
    
    Provides automatic memory management, optimization strategies,
    and monitoring for efficient model training and inference.
    """
    
    def __init__(self, 
                 target_gpu_memory_gb: float = 7.5,  # Leave 0.5GB buffer
                 enable_monitoring: bool = True):
        """
        Initialize the Memory Optimizer.
        
        Args:
            target_gpu_memory_gb: Target GPU memory usage in GB
            enable_monitoring: Enable continuous memory monitoring
        """
        self.target_gpu_memory_bytes = target_gpu_memory_gb * 1024**3
        self.enable_monitoring = enable_monitoring
        
        # Memory tracking
        self.memory_history: List[MemoryStats] = []
        self.peak_memory_usage = 0.0
        
        # Optimization flags
        self.gradient_checkpointing_enabled = False
        self.mixed_precision_enabled = False
        
        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_properties = torch.cuda.get_device_properties(0)
            self.total_gpu_memory = self.gpu_properties.total_memory
            console.print(f"[blue]🔧 GPU Memory Optimizer initialized: {self.total_gpu_memory / 1024**3:.1f}GB available[/blue]")
        else:
            console.print("[yellow]⚠️  No GPU detected. CPU-only optimizations enabled.[/yellow]")
        
        # Set memory fraction
        if self.gpu_available:
            torch.cuda.set_per_process_memory_fraction(target_gpu_memory_gb / (self.total_gpu_memory / 1024**3))
    
    def get_current_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        stats = MemoryStats(
            gpu_allocated_mb=0.0,
            gpu_reserved_mb=0.0,
            gpu_total_mb=0.0,
            cpu_memory_mb=psutil.Process().memory_info().rss / 1024**2,
            cpu_percent=psutil.cpu_percent(interval=None),
            timestamp=time.time()
        )
        
        if self.gpu_available:
            stats.gpu_allocated_mb = torch.cuda.memory_allocated() / 1024**2
            stats.gpu_reserved_mb = torch.cuda.memory_reserved() / 1024**2
            stats.gpu_total_mb = self.total_gpu_memory / 1024**2
        
        return stats
    
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage with context."""
        stats = self.get_current_memory_stats()
        self.memory_history.append(stats)
        
        if self.gpu_available:
            usage_percent = (stats.gpu_allocated_mb / stats.gpu_total_mb) * 100
            console.print(f"[dim]💾 {context} - GPU: {stats.gpu_allocated_mb:.1f}MB ({usage_percent:.1f}%), "
                         f"CPU: {stats.cpu_memory_mb:.1f}MB[/dim]")
            
            # Update peak usage
            if stats.gpu_allocated_mb > self.peak_memory_usage:
                self.peak_memory_usage = stats.gpu_allocated_mb
        else:
            console.print(f"[dim]💾 {context} - CPU: {stats.cpu_memory_mb:.1f}MB ({stats.cpu_percent:.1f}%)[/dim]")
    
    @contextmanager
    def memory_monitor(self, operation_name: str):
        """Context manager for monitoring memory during operations."""
        self.log_memory_usage(f"Before {operation_name}")
        
        try:
            yield
        finally:
            self.log_memory_usage(f"After {operation_name}")
            self.cleanup_memory()
    
    def cleanup_memory(self):
        """Perform comprehensive memory cleanup."""
        if self.gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Python garbage collection
        gc.collect()
        
        # Log cleanup results
        if self.enable_monitoring:
            self.log_memory_usage("After cleanup")
    
    def optimize_model_for_training(self, 
                                   model: nn.Module,
                                   enable_gradient_checkpointing: bool = True,
                                   enable_mixed_precision: bool = True) -> nn.Module:
        """
        Optimize model for memory-efficient training.
        
        Args:
            model: Model to optimize
            enable_gradient_checkpointing: Enable gradient checkpointing
            enable_mixed_precision: Enable mixed precision training
            
        Returns:
            Optimized model
        """
        console.print("[blue]🔧 Optimizing model for memory-efficient training...[/blue]")
        
        # Enable gradient checkpointing
        if enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            self.gradient_checkpointing_enabled = True
            console.print("[green]✅ Gradient checkpointing enabled[/green]")
        
        # Convert to half precision for inference
        if enable_mixed_precision and self.gpu_available:
            model = model.half()
            self.mixed_precision_enabled = True
            console.print("[green]✅ Mixed precision enabled[/green]")
        
        # Optimize memory layout
        if self.gpu_available:
            model = model.cuda()
        
        return model
    
    def calculate_optimal_batch_size(self, 
                                    model: nn.Module,
                                    sample_input: torch.Tensor,
                                    max_batch_size: int = 32,
                                    memory_fraction: float = 0.8) -> int:
        """
        Calculate optimal batch size for current memory constraints.
        
        Args:
            model: Model to test
            sample_input: Sample input tensor
            max_batch_size: Maximum batch size to test
            memory_fraction: Fraction of available memory to use
            
        Returns:
            Optimal batch size
        """
        if not self.gpu_available:
            return min(4, max_batch_size)  # Conservative for CPU
        
        console.print("[blue]🔍 Finding optimal batch size...[/blue]")
        
        model.eval()
        optimal_batch_size = 1
        
        for batch_size in range(1, max_batch_size + 1):
            try:
                # Create batch
                batch_input = sample_input.repeat(batch_size, 1)
                
                # Test memory usage
                with torch.no_grad():
                    _ = model(batch_input)
                
                current_memory = torch.cuda.memory_allocated()
                max_memory = self.total_gpu_memory * memory_fraction
                
                if current_memory < max_memory:
                    optimal_batch_size = batch_size
                else:
                    break
                    
                # Cleanup
                del batch_input
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise e
        
        console.print(f"[green]✅ Optimal batch size: {optimal_batch_size}[/green]")
        return optimal_batch_size
    
    def create_memory_efficient_dataloader(self, 
                                          dataset,
                                          batch_size: int = None,
                                          num_workers: int = None,
                                          pin_memory: bool = None) -> DataLoader:
        """
        Create memory-efficient DataLoader.
        
        Args:
            dataset: Dataset to load
            batch_size: Batch size (auto-calculated if None)
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            
        Returns:
            Optimized DataLoader
        """
        # Auto-configure parameters for 8GB GPU
        if batch_size is None:
            batch_size = 2 if self.gpu_available else 1
        
        if num_workers is None:
            # Conservative worker count to avoid memory issues
            num_workers = min(2, psutil.cpu_count() // 2)
        
        if pin_memory is None:
            pin_memory = self.gpu_available
        
        console.print(f"[blue]🔧 Creating DataLoader: batch_size={batch_size}, workers={num_workers}[/blue]")
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
            drop_last=True,  # Avoid memory spikes from uneven batches
            persistent_workers=num_workers > 0
        )
    
    def monitor_training_memory(self, 
                               callback_frequency: int = 10) -> Callable:
        """
        Create a callback function for monitoring memory during training.
        
        Args:
            callback_frequency: How often to log memory (every N steps)
            
        Returns:
            Callback function
        """
        step_counter = [0]  # Use list for closure
        
        def memory_callback():
            step_counter[0] += 1
            
            if step_counter[0] % callback_frequency == 0:
                stats = self.get_current_memory_stats()
                
                if self.gpu_available:
                    usage_percent = (stats.gpu_allocated_mb / stats.gpu_total_mb) * 100
                    
                    # Warning for high memory usage
                    if usage_percent > 90:
                        console.print(f"[red]⚠️  High GPU memory usage: {usage_percent:.1f}%[/red]")
                    elif usage_percent > 80:
                        console.print(f"[yellow]⚠️  GPU memory usage: {usage_percent:.1f}%[/yellow]")
                    
                    # Auto cleanup if needed
                    if usage_percent > 85:
                        self.cleanup_memory()
        
        return memory_callback
    
    def detect_memory_leaks(self, tolerance_mb: float = 100.0) -> bool:
        """
        Detect potential memory leaks by comparing memory usage over time.
        
        Args:
            tolerance_mb: Memory increase tolerance in MB
            
        Returns:
            True if potential leak detected
        """
        if len(self.memory_history) < 10:
            return False
        
        # Compare current vs baseline (10 steps ago)
        current_stats = self.memory_history[-1]
        baseline_stats = self.memory_history[-10]
        
        if self.gpu_available:
            memory_increase = current_stats.gpu_allocated_mb - baseline_stats.gpu_allocated_mb
        else:
            memory_increase = current_stats.cpu_memory_mb - baseline_stats.cpu_memory_mb
        
        if memory_increase > tolerance_mb:
            console.print(f"[red]⚠️  Potential memory leak detected: +{memory_increase:.1f}MB[/red]")
            return True
        
        return False
    
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for memory-efficient inference.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        console.print("[blue]🔧 Optimizing model for inference...[/blue]")
        
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Enable inference optimizations
        if hasattr(model, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                console.print("[green]✅ Torch compile enabled[/green]")
            except:
                console.print("[yellow]⚠️  Torch compile not available[/yellow]")
        
        return model
    
    def suggest_optimizations(self) -> List[str]:
        """
        Suggest optimizations based on current memory usage patterns.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        if not self.memory_history:
            return ["No memory history available for analysis"]
        
        current_stats = self.get_current_memory_stats()
        
        if self.gpu_available:
            usage_percent = (current_stats.gpu_allocated_mb / current_stats.gpu_total_mb) * 100
            
            if usage_percent > 90:
                suggestions.extend([
                    "Reduce batch size",
                    "Enable gradient checkpointing",
                    "Use gradient accumulation instead of larger batches",
                    "Consider using CPU offloading for optimizer states"
                ])
            elif usage_percent > 80:
                suggestions.extend([
                    "Consider reducing sequence length",
                    "Enable mixed precision training",
                    "Use gradient accumulation"
                ])
            
            if not self.gradient_checkpointing_enabled:
                suggestions.append("Enable gradient checkpointing for memory savings")
            
            if not self.mixed_precision_enabled:
                suggestions.append("Enable mixed precision (FP16) training")
        
        # CPU-specific suggestions
        cpu_usage = current_stats.cpu_percent
        if cpu_usage > 90:
            suggestions.extend([
                "Reduce number of DataLoader workers",
                "Consider using smaller models",
                "Enable CPU memory mapping for large datasets"
            ])
        
        # Check for potential memory leaks
        if self.detect_memory_leaks():
            suggestions.append("Investigate potential memory leaks - consider manual garbage collection")
        
        return suggestions or ["Memory usage is optimal"]
    
    def get_memory_report(self) -> str:
        """
        Generate a comprehensive memory usage report.
        
        Returns:
            Formatted memory report
        """
        if not self.memory_history:
            return "No memory data available"
        
        current_stats = self.get_current_memory_stats()
        
        report_lines = [
            "# Memory Usage Report",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Current Status"
        ]
        
        if self.gpu_available:
            usage_percent = (current_stats.gpu_allocated_mb / current_stats.gpu_total_mb) * 100
            report_lines.extend([
                f"- **GPU Memory:** {current_stats.gpu_allocated_mb:.1f}MB / {current_stats.gpu_total_mb:.1f}MB ({usage_percent:.1f}%)",
                f"- **GPU Reserved:** {current_stats.gpu_reserved_mb:.1f}MB",
                f"- **Peak Usage:** {self.peak_memory_usage:.1f}MB"
            ])
        
        report_lines.extend([
            f"- **CPU Memory:** {current_stats.cpu_memory_mb:.1f}MB",
            f"- **CPU Usage:** {current_stats.cpu_percent:.1f}%",
            "",
            "## Optimizations Enabled"
        ])
        
        if self.gradient_checkpointing_enabled:
            report_lines.append("- ✅ Gradient Checkpointing")
        else:
            report_lines.append("- ❌ Gradient Checkpointing")
        
        if self.mixed_precision_enabled:
            report_lines.append("- ✅ Mixed Precision")
        else:
            report_lines.append("- ❌ Mixed Precision")
        
        # Suggestions
        suggestions = self.suggest_optimizations()
        if suggestions:
            report_lines.extend([
                "",
                "## Optimization Suggestions"
            ])
            for suggestion in suggestions:
                report_lines.append(f"- {suggestion}")
        
        return "\n".join(report_lines)
    
    def clear_history(self):
        """Clear memory usage history."""
        self.memory_history.clear()
        self.peak_memory_usage = 0.0
        console.print("[blue]🧹 Memory history cleared[/blue]")


# Convenience functions for common operations

def optimize_model_for_8gb_gpu(model: nn.Module) -> nn.Module:
    """
    Apply standard optimizations for 8GB GPU environment.
    
    Args:
        model: Model to optimize
        
    Returns:
        Optimized model
    """
    optimizer = MemoryOptimizer(target_gpu_memory_gb=7.5)
    return optimizer.optimize_model_for_training(
        model,
        enable_gradient_checkpointing=True,
        enable_mixed_precision=True
    )

@contextmanager
def memory_efficient_training():
    """Context manager for memory-efficient training session."""
    optimizer = MemoryOptimizer()
    
    optimizer.log_memory_usage("Training session start")
    
    try:
        yield optimizer
    finally:
        optimizer.cleanup_memory()
        optimizer.log_memory_usage("Training session end")
        
        # Show final report
        if optimizer.memory_history:
            console.print("\n[cyan]📊 Memory Usage Summary:[/cyan]")
            suggestions = optimizer.suggest_optimizations()
            for suggestion in suggestions[:3]:  # Show top 3 suggestions
                console.print(f"[dim]💡 {suggestion}[/dim]")

def auto_tune_batch_size(model: nn.Module, 
                        sample_input: torch.Tensor,
                        target_memory_gb: float = 7.0) -> int:
    """
    Automatically find optimal batch size for given model and input.
    
    Args:
        model: Model to test
        sample_input: Sample input tensor
        target_memory_gb: Target memory usage
        
    Returns:
        Optimal batch size
    """
    optimizer = MemoryOptimizer(target_gpu_memory_gb=target_memory_gb)
    return optimizer.calculate_optimal_batch_size(model, sample_input)