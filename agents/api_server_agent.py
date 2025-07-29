#!/usr/bin/env python3
"""
API Server Agent - Serves quantized models as API endpoints
Allows users to load quantized models and experiment with them via REST API
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Import LQMF agents
import sys
sys.path.append(str(Path(__file__).parent.parent))
from agents.memory_agent import MemoryAgent
from agents.adapter_manager import AdapterManager

console = Console()

class ModelStatus(Enum):
    """Status of loaded models"""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"

@dataclass
class LoadedModel:
    """Information about a loaded model"""
    model_name: str
    model_path: str
    quantization_method: str
    bit_width: int
    model: Any
    tokenizer: Any
    status: ModelStatus
    load_time: float
    memory_usage: float
    active_adapter: Optional[str] = None  # Currently loaded adapter
    endpoint_url: Optional[str] = None
    error_message: Optional[str] = None

class ChatRequest(BaseModel):
    """Request model for chat completion"""
    message: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stop: Optional[List[str]] = None

class ChatResponse(BaseModel):
    """Response model for chat completion"""
    response: str
    tokens_generated: int
    time_taken: float
    model_name: str

class APIServerAgent:
    """Agent for serving quantized models as API endpoints"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.quantized_models_dir = self.base_dir / "quantized-models"
        self.memory_agent = MemoryAgent()
        
        # Initialize adapter manager
        self.adapter_manager = AdapterManager(
            adapters_dir=str(self.base_dir / "adapters"),
            max_loaded_adapters=3
        )
        
        # Track loaded models
        self.loaded_models: Dict[str, LoadedModel] = {}
        self.server_thread = None
        self.server_port = 8000
        self.server_host = "127.0.0.1"
        self.app = None
        
        # Initialize FastAPI if available
        if FASTAPI_AVAILABLE:
            self._setup_fastapi()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _setup_fastapi(self):
        """Setup FastAPI application"""
        self.app = FastAPI(
            title="LQMF Model API",
            description="Local Quantized Model Factory API for serving quantized models",
            version="1.0.0"
        )
        
        @self.app.get("/")
        async def root():
            return {"message": "LQMF Model API is running", "loaded_models": list(self.loaded_models.keys())}
        
        @self.app.get("/models")
        async def list_models():
            """List all loaded models"""
            return {
                "loaded_models": {
                    name: {
                        "status": model.status.value,
                        "quantization_method": model.quantization_method,
                        "bit_width": model.bit_width,
                        "memory_usage": model.memory_usage,
                        "load_time": model.load_time,
                        "endpoint_url": model.endpoint_url
                    }
                    for name, model in self.loaded_models.items()
                }
            }
        
        @self.app.get("/models/available")
        async def list_available_models():
            """List all available quantized models"""
            return {"available_models": self.discover_quantized_models()}
        
        @self.app.post("/models/{model_name}/load")
        async def load_model_endpoint(model_name: str, background_tasks: BackgroundTasks):
            """Load a specific model"""
            background_tasks.add_task(self.load_model, model_name)
            return {"message": f"Loading model {model_name} in background"}
        
        @self.app.post("/models/{model_name}/unload")
        async def unload_model_endpoint(model_name: str):
            """Unload a specific model"""
            success = self.unload_model(model_name)
            if success:
                return {"message": f"Model {model_name} unloaded successfully"}
            else:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        @self.app.post("/models/{model_name}/chat")
        async def chat_with_model(model_name: str, request: ChatRequest):
            """Chat with a loaded model"""
            if model_name not in self.loaded_models:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not loaded")
            
            model_info = self.loaded_models[model_name]
            if model_info.status != ModelStatus.READY:
                raise HTTPException(status_code=503, detail=f"Model {model_name} is not ready")
            
            try:
                start_time = time.time()
                response = self._generate_response(model_info, request.message, request.max_tokens, 
                                                 request.temperature, request.top_p, request.stop)
                time_taken = time.time() - start_time
                
                return ChatResponse(
                    response=response,
                    tokens_generated=len(response.split()),  # Simple token count
                    time_taken=time_taken,
                    model_name=model_name
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
    
    def discover_quantized_models(self) -> List[Dict[str, Any]]:
        """Discover all available quantized models"""
        models = []
        
        if not self.quantized_models_dir.exists():
            return models
        
        for model_dir in self.quantized_models_dir.iterdir():
            if model_dir.is_dir():
                model_info = self._parse_model_directory(model_dir)
                if model_info:
                    models.append(model_info)
        
        return models
    
    def _parse_model_directory(self, model_dir: Path) -> Optional[Dict[str, Any]]:
        """Parse model directory to extract information"""
        try:
            # Parse directory name: {model_name}_{method}_{bits}bit
            dir_name = model_dir.name
            parts = dir_name.split('_')
            
            if len(parts) < 3:
                return None
            
            # Extract quantization method and bit width
            method = parts[-2]
            bit_part = parts[-1]
            
            if not bit_part.endswith('bit'):
                return None
            
            bit_width = int(bit_part.replace('bit', ''))
            model_name = '_'.join(parts[:-2])
            
            # Check if model files exist
            config_file = model_dir / "config.json"
            model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.gguf"))
            
            if not config_file.exists() and not model_files:
                return None
            
            # Get model size
            model_size = sum(f.stat().st_size for f in model_files) / (1024 * 1024)  # MB
            
            return {
                "model_name": model_name,
                "display_name": dir_name,
                "quantization_method": method,
                "bit_width": bit_width,
                "model_path": str(model_dir),
                "model_size_mb": round(model_size, 2),
                "has_config": config_file.exists(),
                "model_files": [f.name for f in model_files]
            }
            
        except Exception as e:
            console.print(f"[yellow]Error parsing model directory {model_dir}: {e}[/yellow]")
            return None
    
    def load_model(self, model_name: str) -> bool:
        """Load a quantized model for serving"""
        try:
            # Find the model
            available_models = self.discover_quantized_models()
            model_info = None
            
            for model in available_models:
                if model['display_name'] == model_name or model['model_name'] == model_name:
                    model_info = model
                    break
            
            if not model_info:
                console.print(f"[red]Model {model_name} not found[/red]")
                return False
            
            # Update status to loading
            loaded_model = LoadedModel(
                model_name=model_info['display_name'],
                model_path=model_info['model_path'],
                quantization_method=model_info['quantization_method'],
                bit_width=model_info['bit_width'],
                model=None,
                tokenizer=None,
                status=ModelStatus.LOADING,
                load_time=0,
                memory_usage=0
            )
            
            self.loaded_models[model_info['display_name']] = loaded_model
            
            console.print(f"[blue]Loading model {model_name}...[/blue]")
            start_time = time.time()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_info['model_path'])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model based on quantization method
            if model_info['quantization_method'] == 'bnb':
                model = self._load_bnb_model(model_info['model_path'])
            elif model_info['quantization_method'] == 'gptq':
                model = self._load_gptq_model(model_info['model_path'])
            elif model_info['quantization_method'] == 'gguf':
                model = self._load_gguf_model(model_info['model_path'])
            else:
                # Fallback to standard loading
                model = AutoModelForCausalLM.from_pretrained(model_info['model_path'])
            
            load_time = time.time() - start_time
            
            # Calculate memory usage
            memory_usage = self._calculate_memory_usage(model)
            
            # Update loaded model info
            loaded_model.model = model
            loaded_model.tokenizer = tokenizer
            loaded_model.status = ModelStatus.READY
            loaded_model.load_time = load_time
            loaded_model.memory_usage = memory_usage
            loaded_model.endpoint_url = f"http://{self.server_host}:{self.server_port}/models/{model_name}/chat"
            
            console.print(f"[green]Model {model_name} loaded successfully![/green]")
            console.print(f"[dim]Load time: {load_time:.2f}s, Memory usage: {memory_usage:.2f}MB[/dim]")
            
            # Log to memory agent (with error handling)
            try:
                self.memory_agent.log_experiment(
                    model_name=model_info['model_name'],
                    quantization_method=model_info['quantization_method'],
                    bit_width=model_info['bit_width'],
                    status="api_loaded",
                    metadata={
                        "load_time": load_time,
                        "memory_usage": memory_usage,
                        "endpoint_url": loaded_model.endpoint_url
                    }
                )
            except Exception as logging_error:
                console.print(f"[yellow]Warning: Could not log to memory agent: {logging_error}[/yellow]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]Error loading model {model_name}: {e}[/red]")
            if model_name in self.loaded_models:
                self.loaded_models[model_name].status = ModelStatus.ERROR
                self.loaded_models[model_name].error_message = str(e)
            return False
    
    def _load_bnb_model(self, model_path: str):
        """Load BitsAndBytes quantized model"""
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    def _load_gptq_model(self, model_path: str):
        """Load GPTQ quantized model"""
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    def _load_gguf_model(self, model_path: str):
        """Load GGUF quantized model"""
        # For GGUF, we would use llama-cpp-python if available
        # For now, fallback to standard loading
        try:
            from llama_cpp import Llama
            gguf_files = list(Path(model_path).glob("*.gguf"))
            if gguf_files:
                return Llama(model_path=str(gguf_files[0]))
        except ImportError:
            pass
        
        return AutoModelForCausalLM.from_pretrained(model_path)
    
    def _calculate_memory_usage(self, model) -> float:
        """Calculate approximate memory usage of loaded model"""
        try:
            if hasattr(model, 'get_memory_footprint'):
                return model.get_memory_footprint() / (1024 * 1024)  # MB
            elif hasattr(model, 'num_parameters'):
                # Rough estimate: 4 bytes per parameter for float32, 2 for float16
                return (model.num_parameters() * 2) / (1024 * 1024)
            else:
                return 0.0
        except:
            return 0.0
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        if model_name not in self.loaded_models:
            return False
        
        try:
            loaded_model = self.loaded_models[model_name]
            
            # Clear model from memory
            if loaded_model.model:
                del loaded_model.model
            if loaded_model.tokenizer:
                del loaded_model.tokenizer
            
            # Force garbage collection
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Remove from loaded models
            del self.loaded_models[model_name]
            
            console.print(f"[green]Model {model_name} unloaded successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error unloading model {model_name}: {e}[/red]")
            return False
    
    def _generate_response(self, model_info: LoadedModel, message: str, max_tokens: int = 100, 
                          temperature: float = 0.7, top_p: float = 0.9, stop: List[str] = None) -> str:
        """Generate response from loaded model"""
        try:
            # Tokenize input
            inputs = model_info.tokenizer(message, return_tensors="pt", truncation=True, max_length=512)
            
            # Move to appropriate device
            device = next(model_info.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model_info.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=model_info.tokenizer.eos_token_id,
                    eos_token_id=model_info.tokenizer.eos_token_id
                )
            
            # Decode response
            response = model_info.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input from response
            response = response[len(message):].strip()
            
            return response
            
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def start_server(self, port: int = 8000, host: str = "127.0.0.1"):
        """Start the API server"""
        if not FASTAPI_AVAILABLE:
            console.print("[red]FastAPI not available. Install with: pip install fastapi uvicorn[/red]")
            return False
        
        self.server_port = port
        self.server_host = host
        
        try:
            console.print(f"[blue]Starting API server on {host}:{port}[/blue]")
            uvicorn.run(self.app, host=host, port=port, log_level="info")
            return True
        except Exception as e:
            console.print(f"[red]Error starting server: {e}[/red]")
            return False
    
    def start_server_background(self, port: int = 8000, host: str = "127.0.0.1"):
        """Start the API server in background"""
        if not FASTAPI_AVAILABLE:
            console.print("[red]FastAPI not available. Install with: pip install fastapi uvicorn[/red]")
            return False
        
        self.server_port = port
        self.server_host = host
        
        def run_server():
            uvicorn.run(self.app, host=host, port=port, log_level="error")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Wait a bit for server to start
        time.sleep(2)
        
        console.print(f"[green]API server started on {host}:{port}[/green]")
        console.print(f"[dim]Access the API at: http://{host}:{port}[/dim]")
        return True
    
    def stop_server(self):
        """Stop the API server"""
        if self.server_thread and self.server_thread.is_alive():
            console.print("[yellow]Stopping API server...[/yellow]")
            # Note: This is a simplified stop - in production, you'd want proper shutdown
            return True
        return False
    
    def show_loaded_models(self):
        """Display information about loaded models"""
        if not self.loaded_models:
            console.print("[yellow]No models currently loaded[/yellow]")
            return
        
        table = Table(title="Loaded Models")
        table.add_column("Model Name", style="cyan")
        table.add_column("Method", style="green")
        table.add_column("Bits", style="yellow")
        table.add_column("Status", style="blue")
        table.add_column("Memory (MB)", style="magenta")
        table.add_column("Load Time (s)", style="red")
        table.add_column("Endpoint URL", style="dim")
        
        for name, model in self.loaded_models.items():
            table.add_row(
                name,
                model.quantization_method,
                str(model.bit_width),
                model.status.value,
                f"{model.memory_usage:.1f}",
                f"{model.load_time:.2f}",
                model.endpoint_url or "N/A"
            )
        
        console.print(table)
    
    def show_available_models(self):
        """Display available quantized models"""
        models = self.discover_quantized_models()
        
        if not models:
            console.print("[yellow]No quantized models found[/yellow]")
            return
        
        table = Table(title="Available Quantized Models")
        table.add_column("Model Name", style="cyan")
        table.add_column("Method", style="green")
        table.add_column("Bits", style="yellow")
        table.add_column("Size (MB)", style="magenta")
        table.add_column("Status", style="blue")
        
        for model in models:
            status = "Loaded" if model['display_name'] in self.loaded_models else "Available"
            table.add_row(
                model['display_name'],
                model['quantization_method'],
                str(model['bit_width']),
                str(model['model_size_mb']),
                status
            )
        
        console.print(table)
    
    def interactive_model_selection(self) -> Optional[str]:
        """Interactive model selection interface"""
        models = self.discover_quantized_models()
        
        if not models:
            console.print("[yellow]No quantized models found[/yellow]")
            return None
        
        self.show_available_models()
        
        while True:
            try:
                choice = input("\nEnter model name to load (or 'quit' to exit): ").strip()
                
                if choice.lower() == 'quit':
                    return None
                
                # Find matching model
                for model in models:
                    if model['display_name'] == choice or model['model_name'] == choice:
                        return model['display_name']
                
                console.print(f"[red]Model '{choice}' not found. Please try again.[/red]")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Selection cancelled[/yellow]")
                return None
    
    # Adapter Management Methods
    
    def load_adapter(self, adapter_name: str, model_name: Optional[str] = None) -> bool:
        """
        Load an adapter for a model.
        
        Args:
            adapter_name: Name of the adapter to load
            model_name: Name of the model (optional, uses first loaded model if not specified)
            
        Returns:
            True if adapter loaded successfully, False otherwise
        """
        # Find target model
        if model_name is None:
            if not self.loaded_models:
                console.print("[red]❌ No models loaded. Load a model first.[/red]")
                return False
            model_name = next(iter(self.loaded_models.keys()))
        
        if model_name not in self.loaded_models:
            console.print(f"[red]❌ Model not loaded: {model_name}[/red]")
            return False
        
        loaded_model = self.loaded_models[model_name]
        base_model_name = loaded_model.model_name
        
        # Load adapter using adapter manager
        success = self.adapter_manager.load_adapter(adapter_name, base_model_name)
        
        if success:
            # Update loaded model info
            loaded_model.active_adapter = adapter_name
            console.print(f"[green]✅ Adapter '{adapter_name}' loaded for model '{model_name}'[/green]")
        
        return success
    
    def unload_adapter(self, model_name: Optional[str] = None) -> bool:
        """
        Unload the current adapter from a model.
        
        Args:
            model_name: Name of the model (optional)
            
        Returns:
            True if adapter unloaded successfully, False otherwise
        """
        # Find target model
        if model_name is None:
            if not self.loaded_models:
                console.print("[red]❌ No models loaded.[/red]")
                return False
            model_name = next(iter(self.loaded_models.keys()))
        
        if model_name not in self.loaded_models:
            console.print(f"[red]❌ Model not loaded: {model_name}[/red]")
            return False
        
        loaded_model = self.loaded_models[model_name]
        
        if not loaded_model.active_adapter:
            console.print(f"[yellow]⚠️  No adapter loaded for model '{model_name}'[/yellow]")
            return True
        
        adapter_name = loaded_model.active_adapter
        base_model_name = loaded_model.model_name
        
        # Unload adapter using adapter manager
        success = self.adapter_manager.unload_adapter(adapter_name, base_model_name)
        
        if success:
            # Update loaded model info
            loaded_model.active_adapter = None
            console.print(f"[green]✅ Adapter '{adapter_name}' unloaded from model '{model_name}'[/green]")
        
        return success
    
    def switch_adapter(self, new_adapter: str, model_name: Optional[str] = None) -> bool:
        """
        Switch to a different adapter for a model.
        
        Args:
            new_adapter: Name of the new adapter to load
            model_name: Name of the model (optional)
            
        Returns:
            True if switch successful, False otherwise
        """
        # Find target model
        if model_name is None:
            if not self.loaded_models:
                console.print("[red]❌ No models loaded.[/red]")
                return False
            model_name = next(iter(self.loaded_models.keys()))
        
        if model_name not in self.loaded_models:
            console.print(f"[red]❌ Model not loaded: {model_name}[/red]")
            return False
        
        loaded_model = self.loaded_models[model_name]
        old_adapter = loaded_model.active_adapter
        base_model_name = loaded_model.model_name
        
        # Switch adapter using adapter manager
        success = self.adapter_manager.switch_adapter(old_adapter, new_adapter, base_model_name)
        
        if success:
            # Update loaded model info
            loaded_model.active_adapter = new_adapter
            console.print(f"[green]✅ Switched to adapter '{new_adapter}' for model '{model_name}'[/green]")
        
        return success
    
    def list_adapters(self) -> List[Dict[str, Any]]:
        """List all available adapters."""
        adapters = self.adapter_manager.available_adapters
        adapter_list = []
        
        for name, info in adapters.items():
            adapter_list.append({
                "name": name,
                "base_model": info.base_model,
                "task_type": info.task_type.value,
                "size_mb": info.size_mb,
                "status": info.status.value,
                "created_at": info.created_at
            })
        
        return adapter_list
    
    def get_adapter_info(self, adapter_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific adapter."""
        adapter_info = self.adapter_manager.get_adapter(adapter_name)
        
        if not adapter_info:
            return None
        
        return {
            "name": adapter_info.adapter_name,
            "base_model": adapter_info.base_model,
            "task_type": adapter_info.task_type.value,
            "size_mb": adapter_info.size_mb,
            "status": adapter_info.status.value,
            "created_at": adapter_info.created_at,
            "adapter_path": adapter_info.adapter_path,
            "training_stats": adapter_info.training_stats,
            "performance_metrics": adapter_info.performance_metrics
        }
    
    def benchmark_adapter(self, adapter_name: str, test_prompts: List[str], model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Benchmark an adapter's performance.
        
        Args:
            adapter_name: Name of the adapter to benchmark
            test_prompts: List of test prompts
            model_name: Name of the model (optional)
            
        Returns:
            Benchmark results
        """
        # Find target model
        if model_name is None:
            if not self.loaded_models:
                return {"error": "No models loaded"}
            model_name = next(iter(self.loaded_models.keys()))
        
        if model_name not in self.loaded_models:
            return {"error": f"Model not loaded: {model_name}"}
        
        loaded_model = self.loaded_models[model_name]
        base_model_name = loaded_model.model_name
        
        return self.adapter_manager.benchmark_adapter(adapter_name, test_prompts, base_model_name)
    
    def get_model_with_adapter(self, model_name: str) -> Optional[Any]:
        """
        Get the model instance with any loaded adapter.
        
        Args:
            model_name: Name of the loaded model
            
        Returns:
            Model instance (with adapter if loaded), None if not found
        """
        if model_name not in self.loaded_models:
            return None
        
        loaded_model = self.loaded_models[model_name]
        
        # If an adapter is loaded, get the adapted model
        if loaded_model.active_adapter:
            adapted_model = self.adapter_manager.get_adapter_model(
                loaded_model.active_adapter, 
                loaded_model.model_name
            )
            if adapted_model:
                return adapted_model
        
        # Return base model if no adapter
        return loaded_model.model


def main():
    """Test the API Server Agent"""
    api_agent = APIServerAgent()
    
    console.print(Panel.fit("🚀 Testing API Server Agent", style="bold blue"))
    
    # Show available models
    console.print("\n[cyan]Available Models:[/cyan]")
    api_agent.show_available_models()
    
    # Test model loading
    models = api_agent.discover_quantized_models()
    if models:
        test_model = models[0]['display_name']
        console.print(f"\n[cyan]Testing model loading: {test_model}[/cyan]")
        success = api_agent.load_model(test_model)
        
        if success:
            console.print("\n[cyan]Loaded Models:[/cyan]")
            api_agent.show_loaded_models()
            
            # Test unloading
            console.print(f"\n[cyan]Testing model unloading: {test_model}[/cyan]")
            api_agent.unload_model(test_model)


if __name__ == "__main__":
    main()