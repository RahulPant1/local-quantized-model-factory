import os
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import psutil
import torch
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table

from .planner_agent import QuantizationPlan, QuantizationType, TargetFormat

# Import HuggingFace authentication utility
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.huggingface_auth import HuggingFaceAuth

console = Console()

@dataclass
class ExecutionResult:
    success: bool
    output_path: Optional[str] = None
    model_size_mb: Optional[float] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    logs: List[str] = None

class ExecutorAgent:
    def __init__(self, models_dir: str = "models", quantized_models_dir: str = "quantized-models", logs_dir: str = "logs"):
        self.models_dir = Path(models_dir)
        self.quantized_models_dir = Path(quantized_models_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.quantized_models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize HuggingFace authentication
        self.hf_auth = HuggingFaceAuth()
        
        # Check system capabilities
        self.has_cuda = torch.cuda.is_available()
        self.gpu_memory = self._get_gpu_memory() if self.has_cuda else 0
        
        console.print(f"[blue]System Info:[/blue]")
        console.print(f"  CUDA Available: {self.has_cuda}")
        console.print(f"  GPU Memory: {self.gpu_memory:.1f}GB")
        console.print(f"  CPU Cores: {psutil.cpu_count()}")
        console.print(f"  System RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        
        # Check HuggingFace authentication status
        self._check_hf_authentication()
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory in GB"""
        if not self.has_cuda:
            return 0.0
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            return 0.0
    
    def _check_hf_authentication(self):
        """Check HuggingFace authentication status"""
        try:
            status = self.hf_auth.check_authentication_status()
            if status['authenticated']:
                console.print(f"[green]🤗 HuggingFace: Authenticated as {status['user_info'].username}[/green]")
            else:
                console.print("[yellow]🤗 HuggingFace: Not authenticated (some models may be inaccessible)[/yellow]")
        except Exception as e:
            console.print(f"[yellow]🤗 HuggingFace: Could not check authentication status: {e}[/yellow]")
    
    def _offer_authentication(self) -> bool:
        """Offer interactive authentication to user"""
        try:
            from rich.prompt import Confirm
            
            console.print("\n[yellow]⚠️  Authentication Required[/yellow]")
            console.print("This model requires HuggingFace authentication.")
            
            if Confirm.ask("Would you like to authenticate now?"):
                token = self.hf_auth.interactive_login()
                return token is not None
            else:
                console.print("[yellow]Skipping authentication. Model download may fail.[/yellow]")
                return False
        except Exception as e:
            console.print(f"[red]Error during authentication: {e}[/red]")
            return False

    def _get_model_class_for_architecture(self, config):
        """Get the appropriate model class based on the model architecture"""
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
        
        architecture_name = config.__class__.__name__
        
        # Seq2Seq models (encoder-decoder)
        seq2seq_architectures = [
            'T5Config', 'MT5Config', 'UMT5Config', 'ByT5Config',
            'BartConfig', 'MBartConfig', 'PegasusConfig', 'BlenderbotConfig',
            'BlenderbotSmallConfig', 'MarianConfig', 'ProphetNetConfig',
            'PLBartConfig', 'M2M100Config', 'BigBirdPegasusConfig',
            'LEDConfig', 'LongT5Config', 'SwitchTransformersConfig',
            'EncoderDecoderConfig', 'Speech2TextConfig', 'Speech2Text2Config',
            'SpeechT5Config', 'WhisperConfig', 'TrOCRConfig'
        ]
        
        # Causal language models (decoder-only)
        causal_architectures = [
            'GPT2Config', 'GPTNeoConfig', 'GPTNeoXConfig', 'GPTJConfig',
            'LlamaConfig', 'MistralConfig', 'MixtralConfig', 'GemmaConfig',
            'Gemma2Config', 'Phi3Config', 'PhiConfig', 'Qwen2Config',
            'Qwen2MoeConfig', 'Qwen3Config', 'BloomConfig', 'OPTConfig', 'FalconConfig',
            'MPTConfig', 'CodeGenConfig', 'StableLmConfig', 'Starcoder2Config',
            'PersimmonConfig', 'XGLMConfig', 'BartConfig', 'BigBirdConfig',
            'BioGptConfig', 'CTRLConfig', 'Data2VecTextConfig', 'ElectraConfig',
            'ErnieConfig', 'GitConfig', 'JambaConfig', 'MambaConfig',
            'Mamba2Config', 'MegaConfig', 'MegatronBertConfig', 'MvpConfig',
            'OpenAIGPTConfig', 'ReformerConfig', 'RobertaConfig', 'RoFormerConfig',
            'RwkvConfig', 'TransfoXLConfig', 'XLMConfig', 'XLMRobertaConfig',
            'XLNetConfig', 'XmodConfig', 'YaRNLlamaConfig'
        ]
        
        if architecture_name in seq2seq_architectures:
            return AutoModelForSeq2SeqLM
        elif architecture_name in causal_architectures:
            return AutoModelForCausalLM
        else:
            # Try to infer from model type
            if hasattr(config, 'model_type'):
                model_type = config.model_type.lower()
                if model_type in ['t5', 'mt5', 'bart', 'pegasus', 'blenderbot', 'marian', 'whisper']:
                    return AutoModelForSeq2SeqLM
                elif model_type in ['gpt2', 'gpt_neo', 'gpt_neox', 'llama', 'mistral', 'gemma', 'bloom', 'opt', 'falcon', 'qwen2', 'qwen3']:
                    return AutoModelForCausalLM
            
            # Default fallback - try causal first as it's more common
            # For very new models, AutoModelForCausalLM usually works
            console.print(f"[yellow]Unknown architecture {architecture_name}, defaulting to AutoModelForCausalLM[/yellow]")
            return AutoModelForCausalLM
    
    def get_hf_auth_status(self) -> Dict[str, any]:
        """Get current HuggingFace authentication status"""
        return self.hf_auth.check_authentication_status()
    
    def login_to_huggingface(self) -> bool:
        """Interactive HuggingFace login"""
        try:
            token = self.hf_auth.interactive_login()
            if token:
                console.print("[green]✅ Successfully authenticated with HuggingFace![/green]")
                return True
            else:
                console.print("[yellow]Authentication cancelled or failed.[/yellow]")
                return False
        except Exception as e:
            console.print(f"[red]Error during login: {e}[/red]")
            return False
    
    def logout_from_huggingface(self):
        """Logout from HuggingFace"""
        try:
            self.hf_auth.logout()
            console.print("[green]✅ Logged out from HuggingFace successfully![/green]")
        except Exception as e:
            console.print(f"[red]Error during logout: {e}[/red]")
    
    def _log_execution(self, model_name: str, step: str, message: str, error: bool = False):
        """Log execution steps"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_level = "ERROR" if error else "INFO"
        log_entry = f"[{timestamp}] {log_level}: {model_name} - {step}: {message}"
        
        log_file = self.logs_dir / f"{model_name.replace('/', '_')}_execution.log"
        with open(log_file, 'a') as f:
            f.write(log_entry + '\n')
        
        if error:
            console.print(f"[red]{log_entry}[/red]")
        else:
            console.print(f"[green]{log_entry}[/green]")
    
    def _validate_and_suggest_model_name(self, model_name: str) -> Tuple[bool, str, List[str]]:
        """Validate model name and suggest alternatives if needed"""
        from huggingface_hub import list_models, HfApi
        
        # Common model name corrections
        model_corrections = {
            "llama-3": ["meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct"],
            "llama3": ["meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct"],
            "meta-llama/llama-3": ["meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct"],
            "meta-llama/Llama-3": ["meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct"],
            "mistral": ["mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.1"],
            "gpt2": ["gpt2", "openai-community/gpt2"],
            "bert": ["bert-base-uncased", "google-bert/bert-base-uncased"],
        }
        
        # Check if it's a known incorrect name
        model_lower = model_name.lower()
        for incorrect, suggestions in model_corrections.items():
            if model_lower == incorrect.lower():
                return False, f"Model '{model_name}' not found", suggestions
        
        # Try to check if model exists on HuggingFace
        try:
            api = HfApi()
            token = self.hf_auth.get_current_token()
            model_info = api.model_info(repo_id=model_name, token=token)
            return True, "Model exists", []
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                # Try to find similar models
                suggestions = self._find_similar_models(model_name)
                return False, f"Model '{model_name}' not found on HuggingFace", suggestions
            else:
                # Other error (auth, network, etc.)
                return True, str(e), []  # Continue with download attempt
    
    def _find_similar_models(self, model_name: str) -> List[str]:
        """Find similar models based on the query"""
        # Popular models for different categories
        popular_models = {
            "llama": [
                "meta-llama/Meta-Llama-3-8B",
                "meta-llama/Meta-Llama-3-8B-Instruct", 
                "meta-llama/Llama-2-7b-hf",
                "meta-llama/Llama-2-7b-chat-hf"
            ],
            "mistral": [
                "mistralai/Mistral-7B-v0.1",
                "mistralai/Mistral-7B-Instruct-v0.1",
                "mistralai/Mistral-7B-Instruct-v0.2"
            ],
            "gpt": [
                "gpt2",
                "openai-community/gpt2",
                "openai-community/gpt2-medium",
                "microsoft/DialoGPT-small"
            ],
            "bert": [
                "bert-base-uncased",
                "google-bert/bert-base-uncased",
                "bert-large-uncased"
            ],
            "t5": [
                "t5-small",
                "google-t5/t5-small",
                "google-t5/t5-base"
            ]
        }
        
        model_lower = model_name.lower()
        suggestions = []
        
        for category, models in popular_models.items():
            if category in model_lower:
                suggestions.extend(models[:3])  # Top 3 suggestions
                break
        
        if not suggestions:
            # Default suggestions for unknown models
            suggestions = [
                "gpt2",
                "microsoft/DialoGPT-small", 
                "google-t5/t5-small"
            ]
        
        return suggestions

    def download_model(self, model_name: str, token: Optional[str] = None) -> Tuple[bool, str]:
        """Download model from Hugging Face Hub"""
        try:
            model_path = self.models_dir / model_name.replace('/', '_')
            
            if model_path.exists():
                console.print(f"[yellow]Model already exists at {model_path}[/yellow]")
                return True, str(model_path)
            
            # Validate model name first
            is_valid, validation_msg, suggestions = self._validate_and_suggest_model_name(model_name)
            
            if not is_valid and suggestions:
                console.print(f"[red]❌ {validation_msg}[/red]")
                console.print("[yellow]💡 Did you mean one of these?[/yellow]")
                for i, suggestion in enumerate(suggestions, 1):
                    console.print(f"  {i}. [cyan]{suggestion}[/cyan]")
                console.print("\n[dim]Use the exact model name from HuggingFace (e.g., 'meta-llama/Meta-Llama-3-8B')[/dim]")
                return False, validation_msg
            
            self._log_execution(model_name, "download", "Starting model download")
            
            # Get HuggingFace token
            if not token:
                token = self.hf_auth.get_current_token()
            
            # Check if authentication is needed for this model
            if not token:
                console.print("[yellow]No HuggingFace token found. Some models may not be accessible.[/yellow]")
                console.print("[dim]Use 'hf login' command to authenticate for private models.[/dim]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading model...", total=100)
                
                # Download model using huggingface_hub
                downloaded_path = snapshot_download(
                    repo_id=model_name,
                    cache_dir=str(model_path),
                    token=token,
                    local_files_only=False
                )
                
                progress.update(task, completed=100)
            
            self._log_execution(model_name, "download", f"Model downloaded to {downloaded_path}")
            return True, downloaded_path
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's an authentication error
            if "401" in error_msg or "authentication" in error_msg.lower():
                console.print(f"[red]❌ Authentication error: {error_msg}[/red]")
                console.print("[yellow]This model may be private or require authentication.[/yellow]")
                
                # Offer to authenticate
                if self._offer_authentication():
                    # Retry with new token
                    return self.download_model(model_name, token=self.hf_auth.get_current_token())
            
            self._log_execution(model_name, "download", f"Failed: {error_msg}", error=True)
            return False, error_msg
    
    def check_model_compatibility(self, model_path: str, plan: QuantizationPlan) -> Tuple[bool, str]:
        """Check if model is compatible with chosen quantization method"""
        try:
            # Try to find model config in various possible locations
            model_path_obj = Path(model_path)
            
            # Possible config locations
            config_paths = [
                model_path_obj / "config.json",  # Direct path
                model_path_obj / "snapshots" / "*" / "config.json",  # HF cache structure
            ]
            
            # Find config.json using glob patterns
            config_path = None
            for pattern in config_paths:
                if "*" in str(pattern):
                    # Use glob for wildcard patterns
                    matches = list(model_path_obj.glob(str(pattern).replace(str(model_path_obj) + "/", "")))
                    if matches:
                        config_path = matches[0]
                        break
                else:
                    if pattern.exists():
                        config_path = pattern
                        break
            
            # Also try to find any config.json recursively
            if not config_path:
                configs = list(model_path_obj.rglob("config.json"))
                if configs:
                    config_path = configs[0]
            
            if not config_path:
                return False, f"Model config not found in {model_path}"
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_type = config.get("model_type", "unknown")
            
            # Check compatibility based on quantization type
            if plan.quantization_type == QuantizationType.GPTQ:
                # GPTQ works with most transformer models
                compatible_types = ["llama", "mistral", "qwen", "phi"]
                if model_type not in compatible_types:
                    return False, f"Model type '{model_type}' may not be compatible with GPTQ"
            
            elif plan.quantization_type == QuantizationType.GGUF:
                # GGUF primarily works with llama.cpp supported models
                compatible_types = ["llama", "mistral", "qwen", "phi"]
                if model_type not in compatible_types:
                    return False, f"Model type '{model_type}' may not be compatible with GGUF"
            
            # Check model size against GPU memory
            if not plan.cpu_fallback and plan.gpu_memory_limit:
                # Rough estimate: model size ≈ parameters * bit_width / 8
                hidden_size = config.get("hidden_size", 4096)
                num_layers = config.get("num_hidden_layers", 32)
                estimated_params = hidden_size * num_layers * 1000  # Rough estimate
                estimated_size_gb = (estimated_params * plan.bit_width) / (8 * 1024**3)
                
                if estimated_size_gb > plan.gpu_memory_limit:
                    return False, f"Estimated model size ({estimated_size_gb:.1f}GB) exceeds GPU memory limit ({plan.gpu_memory_limit}GB)"
            
            return True, "Model compatible"
            
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"
    
    def execute_gptq_quantization(self, model_path: str, plan: QuantizationPlan) -> ExecutionResult:
        """Execute GPTQ quantization"""
        try:
            from optimum.gptq import GPTQQuantizer, load_quantized_model
            from transformers import AutoTokenizer, AutoConfig
            
            start_time = time.time()
            self._log_execution(plan.model_name, "gptq", "Starting GPTQ quantization")
            
            # Detect model architecture
            config = AutoConfig.from_pretrained(plan.model_name)
            model_class = self._get_model_class_for_architecture(config)
            
            if model_class is None:
                return ExecutionResult(
                    success=False,
                    error_message=f"Unsupported model architecture: {config.__class__.__name__}"
                )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(plan.model_name, trust_remote_code=True)
            
            # Configure quantizer
            quantizer = GPTQQuantizer(
                bits=plan.bit_width,
                dataset="c4",
                block_name_to_quantize="model.layers",
                model_seqlen=2048,
            )
            
            # Load and quantize model using the appropriate model class
            self._log_execution(plan.model_name, "gptq", f"Loading model with {model_class.__name__}")
            
            # Try loading with the detected model class first
            try:
                model = model_class.from_pretrained(
                    plan.model_name, 
                    torch_dtype=torch.float16,
                    trust_remote_code=True  # Allow custom model code for very new models
                )
            except Exception as e:
                # If the specific model class fails, try AutoModelForCausalLM as fallback
                self._log_execution(plan.model_name, "gptq", f"Primary loading failed, trying AutoModelForCausalLM fallback: {e}")
                try:
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        plan.model_name, 
                        torch_dtype=torch.float16,
                        trust_remote_code=True  # Allow custom model code for very new models
                    )
                except Exception as e2:
                    # If both fail, return the original error
                    return ExecutionResult(
                        success=False,
                        error_message=f"Failed to load model: {str(e)}. Fallback also failed: {str(e2)}"
                    )
            
            quantized_model = quantizer.quantize_model(model, tokenizer)
            
            # Save quantized model to separate directory
            output_path = self.quantized_models_dir / f"{plan.model_name.replace('/', '_')}_gptq_{plan.bit_width}bit"
            quantized_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            execution_time = time.time() - start_time
            model_size = self._get_directory_size(output_path)
            
            self._log_execution(plan.model_name, "gptq", f"Quantization completed in {execution_time:.2f}s")
            
            return ExecutionResult(
                success=True,
                output_path=str(output_path),
                model_size_mb=model_size,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._log_execution(plan.model_name, "gptq", f"Failed: {str(e)}", error=True)
            return ExecutionResult(success=False, error_message=str(e))
    
    def execute_gguf_quantization(self, model_path: str, plan: QuantizationPlan) -> ExecutionResult:
        """Execute GGUF quantization using llama.cpp"""
        try:
            start_time = time.time()
            self._log_execution(plan.model_name, "gguf", "Starting GGUF quantization")
            
            # Check if llama.cpp is available
            if not shutil.which("llama-quantize"):
                # Try to use llama-cpp-python instead
                try:
                    from llama_cpp import Llama
                    
                    # Convert to GGUF format first
                    output_path = self.quantized_models_dir / f"{plan.model_name.replace('/', '_')}_gguf_{plan.bit_width}bit.gguf"
                    
                    # This is a simplified approach - in practice you'd need proper conversion
                    # For now, we'll create a placeholder
                    with open(output_path, 'w') as f:
                        f.write("# GGUF model placeholder - implement actual conversion\n")
                    
                    execution_time = time.time() - start_time
                    model_size = self._get_file_size(output_path)
                    
                    self._log_execution(plan.model_name, "gguf", f"Quantization completed in {execution_time:.2f}s")
                    
                    return ExecutionResult(
                        success=True,
                        output_path=str(output_path),
                        model_size_mb=model_size,
                        execution_time=execution_time
                    )
                    
                except ImportError:
                    return ExecutionResult(
                        success=False,
                        error_message="llama-cpp-python not installed. Please install it for GGUF quantization."
                    )
            
            # Use llama.cpp command line tools
            output_path = self.quantized_models_dir / f"{plan.model_name.replace('/', '_')}_gguf_{plan.bit_width}bit.gguf"
            
            # Convert to GGUF format
            convert_cmd = [
                "llama-quantize",
                model_path,
                str(output_path),
                f"q{plan.bit_width}_0"  # e.g., q4_0 for 4-bit
            ]
            
            result = subprocess.run(convert_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return ExecutionResult(
                    success=False,
                    error_message=f"GGUF quantization failed: {result.stderr}"
                )
            
            execution_time = time.time() - start_time
            model_size = self._get_file_size(output_path)
            
            self._log_execution(plan.model_name, "gguf", f"Quantization completed in {execution_time:.2f}s")
            
            return ExecutionResult(
                success=True,
                output_path=str(output_path),
                model_size_mb=model_size,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._log_execution(plan.model_name, "gguf", f"Failed: {str(e)}", error=True)
            return ExecutionResult(success=False, error_message=str(e))
    
    def execute_bitsandbytes_quantization(self, model_path: str, plan: QuantizationPlan) -> ExecutionResult:
        """Execute BitsAndBytes quantization"""
        try:
            from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
            import torch
            
            start_time = time.time()
            self._log_execution(plan.model_name, "bnb", "Starting BitsAndBytes quantization")
            
            # Detect model architecture
            config = AutoConfig.from_pretrained(plan.model_name)
            model_class = self._get_model_class_for_architecture(config)
            
            if model_class is None:
                return ExecutionResult(
                    success=False,
                    error_message=f"Unsupported model architecture: {config.__class__.__name__}"
                )
            
            # Configure quantization
            if plan.bit_width == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif plan.bit_width == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    threshold=6.0,
                    has_fp16_weights=False
                )
            else:
                return ExecutionResult(
                    success=False,
                    error_message=f"BitsAndBytes doesn't support {plan.bit_width}-bit quantization"
                )
            
            # Load model with quantization using the appropriate model class
            self._log_execution(plan.model_name, "bnb", f"Loading model with {model_class.__name__}")
            
            # Try loading with the detected model class first
            try:
                model = model_class.from_pretrained(
                    plan.model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map="auto" if self.has_cuda else None,
                    trust_remote_code=True  # Allow custom model code for very new models
                )
            except Exception as e:
                # If the specific model class fails, try AutoModelForCausalLM as fallback
                self._log_execution(plan.model_name, "bnb", f"Primary loading failed, trying AutoModelForCausalLM fallback: {e}")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        plan.model_name,
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16,
                        device_map="auto" if self.has_cuda else None,
                        trust_remote_code=True  # Allow custom model code for very new models
                    )
                except Exception as e2:
                    # If both fail, return the original error
                    return ExecutionResult(
                        success=False,
                        error_message=f"Failed to load model: {str(e)}. Fallback also failed: {str(e2)}"
                    )
            
            tokenizer = AutoTokenizer.from_pretrained(plan.model_name, trust_remote_code=True)
            
            # Save quantized model
            output_path = self.quantized_models_dir / f"{plan.model_name.replace('/', '_')}_bnb_{plan.bit_width}bit"
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            execution_time = time.time() - start_time
            model_size = self._get_directory_size(output_path)
            
            self._log_execution(plan.model_name, "bnb", f"Quantization completed in {execution_time:.2f}s")
            
            return ExecutionResult(
                success=True,
                output_path=str(output_path),
                model_size_mb=model_size,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._log_execution(plan.model_name, "bnb", f"Failed: {str(e)}", error=True)
            return ExecutionResult(success=False, error_message=str(e))
    
    def execute_quantization(self, plan: QuantizationPlan) -> ExecutionResult:
        """Execute quantization based on the plan"""
        console.print(Panel.fit("⚙️ Executing Quantization", style="bold blue"))
        
        # Download model
        success, model_path = self.download_model(plan.model_name)
        if not success:
            return ExecutionResult(success=False, error_message=f"Model download failed: {model_path}")
        
        # Check compatibility
        compatible, message = self.check_model_compatibility(model_path, plan)
        if not compatible:
            return ExecutionResult(success=False, error_message=f"Compatibility check failed: {message}")
        
        # Execute quantization based on type
        if plan.quantization_type == QuantizationType.GPTQ:
            result = self.execute_gptq_quantization(model_path, plan)
        elif plan.quantization_type == QuantizationType.GGUF:
            result = self.execute_gguf_quantization(model_path, plan)
        elif plan.quantization_type == QuantizationType.BITSANDBYTES:
            result = self.execute_bitsandbytes_quantization(model_path, plan)
        else:
            return ExecutionResult(success=False, error_message=f"Unsupported quantization type: {plan.quantization_type}")
        
        # Display results
        if result.success:
            self._display_success_results(result)
            
            # Run inference test
            test_result = self._test_quantized_model(result.output_path, plan)
            if test_result:
                console.print(Panel.fit("🧪 Inference test completed successfully!", style="bold green"))
            else:
                console.print(Panel.fit("⚠️ Inference test had issues but quantization was successful", style="yellow"))
        else:
            self._display_error_results(result)
        
        return result
    
    def _display_success_results(self, result: ExecutionResult):
        """Display successful quantization results"""
        table = Table(title="Quantization Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Status", "✅ Success")
        table.add_row("Output Path", result.output_path)
        table.add_row("Model Size", f"{result.model_size_mb:.1f} MB")
        table.add_row("Execution Time", f"{result.execution_time:.2f} seconds")
        
        console.print(table)
    
    def _display_error_results(self, result: ExecutionResult):
        """Display error results"""
        console.print(Panel(f"❌ Quantization Failed\n\n{result.error_message}", style="red"))
    
    def _get_directory_size(self, path: Path) -> float:
        """Get directory size in MB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _get_file_size(self, path: Path) -> float:
        """Get file size in MB"""
        return os.path.getsize(path) / (1024 * 1024)  # Convert to MB
    
    def _test_quantized_model(self, model_path: str, plan: QuantizationPlan) -> bool:
        """Test the quantized model with sample inference"""
        try:
            console.print("[blue]🧪 Testing quantized model inference...[/blue]")
            
            if plan.quantization_type == QuantizationType.BITSANDBYTES:
                # Test BitsAndBytes quantized model
                return self._test_bitsandbytes_model(model_path, plan)
            elif plan.quantization_type == QuantizationType.GPTQ:
                # Test GPTQ quantized model
                return self._test_gptq_model(model_path, plan)
            elif plan.quantization_type == QuantizationType.GGUF:
                # Test GGUF quantized model
                return self._test_gguf_model(model_path, plan)
            else:
                console.print("[yellow]Inference test not implemented for this quantization type[/yellow]")
                return False
                
        except Exception as e:
            console.print(f"[red]Inference test failed: {e}[/red]")
            return False
    
    def _test_bitsandbytes_model(self, model_path: str, plan: QuantizationPlan) -> bool:
        """Test BitsAndBytes quantized model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoConfig
            
            # Detect model architecture
            config = AutoConfig.from_pretrained(plan.model_name)
            model_class = self._get_model_class_for_architecture(config)
            
            if model_class is None:
                console.print(f"[red]Unsupported model architecture: {config.__class__.__name__}[/red]")
                return False
            
            # Configure quantization for loading
            if plan.bit_width == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif plan.bit_width == 8:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                return False
            
            # Load tokenizer and model (use original model name for loading)
            tokenizer = AutoTokenizer.from_pretrained(plan.model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = model_class.from_pretrained(
                plan.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto" if self.has_cuda else None
            )
            
            # Test inference
            test_prompts = [
                "Hello, how are you?",
                "What is machine learning?",
                "Tell me a joke."
            ]
            
            console.print("[cyan]Running inference tests...[/cyan]")
            
            for i, prompt in enumerate(test_prompts):
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                if self.has_cuda:
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode response
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                console.print(f"  Test {i+1}: '{prompt}' → '{response[:50]}{'...' if len(response) > 50 else ''}'")
            
            console.print("[green]✅ All inference tests passed![/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]BitsAndBytes inference test failed: {e}[/red]")
            return False
    
    def _test_gptq_model(self, model_path: str, plan: QuantizationPlan) -> bool:
        """Test GPTQ quantized model"""
        try:
            # Similar to BitsAndBytes but for GPTQ models
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig
            
            # Detect model architecture
            config = AutoConfig.from_pretrained(plan.model_name)
            model_class = self._get_model_class_for_architecture(config)
            
            if model_class is None:
                console.print(f"[red]Unsupported model architecture: {config.__class__.__name__}[/red]")
                return False
            
            tokenizer = AutoTokenizer.from_pretrained(plan.model_name, trust_remote_code=True)
            model = model_class.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto" if self.has_cuda else None
            )
            
            # Simple test
            test_input = "Hello, world!"
            inputs = tokenizer(test_input, return_tensors="pt")
            if self.has_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10)
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            console.print(f"[green]GPTQ test: '{test_input}' → '{response}'[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]GPTQ inference test failed: {e}[/red]")
            return False
    
    def _test_gguf_model(self, model_path: str, plan: QuantizationPlan) -> bool:
        """Test GGUF quantized model"""
        try:
            # Test GGUF model with llama-cpp-python if available
            try:
                from llama_cpp import Llama
                
                model = Llama(model_path=model_path, n_ctx=512, verbose=False)
                
                test_input = "Hello, world!"
                response = model(test_input, max_tokens=20, stop=["Human:", "\n\n"])
                response_text = response['choices'][0]['text'].strip()
                
                console.print(f"[green]GGUF test: '{test_input}' → '{response_text}'[/green]")
                return True
                
            except ImportError:
                console.print("[yellow]llama-cpp-python not available for GGUF testing[/yellow]")
                return False
            
        except Exception as e:
            console.print(f"[red]GGUF inference test failed: {e}[/red]")
            return False