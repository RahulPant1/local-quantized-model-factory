#!/usr/bin/env python3
"""
Fine-Tuning Agent for LQMF - Parameter-Efficient Fine-Tuning with LoRA/QLoRA

This agent provides lightweight fine-tuning capabilities for quantized models using:
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Parameter-efficient tuning for 8GB GPU environments
- Adapter hot-swapping capabilities
- Dataset preprocessing and training management

Author: LQMF Development Team
Version: 1.0.0
Compatibility: Python 3.8+, PyTorch 2.0+

Features:
- Support for multiple task types (chat, classification, summarization)
- Automatic dataset preprocessing from CSV, JSONL, or text files
- Memory-efficient training with gradient checkpointing
- Adapter versioning and management
- Real-time training progress monitoring
- Automatic model evaluation and benchmarking
"""

import os
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling, AutoConfig
)

try:
    from peft import (
        LoraConfig, get_peft_model, TaskType as PeftTaskType, PeftModel,
        prepare_model_for_kbit_training, get_peft_model_state_dict
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    # Create dummy classes for import compatibility
    class LoraConfig:
        pass
    class PeftTaskType:
        CAUSAL_LM = "CAUSAL_LM"

try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm

# Import LQMF components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from agents.memory_agent import MemoryAgent

console = Console()
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Supported fine-tuning task types"""
    CHAT = "chat"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    INSTRUCTION_FOLLOWING = "instruction_following"

class AdapterStatus(Enum):
    """Status of adapter training"""
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    LOADED = "loaded"
    UNLOADED = "unloaded"

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning job"""
    job_name: str
    base_model: str
    task_type: TaskType
    dataset_path: str
    output_dir: str
    
    # LoRA Configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    
    # Training Configuration
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = -1
    warmup_steps: int = 100
    
    # Memory Optimization
    use_gradient_checkpointing: bool = True
    use_4bit_quantization: bool = True
    use_8bit_optimizer: bool = True
    max_seq_length: int = 512
    
    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_steps: int = 100
    
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

@dataclass
class AdapterInfo:
    """Information about a trained adapter"""
    adapter_name: str
    base_model: str
    task_type: TaskType
    adapter_path: str
    config_path: str
    size_mb: float
    status: AdapterStatus
    created_at: str
    training_stats: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None

class CustomDataset(Dataset):
    """Custom dataset class for different task types"""
    
    def __init__(self, data: List[Dict[str, str]], tokenizer, task_type: TaskType, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.task_type == TaskType.CHAT:
            # Format: {"input": "user message", "output": "assistant response"}
            text = f"### Human: {item['input']}\n### Assistant: {item['output']}"
        elif self.task_type == TaskType.INSTRUCTION_FOLLOWING:
            # Format: {"instruction": "task", "input": "context", "output": "response"}
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output_text = item.get('output', '')
            if input_text:
                text = f"### Instruction: {instruction}\n### Input: {input_text}\n### Response: {output_text}"
            else:
                text = f"### Instruction: {instruction}\n### Response: {output_text}"
        elif self.task_type == TaskType.CLASSIFICATION:
            # Format: {"text": "input text", "label": "classification label"}
            text = f"Text: {item['text']}\nLabel: {item['label']}"
        elif self.task_type == TaskType.SUMMARIZATION:
            # Format: {"text": "long text", "summary": "short summary"}
            text = f"Text: {item['text']}\nSummary: {item['summary']}"
        else:
            # Generic format
            text = f"{item.get('input', '')}\n{item.get('output', '')}"
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].flatten(),
            "attention_mask": encodings["attention_mask"].flatten(),
            "labels": encodings["input_ids"].flatten()
        }

class FineTuningAgent:
    """
    Main agent for handling LoRA/QLoRA fine-tuning operations.
    
    Provides comprehensive fine-tuning capabilities while maintaining
    memory efficiency for 8GB GPU environments.
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 adapters_dir: str = "adapters", 
                 logs_dir: str = "logs"):
        """
        Initialize the Fine-Tuning Agent.
        
        Args:
            models_dir: Directory containing base models
            adapters_dir: Directory for storing trained adapters
            logs_dir: Directory for training logs
        """
        self.models_dir = Path(models_dir)
        self.adapters_dir = Path(adapters_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        self.adapters_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize memory agent for tracking
        self.memory_agent = MemoryAgent()
        
        # Check system capabilities
        self._check_dependencies()
        self._check_gpu_memory()
        
        # Adapter registry
        self.adapters_registry_path = self.adapters_dir / "registry.json"
        self.adapters: Dict[str, AdapterInfo] = self._load_adapters_registry()
        
        console.print("[green]✅ Fine-Tuning Agent initialized successfully[/green]")
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library not available. Install with: pip install peft")
        
        if not BNB_AVAILABLE:
            console.print("[yellow]⚠️  BitsAndBytes not available. 4-bit quantization disabled.[/yellow]")
        
        console.print("[green]✅ Dependencies check passed[/green]")
    
    def _check_gpu_memory(self) -> None:
        """Check GPU memory availability."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            console.print(f"[blue]🔧 GPU Memory: {gpu_memory:.1f}GB[/blue]")
            
            if gpu_memory < 6:
                console.print("[yellow]⚠️  Low GPU memory detected. Consider using CPU or smaller batch sizes.[/yellow]")
        else:
            console.print("[yellow]⚠️  No GPU detected. Fine-tuning will be slow on CPU.[/yellow]")
    
    def _load_adapters_registry(self) -> Dict[str, AdapterInfo]:
        """Load the adapters registry from disk."""
        if self.adapters_registry_path.exists():
            try:
                with open(self.adapters_registry_path, 'r') as f:
                    data = json.load(f)
                    adapters = {}
                    for name, info in data.items():
                        adapters[name] = AdapterInfo(**info)
                    return adapters
            except Exception as e:
                logger.error(f"Failed to load adapters registry: {e}")
                return {}
        return {}
    
    def _save_adapters_registry(self) -> None:
        """Save the adapters registry to disk."""
        try:
            data = {}
            for name, info in self.adapters.items():
                data[name] = asdict(info)
            
            with open(self.adapters_registry_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save adapters registry: {e}")
    
    def load_dataset(self, 
                    dataset_path: str, 
                    task_type: TaskType,
                    test_split: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """
        Load and preprocess dataset from various formats.
        
        Args:
            dataset_path: Path to dataset file or directory
            task_type: Type of task for proper formatting
            test_split: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_data, test_data)
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        console.print(f"[blue]📁 Loading dataset from: {dataset_path}[/blue]")
        
        data = []
        
        if dataset_path.is_file():
            if dataset_path.suffix == '.csv':
                df = pd.read_csv(dataset_path)
                data = df.to_dict('records')
            elif dataset_path.suffix == '.jsonl':
                with open(dataset_path, 'r') as f:
                    data = [json.loads(line) for line in f]
            elif dataset_path.suffix == '.json':
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
            else:
                raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
        
        elif dataset_path.is_dir():
            # Load all text files from directory
            for file_path in dataset_path.glob("*.txt"):
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    # Simple split for chat format
                    if task_type == TaskType.CHAT:
                        # Assume each file contains alternating user/assistant messages
                        lines = content.split('\n')
                        for i in range(0, len(lines)-1, 2):
                            if i+1 < len(lines):
                                data.append({
                                    'input': lines[i].strip(),
                                    'output': lines[i+1].strip()
                                })
                    else:
                        data.append({'text': content})
        
        if not data:
            raise ValueError("No data loaded from dataset")
        
        # Validate data format
        self._validate_dataset_format(data, task_type)
        
        # Split into train/test
        split_idx = int(len(data) * (1 - test_split))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        console.print(f"[green]✅ Loaded {len(train_data)} training samples, {len(test_data)} test samples[/green]")
        
        return train_data, test_data
    
    def _validate_dataset_format(self, data: List[Dict], task_type: TaskType) -> None:
        """Validate dataset format for the specified task type."""
        if not data:
            raise ValueError("Dataset is empty")
        
        sample = data[0]
        
        if task_type == TaskType.CHAT:
            required_keys = ['input', 'output']
        elif task_type == TaskType.INSTRUCTION_FOLLOWING:
            required_keys = ['instruction', 'output']
        elif task_type == TaskType.CLASSIFICATION:
            required_keys = ['text', 'label']
        elif task_type == TaskType.SUMMARIZATION:
            required_keys = ['text', 'summary']
        else:
            return  # Skip validation for unknown types
        
        missing_keys = [key for key in required_keys if key not in sample]
        if missing_keys:
            raise ValueError(f"Dataset missing required keys for {task_type.value}: {missing_keys}")
    
    def create_lora_config(self, 
                          base_model: str,
                          config: FineTuningConfig) -> LoraConfig:
        """
        Create LoRA configuration based on model architecture.
        
        Args:
            base_model: Name of the base model
            config: Fine-tuning configuration
            
        Returns:
            LoRA configuration object
        """
        # Default target modules for different model families
        if config.target_modules is None:
            if "llama" in base_model.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "mistral" in base_model.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "falcon" in base_model.lower():
                target_modules = ["query_key_value", "dense"]
            elif "gpt" in base_model.lower():
                target_modules = ["c_attn", "c_proj"]
            else:
                # Generic targets for transformer models
                target_modules = ["q_proj", "v_proj"]
        else:
            target_modules = config.target_modules
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=PeftTaskType.CAUSAL_LM,
        )
        
        console.print(f"[blue]🔧 LoRA Config: r={config.lora_r}, alpha={config.lora_alpha}, targets={target_modules}[/blue]")
        
        return lora_config
    
    def start_fine_tuning(self, config: FineTuningConfig) -> bool:
        """
        Start fine-tuning process with the given configuration.
        
        Args:
            config: Fine-tuning configuration
            
        Returns:
            True if training completed successfully, False otherwise
        """
        console.print(Panel.fit(
            f"🚀 Starting Fine-Tuning Job: {config.job_name}",
            style="bold blue"
        ))
        
        try:
            # Load dataset
            train_data, test_data = self.load_dataset(
                config.dataset_path, 
                config.task_type
            )
            
            # Load base model and tokenizer
            model, tokenizer = self._load_base_model(config)
            
            # Create LoRA configuration
            lora_config = self.create_lora_config(config.base_model, config)
            
            # Apply LoRA to model
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Prepare datasets
            train_dataset = CustomDataset(train_data, tokenizer, config.task_type, config.max_seq_length)
            eval_dataset = CustomDataset(test_data, tokenizer, config.task_type, config.max_seq_length)
            
            # Configure training arguments
            training_args = self._create_training_arguments(config)
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            
            # Start training
            console.print("[blue]🎯 Starting training...[/blue]")
            start_time = time.time()
            
            trainer.train()
            
            training_time = time.time() - start_time
            console.print(f"[green]✅ Training completed in {training_time:.2f}s[/green]")
            
            # Save adapter
            adapter_path = self.adapters_dir / config.job_name
            adapter_path.mkdir(exist_ok=True)
            
            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            
            # Save configuration
            config_path = adapter_path / "fine_tuning_config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2, default=str)
            
            # Calculate adapter size
            adapter_size = sum(f.stat().st_size for f in adapter_path.rglob('*') if f.is_file()) / 1024**2
            
            # Register adapter
            adapter_info = AdapterInfo(
                adapter_name=config.job_name,
                base_model=config.base_model,
                task_type=config.task_type,
                adapter_path=str(adapter_path),
                config_path=str(config_path),
                size_mb=adapter_size,
                status=AdapterStatus.COMPLETED,
                created_at=datetime.now().isoformat(),
                training_stats={
                    "training_time": training_time,
                    "num_epochs": config.num_train_epochs,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.per_device_train_batch_size
                }
            )
            
            self.adapters[config.job_name] = adapter_info
            self._save_adapters_registry()
            
            console.print(f"[green]✅ Adapter saved: {adapter_size:.1f}MB[/green]")
            
            return True
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            console.print(f"[red]❌ Fine-tuning failed: {e}[/red]")
            return False
    
    def _load_base_model(self, config: FineTuningConfig) -> Tuple[nn.Module, Any]:
        """Load base model with appropriate quantization settings."""
        console.print(f"[blue]📥 Loading base model: {config.base_model}[/blue]")
        
        # Configure quantization
        quantization_config = None
        if config.use_4bit_quantization and BNB_AVAILABLE:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare model for training
        if config.use_4bit_quantization:
            model = prepare_model_for_kbit_training(model)
        
        if config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model, tokenizer
    
    def _create_training_arguments(self, config: FineTuningConfig) -> TrainingArguments:
        """Create training arguments from configuration."""
        return TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps if config.max_steps > 0 else -1,
            evaluation_strategy=config.evaluation_strategy,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            logging_steps=50,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,
            optim="adamw_8bit" if config.use_8bit_optimizer and BNB_AVAILABLE else "adamw_hf",
            gradient_checkpointing=config.use_gradient_checkpointing,
            dataloader_pin_memory=False,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    
    def list_adapters(self) -> List[AdapterInfo]:
        """List all available adapters."""
        return list(self.adapters.values())
    
    def get_adapter(self, adapter_name: str) -> Optional[AdapterInfo]:
        """Get information about a specific adapter."""
        return self.adapters.get(adapter_name)
    
    def export_adapter(self, adapter_name: str, export_path: str) -> bool:
        """
        Export an adapter to a specified location.
        
        Args:
            adapter_name: Name of the adapter to export
            export_path: Path where to export the adapter
            
        Returns:
            True if export successful, False otherwise
        """
        if adapter_name not in self.adapters:
            console.print(f"[red]❌ Adapter not found: {adapter_name}[/red]")
            return False
        
        adapter_info = self.adapters[adapter_name]
        source_path = Path(adapter_info.adapter_path)
        export_path = Path(export_path)
        
        try:
            if export_path.exists():
                shutil.rmtree(export_path)
            
            shutil.copytree(source_path, export_path)
            console.print(f"[green]✅ Adapter exported to: {export_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Export failed: {e}[/red]")
            return False
    
    def delete_adapter(self, adapter_name: str) -> bool:
        """
        Delete an adapter and its files.
        
        Args:
            adapter_name: Name of the adapter to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        if adapter_name not in self.adapters:
            console.print(f"[red]❌ Adapter not found: {adapter_name}[/red]")
            return False
        
        adapter_info = self.adapters[adapter_name]
        adapter_path = Path(adapter_info.adapter_path)
        
        try:
            if adapter_path.exists():
                shutil.rmtree(adapter_path)
            
            del self.adapters[adapter_name]
            self._save_adapters_registry()
            
            console.print(f"[green]✅ Adapter deleted: {adapter_name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Deletion failed: {e}[/red]")
            return False