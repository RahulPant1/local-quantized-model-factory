import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
# Import handled gracefully in initialization
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    genai = None
    GOOGLE_AI_AVAILABLE = False
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

console = Console()

class QuantizationType(Enum):
    GPTQ = "gptq"
    BITSANDBYTES = "bitsandbytes"
    GGUF = "gguf"
    AWQGPTQ = "awq"

class TargetFormat(Enum):
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    PYTORCH = "pytorch"

@dataclass
class QuantizationPlan:
    model_name: str
    quantization_type: QuantizationType
    target_format: TargetFormat
    bit_width: int
    cpu_fallback: bool
    gpu_memory_limit: Optional[int] = None
    specific_params: Optional[Dict] = None

class PlannerAgent:
    def __init__(self, gemini_api_key: Optional[str] = None):
        # Check if Google AI is available
        if not GOOGLE_AI_AVAILABLE:
            console.print("[yellow]⚠️ Google AI not available, some planning features disabled[/yellow]")
        
        # Use centralized LLM configuration
        try:
            from utils.llm_config import get_llm_manager
            self.llm_manager = get_llm_manager()
        except Exception as e:
            console.print(f"[yellow]⚠️ LLM manager initialization failed: {e}[/yellow]")
            self.llm_manager = None
        
        # Keep backward compatibility for direct model access
        if self.llm_manager and self.llm_manager.current_provider:
            provider_instance = self.llm_manager.provider_instances.get(self.llm_manager.current_provider, {})
            self.model = provider_instance.get('model') if isinstance(provider_instance, dict) else None
        else:
            self.model = None
            
        self.gemini_api_key = gemini_api_key
    
    def parse_user_request(self, user_input: str) -> Dict:
        """Parse user request to extract model name and quantization preferences"""
        parsed = {
            "model_name": None,
            "quantization_type": None,
            "target_format": None,
            "bit_width": None,
            "cpu_fallback": False,
            "ambiguous_parts": []
        }
        
        # Extract model name patterns (improved to handle complex names)
        model_patterns = [
            r'(mistralai/[A-Za-z0-9._-]+)',
            r'(meta-llama/[A-Za-z0-9._-]+)',
            r'(microsoft/[A-Za-z0-9._-]+)',
            r'(Qwen/[A-Za-z0-9._-]+)',
            r'(google/[A-Za-z0-9._-]+)',
            r'(huggingface/[A-Za-z0-9._-]+)',
            r'(facebook/[A-Za-z0-9._-]+)',
            r'(openai/[A-Za-z0-9._-]+)',
            r'([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)',  # Generic pattern last
        ]
        
        for pattern in model_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                parsed["model_name"] = match.group(1)
                break
        
        # Extract quantization preferences
        user_lower = user_input.lower()
        
        if "gguf" in user_lower:
            parsed["quantization_type"] = QuantizationType.GGUF
            parsed["target_format"] = TargetFormat.GGUF
        elif "gptq" in user_lower:
            parsed["quantization_type"] = QuantizationType.GPTQ
            parsed["target_format"] = TargetFormat.SAFETENSORS
        elif "bitsandbytes" in user_lower or "bnb" in user_lower:
            parsed["quantization_type"] = QuantizationType.BITSANDBYTES
            parsed["target_format"] = TargetFormat.PYTORCH
        
        # Extract bit width
        bit_patterns = [r'(\d+)-?bit', r'q(\d+)', r'int(\d+)']
        for pattern in bit_patterns:
            match = re.search(pattern, user_lower)
            if match:
                parsed["bit_width"] = int(match.group(1))
                break
        
        # Check for CPU preference
        if "cpu" in user_lower:
            parsed["cpu_fallback"] = True
        
        # Identify ambiguous parts
        if not parsed["model_name"]:
            parsed["ambiguous_parts"].append("model_name")
        if not parsed["quantization_type"]:
            parsed["ambiguous_parts"].append("quantization_type")
        if not parsed["bit_width"]:
            parsed["ambiguous_parts"].append("bit_width")
        
        return parsed
    
    def get_gemini_suggestion(self, user_input: str, parsed_info: Dict) -> Optional[str]:
        """Get suggestions from LLM for ambiguous requests"""
        if not self.llm_manager.is_available():
            return None
        
        try:
            # Convert enum objects to strings for JSON serialization
            serializable_info = {}
            for key, value in parsed_info.items():
                if hasattr(value, 'value'):  # Enum object
                    serializable_info[key] = value.value
                elif hasattr(value, 'name'):  # Enum object (alternative check)
                    serializable_info[key] = value.name
                else:
                    serializable_info[key] = value
            
            prompt = f"""
            User wants to quantize a model with this request: "{user_input}"
            
            Parsed information so far: {json.dumps(serializable_info, indent=2)}
            
            Please provide specific suggestions for:
            1. Best quantization method (GPTQ, GGUF, BitsAndBytes) for the use case
            2. Recommended bit width (4-bit, 8-bit, etc.)
            3. Target format compatibility
            4. Any compatibility concerns for 8GB GPU constraint
            
            Respond with practical, specific recommendations in a concise format.
            """
            
            from utils.llm_config import QueryType
            response = self.llm_manager.query(prompt, QueryType.QUANTIZATION_SUGGESTION)
            return response.content if response.success else None
        except Exception as e:
            console.print(f"[yellow]Gemini API error: {e}[/yellow]")
            return None
    
    def interactive_planning(self, user_input: str) -> QuantizationPlan:
        """Interactive planning session with user"""
        console.print(Panel.fit("🧠 Planning Quantization", style="bold blue"))
        
        # Parse initial request
        parsed = self.parse_user_request(user_input)
        
        # Show quantization compatibility info if model is detected
        if parsed["model_name"]:
            self._show_quantization_compatibility(parsed["model_name"])
        
        # Get LLM suggestions if available (with timeout)
        llm_suggestion = None
        if parsed["ambiguous_parts"] and self.llm_manager.is_available():
            provider = self.llm_manager.get_provider()
            console.print(f"[blue]Consulting {provider.value.title() if provider else 'LLM'} for recommendations (5s timeout)...[/blue]")
            
            # Add timeout handling for the LLM API call
            try:
                from concurrent.futures import ThreadPoolExecutor, TimeoutError
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(self.get_gemini_suggestion, user_input, parsed)
                    try:
                        llm_suggestion = future.result(timeout=5)  # 5 second timeout
                    except TimeoutError:
                        console.print(f"[yellow]⏱️ {provider.value.title() if provider else 'LLM'} API timeout - continuing without suggestions[/yellow]")
                        llm_suggestion = None
            except Exception as e:
                console.print(f"[yellow]Skipping LLM suggestions: {e}[/yellow]")
                llm_suggestion = None
                
            if llm_suggestion:
                console.print(Panel(llm_suggestion, title=f"🤖 {provider.value.title() if provider else 'LLM'} Suggestions", style="cyan"))
            else:
                console.print(f"[dim]Proceeding without {provider.value.title() if provider else 'LLM'} suggestions...[/dim]")
        
        # Interactive clarification
        model_name = parsed["model_name"]
        if not model_name:
            model_name = Prompt.ask("Enter the Hugging Face model name (e.g., mistralai/Mistral-7B-Instruct)")
        
        # Quantization type selection
        quant_type = parsed["quantization_type"]
        if not quant_type:
            console.print("\n[bold]Available quantization methods:[/bold]")
            console.print("1. GPTQ - Fast inference, good for GPU")
            console.print("2. GGUF - CPU-friendly, llama.cpp compatible")
            console.print("3. BitsAndBytes - Easy to use, good for experimentation")
            
            choice = Prompt.ask("Select quantization method", choices=["1", "2", "3"], default="2")
            quant_map = {"1": QuantizationType.GPTQ, "2": QuantizationType.GGUF, "3": QuantizationType.BITSANDBYTES}
            quant_type = quant_map[choice]
        
        # Bit width selection
        bit_width = parsed["bit_width"]
        if not bit_width:
            bit_width = int(Prompt.ask("Select bit width", choices=["4", "8", "16"], default="4"))
        
        # Target format based on quantization type
        format_map = {
            QuantizationType.GPTQ: TargetFormat.SAFETENSORS,
            QuantizationType.GGUF: TargetFormat.GGUF,
            QuantizationType.BITSANDBYTES: TargetFormat.PYTORCH
        }
        target_format = format_map[quant_type]
        
        # CPU fallback
        cpu_fallback = parsed["cpu_fallback"]
        if not cpu_fallback:
            cpu_fallback = Confirm.ask("Enable CPU fallback for inference?", default=True)
        
        # GPU memory limit
        gpu_memory = None
        if not cpu_fallback:
            gpu_memory = int(Prompt.ask("GPU memory limit (GB)", default="8"))
        
        plan = QuantizationPlan(
            model_name=model_name,
            quantization_type=quant_type,
            target_format=target_format,
            bit_width=bit_width,
            cpu_fallback=cpu_fallback,
            gpu_memory_limit=gpu_memory
        )
        
        # Display final plan
        self.display_plan(plan)
        
        if not Confirm.ask("Proceed with this plan?", default=True):
            console.print("[yellow]Planning cancelled.[/yellow]")
            return None
        
        return plan
    
    def display_plan(self, plan: QuantizationPlan):
        """Display the quantization plan"""
        plan_text = f"""
[bold]Model:[/bold] {plan.model_name}
[bold]Quantization:[/bold] {plan.quantization_type.value.upper()}
[bold]Target Format:[/bold] {plan.target_format.value.upper()}
[bold]Bit Width:[/bold] {plan.bit_width}-bit
[bold]CPU Fallback:[/bold] {plan.cpu_fallback}
[bold]GPU Memory Limit:[/bold] {plan.gpu_memory_limit}GB
        """
        console.print(Panel(plan_text.strip(), title="📋 Quantization Plan", style="green"))
    
    def save_plan(self, plan: QuantizationPlan, config_path: str):
        """Save quantization plan to configuration file"""
        config = {
            "model_name": plan.model_name,
            "quantization_type": plan.quantization_type.value,
            "target_format": plan.target_format.value,
            "bit_width": plan.bit_width,
            "cpu_fallback": plan.cpu_fallback,
            "gpu_memory_limit": plan.gpu_memory_limit,
            "specific_params": plan.specific_params
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        console.print(f"[green]Plan saved to {config_path}[/green]")
    
    def _show_quantization_compatibility(self, model_name: str):
        """Show quantization compatibility information for the model"""
        try:
            from utils.quantization_compatibility import get_quantization_suggestions
            
            console.print("\n[cyan]📊 Analyzing Model Compatibility...[/cyan]")
            suggestions = get_quantization_suggestions(model_name)
            console.print(Panel(suggestions, title="🎯 Quantization Recommendations", style="cyan"))
            
        except ImportError:
            console.print("[yellow]⚠️ Quantization compatibility checker not available[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not analyze model compatibility: {e}[/yellow]")

    
    # Phase 1: AI-Powered Intelligence - Enhanced planning capabilities
    def suggest_optimal_lora_config(self, model_name: str, task_type: str, 
                                   dataset_size: int, gpu_memory_gb: int = 8) -> Dict:
        """
        AI-powered LoRA configuration suggestions
        
        Args:
            model_name: Name of the base model to fine-tune
            task_type: Type of fine-tuning task (chat, classification, etc.)
            dataset_size: Number of training samples
            gpu_memory_gb: Available GPU memory in GB
            
        Returns:
            Dictionary with optimal LoRA configuration
        """
        console.print(f"[blue]🧠 Analyzing optimal LoRA config for {model_name}...[/blue]")
        
        try:
            if self.llm_manager.is_available():
                response = self.llm_manager.suggest_lora_config(
                    model_name, task_type, dataset_size, gpu_memory_gb
                )
                
                if response.success:
                    try:
                        # Try to parse JSON response
                        config_data = json.loads(response.content)
                        
                        console.print("[green]✅ AI-powered LoRA config generated![/green]")
                        return config_data
                        
                    except json.JSONDecodeError:
                        # Fall back to parsing text response
                        config = self._parse_lora_response(response.content)
                        console.print("[yellow]⚠️ Using parsed AI recommendations[/yellow]")
                        return config
                else:
                    console.print(f"[yellow]⚠️ AI analysis failed: {response.error_message}[/yellow]")
                    return self._fallback_lora_config(model_name, task_type, dataset_size, gpu_memory_gb)
            else:
                console.print("[yellow]⚠️ AI unavailable, using rule-based recommendations[/yellow]")
                return self._fallback_lora_config(model_name, task_type, dataset_size, gpu_memory_gb)
                
        except Exception as e:
            console.print(f"[red]❌ Error generating LoRA config: {e}[/red]")
            return self._fallback_lora_config(model_name, task_type, dataset_size, gpu_memory_gb)
    
    def predict_training_time(self, config: Dict) -> Dict:
        """
        Predict training duration and resource usage
        
        Args:
            config: Fine-tuning configuration dictionary
            
        Returns:
            Dictionary with training predictions
        """
        console.print("[blue]⏱️ Predicting training requirements...[/blue]")
        
        try:
            if self.llm_manager.is_available():
                model_name = config.get('base_model', 'unknown')
                response = self.llm_manager.predict_training_requirements(model_name, config)
                
                if response.success:
                    predictions = self._parse_training_predictions(response.content)
                    console.print("[green]✅ Training predictions generated![/green]")
                    return predictions
                else:
                    console.print(f"[yellow]⚠️ AI prediction failed: {response.error_message}[/yellow]")
                    return self._fallback_training_predictions(config)
            else:
                return self._fallback_training_predictions(config)
                
        except Exception as e:
            console.print(f"[red]❌ Error predicting training time: {e}[/red]")
            return self._fallback_training_predictions(config)
    
    def suggest_training_strategy(self, model_name: str, task_type: str, constraints: Dict) -> Dict:
        """
        Get comprehensive training strategy recommendations
        
        Args:
            model_name: Name of the model to fine-tune
            task_type: Type of fine-tuning task
            constraints: Hardware and other constraints
            
        Returns:
            Dictionary with training strategy recommendations
        """
        console.print("[blue]📋 Generating training strategy...[/blue]")
        
        try:
            if self.llm_manager.is_available():
                response = self.llm_manager.suggest_training_strategy(model_name, task_type, constraints)
                
                if response.success:
                    strategy = self._parse_training_strategy(response.content)
                    console.print("[green]✅ Training strategy generated![/green]")
                    return strategy
                else:
                    console.print(f"[yellow]⚠️ AI strategy failed: {response.error_message}[/yellow]")
                    return self._fallback_training_strategy(task_type, constraints)
            else:
                return self._fallback_training_strategy(task_type, constraints)
                
        except Exception as e:
            console.print(f"[red]❌ Error generating strategy: {e}[/red]")
            return self._fallback_training_strategy(task_type, constraints)
    
    def analyze_model_compatibility(self, model_name: str, task_type: str) -> Dict:
        """
        Analyze model compatibility for fine-tuning
        
        Args:
            model_name: Name of the model to analyze
            task_type: Type of fine-tuning task
            
        Returns:
            Dictionary with compatibility analysis
        """
        console.print(f"[blue]🔍 Analyzing {model_name} compatibility for {task_type}...[/blue]")
        
        compatibility = {
            "compatible": True,
            "confidence": 0.8,
            "recommendations": [],
            "warnings": [],
            "optimal_settings": {}
        }
        
        # Basic compatibility checks
        model_lower = model_name.lower()
        
        # Check for known incompatible models
        if "gpt-4" in model_lower or "claude" in model_lower:
            compatibility["compatible"] = False
            compatibility["warnings"].append("Proprietary model - fine-tuning not available")
            compatibility["confidence"] = 0.9
            return compatibility
        
        # Check model size compatibility
        if "70b" in model_lower or "65b" in model_lower:
            compatibility["warnings"].append("Large model - may require significant GPU memory")
            compatibility["recommendations"].append("Consider using QLoRA for memory efficiency")
        
        # Task-specific recommendations
        if task_type == "chat":
            compatibility["recommendations"].extend([
                "Use chat-based instruction templates",
                "Consider conversation turn formatting",
                "Monitor response coherence during training"
            ])
        elif task_type == "classification":
            compatibility["recommendations"].extend([
                "Ensure balanced class distribution",
                "Use appropriate loss functions",
                "Monitor classification metrics"
            ])
        
        console.print("[green]✅ Compatibility analysis complete![/green]")
        return compatibility
    
    def _parse_lora_response(self, response_content: str) -> Dict:
        """Parse AI response for LoRA configuration"""
        config = {
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'target_modules': ['q_proj', 'v_proj'],
            'learning_rate': 2e-4,
            'per_device_train_batch_size': 2,
            'gradient_accumulation_steps': 8,
            'num_train_epochs': 3,
            'estimated_memory_gb': 6.0,
            'estimated_training_hours': 2.0,
            'reasoning': 'Fallback configuration'
        }
        
        try:
            # Extract numeric values from response
            import re
            
            # Extract LoRA rank
            rank_match = re.search(r'(?:rank|lora_r).*?(\d+)', response_content.lower())
            if rank_match:
                config['lora_r'] = int(rank_match.group(1))
            
            # Extract alpha
            alpha_match = re.search(r'(?:alpha|lora_alpha).*?(\d+)', response_content.lower())
            if alpha_match:
                config['lora_alpha'] = int(alpha_match.group(1))
            
            # Extract learning rate
            lr_match = re.search(r'learning.rate.*?(\d+\.?\d*e?-?\d*)', response_content.lower())
            if lr_match:
                config['learning_rate'] = float(lr_match.group(1))
            
            # Extract batch size
            batch_match = re.search(r'batch.size.*?(\d+)', response_content.lower())
            if batch_match:
                config['per_device_train_batch_size'] = int(batch_match.group(1))
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not parse all AI recommendations: {e}[/yellow]")
        
        return config
    
    def _parse_training_predictions(self, response_content: str) -> Dict:
        """Parse AI response for training predictions"""
        predictions = {
            'estimated_memory_gb': 6.0,
            'training_time_per_epoch_minutes': 30,
            'total_training_hours': 2.0,
            'cpu_memory_gb': 8.0,
            'storage_gb': 5.0,
            'recommendations': ['Monitor GPU memory usage', 'Use gradient checkpointing'],
            'confidence': 0.7
        }
        
        try:
            import re
            
            # Extract memory usage
            memory_match = re.search(r'(\d+\.?\d*)\s*gb?\s*(?:gpu|memory)', response_content.lower())
            if memory_match:
                predictions['estimated_memory_gb'] = float(memory_match.group(1))
            
            # Extract training time
            time_match = re.search(r'(\d+\.?\d*)\s*(?:hours?|hrs?)', response_content.lower())
            if time_match:
                predictions['total_training_hours'] = float(time_match.group(1))
            
            # Extract recommendations
            lines = response_content.split('\n')
            recommendations = []
            for line in lines:
                if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'consider']):
                    clean_line = line.strip('- •*0123456789. ')
                    if clean_line and len(clean_line) > 10:
                        recommendations.append(clean_line)
            
            if recommendations:
                predictions['recommendations'] = recommendations[:5]  # Top 5 recommendations
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not parse all predictions: {e}[/yellow]")
        
        return predictions
    
    def _parse_training_strategy(self, response_content: str) -> Dict:
        """Parse AI response for training strategy"""
        strategy = {
            'training_phases': ['warm-up', 'main training', 'fine-tuning'],
            'learning_rate_schedule': 'cosine with warm-up',
            'regularization': ['dropout', 'weight decay'],
            'monitoring_metrics': ['loss', 'accuracy', 'perplexity'],
            'early_stopping': {'patience': 3, 'metric': 'validation_loss'},
            'checkpoint_strategy': 'save best and every epoch',
            'recommendations': ['Use gradient checkpointing', 'Monitor for overfitting'],
            'success_indicators': ['Decreasing validation loss', 'Improved task performance']
        }
        
        try:
            # Extract key strategy elements from AI response
            lines = response_content.lower().split('\n')
            
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect sections
                if 'phase' in line or 'training' in line:
                    current_section = 'training_phases'
                elif 'learning rate' in line or 'schedule' in line:
                    current_section = 'learning_rate_schedule'
                elif 'metric' in line or 'monitor' in line:
                    current_section = 'monitoring_metrics'
                elif 'recommend' in line:
                    current_section = 'recommendations'
                elif 'success' in line or 'indicator' in line:
                    current_section = 'success_indicators'
                elif line.startswith(('-', '•', '*', '1.', '2.')):
                    # Extract bullet points
                    item = line.lstrip('-•*0123456789. ')
                    if current_section and isinstance(strategy[current_section], list):
                        strategy[current_section].append(item.capitalize())
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not parse all strategy elements: {e}[/yellow]")
        
        return strategy
    
    def _fallback_lora_config(self, model_name: str, task_type: str, 
                             dataset_size: int, gpu_memory_gb: int) -> Dict:
        """Fallback LoRA configuration when AI is unavailable"""
        # Rule-based configuration based on model size and constraints
        model_lower = model_name.lower()
        
        # Determine model size category
        if "7b" in model_lower or "6b" in model_lower:
            base_config = {
                'lora_r': 16,
                'lora_alpha': 32,
                'learning_rate': 2e-4,
                'per_device_train_batch_size': 2 if gpu_memory_gb >= 8 else 1
            }
        elif "13b" in model_lower:
            base_config = {
                'lora_r': 32,
                'lora_alpha': 64,
                'learning_rate': 1e-4,
                'per_device_train_batch_size': 1
            }
        else:  # Default for unknown sizes
            base_config = {
                'lora_r': 16,
                'lora_alpha': 32,
                'learning_rate': 2e-4,
                'per_device_train_batch_size': 2
            }
        
        # Adjust for dataset size
        if dataset_size < 500:
            base_config['num_train_epochs'] = 5
            base_config['lora_dropout'] = 0.05
        elif dataset_size > 5000:
            base_config['num_train_epochs'] = 2
            base_config['lora_dropout'] = 0.1
        else:
            base_config['num_train_epochs'] = 3
            base_config['lora_dropout'] = 0.1
        
        # Task-specific adjustments
        if task_type == "classification":
            base_config['target_modules'] = ['q_proj', 'v_proj', 'o_proj']
        else:
            base_config['target_modules'] = ['q_proj', 'v_proj']
        
        # Memory optimization
        base_config['gradient_accumulation_steps'] = max(1, 16 // base_config['per_device_train_batch_size'])
        
        base_config.update({
            'estimated_memory_gb': gpu_memory_gb * 0.8,
            'estimated_training_hours': (dataset_size * base_config['num_train_epochs']) / 1000,
            'reasoning': 'Rule-based configuration'
        })
        
        return base_config
    
    def _fallback_training_predictions(self, config: Dict) -> Dict:
        """Fallback training predictions when AI is unavailable"""
        batch_size = config.get('per_device_train_batch_size', 2)
        epochs = config.get('num_train_epochs', 3)
        
        return {
            'estimated_memory_gb': 6.0 + (batch_size * 0.5),
            'training_time_per_epoch_minutes': max(20, batch_size * 10),
            'total_training_hours': max(1, epochs * batch_size * 0.5),
            'cpu_memory_gb': 8.0,
            'storage_gb': 3.0 + epochs,
            'recommendations': [
                'Use gradient checkpointing for memory efficiency',
                'Monitor GPU utilization',
                'Save checkpoints regularly'
            ],
            'confidence': 0.6
        }
    
    def _fallback_training_strategy(self, task_type: str, constraints: Dict) -> Dict:
        """Fallback training strategy when AI is unavailable"""
        return {
            'training_phases': ['warm-up (10%)', 'main training (80%)', 'fine-tuning (10%)'],
            'learning_rate_schedule': 'Linear warm-up then cosine decay',
            'regularization': ['LoRA dropout', 'gradient clipping'],
            'monitoring_metrics': ['training_loss', 'validation_loss', 'learning_rate'],
            'early_stopping': {'patience': 3, 'metric': 'validation_loss', 'min_delta': 0.001},
            'checkpoint_strategy': 'Save best model and every 500 steps',
            'recommendations': [
                'Start with conservative learning rates',
                'Use gradient accumulation for larger effective batch sizes',
                'Monitor for overfitting with small datasets'
            ],
            'success_indicators': [
                'Steady decrease in training loss',
                'Validation loss follows training loss',
                'No significant performance degradation'
            ]
        }
