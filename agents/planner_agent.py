import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
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
        # Use centralized LLM configuration
        from utils.llm_config import get_llm_manager
        self.llm_manager = get_llm_manager()
        
        # Keep backward compatibility for direct model access
        self.model = getattr(self.llm_manager.provider_instances.get(self.llm_manager.current_provider, {}), 'get', lambda x: None)('model')
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