"""
Centralized LLM Configuration and Provider Management
Supports multiple LLM providers: Gemini, Claude, OpenAI

All LLM parameters (models, timeouts, etc.) are configured here in one place.
config.json only specifies which provider to use.
"""

import os
import json
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    from rich.console import Console
    console = Console()
except ImportError:
    # Fallback if rich is not available
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console = Console()

class LLMProvider(Enum):
    GEMINI = "gemini"
    CLAUDE = "claude"
    OPENAI = "openai"

class QueryType(Enum):
    QUANTIZATION_SUGGESTION = "quantization_suggestion"
    COMPATIBILITY_CHECK = "compatibility_check"
    ERROR_ANALYSIS = "error_analysis"
    OPTIMIZATION_ADVICE = "optimization_advice"
    MODEL_RECOMMENDATION = "model_recommendation"
    PLANNING = "planning"
    GENERAL = "general"

@dataclass
class LLMResponse:
    success: bool
    content: Optional[str] = None
    error_message: Optional[str] = None
    query_type: Optional[QueryType] = None
    metadata: Optional[Dict] = None
    provider: Optional[LLMProvider] = None

@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    rate_limit_delay: float = 1.0
    request_timeout: int = 8
    max_retries: int = 3
    base_url: Optional[str] = None
    extra_params: Optional[Dict[str, Any]] = None

# Default configurations for each provider
DEFAULT_LLM_CONFIGS = {
    LLMProvider.GEMINI: LLMConfig(
        provider=LLMProvider.GEMINI,
        model_name="gemini-2.5-flash",
        rate_limit_delay=1.0,
        request_timeout=8,
        max_retries=3
    ),
    LLMProvider.CLAUDE: LLMConfig(
        provider=LLMProvider.CLAUDE,
        model_name="claude-3-sonnet-20240229",
        rate_limit_delay=1.0,
        request_timeout=8,
        max_retries=3
    ),
    LLMProvider.OPENAI: LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4",
        rate_limit_delay=1.0,
        request_timeout=8,
        max_retries=3
    )
}

class LLMManager:
    """Centralized LLM provider management"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.current_provider = None
        self.provider_instances = {}
        self.last_request_time = 0
        
        # Initialize based on configuration
        self._initialize_provider()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from config.json and create LLM config"""
        if config_path is None:
            current_dir = Path(__file__).parent
            config_path = current_dir.parent / "config.json"
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Get the provider choice from config.json
            llm_provider = config.get("llm_provider", "gemini")
            
            # Get the default config for this provider
            provider_enum = LLMProvider(llm_provider)
            llm_config = DEFAULT_LLM_CONFIGS[provider_enum]
            
            # Set API key from environment variables
            if provider_enum == LLMProvider.GEMINI:
                llm_config.api_key = os.getenv("GEMINI_API_KEY")
            elif provider_enum == LLMProvider.CLAUDE:
                llm_config.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif provider_enum == LLMProvider.OPENAI:
                llm_config.api_key = os.getenv("OPENAI_API_KEY")
            
            config["llm_config"] = llm_config
            return config
            
        except FileNotFoundError:
            console.print(f"[yellow]⚠️ Config file not found at {config_path}, using defaults[/yellow]")
            return {"llm_provider": "gemini", "llm_config": DEFAULT_LLM_CONFIGS[LLMProvider.GEMINI]}
        except json.JSONDecodeError:
            console.print(f"[yellow]⚠️ Invalid JSON in config file, using defaults[/yellow]")
            return {"llm_provider": "gemini", "llm_config": DEFAULT_LLM_CONFIGS[LLMProvider.GEMINI]}
        except ValueError as e:
            console.print(f"[yellow]⚠️ Invalid provider in config: {e}, using gemini[/yellow]")
            return {"llm_provider": "gemini", "llm_config": DEFAULT_LLM_CONFIGS[LLMProvider.GEMINI]}
    
    def _initialize_provider(self):
        """Initialize the configured LLM provider"""
        llm_config = self.config.get("llm_config")
        if not llm_config:
            console.print("[yellow]⚠️ No LLM configuration found, using offline mode[/yellow]")
            return
        
        provider = llm_config.provider
        
        try:
            if provider == LLMProvider.GEMINI:
                self._initialize_gemini(llm_config)
            elif provider == LLMProvider.CLAUDE:
                self._initialize_claude(llm_config)
            elif provider == LLMProvider.OPENAI:
                self._initialize_openai(llm_config)
            
            self.current_provider = provider
            console.print(f"[green]✅ {provider.value.title()} initialized successfully with {llm_config.model_name}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to initialize {provider.value}: {e}[/red]")
            self.current_provider = None
    
    def _initialize_gemini(self, config: LLMConfig):
        """Initialize Gemini provider"""
        if not config.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.api_key)
            model = genai.GenerativeModel(config.model_name)
            
            self.provider_instances[LLMProvider.GEMINI] = {
                "model": model,
                "config": config
            }
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    
    def _initialize_claude(self, config: LLMConfig):
        """Initialize Claude provider"""
        if not config.api_key:
            raise ValueError("Claude API key not provided. Set ANTHROPIC_API_KEY environment variable.")
        
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=config.api_key)
            
            self.provider_instances[LLMProvider.CLAUDE] = {
                "client": client,
                "config": config
            }
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    def _initialize_openai(self, config: LLMConfig):
        """Initialize OpenAI provider"""
        if not config.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        try:
            import openai
            client = openai.OpenAI(api_key=config.api_key)
            
            self.provider_instances[LLMProvider.OPENAI] = {
                "client": client,
                "config": config
            }
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    def is_available(self) -> bool:
        """Check if any LLM provider is available"""
        return self.current_provider is not None
    
    def get_provider(self) -> Optional[LLMProvider]:
        """Get current provider"""
        return self.current_provider
    
    def _rate_limit(self):
        """Implement basic rate limiting"""
        if not self.current_provider:
            return
        
        config = self.provider_instances[self.current_provider]["config"]
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < config.rate_limit_delay:
            sleep_time = config.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, prompt: str, max_retries: Optional[int] = None) -> LLMResponse:
        """Make a request to the current LLM provider with retries"""
        if not self.current_provider:
            return LLMResponse(
                success=False,
                error_message="No LLM provider initialized"
            )
        
        provider_data = self.provider_instances[self.current_provider]
        config = provider_data["config"]
        
        if max_retries is None:
            max_retries = config.max_retries
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                if self.current_provider == LLMProvider.GEMINI:
                    response = self._make_gemini_request(provider_data, prompt, config.request_timeout)
                elif self.current_provider == LLMProvider.CLAUDE:
                    response = self._make_claude_request(provider_data, prompt, config.request_timeout)
                elif self.current_provider == LLMProvider.OPENAI:
                    response = self._make_openai_request(provider_data, prompt, config.request_timeout)
                else:
                    return LLMResponse(
                        success=False,
                        error_message=f"Unsupported provider: {self.current_provider}"
                    )
                
                if response:
                    return LLMResponse(
                        success=True,
                        content=response,
                        provider=self.current_provider
                    )
                else:
                    return LLMResponse(
                        success=False,
                        error_message=f"Empty response from {self.current_provider.value}",
                        provider=self.current_provider
                    )
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return LLMResponse(
                        success=False,
                        error_message=f"Failed after {max_retries} attempts: {str(e)}",
                        provider=self.current_provider
                    )
                else:
                    console.print(f"[yellow]Retry {attempt + 1}/{max_retries} after error: {e}[/yellow]")
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return LLMResponse(success=False, error_message="Unexpected error")
    
    def _make_gemini_request(self, provider_data: Dict, prompt: str, timeout: int) -> Optional[str]:
        """Make request to Gemini"""
        model = provider_data["model"]
        
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        with ThreadPoolExecutor() as executor:
            future = executor.submit(model.generate_content, prompt)
            try:
                response = future.result(timeout=timeout)
                return response.text if response.text else None
            except TimeoutError:
                raise Exception(f"API call timed out after {timeout} seconds")
    
    def _make_claude_request(self, provider_data: Dict, prompt: str, timeout: int) -> Optional[str]:
        """Make request to Claude"""
        client = provider_data["client"]
        config = provider_data["config"]
        
        message = client.messages.create(
            model=config.model_name,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text if message.content else None
    
    def _make_openai_request(self, provider_data: Dict, prompt: str, timeout: int) -> Optional[str]:
        """Make request to OpenAI"""
        client = provider_data["client"]
        config = provider_data["config"]
        
        response = client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        
        return response.choices[0].message.content if response.choices else None
    
    def query(self, prompt: str, query_type: QueryType = QueryType.GENERAL) -> LLMResponse:
        """Generic query method for any prompt"""
        response = self._make_request(prompt)
        response.query_type = query_type
        return response
    
    # Convenience methods for common operations
    def suggest_quantization_method(self, model_name: str, use_case: str, 
                                  hardware_constraints: Dict[str, Any]) -> LLMResponse:
        """Get quantization method suggestions"""
        prompt = f"""
        As an expert in model quantization, please recommend the best quantization approach for:
        
        Model: {model_name}
        Use Case: {use_case}
        Hardware Constraints: {json.dumps(hardware_constraints, indent=2)}
        
        Please provide specific recommendations for:
        1. Quantization method (GPTQ, GGUF, BitsAndBytes)
        2. Bit width (4-bit, 8-bit, etc.)
        3. Target format (safetensors, GGUF, pytorch)
        4. Expected performance trade-offs
        5. Memory requirements
        
        Format your response as practical, actionable advice focusing on the optimal choice for the given constraints.
        Keep it concise but informative.
        """
        
        response = self._make_request(prompt)
        response.query_type = QueryType.QUANTIZATION_SUGGESTION
        return response
    
    def check_model_compatibility(self, model_name: str, quantization_method: str, 
                                 target_format: str) -> LLMResponse:
        """Check model compatibility with quantization method"""
        prompt = f"""
        Please analyze the compatibility of this quantization setup:
        
        Model: {model_name}
        Quantization Method: {quantization_method}
        Target Format: {target_format}
        
        Provide information about:
        1. Compatibility status (fully compatible / partially compatible / incompatible)
        2. Any known issues or limitations
        3. Alternative approaches if incompatible
        4. Expected model size reduction
        5. Performance characteristics
        
        Base your analysis on the model architecture and the quantization method's capabilities.
        Provide a clear recommendation.
        """
        
        response = self._make_request(prompt)
        response.query_type = QueryType.COMPATIBILITY_CHECK
        return response

# Global instance
_llm_manager = None

def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager

def change_provider(provider: str):
    """Change the LLM provider and reinitialize"""
    global _llm_manager
    
    # Validate provider
    try:
        LLMProvider(provider)
    except ValueError:
        console.print(f"[red]❌ Invalid provider: {provider}. Must be one of: gemini, claude, openai[/red]")
        return
    
    # Update config file
    config_path = Path(__file__).parent.parent / "config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config["llm_provider"] = provider
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Reinitialize manager
        _llm_manager = LLMManager()
        console.print(f"[green]✅ Switched to {provider}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to change provider: {e}[/red]")

def get_current_config() -> Optional[LLMConfig]:
    """Get the current LLM configuration"""
    manager = get_llm_manager()
    if manager.current_provider:
        return manager.provider_instances[manager.current_provider]["config"]
    return None

def update_model_for_provider(provider: str, model_name: str):
    """Update the model name for a specific provider"""
    try:
        provider_enum = LLMProvider(provider)
        DEFAULT_LLM_CONFIGS[provider_enum].model_name = model_name
        console.print(f"[green]✅ Updated {provider} model to {model_name}[/green]")
        
        # If this is the current provider, reinitialize
        manager = get_llm_manager()
        if manager.current_provider == provider_enum:
            global _llm_manager
            _llm_manager = LLMManager()
            console.print(f"[green]✅ Reinitialized {provider} with new model[/green]")
            
    except ValueError:
        console.print(f"[red]❌ Invalid provider: {provider}[/red]")

# Backward compatibility functions
def get_quantization_suggestion(model_name: str, use_case: str, hardware_constraints: Dict[str, Any]) -> Optional[str]:
    """Convenience function to get quantization suggestions"""
    manager = get_llm_manager()
    if not manager.is_available():
        return None
    
    response = manager.suggest_quantization_method(model_name, use_case, hardware_constraints)
    return response.content if response.success else None

def analyze_error(error_message: str, model_name: str, config: Dict[str, Any]) -> Optional[str]:
    """Convenience function to analyze errors"""
    manager = get_llm_manager()
    if not manager.is_available():
        return None
    
    prompt = f"""
    A quantization process failed with the following error:
    
    Error: {error_message}
    Model: {model_name}
    Configuration: {json.dumps(config, indent=2)}
    
    Please analyze this error and provide:
    1. Root cause analysis
    2. Specific remediation steps
    3. Alternative configurations to try
    4. Prevention strategies for similar errors
    5. Whether this is a recoverable error
    
    Focus on actionable solutions that can be implemented immediately.
    """
    
    response = manager.query(prompt, QueryType.ERROR_ANALYSIS)
    return response.content if response.success else None

def check_compatibility(model_name: str, quantization_method: str, target_format: str) -> Optional[str]:
    """Convenience function to check compatibility"""
    manager = get_llm_manager()
    if not manager.is_available():
        return None
    
    response = manager.check_model_compatibility(model_name, quantization_method, target_format)
    return response.content if response.success else None