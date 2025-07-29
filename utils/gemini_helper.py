"""
Gemini API Helper Functions
Provides utility functions for interacting with Google's Gemini API
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import google.generativeai as genai
from rich.console import Console

console = Console()

class QueryType(Enum):
    QUANTIZATION_SUGGESTION = "quantization_suggestion"
    COMPATIBILITY_CHECK = "compatibility_check"
    ERROR_ANALYSIS = "error_analysis"
    OPTIMIZATION_ADVICE = "optimization_advice"
    MODEL_RECOMMENDATION = "model_recommendation"
    PLANNING = "planning"
    GENERAL = "general"

@dataclass
class GeminiResponse:
    success: bool
    content: Optional[str] = None
    error_message: Optional[str] = None
    query_type: Optional[QueryType] = None
    metadata: Optional[Dict] = None

class GeminiHelper:
    """Helper class for Gemini API interactions"""
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize API key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or self.config.get("gemini_api_key")
        
        # Initialize model configuration
        self.model_name = self.config.get("gemini_model", "gemini-2.0-flash-exp")
        self.rate_limit_delay = self.config.get("gemini_rate_limit_delay", 1.0)
        self.request_timeout = self.config.get("gemini_request_timeout", 8)
        self.max_retries = self.config.get("gemini_max_retries", 3)
        
        self.model = None
        self.last_request_time = 0
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                console.print(f"[green]✅ Gemini API initialized successfully with {self.model_name}[/green]")
            except Exception as e:
                console.print(f"[red]❌ Failed to initialize Gemini API: {e}[/red]")
                self.model = None
        else:
            console.print("[yellow]⚠️ No Gemini API key provided. Running in offline mode.[/yellow]")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from config.json"""
        if config_path is None:
            # Try to find config.json in the project root
            current_dir = Path(__file__).parent
            config_path = current_dir.parent / "config.json"
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            console.print(f"[yellow]⚠️ Config file not found at {config_path}, using defaults[/yellow]")
            return {}
        except json.JSONDecodeError:
            console.print(f"[yellow]⚠️ Invalid JSON in config file, using defaults[/yellow]")
            return {}
    
    def is_available(self) -> bool:
        """Check if Gemini API is available"""
        return self.model is not None
    
    def _rate_limit(self):
        """Implement basic rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, prompt: str, max_retries: Optional[int] = None) -> GeminiResponse:
        """Make a request to Gemini API with retries"""
        if not self.model:
            return GeminiResponse(
                success=False,
                error_message="Gemini API not initialized"
            )
        
        # Use configured max_retries if not specified
        if max_retries is None:
            max_retries = self.max_retries
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                # Add timeout to the API call using configured timeout
                from concurrent.futures import ThreadPoolExecutor, TimeoutError
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(self.model.generate_content, prompt)
                    try:
                        response = future.result(timeout=self.request_timeout)
                    except TimeoutError:
                        raise Exception(f"API call timed out after {self.request_timeout} seconds")
                
                if response.text:
                    return GeminiResponse(
                        success=True,
                        content=response.text
                    )
                else:
                    return GeminiResponse(
                        success=False,
                        error_message="Empty response from Gemini"
                    )
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return GeminiResponse(
                        success=False,
                        error_message=f"Failed after {max_retries} attempts: {str(e)}"
                    )
                else:
                    console.print(f"[yellow]Retry {attempt + 1}/{max_retries} after error: {e}[/yellow]")
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return GeminiResponse(success=False, error_message="Unexpected error")
    
    def suggest_quantization_method(self, model_name: str, use_case: str, 
                                  hardware_constraints: Dict[str, Any]) -> GeminiResponse:
        """Get quantization method suggestions from Gemini"""
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
                                 target_format: str) -> GeminiResponse:
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
    
    def analyze_quantization_error(self, error_message: str, model_name: str, 
                                  quantization_config: Dict[str, Any]) -> GeminiResponse:
        """Analyze quantization errors and suggest fixes"""
        prompt = f"""
        A quantization process failed with the following error:
        
        Error: {error_message}
        Model: {model_name}
        Configuration: {json.dumps(quantization_config, indent=2)}
        
        Please analyze this error and provide:
        1. Root cause analysis
        2. Specific remediation steps
        3. Alternative configurations to try
        4. Prevention strategies for similar errors
        5. Whether this is a recoverable error
        
        Focus on actionable solutions that can be implemented immediately.
        """
        
        response = self._make_request(prompt)
        response.query_type = QueryType.ERROR_ANALYSIS
        return response
    
    def get_optimization_advice(self, model_name: str, current_config: Dict[str, Any], 
                               performance_metrics: Dict[str, Any]) -> GeminiResponse:
        """Get optimization advice based on current performance"""
        prompt = f"""
        Based on the current quantization results, please provide optimization advice:
        
        Model: {model_name}
        Current Configuration: {json.dumps(current_config, indent=2)}
        Performance Metrics: {json.dumps(performance_metrics, indent=2)}
        
        Please suggest optimizations for:
        1. Inference speed improvement
        2. Memory usage reduction
        3. Model quality preservation
        4. Alternative quantization approaches
        5. Hardware-specific optimizations
        
        Prioritize suggestions that will have the biggest impact on the specific metrics provided.
        """
        
        response = self._make_request(prompt)
        response.query_type = QueryType.OPTIMIZATION_ADVICE
        return response
    
    def recommend_similar_models(self, model_name: str, requirements: Dict[str, Any]) -> GeminiResponse:
        """Recommend similar models that might be better suited"""
        prompt = f"""
        The user is working with {model_name} but may want alternatives. Based on these requirements:
        
        Requirements: {json.dumps(requirements, indent=2)}
        
        Please recommend:
        1. Similar models with better quantization characteristics
        2. Smaller models that might meet the same use case
        3. Models specifically optimized for the target hardware
        4. Trade-offs between model choice and quantization approach
        5. Ranking of recommendations with justification
        
        Focus on models that are publicly available on Hugging Face.
        Consider both performance and resource efficiency.
        """
        
        response = self._make_request(prompt)
        response.query_type = QueryType.MODEL_RECOMMENDATION
        return response
    
    def explain_quantization_concepts(self, concept: str, user_level: str = "intermediate") -> GeminiResponse:
        """Explain quantization concepts to users"""
        prompt = f"""
        Please explain the quantization concept: {concept}
        
        Target audience: {user_level} level user
        
        Provide:
        1. Clear definition and explanation
        2. Practical examples
        3. When to use this approach
        4. Pros and cons
        5. Relationship to other quantization methods
        
        Keep the explanation accessible but technically accurate.
        Use examples relevant to model deployment and inference.
        """
        
        response = self._make_request(prompt)
        return response
    
    def generate_quantization_plan(self, user_request: str, system_capabilities: Dict[str, Any]) -> GeminiResponse:
        """Generate a complete quantization plan from user request"""
        prompt = f"""
        User Request: "{user_request}"
        System Capabilities: {json.dumps(system_capabilities, indent=2)}
        
        Generate a comprehensive quantization plan including:
        1. Parsed user requirements
        2. Recommended quantization approach
        3. Step-by-step execution plan
        4. Expected outcomes and metrics
        5. Fallback options if primary approach fails
        6. Resource requirements and timeline
        
        Format as a structured plan that can guide the quantization process.
        Be specific about technical choices and justify recommendations.
        """
        
        response = self._make_request(prompt)
        return response
    
    def validate_user_input(self, user_input: str) -> Dict[str, Any]:
        """Parse and validate user input for quantization requests"""
        if not self.is_available():
            return {"valid": False, "reason": "Gemini API not available"}
        
        prompt = f"""
        Parse this quantization request and extract structured information:
        
        User Input: "{user_input}"
        
        Extract and return as JSON:
        {{
            "model_name": "extracted model name or null",
            "quantization_method": "gptq/gguf/bitsandbytes or null",
            "bit_width": "4/8/16 or null",
            "target_format": "gguf/safetensors/pytorch or null",
            "cpu_fallback": "true/false or null",
            "use_case": "extracted use case description",
            "hardware_mentioned": "any hardware constraints mentioned",
            "ambiguous_parts": ["list of unclear requirements"],
            "confidence": "high/medium/low",
            "suggestions": ["clarifying questions to ask user"]
        }}
        
        Only return valid JSON, no additional text.
        """
        
        response = self._make_request(prompt)
        
        if response.success:
            try:
                parsed = json.loads(response.content)
                parsed["valid"] = True
                return parsed
            except json.JSONDecodeError:
                return {"valid": False, "reason": "Failed to parse Gemini response"}
        else:
            return {"valid": False, "reason": response.error_message}
    
    def query(self, prompt: str, query_type: QueryType = QueryType.GENERAL) -> GeminiResponse:
        """Generic query method for any prompt"""
        response = self._make_request(prompt)
        response.query_type = query_type
        return response

# Convenience functions for common operations
def get_quantization_suggestion(model_name: str, use_case: str, hardware_constraints: Dict[str, Any]) -> Optional[str]:
    """Convenience function to get quantization suggestions"""
    helper = GeminiHelper()
    if not helper.is_available():
        return None
    
    response = helper.suggest_quantization_method(model_name, use_case, hardware_constraints)
    return response.content if response.success else None

def analyze_error(error_message: str, model_name: str, config: Dict[str, Any]) -> Optional[str]:
    """Convenience function to analyze errors"""
    helper = GeminiHelper()
    if not helper.is_available():
        return None
    
    response = helper.analyze_quantization_error(error_message, model_name, config)
    return response.content if response.success else None

def check_compatibility(model_name: str, quantization_method: str, target_format: str) -> Optional[str]:
    """Convenience function to check compatibility"""
    helper = GeminiHelper()
    if not helper.is_available():
        return None
    
    response = helper.check_model_compatibility(model_name, quantization_method, target_format)
    return response.content if response.success else None