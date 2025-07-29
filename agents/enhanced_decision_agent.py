#!/usr/bin/env python3
"""
Enhanced Decision Agent - LLM-first intelligent command routing and user intent understanding
Uses LLM as primary method for understanding user requests
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from rich.console import Console

# Import system paths for utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

console = Console()

class UserIntent(Enum):
    """Types of user intents"""
    QUANTIZE_MODEL = "quantize_model"
    DOWNLOAD_MODEL = "download_model"
    TEST_MODEL = "test_model" 
    HF_AUTHENTICATION = "hf_auth"
    SEARCH_MODELS = "search_models"
    DISCOVER_MODELS = "discover_models"
    LIST_EXPERIMENTS = "list_experiments"
    SHOW_STATISTICS = "show_statistics"
    SHOW_CONFIG = "show_config"
    SHOW_HELP = "show_help"
    GENERAL_QUESTION = "general_question"
    GREETING = "greeting"
    EXIT = "exit"
    UNKNOWN = "unknown"
    # API-related intents
    LOAD_MODEL_API = "load_model_api"
    UNLOAD_MODEL_API = "unload_model_api" 
    START_API_SERVER = "start_api_server"
    STOP_API_SERVER = "stop_api_server"
    SHOW_API_STATUS = "show_api_status"
    CHAT_WITH_MODEL = "chat_with_model"
    EXPERIMENT_WITH_MODEL = "experiment_with_model"

@dataclass
class IntentResult:
    """Result of intent analysis"""
    intent: UserIntent
    confidence: float
    extracted_info: Dict[str, Any]
    response_message: Optional[str] = None
    requires_llm_response: bool = False

class EnhancedDecisionAgent:
    """Enhanced Agent for understanding user intent using LLM-first approach"""
    
    def __init__(self):
        from utils.llm_config import get_llm_manager
        self.llm_manager = get_llm_manager()
        self.conversation_history = []
        
    def analyze_user_intent(self, user_input: str) -> IntentResult:
        """Analyze user input to determine intent using LLM-first approach"""
        user_input_lower = user_input.lower().strip()
        
        # Handle empty input
        if not user_input_lower:
            return IntentResult(
                intent=UserIntent.SHOW_HELP,
                confidence=1.0,
                extracted_info={},
                response_message="How can I help you? Type 'help' to see available commands."
            )
        
        # Try LLM first for natural language understanding
        if self.llm_manager.is_available():
            llm_result = self._llm_intent_analysis(user_input)
            if llm_result.confidence > 0.5:  # Trust LLM for any reasonable confidence
                return llm_result
        
        # Fallback to simple pattern matching for basic commands
        return self._simple_pattern_fallback(user_input_lower, user_input)
    
    def _llm_intent_analysis(self, user_input: str) -> IntentResult:
        """Use LLM to understand user intent with comprehensive prompt"""
        
        prompt = f"""
        You are an AI assistant for a model quantization tool called LQMF (Local Quantized Model Factory).
        
        Analyze the following user input and determine their intent:
        
        User input: "{user_input}"
        
        Available intents (choose the most appropriate one):
        
        1. DISCOVER_MODELS: User wants model recommendations/suggestions from HuggingFace
           - Examples: "suggest models to quantize", "recommend good models", "what models should I use", "find models for chat"
           
        2. QUANTIZE_MODEL: User wants to quantize/compress a specific model  
           - Examples: "quantize mistral 7B", "compress llama2 to 4-bit", "quantize microsoft/DialoGPT"
           
        3. SEARCH_MODELS: User wants to search/browse HuggingFace models
           - Examples: "search for chat models", "browse models", "find llama models"
           
        4. TEST_MODEL: User wants to test/benchmark a quantized model
           - Examples: "test my model", "benchmark performance", "run inference test"
           
        5. DOWNLOAD_MODEL: User wants to download a model from HuggingFace
           - Examples: "download mistral", "get model from huggingface", "fetch llama2"
           
        6. HF_AUTHENTICATION: User wants to login/logout/check HuggingFace auth
           - Examples: "hf login", "huggingface status", "logout from hf"
           
        7. LIST_EXPERIMENTS: User wants to see previous experiments/history
           - Examples: "show experiments", "list history", "previous quantizations"
           
        8. SHOW_STATISTICS: User wants to see stats/metrics/performance data  
           - Examples: "show stats", "performance metrics", "success rates"
           
        9. SHOW_CONFIG: User wants to see configuration/settings
           - Examples: "show config", "settings", "configuration"
           
        10. SHOW_HELP: User needs help/instructions
            - Examples: "help", "how to use", "what can you do"
            
        11. GENERAL_QUESTION: User has a question about quantization/AI/models
            - Examples: "what is quantization", "how does GPTQ work", "explain bit width"
            
        12. GREETING: User is greeting/saying hello
            - Examples: "hello", "hi", "good morning"
            
        13. EXIT: User wants to exit/quit
            - Examples: "exit", "quit", "bye", "goodbye"
            
        14. LOAD_MODEL_API: User wants to load a model for API serving
            - Examples: "load model for api", "serve model", "load mistral for endpoint"
            
        15. UNLOAD_MODEL_API: User wants to unload a model from API serving
            - Examples: "unload model", "stop serving model", "remove model from api"
            
        16. START_API_SERVER: User wants to start the API server
            - Examples: "start api server", "run server", "launch api"
            
        17. STOP_API_SERVER: User wants to stop the API server  
            - Examples: "stop api server", "shutdown server", "kill api"
            
        18. SHOW_API_STATUS: User wants to see API server status or loaded models
            - Examples: "api status", "show loaded models", "server status"
            
        19. CHAT_WITH_MODEL: User wants to chat with a loaded model
            - Examples: "chat with model", "talk to mistral", "ask the model"
            
        20. EXPERIMENT_WITH_MODEL: User wants to experiment/test with models
            - Examples: "experiment with model", "test model interactively", "benchmark model"
            
        21. UNKNOWN: Intent is unclear or doesn't match any category
        
        IMPORTANT DISTINCTIONS:
        - Use DISCOVER_MODELS when user asks for suggestions/recommendations about which models to use
        - Use QUANTIZE_MODEL only when user specifies a particular model to quantize
        - Use SEARCH_MODELS when user wants to search/browse without specific recommendations
        
        CRITICAL: Respond with ONLY a valid JSON object, no other text. Format:
        {{
            "intent": "INTENT_NAME",
            "confidence": 0.9,
            "model_name": "extracted_model_name_if_any",
            "quantization_method": "gptq|gguf|bitsandbytes|null",
            "bit_width": 4,
            "goals": "extracted_user_goals_for_discovery",
            "explanation": "Brief explanation of why you chose this intent"
        }}
        
        Rules:
        - confidence: float between 0.1-1.0
        - Use null for missing values, not empty strings
        - No markdown formatting, no explanation outside JSON
        """
        
        try:
            from utils.llm_config import QueryType
            response = self.llm_manager.query(prompt, QueryType.PLANNING)
            
            if response and response.success:
                # Clean and parse the JSON response
                try:
                    # Clean the response content
                    content = response.content.strip()
                    
                    # Remove markdown code blocks if present
                    if content.startswith('```'):
                        lines = content.split('\n')
                        content = '\n'.join(lines[1:-1])
                    
                    # Remove any leading/trailing text that isn't JSON
                    start_idx = content.find('{')
                    end_idx = content.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        content = content[start_idx:end_idx + 1]
                    
                    result_data = json.loads(content)
                    
                    intent_name = result_data.get('intent', 'UNKNOWN')
                    confidence = float(result_data.get('confidence', 0.5))
                    
                    # Convert intent name to enum
                    intent_mapping = {
                        'QUANTIZE_MODEL': UserIntent.QUANTIZE_MODEL,
                        'DOWNLOAD_MODEL': UserIntent.DOWNLOAD_MODEL,
                        'TEST_MODEL': UserIntent.TEST_MODEL,
                        'HF_AUTHENTICATION': UserIntent.HF_AUTHENTICATION,
                        'SEARCH_MODELS': UserIntent.SEARCH_MODELS,
                        'DISCOVER_MODELS': UserIntent.DISCOVER_MODELS,
                        'LIST_EXPERIMENTS': UserIntent.LIST_EXPERIMENTS,
                        'SHOW_STATISTICS': UserIntent.SHOW_STATISTICS,
                        'SHOW_CONFIG': UserIntent.SHOW_CONFIG,
                        'SHOW_HELP': UserIntent.SHOW_HELP,
                        'GENERAL_QUESTION': UserIntent.GENERAL_QUESTION,
                        'GREETING': UserIntent.GREETING,
                        'EXIT': UserIntent.EXIT,
                        'LOAD_MODEL_API': UserIntent.LOAD_MODEL_API,
                        'UNLOAD_MODEL_API': UserIntent.UNLOAD_MODEL_API,
                        'START_API_SERVER': UserIntent.START_API_SERVER,
                        'STOP_API_SERVER': UserIntent.STOP_API_SERVER,
                        'SHOW_API_STATUS': UserIntent.SHOW_API_STATUS,
                        'CHAT_WITH_MODEL': UserIntent.CHAT_WITH_MODEL,
                        'EXPERIMENT_WITH_MODEL': UserIntent.EXPERIMENT_WITH_MODEL,
                        'UNKNOWN': UserIntent.UNKNOWN
                    }
                    
                    intent = intent_mapping.get(intent_name, UserIntent.UNKNOWN)
                    
                    # Extract information
                    extracted_info = {}
                    if result_data.get('model_name'):
                        extracted_info['model_name'] = result_data['model_name']
                    if result_data.get('quantization_method'):
                        extracted_info['quantization_method'] = result_data['quantization_method']
                    if result_data.get('bit_width'):
                        extracted_info['bit_width'] = result_data['bit_width']
                    if result_data.get('goals'):
                        extracted_info['goals'] = result_data['goals']
                    if result_data.get('explanation'):
                        extracted_info['explanation'] = result_data['explanation']
                    
                    return IntentResult(
                        intent=intent,
                        confidence=confidence,
                        extracted_info=extracted_info
                    )
                    
                except json.JSONDecodeError:
                    # Silently fall back to pattern matching - this is normal behavior
                    return self._simple_pattern_fallback(user_input.lower(), user_input)
                    
        except Exception as e:
            console.print(f"[yellow]LLM analysis failed: {e}[/yellow]")
        
        # Fallback to pattern matching
        return self._simple_pattern_fallback(user_input.lower(), user_input)
    
    def _simple_pattern_fallback(self, user_input_lower: str, original_input: str) -> IntentResult:
        """Simple pattern matching fallback when LLM is not available"""
        
        # Basic exact command patterns
        simple_patterns = {
            UserIntent.EXIT: ['exit', 'quit', 'bye', 'goodbye'],
            UserIntent.SHOW_HELP: ['help', '?'],
            UserIntent.SHOW_STATISTICS: ['stats', 'statistics'],
            UserIntent.LIST_EXPERIMENTS: ['list', 'experiments', 'history'],
            UserIntent.SHOW_CONFIG: ['config', 'configuration', 'settings'],
            UserIntent.GREETING: ['hi', 'hello', 'hey']
        }
        
        # Check exact matches first
        for intent, keywords in simple_patterns.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return IntentResult(
                    intent=intent,
                    confidence=0.8,
                    extracted_info={}
                )
        
        # Check for model suggestions (key pattern to fix)
        suggestion_keywords = ['suggest', 'recommend', 'what should', 'which model', 'give.*model', 'show.*model', 'example.*model', 'model.*example']
        if any(re.search(keyword, user_input_lower) for keyword in suggestion_keywords):
            return IntentResult(
                intent=UserIntent.DISCOVER_MODELS,
                confidence=0.7,
                extracted_info={'goals': original_input}
            )
        
        # Check for quantization with model names
        if 'quantize' in user_input_lower and any(char in original_input for char in ['/', '-', '_']):
            # Likely has a model name
            return IntentResult(
                intent=UserIntent.QUANTIZE_MODEL,
                confidence=0.7,
                extracted_info={'model_name': self._extract_potential_model_name(original_input)}
            )
        
        # Default fallback
        return IntentResult(
            intent=UserIntent.UNKNOWN,
            confidence=0.3,
            extracted_info={},
            requires_llm_response=True
        )
    
    def _extract_potential_model_name(self, text: str) -> Optional[str]:
        """Extract potential model name from text"""
        # Look for patterns like org/model or model-name
        import re
        patterns = [
            r'([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)',  # org/model
            r'([a-zA-Z0-9_-]+)', # simple model name
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        return None
    
    def should_route_to_agent(self, intent_result: IntentResult) -> bool:
        """Determine if request should be routed to an agent"""
        return intent_result.intent != UserIntent.UNKNOWN and intent_result.confidence > 0.5
    
    def add_to_conversation_history(self, user_input: str, response: str):
        """Add interaction to conversation history"""
        self.conversation_history.append({
            'user': user_input,
            'assistant': response
        })
        
        # Keep only last 10 interactions
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def generate_conversational_response(self, user_input: str, intent_result: IntentResult) -> Optional[str]:
        """Generate conversational responses for unclear intents"""
        
        if intent_result.intent == UserIntent.UNKNOWN:
            # Try to provide helpful suggestions based on the input
            user_lower = user_input.lower()
            
            if any(word in user_lower for word in ['model', 'suggest', 'recommend', 'example']):
                return "I can help you discover and recommend models! Try: 'discover models for chat' or 'find models for code generation'"
            
            if any(word in user_lower for word in ['quantize', 'compress']):
                return "I can help you quantize models! Try: 'quantize <model-name>' or 'discover models to quantize'"
            
            if any(word in user_lower for word in ['test', 'benchmark']):
                return "I can help you test models! Try: 'test list' to see available models or 'test <model-name>'"
            
            return "I'm not sure what you'd like to do. Try 'help' to see available commands or describe what you want to accomplish."
        
        elif intent_result.intent == UserIntent.GENERAL_QUESTION:
            return "I'd be happy to answer your question! What would you like to know about model quantization?"
        
        elif intent_result.intent == UserIntent.GREETING:
            return "Hello! I'm here to help you with model quantization. What would you like to do today?"
        
        return None