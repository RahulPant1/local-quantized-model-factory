"""
Conversational Quantization Copilot Agent
Unified natural language interface for all LQMF functionality.

This agent consolidates quantization, fine-tuning, and API serving
into a single conversational experience with multi-turn planning
and intelligent context management.
"""

import json
import re
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging

# Set up logging and console
console = Console()
logger = logging.getLogger(__name__)

class ConversationMode(Enum):
    """Different conversation modes for specialized workflows"""
    GENERAL = "general"              # General LQMF assistance
    QUANTIZATION = "quantization"    # Focused on quantization tasks
    FINETUNING = "finetuning"       # Focused on fine-tuning workflows
    API_SERVING = "api_serving"      # Focused on API serving and deployment
    EXPLORATION = "exploration"      # Model discovery and experimentation
    TUTORIAL = "tutorial"           # Guided learning and tutorials

class ConversationContext(Enum):
    """Context types for conversation management"""
    MODEL_SELECTION = "model_selection"
    QUANTIZATION_PLANNING = "quantization_planning"
    FINETUNING_SETUP = "finetuning_setup"
    API_DEPLOYMENT = "api_deployment"
    TROUBLESHOOTING = "troubleshooting"
    PERFORMANCE_ANALYSIS = "performance_analysis"

@dataclass
class SessionState:
    """Maintains conversation state across interactions"""
    session_id: str
    current_mode: ConversationMode
    active_context: Optional[ConversationContext]
    conversation_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    current_model: Optional[str]
    active_workflow: Optional[str]
    pending_actions: List[Dict[str, Any]]
    last_results: Dict[str, Any]

@dataclass
class ConversationIntent:
    """Enhanced intent recognition for conversational flows"""
    primary_intent: str
    sub_intents: List[str]
    entities: Dict[str, Any]
    confidence: float
    suggested_mode: ConversationMode
    context_needed: List[ConversationContext]
    multi_turn_flow: bool
    urgency: str  # low, medium, high

@dataclass
class CopilotResponse:
    """Structured response from the copilot"""
    response_text: str
    actions_taken: List[Dict[str, Any]]
    suggestions: List[str]
    next_steps: List[str]
    mode_change: Optional[ConversationMode]
    requires_confirmation: bool
    error_occurred: bool
    tutorial_triggered: bool

class ConversationalCopilotAgent:
    """
    Unified conversational interface for all LQMF functionality.
    
    Provides natural language interaction with multi-turn planning,
    intelligent context management, and seamless workflow integration.
    """
    
    def __init__(self):
        """Initialize the Conversational Copilot"""
        try:
            from utils.llm_config import get_llm_manager
            self.llm_manager = get_llm_manager()
        except Exception as e:
            console.print(f"[yellow]⚠️ LLM manager initialization failed: {e}[/yellow]")
            console.print("[yellow]⚠️ Some AI-powered features may be limited[/yellow]")
            self.llm_manager = None
        
        # Import all existing agents
        self._initialize_agents()
        
        # Session management
        self.current_session = SessionState(
            session_id=self._generate_session_id(),
            current_mode=ConversationMode.GENERAL,
            active_context=None,
            conversation_history=[],
            user_preferences={},
            current_model=None,
            active_workflow=None,
            pending_actions=[],
            last_results={}
        )
        
        # Conversation patterns and flows
        self.conversation_patterns = self._load_conversation_patterns()
        self.tutorial_flows = self._load_tutorial_flows()
        
        console.print("[green]🤖 Conversational Copilot initialized![/green]")
    
    def _initialize_agents(self):
        """Initialize all existing LQMF agents"""
        try:
            from agents.enhanced_decision_agent import EnhancedDecisionAgent
            from agents.planner_agent import PlannerAgent
            from agents.executor_agent import ExecutorAgent
            from agents.memory_agent import MemoryAgent
            from agents.feedback_agent import FeedbackAgent
            from agents.finetuning_agent import FineTuningAgent
            from agents.api_server_agent import APIServerAgent
            from agents.dataset_intelligence_agent import DatasetIntelligenceAgent
            
            # Core agents
            self.decision_agent = EnhancedDecisionAgent()
            self.planner_agent = PlannerAgent()
            self.executor_agent = ExecutorAgent()
            self.memory_agent = MemoryAgent()
            self.feedback_agent = FeedbackAgent()
            
            # Specialized agents
            self.finetuning_agent = FineTuningAgent()
            self.api_server_agent = APIServerAgent()
            self.dataset_agent = DatasetIntelligenceAgent()
            
            console.print("[dim]✅ All agents loaded successfully[/dim]")
            
        except Exception as e:
            console.print(f"[red]⚠️ Error initializing agents: {e}[/red]")
            logger.error(f"Agent initialization failed: {e}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier"""
        import uuid
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"copilot_{timestamp}_{short_uuid}"
    
    def _load_conversation_patterns(self) -> Dict[str, Any]:
        """Load conversation patterns for multi-turn flows"""
        return {
            "quantization_flow": {
                "steps": ["model_selection", "method_choice", "parameter_tuning", "execution", "testing"],
                "decision_points": ["confirm_model", "confirm_method", "confirm_parameters"],
                "fallback_actions": ["suggest_alternatives", "explain_options", "show_examples"]
            },
            "finetuning_flow": {
                "steps": ["dataset_analysis", "config_generation", "training_setup", "execution", "evaluation"],
                "decision_points": ["confirm_dataset", "confirm_config", "confirm_training"],
                "fallback_actions": ["analyze_dataset", "suggest_config", "explain_process"]
            },
            "api_deployment_flow": {
                "steps": ["model_loading", "server_setup", "endpoint_testing", "deployment_verification"],
                "decision_points": ["confirm_model", "confirm_config", "confirm_deployment"],
                "fallback_actions": ["list_models", "show_status", "test_endpoint"]
            },
            "exploration_flow": {
                "steps": ["goal_understanding", "model_discovery", "compatibility_analysis", "recommendations"],
                "decision_points": ["confirm_goals", "select_models", "proceed_with_action"],
                "fallback_actions": ["clarify_goals", "show_examples", "suggest_alternatives"]
            }
        }
    
    def _load_tutorial_flows(self) -> Dict[str, Any]:
        """Load interactive tutorial flows"""
        return {
            "beginner_quantization": {
                "title": "Getting Started with Model Quantization",
                "steps": [
                    {"type": "explanation", "content": "Model quantization reduces model size while preserving performance"},
                    {"type": "demonstration", "content": "Let's quantize a small model step by step"},
                    {"type": "hands_on", "content": "Try quantizing microsoft/DialoGPT-small"},
                    {"type": "analysis", "content": "Understanding the results and metrics"}
                ],
                "estimated_time": "10-15 minutes"
            },
            "advanced_finetuning": {
                "title": "Advanced Fine-tuning with LoRA",
                "steps": [
                    {"type": "explanation", "content": "Understanding LoRA and parameter-efficient fine-tuning"},
                    {"type": "dataset_prep", "content": "Preparing your dataset for fine-tuning"},
                    {"type": "config_optimization", "content": "Optimizing LoRA configuration for your use case"},
                    {"type": "training", "content": "Running the fine-tuning process"},
                    {"type": "evaluation", "content": "Evaluating and deploying your fine-tuned model"}
                ],
                "estimated_time": "30-45 minutes"
            },
            "api_deployment": {
                "title": "Deploying Models as APIs",
                "steps": [
                    {"type": "explanation", "content": "Understanding model serving and API deployment"},
                    {"type": "model_prep", "content": "Preparing quantized models for serving"},
                    {"type": "server_setup", "content": "Setting up the API server"},
                    {"type": "testing", "content": "Testing endpoints and performance"},
                    {"type": "production", "content": "Production deployment considerations"}
                ],
                "estimated_time": "20-30 minutes"
            }
        }
    
    async def process_conversation(self, user_input: str) -> CopilotResponse:
        """
        Main conversation processing method with enhanced natural language understanding
        """
        try:
            # Add to conversation history
            self.current_session.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": self._get_timestamp(),
                "mode": self.current_session.current_mode.value
            })
            
            # Enhanced intent recognition
            intent = await self._analyze_conversational_intent(user_input)
            
            # Determine if we need mode switching
            if intent.suggested_mode != self.current_session.current_mode:
                if await self._should_switch_mode(intent):
                    await self._switch_conversation_mode(intent.suggested_mode)
            
            # Process based on current mode and intent
            response = await self._process_mode_specific_conversation(intent, user_input)
            
            # Add response to history
            self.current_session.conversation_history.append({
                "role": "assistant", 
                "content": response.response_text,
                "timestamp": self._get_timestamp(),
                "mode": self.current_session.current_mode.value,
                "actions": response.actions_taken
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Conversation processing error: {e}")
            return CopilotResponse(
                response_text=f"I encountered an error processing your request: {e}. Let me help you get back on track.",
                actions_taken=[],
                suggestions=["Try rephrasing your request", "Ask for help with specific tasks"],
                next_steps=["Type 'help' for available commands"],
                mode_change=None,
                requires_confirmation=False,
                error_occurred=True,
                tutorial_triggered=False
            )
    
    async def _analyze_conversational_intent(self, user_input: str) -> ConversationIntent:
        """Enhanced intent analysis with conversational context"""
        
        # Use existing decision agent for basic intent recognition
        basic_intent = self.decision_agent.analyze_user_intent(user_input)
        
        # Enhance with conversational AI analysis
        if self.llm_manager.is_available():
            enhanced_intent = await self._llm_enhanced_intent_analysis(user_input, basic_intent)
            return enhanced_intent
        
        # Fallback to rule-based enhancement
        return self._rule_based_intent_enhancement(user_input, basic_intent)
    
    async def _llm_enhanced_intent_analysis(self, user_input: str, basic_intent) -> ConversationIntent:
        """LLM-powered conversational intent analysis"""
        
        # Check if LLM manager is available
        if not self.llm_manager or not self.llm_manager.current_provider:
            console.print("[yellow]⚠️ LLM not available, using rule-based analysis[/yellow]")
            return self._rule_based_intent_enhancement(user_input, basic_intent)
        
        # Build context-aware prompt
        context_info = self._build_conversation_context()
        
        prompt = f"""
        You are the LQMF Conversational Copilot, analyzing user intent in a multi-turn conversation about model quantization, fine-tuning, and API serving.

        Current Session Context:
        - Mode: {self.current_session.current_mode.value}
        - Active Context: {self.current_session.active_context.value if self.current_session.active_context else 'None'}
        - Current Model: {self.current_session.current_model or 'None'}
        - Recent History: {json.dumps(self.current_session.conversation_history[-3:], indent=2) if self.current_session.conversation_history else 'None'}

        User Input: "{user_input}"
        Basic Intent: {basic_intent.intent.value if hasattr(basic_intent, 'intent') else 'unknown'}

        Analyze this in the conversational context and respond with JSON:
        {{
            "primary_intent": "quantize|finetune|serve_api|explore|help|configure|troubleshoot|learn",
            "sub_intents": ["list of related sub-intents"],
            "entities": {{
                "model_name": "extracted model name or null",
                "method": "quantization/training method or null", 
                "parameters": {{"key": "value"}},
                "goals": "user goals description"
            }},
            "confidence": 0.85,
            "suggested_mode": "general|quantization|finetuning|api_serving|exploration|tutorial",
            "context_needed": ["model_selection", "quantization_planning", etc],
            "multi_turn_flow": true,
            "urgency": "low|medium|high",
            "conversation_cues": {{
                "is_follow_up": true,
                "references_previous": true,
                "needs_clarification": false,
                "expresses_frustration": false,
                "shows_learning_intent": true
            }},
            "response_tone": "helpful|educational|technical|encouraging",
            "suggested_actions": ["immediate actions to take"],
            "reasoning": "brief explanation of analysis"
        }}
        """
        
        try:
            from utils.llm_config import QueryType
            response = self.llm_manager.query(prompt, QueryType.PLANNING)
            
            if response and response.success:
                try:
                    result_data = json.loads(response.content.strip())
                    
                    return ConversationIntent(
                        primary_intent=result_data.get('primary_intent', 'help'),
                        sub_intents=result_data.get('sub_intents', []),
                        entities=result_data.get('entities', {}),
                        confidence=float(result_data.get('confidence', 0.6)),
                        suggested_mode=ConversationMode(result_data.get('suggested_mode', 'general')),
                        context_needed=[ConversationContext(ctx) for ctx in result_data.get('context_needed', []) if ctx in [c.value for c in ConversationContext]],
                        multi_turn_flow=result_data.get('multi_turn_flow', False),
                        urgency=result_data.get('urgency', 'medium')
                    )
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Could not parse LLM intent analysis: {e}")
                    return self._rule_based_intent_enhancement(user_input, basic_intent)
            
        except Exception as e:
            logger.warning(f"LLM intent analysis failed: {e}")
        
        return self._rule_based_intent_enhancement(user_input, basic_intent)
    
    def _rule_based_intent_enhancement(self, user_input: str, basic_intent) -> ConversationIntent:
        """Fallback rule-based intent enhancement"""
        
        user_lower = user_input.lower()
        
        # Determine primary intent
        primary_intent = "help"
        confidence = 0.6
        suggested_mode = ConversationMode.GENERAL
        multi_turn_flow = False
        urgency = "medium"
        
        # Intent patterns
        if any(word in user_lower for word in ['quantize', 'compress', 'reduce size']):
            primary_intent = "quantize"
            suggested_mode = ConversationMode.QUANTIZATION
            multi_turn_flow = True
            
        elif any(word in user_lower for word in ['finetune', 'fine-tune', 'train', 'adapt']):
            primary_intent = "finetune"
            suggested_mode = ConversationMode.FINETUNING
            multi_turn_flow = True
            
        elif any(word in user_lower for word in ['serve', 'api', 'deploy', 'endpoint']):
            primary_intent = "serve_api"
            suggested_mode = ConversationMode.API_SERVING
            multi_turn_flow = True
            
        elif any(word in user_lower for word in ['find', 'search', 'discover', 'recommend']):
            primary_intent = "explore"
            suggested_mode = ConversationMode.EXPLORATION
            
        elif any(word in user_lower for word in ['learn', 'tutorial', 'guide', 'teach']):
            primary_intent = "learn"
            suggested_mode = ConversationMode.TUTORIAL
            
        elif any(word in user_lower for word in ['error', 'failed', 'problem', 'issue']):
            primary_intent = "troubleshoot"
            urgency = "high"
        
        # Extract entities
        entities = {}
        
        # Model name extraction
        model_patterns = [
            r'([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)',  # org/model
            r'(mistral|llama|gpt|opt|bloom|bert|t5)[\w-]*',  # common model families
        ]
        
        for pattern in model_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            if matches:
                entities['model_name'] = matches[0]
                break
        
        # Method extraction
        if any(method in user_lower for method in ['gptq', 'gguf', 'bitsandbytes', 'bnb']):
            method_map = {'bnb': 'bitsandbytes', 'gptq': 'gptq', 'gguf': 'gguf'}
            for method in method_map:
                if method in user_lower:
                    entities['method'] = method_map[method]
                    break
        
        return ConversationIntent(
            primary_intent=primary_intent,
            sub_intents=[],
            entities=entities,
            confidence=confidence,
            suggested_mode=suggested_mode,
            context_needed=[],
            multi_turn_flow=multi_turn_flow,
            urgency=urgency
        )
    
    def _build_conversation_context(self) -> Dict[str, Any]:
        """Build context information for LLM analysis"""
        return {
            "session_info": {
                "mode": self.current_session.current_mode.value,
                "active_context": self.current_session.active_context.value if self.current_session.active_context else None,
                "current_model": self.current_session.current_model,
                "active_workflow": self.current_session.active_workflow
            },
            "recent_actions": self.current_session.pending_actions[-3:] if self.current_session.pending_actions else [],
            "last_results": self.current_session.last_results,
            "conversation_length": len(self.current_session.conversation_history)
        }
    
    async def _should_switch_mode(self, intent: ConversationIntent) -> bool:
        """Determine if conversation mode should be switched"""
        
        # High confidence intents should trigger mode switches
        if intent.confidence > 0.8 and intent.suggested_mode != self.current_session.current_mode:
            return True
        
        # Multi-turn flows in different domains should switch
        if intent.multi_turn_flow and intent.suggested_mode != self.current_session.current_mode:
            return True
        
        # High urgency should switch to appropriate mode
        if intent.urgency == "high":
            return True
        
        return False
    
    async def _switch_conversation_mode(self, new_mode: ConversationMode):
        """Switch conversation mode with appropriate messaging"""
        old_mode = self.current_session.current_mode
        self.current_session.current_mode = new_mode
        
        mode_descriptions = {
            ConversationMode.GENERAL: "General LQMF assistance",
            ConversationMode.QUANTIZATION: "Model quantization workflow",
            ConversationMode.FINETUNING: "Fine-tuning and training workflow", 
            ConversationMode.API_SERVING: "API serving and deployment workflow",
            ConversationMode.EXPLORATION: "Model discovery and exploration",
            ConversationMode.TUTORIAL: "Interactive learning and tutorials"
        }
        
        console.print(f"[blue]🔄 Switching to {mode_descriptions[new_mode]} mode[/blue]")
    
    async def _process_mode_specific_conversation(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Process conversation based on current mode"""
        
        mode_handlers = {
            ConversationMode.GENERAL: self._handle_general_conversation,
            ConversationMode.QUANTIZATION: self._handle_quantization_conversation,
            ConversationMode.FINETUNING: self._handle_finetuning_conversation,
            ConversationMode.API_SERVING: self._handle_api_conversation,
            ConversationMode.EXPLORATION: self._handle_exploration_conversation,
            ConversationMode.TUTORIAL: self._handle_tutorial_conversation
        }
        
        handler = mode_handlers.get(self.current_session.current_mode, self._handle_general_conversation)
        return await handler(intent, user_input)
    
    async def _handle_general_conversation(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Handle general conversation and routing"""
        
        # Route to specific functionality
        if intent.primary_intent == "quantize":
            return await self._initiate_quantization_flow(intent, user_input)
        elif intent.primary_intent == "finetune": 
            return await self._initiate_finetuning_flow(intent, user_input)
        elif intent.primary_intent == "serve_api":
            return await self._initiate_api_flow(intent, user_input)
        elif intent.primary_intent == "explore":
            return await self._initiate_exploration_flow(intent, user_input)
        elif intent.primary_intent == "learn":
            return await self._initiate_tutorial_flow(intent, user_input)
        elif intent.primary_intent == "help":
            return self._provide_comprehensive_help()
        else:
            return await self._provide_conversational_guidance(intent, user_input)
    
    async def _handle_quantization_conversation(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Handle quantization-focused conversation flow"""
        
        # Check current workflow state
        if not self.current_session.active_workflow:
            return await self._start_quantization_workflow(intent, user_input)
        
        # Continue existing workflow
        return await self._continue_quantization_workflow(intent, user_input)
    
    async def _handle_finetuning_conversation(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Handle fine-tuning focused conversation flow"""
        
        if not self.current_session.active_workflow:
            return await self._start_finetuning_workflow(intent, user_input)
        
        return await self._continue_finetuning_workflow(intent, user_input)
    
    async def _handle_api_conversation(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Handle API serving focused conversation flow"""
        
        if not self.current_session.active_workflow:
            return await self._start_api_workflow(intent, user_input)
        
        return await self._continue_api_workflow(intent, user_input)
    
    async def _handle_exploration_conversation(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Handle model exploration and discovery conversation"""
        
        # Use memory agent for model discovery
        try:
            goals = intent.entities.get('goals', user_input)
            recommendations = self.memory_agent.discover_huggingface_models(goals, limit=5)
            
            if recommendations:
                response_text = f"I found {len(recommendations)} models that match your criteria:\n\n"
                suggestions = []
                
                for i, rec in enumerate(recommendations[:3], 1):
                    response_text += f"{i}. **{rec.model_name}** - {rec.category}\n"
                    response_text += f"   Overall Score: {rec.overall_score:.1%}\n"
                    response_text += f"   Key Reason: {rec.reasoning[0] if rec.reasoning else 'Good general choice'}\n\n"
                    
                    suggestions.append(f"Quantize {rec.model_name}")
                
                response_text += "Would you like to quantize one of these models or explore more options?"
                
                return CopilotResponse(
                    response_text=response_text,
                    actions_taken=[{"action": "model_discovery", "count": len(recommendations)}],
                    suggestions=suggestions,
                    next_steps=["Select a model to quantize", "Refine search criteria", "Ask for more details"],
                    mode_change=None,
                    requires_confirmation=True,
                    error_occurred=False,
                    tutorial_triggered=False
                )
            else:
                return CopilotResponse(
                    response_text="I couldn't find specific model recommendations. Let me help you explore available options or clarify your requirements.",
                    actions_taken=[],
                    suggestions=["Try different search terms", "Browse popular models", "Get quantization suggestions"],
                    next_steps=["Describe your use case in more detail"],
                    mode_change=None,
                    requires_confirmation=False,
                    error_occurred=False,
                    tutorial_triggered=False
                )
                
        except Exception as e:
            return CopilotResponse(
                response_text=f"I encountered an issue during model exploration: {e}. Let me help you with a different approach.",
                actions_taken=[],
                suggestions=["Try browsing popular models", "Get general recommendations"],
                next_steps=["Describe what you're looking for"],
                mode_change=None,
                requires_confirmation=False,
                error_occurred=True,
                tutorial_triggered=False
            )
    
    async def _handle_tutorial_conversation(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Handle tutorial and learning conversation"""
        
        # Determine tutorial type based on intent
        tutorial_type = "beginner_quantization"  # Default
        
        if "finetuning" in user_input.lower() or "training" in user_input.lower():
            tutorial_type = "advanced_finetuning"
        elif "api" in user_input.lower() or "deploy" in user_input.lower():
            tutorial_type = "api_deployment"
        
        tutorial = self.tutorial_flows.get(tutorial_type)
        if not tutorial:
            return CopilotResponse(
                response_text="I have several tutorials available. What would you like to learn about?",
                actions_taken=[],
                suggestions=["Model quantization basics", "Advanced fine-tuning", "API deployment"],
                next_steps=["Choose a tutorial topic"],
                mode_change=None,
                requires_confirmation=True,
                error_occurred=False,
                tutorial_triggered=True
            )
        
        response_text = f"**{tutorial['title']}**\n\n"
        response_text += f"Estimated time: {tutorial['estimated_time']}\n\n"
        response_text += "This tutorial will cover:\n"
        
        for i, step in enumerate(tutorial['steps'], 1):
            response_text += f"{i}. {step['content']}\n"
        
        response_text += "\nWould you like to start this tutorial?"
        
        return CopilotResponse(
            response_text=response_text,
            actions_taken=[{"action": "tutorial_offered", "type": tutorial_type}],
            suggestions=["Start tutorial", "Choose different tutorial", "Get overview first"],
            next_steps=["Confirm tutorial start"],
            mode_change=None,
            requires_confirmation=True,
            error_occurred=False,
            tutorial_triggered=True
        )
    
    # Workflow initiation methods
    async def _initiate_quantization_flow(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Start quantization workflow"""
        
        model_name = intent.entities.get('model_name')
        method = intent.entities.get('method')
        
        if model_name:
            response_text = f"Great! I'll help you quantize **{model_name}**.\n\n"
            
            if method:
                response_text += f"You mentioned using **{method.upper()}** quantization. "
            else:
                response_text += "Let me suggest the best quantization method for this model. "
            
            # Get AI recommendations
            try:
                if self.llm_manager.is_available():
                    compat_response = self.llm_manager.check_model_compatibility(model_name, method or "auto", "auto")
                    if compat_response.success:
                        response_text += f"\n\n**AI Analysis:**\n{compat_response.content[:200]}..."
            except Exception:
                pass
            
            response_text += "\n\nNext steps:\n1. Confirm model and method\n2. Configure parameters\n3. Execute quantization\n4. Test results"
            
            # Set workflow state
            self.current_session.active_workflow = "quantization"
            self.current_session.current_model = model_name
            
            return CopilotResponse(
                response_text=response_text,
                actions_taken=[{"action": "quantization_initiated", "model": model_name}],
                suggestions=[f"Proceed with {model_name}", "Choose different model", "Get more details"],
                next_steps=["Confirm quantization parameters", "Start quantization process"],
                mode_change=ConversationMode.QUANTIZATION,
                requires_confirmation=True,
                error_occurred=False,
                tutorial_triggered=False
            )
        else:
            return CopilotResponse(
                response_text="I'd love to help you quantize a model! Which model would you like to work with? I can help you find the right one for your needs.",
                actions_taken=[],
                suggestions=["Browse popular models", "Get model recommendations", "Search specific model"],
                next_steps=["Tell me your model name or describe what you need"],
                mode_change=ConversationMode.EXPLORATION,
                requires_confirmation=False,
                error_occurred=False,
                tutorial_triggered=False
            )
    
    # Continue workflow methods (implemented similarly for each workflow type)
    async def _continue_quantization_workflow(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Continue active quantization workflow"""
        
        # This would implement the multi-turn quantization flow
        # with parameter confirmation, execution, and testing
        
        return CopilotResponse(
            response_text="Continuing with quantization workflow...",
            actions_taken=[],
            suggestions=[],
            next_steps=[],
            mode_change=None,
            requires_confirmation=False,
            error_occurred=False,
            tutorial_triggered=False
        )
    
    # Helper methods
    def _provide_comprehensive_help(self) -> CopilotResponse:
        """Provide comprehensive help information"""
        
        help_text = """
# 🤖 LQMF Conversational Copilot

I'm your AI assistant for model quantization, fine-tuning, and deployment. I can help you with natural language conversations!

## What I Can Do:

**🔧 Model Quantization**
- "Quantize Mistral 7B for my 8GB GPU"
- "Find me a lightweight chat model to compress"
- "What's the best quantization method for GPT models?"

**🎯 Fine-tuning** 
- "Help me fine-tune a model for customer support"
- "Analyze my dataset for training quality"
- "Create a LoRA configuration for my use case"

**🌐 API Serving**
- "Deploy my quantized model as an API"
- "Show me how to test my model endpoints"
- "Set up model serving for production"

**🔍 Model Discovery**
- "Find me models for code generation"
- "Recommend fast inference models under 2GB"
- "What are the best chat models for mobile?"

**📚 Learning & Tutorials**
- "Teach me about quantization"
- "Guide me through fine-tuning"
- "Show me API deployment best practices"

## Natural Language Examples:
- "I need to make LLaMA 2 smaller for my laptop"
- "Help me train a model to write better emails"
- "Which models work best for chatbots?"
- "My quantization failed, what went wrong?"
- "Show me my fastest models under 2GB"

Just tell me what you want to do in natural language - I'll understand and guide you through the process!
        """
        
        return CopilotResponse(
            response_text=help_text,
            actions_taken=[{"action": "help_provided", "type": "comprehensive"}],
            suggestions=["Try quantizing a model", "Explore model recommendations", "Start a tutorial"],
            next_steps=["Ask me anything about models, quantization, or fine-tuning!"],
            mode_change=None,
            requires_confirmation=False,
            error_occurred=False,
            tutorial_triggered=False
        )
    
    async def _provide_conversational_guidance(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Provide conversational guidance when intent is unclear"""
        
        # Use LLM to generate contextual guidance
        if self.llm_manager.is_available():
            try:
                guidance_prompt = f"""
                You are the LQMF Conversational Copilot. The user said: "{user_input}"
                
                Their intent seems unclear or general. Provide a helpful, conversational response that:
                1. Acknowledges their input
                2. Asks clarifying questions 
                3. Suggests specific actions they might want to take
                4. Maintains a friendly, helpful tone
                
                Focus on model quantization, fine-tuning, and API serving capabilities.
                Keep response concise and actionable.
                """
                
                from utils.llm_config import QueryType
                response = self.llm_manager.query(guidance_prompt, QueryType.GENERAL)
                
                if response.success:
                    return CopilotResponse(
                        response_text=response.content,
                        actions_taken=[],
                        suggestions=["Tell me about your specific goals", "Browse available models", "Start with a tutorial"],
                        next_steps=["Be more specific about what you'd like to do"],
                        mode_change=None,
                        requires_confirmation=False,
                        error_occurred=False,
                        tutorial_triggered=False
                    )
            except Exception:
                pass
        
        # Fallback guidance
        return CopilotResponse(
            response_text="I'm here to help with model quantization, fine-tuning, and API serving! Could you tell me more about what you're trying to accomplish?",
            actions_taken=[],
            suggestions=["Quantize a specific model", "Find model recommendations", "Learn about the process"],
            next_steps=["Describe your goal or ask a specific question"],
            mode_change=None,
            requires_confirmation=False,
            error_occurred=False,
            tutorial_triggered=False
        )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    # Placeholder methods for additional workflow implementations
    async def _initiate_finetuning_flow(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Start fine-tuning workflow - to be fully implemented"""
        return CopilotResponse(
            response_text="Fine-tuning workflow initiated. Full implementation coming soon!",
            actions_taken=[], suggestions=[], next_steps=[], 
            mode_change=ConversationMode.FINETUNING,
            requires_confirmation=False, error_occurred=False, tutorial_triggered=False
        )
    
    async def _initiate_api_flow(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Start API serving workflow - to be fully implemented"""
        return CopilotResponse(
            response_text="API serving workflow initiated. Full implementation coming soon!",
            actions_taken=[], suggestions=[], next_steps=[],
            mode_change=ConversationMode.API_SERVING, 
            requires_confirmation=False, error_occurred=False, tutorial_triggered=False
        )
    
    async def _initiate_tutorial_flow(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Start tutorial flow - delegates to tutorial handler"""
        return await self._handle_tutorial_conversation(intent, user_input)
    
    # Additional workflow continuation methods would be implemented here
    async def _start_quantization_workflow(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Start new quantization workflow"""
        return await self._initiate_quantization_flow(intent, user_input)
    
    async def _continue_finetuning_workflow(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Continue fine-tuning workflow - to be implemented"""
        return await self._initiate_finetuning_flow(intent, user_input)
    
    async def _start_finetuning_workflow(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Start fine-tuning workflow - to be implemented"""
        return await self._initiate_finetuning_flow(intent, user_input)
    
    async def _start_api_workflow(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Start API workflow - to be implemented"""
        return await self._initiate_api_flow(intent, user_input)
    
    async def _continue_api_workflow(self, intent: ConversationIntent, user_input: str) -> CopilotResponse:
        """Continue API workflow - to be implemented"""
        return await self._initiate_api_flow(intent, user_input)

# Example usage and testing
if __name__ == "__main__":
    async def test_copilot():
        copilot = ConversationalCopilotAgent()
        
        test_inputs = [
            "I want to quantize Mistral 7B for my 8GB GPU",
            "Show me my fastest chat models under 2GB", 
            "Help me fine-tune a model for customer support",
            "What models are good for code generation?",
            "My quantization failed, what went wrong?"
        ]
        
        for test_input in test_inputs:
            print(f"\nUser: {test_input}")
            response = await copilot.process_conversation(test_input)
            print(f"Copilot: {response.response_text[:200]}...")
    
    # Run test if executed directly
    import asyncio
    asyncio.run(test_copilot())