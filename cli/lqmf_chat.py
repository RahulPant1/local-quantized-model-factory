#!/usr/bin/env python3
"""
LQMF Conversational Chat Interface

Unified natural language interface for all LQMF functionality including:
- Quantization workflows
- Fine-tuning operations  
- API serving management
- Model exploration and benchmarking

Usage:
    python cli/lqmf_chat.py

Features:
- Natural language understanding
- Multi-turn conversations
- Context-aware responses
- Seamless workflow transitions
- Interactive tutorials
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from agents.conversational_copilot_agent import ConversationalCopilotAgent, CopilotResponse
    from utils.llm_config import get_llm_manager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed and the project is properly set up.")
    sys.exit(1)


class LQMFChatInterface:
    """Main chat interface for LQMF unified conversational experience."""
    
    def __init__(self):
        """Initialize the chat interface."""
        self.copilot = None
        self.running = True
        
    async def initialize(self):
        """Initialize the conversational copilot."""
        try:
            print("🚀 Initializing LQMF Conversational Copilot...")
            self.copilot = ConversationalCopilotAgent()
            print("✅ LQMF Copilot ready!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize copilot: {e}")
            print(f"Error details: {str(e)}")
            return False
    
    def print_welcome(self):
        """Print welcome message and instructions."""
        print("\n" + "="*60)
        print("🧠 LQMF Conversational Copilot")
        print("="*60)
        print("Natural language interface for all LQMF functionality!")
        print("\n💬 What can I help you with today?")
        print("   • Quantize models: 'Quantize Mistral 7B to GGUF format'")
        print("   • Fine-tune models: 'Train a chat adapter for customer support'")
        print("   • API serving: 'Start serving my quantized models'")
        print("   • Explore models: 'Show me my fastest models under 2GB'")
        print("   • Get help: 'Show me a tutorial' or 'help'")
        print("\n📝 Commands:")
        print("   • /tutorial - Start interactive tutorial")
        print("   • /help - Show detailed help")
        print("   • /status - Show system status")
        print("   • /clear - Clear conversation history")
        print("   • /quit or /exit - Exit chat")
        print("\n" + "="*60)
    
    def print_response(self, response: CopilotResponse):
        """Print formatted copilot response."""
        print(f"\n🤖 {response.message}")
        
        if response.suggestions:
            print("\n💡 Suggestions:")
            for suggestion in response.suggestions:
                print(f"   • {suggestion}")
        
        if response.actions_taken:
            print("\n✅ Actions completed:")
            for action in response.actions_taken:
                print(f"   • {action}")
        
        if response.next_steps:
            print("\n🔄 Next steps:")
            for step in response.next_steps:
                print(f"   • {step}")
    
    async def handle_special_commands(self, user_input: str) -> bool:
        """Handle special chat commands. Returns True if command was handled."""
        command = user_input.lower().strip()
        
        if command in ['/quit', '/exit', 'quit', 'exit']:
            print("\n👋 Thanks for using LQMF Conversational Copilot!")
            self.running = False
            return True
        
        elif command == '/clear':
            print("\n🧹 Clearing conversation history...")
            if self.copilot:
                self.copilot.clear_conversation_history()
            print("✅ Conversation history cleared.")
            return True
        
        elif command == '/help':
            self.print_help()
            return True
        
        elif command == '/status':
            if self.copilot:
                response = await self.copilot.process_conversation("Show system status")
                self.print_response(response)
            return True
        
        elif command == '/tutorial':
            if self.copilot:
                response = await self.copilot.process_conversation("Start the interactive tutorial")
                self.print_response(response)
            return True
        
        return False
    
    def print_help(self):
        """Print detailed help information."""
        print("\n" + "="*50)
        print("📚 LQMF Conversational Copilot Help")
        print("="*50)
        print("\n🎯 What you can do:")
        print("   • Quantization: Convert models to different formats")
        print("   • Fine-tuning: Train adapters for specific tasks")
        print("   • API Serving: Deploy and manage model APIs")
        print("   • Exploration: Discover and analyze models")
        print("\n💬 Natural Language Examples:")
        print("   • 'I want to quantize Llama 2 7B to 4-bit GGUF'")
        print("   • 'Help me fine-tune a model for legal documents'")
        print("   • 'Show me my models that use less than 4GB memory'")
        print("   • 'Start an API server with my best chat model'")
        print("   • 'What's the fastest way to get started?'")
        print("\n🔧 Special Commands:")
        print("   /tutorial  - Interactive guided tutorial")
        print("   /help      - This help message")
        print("   /status    - Show system and model status")
        print("   /clear     - Clear conversation history")
        print("   /quit      - Exit the chat interface")
        print("\n💡 Tips:")
        print("   • Be conversational - I understand natural language!")
        print("   • Ask follow-up questions - I remember our conversation")
        print("   • Request explanations - I can guide you through processes")
        print("   • Experiment freely - I'll help you learn as you go")
        print("="*50)
    
    async def run_chat_loop(self):
        """Run the main chat interaction loop."""
        while self.running:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if await self.handle_special_commands(user_input):
                    continue
                
                # Process with copilot
                if self.copilot:
                    print("🤔 Processing...")
                    response = await self.copilot.process_conversation(user_input)
                    self.print_response(response)
                else:
                    print("❌ Copilot not initialized. Please restart the application.")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type '/help' for assistance.")
    
    async def run(self):
        """Main run method."""
        # Initialize
        if not await self.initialize():
            return 1
        
        # Show welcome
        self.print_welcome()
        
        # Run chat loop
        await self.run_chat_loop()
        
        return 0


async def main():
    """Main entry point."""
    chat_interface = LQMFChatInterface()
    return await chat_interface.run()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)