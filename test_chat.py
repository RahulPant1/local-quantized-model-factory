#!/usr/bin/env python3
"""
Test script for LQMF Chat Interface
This version handles missing dependencies gracefully for testing
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test which modules can be imported"""
    print("🧪 Testing imports...")
    
    # Test basic imports
    try:
        from agents.conversational_copilot_agent import ConversationalCopilotAgent
        print("✅ ConversationalCopilotAgent imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import ConversationalCopilotAgent: {e}")
        return False

def test_basic_functionality():
    """Test basic agent functionality"""
    try:
        from agents.conversational_copilot_agent import ConversationalCopilotAgent
        print("🧪 Testing agent initialization...")
        
        agent = ConversationalCopilotAgent()
        print("✅ Agent created successfully")
        
        # Test basic properties
        print(f"📊 Session ID: {agent.current_session.session_id}")
        print(f"📊 Current Mode: {agent.current_session.current_mode}")
        print(f"📊 Conversation History: {len(agent.current_session.conversation_history)} messages")
        
        return True
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False

async def test_conversation():
    """Test conversation processing"""
    try:
        from agents.conversational_copilot_agent import ConversationalCopilotAgent
        print("🧪 Testing conversation processing...")
        
        agent = ConversationalCopilotAgent()
        
        # Test with a simple message
        response = await agent.process_conversation("hello")
        print(f"✅ Got response: {response.message[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Conversation processing failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 LQMF Chat Interface Test")
    print("="*50)
    
    # Test imports
    if not test_imports():
        return 1
    
    # Test basic functionality
    if not test_basic_functionality():
        return 1
    
    # Test conversation
    if not await test_conversation():
        return 1
    
    print("\n✅ All tests passed!")
    return 0

if __name__ == "__main__":
    import asyncio
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted")
        sys.exit(0)