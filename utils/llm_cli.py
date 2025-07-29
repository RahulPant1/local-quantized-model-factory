#!/usr/bin/env python3
"""
LLM Configuration CLI
Simple CLI tool to manage LLM provider settings
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.llm_config import (
    change_provider, get_llm_manager, get_current_config, 
    update_model_for_provider, DEFAULT_LLM_CONFIGS,
    LLMProvider
)

def show_current_config():
    """Show current LLM configuration"""
    manager = get_llm_manager()
    
    if not manager.current_provider:
        print("❌ No LLM provider configured")
        return
    
    config = get_current_config()
    print(f"✅ Current Provider: {manager.current_provider.value}")
    print(f"   Model: {config.model_name}")
    print(f"   Available: {'Yes' if manager.is_available() else 'No (API key missing)'}")
    print(f"   Timeout: {config.request_timeout}s")
    print(f"   Max Retries: {config.max_retries}")

def show_all_configs():
    """Show all available provider configurations"""
    print("📋 Available LLM Providers:")
    print()
    
    current_provider = get_llm_manager().current_provider
    
    for provider, config in DEFAULT_LLM_CONFIGS.items():
        status = "🟢 ACTIVE" if provider == current_provider else "⚫ Available"
        print(f"{status} {provider.value.upper()}")
        print(f"    Model: {config.model_name}")
        print(f"    Timeout: {config.request_timeout}s")
        print(f"    Retries: {config.max_retries}")
        print()

def main():
    """Main CLI function"""
    if len(sys.argv) < 2:
        print("LLM Configuration CLI")
        print()
        print("Usage:")
        print("  python utils/llm_cli.py status          - Show current configuration")
        print("  python utils/llm_cli.py list           - List all available providers")
        print("  python utils/llm_cli.py switch <provider> - Switch to provider (gemini/claude/openai)")
        print("  python utils/llm_cli.py model <provider> <model> - Update model for provider")
        print()
        print("Environment Variables:")
        print("  GEMINI_API_KEY     - Gemini API key")
        print("  ANTHROPIC_API_KEY  - Claude API key") 
        print("  OPENAI_API_KEY     - OpenAI API key")
        return
    
    command = sys.argv[1].lower()
    
    if command == "status":
        show_current_config()
        
    elif command == "list":
        show_all_configs()
        
    elif command == "switch":
        if len(sys.argv) < 3:
            print("❌ Please specify provider: gemini, claude, or openai")
            return
        provider = sys.argv[2].lower()
        change_provider(provider)
        
    elif command == "model":
        if len(sys.argv) < 4:
            print("❌ Usage: python utils/llm_cli.py model <provider> <model_name>")
            return
        provider = sys.argv[2].lower()
        model_name = sys.argv[3]
        update_model_for_provider(provider, model_name)
        
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python utils/llm_cli.py' for usage help")

if __name__ == "__main__":
    main()