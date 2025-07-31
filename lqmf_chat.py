#!/usr/bin/env python3
"""
LQMF Conversational Chat - Main Entry Point

Quick start script for the unified LQMF chat interface.
This provides natural language access to all LQMF functionality.

Usage:
    python lqmf_chat.py

Features:
- Natural language quantization: "Quantize Mistral 7B to GGUF"
- Conversational fine-tuning: "Train a chat adapter for customer service"
- Interactive API management: "Start serving my best models"
- Model exploration: "Show me models under 2GB"
- Guided tutorials and help
"""

import sys
from pathlib import Path

# Ensure we can import from cli directory
project_root = Path(__file__).parent
cli_path = project_root / "cli"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(cli_path))

if __name__ == "__main__":
    try:
        from cli.lqmf_chat import main
        import asyncio
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except ImportError as e:
        print(f"❌ Failed to import LQMF chat interface: {e}")
        print("\nPlease ensure:")
        print("1. All dependencies are installed: pip install -r requirements.txt")
        print("2. The project structure is intact")
        print("3. Run from the project root directory")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)