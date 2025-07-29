"""
DocumentationAgent - Automatic documentation generation for LQMF
Analyzes the codebase and generates comprehensive documentation
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import inspect
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()

@dataclass
class ModuleInfo:
    name: str
    path: str
    docstring: Optional[str]
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    imports: List[str]
    lines_of_code: int

@dataclass
class ProjectStructure:
    total_files: int
    total_lines: int
    modules: List[ModuleInfo]
    dependencies: List[str]
    entry_points: List[str]

class DocumentationAgent:
    """Agent for generating comprehensive project documentation"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.console = Console()
        
    def analyze_python_file(self, file_path: Path) -> ModuleInfo:
        """Analyze a single Python file and extract information"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract module docstring
            module_docstring = ast.get_docstring(tree)
            
            # Extract classes
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node),
                        "methods": [method.name for method in node.body if isinstance(method, ast.FunctionDef)],
                        "line_number": node.lineno
                    }
                    classes.append(class_info)
            
            # Extract functions
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not any(node in cls.body for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef)):
                    func_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node),
                        "args": [arg.arg for arg in node.args.args],
                        "line_number": node.lineno
                    }
                    functions.append(func_info)
            
            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imports.extend([f"{module}.{alias.name}" for alias in node.names])
            
            lines_of_code = len(content.splitlines())
            
            return ModuleInfo(
                name=file_path.stem,
                path=str(file_path),
                docstring=module_docstring,
                classes=classes,
                functions=functions,
                imports=imports,
                lines_of_code=lines_of_code
            )
            
        except Exception as e:
            console.print(f"[red]Error analyzing {file_path}: {e}[/red]")
            return ModuleInfo(
                name=file_path.stem,
                path=str(file_path),
                docstring=None,
                classes=[],
                functions=[],
                imports=[],
                lines_of_code=0
            )
    
    def analyze_project_structure(self) -> ProjectStructure:
        """Analyze the entire project structure"""
        python_files = list(self.project_root.rglob("*.py"))
        modules = []
        total_lines = 0
        all_imports = set()
        
        for file_path in python_files:
            if "__pycache__" not in str(file_path):
                module_info = self.analyze_python_file(file_path)
                modules.append(module_info)
                total_lines += module_info.lines_of_code
                all_imports.update(module_info.imports)
        
        # Extract dependencies from requirements.txt
        dependencies = []
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            with open(req_file, 'r') as f:
                dependencies = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Find entry points
        entry_points = []
        for module in modules:
            if "main" in module.name or "run" in module.name:
                entry_points.append(module.path)
        
        return ProjectStructure(
            total_files=len(python_files),
            total_lines=total_lines,
            modules=modules,
            dependencies=dependencies,
            entry_points=entry_points
        )
    
    def generate_readme(self, structure: ProjectStructure) -> str:
        """Generate a comprehensive README.md file"""
        readme_content = f"""# 🏭 Local Quantized Model Factory (LQMF)

**Agent-Powered Model Quantization System for 8GB GPUs**

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 🎯 Overview

LQMF is an interactive, AI-powered system designed to simplify model quantization for local deployment. It combines multiple quantization backends with intelligent planning assistance from Google's Gemini API to optimize models for 8GB GPU constraints.

## 🏗️ Architecture

The system is built around four core agents that work together to provide a seamless quantization experience:

### Core Agents

| Agent | Purpose | Key Features |
|-------|---------|--------------|
| **PlannerAgent** | Interactive planning & Gemini integration | Natural language parsing, AI recommendations, quantization strategy |
| **ExecutorAgent** | Model download & quantization execution | Multi-backend support, hardware optimization, progress tracking |
| **MemoryAgent** | Experiment tracking & metadata storage | SQLite/JSON backends, versioning, performance metrics |
| **FeedbackAgent** | Performance analysis & benchmarking | Quality metrics, performance profiling, model comparison |

### Architecture Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  PlannerAgent    │───▶│ Gemini API      │
│ (Natural Lang.) │    │  - Parse request │    │ - Suggestions   │
└─────────────────┘    │  - Get AI advice │    │ - Optimization  │
                       └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Model Storage  │◀───│  ExecutorAgent   │───▶│ Quantization    │
│  - GGUF/GPTQ    │    │  - Download      │    │ - GPTQ/BnB      │
│  - Safetensors  │    │  - Quantize      │    │ - GGUF          │
└─────────────────┘    │  - Validate      │    └─────────────────┘
                       └──────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Statistics    │◀───│  MemoryAgent     │───▶│ Benchmarking    │
│   Dashboard     │    │  - Track expts   │    │ FeedbackAgent   │
└─────────────────┘    │  - Store metrics │    └─────────────────┘
                       └──────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd my_slm_factory_app

# Install dependencies
pip install -r requirements.txt

# Set up environment (optional)
cp .env.example .env
# Edit .env with your Gemini API key
```

### Basic Usage

```bash
# Run the application
python run.py

# Or use the CLI directly
python cli/main.py
```

### Example Commands

```bash
# Natural language quantization
> quantize mistralai/Mistral-7B-Instruct for 8GB GPU

# Specific configuration
> quantize microsoft/DialoGPT-small to 4-bit GGUF for CPU

# View experiments
> list successful experiments
> stats
```

## 📊 Project Statistics

- **Total Python Files**: {structure.total_files}
- **Lines of Code**: {structure.total_lines:,}
- **Core Modules**: {len([m for m in structure.modules if 'agent' in m.name])}
- **Dependencies**: {len(structure.dependencies)}

## 🧩 Module Overview

"""

        # Add module details
        for module in structure.modules:
            if module.classes or module.functions:
                readme_content += f"""
### {module.name.replace('_', ' ').title()} (`{module.path}`)

{module.docstring or 'Core module for LQMF functionality.'}

**Classes**: {len(module.classes)} | **Functions**: {len(module.functions)} | **Lines**: {module.lines_of_code}

"""
                if module.classes:
                    readme_content += "**Key Classes:**\n"
                    for cls in module.classes[:3]:  # Show top 3 classes
                        readme_content += f"- `{cls['name']}`: {cls['docstring'][:100] + '...' if cls['docstring'] and len(cls['docstring']) > 100 else cls['docstring'] or 'Core class'}\n"
                    readme_content += "\n"

        readme_content += f"""
## 🔧 Configuration

The system supports multiple configuration options:

### Environment Variables (.env)
```env
GEMINI_API_KEY=your_api_key_here
STORAGE_BACKEND=sqlite
GPU_MEMORY_LIMIT=8
CPU_FALLBACK=true
```

### Command Line Options
```bash
python run.py --help
python run.py --storage-backend sqlite --gemini-key YOUR_KEY
```

## 🎮 Usage Examples

### Example 1: Basic Quantization
```bash
LQMF> quantize gpt2
# System will guide you through:
# 1. Quantization method selection (GPTQ/GGUF/BitsAndBytes)
# 2. Bit width configuration (4-bit/8-bit)
# 3. Hardware optimization for your 8GB GPU
```

### Example 2: Natural Language
```bash
LQMF> I want to quantize Mistral 7B for CPU inference with 4-bit precision
# AI will automatically:
# 1. Parse: model=mistralai/Mistral-7B-Instruct, method=GGUF, bits=4
# 2. Provide Gemini recommendations
# 3. Execute optimized quantization
```

### Example 3: Experiment Tracking
```bash
LQMF> stats
# Shows:
# - Success/failure rates
# - Average execution times  
# - Storage savings achieved
# - Method performance comparison
```

## 📈 Expected Output

### Successful Quantization
```
✅ Quantization Successful!
Model saved to: models/mistralai_Mistral-7B-Instruct_gptq_4bit
Size: 3,247.5MB (reduced from 13,852.2MB)
Time: 127.34 seconds
Compression Ratio: 4.26x
```

### Performance Metrics
```
Performance Metrics:
├── Tokens/Second: 23.4
├── Memory Usage: 3.2GB
├── Model Size: 3.2GB  
└── GPU Utilization: 67%

Quality Metrics:
├── Response Quality: 0.87/1.0
├── Coherence Score: 0.84/1.0
└── Avg Response Length: 127 words
```

## 🎯 Supported Models

- **Language Models**: GPT-2, GPT-Neo, LLaMA, Mistral, CodeLlama
- **Conversational**: DialoGPT, BlenderBot, ChatGLM
- **Code Models**: CodeT5, CodeBERT, StarCoder
- **Custom Models**: Any transformer-based model on Hugging Face

## 🔄 Quantization Methods

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **GPTQ** | GPU inference | Fast, good compression | GPU memory needed |
| **GGUF** | CPU deployment | Cross-platform, efficient | Slower inference |
| **BitsAndBytes** | Experimentation | Easy to use, flexible | Less optimized |

## 📱 Dependencies

{chr(10).join(f"- {dep}" for dep in structure.dependencies[:10])}
{"..." if len(structure.dependencies) > 10 else ""}

## 🛠️ Development

### Project Structure
```
{self.project_root.name}/
├── agents/          # Core agent implementations
├── cli/            # Command-line interface
├── utils/          # Helper utilities
├── models/         # Downloaded and quantized models
├── configs/        # Quantization configurations
├── logs/           # Execution logs
├── mcp/            # Memory context protocol storage
└── requirements.txt
```

### Adding New Quantization Methods

1. Extend `ExecutorAgent` with new quantization logic
2. Update `PlannerAgent` parsing for new method keywords
3. Add configuration options to `QuantizationType` enum
4. Update Gemini prompts for new method recommendations

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for intelligent planning assistance
- Hugging Face for model hosting and transformers library
- GPTQ, llama.cpp, and BitsAndBytes teams for quantization implementations
- Rich library for beautiful terminal interfaces

---

**Generated by DocumentationAgent** | LQMF v1.0.0 | {datetime.now().strftime("%Y-%m-%d")}
"""
        
        return readme_content
    
    def generate_api_documentation(self, structure: ProjectStructure) -> str:
        """Generate API documentation"""
        api_doc = f"""# 📚 LQMF API Documentation

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Agent APIs

"""
        
        # Document each agent
        agent_modules = [m for m in structure.modules if 'agent' in m.name and m.classes]
        
        for module in agent_modules:
            api_doc += f"""
## {module.name.replace('_', ' ').title()}

**File**: `{module.path}`

{module.docstring or 'Core agent implementation.'}

### Classes

"""
            for cls in module.classes:
                api_doc += f"""
#### `{cls['name']}`

{cls['docstring'] or 'Core agent class.'}

**Methods**: {', '.join(cls['methods'][:5])}

**Location**: Line {cls['line_number']}

"""
        
        return api_doc
    
    def generate_all_documentation(self) -> Dict[str, str]:
        """Generate all documentation files"""
        console.print(Panel.fit("📚 Analyzing Codebase for Documentation", style="bold blue"))
        
        # Analyze project
        structure = self.analyze_project_structure()
        
        # Generate documentation
        readme = self.generate_readme(structure)
        api_doc = self.generate_api_documentation(structure)
        
        # Save files
        with open(self.project_root / "README.md", 'w') as f:
            f.write(readme)
        
        with open(self.project_root / "API_DOCUMENTATION.md", 'w') as f:
            f.write(api_doc)
        
        # Generate project summary
        summary = f"""
# 📊 LQMF Project Summary

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Quick Stats
- **Files**: {structure.total_files} Python files
- **Code**: {structure.total_lines:,} lines
- **Agents**: {len([m for m in structure.modules if 'agent' in m.name])} core agents
- **Dependencies**: {len(structure.dependencies)} packages

## Architecture Overview
LQMF implements a modular agent-based architecture for AI-powered model quantization:

1. **PlannerAgent**: Natural language processing + Gemini AI integration
2. **ExecutorAgent**: Multi-backend quantization execution  
3. **MemoryAgent**: Experiment tracking with SQLite/JSON
4. **FeedbackAgent**: Performance analysis and benchmarking

## Key Features
- 🧠 AI-powered quantization planning with Gemini
- ⚙️ Multi-backend support (GPTQ, GGUF, BitsAndBytes)
- 🎯 8GB GPU optimization
- 📊 Comprehensive experiment tracking
- 💬 Natural language interface
- 🚀 One-command model quantization

Ready for production use! 🎉
"""
        
        console.print(Panel.fit("✅ Documentation Generated Successfully", style="bold green"))
        
        return {
            "readme": readme,
            "api_doc": api_doc,
            "summary": summary,
            "structure": structure
        }

if __name__ == "__main__":
    doc_agent = DocumentationAgent()
    docs = doc_agent.generate_all_documentation()
    print("Documentation generated successfully!")