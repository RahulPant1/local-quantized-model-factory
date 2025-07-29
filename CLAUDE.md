# Project: Local Quantized Model Factory (LQMF)
## Objective:
Build a local agent-powered system that allows me to:
- Select a Hugging Face model (via chat or CLI)
- Uses an agentic flow to **plan and execute quantization**
- Run quantized models locally on an 8GB GPU (or CPU fallback)
- Store versioned outputs and metadata (via MCP-like memory)
- Enable conversational planning, follow-ups, and experiment tracking
- Interacts with the user via a **chat-first interface**
- Uses **Multiple LLM APIs** for planning assistance (Gemini, Claude, OpenAI - configurable)


---

## Guidelines:
- Modular design: Create composable agents/functions (download, quantize, log, benchmark)
- Conversational flow: Use chat-based interface to drive setup and refinement
- Minimize memory footprint: All operations should run on 8GB GPU or offload to CPU
- Orchestration with context: Track state using a lightweight MCP schema (JSON or SQLite)
- Local-first, offline-capable: No cloud dependency after initial download
- Transparent logs and metadata for every model and experiment

---

## Core Tasks (Decomposed)
### 🧩 Phase 1: Model Acquisition
- Input: HF model name (e.g., "mistralai/Mistral-7B-Instruct")
- Agent downloads model + tokenizer to `/models/`
- Checks compatibility for quant (transformers or llama.cpp)

### 🔧 Phase 2: Interactive Quantization via Chat
- Agent parses user prompt like:
  > "Quantize to 4-bit GGUF for CPU inference"
- Ask follow-ups:
  - Quant type (e.g., GPTQ, BnB, llama.cpp q4_K_M)
  - Target format (e.g., GGUF, safetensors)
  - CPU fallback required?

### 🧠 Phase 3: Model Context Protocol (MCP)
- Create a memory schema to store:
  - Model metadata
  - Quantization parameters
  - Execution logs and retry attempts
- Store as local JSON or SQLite under `/mcp/`

### ⚙️ Phase 4: Execute Quantization
- Based on config, run:
  - transformers/optimum for GPTQ
  - llama.cpp tooling for GGUF
- Log:
  - Quant success/failure
  - Model size
  - RAM usage and speed (optional)

### 🗃️ Phase 5: Store Versioned Output
- Save model in `/models/`
- Save config in `/configs/`
- Save logs in `/logs/`
- MCP entry updated with timestamp and metrics

---

## 🔁 User Interaction Flow

This tool is designed to be **interactive and conversational** from the start.

Example user flow:

> User: “Quantize Mistral 7B for CPU in GGUF format.”

1. Planner Agent confirms:
   - Target bit width (4-bit, 5-bit?)
   - Quantization type (GGUF via `llama.cpp`, GPTQ via `transformers`)
   - CPU or GPU fallback needed?

2. LLM API (configurable: Gemini/Claude/OpenAI) is invoked to:
   - Suggest optimal quantization format
   - Check model compatibility with toolchains
   - Assist in filling gaps in user request

3. Executor Agent:
   - Downloads model from Hugging Face
   - Runs conversion script
   - Logs progress/errors

4. Memory Agent (MCP) stores:
   - Model config
   - Version
   - Output metadata
   - Logs

5. Feedback loop (optional): Model is run and token/speed/quality logged

---

## 🧩 Agent Overview

### Core Agents (Main CLI)
| Agent | Role | File |
|-------|------|------|
| **EnhancedDecisionAgent** | Natural language intent understanding and routing | `enhanced_decision_agent.py` |
| **PlannerAgent** | Interactive quantization planning with LLM assistance | `planner_agent.py` |
| **ExecutorAgent** | Model download, conversion, and quantization execution | `executor_agent.py` |
| **MemoryAgent** | Experiment tracking and metadata storage (JSON/SQLite) | `memory_agent.py` |
| **FeedbackAgent** | Performance benchmarking and quality metrics | `feedback_agent.py` |

### API Server Agents (API CLI)
| Agent | Role | File |
|-------|------|------|
| **APIServerAgent** | Model serving and REST API endpoints | `api_server_agent.py` |
| **ModelExperimentAgent** | Interactive model experimentation and testing | `model_experiment_agent.py` |

### Utility Agents
| Agent | Role | File |
|-------|------|------|
| **DocumentationAgent** | Automatic documentation generation | `documentation_agent.py` |

### Removed/Consolidated
- ~~**DecisionAgent**~~ → Consolidated into **EnhancedDecisionAgent**

---

## 🔧 Tech Stack Summary

- **LLM APIs**: Configurable planning assistant (Gemini/Claude/OpenAI)
  - Centralized configuration in `utils/llm_config.py`
  - Switch providers via `config.json` or CLI tool
- **Python CLI**: Dual interface (main CLI + API server CLI)
- **Hugging Face Hub**: Source of models
- **Quantization Tools**:
  - `llama.cpp` for GGUF quantization
  - `AutoGPTQ` / `BitsAndBytes` for transformer-based
- **Storage**: Local filesystem for models, configs, logs, SQLite for MCP
- **API Server**: FastAPI-based model serving endpoints
- **Inference**: llama.cpp, transformers, or custom API endpoints

## 🛠️ Configuration

### LLM Provider Configuration
All LLM settings centralized in `utils/llm_config.py`. To change providers:

**Option 1: Edit config.json**
```json
{
  "llm_provider": "claude"  // or "gemini", "openai"
}
```

**Option 2: Use CLI tool**
```bash
python utils/llm_cli.py switch claude
python utils/llm_cli.py status
```

**Environment Variables:**
```bash
export GEMINI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"  
export OPENAI_API_KEY="your_key"
```

