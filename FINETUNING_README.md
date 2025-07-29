# 🧠 LQMF Fine-Tuning Guide: LoRA/QLoRA Parameter-Efficient Training

The Local Quantized Model Factory (LQMF) now supports lightweight fine-tuning capabilities using LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) for domain-specific customization within 8GB GPU environments.

## 🎯 Overview

This enhancement allows you to:
- **Fine-tune quantized models** with small datasets
- **Hot-swap adapters** during inference without restarting
- **Benchmark performance** of fine-tuned models
- **Manage multiple adapters** efficiently
- **Stay within 8GB GPU limits** through memory optimizations

## 🚀 Quick Start

### 1. Start Fine-Tuning CLI
```bash
python cli/finetuning_cli.py
```

### 2. Train Your First Adapter
```bash
FT> finetune mistral-7b ./examples/datasets/chat_training.csv chat
```

### 3. Load and Test Adapter
```bash
FT> load adapter mistral-7b-chat-xxxxx
FT> benchmark mistral-7b-chat-xxxxx
```

## 📋 System Requirements

- **GPU**: 8GB+ VRAM (GTX 3070, RTX 4060, etc.)
- **RAM**: 16GB+ system memory
- **Storage**: 5GB+ free space for adapters
- **Python**: 3.8+ with PyTorch 2.0+

### Required Dependencies
```bash
pip install peft transformers bitsandbytes datasets accelerate
```

## 🔧 Fine-Tuning Process

### Step 1: Prepare Your Dataset

LQMF supports multiple dataset formats:

#### Chat Format (CSV)
```csv
input,output
"What is AI?","Artificial Intelligence is..."
"Explain ML","Machine Learning is..."
```

#### Instruction Following (JSONL)
```jsonl
{"instruction": "Summarize this text", "input": "Long text...", "output": "Summary..."}
{"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}
```

#### Classification (CSV)
```csv
text,label
"Great product!",positive
"Poor quality",negative
```

### Step 2: Configure Training Parameters

#### Basic Configuration (Recommended for 8GB GPU)
```bash
FT> finetune mistral-7b dataset.csv chat
# Uses optimized defaults:
# - LoRA rank: 16
# - Batch size: 2
# - Gradient accumulation: 8
# - 4-bit quantization: enabled
# - Gradient checkpointing: enabled
```

#### Advanced Configuration
```bash
FT> config create my_training
# Edit the generated config file, then:
FT> config edit my_training
```

### Step 3: Monitor Training

Training automatically includes:
- **Real-time memory monitoring**
- **Progress tracking with ETA**
- **Automatic memory cleanup**
- **Performance metrics logging**

## 🎛️ CLI Commands Reference

### Fine-Tuning Commands
| Command | Description | Example |
|---------|-------------|---------|
| `finetune <model> <dataset> <task>` | Start training | `finetune mistral-7b data.csv chat` |
| `config create <name>` | Create config template | `config create finance_model` |
| `config edit <name>` | Use saved configuration | `config edit finance_model` |

### Adapter Management
| Command | Description | Example |
|---------|-------------|---------|
| `list adapters` | Show all trained adapters | `list adapters` |
| `load adapter <name>` | Load adapter for inference | `load adapter finance-style` |
| `unload adapter` | Unload current adapter | `unload adapter` |
| `switch adapter <name>` | Hot-swap to different adapter | `switch adapter legal-style` |
| `export adapter <name> <path>` | Export adapter | `export adapter finance-style ./exports/` |

### Testing & Benchmarking
| Command | Description | Example |
|---------|-------------|---------|
| `benchmark <adapter>` | Run performance benchmark | `benchmark finance-style` |
| `test <adapter> "<prompt>"` | Test single prompt | `test finance-style "Explain ROI"` |

### System Commands
| Command | Description | Example |
|---------|-------------|---------|
| `status` | Show system status | `status` |
| `list models` | Show available base models | `list models` |
| `clear` | Clear screen | `clear` |
| `help` | Show command help | `help` |

## 📊 Supported Task Types

### 1. Chat Fine-Tuning
Perfect for creating conversational AI with specific personalities or expertise.

**Dataset Format:**
```csv
input,output
"How do I invest in stocks?","To invest in stocks, start by..."
"What's a portfolio?","A portfolio is a collection..."
```

**Use Cases:**
- Customer support bots
- Domain expert assistants
- Personality-driven chatbots

### 2. Instruction Following
Train models to follow specific instruction patterns and formats.

**Dataset Format:**
```jsonl
{"instruction": "Write an email", "input": "Decline meeting politely", "output": "Dear [Name]..."}
{"instruction": "Analyze data", "input": "Sales increased 20%", "output": "Analysis shows..."}
```

**Use Cases:**
- Task-specific assistants
- Content generation tools
- Code documentation helpers

### 3. Text Classification
Fine-tune for categorizing text into predefined classes.

**Dataset Format:**
```csv
text,label
"Outstanding service!",positive
"Poor experience",negative
"Average quality",neutral
```

**Use Cases:**
- Sentiment analysis
- Content moderation
- Document categorization

### 4. Summarization
Train models to create concise summaries of longer texts.

**Dataset Format:**
```csv
text,summary
"Long article text...","Brief summary..."
"Complex report...","Key points..."
```

**Use Cases:**
- Document summarization
- News article condensing
- Report generation

## ⚡ Memory Optimization Features

### Automatic 8GB GPU Optimization

LQMF automatically applies these optimizations:

1. **4-bit Quantization (QLoRA)**
   - Reduces memory usage by 75%
   - Maintains training quality
   - Automatically enabled

2. **Gradient Checkpointing**
   - Trades compute for memory
   - Reduces VRAM usage significantly
   - Minimal performance impact

3. **Smart Batch Sizing**
   - Auto-detects optimal batch size
   - Uses gradient accumulation
   - Prevents OOM errors

4. **Memory Monitoring**
   - Real-time VRAM tracking
   - Automatic cleanup
   - Leak detection

### Manual Memory Management

```python
from utils.memory_optimizer import MemoryOptimizer

optimizer = MemoryOptimizer(target_gpu_memory_gb=7.5)

# Monitor memory during training
with optimizer.memory_monitor("Training"):
    # Your training code here
    pass

# Get optimization suggestions
suggestions = optimizer.suggest_optimizations()
```

## 🔄 Adapter Hot-Swapping

One of LQMF's key features is the ability to dynamically switch between adapters without restarting the inference server.

### Loading Adapters
```bash
# Start API server with base model
python cli/api_cli.py
API> load mistral-7b

# Switch to fine-tuning CLI
python cli/finetuning_cli.py
FT> load adapter finance-style
```

### Switching Adapters
```bash
FT> switch adapter legal-style
# Model now uses legal-style adapter
# No server restart required!
```

### Performance Impact
- **Load time**: 2-5 seconds
- **Memory overhead**: ~50-200MB per adapter
- **Inference speed**: Minimal impact (~5%)

## 📈 Benchmarking & Evaluation

### Automated Benchmarking

```bash
FT> benchmark finance-style
```

**Metrics Collected:**
- Response time (avg, min, max)
- Tokens per second
- Memory usage
- Quality scores (when applicable)

### Custom Benchmark Prompts

```bash
# Create test_prompts.txt with your prompts
FT> benchmark finance-style test_prompts.txt
```

### Comparative Analysis

Load multiple benchmark results to compare adapter performance:

```python
from agents.benchmark_agent import BenchmarkAgent

agent = BenchmarkAgent()
results = agent.load_benchmark_results("finance*")
comparison_table = agent.compare_models(results)
```

## 📁 File Structure

```
my_slm_factory_app/
├── agents/
│   ├── finetuning_agent.py      # Core fine-tuning logic
│   ├── adapter_manager.py       # Adapter hot-swapping
│   ├── benchmark_agent.py       # Performance testing
│   └── api_server_agent.py      # Enhanced with adapter support
├── cli/
│   ├── finetuning_cli.py        # Fine-tuning CLI interface
│   └── api_cli.py               # Enhanced API CLI
├── utils/
│   └── memory_optimizer.py      # 8GB GPU optimizations
├── examples/
│   └── datasets/                # Sample training datasets
├── adapters/                    # Trained adapters storage
├── benchmarks/                  # Benchmark results
└── reports/                     # Generated reports
```

## 🎯 Best Practices

### Dataset Preparation
1. **Size**: 50-1000 examples per task
2. **Quality**: High-quality, diverse examples
3. **Balance**: Even distribution across categories
4. **Validation**: Reserve 10-20% for testing

### Training Configuration
1. **Start Small**: Use default settings first
2. **Monitor Memory**: Watch GPU usage
3. **Validate Early**: Test after each epoch
4. **Save Checkpoints**: Enable automatic saving

### Adapter Management
1. **Naming**: Use descriptive names (domain-task-version)
2. **Versioning**: Keep track of training data versions
3. **Testing**: Always benchmark new adapters
4. **Cleanup**: Remove unused adapters regularly

## 🔧 Advanced Configuration

### Custom LoRA Configuration

```json
{
  "job_name": "custom_finance_model",
  "base_model": "mistral-7b",
  "task_type": "chat",
  "dataset_path": "./data/finance_chat.csv",
  "lora_r": 32,
  "lora_alpha": 64,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "learning_rate": 1e-4,
  "num_train_epochs": 5,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16
}
```

### Memory-Constrained Settings

For 6GB GPUs or lower:
```json
{
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "max_seq_length": 256,
  "use_gradient_checkpointing": true,
  "use_4bit_quantization": true
}
```

## 🐛 Troubleshooting

### Common Issues

#### Out of Memory Errors
```bash
# Reduce batch size
FT> config create small_batch
# Edit config: per_device_train_batch_size: 1
# Edit config: gradient_accumulation_steps: 16
```

#### Slow Training
```bash
# Check GPU utilization
FT> status
# Enable mixed precision if not already enabled
```

#### Adapter Not Loading
```bash
# Check adapter registry
FT> list adapters
# Verify base model compatibility
FT> list models
```

### Performance Optimization

1. **GPU Utilization**: Aim for 80-90% GPU memory usage
2. **Batch Size**: Increase until you hit memory limits
3. **Sequence Length**: Use shortest acceptable length
4. **Learning Rate**: Start with 2e-4, adjust based on loss

## 📚 Example Workflows

### Workflow 1: Customer Support Bot
```bash
# 1. Prepare customer service conversations
# 2. Train adapter
FT> finetune mistral-7b support_data.csv chat

# 3. Load adapter
FT> load adapter mistral-7b-chat-xxxxx

# 4. Test and benchmark
FT> test mistral-7b-chat-xxxxx "How do I return an item?"
FT> benchmark mistral-7b-chat-xxxxx
```

### Workflow 2: Domain Expert Assistant
```bash
# 1. Create instruction dataset
FT> finetune llama-7b legal_instructions.jsonl instruction_following

# 2. Compare with base model
FT> benchmark llama-7b-instruction-xxxxx

# 3. Deploy for inference
# Switch to API CLI
API> load adapter llama-7b-instruction-xxxxx
```

### Workflow 3: Content Classification
```bash
# 1. Prepare labeled dataset
FT> finetune bert-base sentiment_data.csv classification

# 2. Evaluate performance
FT> benchmark bert-base-classification-xxxxx classification_test.txt

# 3. Export for production
FT> export adapter bert-base-classification-xxxxx ./production/
```

## 🔗 Integration with Main LQMF

The fine-tuning capabilities seamlessly integrate with existing LQMF features:

1. **Model Quantization**: Use pre-quantized models as base
2. **API Serving**: Hot-swap adapters in running server
3. **Memory Management**: Shared optimization strategies
4. **CLI Interface**: Unified command structure

## 📞 Support & Community

- **Documentation**: Check the main LQMF README
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Join the LQMF community discussions
- **Examples**: More examples in `/examples/` directory

## 🔮 Future Enhancements

Planned features for upcoming releases:
- **Multi-adapter inference**: Use multiple adapters simultaneously
- **Federated fine-tuning**: Collaborative training across devices
- **Automated hyperparameter tuning**: Smart parameter optimization
- **Model compression**: Further reduce adapter sizes
- **Real-time training**: Continuous learning from user interactions

---

## 📝 License & Contributing

This fine-tuning enhancement follows the same license and contribution guidelines as the main LQMF project. Contributions are welcome!

---

*Happy fine-tuning! 🚀*