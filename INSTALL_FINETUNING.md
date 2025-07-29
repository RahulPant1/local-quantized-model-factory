# 🚀 LQMF Fine-Tuning Installation Guide

This guide will help you install and set up the LoRA/QLoRA fine-tuning capabilities for LQMF.

## ✅ **System is Ready!**

Your LQMF system has been successfully enhanced with fine-tuning capabilities. Here's what was installed:

### 📦 **Dependencies Installed**
- ✅ **PEFT v0.16.0** - Parameter Efficient Fine-Tuning library
- ✅ **Datasets** - Hugging Face datasets library
- ✅ **Rouge-Score** - Text quality evaluation metrics
- ✅ **BitsAndBytes** - Quantization support (already installed)
- ✅ **Transformers** - Core model library (already installed)
- ✅ **PyTorch** - Deep learning framework (already installed)

### 🎯 **System Verification Complete**
```
GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8.0GB) ✅
Python: 3.11.4 ✅
PyTorch: 2.5.1+cu124 ✅
All dependencies: ✅
Example datasets: ✅
```

## 🚀 **Quick Start**

### 1. **Start Fine-Tuning**
```bash
python cli/finetuning_cli.py
```

### 2. **Try Your First Training**
```bash
FT> finetune mistral-7b examples/datasets/chat_training.csv chat
```

### 3. **Test Your Adapter**
```bash
FT> list adapters
FT> load adapter [your_adapter_name]
FT> benchmark [your_adapter_name]
```

## 📋 **Available Commands**

| Category | Command | Description |
|----------|---------|-------------|
| **Training** | `finetune <model> <dataset> <task>` | Start fine-tuning |
| **Management** | `list adapters` | Show trained adapters |
| **Management** | `load adapter <name>` | Load adapter for inference |
| **Management** | `switch adapter <name>` | Hot-swap adapters |
| **Testing** | `benchmark <adapter>` | Performance testing |
| **Export** | `export adapter <name> <path>` | Export adapter |

## 📁 **Example Datasets Available**

1. **Chat Training** (`examples/datasets/chat_training.csv`)
   - 20 Q&A pairs about machine learning
   - Perfect for conversational AI training

2. **Instruction Following** (`examples/datasets/instruction_following.jsonl`)
   - 15 diverse instruction-response pairs
   - Great for task-specific assistants

3. **Classification** (`examples/datasets/classification_training.csv`)
   - 30 sentiment analysis examples
   - Ideal for text classification tasks

## 🎛️ **Memory Optimization**

Your 8GB GPU setup is automatically optimized with:
- **Batch size**: 1-2 (optimal for 8GB)
- **4-bit quantization**: Enabled (QLoRA)
- **Gradient checkpointing**: Enabled
- **Sequence length**: 256-512 tokens

## 🔧 **Advanced Setup (Optional)**

### Custom Configuration
```bash
FT> config create my_custom_training
# Edit the generated config file
FT> config edit my_custom_training
```

### Integration with API Server
```bash
# Terminal 1: Start API server
python cli/api_cli.py
API> load mistral-7b

# Terminal 2: Load adapters
python cli/finetuning_cli.py
FT> load adapter finance-style
FT> switch adapter legal-style  # Hot-swap!
```

## 📚 **Documentation**

- **Complete Guide**: `FINETUNING_README.md`
- **Examples**: `/examples/datasets/`
- **Architecture**: Main README.md

## 🐛 **Troubleshooting**

### If you see import errors:
```bash
python install_finetuning_deps.py
```

### If you have memory issues:
```bash
FT> status  # Check memory usage
# Reduce batch size in config
```

### If training is slow:
- Ensure GPU is being used
- Check `nvidia-smi` for GPU utilization
- Consider reducing sequence length

## 🎉 **You're Ready!**

The LQMF fine-tuning system is now fully operational. Start with the example datasets to get familiar with the workflow, then move on to your own data.

### Next Steps:
1. **Try the examples**: Start with provided datasets
2. **Create your data**: Follow the format in examples
3. **Experiment**: Try different task types and parameters
4. **Share**: Export adapters to share with others

**Happy fine-tuning! 🧠✨**