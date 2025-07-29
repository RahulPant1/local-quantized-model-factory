# LQMF API Serving & Experimentation Module

## Overview

The LQMF (Local Quantized Model Factory) now includes comprehensive API serving and model experimentation capabilities. This module allows users to:

- Load quantized models as API endpoints
- Serve models via REST API
- Experiment with models interactively
- Run benchmarks and comparisons
- Test models via chat interface

## Architecture

### Core Components

#### 1. API Server Agent (`api_server_agent.py`)
- **Purpose**: Manages loading/unloading of quantized models and serves them via REST API
- **Key Features**:
  - Automatic model discovery from `quantized-models/` directory
  - Support for all quantization formats (BitsAndBytes, GPTQ, GGUF)
  - FastAPI-based REST API with automatic documentation
  - Memory usage tracking and optimization
  - Background server management

#### 2. Model Experiment Agent (`model_experiment_agent.py`)
- **Purpose**: Provides interactive testing and experimentation interface
- **Key Features**:
  - Single prompt testing
  - Batch prompt testing
  - Conversational testing
  - Benchmark experiments
  - Model comparison experiments
  - Quality scoring and performance metrics

#### 3. Enhanced Decision Agent (`decision_agent.py`)
- **Purpose**: Extended to handle API-related commands with natural language understanding
- **New Intents**:
  - `LOAD_MODEL_API`: Load models for API serving
  - `UNLOAD_MODEL_API`: Unload models from API
  - `START_API_SERVER`: Start the API server
  - `STOP_API_SERVER`: Stop the API server
  - `SHOW_API_STATUS`: Show API and model status
  - `CHAT_WITH_MODEL`: Interactive chat with models
  - `EXPERIMENT_WITH_MODEL`: Run model experiments

#### 4. API CLI (`api_cli.py`)
- **Purpose**: Command-line interface that integrates all API functionality
- **Features**:
  - Interactive command processing
  - Session management
  - Statistics tracking
  - Help system

## Installation

### Dependencies

Install the additional dependencies for API serving:

```bash
pip install fastapi uvicorn requests
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Starting the API CLI

```bash
python cli/api_cli.py
```

### Basic Commands

#### Server Management
```bash
# Start API server
> start server
> start api server

# Stop API server
> stop server
> stop api server

# Check server status
> show api status
> api status
```

#### Model Management
```bash
# Load a model for API serving
> load model api model_name
> serve model model_name

# Unload a model
> unload model model_name

# Show loaded models
> show loaded models

# Show available models
> show api status
```

#### Model Experimentation
```bash
# Chat with a model
> chat with model model_name
> talk to model model_name

# Run experiments
> experiment with model
> benchmark model
> compare models

# Show experiment history
> list experiments
> show statistics
```

### API Endpoints

Once the server is running (default: `http://localhost:8000`), the following endpoints are available:

#### Core Endpoints
- `GET /` - Server status and loaded models
- `GET /docs` - Interactive API documentation
- `GET /models` - List loaded models with details
- `GET /models/available` - List all available quantized models

#### Model Management
- `POST /models/{model_name}/load` - Load a model for serving
- `POST /models/{model_name}/unload` - Unload a model

#### Model Interaction
- `POST /models/{model_name}/chat` - Chat with a loaded model

#### Example API Request
```bash
curl -X POST "http://localhost:8000/models/my_model/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Supported Model Formats

The API server supports all quantization formats available in LQMF:

### BitsAndBytes (BnB)
- **Format**: PyTorch/SafeTensors
- **Bit widths**: 4-bit, 8-bit
- **Loading**: Automatic with device mapping
- **Memory**: Optimized for GPU inference

### GPTQ
- **Format**: SafeTensors
- **Bit widths**: 4-bit (typically)
- **Loading**: Via optimum.gptq
- **Memory**: Efficient GPU utilization

### GGUF
- **Format**: GGUF files
- **Bit widths**: Various (q4_K_M, q5_K_M, etc.)
- **Loading**: Via llama-cpp-python (if available)
- **Memory**: CPU/GPU flexible

## Model Discovery

The system automatically discovers quantized models from the `quantized-models/` directory. Models should be organized as:

```
quantized-models/
├── model_name_method_bitwidth/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
```

Example:
```
quantized-models/
├── Qwen_Qwen2.5-1.5B-Instruct_bnb_4bit/
├── meta-llama_Llama-3.2-1B_bnb_4bit/
└── mistral-7b-instruct_gptq_4bit/
```

## Experimentation Features

### Single Prompt Testing
- Test individual prompts on loaded models
- Configurable parameters (max_tokens, temperature, etc.)
- Quality scoring and performance metrics

### Batch Testing
- Test multiple prompts on a model
- Progress tracking and summary statistics
- Customizable prompt sets

### Conversational Testing
- Multi-turn conversations with models
- Context management
- Natural conversation flow

### Benchmark Testing
- Comprehensive model evaluation
- Standard prompt sets
- Performance metrics (tokens/sec, quality scores)

### Model Comparison
- Compare multiple models on same prompts
- Side-by-side performance analysis
- Ranking and recommendations

## Performance Monitoring

### Metrics Tracked
- **Response Time**: Time to generate responses
- **Token Generation Rate**: Tokens per second
- **Memory Usage**: Model memory footprint
- **Quality Scores**: Heuristic-based response quality
- **Success Rate**: Percentage of successful responses

### Quality Scoring
The system uses heuristic-based quality scoring:
- Length appropriateness (0.3 max)
- Relevance to prompt (0.3 max)
- Coherence and non-repetitiveness (0.4 max)
- Total score: 0.0 to 1.0

## Integration with Existing LQMF

The API module seamlessly integrates with existing LQMF functionality:

### Memory Agent Integration
- Experiment tracking and history
- Performance data storage
- Recovery suggestions

### Decision Agent Integration
- Natural language command processing
- Intent recognition for API commands
- Conversational responses

### Executor Agent Integration
- Uses existing quantized models
- Leverages quantization metadata
- Compatible with all supported formats

## Use Cases

### Development & Testing
- **Rapid Prototyping**: Quickly test model responses
- **A/B Testing**: Compare different quantized versions
- **Performance Tuning**: Optimize quantization parameters

### Production Deployment
- **API Serving**: Serve models via REST API
- **Load Testing**: Benchmark model performance
- **Monitoring**: Track model health and performance

### Research & Analysis
- **Model Comparison**: Compare different models/quantizations
- **Quality Assessment**: Evaluate response quality
- **Performance Analysis**: Analyze speed vs. quality trade-offs

## Configuration

### Server Configuration
```python
# Default settings
HOST = "127.0.0.1"
PORT = 8000
MAX_MODELS = 5  # Limit concurrent loaded models
```

### Model Loading Configuration
```python
# Model loading parameters
DEVICE_MAP = "auto"
TORCH_DTYPE = torch.float16
MAX_MEMORY_USAGE = 8192  # MB
```

## Error Handling

The system includes comprehensive error handling:
- **Model Loading Errors**: Graceful fallback and error reporting
- **API Errors**: Proper HTTP status codes and error messages
- **Resource Errors**: Memory management and cleanup
- **Server Errors**: Automatic recovery and logging

## Security Considerations

- **Local Only**: Server runs on localhost by default
- **No Authentication**: Designed for local development use
- **Resource Limits**: Built-in memory and model limits
- **Input Validation**: Sanitized input handling

## Future Enhancements

### Planned Features
- **Authentication**: Add API key support
- **Model Versioning**: Track model versions and updates
- **Distributed Serving**: Multi-GPU and multi-node support
- **Advanced Metrics**: More sophisticated quality scoring
- **Streaming Responses**: Real-time response streaming

### Integration Opportunities
- **Gradio Interface**: Web-based GUI for model interaction
- **OpenAI API Compatibility**: Drop-in replacement for OpenAI API
- **Container Support**: Docker containerization
- **Cloud Deployment**: Cloud platform integration

## Troubleshooting

### Common Issues

#### Model Loading Fails
```bash
# Check model format
> show api status

# Verify model files exist
ls quantized-models/your_model_name/

# Check memory usage
> show statistics
```

#### Server Won't Start
```bash
# Check port availability
netstat -an | grep 8000

# Install missing dependencies
pip install fastapi uvicorn

# Check error logs
tail -f logs/api_server.log
```

#### Poor Performance
```bash
# Check memory usage
> show statistics

# Unload unused models
> unload model unused_model

# Adjust model parameters
# Edit model configuration
```

### Debugging Commands
```bash
# Enable debug mode
export LQMF_DEBUG=1

# Verbose logging
export LQMF_LOG_LEVEL=DEBUG

# Check system resources
> show statistics
```

## Examples

### Complete Workflow Example

```bash
# 1. Start the API CLI
python cli/api_cli.py

# 2. Check available models
> show api status

# 3. Load a model
> load model api Qwen_Qwen2.5-1.5B-Instruct_bnb_4bit

# 4. Start the server
> start server

# 5. Chat with the model
> chat with model Qwen_Qwen2.5-1.5B-Instruct_bnb_4bit

# 6. Run experiments
> experiment with model

# 7. Check performance
> show statistics

# 8. Stop and exit
> stop server
> exit
```

### API Usage Example

```python
import requests

# Load a model
response = requests.post(
    "http://localhost:8000/models/my_model/load"
)

# Chat with the model
response = requests.post(
    "http://localhost:8000/models/my_model/chat",
    json={
        "message": "Explain quantum computing",
        "max_tokens": 150,
        "temperature": 0.7
    }
)

result = response.json()
print(f"Response: {result['response']}")
print(f"Tokens: {result['tokens_generated']}")
print(f"Time: {result['time_taken']:.2f}s")
```

## Conclusion

The LQMF API Serving & Experimentation Module provides a comprehensive solution for serving and testing quantized models. It combines the power of local quantization with the flexibility of API serving and the depth of interactive experimentation.

The module is designed to be:
- **Easy to use**: Natural language commands and intuitive interface
- **Flexible**: Support for all quantization formats and model types
- **Comprehensive**: Full range of testing and experimentation features
- **Extensible**: Modular architecture for future enhancements

Whether you're developing AI applications, conducting research, or deploying production systems, this module provides the tools you need to work effectively with quantized models.