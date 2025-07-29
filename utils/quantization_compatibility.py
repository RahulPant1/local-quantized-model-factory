"""
Quantization Compatibility Detection
Analyzes HuggingFace models to suggest appropriate quantization methods
"""

from huggingface_hub import model_info
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class QuantMethod(Enum):
    GPTQ = "gptq"
    GGUF = "gguf" 
    BITSANDBYTES = "bnb"

@dataclass
class QuantizationRecommendation:
    """Recommendation for quantization method"""
    method: QuantMethod
    confidence: float  # 0.0 to 1.0
    reason: str
    bit_widths: List[int]  # Supported bit widths
    target_hardware: str  # "gpu", "cpu", "both"

@dataclass
class ModelCompatibility:
    """Model quantization compatibility information"""
    model_id: str
    architecture: str
    model_type: str
    recommendations: List[QuantizationRecommendation]
    warnings: List[str]
    already_quantized: bool

# Architecture-based compatibility matrix
ARCHITECTURE_COMPATIBILITY = {
    'llama': {
        QuantMethod.GPTQ: QuantizationRecommendation(
            method=QuantMethod.GPTQ,
            confidence=0.95,
            reason="LLaMA models have excellent GPTQ support with high quality",
            bit_widths=[4, 8],
            target_hardware="gpu"
        ),
        QuantMethod.GGUF: QuantizationRecommendation(
            method=QuantMethod.GGUF,
            confidence=0.95,
            reason="LLaMA is the primary target for GGUF quantization",
            bit_widths=[4, 5, 6, 8],
            target_hardware="cpu"
        ),
        QuantMethod.BITSANDBYTES: QuantizationRecommendation(
            method=QuantMethod.BITSANDBYTES,
            confidence=0.9,
            reason="BitsAndBytes supports all LLaMA variants",
            bit_widths=[4, 8],
            target_hardware="gpu"
        )
    },
    'mistral': {
        QuantMethod.GPTQ: QuantizationRecommendation(
            method=QuantMethod.GPTQ,
            confidence=0.95,
            reason="Mistral has excellent GPTQ support",
            bit_widths=[4, 8],
            target_hardware="gpu"
        ),
        QuantMethod.GGUF: QuantizationRecommendation(
            method=QuantMethod.GGUF,
            confidence=0.9,
            reason="Mistral is well-supported in GGUF format",
            bit_widths=[4, 5, 6, 8],
            target_hardware="cpu"
        ),
        QuantMethod.BITSANDBYTES: QuantizationRecommendation(
            method=QuantMethod.BITSANDBYTES,
            confidence=0.9,
            reason="BitsAndBytes works well with Mistral",
            bit_widths=[4, 8],
            target_hardware="gpu"
        )
    },
    'qwen': {
        QuantMethod.GPTQ: QuantizationRecommendation(
            method=QuantMethod.GPTQ,
            confidence=0.85,
            reason="Qwen models support GPTQ quantization",
            bit_widths=[4, 8],
            target_hardware="gpu"
        ),
        QuantMethod.GGUF: QuantizationRecommendation(
            method=QuantMethod.GGUF,
            confidence=0.8,
            reason="Qwen models can be converted to GGUF",
            bit_widths=[4, 5, 6, 8],
            target_hardware="cpu"
        ),
        QuantMethod.BITSANDBYTES: QuantizationRecommendation(
            method=QuantMethod.BITSANDBYTES,
            confidence=0.9,
            reason="BitsAndBytes is reliable for Qwen models",
            bit_widths=[4, 8],
            target_hardware="gpu"
        )
    },
    'gpt2': {
        QuantMethod.GPTQ: QuantizationRecommendation(
            method=QuantMethod.GPTQ,
            confidence=0.7,
            reason="GPT-2 can be quantized with GPTQ but less optimized",
            bit_widths=[4, 8],
            target_hardware="gpu"
        ),
        QuantMethod.BITSANDBYTES: QuantizationRecommendation(
            method=QuantMethod.BITSANDBYTES,
            confidence=0.85,
            reason="BitsAndBytes is the most reliable for GPT-2",
            bit_widths=[4, 8],
            target_hardware="gpu"
        )
    },
    'falcon': {
        QuantMethod.GPTQ: QuantizationRecommendation(
            method=QuantMethod.GPTQ,
            confidence=0.8,
            reason="Falcon models support GPTQ quantization",
            bit_widths=[4, 8],
            target_hardware="gpu"
        ),
        QuantMethod.GGUF: QuantizationRecommendation(
            method=QuantMethod.GGUF,
            confidence=0.75,
            reason="Falcon can be converted to GGUF with some limitations",
            bit_widths=[4, 6, 8],
            target_hardware="cpu"
        ),
        QuantMethod.BITSANDBYTES: QuantizationRecommendation(
            method=QuantMethod.BITSANDBYTES,
            confidence=0.85,
            reason="BitsAndBytes is safe choice for Falcon",
            bit_widths=[4, 8],
            target_hardware="gpu"
        )
    },
    # Encoder models - limited quantization options
    'bert': {
        QuantMethod.BITSANDBYTES: QuantizationRecommendation(
            method=QuantMethod.BITSANDBYTES,
            confidence=0.8,
            reason="BitsAndBytes is the primary option for BERT models",
            bit_widths=[8],  # 4-bit often too aggressive for BERT
            target_hardware="gpu"
        )
    },
    't5': {
        QuantMethod.BITSANDBYTES: QuantizationRecommendation(
            method=QuantMethod.BITSANDBYTES,
            confidence=0.8,
            reason="BitsAndBytes works for T5 encoder-decoder models",
            bit_widths=[8],
            target_hardware="gpu"
        )
    }
}

def get_model_architecture_info(model_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Get model architecture and type from HuggingFace
    Returns: (architecture, model_type)
    """
    try:
        info = model_info(model_id)
        
        # Get architecture from config
        architecture = None
        model_type = None
        
        if hasattr(info, 'config') and info.config:
            # Get model type (primary identifier)
            model_type = info.config.get('model_type', '').lower()
            
            # Get architecture list and extract base architecture
            architectures = info.config.get('architectures', [])
            if architectures:
                arch_str = architectures[0].lower()
                # Extract base architecture name
                for base_arch in ['llama', 'mistral', 'qwen', 'gpt2', 'falcon', 'bert', 't5']:
                    if base_arch in arch_str:
                        architecture = base_arch
                        break
        
        return architecture, model_type
        
    except Exception as e:
        print(f"Warning: Could not fetch model info for {model_id}: {e}")
        return None, None

def check_if_already_quantized(model_id: str) -> Tuple[bool, List[str]]:
    """
    Check if model is already quantized
    Returns: (is_quantized, quantization_indicators)
    """
    try:
        info = model_info(model_id)
        
        # Check files for quantization indicators
        files = [f.rfilename for f in info.siblings] if hasattr(info, 'siblings') else []
        
        quant_indicators = []
        is_quantized = False
        
        for file in files:
            file_lower = file.lower()
            if '.gguf' in file_lower:
                quant_indicators.append('GGUF format detected')
                is_quantized = True
            elif 'gptq' in file_lower:
                quant_indicators.append('GPTQ quantization detected')
                is_quantized = True
            elif any(x in file_lower for x in ['quantized', 'int4', 'int8', 'bnb']):
                quant_indicators.append('Quantization files detected')
                is_quantized = True
        
        # Check tags for quantization hints
        if hasattr(info, 'tags') and info.tags:
            quant_tags = [tag for tag in info.tags if any(q in tag.lower() for q in ['quant', 'gptq', 'gguf', 'bnb', 'bit'])]
            if quant_tags:
                quant_indicators.extend([f"Tag: {tag}" for tag in quant_tags])
                is_quantized = True
        
        return is_quantized, quant_indicators
        
    except Exception:
        return False, []

def analyze_quantization_compatibility(model_id: str) -> ModelCompatibility:
    """
    Analyze a model for quantization compatibility and provide recommendations
    """
    architecture, model_type = get_model_architecture_info(model_id)
    is_quantized, quant_indicators = check_if_already_quantized(model_id)
    
    recommendations = []
    warnings = []
    
    if is_quantized:
        warnings.append("Model appears to be already quantized")
        warnings.extend(quant_indicators)
    
    # Get recommendations based on architecture
    if architecture and architecture in ARCHITECTURE_COMPATIBILITY:
        # Sort by confidence (highest first)
        arch_recommendations = list(ARCHITECTURE_COMPATIBILITY[architecture].values())
        arch_recommendations.sort(key=lambda x: x.confidence, reverse=True)
        recommendations = arch_recommendations
    else:
        # Unknown architecture - provide conservative recommendations
        warnings.append(f"Unknown architecture: {architecture or 'N/A'}")
        
        # Default to BitsAndBytes as safest option
        recommendations = [
            QuantizationRecommendation(
                method=QuantMethod.BITSANDBYTES,
                confidence=0.6,
                reason="BitsAndBytes is most compatible with unknown architectures",
                bit_widths=[8],  # Conservative bit width
                target_hardware="gpu"
            )
        ]
    
    return ModelCompatibility(
        model_id=model_id,
        architecture=architecture or "unknown",
        model_type=model_type or "unknown",
        recommendations=recommendations,
        warnings=warnings,
        already_quantized=is_quantized
    )

def get_quantization_suggestions(model_id: str, target_hardware: str = "auto") -> str:
    """
    Get human-readable quantization suggestions for a model
    """
    compatibility = analyze_quantization_compatibility(model_id)
    
    output = [f"\n🔍 Quantization Analysis for: {model_id}"]
    output.append(f"Architecture: {compatibility.architecture}")
    output.append(f"Model Type: {compatibility.model_type}")
    
    if compatibility.warnings:
        output.append("\n⚠️  Warnings:")
        for warning in compatibility.warnings:
            output.append(f"   • {warning}")
    
    if compatibility.already_quantized:
        output.append("\n✅ This model appears to be already quantized!")
        return "\n".join(output)
    
    output.append(f"\n🎯 Recommended Quantization Methods:")
    
    # Filter by target hardware if specified
    filtered_recommendations = compatibility.recommendations
    if target_hardware != "auto":
        filtered_recommendations = [
            rec for rec in compatibility.recommendations 
            if rec.target_hardware == target_hardware or rec.target_hardware == "both"
        ]
    
    for i, rec in enumerate(filtered_recommendations[:3], 1):  # Top 3 recommendations
        confidence_emoji = "🟢" if rec.confidence >= 0.9 else "🟡" if rec.confidence >= 0.7 else "🟠"
        output.append(f"   {i}. {confidence_emoji} {rec.method.value.upper()} ({rec.confidence:.0%} confidence)")
        output.append(f"      💡 {rec.reason}")
        output.append(f"      🎚️  Bit widths: {', '.join(map(str, rec.bit_widths))}")
        output.append(f"      🖥️  Target: {rec.target_hardware.upper()}")
        output.append("")
    
    # Add general guidance
    output.append("💡 General Guidance:")
    output.append("   • GPTQ: Best for GPU inference, fast loading")
    output.append("   • GGUF: Best for CPU inference, cross-platform")
    output.append("   • BitsAndBytes: Most compatible, good for experimentation")
    
    return "\n".join(output)

def main():
    """Test the compatibility checker"""
    test_models = [
        "microsoft/DialoGPT-small",
        "Qwen/Qwen2.5-1.5B-Instruct", 
        "meta-llama/Llama-3.2-1B",
        "mistralai/Mistral-7B-Instruct-v0.1"
    ]
    
    for model_id in test_models:
        print(get_quantization_suggestions(model_id))
        print("-" * 80)

if __name__ == "__main__":
    main()