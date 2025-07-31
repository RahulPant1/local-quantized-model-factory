"""
Dataset Intelligence Agent - Phase 1: AI-Powered Intelligence
Provides AI-powered dataset analysis and recommendations for fine-tuning.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from rich.console import Console
import logging

# Set up logging and console
console = Console()
logger = logging.getLogger(__name__)

class DatasetFormat(Enum):
    CSV = "csv"
    JSONL = "jsonl"
    JSON = "json"
    PARQUET = "parquet"
    UNKNOWN = "unknown"

class TaskType(Enum):
    CHAT = "chat"
    INSTRUCTION_FOLLOWING = "instruction_following"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    Q_AND_A = "q_and_a"
    CODE_GENERATION = "code_generation"
    UNKNOWN = "unknown"

@dataclass
class DatasetStats:
    """Basic dataset statistics"""
    total_samples: int
    columns: List[str]
    missing_values: Dict[str, int]
    data_types: Dict[str, str]
    file_size_mb: float
    format: DatasetFormat
    
@dataclass
class QualityMetrics:
    """Dataset quality assessment metrics"""
    completeness_score: float  # 0-1, percentage of complete rows
    balance_score: float  # 0-1, how balanced the dataset is
    diversity_score: float  # 0-1, text diversity measure
    length_consistency: float  # 0-1, consistency in text lengths
    quality_issues: List[str]  # List of identified issues
    
@dataclass
class DatasetAnalysis:
    """Complete dataset analysis result"""
    stats: DatasetStats
    quality: QualityMetrics
    task_type: TaskType
    recommendations: List[str]
    preprocessing_steps: List[str]
    augmentation_suggestions: List[str]
    train_val_split: Dict[str, float]
    estimated_training_samples: int
    confidence: float  # AI analysis confidence 0-1

@dataclass
class AugmentationStrategy:
    """Data augmentation recommendation"""
    technique: str
    description: str
    implementation: str
    expected_gain: str
    difficulty: str  # easy, medium, hard
    
class DatasetIntelligenceAgent:
    """AI-powered dataset analysis and optimization agent"""
    
    def __init__(self):
        from utils.llm_config import get_llm_manager
        self.llm_manager = get_llm_manager()
        
    def analyze_dataset(self, dataset_path: str, task_type: Optional[str] = None) -> DatasetAnalysis:
        """
        Comprehensive dataset analysis with AI-powered insights
        
        Args:
            dataset_path: Path to the dataset file
            task_type: Optional hint about the task type
            
        Returns:
            DatasetAnalysis with complete analysis results
        """
        console.print(f"[blue]🔍 Analyzing dataset: {dataset_path}[/blue]")
        
        try:
            # Basic file analysis
            stats = self._get_basic_stats(dataset_path)
            
            # Load and analyze content
            data = self._load_dataset(dataset_path, stats.format)
            
            # Detect task type if not provided
            detected_task_type = self._detect_task_type(data, stats.columns, task_type)
            
            # Calculate quality metrics
            quality = self._calculate_quality_metrics(data, detected_task_type)
            
            # Get AI-powered insights
            ai_insights = self._get_ai_insights(stats, data, detected_task_type, quality)
            
            # Create comprehensive analysis
            analysis = DatasetAnalysis(
                stats=stats,
                quality=quality,
                task_type=detected_task_type,
                recommendations=ai_insights.get('recommendations', []),
                preprocessing_steps=ai_insights.get('preprocessing_steps', []),
                augmentation_suggestions=ai_insights.get('augmentation_suggestions', []),
                train_val_split=ai_insights.get('train_val_split', {'train': 0.8, 'val': 0.2}),
                estimated_training_samples=ai_insights.get('estimated_training_samples', len(data)),
                confidence=ai_insights.get('confidence', 0.8)
            )
            
            console.print(f"[green]✅ Dataset analysis complete![/green]")
            return analysis
            
        except Exception as e:
            console.print(f"[red]❌ Dataset analysis failed: {e}[/red]")
            logger.error(f"Dataset analysis error: {e}")
            raise
    
    def _get_basic_stats(self, dataset_path: str) -> DatasetStats:
        """Get basic dataset statistics"""
        path = Path(dataset_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Determine format
        format_map = {
            '.csv': DatasetFormat.CSV,
            '.jsonl': DatasetFormat.JSONL,
            '.json': DatasetFormat.JSON,
            '.parquet': DatasetFormat.PARQUET
        }
        
        file_format = format_map.get(path.suffix.lower(), DatasetFormat.UNKNOWN)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        
        # Quick column analysis
        try:
            if file_format == DatasetFormat.CSV:
                df = pd.read_csv(dataset_path, nrows=100)  # Sample for speed
                columns = df.columns.tolist()
                data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
                missing_values = df.isnull().sum().to_dict()
                total_samples = len(pd.read_csv(dataset_path))
            elif file_format == DatasetFormat.JSONL:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    first_line = json.loads(f.readline())
                    columns = list(first_line.keys())
                
                # Count total lines
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    total_samples = sum(1 for _ in f)
                
                data_types = {col: "object" for col in columns}
                missing_values = {col: 0 for col in columns}  # Will calculate properly later
            else:
                columns = []
                data_types = {}
                missing_values = {}
                total_samples = 0
                
        except Exception as e:
            logger.warning(f"Could not analyze file structure: {e}")
            columns = []
            data_types = {}
            missing_values = {}
            total_samples = 0
        
        return DatasetStats(
            total_samples=total_samples,
            columns=columns,
            missing_values=missing_values,
            data_types=data_types,
            file_size_mb=file_size_mb,
            format=file_format
        )
    
    def _load_dataset(self, dataset_path: str, format: DatasetFormat) -> pd.DataFrame:
        """Load dataset based on format"""
        try:
            if format == DatasetFormat.CSV:
                return pd.read_csv(dataset_path)
            elif format == DatasetFormat.JSONL:
                data = []
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                return pd.DataFrame(data)
            elif format == DatasetFormat.JSON:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            elif format == DatasetFormat.PARQUET:
                return pd.read_parquet(dataset_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")
    
    def _detect_task_type(self, data: pd.DataFrame, columns: List[str], hint: Optional[str] = None) -> TaskType:
        """Detect the likely task type from dataset structure"""
        if hint:
            try:
                return TaskType(hint.lower())
            except ValueError:
                pass
        
        columns_lower = [col.lower() for col in columns]
        
        # Chat detection
        if any(col in columns_lower for col in ['input', 'output', 'user', 'assistant', 'human', 'ai']):
            return TaskType.CHAT
        
        # Instruction following detection
        if any(col in columns_lower for col in ['instruction', 'response', 'system']):
            return TaskType.INSTRUCTION_FOLLOWING
        
        # Classification detection
        if any(col in columns_lower for col in ['label', 'class', 'category', 'sentiment']):
            return TaskType.CLASSIFICATION
        
        # Summarization detection
        if any(col in columns_lower for col in ['text', 'summary', 'article', 'document']):
            return TaskType.SUMMARIZATION
        
        # Q&A detection
        if any(col in columns_lower for col in ['question', 'answer', 'context']):
            return TaskType.Q_AND_A
        
        # Code generation detection
        if any(col in columns_lower for col in ['code', 'function', 'program', 'solution']):
            return TaskType.CODE_GENERATION
        
        return TaskType.UNKNOWN
    
    def _calculate_quality_metrics(self, data: pd.DataFrame, task_type: TaskType) -> QualityMetrics:
        """Calculate dataset quality metrics"""
        try:
            # Completeness score
            total_cells = data.size
            complete_cells = total_cells - data.isnull().sum().sum()
            completeness_score = complete_cells / total_cells if total_cells > 0 else 0
            
            # Balance score (for classification tasks)
            balance_score = 1.0  # Default for non-classification
            if task_type == TaskType.CLASSIFICATION and 'label' in data.columns:
                label_counts = data['label'].value_counts()
                if len(label_counts) > 1:
                    balance_score = 1 - (label_counts.std() / label_counts.mean())
                    balance_score = max(0, min(1, balance_score))
            
            # Text length consistency (for text columns)
            length_consistency = 1.0
            text_columns = []
            
            for col in data.columns:
                if data[col].dtype == 'object':
                    sample_values = data[col].dropna().head(100)
                    if sample_values.apply(lambda x: isinstance(x, str) and len(x) > 10).any():
                        text_columns.append(col)
            
            if text_columns:
                lengths = []
                for col in text_columns:
                    text_lengths = data[col].dropna().astype(str).str.len()
                    lengths.extend(text_lengths.tolist())
                
                if lengths:
                    length_std = np.std(lengths)
                    length_mean = np.mean(lengths)
                    length_consistency = max(0, 1 - (length_std / length_mean) if length_mean > 0 else 0)
            
            # Diversity score (simple approximation)
            diversity_score = min(1.0, len(data) / 1000)  # More samples = more diversity (capped at 1.0)
            
            # Quality issues
            quality_issues = []
            if completeness_score < 0.9:
                quality_issues.append(f"Missing data: {(1-completeness_score)*100:.1f}% incomplete")
            if balance_score < 0.7 and task_type == TaskType.CLASSIFICATION:
                quality_issues.append("Imbalanced classes detected")
            if length_consistency < 0.5:
                quality_issues.append("Highly variable text lengths")
            if len(data) < 100:
                quality_issues.append("Small dataset size (< 100 samples)")
            
            return QualityMetrics(
                completeness_score=completeness_score,
                balance_score=balance_score,
                diversity_score=diversity_score,
                length_consistency=length_consistency,
                quality_issues=quality_issues
            )
            
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
            return QualityMetrics(
                completeness_score=0.5,
                balance_score=0.5,
                diversity_score=0.5,
                length_consistency=0.5,
                quality_issues=["Could not calculate quality metrics"]
            )
    
    def _get_ai_insights(self, stats: DatasetStats, data: pd.DataFrame, 
                        task_type: TaskType, quality: QualityMetrics) -> Dict[str, Any]:
        """Get AI-powered insights about the dataset"""
        if not self.llm_manager.is_available():
            return self._fallback_insights(stats, data, task_type, quality)
        
        # Prepare dataset info for LLM
        dataset_info = {
            "stats": asdict(stats),
            "task_type": task_type.value,
            "quality_metrics": asdict(quality),
            "sample_data": self._get_data_sample(data, 5)  # First 5 rows
        }
        
        try:
            response = self.llm_manager.analyze_dataset_for_finetuning(dataset_info)
            
            if response.success:
                # Try to extract structured insights from LLM response
                insights = self._parse_ai_response(response.content, stats, quality)
                return insights
            else:
                logger.warning(f"LLM analysis failed: {response.error_message}")
                return self._fallback_insights(stats, data, task_type, quality)
                
        except Exception as e:
            logger.warning(f"AI insights generation failed: {e}")
            return self._fallback_insights(stats, data, task_type, quality)
    
    def _get_data_sample(self, data: pd.DataFrame, n: int = 5) -> List[Dict]:
        """Get a sample of the data for LLM analysis"""
        try:
            sample = data.head(n)
            return sample.to_dict('records')
        except Exception:
            return []
    
    def _parse_ai_response(self, response_content: str, stats: DatasetStats, quality: QualityMetrics) -> Dict[str, Any]:
        """Parse AI response and extract structured insights"""
        # Default insights structure
        insights = {
            "recommendations": [],
            "preprocessing_steps": [],
            "augmentation_suggestions": [],
            "train_val_split": {"train": 0.8, "val": 0.2},
            "estimated_training_samples": stats.total_samples,
            "confidence": 0.8
        }
        
        try:
            # Simple keyword extraction from AI response
            lines = response_content.lower().split('\n')
            
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect sections
                if 'recommend' in line or 'suggestion' in line:
                    current_section = 'recommendations'
                elif 'preprocess' in line or 'clean' in line:
                    current_section = 'preprocessing_steps'
                elif 'augment' in line or 'generate' in line:
                    current_section = 'augmentation_suggestions'
                elif line.startswith(('-', '•', '*', '1.', '2.')):
                    # Extract bullet points or numbered items
                    item = line.lstrip('-•*0123456789. ')
                    if current_section and item:
                        insights[current_section].append(item.capitalize())
            
            # Set training samples based on quality
            if quality.completeness_score > 0.9:
                insights["estimated_training_samples"] = int(stats.total_samples * 0.9)
            else:
                insights["estimated_training_samples"] = int(stats.total_samples * quality.completeness_score)
            
            # Adjust split based on dataset size
            if stats.total_samples < 500:
                insights["train_val_split"] = {"train": 0.9, "val": 0.1}
            elif stats.total_samples > 5000:
                insights["train_val_split"] = {"train": 0.8, "val": 0.15, "test": 0.05}
            
            return insights
            
        except Exception as e:
            logger.warning(f"Failed to parse AI insights: {e}")
            return insights
    
    def _fallback_insights(self, stats: DatasetStats, data: pd.DataFrame, 
                          task_type: TaskType, quality: QualityMetrics) -> Dict[str, Any]:
        """Fallback insights when LLM is not available"""
        insights = {
            "recommendations": [],
            "preprocessing_steps": [],
            "augmentation_suggestions": [],
            "train_val_split": {"train": 0.8, "val": 0.2},
            "estimated_training_samples": stats.total_samples,
            "confidence": 0.6
        }
        
        # Basic recommendations based on quality metrics
        if quality.completeness_score < 0.9:
            insights["recommendations"].append("Remove or impute missing values before training")
            insights["preprocessing_steps"].append("Handle missing data")
        
        if quality.balance_score < 0.7 and task_type == TaskType.CLASSIFICATION:
            insights["recommendations"].append("Consider data augmentation to balance classes")
            insights["augmentation_suggestions"].append("Class balancing techniques")
        
        if stats.total_samples < 500:
            insights["recommendations"].append("Small dataset - consider data augmentation")
            insights["augmentation_suggestions"].append("Text augmentation techniques")
            insights["train_val_split"] = {"train": 0.9, "val": 0.1}
        
        if task_type == TaskType.CHAT:
            insights["preprocessing_steps"].extend([
                "Ensure consistent conversation format",
                "Remove overly long conversations",
                "Check for proper turn-taking"
            ])
        elif task_type == TaskType.CLASSIFICATION:
            insights["preprocessing_steps"].extend([
                "Text normalization",
                "Remove duplicates",
                "Verify label consistency"
            ])
        
        return insights
    
    def suggest_data_augmentation(self, analysis: DatasetAnalysis) -> List[AugmentationStrategy]:
        """Suggest specific data augmentation strategies"""
        strategies = []
        
        # Basic augmentation strategies based on task type
        if analysis.task_type == TaskType.CHAT:
            strategies.extend([
                AugmentationStrategy(
                    technique="Paraphrasing",
                    description="Rephrase user inputs and responses while maintaining meaning",
                    implementation="Use paraphrasing models or manual rewrites",
                    expected_gain="20-30% more training examples",
                    difficulty="medium"
                ),
                AugmentationStrategy(
                    technique="Response variation",
                    description="Generate alternative responses to the same input",
                    implementation="Use LLM to generate alternative responses",
                    expected_gain="40-50% more training examples",
                    difficulty="easy"
                )
            ])
        
        elif analysis.task_type == TaskType.CLASSIFICATION:
            strategies.extend([
                AugmentationStrategy(
                    technique="Synonym replacement",
                    description="Replace words with synonyms to create variations",
                    implementation="Use WordNet or similar for synonym replacement",
                    expected_gain="50-100% more training examples",
                    difficulty="easy"
                ),
                AugmentationStrategy(
                    technique="Back translation",
                    description="Translate to another language and back",
                    implementation="Use translation APIs",
                    expected_gain="30-50% more training examples",
                    difficulty="medium"
                )
            ])
        
        # Add strategies based on quality issues
        if analysis.quality.balance_score < 0.7:
            strategies.append(
                AugmentationStrategy(
                    technique="Minority class oversampling",
                    description="Generate more examples for underrepresented classes",
                    implementation="SMOTE or manual generation",
                    expected_gain="Balanced dataset",
                    difficulty="medium"
                )
            )
        
        return strategies
    
    def export_analysis(self, analysis: DatasetAnalysis, output_path: str):
        """Export analysis results to JSON file"""
        try:
            analysis_dict = asdict(analysis)
            # Convert enums to strings for JSON serialization
            analysis_dict['stats']['format'] = analysis_dict['stats']['format']
            analysis_dict['task_type'] = analysis_dict['task_type']
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_dict, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]📊 Analysis exported to: {output_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to export analysis: {e}[/red]")
            raise

# Example usage and testing
if __name__ == "__main__":
    agent = DatasetIntelligenceAgent()
    
    # Test with sample data
    test_dataset = "examples/datasets/chat_training.csv"
    if os.path.exists(test_dataset):
        analysis = agent.analyze_dataset(test_dataset, "chat")
        print(f"Analysis complete: {analysis.stats.total_samples} samples")
        print(f"Quality score: {analysis.quality.completeness_score:.2f}")
        print(f"Recommendations: {len(analysis.recommendations)}")