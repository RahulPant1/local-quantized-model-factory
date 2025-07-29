import os
import json
import sqlite3
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from contextlib import contextmanager
from difflib import SequenceMatcher

from .planner_agent import QuantizationPlan
from .executor_agent import ExecutionResult

class StorageBackend(Enum):
    JSON = "json"
    SQLITE = "sqlite"

@dataclass
class ModelRecord:
    id: str
    model_name: str
    original_size_mb: Optional[float]
    quantized_size_mb: Optional[float]
    quantization_type: str
    bit_width: int
    target_format: str
    cpu_fallback: bool
    gpu_memory_limit: Optional[int]
    output_path: Optional[str]
    success: bool
    execution_time: Optional[float]
    error_message: Optional[str]
    created_at: str
    updated_at: str
    metadata: Optional[Dict] = None

@dataclass 
class ExperimentLog:
    id: str
    model_record_id: str
    step: str
    message: str
    timestamp: str
    log_level: str = "INFO"
    additional_data: Optional[Dict] = None

@dataclass
class SimilarModel:
    """Represents a similar model found in memory"""
    record: ModelRecord
    similarity_score: float
    similarity_reasons: List[str]
    success_rate: float

@dataclass
class QuantizationSuggestion:
    """Suggested quantization strategy based on historical data"""
    recommended_bit_width: int
    recommended_method: str
    recommended_format: str
    cpu_fallback_suggested: bool
    confidence_score: float
    reasoning: List[str]
    similar_models: List[SimilarModel]

@dataclass
class RecoveryAction:
    """Suggested recovery action for failed quantization"""
    action_type: str
    description: str
    parameters: Dict[str, Any]
    success_probability: float
    reasoning: str

@dataclass
class ModelSuggestion:
    """Model recommendation with reasoning"""
    model_name: str
    confidence_score: float
    success_rate: float
    avg_size_mb: Optional[float]
    avg_execution_time: Optional[float]
    reasons: List[str]
    compatible_methods: List[str]
    download_count: Optional[int] = None

@dataclass
class ExperimentPlan:
    """Suggested experiment plan"""
    model_name: str
    quantization_method: str
    bit_width: int
    priority_score: float
    learning_value: str
    expected_insights: List[str]
    resource_requirements: Dict[str, Any]

@dataclass
class ComparisonReport:
    """Detailed comparison between experiments"""
    experiment_ids: List[str]
    models_compared: List[str]
    performance_metrics: Dict[str, Dict[str, Any]]
    success_analysis: Dict[str, Any]
    optimization_suggestions: List[str]
    best_performer: Optional[str]

@dataclass
class LearningPattern:
    """Cross-session learning pattern"""
    pattern_type: str
    description: str
    confidence: float
    supporting_experiments: List[str]
    recommendations: List[str]
    trend_direction: str  # "improving", "declining", "stable"

@dataclass
class PerformanceBenchmark:
    """Performance benchmarking data"""
    model_record_id: str
    inference_speed_tokens_per_sec: Optional[float]
    memory_usage_mb: Optional[float]
    quality_score: Optional[float]
    benchmark_timestamp: str
    test_prompt: Optional[str]
    hardware_info: Dict[str, Any]

@dataclass
class HuggingFaceModelRecommendation:
    """Recommendation for downloading and quantizing HF models"""
    model_name: str
    model_id: str
    description: str
    download_count: int
    model_size_gb: Optional[float]
    architecture: str
    tags: List[str]
    category: str  # e.g., "chat", "code", "text-generation"
    
    # Recommendation scores
    popularity_score: float
    compatibility_score: float
    learning_value_score: float
    resource_fit_score: float
    overall_score: float
    
    # Reasoning
    reasoning: List[str]  # Changed from recommendation_reasons for consistency
    quantization_predictions: Dict[str, Dict[str, Any]]  # method -> {success_probability, reasoning}
    resource_requirements: Dict[str, Any]  # Changed from estimated_resources
    learning_benefits: List[str]
    
    # Metadata
    last_updated: str
    license: Optional[str]
    is_gated: bool = False

class MemoryAgent:
    def __init__(self, storage_dir: str = "mcp", backend: StorageBackend = StorageBackend.SQLITE):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.backend = backend
        
        if backend == StorageBackend.SQLITE:
            self.db_path = self.storage_dir / "memory.db"
            self._init_sqlite_schema()
        else:
            self.json_path = self.storage_dir / "memory.json"
            self._init_json_storage()
    
    def _init_sqlite_schema(self):
        """Initialize SQLite database schema"""
        with self._get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_records (
                    id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    original_size_mb REAL,
                    quantized_size_mb REAL,
                    quantization_type TEXT NOT NULL,
                    bit_width INTEGER NOT NULL,
                    target_format TEXT NOT NULL,
                    cpu_fallback BOOLEAN NOT NULL,
                    gpu_memory_limit INTEGER,
                    output_path TEXT,
                    success BOOLEAN NOT NULL,
                    execution_time REAL,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_logs (
                    id TEXT PRIMARY KEY,
                    model_record_id TEXT NOT NULL,
                    step TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    log_level TEXT DEFAULT 'INFO',
                    additional_data TEXT,
                    FOREIGN KEY (model_record_id) REFERENCES model_records (id)
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON model_records (model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON model_records (created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_model_id ON experiment_logs (model_record_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON experiment_logs (timestamp)")
    
    def _init_json_storage(self):
        """Initialize JSON storage"""
        if not self.json_path.exists():
            initial_data = {
                "model_records": {},
                "experiment_logs": {},
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            with open(self.json_path, 'w') as f:
                json.dump(initial_data, f, indent=2)
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _load_json_data(self) -> Dict:
        """Load data from JSON file"""
        with open(self.json_path, 'r') as f:
            return json.load(f)
    
    def _save_json_data(self, data: Dict):
        """Save data to JSON file"""
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def store_quantization_attempt(self, plan: QuantizationPlan, result: ExecutionResult) -> str:
        """Store a quantization attempt with plan and result"""
        record_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        record = ModelRecord(
            id=record_id,
            model_name=plan.model_name,
            original_size_mb=None,  # Can be populated later
            quantized_size_mb=result.model_size_mb,
            quantization_type=plan.quantization_type.value,
            bit_width=plan.bit_width,
            target_format=plan.target_format.value,
            cpu_fallback=plan.cpu_fallback,
            gpu_memory_limit=plan.gpu_memory_limit,
            output_path=result.output_path,
            success=result.success,
            execution_time=result.execution_time,
            error_message=result.error_message,
            created_at=now,
            updated_at=now,
            metadata=plan.specific_params
        )
        
        if self.backend == StorageBackend.SQLITE:
            self._store_record_sqlite(record)
        else:
            self._store_record_json(record)
        
        return record_id
    
    def _store_record_sqlite(self, record: ModelRecord):
        """Store record in SQLite"""
        with self._get_db_connection() as conn:
            conn.execute("""
                INSERT INTO model_records (
                    id, model_name, original_size_mb, quantized_size_mb,
                    quantization_type, bit_width, target_format, cpu_fallback,
                    gpu_memory_limit, output_path, success, execution_time,
                    error_message, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id, record.model_name, record.original_size_mb, record.quantized_size_mb,
                record.quantization_type, record.bit_width, record.target_format, record.cpu_fallback,
                record.gpu_memory_limit, record.output_path, record.success, record.execution_time,
                record.error_message, record.created_at, record.updated_at,
                json.dumps(record.metadata) if record.metadata else None
            ))
    
    def _store_record_json(self, record: ModelRecord):
        """Store record in JSON"""
        data = self._load_json_data()
        data["model_records"][record.id] = asdict(record)
        self._save_json_data(data)
    
    def log_experiment_step(self, model_record_id: str, step: str, message: str, 
                           log_level: str = "INFO", additional_data: Optional[Dict] = None):
        """Log an experiment step"""
        log_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        log_entry = ExperimentLog(
            id=log_id,
            model_record_id=model_record_id,
            step=step,
            message=message,
            timestamp=now,
            log_level=log_level,
            additional_data=additional_data
        )
        
        if self.backend == StorageBackend.SQLITE:
            self._store_log_sqlite(log_entry)
        else:
            self._store_log_json(log_entry)
    
    def log_experiment(self, model_name: str, quantization_method: str, bit_width: int, 
                      status: str, metadata: Optional[Dict] = None):
        """Log an experiment/operation (API compatibility method)"""
        # Create a record for this experiment
        record = ModelRecord(
            id=str(uuid.uuid4()),
            model_name=model_name,
            original_size_mb=None,
            quantized_size_mb=metadata.get('memory_usage') if metadata else None,
            quantization_type=quantization_method,
            bit_width=bit_width,
            target_format="unknown",
            cpu_fallback=False,
            gpu_memory_limit=None,
            output_path=None,
            success=(status != "error"),
            execution_time=metadata.get('load_time') if metadata else None,
            error_message=None,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata=metadata
        )
        
        if self.backend == StorageBackend.SQLITE:
            self._store_record_sqlite(record)
        else:
            self._store_record_json(record)
        
        # Also log as an experiment step
        self.log_experiment_step(
            model_record_id=record.id,
            step=status,
            message=f"Logged experiment for {model_name} with {quantization_method} {bit_width}-bit",
            additional_data=metadata
        )
        
        return record.id
    
    def get_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments as a list of dictionaries"""
        records = self.get_model_records()
        return [asdict(record) for record in records]
    
    def _store_log_sqlite(self, log_entry: ExperimentLog):
        """Store log entry in SQLite"""
        with self._get_db_connection() as conn:
            conn.execute("""
                INSERT INTO experiment_logs (
                    id, model_record_id, step, message, timestamp, log_level, additional_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                log_entry.id, log_entry.model_record_id, log_entry.step, log_entry.message,
                log_entry.timestamp, log_entry.log_level,
                json.dumps(log_entry.additional_data) if log_entry.additional_data else None
            ))
    
    def _store_log_json(self, log_entry: ExperimentLog):
        """Store log entry in JSON"""
        data = self._load_json_data()
        if "experiment_logs" not in data:
            data["experiment_logs"] = {}
        data["experiment_logs"][log_entry.id] = asdict(log_entry)
        self._save_json_data(data)
    
    def get_model_records(self, model_name: Optional[str] = None, 
                         success_only: bool = False) -> List[ModelRecord]:
        """Get model records with optional filtering"""
        if self.backend == StorageBackend.SQLITE:
            return self._get_records_sqlite(model_name, success_only)
        else:
            return self._get_records_json(model_name, success_only)
    
    def _get_records_sqlite(self, model_name: Optional[str], success_only: bool) -> List[ModelRecord]:
        """Get records from SQLite"""
        query = "SELECT * FROM model_records WHERE 1=1"
        params = []
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        if success_only:
            query += " AND success = 1"
        
        query += " ORDER BY created_at DESC"
        
        with self._get_db_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                metadata = json.loads(row['metadata']) if row['metadata'] else None
                record = ModelRecord(
                    id=row['id'],
                    model_name=row['model_name'],
                    original_size_mb=row['original_size_mb'],
                    quantized_size_mb=row['quantized_size_mb'],
                    quantization_type=row['quantization_type'],
                    bit_width=row['bit_width'],
                    target_format=row['target_format'],
                    cpu_fallback=bool(row['cpu_fallback']),
                    gpu_memory_limit=row['gpu_memory_limit'],
                    output_path=row['output_path'],
                    success=bool(row['success']),
                    execution_time=row['execution_time'],
                    error_message=row['error_message'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=metadata
                )
                records.append(record)
            
            return records
    
    def _get_records_json(self, model_name: Optional[str], success_only: bool) -> List[ModelRecord]:
        """Get records from JSON"""
        data = self._load_json_data()
        records = []
        
        for record_data in data.get("model_records", {}).values():
            if model_name and record_data["model_name"] != model_name:
                continue
            if success_only and not record_data["success"]:
                continue
            
            record = ModelRecord(**record_data)
            records.append(record)
        
        # Sort by created_at descending
        records.sort(key=lambda x: x.created_at, reverse=True)
        return records
    
    def get_experiment_logs(self, model_record_id: str) -> List[ExperimentLog]:
        """Get experiment logs for a specific model record"""
        if self.backend == StorageBackend.SQLITE:
            return self._get_logs_sqlite(model_record_id)
        else:
            return self._get_logs_json(model_record_id)
    
    def _get_logs_sqlite(self, model_record_id: str) -> List[ExperimentLog]:
        """Get logs from SQLite"""
        with self._get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM experiment_logs 
                WHERE model_record_id = ? 
                ORDER BY timestamp ASC
            """, (model_record_id,))
            rows = cursor.fetchall()
            
            logs = []
            for row in rows:
                additional_data = json.loads(row['additional_data']) if row['additional_data'] else None
                log = ExperimentLog(
                    id=row['id'],
                    model_record_id=row['model_record_id'],
                    step=row['step'],
                    message=row['message'],
                    timestamp=row['timestamp'],
                    log_level=row['log_level'],
                    additional_data=additional_data
                )
                logs.append(log)
            
            return logs
    
    def _get_logs_json(self, model_record_id: str) -> List[ExperimentLog]:
        """Get logs from JSON"""
        data = self._load_json_data()
        logs = []
        
        for log_data in data.get("experiment_logs", {}).values():
            if log_data["model_record_id"] == model_record_id:
                log = ExperimentLog(**log_data)
                logs.append(log)
        
        # Sort by timestamp ascending
        logs.sort(key=lambda x: x.timestamp)
        return logs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored models and experiments"""
        if self.backend == StorageBackend.SQLITE:
            return self._get_stats_sqlite()
        else:
            return self._get_stats_json()
    
    def _get_stats_sqlite(self) -> Dict[str, Any]:
        """Get statistics from SQLite"""
        with self._get_db_connection() as conn:
            stats = {}
            
            # Total records
            cursor = conn.execute("SELECT COUNT(*) as count FROM model_records")
            stats["total_records"] = cursor.fetchone()["count"]
            
            # Successful quantizations
            cursor = conn.execute("SELECT COUNT(*) as count FROM model_records WHERE success = 1")
            stats["successful_quantizations"] = cursor.fetchone()["count"]
            
            # Failed quantizations
            cursor = conn.execute("SELECT COUNT(*) as count FROM model_records WHERE success = 0")
            stats["failed_quantizations"] = cursor.fetchone()["count"]
            
            # Quantization types distribution
            cursor = conn.execute("""
                SELECT quantization_type, COUNT(*) as count 
                FROM model_records 
                GROUP BY quantization_type
            """)
            stats["quantization_types"] = {row["quantization_type"]: row["count"] for row in cursor.fetchall()}
            
            # Average execution time for successful quantizations
            cursor = conn.execute("""
                SELECT AVG(execution_time) as avg_time 
                FROM model_records 
                WHERE success = 1 AND execution_time IS NOT NULL
            """)
            result = cursor.fetchone()
            stats["avg_execution_time"] = result["avg_time"] if result["avg_time"] else 0
            
            # Total storage saved (if original sizes are available)
            cursor = conn.execute("""
                SELECT SUM(original_size_mb - quantized_size_mb) as saved 
                FROM model_records 
                WHERE success = 1 AND original_size_mb IS NOT NULL AND quantized_size_mb IS NOT NULL
            """)
            result = cursor.fetchone()
            stats["total_storage_saved_mb"] = result["saved"] if result["saved"] else 0
            
            return stats
    
    def _get_stats_json(self) -> Dict[str, Any]:
        """Get statistics from JSON"""
        data = self._load_json_data()
        records = list(data.get("model_records", {}).values())
        
        stats = {
            "total_records": len(records),
            "successful_quantizations": len([r for r in records if r["success"]]),
            "failed_quantizations": len([r for r in records if not r["success"]]),
            "quantization_types": {},
            "avg_execution_time": 0,
            "total_storage_saved_mb": 0
        }
        
        # Quantization types distribution
        for record in records:
            qt = record["quantization_type"]
            stats["quantization_types"][qt] = stats["quantization_types"].get(qt, 0) + 1
        
        # Average execution time
        successful_times = [r["execution_time"] for r in records if r["success"] and r["execution_time"]]
        if successful_times:
            stats["avg_execution_time"] = sum(successful_times) / len(successful_times)
        
        # Total storage saved
        savings = []
        for record in records:
            if (record["success"] and record["original_size_mb"] and record["quantized_size_mb"]):
                savings.append(record["original_size_mb"] - record["quantized_size_mb"])
        if savings:
            stats["total_storage_saved_mb"] = sum(savings)
        
        return stats
    
    def export_data(self, output_path: str, format: str = "json"):
        """Export all data to a file"""
        if self.backend == StorageBackend.SQLITE:
            records = self._get_records_sqlite(None, False)
            export_data = {
                "model_records": [asdict(r) for r in records],
                "experiment_logs": {},
                "export_timestamp": datetime.now().isoformat()
            }
            
            # Get all logs
            for record in records:
                logs = self._get_logs_sqlite(record.id)
                export_data["experiment_logs"][record.id] = [asdict(log) for log in logs]
        else:
            export_data = self._load_json_data()
            export_data["export_timestamp"] = datetime.now().isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    # === ENHANCED MCP FEATURES - PHASE 2 ===
    
    def store_performance_benchmark(self, model_record_id: str, benchmark_data: 'PerformanceBenchmark'):
        """Store performance benchmark data"""
        if self.backend == StorageBackend.SQLITE:
            self._store_benchmark_sqlite(benchmark_data)
        else:
            self._store_benchmark_json(benchmark_data)
    
    def _store_benchmark_sqlite(self, benchmark: 'PerformanceBenchmark'):
        """Store benchmark in SQLite"""
        with self._get_db_connection() as conn:
            # Create benchmark table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_benchmarks (
                    id TEXT PRIMARY KEY,
                    model_record_id TEXT NOT NULL,
                    inference_speed_tokens_per_sec REAL,
                    memory_usage_mb REAL,
                    quality_score REAL,
                    benchmark_timestamp TEXT NOT NULL,
                    test_prompt TEXT,
                    hardware_info TEXT,
                    FOREIGN KEY (model_record_id) REFERENCES model_records (id)
                )
            """)
            
            benchmark_id = str(uuid.uuid4())
            conn.execute("""
                INSERT INTO performance_benchmarks (
                    id, model_record_id, inference_speed_tokens_per_sec, memory_usage_mb,
                    quality_score, benchmark_timestamp, test_prompt, hardware_info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                benchmark_id, benchmark.model_record_id, benchmark.inference_speed_tokens_per_sec,
                benchmark.memory_usage_mb, benchmark.quality_score, benchmark.benchmark_timestamp,
                benchmark.test_prompt, json.dumps(benchmark.hardware_info)
            ))
    
    def _store_benchmark_json(self, benchmark: 'PerformanceBenchmark'):
        """Store benchmark in JSON"""
        data = self._load_json_data()
        if "performance_benchmarks" not in data:
            data["performance_benchmarks"] = {}
        
        benchmark_id = str(uuid.uuid4())
        data["performance_benchmarks"][benchmark_id] = asdict(benchmark)
        self._save_json_data(data)
    
    def get_model_compatibility_matrix(self) -> Dict[str, Dict[str, float]]:
        """Generate model compatibility matrix showing success rates by family and method"""
        records = self.get_model_records()
        compatibility_matrix = {}
        
        for record in records:
            family = self._extract_model_family(record.model_name)
            method = record.quantization_type
            
            if family not in compatibility_matrix:
                compatibility_matrix[family] = {}
            
            if method not in compatibility_matrix[family]:
                compatibility_matrix[family][method] = {'total': 0, 'successful': 0}
            
            compatibility_matrix[family][method]['total'] += 1
            if record.success:
                compatibility_matrix[family][method]['successful'] += 1
        
        # Convert to success rates
        success_matrix = {}
        for family, methods in compatibility_matrix.items():
            success_matrix[family] = {}
            for method, stats in methods.items():
                if stats['total'] > 0:
                    success_matrix[family][method] = stats['successful'] / stats['total']
                else:
                    success_matrix[family][method] = 0.0
        
        return success_matrix
    
    # === ENHANCED MCP FEATURES - PHASE 1 ===
    
    def get_similar_models(self, model_name: str, limit: int = 5) -> List[SimilarModel]:
        """Find similar models based on name patterns, architecture, and size"""
        records = self.get_model_records()
        similar_models = []
        
        for record in records:
            similarity_score, reasons = self._calculate_model_similarity(model_name, record)
            if similarity_score > 0.3:  # Threshold for similarity
                success_rate = self._calculate_success_rate_for_model_type(record.model_name)
                similar_models.append(SimilarModel(
                    record=record,
                    similarity_score=similarity_score,
                    similarity_reasons=reasons,
                    success_rate=success_rate
                ))
        
        # Sort by similarity score descending
        similar_models.sort(key=lambda x: x.similarity_score, reverse=True)
        return similar_models[:limit]
    
    def _calculate_model_similarity(self, target_model: str, record: ModelRecord) -> Tuple[float, List[str]]:
        """Calculate similarity between target model and existing record"""
        reasons = []
        similarity_score = 0.0
        
        # Name similarity
        name_similarity = SequenceMatcher(None, target_model.lower(), record.model_name.lower()).ratio()
        similarity_score += name_similarity * 0.4
        if name_similarity > 0.5:
            reasons.append(f"Similar name pattern ({name_similarity:.2f} match)")
        
        # Extract model family/organization
        target_parts = target_model.split('/')
        record_parts = record.model_name.split('/')
        
        if len(target_parts) > 1 and len(record_parts) > 1:
            # Same organization/family
            if target_parts[0].lower() == record_parts[0].lower():
                similarity_score += 0.3
                reasons.append(f"Same organization ({target_parts[0]})")
            
            # Similar model names (e.g., "llama", "mistral")
            for family in ['llama', 'mistral', 'gpt', 'opt', 'bloom', 'falcon']:
                if family in target_parts[1].lower() and family in record_parts[1].lower():
                    similarity_score += 0.2
                    reasons.append(f"Same model family ({family})")
        
        # Size patterns (extract numbers that might indicate model size)
        target_size_match = re.search(r'(\d+(?:\.\d+)?)[bB]?', target_model)
        record_size_match = re.search(r'(\d+(?:\.\d+)?)[bB]?', record.model_name)
        
        if target_size_match and record_size_match:
            target_size = float(target_size_match.group(1))
            record_size = float(record_size_match.group(1))
            
            # Similar size models
            size_ratio = min(target_size, record_size) / max(target_size, record_size)
            if size_ratio > 0.7:  # Within 30% of each other
                similarity_score += size_ratio * 0.2
                reasons.append(f"Similar model size ({target_size}B vs {record_size}B)")
        
        return min(similarity_score, 1.0), reasons
    
    def _calculate_success_rate_for_model_type(self, model_name: str) -> float:
        """Calculate success rate for similar model types"""
        # Extract model family for success rate calculation
        model_family = self._extract_model_family(model_name)
        
        if self.backend == StorageBackend.SQLITE:
            with self._get_db_connection() as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) as total, SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
                    FROM model_records 
                    WHERE model_name LIKE ?
                """, (f"%{model_family}%",))
                result = cursor.fetchone()
                total = result["total"]
                successful = result["successful"]
        else:
            data = self._load_json_data()
            records = [r for r in data.get("model_records", {}).values() if model_family.lower() in r["model_name"].lower()]
            total = len(records)
            successful = len([r for r in records if r["success"]])
        
        return successful / total if total > 0 else 0.0
    
    def _extract_model_family(self, model_name: str) -> str:
        """Extract model family from model name"""
        families = ['llama', 'mistral', 'gpt', 'opt', 'bloom', 'falcon', 'chatglm', 'baichuan']
        model_lower = model_name.lower()
        
        for family in families:
            if family in model_lower:
                return family
        
        # Fallback to organization name
        if '/' in model_name:
            return model_name.split('/')[0]
        
        return model_name.split('-')[0] if '-' in model_name else model_name
    
    def suggest_quantization_strategy(self, model_name: str) -> QuantizationSuggestion:
        """Suggest optimal quantization strategy based on historical data"""
        similar_models = self.get_similar_models(model_name, limit=10)
        
        if not similar_models:
            # Default suggestion for new model types
            return QuantizationSuggestion(
                recommended_bit_width=4,
                recommended_method="bnb",
                recommended_format="safetensors",
                cpu_fallback_suggested=True,
                confidence_score=0.3,
                reasoning=["No similar models found, using conservative defaults"],
                similar_models=[]
            )
        
        # Analyze successful quantizations from similar models
        successful_similar = [sm for sm in similar_models if sm.record.success]
        
        if not successful_similar:
            return QuantizationSuggestion(
                recommended_bit_width=8,
                recommended_method="bnb",
                recommended_format="safetensors",
                cpu_fallback_suggested=True,
                confidence_score=0.4,
                reasoning=["Similar models found but no successful quantizations, using safer defaults"],
                similar_models=similar_models[:3]
            )
        
        # Calculate recommendations based on successful attempts
        bit_widths = [sm.record.bit_width for sm in successful_similar]
        methods = [sm.record.quantization_type for sm in successful_similar]
        formats = [sm.record.target_format for sm in successful_similar]
        cpu_fallbacks = [sm.record.cpu_fallback for sm in successful_similar]
        
        # Most common successful configurations
        recommended_bit_width = max(set(bit_widths), key=bit_widths.count)
        recommended_method = max(set(methods), key=methods.count)
        recommended_format = max(set(formats), key=formats.count)
        cpu_fallback_suggested = sum(cpu_fallbacks) > len(cpu_fallbacks) / 2
        
        # Calculate confidence based on success rate and similarity
        avg_similarity = sum(sm.similarity_score for sm in successful_similar) / len(successful_similar)
        avg_success_rate = sum(sm.success_rate for sm in successful_similar) / len(successful_similar)
        confidence_score = (avg_similarity * 0.6 + avg_success_rate * 0.4)
        
        reasoning = [
            f"Based on {len(successful_similar)} successful similar models",
            f"{recommended_bit_width}-bit quantization succeeded {bit_widths.count(recommended_bit_width)}/{len(bit_widths)} times",
            f"{recommended_method.upper()} method succeeded {methods.count(recommended_method)}/{len(methods)} times",
            f"Average success rate for similar models: {avg_success_rate:.1%}"
        ]
        
        return QuantizationSuggestion(
            recommended_bit_width=recommended_bit_width,
            recommended_method=recommended_method,
            recommended_format=recommended_format,
            cpu_fallback_suggested=cpu_fallback_suggested,
            confidence_score=confidence_score,
            reasoning=reasoning,
            similar_models=similar_models[:3]
        )
    
    def suggest_error_recovery(self, error_message: str, failed_plan: 'QuantizationPlan') -> List[RecoveryAction]:
        """Suggest recovery actions based on historical error patterns"""
        recovery_actions = []
        
        # Analyze similar failures
        similar_failures = self._find_similar_failures(error_message, failed_plan)
        
        # Memory-related errors
        if any(keyword in error_message.lower() for keyword in ['memory', 'oom', 'cuda']):
            if not failed_plan.cpu_fallback:
                recovery_actions.append(RecoveryAction(
                    action_type="enable_cpu_fallback",
                    description="Enable CPU fallback to avoid GPU memory issues",
                    parameters={"cpu_fallback": True},
                    success_probability=self._calculate_recovery_success_rate("cpu_fallback", similar_failures),
                    reasoning="GPU memory issues often resolve with CPU fallback"
                ))
            
            if failed_plan.bit_width <= 4:
                recovery_actions.append(RecoveryAction(
                    action_type="increase_bit_width",
                    description="Increase bit width to reduce memory pressure",
                    parameters={"bit_width": min(failed_plan.bit_width + 2, 8)},
                    success_probability=self._calculate_recovery_success_rate("higher_bit_width", similar_failures),
                    reasoning="Higher bit width requires less memory during quantization"
                ))
        
        # Model not found / download errors
        if any(keyword in error_message.lower() for keyword in ['404', 'not found', 'download']):
            recovery_actions.append(RecoveryAction(
                action_type="verify_model_name",
                description="Verify model name and check HuggingFace Hub",
                parameters={"suggested_names": self._suggest_similar_model_names(failed_plan.model_name)},
                success_probability=0.8,
                reasoning="Model name might be incorrect or model might be private"
            ))
        
        # Compatibility errors
        if any(keyword in error_message.lower() for keyword in ['compatibility', 'unsupported', 'format']):
            # Suggest different quantization method
            current_method = failed_plan.quantization_type.value
            alternative_methods = ["bnb", "gptq", "gguf"]
            if current_method in alternative_methods:
                alternative_methods.remove(current_method)
            
            for method in alternative_methods:
                recovery_actions.append(RecoveryAction(
                    action_type="change_method",
                    description=f"Try {method.upper()} quantization method",
                    parameters={"quantization_type": method},
                    success_probability=self._calculate_method_success_rate(method, failed_plan.model_name),
                    reasoning=f"{method.upper()} might be more compatible with this model architecture"
                ))
        
        # Sort by success probability
        recovery_actions.sort(key=lambda x: x.success_probability, reverse=True)
        return recovery_actions[:3]  # Return top 3 suggestions
    
    def _find_similar_failures(self, error_message: str, failed_plan: 'QuantizationPlan') -> List[ModelRecord]:
        """Find records with similar failure patterns"""
        records = self.get_model_records(success_only=False)
        similar_failures = []
        
        for record in records:
            if not record.success and record.error_message:
                # Check for similar error patterns
                error_similarity = SequenceMatcher(None, error_message.lower(), record.error_message.lower()).ratio()
                if error_similarity > 0.3:
                    similar_failures.append(record)
        
        return similar_failures
    
    def _calculate_recovery_success_rate(self, recovery_type: str, similar_failures: List[ModelRecord]) -> float:
        """Calculate success rate for specific recovery actions"""
        # This is a simplified implementation
        # In practice, you'd track recovery attempts and their outcomes
        recovery_rates = {
            "cpu_fallback": 0.7,
            "higher_bit_width": 0.6,
            "change_method": 0.5
        }
        return recovery_rates.get(recovery_type, 0.4)
    
    def _calculate_method_success_rate(self, method: str, model_name: str) -> float:
        """Calculate success rate for a specific method with similar models"""
        model_family = self._extract_model_family(model_name)
        
        if self.backend == StorageBackend.SQLITE:
            with self._get_db_connection() as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) as total, SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
                    FROM model_records 
                    WHERE quantization_type = ? AND model_name LIKE ?
                """, (method, f"%{model_family}%"))
                result = cursor.fetchone()
                total = result["total"]
                successful = result["successful"]
        else:
            data = self._load_json_data()
            records = [
                r for r in data.get("model_records", {}).values() 
                if r["quantization_type"] == method and model_family.lower() in r["model_name"].lower()
            ]
            total = len(records)
            successful = len([r for r in records if r["success"]])
        
        return successful / total if total > 0 else 0.5  # Default 50% if no data
    
    def _suggest_similar_model_names(self, model_name: str) -> List[str]:
        """Suggest similar model names based on existing records"""
        records = self.get_model_records()
        suggestions = []
        
        for record in records:
            similarity = SequenceMatcher(None, model_name.lower(), record.model_name.lower()).ratio()
            if similarity > 0.4:
                suggestions.append(record.model_name)
        
        return list(set(suggestions))[:5]  # Return unique suggestions, max 5
    
    def get_contextual_insights(self, model_name: str) -> Dict[str, Any]:
        """Get contextual insights for a model based on historical data"""
        similar_models = self.get_similar_models(model_name)
        suggestion = self.suggest_quantization_strategy(model_name)
        
        insights = {
            "similar_models_count": len(similar_models),
            "confidence_score": suggestion.confidence_score,
            "recommended_strategy": {
                "bit_width": suggestion.recommended_bit_width,
                "method": suggestion.recommended_method,
                "format": suggestion.recommended_format,
                "cpu_fallback": suggestion.cpu_fallback_suggested
            },
            "success_indicators": suggestion.reasoning,
            "risk_factors": self._identify_risk_factors(model_name, similar_models)
        }
        
        return insights
    
    def _identify_risk_factors(self, model_name: str, similar_models: List[SimilarModel]) -> List[str]:
        """Identify potential risk factors for quantization"""
        risk_factors = []
        
        if not similar_models:
            risk_factors.append("No historical data for similar models")
        
        failed_similar = [sm for sm in similar_models if not sm.record.success]
        if len(failed_similar) > len(similar_models) / 2:
            risk_factors.append("High failure rate for similar models")
        
        # Check for large model indicators
        size_match = re.search(r'(\d+(?:\.\d+)?)[bB]?', model_name)
        if size_match and float(size_match.group(1)) > 10:
            risk_factors.append("Large model size may require more memory")
        
        # Check for specific model families with known issues
        problematic_families = ['chatglm', 'baichuan']  # Example problematic families
        for family in problematic_families:
            if family in model_name.lower():
                risk_factors.append(f"{family.title()} models may have compatibility issues")
        
        return risk_factors
    
    # === ENHANCED MCP FEATURES - PHASE 2 CONTINUED ===
    
    def get_model_recommendations(self, user_goals: str, limit: int = 5) -> List['ModelSuggestion']:
        """Recommend models based on user goals and historical success patterns"""
        recommendations = []
        records = self.get_model_records(success_only=True)
        
        if not records:
            return self._get_default_model_recommendations(user_goals)
        
        # Analyze successful models by family/type
        model_families = {}
        for record in records:
            family = self._extract_model_family(record.model_name)
            if family not in model_families:
                model_families[family] = {
                    'models': [],
                    'success_count': 0,
                    'total_count': 0,
                    'avg_size': 0,
                    'avg_time': 0,
                    'methods': set()
                }
            
            model_families[family]['models'].append(record)
            model_families[family]['success_count'] += 1
            model_families[family]['methods'].add(record.quantization_type)
            if record.quantized_size_mb:
                model_families[family]['avg_size'] += record.quantized_size_mb
            if record.execution_time:
                model_families[family]['avg_time'] += record.execution_time
        
        # Calculate family statistics
        for family, data in model_families.items():
            model_count = len(data['models'])
            if model_count > 0:
                data['avg_size'] /= model_count
                data['avg_time'] /= model_count
                data['success_rate'] = data['success_count'] / model_count
        
        # Generate recommendations based on goals
        goal_keywords = user_goals.lower().split()
        
        for family, data in model_families.items():
            if data['success_count'] == 0:
                continue
                
            confidence = self._calculate_recommendation_confidence(data, goal_keywords)
            reasons = self._generate_recommendation_reasons(family, data, goal_keywords)
            
            # Find the best performing model in this family
            best_model = min(data['models'], key=lambda x: x.execution_time or float('inf'))
            
            recommendation = ModelSuggestion(
                model_name=f"{family} family (e.g., {best_model.model_name})",
                confidence_score=confidence,
                success_rate=data['success_rate'],
                avg_size_mb=data['avg_size'],
                avg_execution_time=data['avg_time'],
                reasons=reasons,
                compatible_methods=list(data['methods'])
            )
            
            recommendations.append(recommendation)
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        return recommendations[:limit]
    
    def _get_default_model_recommendations(self, user_goals: str) -> List['ModelSuggestion']:
        """Provide default recommendations when no historical data exists"""
        defaults = [
            ModelSuggestion(
                model_name="microsoft/DialoGPT-small",
                confidence_score=0.7,
                success_rate=0.8,
                avg_size_mb=200.0,
                avg_execution_time=30.0,
                reasons=["Small, well-tested model", "Good for beginners", "Fast quantization"],
                compatible_methods=["bnb", "gptq"]
            ),
            ModelSuggestion(
                model_name="meta-llama/Llama-2-7b-hf",
                confidence_score=0.6,
                success_rate=0.7,
                avg_size_mb=3000.0,
                avg_execution_time=120.0,
                reasons=["Popular architecture", "Good documentation", "Community support"],
                compatible_methods=["bnb", "gptq", "gguf"]
            )
        ]
        return defaults
    
    def _calculate_recommendation_confidence(self, family_data: Dict, goal_keywords: List[str]) -> float:
        """Calculate confidence score for model family recommendation"""
        base_confidence = family_data['success_rate']
        
        # Boost confidence based on goal alignment
        goal_boost = 0.0
        if any(keyword in ['fast', 'quick', 'speed'] for keyword in goal_keywords):
            if family_data['avg_time'] < 60:  # Fast quantization
                goal_boost += 0.2
        
        if any(keyword in ['small', 'lightweight', 'mobile'] for keyword in goal_keywords):
            if family_data['avg_size'] < 1000:  # Small models
                goal_boost += 0.2
        
        if any(keyword in ['reliable', 'stable', 'production'] for keyword in goal_keywords):
            if family_data['success_rate'] > 0.8:  # High success rate
                goal_boost += 0.2
        
        return min(base_confidence + goal_boost, 1.0)
    
    def _generate_recommendation_reasons(self, family: str, data: Dict, goal_keywords: List[str]) -> List[str]:
        """Generate human-readable reasons for recommendations"""
        reasons = []
        
        if data['success_rate'] > 0.8:
            reasons.append(f"High success rate ({data['success_rate']:.0%}) in your experiments")
        
        if data['avg_time'] < 60:
            reasons.append(f"Fast quantization (avg {data['avg_time']:.0f}s)")
        
        if len(data['methods']) > 1:
            reasons.append(f"Compatible with {len(data['methods'])} quantization methods")
        
        if data['avg_size'] < 500:
            reasons.append("Produces compact quantized models")
        
        # Goal-specific reasons
        if any(keyword in ['beginner', 'learning', 'test'] for keyword in goal_keywords):
            reasons.append("Good for learning and experimentation")
        
        return reasons
    
    def suggest_next_experiments(self, user_goals: str, limit: int = 3) -> List['ExperimentPlan']:
        """Suggest next experiments based on learning value and user goals"""
        experiments = []
        
        # Analyze gaps in current knowledge
        records = self.get_model_records()
        tested_combinations = set()
        
        for record in records:
            combination = f"{record.quantization_type}_{record.bit_width}_{self._extract_model_family(record.model_name)}"
            tested_combinations.add(combination)
        
        # Suggest untested combinations with high learning value
        model_families = ['llama', 'mistral', 'gpt', 'opt']
        methods = ['bnb', 'gptq', 'gguf']
        bit_widths = [4, 8]
        
        for family in model_families:
            for method in methods:
                for bit_width in bit_widths:
                    combination = f"{method}_{bit_width}_{family}"
                    
                    if combination not in tested_combinations:
                        priority = self._calculate_experiment_priority(family, method, bit_width, user_goals)
                        
                        if priority > 0.3:  # Only suggest high-value experiments
                            example_model = self._get_example_model_for_family(family)
                            
                            experiment = ExperimentPlan(
                                model_name=example_model,
                                quantization_method=method,
                                bit_width=bit_width,
                                priority_score=priority,
                                learning_value=self._describe_learning_value(family, method, bit_width),
                                expected_insights=self._predict_experiment_insights(family, method, bit_width),
                                resource_requirements=self._estimate_resource_requirements(family, bit_width)
                            )
                            
                            experiments.append(experiment)
        
        # Sort by priority score
        experiments.sort(key=lambda x: x.priority_score, reverse=True)
        return experiments[:limit]
    
    def _calculate_experiment_priority(self, family: str, method: str, bit_width: int, user_goals: str) -> float:
        """Calculate priority score for suggested experiment"""
        base_priority = 0.5
        
        # Boost based on method popularity
        method_success_rate = self._calculate_method_success_rate(method, family)
        base_priority += method_success_rate * 0.3
        
        # Boost based on user goals
        goal_keywords = user_goals.lower().split()
        if any(keyword in ['performance', 'speed', 'benchmark'] for keyword in goal_keywords):
            if bit_width == 4:  # Lower precision often faster
                base_priority += 0.2
        
        if any(keyword in ['quality', 'accuracy', 'precision'] for keyword in goal_keywords):
            if bit_width == 8:  # Higher precision often better quality
                base_priority += 0.2
        
        return min(base_priority, 1.0)
    
    def _get_example_model_for_family(self, family: str) -> str:
        """Get example model name for a family"""
        examples = {
            'llama': 'meta-llama/Llama-2-7b-hf',
            'mistral': 'mistralai/Mistral-7B-v0.1',
            'gpt': 'microsoft/DialoGPT-medium',
            'opt': 'facebook/opt-1.3b'
        }
        return examples.get(family, f'{family}-example-model')
    
    def _describe_learning_value(self, family: str, method: str, bit_width: int) -> str:
        """Describe the learning value of an experiment"""
        return f"Learn {method.upper()} {bit_width}-bit performance on {family.title()} architecture"
    
    def _predict_experiment_insights(self, family: str, method: str, bit_width: int) -> List[str]:
        """Predict insights from suggested experiment"""
        insights = [
            f"Performance characteristics of {method.upper()} on {family.title()} models",
            f"Memory efficiency of {bit_width}-bit quantization",
            f"Compatibility patterns for {family.title()} architecture"
        ]
        return insights
    
    def _estimate_resource_requirements(self, family: str, bit_width: int) -> Dict[str, Any]:
        """Estimate resource requirements for experiment"""
        # Simplified estimation based on family and bit width
        base_memory = {'llama': 4000, 'mistral': 3500, 'gpt': 2000, 'opt': 1500}
        memory_mb = base_memory.get(family, 2000)
        
        if bit_width == 4:
            memory_mb = int(memory_mb * 0.7)  # 4-bit uses less memory
        
        return {
            'estimated_memory_mb': memory_mb,
            'estimated_time_minutes': 2 if bit_width == 4 else 3,
            'gpu_recommended': memory_mb > 2000
        }
    
    def compare_experiments(self, experiment_ids: List[str]) -> 'ComparisonReport':
        """Generate detailed comparison report across experiments"""
        if self.backend == StorageBackend.SQLITE:
            return self._compare_experiments_sqlite(experiment_ids)
        else:
            return self._compare_experiments_json(experiment_ids)
    
    def _compare_experiments_sqlite(self, experiment_ids: List[str]) -> 'ComparisonReport':
        """Compare experiments using SQLite backend"""
        with self._get_db_connection() as conn:
            placeholders = ','.join(['?' for _ in experiment_ids])
            cursor = conn.execute(f"""
                SELECT * FROM model_records WHERE id IN ({placeholders})
            """, experiment_ids)
            
            records = []
            for row in cursor.fetchall():
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                record = ModelRecord(
                    id=row['id'],
                    model_name=row['model_name'],
                    original_size_mb=row['original_size_mb'],
                    quantized_size_mb=row['quantized_size_mb'],
                    quantization_type=row['quantization_type'],
                    bit_width=row['bit_width'],
                    target_format=row['target_format'],
                    cpu_fallback=bool(row['cpu_fallback']),
                    gpu_memory_limit=row['gpu_memory_limit'],
                    output_path=row['output_path'],
                    success=bool(row['success']),
                    execution_time=row['execution_time'],
                    error_message=row['error_message'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=metadata
                )
                records.append(record)
        
        return self._generate_comparison_report(records)
    
    def _compare_experiments_json(self, experiment_ids: List[str]) -> 'ComparisonReport':
        """Compare experiments using JSON backend"""
        data = self._load_json_data()
        records = []
        
        for exp_id in experiment_ids:
            if exp_id in data.get("model_records", {}):
                record_data = data["model_records"][exp_id]
                record = ModelRecord(**record_data)
                records.append(record)
        
        return self._generate_comparison_report(records)
    
    def _generate_comparison_report(self, records: List[ModelRecord]) -> 'ComparisonReport':
        """Generate comparison report from records"""
        models_compared = [r.model_name for r in records]
        
        # Performance metrics comparison
        performance_metrics = {}
        for record in records:
            performance_metrics[record.id] = {
                'model_name': record.model_name,
                'method': record.quantization_type,
                'bit_width': record.bit_width,
                'success': record.success,
                'size_mb': record.quantized_size_mb,
                'execution_time': record.execution_time,
                'compression_ratio': (
                    record.original_size_mb / record.quantized_size_mb 
                    if record.original_size_mb and record.quantized_size_mb 
                    else None
                )
            }
        
        # Success analysis
        successful_records = [r for r in records if r.success]
        success_analysis = {
            'total_experiments': len(records),
            'successful_experiments': len(successful_records),
            'success_rate': len(successful_records) / len(records) if records else 0,
            'avg_execution_time': (
                sum(r.execution_time for r in successful_records if r.execution_time) / len(successful_records)
                if successful_records else 0
            ),
            'best_compression': (
                max(performance_metrics.values(), key=lambda x: x['compression_ratio'] or 0)['model_name']
                if any(m['compression_ratio'] for m in performance_metrics.values()) else None
            )
        }
        
        # Optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(records)
        
        # Find best performer
        best_performer = None
        if successful_records:
            best_performer = min(
                successful_records, 
                key=lambda x: x.execution_time or float('inf')
            ).model_name
        
        return ComparisonReport(
            experiment_ids=[r.id for r in records],
            models_compared=models_compared,
            performance_metrics=performance_metrics,
            success_analysis=success_analysis,
            optimization_suggestions=optimization_suggestions,
            best_performer=best_performer
        )
    
    def _generate_optimization_suggestions(self, records: List[ModelRecord]) -> List[str]:
        """Generate optimization suggestions based on comparison"""
        suggestions = []
        
        successful_records = [r for r in records if r.success]
        failed_records = [r for r in records if not r.success]
        
        if not successful_records:
            suggestions.append("All experiments failed - consider using more conservative settings")
            return suggestions
        
        # Analyze patterns
        successful_methods = [r.quantization_type for r in successful_records]
        successful_bit_widths = [r.bit_width for r in successful_records]
        
        if len(set(successful_methods)) == 1:
            method = successful_methods[0]
            suggestions.append(f"{method.upper()} method shows consistent success")
        
        if len(set(successful_bit_widths)) == 1:
            bit_width = successful_bit_widths[0]
            suggestions.append(f"{bit_width}-bit quantization appears optimal for these models")
        
        # Performance suggestions
        if successful_records:
            avg_time = sum(r.execution_time for r in successful_records if r.execution_time) / len(successful_records)
            if avg_time > 120:  # More than 2 minutes
                suggestions.append("Consider enabling CPU fallback to reduce execution time")
        
        return suggestions
    
    def discover_learning_patterns(self) -> List['LearningPattern']:
        """Discover patterns across quantization sessions"""
        patterns = []
        records = self.get_model_records()
        
        if len(records) < 5:  # Need sufficient data
            return patterns
        
        # Pattern 1: Success rate trends over time
        time_pattern = self._analyze_success_trends_over_time(records)
        if time_pattern:
            patterns.append(time_pattern)
        
        # Pattern 2: Method effectiveness patterns
        method_pattern = self._analyze_method_effectiveness_patterns(records)
        if method_pattern:
            patterns.append(method_pattern)
        
        # Pattern 3: Model family compatibility patterns
        family_pattern = self._analyze_family_compatibility_patterns(records)
        if family_pattern:
            patterns.append(family_pattern)
        
        return patterns
    
    def _analyze_success_trends_over_time(self, records: List[ModelRecord]) -> Optional['LearningPattern']:
        """Analyze how success rates change over time"""
        # Sort records by creation time
        sorted_records = sorted(records, key=lambda x: x.created_at)
        
        # Split into early and recent experiments
        mid_point = len(sorted_records) // 2
        early_records = sorted_records[:mid_point]
        recent_records = sorted_records[mid_point:]
        
        early_success_rate = sum(1 for r in early_records if r.success) / len(early_records)
        recent_success_rate = sum(1 for r in recent_records if r.success) / len(recent_records)
        
        improvement = recent_success_rate - early_success_rate
        
        if abs(improvement) > 0.2:  # Significant change
            trend = "improving" if improvement > 0 else "declining"
            confidence = min(abs(improvement) * 2, 1.0)
            
            return LearningPattern(
                pattern_type="success_trend",
                description=f"Success rate is {trend} over time ({early_success_rate:.1%} → {recent_success_rate:.1%})",
                confidence=confidence,
                supporting_experiments=[r.id for r in sorted_records],
                recommendations=[
                    f"Continue current approach" if trend == "improving" else "Review recent failures for patterns",
                    "Your quantization skills are improving" if trend == "improving" else "Consider reverting to earlier successful strategies"
                ],
                trend_direction=trend
            )
        
        return None
    
    def _analyze_method_effectiveness_patterns(self, records: List[ModelRecord]) -> Optional['LearningPattern']:
        """Analyze which methods work best"""
        method_stats = {}
        
        for record in records:
            method = record.quantization_type
            if method not in method_stats:
                method_stats[method] = {'total': 0, 'successful': 0}
            
            method_stats[method]['total'] += 1
            if record.success:
                method_stats[method]['successful'] += 1
        
        if len(method_stats) < 2:  # Need multiple methods to compare
            return None
        
        # Calculate success rates
        method_success_rates = {
            method: stats['successful'] / stats['total']
            for method, stats in method_stats.items()
            if stats['total'] >= 2  # Minimum sample size
        }
        
        if not method_success_rates:
            return None
        
        best_method = max(method_success_rates.keys(), key=lambda k: method_success_rates[k])
        best_rate = method_success_rates[best_method]
        
        if best_rate > 0.6:  # Reasonably good success rate
            return LearningPattern(
                pattern_type="method_effectiveness",
                description=f"{best_method.upper()} shows highest success rate ({best_rate:.1%})",
                confidence=best_rate,
                supporting_experiments=[r.id for r in records if r.quantization_type == best_method],
                recommendations=[
                    f"Prioritize {best_method.upper()} method for new quantizations",
                    f"Build expertise in {best_method.upper()} configurations"
                ],
                trend_direction="stable"
            )
        
        return None
    
    def _analyze_family_compatibility_patterns(self, records: List[ModelRecord]) -> Optional['LearningPattern']:
        """Analyze which model families work best with which methods"""
        family_method_success = {}
        
        for record in records:
            family = self._extract_model_family(record.model_name)
            method = record.quantization_type
            key = f"{family}_{method}"
            
            if key not in family_method_success:
                family_method_success[key] = {'total': 0, 'successful': 0}
            
            family_method_success[key]['total'] += 1
            if record.success:
                family_method_success[key]['successful'] += 1
        
        # Find the most successful family-method combination
        best_combinations = []
        for key, stats in family_method_success.items():
            if stats['total'] >= 2:  # Minimum sample size
                success_rate = stats['successful'] / stats['total']
                if success_rate >= 0.8:  # High success rate
                    family, method = key.split('_', 1)
                    best_combinations.append((family, method, success_rate))
        
        if best_combinations:
            # Sort by success rate
            best_combinations.sort(key=lambda x: x[2], reverse=True)
            family, method, rate = best_combinations[0]
            
            return LearningPattern(
                pattern_type="family_compatibility",
                description=f"{family.title()} models work exceptionally well with {method.upper()} ({rate:.1%} success)",
                confidence=rate,
                supporting_experiments=[
                    r.id for r in records 
                    if self._extract_model_family(r.model_name) == family and r.quantization_type == method
                ],
                recommendations=[
                    f"Use {method.upper()} as first choice for {family.title()} models",
                    f"Consider {family.title()} models for reliable quantization results"
                ],
                trend_direction="stable"
            )
        
        return None
    
    def get_learning_insights_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning insights summary"""
        patterns = self.discover_learning_patterns()
        compatibility_matrix = self.get_model_compatibility_matrix()
        stats = self.get_statistics()
        
        # Calculate learning progress score
        records = self.get_model_records()
        if len(records) >= 10:
            recent_records = sorted(records, key=lambda x: x.created_at)[-5:]
            recent_success_rate = sum(1 for r in recent_records if r.success) / len(recent_records)
            learning_progress = "Advanced" if recent_success_rate > 0.8 else "Intermediate" if recent_success_rate > 0.5 else "Beginner"
        else:
            learning_progress = "Getting Started"
        
        return {
            "learning_progress": learning_progress,
            "total_experiments": stats["total_records"],
            "overall_success_rate": stats["successful_quantizations"] / stats["total_records"] if stats["total_records"] > 0 else 0,
            "discovered_patterns": len(patterns),
            "patterns_summary": [p.description for p in patterns],
            "compatibility_insights": self._summarize_compatibility_matrix(compatibility_matrix),
            "recommendations": self._generate_learning_recommendations(patterns, stats)
        }
    
    def _summarize_compatibility_matrix(self, matrix: Dict[str, Dict[str, float]]) -> List[str]:
        """Summarize key insights from compatibility matrix"""
        insights = []
        
        for family, methods in matrix.items():
            if not methods:
                continue
                
            best_method = max(methods.keys(), key=lambda k: methods[k])
            best_rate = methods[best_method]
            
            if best_rate > 0.7:
                insights.append(f"{family.title()} works best with {best_method.upper()} ({best_rate:.1%} success)")
        
        return insights
    
    def _generate_learning_recommendations(self, patterns: List['LearningPattern'], stats: Dict) -> List[str]:
        """Generate learning recommendations based on patterns and stats"""
        recommendations = []
        
        if stats["total_records"] < 5:
            recommendations.append("Experiment with different model families to build knowledge base")
        
        success_rate = stats["successful_quantizations"] / stats["total_records"] if stats["total_records"] > 0 else 0
        
        if success_rate < 0.3:
            recommendations.append("Focus on smaller, well-documented models to improve success rate")
        elif success_rate > 0.8:
            recommendations.append("Consider experimenting with larger or more challenging models")
        
        # Pattern-based recommendations
        for pattern in patterns:
            if pattern.pattern_type == "success_trend" and pattern.trend_direction == "declining":
                recommendations.append("Review recent changes - consider reverting to earlier successful approaches")
            elif pattern.pattern_type == "method_effectiveness":
                recommendations.extend(pattern.recommendations)
        
        return recommendations
    
    # === HUGGINGFACE MODEL DISCOVERY & RECOMMENDATION SYSTEM ===
    
    def discover_huggingface_models(self, user_goals: str = "general quantization", 
                                   hardware_memory_gb: int = 8, limit: int = 10) -> List[HuggingFaceModelRecommendation]:
        """Discover and recommend HuggingFace models for download and quantization"""
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            
            # Get popular models across different categories
            model_categories = self._get_model_search_categories(user_goals)
            all_recommendations = []
            
            for category in model_categories:
                models = self._fetch_models_by_category(api, category, limit=20)
                category_recommendations = self._analyze_models_for_recommendations(
                    models, user_goals, hardware_memory_gb, category
                )
                all_recommendations.extend(category_recommendations)
            
            # Sort by overall score and remove duplicates
            unique_recommendations = self._deduplicate_recommendations(all_recommendations)
            unique_recommendations.sort(key=lambda x: x.overall_score, reverse=True)
            
            return unique_recommendations[:limit]
            
        except Exception as e:
            print(f"Error discovering HuggingFace models: {e}")
            return self._get_fallback_model_recommendations(user_goals, hardware_memory_gb, limit)
    
    def _get_model_search_categories(self, user_goals: str) -> List[Dict[str, str]]:
        """Get search categories based on user goals"""
        goal_keywords = user_goals.lower().split()
        
        categories = []
        
        # Base categories always included
        categories.append({"search": "text-generation", "filter": "transformers", "name": "text-generation"})
        
        # Goal-specific categories
        if any(keyword in goal_keywords for keyword in ['chat', 'conversation', 'dialogue']):
            categories.append({"search": "conversational", "filter": "transformers", "name": "conversational"})
        
        if any(keyword in goal_keywords for keyword in ['code', 'programming', 'coding']):
            categories.append({"search": "code", "filter": "transformers", "name": "code-generation"})
        
        if any(keyword in goal_keywords for keyword in ['small', 'mobile', 'edge', 'lightweight']):
            categories.append({"search": "text-generation", "filter": "safetensors", "name": "small-models"})
        
        if any(keyword in goal_keywords for keyword in ['instruction', 'instruct', 'fine-tuned']):
            categories.append({"search": "instruction-tuned", "filter": "transformers", "name": "instruction-following"})
        
        return categories
    
    def _fetch_models_by_category(self, api: 'HfApi', category: Dict[str, str], limit: int = 20) -> List:
        """Fetch models from HuggingFace API by category"""
        try:
            models = list(api.list_models(
                search=category["search"],
                filter=category["filter"],
                sort="downloads",
                direction=-1,
                limit=limit
            ))
            return models
        except Exception as e:
            print(f"Error fetching models for category {category['name']}: {e}")
            return []
    
    def _analyze_models_for_recommendations(self, models: List, user_goals: str, 
                                          hardware_memory_gb: int, category: str) -> List[HuggingFaceModelRecommendation]:
        """Analyze models and create recommendations"""
        recommendations = []
        
        for model in models:
            try:
                # Extract model information
                model_info = self._extract_model_info(model)
                
                # Skip if model is too large for hardware
                if model_info['size_gb'] and model_info['size_gb'] > hardware_memory_gb * 2:
                    continue
                
                # Calculate recommendation scores
                scores = self._calculate_recommendation_scores(model_info, user_goals, hardware_memory_gb, category)
                
                # Generate reasoning and predictions
                reasoning = self._generate_recommendation_reasoning(model_info, scores, user_goals)
                predictions = self._predict_quantization_success(model_info)
                learning_benefits = self._identify_learning_benefits(model_info, user_goals)
                
                recommendation = HuggingFaceModelRecommendation(
                    model_name=model_info['id'],  # Use full model ID, not truncated name
                    model_id=model_info['id'],
                    description=model_info['description'],
                    download_count=model_info['downloads'],
                    model_size_gb=model_info['size_gb'],
                    architecture=model_info['architecture'],
                    tags=model_info['tags'],
                    category=self._categorize_model(model_info, user_goals),
                    
                    popularity_score=scores['popularity'],
                    compatibility_score=scores['compatibility'],
                    learning_value_score=scores['learning_value'],
                    resource_fit_score=scores['resource_fit'],
                    overall_score=scores['overall'],
                    
                    reasoning=reasoning,
                    quantization_predictions=predictions,
                    resource_requirements=self._estimate_model_resources(model_info),
                    learning_benefits=learning_benefits,
                    
                    last_updated=model_info['last_updated'],
                    license=model_info['license'],
                    is_gated=model_info['is_gated']
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                print(f"Error analyzing model {model.modelId}: {e}")
                continue
        
        return recommendations
    
    def _extract_model_info(self, model) -> Dict[str, Any]:
        """Extract relevant information from HuggingFace model object"""
        # Estimate model size from model name patterns
        size_gb = self._estimate_model_size_from_name(model.modelId)
        
        # Extract architecture from tags or model name
        architecture = self._extract_architecture(model.tags or [], model.modelId)
        
        # Clean up model name for display
        display_name = model.modelId.split('/')[-1] if '/' in model.modelId else model.modelId
        
        return {
            'id': model.modelId,
            'name': display_name,
            'description': getattr(model, 'description', 'No description available'),
            'downloads': getattr(model, 'downloads', 0) or 0,
            'size_gb': size_gb,
            'architecture': architecture,
            'tags': model.tags or [],
            'last_updated': (model.lastModified.isoformat() if model.lastModified else datetime.now().isoformat()) if hasattr(model, 'lastModified') else datetime.now().isoformat(),
            'license': getattr(model, 'license', None),
            'is_gated': getattr(model, 'gated', False)
        }
    
    def _estimate_model_size_from_name(self, model_name: str) -> Optional[float]:
        """Estimate model size in GB from model name patterns"""
        # Common size patterns in model names
        size_patterns = [
            (r'(\d+(?:\.\d+)?)[bB]', 1),  # "7B", "1.3B", etc.
            (r'(\d+(?:\.\d+)?)-?[bB]', 1),  # "7-B", "1.3-B", etc.
            (r'small', 0.5),
            (r'medium', 2),
            (r'large', 7),
            (r'xl', 20),
            (r'xxl', 50)
        ]
        
        model_name_lower = model_name.lower()
        
        for pattern, multiplier in size_patterns:
            if isinstance(pattern, str):
                if pattern in model_name_lower:
                    return multiplier
            else:
                import re
                match = re.search(pattern, model_name_lower)
                if match:
                    return float(match.group(1)) * multiplier
        
        return None
    
    def _extract_architecture(self, tags: List[str], model_name: str) -> str:
        """Extract model architecture from tags or name"""
        architecture_map = {
            'llama': 'Llama',
            'mistral': 'Mistral',
            'gpt': 'GPT',
            'opt': 'OPT',
            'bloom': 'BLOOM',
            'falcon': 'Falcon',
            'mpt': 'MPT',
            'chatglm': 'ChatGLM',
            'baichuan': 'Baichuan',
            'qwen': 'Qwen',
            'phi': 'Phi',
            'gemma': 'Gemma',
            'stablelm': 'StableLM'
        }
        
        # Check tags first
        for tag in tags:
            tag_lower = tag.lower()
            for key, arch in architecture_map.items():
                if key in tag_lower:
                    return arch
        
        # Check model name
        model_name_lower = model_name.lower()
        for key, arch in architecture_map.items():
            if key in model_name_lower:
                return arch
        
        return 'Unknown'
    
    def _calculate_recommendation_scores(self, model_info: Dict, user_goals: str, 
                                       hardware_memory_gb: int, category: str) -> Dict[str, float]:
        """Calculate various recommendation scores"""
        scores = {}
        
        # Popularity score (0-1)
        max_downloads = 10000000  # 10M downloads as reference
        scores['popularity'] = min(model_info['downloads'] / max_downloads, 1.0)
        
        # Compatibility score based on MCP history
        scores['compatibility'] = self._calculate_compatibility_score(model_info)
        
        # Learning value score
        scores['learning_value'] = self._calculate_learning_value_score(model_info, user_goals)
        
        # Resource fit score
        scores['resource_fit'] = self._calculate_resource_fit_score(model_info, hardware_memory_gb)
        
        # Overall score (weighted combination)
        scores['overall'] = (
            scores['popularity'] * 0.2 +
            scores['compatibility'] * 0.3 +
            scores['learning_value'] * 0.3 +
            scores['resource_fit'] * 0.2
        )
        
        return scores
    
    def _calculate_compatibility_score(self, model_info: Dict) -> float:
        """Calculate compatibility score based on MCP history"""
        architecture = model_info['architecture']
        family = architecture.lower()
        
        # Get success rate for this model family from MCP history
        if family in ['llama', 'mistral', 'gpt', 'opt', 'bloom', 'falcon']:
            family_success_rate = self._calculate_success_rate_for_model_type(family)
            return family_success_rate
        
        # For unknown architectures, use overall success rate
        stats = self.get_statistics()
        if stats['total_records'] > 0:
            return stats['successful_quantizations'] / stats['total_records']
        
        return 0.5  # Default middle score
    
    def _calculate_learning_value_score(self, model_info: Dict, user_goals: str) -> float:
        """Calculate learning value score"""
        score = 0.5  # Base score
        
        # Check if we haven't tested this architecture yet
        architecture = model_info['architecture'].lower()
        records = self.get_model_records()
        tested_architectures = set(self._extract_model_family(r.model_name) for r in records)
        
        if architecture not in tested_architectures:
            score += 0.3  # Bonus for new architecture
        
        # Check if model size is different from what we've tested
        if model_info['size_gb']:
            tested_sizes = [self._estimate_model_size_from_name(r.model_name) for r in records]
            tested_sizes = [s for s in tested_sizes if s is not None]
            
            if not tested_sizes or not any(abs(s - model_info['size_gb']) < 1 for s in tested_sizes):
                score += 0.2  # Bonus for new size range
        
        # Goal-specific bonuses
        goal_keywords = user_goals.lower().split()
        model_tags = [tag.lower() for tag in model_info['tags']]
        
        if any(keyword in goal_keywords for keyword in ['chat', 'conversation']) and any('chat' in tag or 'instruct' in tag for tag in model_tags):
            score += 0.1
        
        if any(keyword in goal_keywords for keyword in ['code', 'programming']) and any('code' in tag for tag in model_tags):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_resource_fit_score(self, model_info: Dict, hardware_memory_gb: int) -> float:
        """Calculate how well the model fits available resources"""
        if not model_info['size_gb']:
            return 0.7  # Default score for unknown size
        
        model_size = model_info['size_gb']
        
        # Perfect fit: model uses 50-70% of available memory
        ideal_min = hardware_memory_gb * 0.3
        ideal_max = hardware_memory_gb * 0.7
        
        if ideal_min <= model_size <= ideal_max:
            return 1.0
        elif model_size < ideal_min:
            # Small model - good but not optimal resource usage
            return 0.8
        elif model_size <= hardware_memory_gb:
            # Fits but tight on memory
            return 0.6
        else:
            # Too large - very low score
            return 0.1
    
    def _generate_recommendation_reasoning(self, model_info: Dict, scores: Dict, user_goals: str) -> List[str]:
        """Generate human-readable reasoning for recommendations"""
        reasons = []
        
        # Popularity reasons
        if scores['popularity'] > 0.7:
            reasons.append(f"Highly popular with {model_info['downloads']:,} downloads")
        elif scores['popularity'] > 0.3:
            reasons.append(f"Well-adopted with {model_info['downloads']:,} downloads")
        
        # Compatibility reasons
        if scores['compatibility'] > 0.7:
            reasons.append(f"High success rate expected for {model_info['architecture']} architecture")
        elif scores['compatibility'] > 0.4:
            reasons.append(f"Good compatibility expected based on similar models")
        else:
            reasons.append(f"Experimental - limited data for {model_info['architecture']} architecture")
        
        # Learning value reasons
        if scores['learning_value'] > 0.7:
            reasons.append("High learning value - new architecture or size to explore")
        elif scores['learning_value'] > 0.5:
            reasons.append("Good learning opportunity for skill development")
        
        # Resource fit reasons
        if scores['resource_fit'] > 0.8:
            reasons.append("Excellent fit for your hardware resources")
        elif scores['resource_fit'] > 0.6:
            reasons.append("Good fit for available memory")
        elif scores['resource_fit'] < 0.3:
            reasons.append("May require CPU fallback due to size")
        
        # Size-specific reasons
        if model_info['size_gb']:
            if model_info['size_gb'] < 2:
                reasons.append("Lightweight model - fast quantization and inference")
            elif model_info['size_gb'] > 10:
                reasons.append("Large model - comprehensive capabilities but resource-intensive")
        
        # Goal-specific reasons
        goal_keywords = user_goals.lower().split()
        if any(keyword in goal_keywords for keyword in ['beginner', 'learning', 'start']):
            if model_info['size_gb'] and model_info['size_gb'] < 3:
                reasons.append("Beginner-friendly size for learning quantization")
        
        return reasons
    
    def _predict_quantization_success(self, model_info: Dict) -> Dict[str, Dict[str, Any]]:
        """Predict quantization success probability for different methods"""
        architecture = model_info['architecture'].lower()
        size_gb = model_info['size_gb'] or 5  # Default size if unknown
        
        # Base success rates by method
        base_rates = {
            'bnb': 0.7,
            'gptq': 0.6,
            'gguf': 0.5
        }
        
        # Adjust based on architecture compatibility from MCP history
        family_success_rate = self._calculate_success_rate_for_model_type(architecture)
        
        # Adjust based on model size
        size_factor = 1.0
        if size_gb < 2:
            size_factor = 1.1  # Small models easier to quantize
        elif size_gb > 10:
            size_factor = 0.8  # Large models more challenging
        
        predictions = {}
        for method, base_rate in base_rates.items():
            # Combine base rate, architecture compatibility, and size factor
            predicted_rate = (base_rate * 0.5 + family_success_rate * 0.3) * size_factor
            success_probability = min(predicted_rate, 1.0)
            
            # Generate reasoning for the prediction
            reasoning = self._generate_prediction_reasoning(method, success_probability, architecture, size_gb, family_success_rate)
            
            predictions[method] = {
                'success_probability': success_probability,
                'reasoning': reasoning
            }
        
        return predictions
    
    def _generate_prediction_reasoning(self, method: str, success_probability: float, architecture: str, size_gb: float, family_success_rate: float) -> str:
        """Generate human-readable reasoning for quantization success prediction"""
        
        # Base reasoning by method
        method_reasoning = {
            'bnb': f"BitsAndBytes quantization generally works well with {architecture} architecture",
            'gptq': f"GPTQ quantization is moderately compatible with {architecture} models",
            'gguf': f"GGUF quantization requires model conversion and is architecture-dependent"
        }
        
        base_reason = method_reasoning.get(method, f"{method.upper()} quantization compatibility")
        
        # Size considerations
        if size_gb < 2:
            size_reason = "Small model size makes quantization more reliable"
        elif size_gb > 10:
            size_reason = "Large model size increases quantization complexity"
        else:
            size_reason = "Model size is in good range for quantization"
        
        # Historical success rate
        if family_success_rate > 0.7:
            history_reason = "Strong historical success rate for this architecture"
        elif family_success_rate > 0.4:
            history_reason = "Moderate historical success rate for this architecture"
        else:
            history_reason = "Limited historical data for this architecture"
        
        # Combine reasoning
        if success_probability > 0.7:
            return f"{base_reason}. {size_reason}. {history_reason}."
        elif success_probability > 0.4:
            return f"{base_reason}. {size_reason}. May require adjustment based on specific model."
        else:
            return f"{base_reason}. {size_reason}. Consider alternative methods or manual tuning."
    
    def _identify_learning_benefits(self, model_info: Dict, user_goals: str) -> List[str]:
        """Identify learning benefits from quantizing this model"""
        benefits = []
        
        architecture = model_info['architecture']
        size_gb = model_info['size_gb']
        
        # Architecture-specific benefits
        if architecture not in [self._extract_model_family(r.model_name) for r in self.get_model_records()]:
            benefits.append(f"Learn quantization patterns for {architecture} architecture")
        
        # Size-specific benefits
        if size_gb:
            if size_gb < 1:
                benefits.append("Practice with lightweight models - fast iteration")
            elif size_gb > 10:
                benefits.append("Experience with large models - advanced optimization")
            else:
                benefits.append("Work with mid-size models - balanced complexity")
        
        # Goal-specific benefits
        goal_keywords = user_goals.lower().split()
        if any(keyword in goal_keywords for keyword in ['production', 'deployment']):
            benefits.append("Production-ready model for deployment scenarios")
        
        if any(keyword in goal_keywords for keyword in ['performance', 'benchmark']):
            benefits.append("High-performance model for benchmarking quantization methods")
        
        # General benefits
        benefits.append("Expand your quantization experience")
        benefits.append("Build expertise with diverse model types")
        
        return benefits
    
    def _categorize_model(self, model_info: Dict, user_goals: str) -> str:
        """Categorize model based on tags and user goals"""
        tags = [tag.lower() for tag in model_info.get('tags', [])]
        model_name = model_info['name'].lower()
        goals = user_goals.lower()
        
        # Check user goals first
        if any(word in goals for word in ['chat', 'conversation', 'dialogue']):
            return 'chat'
        elif any(word in goals for word in ['code', 'programming', 'coding']):
            return 'code'
        elif any(word in goals for word in ['text-generation', 'generation', 'writing']):
            return 'text-generation'
        elif any(word in goals for word in ['embedding', 'retrieval', 'similarity']):
            return 'embedding'
        
        # Check tags
        if any(tag in tags for tag in ['conversational', 'chat', 'dialogue']):
            return 'chat'
        elif any(tag in tags for tag in ['code', 'programming']):
            return 'code'
        elif any(tag in tags for tag in ['text-generation', 'generation']):
            return 'text-generation'
        elif any(tag in tags for tag in ['sentence-similarity', 'embeddings']):
            return 'embedding'
        
        # Check model name patterns
        if any(word in model_name for word in ['chat', 'dialog', 'conversation']):
            return 'chat'
        elif any(word in model_name for word in ['code', 'programming']):
            return 'code'
        elif any(word in model_name for word in ['generation', 'gpt', 'llama']):
            return 'text-generation'
        
        return 'general'
    
    def _estimate_model_resources(self, model_info: Dict) -> Dict[str, Any]:
        """Estimate resource requirements for model quantization"""
        size_gb = model_info['size_gb'] or 5
        
        return {
            'estimated_download_size_gb': size_gb * 2,  # Account for tokenizer, config, etc.
            'estimated_quantization_memory_gb': size_gb * 1.5,  # Peak memory during quantization
            'estimated_quantized_size_gb': size_gb * 0.6,  # Typical compression ratio
            'estimated_quantization_time_minutes': max(2, size_gb * 0.5),  # Rough estimate
            'gpu_recommended': size_gb > 2,
            'cpu_fallback_viable': size_gb < 15
        }
    
    def _deduplicate_recommendations(self, recommendations: List[HuggingFaceModelRecommendation]) -> List[HuggingFaceModelRecommendation]:
        """Remove duplicate recommendations"""
        seen = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec.model_id not in seen:
                seen.add(rec.model_id)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _get_fallback_model_recommendations(self, user_goals: str, hardware_memory_gb: int, limit: int) -> List[HuggingFaceModelRecommendation]:
        """Provide fallback recommendations when API fails"""
        fallback_models = [
            {
                'id': 'microsoft/DialoGPT-small',
                'name': 'DialoGPT-small',
                'description': 'Small conversational model, great for beginners',
                'downloads': 500000,
                'size_gb': 0.5,
                'architecture': 'GPT',
                'tags': ['conversational', 'pytorch']
            },
            {
                'id': 'microsoft/DialoGPT-medium',
                'name': 'DialoGPT-medium',
                'description': 'Medium-sized conversational model',
                'downloads': 300000,
                'size_gb': 1.5,
                'architecture': 'GPT',
                'tags': ['conversational', 'pytorch']
            },
            {
                'id': 'facebook/opt-1.3b',
                'name': 'OPT-1.3B',
                'description': 'Open Pretrained Transformer by Meta',
                'downloads': 200000,
                'size_gb': 3.0,
                'architecture': 'OPT',
                'tags': ['text-generation', 'pytorch']
            }
        ]
        
        recommendations = []
        for model_info in fallback_models:
            if model_info['size_gb'] <= hardware_memory_gb:
                scores = self._calculate_recommendation_scores(model_info, user_goals, hardware_memory_gb, 'fallback')
                reasoning = self._generate_recommendation_reasoning(model_info, scores, user_goals)
                predictions = self._predict_quantization_success(model_info)
                learning_benefits = self._identify_learning_benefits(model_info, user_goals)
                
                recommendation = HuggingFaceModelRecommendation(
                    model_name=model_info['id'],  # Use full model ID, not truncated name
                    model_id=model_info['id'],
                    description=model_info['description'],
                    download_count=model_info['downloads'],
                    model_size_gb=model_info['size_gb'],
                    architecture=model_info['architecture'],
                    tags=model_info['tags'],
                    category='fallback',
                    
                    popularity_score=scores['popularity'],
                    compatibility_score=scores['compatibility'],
                    learning_value_score=scores['learning_value'],
                    resource_fit_score=scores['resource_fit'],
                    overall_score=scores['overall'],
                    
                    reasoning=reasoning,
                    quantization_predictions=predictions,
                    resource_requirements=self._estimate_model_resources(model_info),
                    learning_benefits=learning_benefits,
                    
                    last_updated=datetime.now().isoformat(),
                    license='MIT',
                    is_gated=False
                )
                
                recommendations.append(recommendation)
        
        return recommendations[:limit]