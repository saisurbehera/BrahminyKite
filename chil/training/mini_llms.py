"""
Mini-LLM registry and configurations for each framework domain.

Each framework gets a specialized small language model optimized
for its specific verification tasks.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    T5ForConditionalGeneration, T5Config,
    BertModel, BertConfig,
    RobertaModel, RobertaConfig,
    DistilBertModel, DistilBertConfig,
    AlbertModel, AlbertConfig
)
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class FrameworkType(Enum):
    """Framework types for specialized models."""
    EMPIRICAL = "empirical"
    CONTEXTUAL = "contextual" 
    CONSISTENCY = "consistency"
    POWER_DYNAMICS = "power_dynamics"
    UTILITY = "utility"
    EVOLUTION = "evolution"


@dataclass
class MiniLLMSpec:
    """Specification for a mini-LLM."""
    
    # Model architecture
    base_model: str  # HuggingFace model name or architecture type
    model_size: str  # "tiny", "small", "base"
    max_length: int = 512
    vocab_size: int = 32000
    
    # Domain-specific parameters
    task_type: str  # "classification", "generation", "verification"
    num_labels: Optional[int] = None
    output_dim: int = 768
    
    # Performance constraints
    max_memory_mb: int = 500  # Maximum memory usage
    max_latency_ms: int = 100  # Maximum inference latency
    quantization: str = "int8"  # "fp16", "int8", "int4"
    
    # Training parameters
    learning_rate: float = 5e-5
    batch_size: int = 16
    gradient_accumulation: int = 2
    
    # Domain-specific tools integration
    tool_endpoints: List[str] = None
    preprocessing_steps: List[str] = None


class MiniLLMRegistry:
    """Registry of mini-LLMs for each framework."""
    
    # Framework-specific model specifications
    FRAMEWORK_SPECS = {
        FrameworkType.EMPIRICAL: MiniLLMSpec(
            base_model="google/flan-t5-small",
            model_size="small",
            max_length=512,
            task_type="verification",
            output_dim=512,
            max_memory_mb=400,
            max_latency_ms=80,
            quantization="int8",
            tool_endpoints=["z3", "lean", "duckdb", "sparql"],
            preprocessing_steps=["logical_parsing", "fact_extraction"]
        ),
        
        FrameworkType.CONTEXTUAL: MiniLLMSpec(
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            model_size="small", 
            max_length=384,
            task_type="classification",
            num_labels=3,  # positive, negative, neutral
            output_dim=384,
            max_memory_mb=300,
            max_latency_ms=60,
            quantization="int8",
            tool_endpoints=["spacy", "gensim", "faiss", "sentiment"],
            preprocessing_steps=["tokenization", "embedding", "cultural_context"]
        ),
        
        FrameworkType.CONSISTENCY: MiniLLMSpec(
            base_model="microsoft/DialoGPT-small",
            model_size="small",
            max_length=256,
            task_type="verification",
            output_dim=768,
            max_memory_mb=350,
            max_latency_ms=70,
            quantization="int8",
            tool_endpoints=["sqlite_fts", "datalog"],
            preprocessing_steps=["logical_extraction", "pattern_matching"]
        ),
        
        FrameworkType.POWER_DYNAMICS: MiniLLMSpec(
            base_model="unitary/toxic-bert",
            model_size="small",
            max_length=512,
            task_type="classification",
            num_labels=5,  # Multiple bias types
            output_dim=768,
            max_memory_mb=400,
            max_latency_ms=90,
            quantization="int8",
            tool_endpoints=["fairness", "network_analysis", "clustering", "bias_detection"],
            preprocessing_steps=["bias_preprocessing", "demographic_inference"]
        ),
        
        FrameworkType.UTILITY: MiniLLMSpec(
            base_model="distilbert-base-uncased",
            model_size="small",
            max_length=256,
            task_type="generation",
            output_dim=768,
            max_memory_mb=300,
            max_latency_ms=50,
            quantization="int8",
            tool_endpoints=["xgboost", "optimization", "game_theory", "utility_calc"],
            preprocessing_steps=["feature_extraction", "outcome_modeling"]
        ),
        
        FrameworkType.EVOLUTION: MiniLLMSpec(
            base_model="microsoft/DialoGPT-medium",
            model_size="medium",
            max_length=512,
            task_type="generation",
            output_dim=1024,
            max_memory_mb=600,
            max_latency_ms=120,
            quantization="int8",
            tool_endpoints=["genetic_algorithm", "evolution_strategy", "rl_training"],
            preprocessing_steps=["temporal_analysis", "adaptation_tracking"]
        )
    }
    
    def __init__(self, device: str = "cuda", cache_dir: str = "./models/mini_llms"):
        self.device = device
        self.cache_dir = cache_dir
        self.models: Dict[FrameworkType, torch.nn.Module] = {}
        self.tokenizers: Dict[FrameworkType, Any] = {}
        self.configs: Dict[FrameworkType, MiniLLMSpec] = {}
    
    def load_framework_model(self, framework: FrameworkType) -> torch.nn.Module:
        """Load and configure model for specific framework."""
        if framework in self.models:
            return self.models[framework]
        
        spec = self.FRAMEWORK_SPECS[framework]
        self.configs[framework] = spec
        
        # Load base model and tokenizer
        model, tokenizer = self._create_model_from_spec(spec)
        
        # Apply quantization
        if spec.quantization == "int8":
            model = self._apply_int8_quantization(model)
        elif spec.quantization == "int4":
            model = self._apply_int4_quantization(model)
        
        # Move to device
        model = model.to(self.device)
        model.eval()
        
        # Cache models
        self.models[framework] = model
        self.tokenizers[framework] = tokenizer
        
        return model
    
    def _create_model_from_spec(self, spec: MiniLLMSpec) -> tuple:
        """Create model and tokenizer from specification."""
        
        if "flan-t5" in spec.base_model:
            # T5-based model for empirical reasoning
            model = T5ForConditionalGeneration.from_pretrained(
                spec.base_model,
                cache_dir=self.cache_dir
            )
            tokenizer = AutoTokenizer.from_pretrained(
                spec.base_model,
                cache_dir=self.cache_dir
            )
            
        elif "MiniLM" in spec.base_model:
            # Sentence transformer for contextual analysis
            model = AutoModel.from_pretrained(
                spec.base_model,
                cache_dir=self.cache_dir
            )
            tokenizer = AutoTokenizer.from_pretrained(
                spec.base_model,
                cache_dir=self.cache_dir
            )
            
        elif "DialoGPT" in spec.base_model:
            # Dialog model for consistency/evolution
            model = AutoModel.from_pretrained(
                spec.base_model,
                cache_dir=self.cache_dir
            )
            tokenizer = AutoTokenizer.from_pretrained(
                spec.base_model,
                cache_dir=self.cache_dir,
                pad_token="<pad>"
            )
            
        elif "toxic-bert" in spec.base_model or "distilbert" in spec.base_model:
            # BERT variants for classification tasks
            model = AutoModel.from_pretrained(
                spec.base_model,
                cache_dir=self.cache_dir
            )
            tokenizer = AutoTokenizer.from_pretrained(
                spec.base_model,
                cache_dir=self.cache_dir
            )
            
        else:
            raise ValueError(f"Unsupported base model: {spec.base_model}")
        
        # Add task-specific head if needed
        if spec.task_type == "classification" and spec.num_labels:
            model = self._add_classification_head(model, spec.num_labels, spec.output_dim)
        elif spec.task_type == "verification":
            model = self._add_verification_head(model, spec.output_dim)
        
        return model, tokenizer
    
    def _add_classification_head(self, model: nn.Module, num_labels: int, hidden_dim: int) -> nn.Module:
        """Add classification head to model."""
        class ClassificationModel(nn.Module):
            def __init__(self, base_model, num_labels, hidden_dim):
                super().__init__()
                self.base_model = base_model
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, num_labels)
                )
            
            def forward(self, input_ids, attention_mask=None, **kwargs):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get pooled output (different for different model types)
                if hasattr(outputs, 'pooler_output'):
                    pooled = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    # Mean pooling for models without pooler
                    pooled = outputs.last_hidden_state.mean(dim=1)
                else:
                    pooled = outputs[0].mean(dim=1)
                
                logits = self.classifier(pooled)
                return {'logits': logits, 'hidden_states': outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]}
        
        return ClassificationModel(model, num_labels, hidden_dim)
    
    def _add_verification_head(self, model: nn.Module, hidden_dim: int) -> nn.Module:
        """Add verification head to model."""
        class VerificationModel(nn.Module):
            def __init__(self, base_model, hidden_dim):
                super().__init__()
                self.base_model = base_model
                self.verification_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()  # Verification score between 0 and 1
                )
                self.confidence_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 1),
                    nn.Sigmoid()  # Confidence score
                )
            
            def forward(self, input_ids, attention_mask=None, **kwargs):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                
                if hasattr(outputs, 'pooler_output'):
                    pooled = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    pooled = outputs.last_hidden_state.mean(dim=1)
                else:
                    pooled = outputs[0].mean(dim=1)
                
                verification_score = self.verification_head(pooled)
                confidence_score = self.confidence_head(pooled)
                
                return {
                    'verification_score': verification_score,
                    'confidence_score': confidence_score,
                    'hidden_states': outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                }
        
        return VerificationModel(model, hidden_dim)
    
    def _apply_int8_quantization(self, model: nn.Module) -> nn.Module:
        """Apply INT8 quantization to model."""
        try:
            from transformers import BitsAndBytesConfig
            
            # This is a simplified version - in practice, you'd use proper quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            return model
        except ImportError:
            print("BitsAndBytesConfig not available, skipping quantization")
            return model
    
    def _apply_int4_quantization(self, model: nn.Module) -> nn.Module:
        """Apply INT4 quantization to model."""
        # Placeholder for INT4 quantization
        return model
    
    def get_model(self, framework: FrameworkType) -> torch.nn.Module:
        """Get model for framework (load if not cached)."""
        if framework not in self.models:
            self.load_framework_model(framework)
        return self.models[framework]
    
    def get_tokenizer(self, framework: FrameworkType):
        """Get tokenizer for framework."""
        if framework not in self.tokenizers:
            self.load_framework_model(framework)
        return self.tokenizers[framework]
    
    def get_spec(self, framework: FrameworkType) -> MiniLLMSpec:
        """Get specification for framework."""
        return self.FRAMEWORK_SPECS[framework]
    
    def estimate_memory_usage(self) -> Dict[FrameworkType, float]:
        """Estimate memory usage for each framework model."""
        memory_usage = {}
        
        for framework, spec in self.FRAMEWORK_SPECS.items():
            # Base memory estimate
            base_memory = spec.max_memory_mb
            
            # Quantization reduction
            if spec.quantization == "int8":
                base_memory *= 0.5
            elif spec.quantization == "int4":
                base_memory *= 0.25
            
            memory_usage[framework] = base_memory
        
        return memory_usage
    
    def benchmark_latency(self, framework: FrameworkType, batch_size: int = 1) -> float:
        """Benchmark inference latency for framework model."""
        model = self.get_model(framework)
        tokenizer = self.get_tokenizer(framework)
        
        # Create dummy input
        dummy_text = "This is a test claim for benchmarking."
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.get_spec(framework).max_length
        ).to(self.device)
        
        # Warm up
        for _ in range(3):
            with torch.no_grad():
                _ = model(**inputs)
        
        # Benchmark
        import time
        torch.cuda.synchronize() if self.device == "cuda" else None
        
        start_time = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = model(**inputs)
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_time = time.time()
        
        avg_latency_ms = (end_time - start_time) / 10 * 1000
        return avg_latency_ms
    
    def load_all_models(self):
        """Load all framework models."""
        for framework in FrameworkType:
            print(f"Loading {framework.value} model...")
            self.load_framework_model(framework)
        
        print("All models loaded successfully!")
    
    def unload_model(self, framework: FrameworkType):
        """Unload model to free memory."""
        if framework in self.models:
            del self.models[framework]
            del self.tokenizers[framework]
            torch.cuda.empty_cache() if self.device == "cuda" else None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all registered models."""
        info = {}
        
        for framework, spec in self.FRAMEWORK_SPECS.items():
            info[framework.value] = {
                'base_model': spec.base_model,
                'model_size': spec.model_size,
                'task_type': spec.task_type,
                'max_memory_mb': spec.max_memory_mb,
                'max_latency_ms': spec.max_latency_ms,
                'quantization': spec.quantization,
                'tool_endpoints': spec.tool_endpoints,
                'loaded': framework in self.models
            }
        
        return info


# Convenience function for quick model access
def get_framework_model(framework: str, device: str = "cuda") -> torch.nn.Module:
    """Quick access to framework model."""
    registry = MiniLLMRegistry(device=device)
    framework_enum = FrameworkType(framework)
    return registry.get_model(framework_enum)