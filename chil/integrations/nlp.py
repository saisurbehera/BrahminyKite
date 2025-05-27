"""
NLP Model Integrations

Provides integration with NLP models and libraries like spaCy and Transformers.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

# Try to import NLP libraries
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    spacy = None

try:
    from transformers import (
        pipeline, 
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelForQuestionAnswering
    )
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None


@dataclass
class Entity:
    """Named entity"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class Sentiment:
    """Sentiment analysis result"""
    label: str  # POSITIVE, NEGATIVE, NEUTRAL
    score: float  # Confidence score
    
    @property
    def polarity(self) -> float:
        """Get polarity value (-1 to 1)"""
        if self.label == "POSITIVE":
            return self.score
        elif self.label == "NEGATIVE":
            return -self.score
        else:
            return 0.0


@dataclass
class TextClassification:
    """Text classification result"""
    label: str
    score: float


@dataclass
class QuestionAnswer:
    """Question answering result"""
    answer: str
    score: float
    start: int
    end: int


@dataclass
class TextEmbedding:
    """Text embedding/vector representation"""
    vector: np.ndarray
    model: str
    
    def similarity(self, other: 'TextEmbedding') -> float:
        """Calculate cosine similarity with another embedding"""
        if self.vector.shape != other.vector.shape:
            raise ValueError("Embeddings must have same dimensions")
        
        dot_product = np.dot(self.vector, other.vector)
        norm_a = np.linalg.norm(self.vector)
        norm_b = np.linalg.norm(other.vector)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


class NLPProcessor(ABC):
    """Abstract base class for NLP processors"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text"""
        pass
    
    @abstractmethod
    def analyze_sentiment(self, text: str) -> Sentiment:
        """Analyze sentiment of text"""
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> TextEmbedding:
        """Get vector embedding of text"""
        pass


class SpacyProcessor(NLPProcessor):
    """spaCy-based NLP processing"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        super().__init__()
        
        if not HAS_SPACY:
            raise ImportError("spaCy not installed. Install with: pip install spacy")
        
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            self.logger.warning(f"Model {model_name} not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
        
        self.logger.info(f"Loaded spaCy model: {model_name}")
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities using spaCy"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char
            ))
        
        return entities
    
    def analyze_sentiment(self, text: str) -> Sentiment:
        """Basic sentiment analysis using spaCy (limited)"""
        # spaCy doesn't have built-in sentiment, so we'll use a simple heuristic
        doc = self.nlp(text)
        
        # Count positive and negative words (very basic)
        positive_words = ["good", "great", "excellent", "positive", "wonderful", "best", "love", "happy"]
        negative_words = ["bad", "terrible", "negative", "worst", "hate", "awful", "poor", "sad"]
        
        positive_count = sum(1 for token in doc if token.text.lower() in positive_words)
        negative_count = sum(1 for token in doc if token.text.lower() in negative_words)
        
        if positive_count > negative_count:
            return Sentiment(label="POSITIVE", score=0.7)
        elif negative_count > positive_count:
            return Sentiment(label="NEGATIVE", score=0.7)
        else:
            return Sentiment(label="NEUTRAL", score=0.5)
    
    def get_embedding(self, text: str) -> TextEmbedding:
        """Get text embedding using spaCy vectors"""
        doc = self.nlp(text)
        
        # Get document vector (average of word vectors)
        if doc.has_vector:
            vector = doc.vector
        else:
            # Fallback to average of token vectors
            vectors = [token.vector for token in doc if token.has_vector]
            if vectors:
                vector = np.mean(vectors, axis=0)
            else:
                # Random vector if no word vectors available
                vector = np.random.rand(96)  # Default spaCy vector size
        
        return TextEmbedding(vector=vector, model=self.nlp.meta['name'])
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using spaCy (noun phrases and important tokens)"""
        doc = self.nlp(text)
        
        # Extract noun phrases
        keywords = []
        
        # Add noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to short phrases
                keywords.append((chunk.text.lower(), 1.0))
        
        # Add important single tokens (nouns, verbs)
        for token in doc:
            if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                not token.is_stop and 
                len(token.text) > 2):
                keywords.append((token.text.lower(), 0.8))
        
        # Count frequencies
        keyword_counts = {}
        for keyword, score in keywords:
            if keyword in keyword_counts:
                keyword_counts[keyword] += score
            else:
                keyword_counts[keyword] = score
        
        # Sort by frequency and return top N
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_n]
    
    def dependency_parse(self, text: str) -> List[Dict[str, Any]]:
        """Get dependency parse of text"""
        doc = self.nlp(text)
        dependencies = []
        
        for token in doc:
            dependencies.append({
                'text': token.text,
                'pos': token.pos_,
                'dep': token.dep_,
                'head': token.head.text,
                'children': [child.text for child in token.children]
            })
        
        return dependencies


class TransformerProcessor(NLPProcessor):
    """Transformer-based NLP processing using Hugging Face"""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers not installed. Install with: pip install transformers torch")
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize pipelines (lazy loading)
        self._ner_pipeline = None
        self._sentiment_pipeline = None
        self._embedding_model = None
        self._qa_pipeline = None
        self._classification_pipeline = None
    
    @property
    def ner_pipeline(self):
        """Get or create NER pipeline"""
        if self._ner_pipeline is None:
            self._ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1
            )
        return self._ner_pipeline
    
    @property
    def sentiment_pipeline(self):
        """Get or create sentiment pipeline"""
        if self._sentiment_pipeline is None:
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if self.device == "cuda" else -1
            )
        return self._sentiment_pipeline
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities using transformers"""
        try:
            results = self.ner_pipeline(text)
            entities = []
            
            for result in results:
                entities.append(Entity(
                    text=result['word'],
                    label=result['entity_group'],
                    start=result['start'],
                    end=result['end'],
                    confidence=result['score']
                ))
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Entity extraction error: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Sentiment:
        """Analyze sentiment using transformers"""
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            results = self.sentiment_pipeline(text)
            
            if results:
                result = results[0]
                label = result['label']
                
                # Convert star ratings to sentiment labels
                if '5 stars' in label or '4 stars' in label:
                    return Sentiment(label="POSITIVE", score=result['score'])
                elif '1 star' in label or '2 stars' in label:
                    return Sentiment(label="NEGATIVE", score=result['score'])
                else:
                    return Sentiment(label="NEUTRAL", score=result['score'])
            
            return Sentiment(label="NEUTRAL", score=0.5)
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return Sentiment(label="NEUTRAL", score=0.5)
    
    def get_embedding(self, text: str) -> TextEmbedding:
        """Get text embedding using sentence transformers"""
        try:
            # Use sentence-transformers for better embeddings
            from sentence_transformers import SentenceTransformer
            
            if self._embedding_model is None:
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                if self.device == "cuda":
                    self._embedding_model = self._embedding_model.to('cuda')
            
            # Get embedding
            embedding = self._embedding_model.encode(text, convert_to_numpy=True)
            
            return TextEmbedding(vector=embedding, model='all-MiniLM-L6-v2')
            
        except ImportError:
            self.logger.warning("sentence-transformers not installed, using basic BERT embeddings")
            
            # Fallback to basic BERT embeddings
            from transformers import AutoModel
            
            if self._embedding_model is None:
                self._embedding_model = AutoModel.from_pretrained('bert-base-uncased')
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                if self.device == "cuda":
                    self._embedding_model = self._embedding_model.to('cuda')
            
            # Tokenize and get embeddings
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            if self.device == "cuda":
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._embedding_model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
            
            return TextEmbedding(vector=embedding, model='bert-base-uncased')
            
        except Exception as e:
            self.logger.error(f"Embedding generation error: {e}")
            # Return random embedding as fallback
            return TextEmbedding(vector=np.random.rand(384), model='random')
    
    def classify_text(self, text: str, labels: List[str]) -> List[TextClassification]:
        """Classify text into given labels using zero-shot classification"""
        try:
            if self._classification_pipeline is None:
                self._classification_pipeline = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if self.device == "cuda" else -1
                )
            
            result = self._classification_pipeline(text, candidate_labels=labels)
            
            classifications = []
            for label, score in zip(result['labels'], result['scores']):
                classifications.append(TextClassification(label=label, score=score))
            
            return classifications
            
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return []
    
    def answer_question(self, question: str, context: str) -> Optional[QuestionAnswer]:
        """Answer a question given context"""
        try:
            if self._qa_pipeline is None:
                self._qa_pipeline = pipeline(
                    "question-answering",
                    model="deepset/roberta-base-squad2",
                    device=0 if self.device == "cuda" else -1
                )
            
            result = self._qa_pipeline(question=question, context=context)
            
            return QuestionAnswer(
                answer=result['answer'],
                score=result['score'],
                start=result['start'],
                end=result['end']
            )
            
        except Exception as e:
            self.logger.error(f"Question answering error: {e}")
            return None
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize text using transformers"""
        try:
            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device == "cuda" else -1
            )
            
            # Handle long texts by chunking
            max_chunk_length = 1024
            if len(text) > max_chunk_length:
                # Simple chunking by sentences
                sentences = text.split('. ')
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < max_chunk_length:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Summarize each chunk
                summaries = []
                for chunk in chunks:
                    summary = summarizer(chunk, max_length=max_length // len(chunks), min_length=30)
                    summaries.append(summary[0]['summary_text'])
                
                return " ".join(summaries)
            else:
                summary = summarizer(text, max_length=max_length, min_length=30)
                return summary[0]['summary_text']
                
        except Exception as e:
            self.logger.error(f"Summarization error: {e}")
            # Simple fallback - return first sentences
            sentences = text.split('. ')
            return '. '.join(sentences[:3]) + '.'


class NLPHub:
    """Hub for coordinating multiple NLP processors"""
    
    def __init__(self, use_gpu: bool = True):
        self.logger = logging.getLogger(__name__)
        self.processors = {}
        
        # Initialize available processors
        if HAS_SPACY:
            try:
                self.processors['spacy'] = SpacyProcessor()
                self.logger.info("SpaCy processor initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize SpaCy: {e}")
        
        if HAS_TRANSFORMERS:
            try:
                device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
                self.processors['transformers'] = TransformerProcessor(device)
                self.logger.info(f"Transformers processor initialized on {device}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Transformers: {e}")
        
        if not self.processors:
            self.logger.error("No NLP processors available!")
    
    def extract_entities(self, text: str, processor: str = 'transformers') -> List[Entity]:
        """Extract entities using specified processor"""
        if processor in self.processors:
            return self.processors[processor].extract_entities(text)
        elif self.processors:
            # Fallback to any available processor
            fallback = list(self.processors.keys())[0]
            self.logger.warning(f"Processor {processor} not available, using {fallback}")
            return self.processors[fallback].extract_entities(text)
        else:
            return []
    
    def analyze_sentiment(self, text: str, processor: str = 'transformers') -> Sentiment:
        """Analyze sentiment using specified processor"""
        if processor in self.processors:
            return self.processors[processor].analyze_sentiment(text)
        elif self.processors:
            fallback = list(self.processors.keys())[0]
            return self.processors[fallback].analyze_sentiment(text)
        else:
            return Sentiment(label="NEUTRAL", score=0.5)
    
    def get_embedding(self, text: str, processor: str = 'transformers') -> Optional[TextEmbedding]:
        """Get embedding using specified processor"""
        if processor in self.processors:
            return self.processors[processor].get_embedding(text)
        elif self.processors:
            fallback = list(self.processors.keys())[0]
            return self.processors[fallback].get_embedding(text)
        else:
            return None
    
    def calculate_similarity(self, text1: str, text2: str, processor: str = 'transformers') -> float:
        """Calculate semantic similarity between two texts"""
        emb1 = self.get_embedding(text1, processor)
        emb2 = self.get_embedding(text2, processor)
        
        if emb1 and emb2:
            return emb1.similarity(emb2)
        else:
            return 0.0
    
    def extract_keywords_all(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract keywords using all available processors"""
        results = {}
        
        if 'spacy' in self.processors:
            spacy_proc = self.processors['spacy']
            if hasattr(spacy_proc, 'extract_keywords'):
                results['spacy'] = spacy_proc.extract_keywords(text)
        
        # Add more keyword extraction methods as needed
        
        return results