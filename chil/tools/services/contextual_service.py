"""
Contextual Framework Tools Service Implementation.
"""

import grpc
import time
from concurrent import futures
from typing import Dict, List, Any

import spacy
import numpy as np
from gensim import models
import faiss
from textblob import TextBlob

from ..protos import tools_pb2
from ..protos import tools_pb2_grpc


class ContextualToolsServicer(tools_pb2_grpc.ContextualToolsServicer):
    """gRPC service for contextual framework tools."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp = spacy.load(config.get('spacy_model', 'en_core_web_sm'))
        self.faiss_index = self._load_faiss_index(config.get('faiss_index_path'))
        self.topic_models = {}
        self._init_models()
    
    def _init_models(self):
        """Initialize NLP models."""
        # Disable unnecessary spaCy components for speed
        self.nlp.disable_pipes([
            pipe for pipe in self.nlp.pipe_names 
            if pipe not in ['tagger', 'parser', 'ner']
        ])
    
    def _load_faiss_index(self, path: str):
        """Load or create FAISS index."""
        try:
            if path and path != ':memory:':
                return faiss.read_index(path)
        except:
            pass
        # Create new index if not found
        dimension = 384  # Assuming sentence-transformers dimension
        return faiss.IndexFlatL2(dimension)
    
    def AnalyzeText(self, request, context):
        """Analyze text using spaCy."""
        try:
            doc = self.nlp(request.text)
            response = tools_pb2.SpacyResponse()
            
            # Extract entities
            if 'entities' in request.analyses:
                for ent in doc.ents:
                    entity = response.entities.add()
                    entity.text = ent.text
                    entity.label = ent.label_
                    entity.start = ent.start_char
                    entity.end = ent.end_char
            
            # Extract tokens with POS and dependencies
            if 'pos' in request.analyses or 'dependency' in request.analyses:
                for token in doc:
                    token_msg = response.tokens.add()
                    token_msg.text = token.text
                    token_msg.pos = token.pos_
                    token_msg.dep = token.dep_
            
            return response
            
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"spaCy error: {str(e)}")
    
    def ExtractTopics(self, request, context):
        """Extract topics using Gensim."""
        try:
            # Preprocess documents
            texts = [doc.lower().split() for doc in request.documents]
            
            # Create dictionary and corpus
            from gensim import corpora
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            
            # Train LDA model
            lda = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=request.num_topics,
                random_state=42,
                passes=10,
                alpha='auto'
            )
            
            response = tools_pb2.GensimResponse()
            
            # Extract topics
            for topic_id in range(request.num_topics):
                topic = response.topics.add()
                topic.id = topic_id
                
                # Get top words for topic
                for word_id, weight in lda.get_topic_terms(topic_id, topn=10):
                    word_weight = topic.words.add()
                    word_weight.word = dictionary[word_id]
                    word_weight.weight = float(weight)
            
            return response
            
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"Gensim error: {str(e)}")
    
    def SearchSimilar(self, request, context):
        """Search similar vectors using FAISS."""
        try:
            # Convert query vector to numpy array
            query_vector = np.array(request.query_vector, dtype=np.float32)
            query_vector = query_vector.reshape(1, -1)
            
            # Search
            distances, indices = self.faiss_index.search(query_vector, request.k)
            
            response = tools_pb2.FaissResponse()
            
            # Build response
            for i in range(len(indices[0])):
                if indices[0][i] != -1:  # Valid result
                    neighbor = response.neighbors.add()
                    neighbor.id = int(indices[0][i])
                    neighbor.distance = float(distances[0][i])
                    # Add metadata if available
                    neighbor.metadata["score"] = str(1.0 / (1.0 + distances[0][i]))
            
            return response
            
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"FAISS error: {str(e)}")
    
    def AnalyzeSentiment(self, request, context):
        """Analyze sentiment using TextBlob."""
        try:
            response = tools_pb2.SentimentResponse()
            
            if request.model == "textblob" or not request.model:
                blob = TextBlob(request.text)
                response.polarity = blob.sentiment.polarity
                response.subjectivity = blob.sentiment.subjectivity
                
                # Determine label
                if response.polarity > 0.1:
                    response.label = "positive"
                elif response.polarity < -0.1:
                    response.label = "negative"
                else:
                    response.label = "neutral"
            
            return response
            
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"Sentiment error: {str(e)}")


def serve(port: int = 50052, config: Dict[str, Any] = None):
    """Start the contextual tools gRPC server."""
    config = config or {}
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    tools_pb2_grpc.add_ContextualToolsServicer_to_server(
        ContextualToolsServicer(config), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    return server


if __name__ == '__main__':
    server = serve()
    server.wait_for_termination()