"""
Knowledge graph and retrieval components for neuro-symbolic reasoning.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from sentence_transformers import SentenceTransformer
import spacy
from collections import defaultdict


class KnowledgeGraph:
    """Base knowledge graph class."""
    
    def __init__(self, name: str):
        self.name = name
        self.triples = []
        self.entity_to_triples = defaultdict(list)
        self.relation_to_triples = defaultdict(list)
    
    def add_triple(self, subject: str, relation: str, object: str, confidence: float = 1.0):
        """Add a triple to the knowledge graph."""
        triple = {
            "subject": subject,
            "relation": relation,
            "object": object,
            "confidence": confidence
        }
        self.triples.append(triple)
        self.entity_to_triples[subject].append(triple)
        self.entity_to_triples[object].append(triple)
        self.relation_to_triples[relation].append(triple)
    
    def get_triples_for_entity(self, entity: str) -> List[Dict[str, Any]]:
        """Get all triples involving a specific entity."""
        return self.entity_to_triples.get(entity, [])
    
    def get_triples_for_relation(self, relation: str) -> List[Dict[str, Any]]:
        """Get all triples with a specific relation."""
        return self.relation_to_triples.get(relation, [])
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant triples given a query."""
        # Simple keyword matching - can be extended with embeddings
        query_lower = query.lower()
        results = []
        
        for triple in self.triples:
            score = 0
            if query_lower in triple["subject"].lower():
                score += 1
            if query_lower in triple["relation"].lower():
                score += 1
            if query_lower in triple["object"].lower():
                score += 1
            
            if score > 0:
                results.append((triple, score))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return [triple for triple, score in results[:top_k]]
    
    def save(self, path: str):
        """Save knowledge graph to file."""
        with open(path, 'w') as f:
            json.dump({
                "name": self.name,
                "triples": self.triples
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        """Load knowledge graph from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        kg = cls(data["name"])
        for triple in data["triples"]:
            kg.add_triple(
                triple["subject"],
                triple["relation"],
                triple["object"],
                triple.get("confidence", 1.0)
            )
        
        return kg


class ConceptNetKnowledge(KnowledgeGraph):
    """ConceptNet knowledge graph wrapper."""
    
    def __init__(self, cache_path: str = "data/conceptnet_cache.json"):
        super().__init__("conceptnet")
        self.cache_path = cache_path
        self.load_cache()
    
    def load_cache(self):
        """Load cached ConceptNet triples."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
                for triple in data["triples"]:
                    self.add_triple(
                        triple["subject"],
                        triple["relation"],
                        triple["object"],
                        triple.get("confidence", 1.0)
                    )
    
    def create_cache(self, conceptnet_path: str, max_triples: int = 100000):
        """Create cache from ConceptNet file."""
        # This is a simplified version - in practice you'd want to process the full ConceptNet
        triples = []
        
        with open(conceptnet_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_triples:
                    break
                
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    subject = parts[1].replace('/c/en/', '').replace('_', ' ')
                    relation = parts[0].replace('/r/', '')
                    object = parts[2].replace('/c/en/', '').replace('_', ' ')
                    confidence = float(parts[3]) if len(parts) > 3 else 1.0
                    
                    triples.append({
                        "subject": subject,
                        "relation": relation,
                        "object": object,
                        "confidence": confidence
                    })
        
        # Save cache
        with open(self.cache_path, 'w') as f:
            json.dump({"triples": triples}, f, indent=2)
        
        # Load into memory
        for triple in triples:
            self.add_triple(
                triple["subject"],
                triple["relation"],
                triple["object"],
                triple["confidence"]
            )


class KnowledgeRetriever:
    """Knowledge retriever using bi-encoder similarity."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.kg = knowledge_graph
        self.encoder = SentenceTransformer(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Pre-encode all triples
        self.triple_embeddings = None
        self.triple_texts = []
        self._encode_triples()
    
    def _encode_triples(self):
        """Pre-encode all triples in the knowledge graph."""
        triple_texts = []
        for triple in self.kg.triples:
            text = f"{triple['subject']} {triple['relation']} {triple['object']}"
            triple_texts.append(text)
        
        self.triple_texts = triple_texts
        self.triple_embeddings = self.encoder.encode(triple_texts, convert_to_tensor=True)
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using spaCy."""
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append(ent.text.lower())
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            entities.append(chunk.text.lower())
        
        # Extract important words (nouns, verbs, adjectives)
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and len(token.text) > 2:
                entities.append(token.text.lower())
        
        return list(set(entities))
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge triples for a query."""
        # Extract entities from query
        entities = self.extract_entities(query)
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_tensor=True)
        
        # Compute similarities
        similarities = F.cosine_similarity(query_embedding, self.triple_embeddings)
        
        # Get top-k results
        top_indices = torch.topk(similarities, min(top_k, len(self.kg.triples))).indices
        
        results = []
        for idx in top_indices:
            triple = self.kg.triples[idx]
            similarity = similarities[idx].item()
            
            # Check if any entities from query match triple entities
            entity_match = any(
                entity in triple["subject"].lower() or 
                entity in triple["object"].lower() 
                for entity in entities
            )
            
            results.append({
                "triple": triple,
                "similarity": similarity,
                "entity_match": entity_match,
                "score": similarity * (1.2 if entity_match else 1.0)
            })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get_facts_for_question(self, question: str, choices: List[str], 
                              top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Get relevant facts for a question and its choices."""
        facts = {}
        
        # Get facts for question
        question_facts = self.retrieve(question, top_k)
        facts["question"] = question_facts
        
        # Get facts for each choice
        for i, choice in enumerate(choices):
            choice_facts = self.retrieve(choice, top_k)
            facts[f"choice_{i}"] = choice_facts
        
        return facts 