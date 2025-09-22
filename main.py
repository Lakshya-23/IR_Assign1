import os
import re
import json
from collections import defaultdict


from preprocessor import Preprocessor
from indexer import Indexer
import search_handlers

class AdvancedSearchEngine:
    
    def __init__(self, corpus_path):
        self.preprocessor = Preprocessor()
        self.indexer = Indexer(self.preprocessor)
        self.index_file = self.indexer.index_file
        self.doc_lengths_file = self.indexer.doc_lengths_file
        
        self._load_or_build_index(corpus_path)

    def _load_or_build_index(self, corpus_path):
        #Loads index from file
        if os.path.exists(self.index_file) and os.path.exists(self.doc_lengths_file):
            print("Loading existing positional index from files...")
            with open(self.index_file, 'r') as f:
                index_data = json.load(f)
                self.index = defaultdict(lambda: {'df': 0, 'postings': defaultdict(list)}, index_data['inverted_index'])
                self.soundex_map = defaultdict(set, {code: set(terms) for code, terms in index_data['soundex_map'].items()})
                self.total_docs = index_data['total_docs']
            with open(self.doc_lengths_file, 'r') as f:
                self.doc_lengths = json.load(f)
            print("Index loaded successfully.")
        else:
            print("No index files found. Building positional index from corpus...")
            self.index, self.doc_lengths, self.soundex_map, self.total_docs = self.indexer.build(corpus_path)
            print("Index built successfully.")

    def search(self, query):
        #Detects query type 
        query = query.strip()
        proximity_match = re.match(r'"(.*?)"\s*w/(\d+)\s*"(.*?)"', query, re.IGNORECASE)
        phrase_match = re.match(r'"([^"]+)"', query)
        
        if proximity_match:
            term1_raw, k, term2_raw = proximity_match.groups()
            print(f"--- Find '{term1_raw}' within {k} words of '{term2_raw}' ---")
            # For proximity, we remove stopwords to get to the core terms
            term1_lemmas = self.preprocessor.process(term1_raw, remove_stopwords=True)
            term2_lemmas = self.preprocessor.process(term2_raw, remove_stopwords=True)
            if not term1_lemmas or not term2_lemmas: return []
            return search_handlers.handle_proximity_query(term1_lemmas[0], term2_lemmas[0], int(k), self.index)

        elif phrase_match:
            phrase = phrase_match.groups()[0]
            print(f"--- Find exact phrase '{phrase}' ---")
            phrase_terms = self.preprocessor.process(phrase, remove_stopwords=True)
            return search_handlers.handle_phrase_query(phrase_terms, self.index)

        else:
            print("--- VSM Cosine Similarity Search ---")
            
            query_terms = self.preprocessor.process(query, remove_stopwords=True)
            return search_handlers.handle_vsm_query(query_terms, self.index, self.doc_lengths, self.total_docs, self.soundex_map)

def run_assignment_test_cases(engine):
   
    print("\n" + "-"*50)
    print("RUNNING ASSIGNMENT TEST CASES")
    print("-"*50)

    test_queries = [
        "Developing your Zomato business account and profile is a great way to boost your restaurant's online reputation",
        'Warwickshire, came from an ancient family and was the heiress to some land'
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n[Test Case Q{i}] Query: '{q}'")
        results = engine.search(q)
        print_results(results)

def start_interactive_mode(engine):
    #Interactive CLI for search engine
    print("\n" + "-"*50)
    print("INTERACTIVE CLI")
    print("For proximity search, use the format: \"term1\" w/k \"term2\"")
    print("Type 'exit' to quit.")
    print("-"*50)

    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
        
        results = engine.search(query)
        print_results(results)

def print_results(results):
   
    if not results:
        print("No matching documents found.")
    elif isinstance(results[0], tuple):
        print("\nResults (Ranked by Cosine Similarity):")
        for rank, (doc_id, score) in enumerate(results, 1):
            print(f"{rank}. Document: {doc_id}, Score: {score:.6f}")
    else:
        print("\nDocuments Matching Query:")
        for doc_id in results:
            print(f"- {doc_id}")

if __name__ == '__main__':
    CORPUS_DIR = 'corpus'
    
    if not os.path.exists(CORPUS_DIR):
        print(f"Error: The corpus folder '{CORPUS_DIR}' does not exist. Please create it.")
    else:
        # if os.path.exists('positional_index.json'):
        #     os.remove('positional_index.json')
        # if os.path.exists('doc_lengths.json'):
        #     os.remove('doc_lengths.json')
            
        search_engine = AdvancedSearchEngine(CORPUS_DIR)
        run_assignment_test_cases(search_engine)
        start_interactive_mode(search_engine)