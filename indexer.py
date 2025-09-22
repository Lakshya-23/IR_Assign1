# indexer.py

import os
import json
import math
from collections import defaultdict
from soundex import generate_soundex

class Indexer:
    #Builds the positional index and document lengths from a corpus.
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.index_file = 'positional_index.json'
        self.doc_lengths_file = 'doc_lengths.json'

    def build(self, corpus_path):
        
        #Reads documents, processes them, and builds the positional index.
        #Saves the index and document lengths to disk.
      
        doc_files = sorted([f for f in os.listdir(corpus_path) if f.endswith('.txt')])
        total_docs = len(doc_files)
        inverted_index = defaultdict(lambda: {'df': 0, 'postings': defaultdict(list)})
        soundex_map = defaultdict(set)

        for doc_id in doc_files:
            file_path = os.path.join(corpus_path, doc_id)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
 
            # The indexerremoves stopwords from documents.
            processed_tokens = self.preprocessor.process(content, remove_stopwords=True)
            
            unique_terms_in_doc = set()
            for i, term in enumerate(processed_tokens):
                inverted_index[term]['postings'][doc_id].append(i)
                soundex_map[generate_soundex(term)].add(term)
                unique_terms_in_doc.add(term)

            for term in unique_terms_in_doc:
                inverted_index[term]['df'] += 1
        
        doc_lengths = {}
        for doc_id in doc_files:
            sum_of_squares = 0
            for term, data in inverted_index.items():
                if doc_id in data['postings']:
                    tf_raw = len(data['postings'][doc_id])
                    weight = 1 + math.log10(tf_raw)
                    sum_of_squares += weight ** 2
            doc_lengths[doc_id] = math.sqrt(sum_of_squares)
        
        self._save_files(inverted_index, doc_lengths, soundex_map, total_docs)
        
        return inverted_index, doc_lengths, soundex_map, total_docs

    def _save_files(self, index, doc_lengths, soundex_map, total_docs):
        """Serializes and saves the created data structures to JSON files."""
        serializable_soundex = {code: list(terms) for code, terms in soundex_map.items()}
        index_data = {
            'inverted_index': index,
            'soundex_map': serializable_soundex,
            'total_docs': total_docs
        }
        with open(self.index_file, 'w') as f:
            json.dump(index_data, f, indent=4)
        with open(self.doc_lengths_file, 'w') as f:
            json.dump(doc_lengths, f, indent=4)
        print(f"Successfully generated '{self.index_file}' and '{self.doc_lengths_file}'.")