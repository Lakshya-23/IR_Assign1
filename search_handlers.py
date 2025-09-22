import math
from collections import defaultdict, Counter
from soundex import generate_soundex

def handle_vsm_query(query_terms, index, doc_lengths, total_docs, soundex_map):
    #Handles  VSM search with cosine similarity and Soundex.
    if not query_terms: return []

    query_tf_counts = Counter(query_terms)
    query_vector = defaultdict(float)
    query_sum_of_squares = 0

    for term, tf_raw in query_tf_counts.items():
        tf_weight = 1 + math.log10(tf_raw)
        search_terms = []
        if term in index:
            search_terms.append(term)
        else:
            soundex_code = generate_soundex(term)
            expanded_terms = soundex_map.get(soundex_code, [])
            if expanded_terms:
                print(f"'{term}' not found. Using Soundex matches: {list(expanded_terms)}")
                search_terms.extend(list(expanded_terms))

        for st in search_terms:
            if st in index:
                df = index[st]['df']
                idf = math.log10(total_docs / df) if df > 0 else 0
                weight = tf_weight * idf
                if st not in query_vector:
                    query_vector[st] = weight
                    query_sum_of_squares += weight**2

    query_length = math.sqrt(query_sum_of_squares) if query_sum_of_squares > 0 else 1.0
    scores = defaultdict(float)

    for term, query_weight in query_vector.items():
        if term in index:
            for doc_id, positions in index[term]['postings'].items():
                doc_term_weight = 1 + math.log10(len(positions))
                scores[doc_id] += query_weight * doc_term_weight

    final_scores = {
        doc_id: dot_product / (doc_lengths.get(doc_id, 1.0) * query_length)
        for doc_id, dot_product in scores.items() if doc_lengths.get(doc_id, 0) > 0
    }
    
    ranked_results = sorted(final_scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked_results[:10]

def handle_phrase_query(phrase_terms, index):
    #Handles exact phrase search using positional intersection.
    if not phrase_terms: return []
    first_term = phrase_terms[0]
    if first_term not in index: return []
    
    results = {doc_id: pos_list for doc_id, pos_list in index[first_term]['postings'].items()}

    for i in range(1, len(phrase_terms)):
        current_term = phrase_terms[i]
        if current_term not in index: return []
        
        new_results = {}
        term_postings = index[current_term]['postings']
        
        for doc_id, positions in results.items():
            if doc_id in term_postings:
                new_positions = [p + 1 for p in positions if (p + 1) in term_postings[doc_id]]
                if new_positions:
                    new_results[doc_id] = new_positions
        
        results = new_results
        if not results: return []

    return sorted(list(results.keys()))

def handle_proximity_query(term1, term2, k, index):
    #Handles proximity search using a two-pointer scan.
    if term1 not in index or term2 not in index: return []

    postings1, postings2 = index[term1]['postings'], index[term2]['postings']
    common_docs = set(postings1.keys()) & set(postings2.keys())
    matching_docs = []

    for doc_id in common_docs:
        pos_list1, pos_list2 = postings1[doc_id], postings2[doc_id]
        p1, p2 = 0, 0
        while p1 < len(pos_list1) and p2 < len(pos_list2):
            if abs(pos_list1[p1] - pos_list2[p2]) <= k:
                matching_docs.append(doc_id)
                break
            elif pos_list1[p1] < pos_list2[p2]: p1 += 1
            else: p2 += 1
            
    return sorted(matching_docs)