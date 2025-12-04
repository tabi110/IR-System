import csv
import sys
import os
import time
import re
from collections import defaultdict

# --- Specified Libraries ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Use try/except to handle the missing external dependency gracefully
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    # If the user hasn't installed rank-bm25, set BM25Okapi to None 
    # and provide a clear message.
    print("---------------------------------------------------------------", file=sys.stderr)
    print("WARNING: 'rank-bm25' library is missing.", file=sys.stderr)
    print("The BM25 search option (Menu 2) will be disabled.", file=sys.stderr)
    print("To enable BM25 search, please run: pip install rank-bm25", file=sys.stderr)
    print("---------------------------------------------------------------", file=sys.stderr)
    BM25Okapi = None

# --- Custom Preprocessing Definitions ---

# Hardcoded Stop Word List (Standard English)
# This list is passed directly to TfidfVectorizer's stop_words parameter
STOP_WORDS = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', 
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', 
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', 
    "wouldn't"
]

def custom_tokenizer(text):
    """
    Tokenizes the text by finding sequences of letters and applying lowercasing.
    Stop word filtering is handled by the TfidfVectorizer's stop_words parameter.
    """
    # Lowercasing and tokenization
    return re.findall(r'[a-z]+', text.lower())

def clean_text(text):
    """Removes HTML tags and non-alphanumeric characters from text."""
    # Remove HTML tags (including attributes like style="...")
    clean = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(clean, '', text)
    # Remove extra whitespace and trim
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class IRSystem:
    """
    Implements a menu-driven Information Retrieval system supporting both 
    Vector Space Model (VSM) and BM25 ranking.
    """
    def __init__(self):
        self.documents = {}    # {doc_id: {'Heading': str, 'Article': str}}
        self.raw_documents = [] # List of concatenated, cleaned document strings
        self.N = 0             # Total number of documents (corpus size)
        
        # VSM (TF-IDF) components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # BM25 components
        self.bm25 = None

    def load_and_index_data(self, filepath):
        """Loads articles from CSV file and builds both VSM and BM25 indices."""
        if not os.path.exists(filepath):
            print(f"\n[!] Error: File '{filepath}' not found.")
            return
        
        try:
            self.documents = {}
            self.raw_documents = []
            
            # Try multiple encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            content = None
            
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    print(f"[*] Successfully read file with {encoding} encoding")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if content is None:
                print(f"\n[!] Error: Could not read file with any supported encoding.")
                return
            
            # Parse CSV from the content
            import io
            reader = csv.DictReader(io.StringIO(content))
            doc_id = 1
            
            for row in reader:
                heading = row.get('Heading', '')
                article = row.get('Article', '')
                
                # Clean the text
                heading = clean_text(heading)
                article = clean_text(article)
                
                self.documents[doc_id] = {
                    'Heading': heading,
                    'Article': article
                }
                
                # Concatenate heading and article for indexing
                combined = f"{heading} {article}"
                self.raw_documents.append(combined)
                doc_id += 1
            
            self.N = len(self.documents)
            
            if self.N == 0:
                print("\n[!] No documents loaded from CSV file.")
                return
            
            # Build TF-IDF index
            self.tfidf_vectorizer = TfidfVectorizer(
                tokenizer=custom_tokenizer,
                stop_words=STOP_WORDS,
                lowercase=True,
                min_df=1,
                max_df=0.95
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.raw_documents)
            
            # Build BM25 index
            if BM25Okapi is not None:
                tokenized_docs = [custom_tokenizer(doc) for doc in self.raw_documents]
                self.bm25 = BM25Okapi(tokenized_docs)
            
            print(f"\n[✓] Data indexed successfully!")
            print(f"    Total documents: {self.N}")
            print(f"    VSM (TF-IDF) index: Built")
            print(f"    BM25 index: {'Built' if self.bm25 else 'Disabled (rank-bm25 not installed)'}")
        
        except Exception as e:
            print(f"\n[!] Error loading data: {e}")

    def _rank_results(self, scores, top_k=10):
        """Helper method to rank and format results."""
        doc_ids = np.arange(1, self.N + 1)
        
        # Pair scores with doc_ids
        scored_documents = list(zip(doc_ids, scores))
        
        # Filter out zero scores and sort in descending order
        ranked_results = sorted(
            [item for item in scored_documents if item[1] > 0], 
            key=lambda item: item[1], 
            reverse=True
        )

        # Prepare final results for display
        final_results = []
        for rank, (doc_id, score) in enumerate(ranked_results[:top_k]):
            doc_data = self.documents.get(doc_id, {'Heading': 'N/A', 'Article': 'N/A'})
            
            # Use the first 200 characters of the article as a snippet
            snippet = doc_data['Article'].replace('\r\n', ' ')[:200] + '...'
            
            final_results.append({
                'rank': rank + 1,
                'score': score,
                'doc_id': doc_id,
                'heading': doc_data['Heading'],
                'snippet': snippet
            })
            
        return final_results

    def search_vsm(self, raw_query, top_k=10):
        """Performs search using the Vector Space Model (TF-IDF + Cosine Similarity)."""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return 0.0, [], "Error: VSM index is not built."
            
        query_start_time = time.time()
        
        # Transform the query into a TF-IDF vector (stop words are removed here by the vectorizer)
        query_vec = self.tfidf_vectorizer.transform([raw_query])
        
        if query_vec.nnz == 0:
            return 0.0, [], "Query contains no terms found in the corpus vocabulary."

        # Calculate cosine similarity between query vector and all document vectors
        cosine_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        query_time = time.time() - query_start_time
        
        final_results = self._rank_results(cosine_scores, top_k)
        
        return query_time, final_results, None

    def search_bm25(self, raw_query, top_k=10):
        """Performs search using the BM25Okapi ranking function."""
        if self.bm25 is None:
            # Check if BM25Okapi class was imported successfully
            if BM25Okapi is None:
                return 0.0, [], "Error: BM25 search disabled. 'rank-bm25' library is missing. Please install it."
            
            # If BM25Okapi exists but the index is not built
            return 0.0, [], "Error: BM25 index is not built."

        query_start_time = time.time()

        # 1. Preprocess the query manually (Tokenize and remove stop words)
        stop_words_set = set(STOP_WORDS)
        query_tokens = [
            token for token in custom_tokenizer(raw_query) if token not in stop_words_set
        ]
        
        if not query_tokens:
            return 0.0, [], "Query contains only stop words or punctuation after processing."
            
        # 2. Compute BM25 scores for all documents
        bm25_scores = self.bm25.get_scores(query_tokens)

        query_time = time.time() - query_start_time

        final_results = self._rank_results(bm25_scores, top_k)
        
        return query_time, final_results, None

# --- Main Application Logic ---

def display_menu(bm25_enabled):
    """Displays the main menu options, conditionally enabling BM25 search."""
    print("\n" + "="*70)
    print(" INFORMATION RETRIEVAL SYSTEM (BM25 & VSM Hybrid)")
    print("="*70)
    print("1. Load and Index Data (Articles.csv)")
    
    if bm25_enabled:
        print("2. Search Corpus (BM25 Ranking)")
    else:
        print("2. Search Corpus (BM25 Ranking) [DISABLED: Install rank-bm25]")
        
    print("3. Search Corpus (VSM / Cosine Similarity Ranking)")
    print("4. Quit")
    print("="*70)

def main():
    """The main function to run the menu-driven application."""
    ir_system = IRSystem()
    data_indexed = False
    data_filepath = "Articles.csv"
    bm25_is_available = BM25Okapi is not None

    while True:
        display_menu(bm25_is_available)
        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            # Option 1: Load and Index Data
            ir_system.load_and_index_data(data_filepath)
            if ir_system.N > 0:
                data_indexed = True
            else:
                data_indexed = False

        elif choice == '2':
            # Option 2: BM25 Search
            if not bm25_is_available:
                print("\n[!] BM25 search is disabled. Please install 'rank-bm25' to use this feature.")
                continue
                
            if not data_indexed:
                print("\n[!] Please load and index the data first (Option 1).")
                continue
                
            query = input("Enter your search query: ").strip()
            if not query:
                print("\n[!] Query cannot be empty.")
                continue

            try:
                top_k = int(input("Enter number of results to display (e.g., 10): ").strip() or 10)
            except ValueError:
                print("\n[!] Invalid number. Defaulting to 10 results.")
                top_k = 10
            
            if top_k <= 0:
                print("\n[!] Number of results must be positive. Defaulting to 10.")
                top_k = 10
            
            ranking_model = "BM25"
            search_func = ir_system.search_bm25

            query_time, results, error = search_func(query, top_k)
            
            if error:
                print(f"\n[!] Search Error ({ranking_model}): {error}")
            elif not results:
                print(f"\n[!] No results found for query: '{query}' using {ranking_model}.")
            else:
                print(f"\n--- Search Results for '{query}' (Ranking Model: {ranking_model}) ---")
                print(f"Query Time: {query_time:.4f} seconds | Found {len(results)} results.")
                print("-" * 70)
                
                for res in results:
                    print(f"RANK: {res['rank']:<3} | SCORE: {res['score']:.4f} | DOC ID: {res['doc_id']}")
                    print(f"HEADING: {res['heading']}")
                    print(f"SNIPPET: {res['snippet']}")
                    print("-" * 70)
        
        elif choice == '3':
            # Option 3: VSM Search
            if not data_indexed:
                print("\n[!] Please load and index the data first (Option 1).")
                continue
                
            query = input("Enter your search query: ").strip()
            if not query:
                print("\n[!] Query cannot be empty.")
                continue

            try:
                top_k = int(input("Enter number of results to display (e.g., 10): ").strip() or 10)
            except ValueError:
                print("\n[!] Invalid number. Defaulting to 10 results.")
                top_k = 10
            
            if top_k <= 0:
                print("\n[!] Number of results must be positive. Defaulting to 10.")
                top_k = 10
            
            ranking_model = "VSM (TF-IDF)"
            search_func = ir_system.search_vsm

            query_time, results, error = search_func(query, top_k)
            
            if error:
                print(f"\n[!] Search Error ({ranking_model}): {error}")
            elif not results:
                print(f"\n[!] No results found for query: '{query}' using {ranking_model}.")
            else:
                print(f"\n--- Search Results for '{query}' (Ranking Model: {ranking_model}) ---")
                print(f"Query Time: {query_time:.4f} seconds | Found {len(results)} results.")
                print("-" * 70)
                
                for res in results:
                    print(f"RANK: {res['rank']:<3} | SCORE: {res['score']:.4f} | DOC ID: {res['doc_id']}")
                    print(f"HEADING: {res['heading']}")
                    print(f"SNIPPET: {res['snippet']}")
                    print("-" * 70)

        elif choice == '4':
            # Option 4: Quit
            print("\nExiting Hybrid IR System. Goodbye!")
            sys.exit(0)

        else:
            print("\n[!] Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    # Ensure working directory is correctly handled if run from a different path
    # (Not strictly necessary in a controlled environment but good practice)
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting.")
        sys.exit(0)