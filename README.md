# Vector Space Model Search (CSD358)

This project is an information retrieval system built for the Information Retrieval Assignment (CSD358). It implements a search engine to include features like phrase search, proximity search, and robust text processing with NLTK.

## Features

- **Multi-Mode Query Engine**: The system intelligently detects the user's intent based on the query syntax.
  - **VSM Keyword Search**: Standard ranked retrieval using the `lnc.ltc` tf-idf weighting scheme.
  - **Exact Phrase Search**: Finds documents containing an exact sequence of words (e.g., `"Apple Computer Inc"`).
  - **Proximity Search**: Finds documents where two terms appear within a specified distance of each other (e.g., `"apple" w/15 "jobs"`).
- **Advanced Text Processing**:
  - **NLTK Integration**: Uses the Natural Language Toolkit for high-quality text analysis.
  - **Lemmatization**: Employs Part-of-Speech (POS) aware lemmatization to accurately reduce words to their dictionary root form.
  - **Stopword Removal**: Uses NLTK's standard English stopword list for efficient indexing.
- **Positional Indexing**: Builds a full positional inverted index, storing the exact location of each term in every document to power phrase and proximity searches.
- **Soundex Fallback**: For VSM queries, if a term is not found in the vocabulary, the Soundex algorithm is used to find and search for phonetically similar words, making the engine resilient to spelling errors.

## System Architecture

The project is designed with a modular architecture to separate concerns, making it easy to understand, maintain, and extend.

- **`main.py`**: The main entry point and orchestrator. Contains the `AdvancedSearchEngine` class and the query dispatcher logic.
- **`preprocessor.py`**: Handles all NLP tasks, including tokenization, POS-tagging, lemmatization, and stopword management using NLTK.
- **`indexer.py`**: Responsible for reading the corpus, processing documents, and building the `positional_index.json` and `doc_lengths.json` files.
- **`search_handlers.py`**: Contains the core algorithmic logic for each of the three search modes (VSM, Phrase, and Proximity).
- **`soundex.py`**: A utility module for the Soundex phonetic algorithm.
- **`corpus/`**: The directory where all text documents for indexing should be placed.


### 2. Setup

1. clone the repository to your local machine:

git clone [your-repository-url]
cd [repository-folder-name]

2. install the required NLTK library:
pip install nltk

3. Populate the Corpus
Place all the .txt document files you want to index into the corpus/ directory.

5. Execute the Program
Run the main script from your terminal:

python main.py
On the first run:
The program will automatically download the necessary NLTK data packages (punkt, wordnet, stopwords, etc.). This requires an internet connection.
It will then build the positional index from the documents in the corpus/ directory and save it as positional_index.json and doc_lengths.json. This process might take a few moments depending on the size of your corpus.
On subsequent runs:
The program will load the existing index files for a much faster startup. (Note: The current main.py is configured to delete and rebuild the index on every run for consistent testing. This can be commented out for faster use.)
Query Syntax Examples
Once the program is running in interactive mode, you can use the following syntax:

VSM Search (default):
  developing zomato business reputation
Phrase Search (use double quotes):
  "The company was founded by Steven Paul Jobs"
Proximity Search (use w/k syntax):
  "company" w/3 "founded"
To exit the interactive mode, simply type exit.
