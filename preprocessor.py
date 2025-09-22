import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

class Preprocessor:
    #Handles all text preprocessing tasks using NLTK
    def __init__(self):
        self._setup_nltk()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

    def _setup_nltk(self):
        #Downloads necessary NLTK data.
        required_packages = [
            'punkt', 'punkt_tab', 'wordnet', 
            'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 
            'stopwords'
        ]
        for package in required_packages:
            try:
                path_prefix = 'tokenizers' if package.startswith('punkt') else 'taggers' if package.startswith('averaged_perceptron') else 'corpora'
                nltk.data.find(f'{path_prefix}/{package}')
            except LookupError:
                print(f"NLTK package '{package}' not found. Downloading...")
                nltk.download(package, quiet=True)
        print("NLTK setup complete.")

    def _get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'): return wordnet.ADJ
        elif treebank_tag.startswith('V'): return wordnet.VERB
        elif treebank_tag.startswith('N'): return wordnet.NOUN
        elif treebank_tag.startswith('R'): return wordnet.ADV
        else: return wordnet.NOUN

    def process(self, text, remove_stopwords=False):
     
        #Tokenizes, POS-tags, and lemmatizes text. Optionally removes stopwords.
        
        tokens = nltk.word_tokenize(text.lower())
        pos_tagged_tokens = nltk.pos_tag(tokens)
        
        lemmatized_tokens = []
        for word, tag in pos_tagged_tokens:
            if word.isalpha():
                if remove_stopwords and word in self.stopwords:
                    continue  
                wn_tag = self._get_wordnet_pos(tag)
                lemma = self.lemmatizer.lemmatize(word, pos=wn_tag)
                lemmatized_tokens.append(lemma)
        return lemmatized_tokens