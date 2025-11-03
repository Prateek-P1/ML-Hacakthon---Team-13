import numpy as np
import pickle
from collections import defaultdict, Counter
import re

class HangmanHMM:
    """
    Hidden Markov Model for Hangman
    States: Letter positions (character at each position)
    Emissions: Observed patterns (masked/unmasked letters)
    """
    
    def __init__(self, max_word_length=25):
        self.max_word_length = max_word_length
        # Letter frequencies by position for different word lengths
        self.position_freq = defaultdict(lambda: defaultdict(Counter))
        # Overall letter frequencies
        self.letter_freq = Counter()
        # Bigram frequencies
        self.bigram_freq = defaultdict(Counter)
        # Pattern matching database
        self.word_patterns = defaultdict(list)
        
    def train(self, corpus_file):
        """Train HMM on corpus"""
        print("Training HMM...")
        words = []
        
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        print(f"Loaded {len(words)} words")
        
        # Build frequency tables
        for word in words:
            word_len = len(word)
            if word_len > self.max_word_length:
                continue
                
            # Position-specific frequencies
            for pos, char in enumerate(word):
                if char.isalpha():
                    self.position_freq[word_len][pos][char] += 1
                    self.letter_freq[char] += 1
            
            # Bigram frequencies
            for i in range(len(word) - 1):
                if word[i].isalpha() and word[i+1].isalpha():
                    self.bigram_freq[word[i]][word[i+1]] += 1
            
            # Store word patterns for quick matching
            pattern = self._word_to_pattern(word)
            self.word_patterns[pattern].append(word)
        
        print("HMM training complete!")
        
    def _word_to_pattern(self, word):
        """Convert word to pattern (e.g., 'hello' -> '5,h-e-l-l-o')"""
        return f"{len(word)},{'-'.join(word)}"
    
    def get_letter_probabilities(self, masked_word, guessed_letters, lives_left):
        """
        Get probability distribution over remaining letters
        
        Args:
            masked_word: str, e.g., "_PPL_"
            guessed_letters: set of already guessed letters
            lives_left: int, remaining lives
            
        Returns:
            dict: {letter: probability}
        """
        word_len = len(masked_word)
        remaining_letters = set('abcdefghijklmnopqrstuvwxyz') - guessed_letters
        
        if not remaining_letters:
            return {}
        
        # Strategy 1: Pattern matching with known words
        matching_words = self._find_matching_words(masked_word, guessed_letters)
        
        if matching_words:
            letter_counts = Counter()
            for word in matching_words:
                for i, char in enumerate(word):
                    if masked_word[i] == '_' and char in remaining_letters:
                        letter_counts[char] += 1
            
            # Normalize to probabilities
            total = sum(letter_counts.values())
            if total > 0:
                probs = {letter: count / total for letter, count in letter_counts.items()}
                # Fill in missing letters with small probability
                for letter in remaining_letters:
                    if letter not in probs:
                        probs[letter] = 0.0001
                return probs
        
        # Strategy 2: Position-based frequency (fallback)
        letter_scores = defaultdict(float)
        
        for i, char in enumerate(masked_word):
            if char == '_':
                pos_freq = self.position_freq[word_len][i]
                if pos_freq:
                    for letter in remaining_letters:
                        letter_scores[letter] += pos_freq.get(letter, 1)
        
        # Strategy 3: Consider bigrams for context
        for i, char in enumerate(masked_word):
            if char != '_':
                # Look at adjacent positions
                if i > 0 and masked_word[i-1] == '_':
                    for letter in remaining_letters:
                        letter_scores[letter] += self.bigram_freq[letter].get(char, 1) * 0.3
                if i < len(masked_word) - 1 and masked_word[i+1] == '_':
                    for letter in remaining_letters:
                        letter_scores[letter] += self.bigram_freq[char].get(letter, 1) * 0.3
        
        # Add baseline frequency
        for letter in remaining_letters:
            letter_scores[letter] += self.letter_freq.get(letter, 1) * 0.1
        
        # Normalize
        total_score = sum(letter_scores.values())
        if total_score > 0:
            probs = {letter: score / total_score for letter, score in letter_scores.items()}
        else:
            # Uniform distribution
            probs = {letter: 1.0 / len(remaining_letters) for letter in remaining_letters}
        
        return probs
    
    def _find_matching_words(self, masked_word, guessed_letters):
        """Find words matching the current pattern"""
        word_len = len(masked_word)
        
        # Create regex pattern
        pattern = ''
        for char in masked_word:
            if char == '_':
                # Match any letter that hasn't been guessed
                # Build a character class manually to avoid regex escaping issues
                excluded_chars = set(guessed_letters)
                allowed_chars = set('abcdefghijklmnopqrstuvwxyz') - excluded_chars
                
                if allowed_chars:
                    # Create character class with allowed letters only
                    allowed_str = ''.join(sorted(allowed_chars))
                    pattern += f'[{allowed_str}]'
                else:
                    # No allowed characters, match nothing
                    pattern += '(?!.)'  # Negative lookahead that never matches
            else:
                # Match the exact known letter (escaped for safety)
                pattern += re.escape(char)
        
        try:
            regex = re.compile(pattern)
        except re.error:
            # If regex fails, return empty list
            return []
        
        # Search through words of same length
        matching = []
        for pattern_key, words in self.word_patterns.items():
            length = int(pattern_key.split(',')[0])
            if length == word_len:
                for word in words:
                    if regex.match(word):
                        matching.append(word)
        
        return matching[:1000]  # Limit for performance
    
    def save(self, filepath):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'position_freq': dict(self.position_freq),
                'letter_freq': self.letter_freq,
                'bigram_freq': dict(self.bigram_freq),
                'word_patterns': dict(self.word_patterns),
                'max_word_length': self.max_word_length
            }, f)
        print(f"HMM saved to {filepath}")
    
    def load(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.position_freq = defaultdict(lambda: defaultdict(Counter), data['position_freq'])
            self.letter_freq = data['letter_freq']
            self.bigram_freq = defaultdict(Counter, data['bigram_freq'])
            self.word_patterns = defaultdict(list, data['word_patterns'])
            self.max_word_length = data['max_word_length']
        print(f"HMM loaded from {filepath}")


# Training script
if __name__ == "__main__":
    hmm = HangmanHMM(max_word_length=30)
    hmm.train('corpus.txt')
    hmm.save('hangman_hmm.pkl')
    
    # Test
    test_word = "_PPL_"
    guessed = set(['e', 's', 'r'])
    probs = hmm.get_letter_probabilities(test_word, guessed, 6)
    print(f"\nTest probabilities for '{test_word}' with guessed {guessed}:")
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]
    for letter, prob in sorted_probs:
        print(f"  {letter}: {prob:.4f}")