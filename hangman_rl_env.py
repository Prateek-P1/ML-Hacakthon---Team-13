import numpy as np
import random
from collections import defaultdict

class HangmanEnvironment:
    """
    Hangman game environment for RL training
    """
    
    def __init__(self, word_list, max_wrong_guesses=6):
        self.word_list = [w.strip().lower() for w in word_list]
        self.max_wrong_guesses = max_wrong_guesses
        self.reset()
        
    def reset(self, word=None):
        """Start a new game"""
        if word is None:
            self.target_word = random.choice(self.word_list)
        else:
            self.target_word = word.lower()
            
        self.target_set = set(self.target_word)
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        self.correct_guesses = 0
        self.done = False
        
        return self._get_state()
    
    def _get_state(self):
        """Get current game state"""
        masked_word = ''.join([c if c in self.guessed_letters else '_' 
                               for c in self.target_word])
        
        return {
            'masked_word': masked_word,
            'guessed_letters': self.guessed_letters.copy(),
            'wrong_guesses': self.wrong_guesses,
            'lives_left': self.max_wrong_guesses - self.wrong_guesses,
            'word_length': len(self.target_word),
            'done': self.done,
            'target_word': self.target_word  # For evaluation only
        }
    
    def step(self, letter):
        """
        Make a guess
        
        Args:
            letter: str, single letter to guess
            
        Returns:
            state: current state dict
            reward: float
            done: bool
            info: dict with additional info
        """
        letter = letter.lower()
        
        # Check for repeated guess
        is_repeated = letter in self.guessed_letters
        
        if is_repeated:
            self.repeated_guesses += 1
            reward = -2  # Penalty for repeated guess
            info = {
                'is_correct': False,
                'is_repeated': True,
                'won': False,
                'lost': False
            }
            return self._get_state(), reward, self.done, info
        
        # Add to guessed letters
        self.guessed_letters.add(letter)
        
        # Check if correct
        is_correct = letter in self.target_set
        
        if is_correct:
            self.correct_guesses += 1
            # Count how many positions this letter appears in
            letter_count = self.target_word.count(letter)
            reward = 1.0 * letter_count  # Reward proportional to letters revealed
            
            # Check if won
            if self.target_set.issubset(self.guessed_letters):
                self.done = True
                # Bonus for winning with lives remaining
                reward += 10 + (self.max_wrong_guesses - self.wrong_guesses) * 2
                info = {
                    'is_correct': True,
                    'is_repeated': False,
                    'won': True,
                    'lost': False
                }
                return self._get_state(), reward, self.done, info
        else:
            self.wrong_guesses += 1
            reward = -5  # Penalty for wrong guess
            
            # Check if lost
            if self.wrong_guesses >= self.max_wrong_guesses:
                self.done = True
                reward -= 10  # Additional penalty for losing
                info = {
                    'is_correct': False,
                    'is_repeated': False,
                    'won': False,
                    'lost': True
                }
                return self._get_state(), reward, self.done, info
        
        info = {
            'is_correct': is_correct,
            'is_repeated': False,
            'won': False,
            'lost': False
        }
        
        return self._get_state(), reward, self.done, info
    
    def get_valid_actions(self):
        """Get list of valid letters to guess"""
        all_letters = set('abcdefghijklmnopqrstuvwxyz')
        return list(all_letters - self.guessed_letters)
    
    def get_stats(self):
        """Get game statistics"""
        return {
            'won': self.done and self.wrong_guesses < self.max_wrong_guesses,
            'wrong_guesses': self.wrong_guesses,
            'repeated_guesses': self.repeated_guesses,
            'correct_guesses': self.correct_guesses,
            'total_guesses': len(self.guessed_letters)
        }


def evaluate_agent(agent, test_words, hmm, num_games=None, verbose=False):
    """
    Evaluate agent performance
    
    Args:
        agent: HangmanAgent instance
        test_words: list of test words
        hmm: trained HMM model
        num_games: number of games to play (None = all words)
        verbose: print progress
        
    Returns:
        dict with evaluation metrics
    """
    if num_games is None:
        test_set = test_words
    else:
        test_set = random.sample(test_words, min(num_games, len(test_words)))
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for i, word in enumerate(test_set):
        env = HangmanEnvironment([word])
        state = env.reset(word)
        
        while not state['done']:
            action = agent.choose_action(state, hmm, epsilon=0.0)  # Greedy
            state, _, _, _ = env.step(action)
        
        stats = env.get_stats()
        if stats['won']:
            wins += 1
        total_wrong += stats['wrong_guesses']
        total_repeated += stats['repeated_guesses']
        
        if verbose and (i + 1) % 100 == 0:
            print(f"Evaluated {i + 1}/{len(test_set)} games...")
    
    success_rate = wins / len(test_set)
    avg_wrong = total_wrong / len(test_set)
    avg_repeated = total_repeated / len(test_set)
    
    # Calculate final score using the given formula
    final_score = (success_rate * len(test_set)) - (total_wrong * 5) - (total_repeated * 2)
    
    return {
        'num_games': len(test_set),
        'wins': wins,
        'success_rate': success_rate,
        'total_wrong_guesses': total_wrong,
        'avg_wrong_guesses': avg_wrong,
        'total_repeated_guesses': total_repeated,
        'avg_repeated_guesses': avg_repeated,
        'final_score': final_score
    }


# Test environment
if __name__ == "__main__":
    test_words = ['apple', 'banana', 'cherry']
    env = HangmanEnvironment(test_words)
    
    # Test game
    state = env.reset('apple')
    print(f"Target word: {state['target_word']}")
    print(f"Initial state: {state['masked_word']}")
    
    for letter in ['e', 'a', 'x', 'p', 'l']:
        state, reward, done, info = env.step(letter)
        print(f"\nGuessed '{letter}': {state['masked_word']}")
        print(f"  Reward: {reward}, Done: {done}, Info: {info}")
        
        if done:
            break
    
    print(f"\nFinal stats: {env.get_stats()}")