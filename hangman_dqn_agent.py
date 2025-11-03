import tensorflow as tf
import numpy as np
from collections import deque
import random
import os

class HangmanDQNAgent:
    """
    Deep Q-Network agent for Hangman
    Uses HMM probabilities as part of state representation
    """
    
    def __init__(self, state_size=100, action_size=26, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """Build neural network for Q-learning"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def _encode_state(self, state, hmm):
        """
        Encode game state into feature vector
        
        Args:
            state: dict from environment
            hmm: HMM model for probability estimation
            
        Returns:
            numpy array of shape (state_size,)
        """
        features = []
        
        masked_word = state['masked_word']
        guessed_letters = state['guessed_letters']
        word_length = state['word_length']
        lives_left = state['lives_left']
        
        # 1. Word pattern features (30 features: position encoding)
        max_len = 30
        pattern = np.zeros(max_len)
        for i, char in enumerate(masked_word[:max_len]):
            if char != '_':
                # Encode as letter position in alphabet (1-26)
                pattern[i] = ord(char) - ord('a') + 1
        features.extend(pattern)
        
        # 2. Guessed letters binary vector (26 features)
        guessed_vector = np.zeros(26)
        for letter in guessed_letters:
            guessed_vector[ord(letter) - ord('a')] = 1
        features.extend(guessed_vector)
        
        # 3. Game state features (4 features)
        features.extend([
            word_length / 30.0,  # Normalized word length
            lives_left / 6.0,  # Normalized lives
            len(guessed_letters) / 26.0,  # Proportion of letters guessed
            masked_word.count('_') / word_length  # Proportion still masked
        ])
        
        # 4. HMM probability distribution (26 features)
        probs = hmm.get_letter_probabilities(masked_word, guessed_letters, lives_left)
        prob_vector = np.zeros(26)
        for letter, prob in probs.items():
            prob_vector[ord(letter) - ord('a')] = prob
        features.extend(prob_vector)
        
        # 5. Top HMM suggestions (14 features: top 7 letters + their probs)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:7]
        for i in range(7):
            if i < len(sorted_probs):
                letter, prob = sorted_probs[i]
                features.extend([ord(letter) - ord('a'), prob])
            else:
                features.extend([0, 0])
        
        # Pad or truncate to state_size
        features = np.array(features[:self.state_size])
        if len(features) < self.state_size:
            features = np.pad(features, (0, self.state_size - len(features)))
        
        return features
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state, hmm, epsilon=None):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: dict from environment
            hmm: HMM model
            epsilon: exploration rate (uses self.epsilon if None)
            
        Returns:
            str: letter to guess
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        valid_actions = state.get('valid_actions', 
                                  list(set('abcdefghijklmnopqrstuvwxyz') - state['guessed_letters']))
        
        if not valid_actions:
            return 'a'  # Fallback
        
        # Epsilon-greedy exploration
        if np.random.rand() <= epsilon:
            return random.choice(valid_actions)
        
        # Exploit: use Q-values
        state_encoded = self._encode_state(state, hmm)
        q_values = self.model.predict(state_encoded.reshape(1, -1), verbose=0)[0]
        
        # Mask invalid actions
        action_indices = [ord(letter) - ord('a') for letter in valid_actions]
        valid_q_values = [(idx, q_values[idx]) for idx in action_indices]
        
        # Choose action with highest Q-value
        best_action_idx = max(valid_q_values, key=lambda x: x[1])[0]
        return chr(best_action_idx + ord('a'))
    
    def replay(self, batch_size=32):
        """Train on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])
        
        # Predict Q-values for starting state
        target = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next state
        target_next = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
        
        # Train model
        self.model.fit(states, target, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save model weights"""
        # Ensure filepath ends with .weights.h5 for newer Keras versions
        if not filepath.endswith('.weights.h5'):
            if filepath.endswith('.h5'):
                filepath = filepath.replace('.h5', '.weights.h5')
            else:
                filepath += '.weights.h5'
        self.model.save_weights(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights"""
        # Handle both old and new Keras weight file extensions
        if not filepath.endswith('.weights.h5') and filepath.endswith('.h5'):
            # Try the new extension first
            new_filepath = filepath.replace('.h5', '.weights.h5')
            if os.path.exists(new_filepath):
                filepath = new_filepath
        self.model.load_weights(filepath)
        self.update_target_model()
        print(f"Model loaded from {filepath}")


# Alternative: Simpler Q-Learning with HMM
class HangmanHybridAgent:
    """
    Hybrid agent that primarily uses HMM with simple heuristics
    Much faster to train and often performs better for Hangman
    """
    
    def __init__(self):
        self.letter_freq = {
            'e': 0.127, 't': 0.091, 'a': 0.082, 'o': 0.075, 'i': 0.070,
            'n': 0.067, 's': 0.063, 'h': 0.061, 'r': 0.060, 'd': 0.043,
            'l': 0.040, 'c': 0.028, 'u': 0.028, 'm': 0.024, 'w': 0.024,
            'f': 0.022, 'g': 0.020, 'y': 0.020, 'p': 0.019, 'b': 0.015,
            'v': 0.010, 'k': 0.008, 'j': 0.002, 'x': 0.002, 'q': 0.001, 'z': 0.001
        }
    
    def choose_action(self, state, hmm, epsilon=0.0):
        """
        Choose action using HMM probabilities with strategic heuristics
        """
        masked_word = state['masked_word']
        guessed_letters = state['guessed_letters']
        lives_left = state['lives_left']
        
        valid_actions = list(set('abcdefghijklmnopqrstuvwxyz') - guessed_letters)
        
        if not valid_actions:
            return 'a'
        
        # Get HMM probabilities
        hmm_probs = hmm.get_letter_probabilities(masked_word, guessed_letters, lives_left)
        
        # Combine HMM with base frequency
        scores = {}
        for letter in valid_actions:
            hmm_score = hmm_probs.get(letter, 0.0001)
            freq_score = self.letter_freq.get(letter, 0.001)
            
            # Weight heavily towards HMM when we have context
            blanks_ratio = masked_word.count('_') / len(masked_word)
            if blanks_ratio < 0.5:  # More than half revealed
                scores[letter] = hmm_score * 0.9 + freq_score * 0.1
            else:
                scores[letter] = hmm_score * 0.7 + freq_score * 0.3
        
        # Epsilon-greedy
        if random.random() < epsilon:
            return random.choice(valid_actions)
        
        return max(scores.items(), key=lambda x: x[1])[0]