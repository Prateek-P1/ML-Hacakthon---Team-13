# Intelligent Hangman Agent using a Hybrid Probabilistic and Learning Model

PES1UG23AM195,  	PES1UG23AM218, 	PES1UG23AM202, 	PES1UG23AM211

## 1. Project Overview

This project presents a hybrid intelligent agent designed to solve the game of Hangman, developed as part of the UE23CS352A Machine Learning Hackathon. The primary objective is to maximize the game success rate while minimizing incorrect guesses.

The solution integrates a statistical model inspired by Hidden Markov Models (HMMs), an explicit word filter, and a trainable agent that employs a multi-tiered heuristic strategy. This hybrid approach allows the agent to make context-aware, data-driven decisions by combining pre-computed linguistic patterns with a simple learning mechanism.

The agent is trained exclusively on the provided `corpus.txt` and evaluated on an unseen `test.txt` vocabulary, simulating a real-world machine learning challenge that requires generalization to new data.

## 2. Architectural Overview

The agent's architecture is a multi-component system where each part provides a different layer of intelligence.

### 2.1. Statistical Model (HMM)

The foundation of the agent's reasoning is a statistical model trained on the 50,000-word corpus. This model learns the underlying patterns of the English language and includes:

* **Positional Probabilities:** The probability of a letter appearing at a specific position for a given word length (e.g., 's' is more common at the end of words).
* **Bigram Probabilities:** The probability of one letter following another (e.g., `P('u'|'q')`), which provides crucial contextual clues when some letters are revealed.
* **Overall Letter Frequencies:** A general fallback measure of letter commonness, calculated for each word length.

The training process is a rapid, one-pass analysis of the corpus, using Laplace smoothing to handle unseen letters and prevent zero-probability errors.

### 2.2. Word Filter

This component is responsible for maintaining a list of valid "candidate words" from the training corpus that are still possible given the current game state. It performs strict filtering based on:

* Matching revealed letters at their exact positions.
* Ensuring the word does not contain any letters that have been confirmed as incorrect guesses.

### 2.3. Trainable Hybrid Agent

The agent's "brain" uses a multi-tiered heuristic strategy to make the final decision. It also incorporates a simple learning mechanism to improve performance over time.

* **State Representation:** `{masked_word, guessed_letters, wrong_guesses, lives_remaining}`
* **Action Space:** The set of un-guessed letters (A-Z).
* **Hybrid Strategy:**
    * **Early Game:** With an empty board, the agent guesses from a pre-defined list of high-frequency letters (E, T, A, O, I, N) to maximize the chance of an early hit.
    * **Mid Game:** The agent combines three sources of information with a weighted average to score each possible guess:
        * Word Filter Frequencies (50% weight): Probabilities derived from the remaining candidate words.
        * HMM Probabilities (30% weight): Statistical scores from the HMM based on positional and bigram data.
        * General Candidate Frequencies (20% weight): Overall letter commonness within the current candidate pool.
    * **Late Game / Fallback:** If the candidate list shrinks to one, the agent solves the word directly. If the list is empty (because the target word is not in the corpus), it relies solely on the statistical HMM.
* **Learning Mechanism:** The agent employs a form of `case-based reasoning` or `pattern memorization`. During a 5,000-game training phase, it populates a `pattern_memory` dictionary. If it successfully guesses a letter for a specific pattern (e.g., for `_A_E`, guessing 'M' was a success), it stores that `(pattern, guess)` pair. In future games, if it encounters the exact same pattern, it will exploit this memory to reuse the successful guess.

## 3. Performance & Results

The agent was evaluated on a hidden test set of 2,000 words. The final performance metrics are as follows:

* **Success Rate:** 26.10% (522 wins / 2000 games)
* **Average Wrong Guesses per Game:** 5.35
* **Final Score:** -52,998.00

The score is heavily negative due to the high penalty for wrong guesses on the ~74% of games that were lost. The agent's performance is respectable given the challenge but falls short of the target success rate.

## 4. Key Insights

* **Corpus-Test Vocabulary Mismatch is the Primary Hurdle:** The final success rate is primarily due to the test set containing words not present in the training corpus. This frequently causes the Word Filter's candidate list to become empty, forcing the agent to rely on its less accurate statistical HMM fallback.
* **Performance Increases with Word Length:** The agent's success rate is significantly higher for longer words (e.g., >50% for words of 15+ letters). Longer words provide more revealed characters, giving the HMM stronger contextual clues to make accurate predictions.
* **Hybrid Heuristics are Powerful but Limited:** The multi-layered strategy with fixed weights (50/30/20) is effective but not adaptive. These weights may not be optimal for all game states.
* **Pattern Memorization is a Limited Form of Learning:** The agent's learning mechanism helps it quickly solve patterns seen during training but cannot generalize to new, unseen patterns, which constitute the majority of states in the test set.

## 5. File Structure

* `hackmantry26.ipynb`: The main Jupyter Notebook containing all code for the model, agent, training, and evaluation.
* `Data/corpus.txt`: The 50,000-word corpus used for training the statistical model.
* `Data/test.txt`: The 2,000-word hidden test set used for final evaluation.
* `improved_hmm.pkl`: The saved, trained statistical model object.
* `trainable_hybrid_agent.pkl`: The saved agent object, including its learned pattern memory.
* `final_evaluation_results.pkl`: A pickle file containing the dictionary of final performance metrics.

## 6. How to Run

1.  **Environment Setup:** Ensure you have Python 3 and the following libraries installed: `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`.
2.  **Directory Structure:** Place the `corpus.txt` and `test.txt` files inside a sub-directory named `Data/`.
3.  **Execution:** Open `hackmantry26.ipynb` and run all cells sequentially from top to bottom. The notebook will:
    * Load the data.
    * Train the statistical model (fast).
    * Train the hybrid agent's pattern memory (approx. 30-40 seconds for 5,000 episodes).
    * Run the final evaluation on the test set.
    * Display performance charts and detailed results.

## 7. Future Improvements

* **True Reinforcement Learning (DQN):** The current pattern memorization is too limited. The next step would be to implement a Deep Q-Network (DQN). The state vector for the network would include the probability distribution from our HMM, allowing it to learn a generalized policy that can handle unseen patterns.
* **Adaptive Weighting:** The fixed weights in the hybrid model could be learned. An RL agent could be trained to dynamically adjust these weights based on the game state.
* **Information-Gain Heuristic:** The agent could be modified to pick the letter that provides the most information gain—the one expected to reduce the pool of candidate words the most—leading to faster convergence.
* **Trie-Based Corpus:** For much faster word filtering, the corpus could be loaded into a Trie data structure, allowing for near-instantaneous retrieval of candidate words.