"""
Complete training pipeline for Hangman AI
Combines HMM + RL for optimal performance

USAGE:
1. Train HMM: python train_hmm.py
2. Train RL: python train_rl.py  
3. Evaluate: python evaluate.py
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time

# Import our modules (ensure they're in the same directory)
# from hangman_hmm import HangmanHMM
# from hangman_env import HangmanEnvironment, evaluate_agent
# from hangman_agent import HangmanDQNAgent, HangmanHybridAgent


def train_hmm(corpus_file='corpus.txt', save_path='hangman_hmm.pkl'):
    """Step 1: Train HMM"""
    print("="*60)
    print("STEP 1: Training Hidden Markov Model")
    print("="*60)
    
    from hangman_hmm import HangmanHMM
    
    hmm = HangmanHMM(max_word_length=30)
    hmm.train(corpus_file)
    hmm.save(save_path)
    
    return hmm


def train_dqn_agent(hmm, corpus_file='corpus.txt', episodes=5000, 
                    save_path='dqn_agent.h5'):
    """Step 2a: Train DQN Agent (Deep RL)"""
    print("\n" + "="*60)
    print("STEP 2: Training DQN Agent")
    print("="*60)
    
    from hangman_rl_env import HangmanEnvironment
    from hangman_dqn_agent import HangmanDQNAgent
    
    # Load words
    with open(corpus_file, 'r') as f:
        words = [line.strip().lower() for line in f if line.strip()]
    
    # Initialize
    agent = HangmanDQNAgent(state_size=100, action_size=26, learning_rate=0.001)
    env = HangmanEnvironment(words)
    
    # Training metrics
    episode_rewards = []
    episode_wins = []
    success_rates = []
    
    print(f"Training for {episodes} episodes...")
    
    for episode in tqdm(range(episodes)):
        state = env.reset()
        total_reward = 0
        
        while not state['done']:
            # Choose action
            action = agent.choose_action(state, hmm)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Encode states
            state_encoded = agent._encode_state(state, hmm)
            next_state_encoded = agent._encode_state(next_state, hmm)
            action_idx = ord(action) - ord('a')
            
            # Remember
            agent.remember(state_encoded, action_idx, reward, next_state_encoded, done)
            
            total_reward += reward
            state = next_state
        
        # Train agent
        if len(agent.memory) > 32:
            agent.replay(batch_size=32)
        
        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_model()
        
        # Track metrics
        episode_rewards.append(total_reward)
        stats = env.get_stats()
        episode_wins.append(1 if stats['won'] else 0)
        
        # Calculate running success rate
        if episode >= 100:
            success_rates.append(np.mean(episode_wins[-100:]))
        
        # Progress report
        if (episode + 1) % 500 == 0:
            recent_sr = np.mean(episode_wins[-100:]) if len(episode_wins) >= 100 else 0
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"  Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
            print(f"  Success Rate (last 100): {recent_sr:.2%}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
    
    # Save model
    agent.save(save_path)
    
    # Plot training progress
    plot_training_progress(episode_rewards, episode_wins, success_rates, 
                          title="DQN Training Progress")
    
    return agent


def train_hybrid_agent(hmm, corpus_file='corpus.txt', test_file='test.txt'):
    """Step 2b: Use Hybrid Agent (HMM-based, no training needed)"""
    print("\n" + "="*60)
    print("STEP 2: Using Hybrid HMM-Based Agent (No training required)")
    print("="*60)
    
    from hangman_dqn_agent import HangmanHybridAgent
    from hangman_rl_env import evaluate_agent
    
    agent = HangmanHybridAgent()
    
    # Quick evaluation on corpus subset
    with open(corpus_file, 'r') as f:
        corpus_words = [line.strip().lower() for line in f if line.strip()]
    
    print("\nQuick validation on 500 corpus words...")
    results = evaluate_agent(agent, corpus_words[:500], hmm, verbose=True)
    
    print(f"\nValidation Results:")
    print(f"  Success Rate: {results['success_rate']:.2%}")
    print(f"  Avg Wrong Guesses: {results['avg_wrong_guesses']:.2f}")
    print(f"  Avg Repeated Guesses: {results['avg_repeated_guesses']:.2f}")
    
    return agent


def evaluate_final(agent, hmm, test_file='test.txt', num_games=2000):
    """Step 3: Final Evaluation"""
    print("\n" + "="*60)
    print("STEP 3: Final Evaluation")
    print("="*60)
    
    from hangman_rl_env import evaluate_agent
    
    # Load test words
    with open(test_file, 'r') as f:
        test_words = [line.strip().lower() for line in f if line.strip()]
    
    print(f"Evaluating on {num_games} games from test set...")
    print(f"Total test words available: {len(test_words)}")
    
    results = evaluate_agent(agent, test_words, hmm, num_games=num_games, verbose=True)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Games Played: {results['num_games']}")
    print(f"Wins: {results['wins']}")
    print(f"Success Rate: {results['success_rate']:.4f} ({results['success_rate']*100:.2f}%)")
    print(f"Total Wrong Guesses: {results['total_wrong_guesses']}")
    print(f"Avg Wrong Guesses: {results['avg_wrong_guesses']:.2f}")
    print(f"Total Repeated Guesses: {results['total_repeated_guesses']}")
    print(f"Avg Repeated Guesses: {results['avg_repeated_guesses']:.2f}")
    print(f"\nFINAL SCORE: {results['final_score']:.2f}")
    print("="*60)
    
    # Save results
    with open('evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results


def plot_training_progress(episode_rewards, episode_wins, success_rates, title="Training Progress"):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    axes[0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
    if len(episode_rewards) >= 100:
        smoothed = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        axes[0].plot(range(99, len(episode_rewards)), smoothed, 
                     label='Smoothed (100 episodes)', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot success rate
    if success_rates:
        axes[1].plot(range(100, 100 + len(success_rates)), success_rates, 
                     label='Success Rate (100-episode window)', linewidth=2)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Success Rate')
        axes[1].set_title('Success Rate Over Time')
        axes[1].set_ylim([0, 1])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150)
    print(f"\nTraining plot saved to: training_progress.png")
    plt.close()


def main_pipeline(use_dqn=False, dqn_episodes=5000):
    """
    Complete training pipeline
    
    Args:
        use_dqn: If True, use DQN agent (slower). If False, use Hybrid agent (faster, often better)
        dqn_episodes: Number of episodes for DQN training
    """
    start_time = time.time()
    
    print("\n" + "="*60)
    print("HANGMAN AI TRAINING PIPELINE")
    print("="*60)
    print(f"Configuration:")
    print(f"  Agent Type: {'Deep Q-Network (DQN)' if use_dqn else 'Hybrid HMM-Based'}")
    print(f"  DQN Episodes: {dqn_episodes if use_dqn else 'N/A'}")
    print("="*60 + "\n")
    
    # Step 1: Train HMM
    hmm = train_hmm('corpus.txt', 'hangman_hmm.pkl')
    
    # Step 2: Train/Setup Agent
    if use_dqn:
        agent = train_dqn_agent(hmm, 'corpus.txt', episodes=dqn_episodes)
    else:
        agent = train_hybrid_agent(hmm, 'corpus.txt', 'test.txt')
    
    # Step 3: Final Evaluation
    results = evaluate_final(agent, hmm, 'test.txt', num_games=2000)
    
    elapsed = time.time() - start_time
    print(f"\nTotal pipeline time: {elapsed/60:.2f} minutes")
    
    return hmm, agent, results


# ============================================================================
# QUICK START SCRIPTS
# ============================================================================

def quick_train_and_test():
    """
    RECOMMENDED: Quick training for maximum performance in 5 hours
    Uses Hybrid agent which is faster and often better for Hangman
    """
    return main_pipeline(use_dqn=False)


def full_dqn_training():
    """
    Full DQN training (takes longer, ~2-3 hours for 5000 episodes)
    May or may not outperform Hybrid agent
    """
    return main_pipeline(use_dqn=True, dqn_episodes=5000)


if __name__ == "__main__":
    # CHOOSE YOUR TRAINING METHOD:
    
    # Option 1: Fast, High-Performance (RECOMMENDED for 5-hour constraint)
    hmm, agent, results = quick_train_and_test()
    
    # Option 2: Deep RL approach (experimental, takes longer)
    # hmm, agent, results = full_dqn_training()