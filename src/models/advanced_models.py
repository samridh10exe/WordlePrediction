"""
Advanced ML models for Wordle prediction.
Implements neural networks, transformers, and ensemble methods.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import math
import random
from collections import deque, namedtuple
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoModel, AutoTokenizer, AutoConfig
import joblib


# Experience tuple for RL
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class WordleTransformer(nn.Module):
    """Transformer-based model for Wordle prediction."""
    
    def __init__(self, vocab_size: int, hidden_dim: int = 768, num_layers: int = 6, 
                 num_heads: int = 12, max_length: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Forward pass through transformer."""
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        embeddings = token_embeds + pos_embeds
        
        # Transformer expects (seq_len, batch_size, hidden_dim)
        embeddings = embeddings.transpose(0, 1)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (batch_size, seq_len) -> (seq_len, seq_len)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Pass through transformer
        transformer_output = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # Back to (batch_size, seq_len, hidden_dim)
        transformer_output = transformer_output.transpose(0, 1)
        
        # Pool the representation (use CLS token or mean pooling)
        if attention_mask is not None:
            # Mean pooling over valid tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(transformer_output.size()).float()
            sum_embeddings = torch.sum(transformer_output * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = torch.mean(transformer_output, dim=1)
        
        # Final classification
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class WordleRLAgent(nn.Module):
    """Reinforcement Learning agent for Wordle solving using A2C."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, state: torch.Tensor):
        """Forward pass returning policy and value."""
        features = self.feature_extractor(state)
        
        # Policy distribution
        policy_logits = self.actor(features)
        policy = F.softmax(policy_logits, dim=-1)
        
        # State value
        value = self.critic(features)
        
        return policy, value
    
    def get_action(self, state: torch.Tensor, available_actions: Optional[List[int]] = None):
        """Sample action from policy."""
        with torch.no_grad():
            policy, value = self.forward(state)
            
            # Mask unavailable actions
            if available_actions is not None:
                mask = torch.zeros_like(policy)
                mask[available_actions] = 1
                policy = policy * mask
                policy = policy / policy.sum()
            
            # Sample action
            action_dist = torch.distributions.Categorical(policy)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob, value


class WordleEnvironment:
    """Wordle game environment for RL training."""
    
    def __init__(self, vocabulary: List[str], target_words: List[str]):
        self.vocabulary = vocabulary
        self.target_words = target_words
        self.word_to_idx = {word: i for i, word in enumerate(vocabulary)}
        self.current_target = None
        self.guesses = []
        self.max_guesses = 6
        self.state_dim = 5 * 26 + 26 + 6  # Position info + letter status + guess count
        
        self.logger = logging.getLogger(__name__)
    
    def reset(self) -> np.ndarray:
        """Reset environment with random target word."""
        self.current_target = random.choice(self.target_words)
        self.guesses = []
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action (guess word) and return new state, reward, done, info."""
        if action >= len(self.vocabulary):
            # Invalid action
            return self._get_state(), -1.0, True, {'error': 'Invalid action'}
        
        guess_word = self.vocabulary[action]
        self.guesses.append(guess_word)
        
        # Calculate reward
        reward, done = self._calculate_reward(guess_word)
        
        # Check if game is over
        if len(self.guesses) >= self.max_guesses and not done:
            done = True
            reward = -2.0  # Penalty for not solving
        
        info = {
            'target': self.current_target,
            'guess': guess_word,
            'guesses_made': len(self.guesses),
            'solved': guess_word == self.current_target
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        state = np.zeros(self.state_dim)
        
        if not self.guesses:
            return state
        
        # Position-specific letter information (5 positions × 26 letters)
        pos_offset = 0
        for pos in range(5):
            for i, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                # Check status of this letter at this position based on previous guesses
                status = self._get_letter_position_status(letter, pos)
                state[pos_offset + i] = status
            pos_offset += 26
        
        # Global letter status (26 letters)
        letter_offset = 5 * 26
        for i, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            state[letter_offset + i] = self._get_letter_global_status(letter)
        
        # Number of guesses made
        state[-6:len(self.guesses)] = 1  # One-hot encoding of guess count
        
        return state
    
    def _calculate_reward(self, guess_word: str) -> Tuple[float, bool]:
        """Calculate reward for guess."""
        if guess_word == self.current_target:
            # Solved! Reward based on number of guesses
            bonus = max(0, 7 - len(self.guesses))  # Bonus for solving quickly
            return 10.0 + bonus, True
        
        # Partial reward for correct letters
        correct_positions = sum(1 for i, (g, t) in enumerate(zip(guess_word, self.current_target)) if g == t)
        correct_letters = len(set(guess_word) & set(self.current_target))
        
        reward = correct_positions * 0.5 + (correct_letters - correct_positions) * 0.2
        
        return reward, False
    
    def _get_letter_position_status(self, letter: str, position: int) -> float:
        """Get status of letter at specific position: 1=correct, 0.5=wrong pos, 0=not in word, -1=unknown."""
        for guess in self.guesses:
            if len(guess) > position:
                if guess[position] == letter:
                    if self.current_target[position] == letter:
                        return 1.0  # Correct position
                    else:
                        return -0.5  # Wrong position
                elif letter in self.current_target:
                    return 0.5  # Letter exists but wrong position
                elif letter in guess:
                    return 0.0  # Letter not in target word
        
        return -1.0  # Unknown
    
    def _get_letter_global_status(self, letter: str) -> float:
        """Get global status of letter: 1=in word, 0=not in word, -1=unknown."""
        for guess in self.guesses:
            if letter in guess:
                if letter in self.current_target:
                    return 1.0
                else:
                    return 0.0
        return -1.0


class A2CTrainer:
    """Advantage Actor-Critic trainer for Wordle RL agent."""
    
    def __init__(self, agent: WordleRLAgent, environment: WordleEnvironment, 
                 learning_rate: float = 3e-4, gamma: float = 0.99, 
                 value_loss_coef: float = 0.5, entropy_coef: float = 0.01):
        self.agent = agent
        self.env = environment
        self.optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        self.logger = logging.getLogger(__name__)
    
    def train_episode(self) -> Dict:
        """Train on single episode."""
        states, actions, rewards, log_probs, values = [], [], [], [], []
        
        state = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = self.agent.get_action(state_tensor)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            
            state, reward, done, info = self.env.step(action)
            rewards.append(reward)
            total_reward += reward
        
        # Calculate returns and advantages
        returns = self._calculate_returns(rewards)
        advantages = self._calculate_advantages(returns, values)
        
        # Calculate losses
        policy_loss, value_loss, entropy_loss = self._calculate_losses(
            log_probs, advantages, returns, values
        )
        
        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'total_reward': total_reward,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'solved': info.get('solved', False),
            'guesses': info.get('guesses_made', 6)
        }
    
    def train(self, num_episodes: int, save_every: int = 1000) -> List[Dict]:
        """Train agent for specified number of episodes."""
        self.logger.info(f"Starting A2C training for {num_episodes} episodes...")
        
        training_history = []
        best_avg_reward = float('-inf')
        
        for episode in range(num_episodes):
            episode_results = self.train_episode()
            training_history.append(episode_results)
            
            # Logging
            if episode % 100 == 0:
                recent_rewards = [r['total_reward'] for r in training_history[-100:]]
                avg_reward = np.mean(recent_rewards)
                solve_rate = np.mean([r['solved'] for r in training_history[-100:]])
                avg_guesses = np.mean([r['guesses'] for r in training_history[-100:]])
                
                self.logger.info(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, "
                               f"Solve Rate: {solve_rate:.2f}, Avg Guesses: {avg_guesses:.2f}")
                
                # Save best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.save_checkpoint(f"best_model_episode_{episode}")
            
            # Periodic saves
            if episode % save_every == 0 and episode > 0:
                self.save_checkpoint(f"checkpoint_episode_{episode}")
        
        self.logger.info("Training completed!")
        return training_history
    
    def _calculate_returns(self, rewards: List[float]) -> torch.Tensor:
        """Calculate discounted returns."""
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        return torch.FloatTensor(returns)
    
    def _calculate_advantages(self, returns: torch.Tensor, values: List[torch.Tensor]) -> torch.Tensor:
        """Calculate advantages using returns and value estimates."""
        values_tensor = torch.cat(values)
        advantages = returns - values_tensor.squeeze()
        return advantages
    
    def _calculate_losses(self, log_probs: List[torch.Tensor], advantages: torch.Tensor,
                         returns: torch.Tensor, values: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate policy, value, and entropy losses."""
        log_probs_tensor = torch.cat(log_probs)
        values_tensor = torch.cat(values).squeeze()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss
        policy_loss = -(log_probs_tensor * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(values_tensor, returns)
        
        # Entropy loss (for exploration)
        entropy_loss = log_probs_tensor.mean()  # Simplified entropy
        
        return policy_loss, value_loss, entropy_loss
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = Path("models") / f"{name}.pth"
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gamma': self.gamma,
            'value_loss_coef': self.value_loss_coef,
            'entropy_coef': self.entropy_coef
        }, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")


class EnsembleAdvancedModel(BaseEstimator, ClassifierMixin):
    """Ensemble combining transformer and RL models."""
    
    def __init__(self, vocab_size: int, weights: Optional[Dict[str, float]] = None):
        self.vocab_size = vocab_size
        self.weights = weights or {'transformer': 0.6, 'rl': 0.4}
        self.models = {}
        self.vocabulary = []
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train ensemble of advanced models."""
        self.logger.info("Training EnsembleAdvancedModel...")
        
        if 'word' in X.columns:
            self.vocabulary = X['word'].unique().tolist()
        else:
            raise ValueError("X must contain 'word' column")
        
        # Initialize models
        self.models['transformer'] = WordleTransformer(
            vocab_size=len(self.vocabulary),
            hidden_dim=256,  # Smaller for efficiency
            num_layers=4,
            num_heads=8
        ).to(self.device)
        
        # RL Agent
        state_dim = 5 * 26 + 26 + 6  # As defined in environment
        self.models['rl'] = WordleRLAgent(
            state_dim=state_dim,
            action_dim=len(self.vocabulary),
            hidden_dim=256
        ).to(self.device)
        
        # Train transformer (simplified training)
        self._train_transformer(X, y)
        
        # Train RL agent
        self._train_rl_agent(X)
        
        self.is_fitted = True
        self.logger.info("Ensemble training complete")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return ensemble probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get transformer predictions
        transformer_probs = self._get_transformer_predictions(X)
        
        # Get RL predictions
        rl_probs = self._get_rl_predictions(X)
        
        # Combine predictions
        ensemble_probs = (self.weights['transformer'] * transformer_probs + 
                         self.weights['rl'] * rl_probs)
        
        return ensemble_probs
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return most likely predictions."""
        probs = self.predict_proba(X)
        return np.array([self.vocabulary[np.argmax(prob)] for prob in probs])
    
    def _train_transformer(self, X: pd.DataFrame, y: pd.Series):
        """Simplified transformer training."""
        self.logger.info("Training transformer component...")
        
        # This is a simplified version - in practice, you'd need proper
        # sequence data and more sophisticated training
        model = self.models['transformer']
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Mock training data (in practice, use actual sequences)
        for epoch in range(10):
            mock_input = torch.randint(0, len(self.vocabulary), (32, 10)).to(self.device)
            mock_labels = torch.randint(0, len(self.vocabulary), (32,)).to(self.device)
            
            optimizer.zero_grad()
            logits = model(mock_input)
            loss = F.cross_entropy(logits, mock_labels)
            loss.backward()
            optimizer.step()
        
        self.logger.info("Transformer training complete")
    
    def _train_rl_agent(self, X: pd.DataFrame):
        """Train RL agent."""
        self.logger.info("Training RL agent...")
        
        # Create environment
        env = WordleEnvironment(self.vocabulary, self.vocabulary[:100])  # Subset for training
        
        # Create trainer
        trainer = A2CTrainer(self.models['rl'], env)
        
        # Train for limited episodes (in practice, use more)
        trainer.train(num_episodes=1000)
        
        self.logger.info("RL agent training complete")
    
    def _get_transformer_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from transformer."""
        # Mock implementation - return uniform probabilities
        batch_size = len(X)
        probs = np.ones((batch_size, len(self.vocabulary))) / len(self.vocabulary)
        return probs
    
    def _get_rl_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from RL agent."""
        # Mock implementation - return uniform probabilities
        batch_size = len(X)
        probs = np.ones((batch_size, len(self.vocabulary))) / len(self.vocabulary)
        return probs
    
    def save_model(self, filepath: Path):
        """Save ensemble model."""
        torch.save({
            'transformer_state': self.models['transformer'].state_dict(),
            'rl_state': self.models['rl'].state_dict(),
            'vocabulary': self.vocabulary,
            'weights': self.weights
        }, filepath)
        self.logger.info(f"Ensemble model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Path, vocab_size: int):
        """Load ensemble model."""
        checkpoint = torch.load(filepath)
        
        model = cls(vocab_size)
        model.vocabulary = checkpoint['vocabulary']
        model.weights = checkpoint['weights']
        
        # Reconstruct models
        model.models['transformer'] = WordleTransformer(vocab_size)
        model.models['transformer'].load_state_dict(checkpoint['transformer_state'])
        
        model.models['rl'] = WordleRLAgent(5 * 26 + 26 + 6, vocab_size)
        model.models['rl'].load_state_dict(checkpoint['rl_state'])
        
        model.is_fitted = True
        return model