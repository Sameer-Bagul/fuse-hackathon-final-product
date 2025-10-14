import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PPOPolicyNetwork(nn.Module):
    """Policy network for PPO agent"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(PPOPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class PPOValueNetwork(nn.Module):
    """Value network for PPO agent"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super(PPOValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent for curriculum-driven learning"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128,
                 lr: float = 3e-4, gamma: float = 0.99, epsilon: float = 0.2,
                 value_coeff: float = 0.5, entropy_coeff: float = 0.01,
                 max_grad_norm: float = 0.5, ppo_epochs: int = 10,
                 mini_batch_size: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon  # PPO clipping parameter
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        # Neural networks
        self.policy_net = PPOPolicyNetwork(state_dim, hidden_dim, action_dim)
        self.value_net = PPOValueNetwork(state_dim, hidden_dim)
        self.policy_old = PPOPolicyNetwork(state_dim, hidden_dim, action_dim)
        self.value_old = PPOValueNetwork(state_dim, hidden_dim)

        # Copy old networks
        self._copy_networks()

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.value_net.to(self.device)
        self.policy_old.to(self.device)
        self.value_old.to(self.device)

        # Experience buffer with size limit
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.max_buffer_size = 100  # Limit buffer to prevent memory issues and dimension mismatches

        # Curriculum learning state
        self.curriculum_state = {
            'current_skill': None,
            'difficulty_level': 0,
            'task_progress': 0,
            'success_streak': 0
        }

        logger.info(f"PPO Agent initialized with state_dim={state_dim}, action_dim={action_dim}")

    def _copy_networks(self):
        """Copy current networks to old networks"""
        self.policy_old.load_state_dict(self.policy_net.state_dict())
        self.value_old.load_state_dict(self.value_net.state_dict())

    def get_state_representation(self, prompt_text: str, curriculum_context: Dict[str, Any]) -> np.ndarray:
        """Convert prompt and curriculum context to state representation"""
        # Simple state representation - can be enhanced with embeddings
        state = np.zeros(self.state_dim)

        # Basic features from prompt
        prompt_length = len(prompt_text.split())
        state[0] = min(prompt_length / 100.0, 1.0)  # Normalized prompt length

        # Curriculum context features
        current_skill = curriculum_context.get('current_skill', '')
        difficulty_level = curriculum_context.get('difficulty_level', 0)
        task_progress = curriculum_context.get('task_progress', 0)
        success_streak = curriculum_context.get('success_streak', 0)

        # Skill encoding (simple one-hot style)
        skill_hash = abs(hash(current_skill)) % 10 if current_skill else 0
        state[1] = skill_hash / 10.0

        # Difficulty and progress
        state[2] = difficulty_level / 4.0  # Assuming 4 difficulty levels
        state[3] = task_progress
        state[4] = min(success_streak / 10.0, 1.0)  # Success streak

        # Add some prompt-based features
        keywords = prompt_text.lower().split()
        ml_keywords = ['machine', 'learning', 'neural', 'network', 'data', 'model']
        code_keywords = ['function', 'class', 'import', 'def', 'return']

        ml_count = sum(1 for kw in keywords if kw in ml_keywords)
        code_count = sum(1 for kw in keywords if kw in code_keywords)

        state[5] = min(ml_count / 5.0, 1.0)
        state[6] = min(code_count / 5.0, 1.0)

        # Ensure state values are valid
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        state = np.clip(state, 0.0, 1.0)  # Ensure values are in [0, 1]

        return state

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get action probabilities from current policy
            action_probs = self.policy_net(state_tensor)
            value = self.value_net(state_tensor)

            # Validate action probabilities
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                logger.warning("Invalid action probabilities detected, using uniform distribution")
                action_probs = torch.ones_like(action_probs) / self.action_dim

            # Ensure probabilities sum to 1 and are non-negative
            action_probs = torch.clamp(action_probs, min=0.0)
            if action_probs.sum() == 0:
                action_probs = torch.ones_like(action_probs) / self.action_dim
            else:
                action_probs = action_probs / action_probs.sum()

            # Sample action
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state: np.ndarray, action: int, log_prob: float,
                        reward: float, value: float, done: bool):
        """Store transition in experience buffer"""
        # Check buffer size limit - clear if too large to prevent memory issues
        if len(self.states) >= self.max_buffer_size:
            logger.warning(f"⚠️  PPO buffer reached max size ({self.max_buffer_size}), clearing old transitions")
            # Keep only the most recent half to maintain some history
            keep_size = self.max_buffer_size // 2
            self.states = self.states[-keep_size:]
            self.actions = self.actions[-keep_size:]
            self.log_probs = self.log_probs[-keep_size:]
            self.rewards = self.rewards[-keep_size:]
            self.values = self.values[-keep_size:]
            self.dones = self.dones[-keep_size:]
            logger.info(f"   Kept most recent {keep_size} transitions")
        
        # Validate state shape
        if state.shape[0] != self.state_dim:
            logger.error(f"🔴 State shape mismatch in store_transition! Expected {self.state_dim}, got {state.shape[0]}")
            logger.error(f"   Buffer sizes before: states={len(self.states)}, actions={len(self.actions)}, rewards={len(self.rewards)}")
            # Reshape or pad/truncate to correct dimension
            if state.shape[0] < self.state_dim:
                padded_state = np.zeros(self.state_dim)
                padded_state[:state.shape[0]] = state
                state = padded_state
                logger.warning(f"   Padded state from {state.shape[0]} to {self.state_dim}")
            else:
                state = state[:self.state_dim]
                logger.warning(f"   Truncated state from {state.shape[0]} to {self.state_dim}")
        
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
        # Log buffer status every 10 transitions
        if len(self.states) % 10 == 0:
            logger.debug(f"📊 PPO Buffer: {len(self.states)} transitions | state_dim={self.state_dim} | max={self.max_buffer_size}")

    def compute_returns_and_advantages(self, rewards: List[float], values: List[float],
                                     dones: List[bool], next_value: float) -> Tuple[List[float], List[float]]:
        """Compute returns and advantages using GAE"""
        returns = []
        advantages = []
        gae = 0
        next_value = next_value

        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0

            # TD error
            delta = rewards[i] + self.gamma * next_value - values[i]

            # Generalized Advantage Estimation
            gae = delta + self.gamma * 0.95 * gae  # lambda = 0.95

            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

            next_value = values[i]

        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages.tolist()

    def update_policy(self):
        """Update policy and value networks using PPO"""
        if len(self.states) == 0:
            return

        # Ensure all buffers are aligned in length. If misaligned, trim to smallest length.
        lengths = [len(self.states), len(self.actions), len(self.log_probs), len(self.rewards), len(self.values), len(self.dones)]
        min_len = min(lengths)
        if min_len == 0:
            logger.warning("PPO buffers have zero-length after alignment, skipping update")
            self.clear_buffer()
            return

        if len(set(lengths)) != 1:
            logger.warning(f"PPO buffers length mismatch, trimming to min length={min_len} | lengths={lengths}")

        # Trim buffers to consistent length
        states_arr = np.array(self.states[:min_len])
        actions_arr = self.actions[:min_len]
        log_probs_arr = self.log_probs[:min_len]
        rewards = self.rewards[:min_len]
        values = self.values[:min_len]
        dones = self.dones[:min_len]
        
        # 🔍 INSTRUMENTATION: Log shapes before tensor conversion
        logger.info(f"🔍 PPO Update Debug:")
        logger.info(f"   Buffer lengths: states={len(self.states)}, actions={len(self.actions)}, rewards={len(self.rewards)}")
        logger.info(f"   States array shape: {states_arr.shape}")
        logger.info(f"   Expected state_dim: {self.state_dim}")
        logger.info(f"   Actions: {len(actions_arr)}, LogProbs: {len(log_probs_arr)}")
        
        # Validate state dimensions
        if states_arr.shape[1] != self.state_dim:
            logger.error(f"🔴 CRITICAL: State dimension mismatch! Expected {self.state_dim}, got {states_arr.shape[1]}")
            logger.error(f"   First state shape: {self.states[0].shape if len(self.states) > 0 else 'N/A'}")
            logger.error(f"   Last state shape: {self.states[-1].shape if len(self.states) > 0 else 'N/A'}")
            # Check if any states have wrong dimensions
            bad_states = [(i, s.shape) for i, s in enumerate(self.states[:min_len]) if s.shape[0] != self.state_dim]
            if bad_states:
                logger.error(f"   Found {len(bad_states)} states with wrong dimensions: {bad_states[:5]}")
            self.clear_buffer()
            return

        # Convert to tensors
        try:
            states = torch.FloatTensor(states_arr).to(self.device)
            actions = torch.LongTensor(actions_arr).to(self.device)
            old_log_probs = torch.FloatTensor(log_probs_arr).to(self.device)
            
            logger.info(f"   Tensor shapes: states={states.shape}, actions={actions.shape}, log_probs={old_log_probs.shape}")
        except Exception as e:
            logger.error(f"Failed to convert PPO buffers to tensors: {e}")
            self.clear_buffer()
            return

        # Compute returns and advantages
        next_value = 0  # Assume episode ends
        returns, advantages = self.compute_returns_and_advantages(rewards, values, dones, next_value)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Create dataset
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=True)

        # PPO update loop with defensive guards against shape mismatches and runtime errors
        try:
            for epoch in range(self.ppo_epochs):
                for batch_idx, (batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages) in enumerate(dataloader):
                    try:
                        # 🔍 Log batch shapes for first epoch
                        if epoch == 0 and batch_idx == 0:
                            logger.info(f"   Batch shapes: states={batch_states.shape}, actions={batch_actions.shape}, advantages={batch_advantages.shape}")
                        
                        # Get current policy outputs
                        action_probs = self.policy_net(batch_states)

                        # Validate action probabilities
                        if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                            logger.warning("Invalid action probabilities in PPO update, using uniform distribution")
                            action_probs = torch.ones_like(action_probs) / self.action_dim

                        # Ensure probabilities are valid
                        action_probs = torch.clamp(action_probs, min=1e-8)  # Avoid exact zeros
                        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

                        dist = Categorical(action_probs)
                        new_log_probs = dist.log_prob(batch_actions)
                        entropy = dist.entropy().mean()

                        # Get current value estimates
                        values_pred = self.value_net(batch_states).squeeze()
                        
                        # 🔍 Check for shape mismatches before computing loss
                        if values_pred.shape != batch_returns.shape:
                            logger.error(f"🔴 Shape mismatch in batch {batch_idx}: values_pred={values_pred.shape}, batch_returns={batch_returns.shape}")
                            logger.error(f"   batch_states.shape={batch_states.shape}, batch_advantages.shape={batch_advantages.shape}")
                            raise RuntimeError(f"Value prediction shape {values_pred.shape} doesn't match returns shape {batch_returns.shape}")

                        # Policy loss (PPO clipped objective)
                        # 🔍 Safety check for tensor dimension mismatch
                        if new_log_probs.shape != batch_old_log_probs.shape:
                            logger.error(f"🔴 Log prob shape mismatch: new={new_log_probs.shape}, old={batch_old_log_probs.shape}")
                            logger.error(f"   Skipping this batch to prevent crash")
                            continue
                        
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Value loss
                        value_loss = F.mse_loss(values_pred, batch_returns)

                        # Total loss
                        loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

                        # Update policy network
                        self.policy_optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                        self.policy_optimizer.step()

                        # Update value network
                        value_loss = F.mse_loss(values_pred, batch_returns)
                        self.value_optimizer.zero_grad()
                        value_loss.backward()
                        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                        self.value_optimizer.step()

                    except Exception as batch_e:
                        logger.warning(f"PPO update batch {batch_idx} failed (epoch {epoch}), skipping this batch: {batch_e}")
                        continue
        except Exception as e:
            logger.error(f"PPO update failed: {e}")

        # Copy updated networks to old networks
        self._copy_networks()

        # Clear experience buffer
        self.clear_buffer()

        logger.info("PPO policy updated")

    def clear_buffer(self):
        """Clear experience buffer"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def update_curriculum_state(self, skill: str, difficulty: int, progress: float, success: bool):
        """Update curriculum learning state"""
        self.curriculum_state['current_skill'] = skill
        self.curriculum_state['difficulty_level'] = difficulty
        self.curriculum_state['task_progress'] = progress

        if success:
            self.curriculum_state['success_streak'] += 1
        else:
            self.curriculum_state['success_streak'] = 0

    def get_curriculum_context(self) -> Dict[str, Any]:
        """Get current curriculum context for state representation"""
        return self.curriculum_state.copy()

    def save_model(self, path: str):
        """Save model parameters"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'curriculum_state': self.curriculum_state
        }, path)

    def load_model(self, path: str):
        """Load model parameters"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.curriculum_state = checkpoint.get('curriculum_state', self.curriculum_state)
        self._copy_networks()

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities for a given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy_net(state_tensor)
        return probs.cpu().numpy().flatten()

    def get_value_estimate(self, state: np.ndarray) -> float:
        """Get value estimate for a given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.value_net(state_tensor)
        return value.item()