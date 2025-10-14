import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, deque
from .history import History
from .meta_learning import LearningStrategy

class ActorCritic(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

class PPOAgent:
    def __init__(self, input_dim, num_actions, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.policy = ActorCritic(input_dim, num_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(input_dim, num_actions)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.policy_old(state)
        action_probs = action_probs.squeeze(0).numpy()
        action = np.random.choice(len(action_probs), p=action_probs)
        return action, action_probs[action]

    def update(self, memory):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).detach()

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            action_probs, state_values = self.policy(old_states)
            dist = torch.distributions.Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.squeeze(-1).detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.mse_loss(state_values.squeeze(-1), rewards) - 0.01*dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class LLM:
    def __init__(self, num_actions, input_dim=10, alpha=0.1, epsilon=0.1, history=None, meta_learning_service=None):
        self.num_actions = num_actions
        self.input_dim = input_dim
        self.ppo_agent = PPOAgent(input_dim, num_actions)
        self.memory = Memory()
        self.alpha = alpha
        self.epsilon = epsilon
        self.history = history or History()
        self.patterns = defaultdict(int)  # For learning patterns from prompts
        self.episode_rewards = deque(maxlen=100)  # Track recent rewards
        self.last_multi_objective_reward = None  # Store last multi-objective reward details

        # Meta-learning integration
        self.meta_learning_service = meta_learning_service
        self.learning_context = {}  # Current learning context for meta-learning
        self.dynamic_parameters = {
            'alpha': alpha,
            'epsilon': epsilon,
            'exploration_weight': 0.5,
            'exploitation_weight': 0.5
        }

    def _get_state_from_prompt(self, prompt_text):
        # Simple state representation: word frequencies as features
        words = prompt_text.lower().split()
        state = np.zeros(self.input_dim, dtype=np.float32)
        for i, word in enumerate(words[:self.input_dim]):
            state[i] = hash(word) % 100 / 100.0  # Normalize hash to [0,1]
        return state

    def process_prompt(self, prompt_text, context=None):
        # Update learning context for meta-learning
        if context:
            self.learning_context.update(context)

        # Apply dynamic parameters from meta-learning
        self._apply_dynamic_parameters()

        # Analyze prompt for patterns
        words = prompt_text.lower().split()
        for word in words:
            self.patterns[word] += 1

        state = self._get_state_from_prompt(prompt_text)

        # Use dynamic epsilon for exploration-exploitation balance
        current_epsilon = self.dynamic_parameters.get('epsilon', self.epsilon)

        # Apply strategy-specific action selection
        if self.meta_learning_service:
            strategy = self.meta_learning_service.meta_learner.current_strategy
            action, logprob = self._strategy_based_action_selection(state, strategy, current_epsilon)
        else:
            action, logprob = self.ppo_agent.select_action(state)

        # Store in memory
        self.memory.states.append(torch.FloatTensor(state))
        self.memory.actions.append(torch.tensor(action))
        self.memory.logprobs.append(torch.tensor(logprob))

        response = self._generate_response(prompt_text, action)
        return response, action

    def _generate_response(self, prompt, action):
        """Generate contextual ML prompts based on user input topics"""
        prompt_lower = prompt.lower()

        # Extract keywords from the prompt (simple approach: split and filter)
        words = prompt_lower.split()
        # Filter out common stop words (basic list)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2][:5]  # Take up to 5 keywords, longer than 2 chars

        if not keywords:
            keywords = words[:3]  # Fallback to first 3 words if all filtered out

        # Join keywords for use in prompts
        keyword_str = ' and '.join(keywords) if len(keywords) > 1 else keywords[0] if keywords else 'general topics'

        # Define ML techniques and tasks
        techniques = [
            'decision tree classifier', 'k-means clustering algorithm', 'linear regression model',
            'random forest model', 'support vector machine (SVM)', 'logistic regression model',
            'neural network', 'gradient boosting classifier', 'principal component analysis (PCA)',
            'hierarchical clustering', 'autoencoder neural network', 'Gaussian mixture model',
            'Q-learning agent', 'policy gradient agent', 'Deep Q-Network (DQN)',
            'convolutional neural network (CNN)', 'recurrent neural network (RNN)',
            'generative adversarial network (GAN)', 'transformer model', 'ResNet model',
            'BERT-based model', 'U-Net architecture', 'variational autoencoder (VAE)',
            'text classification model', 'named entity recognition system',
            'machine translation model', 'chatbot', 'object detection using YOLO',
            'face recognition system', 'image captioning model', 'style transfer network',
            'exploratory data analysis', 'recommendation system', 'time series forecasting model',
            'A/B testing framework', 'stochastic gradient descent', 'genetic algorithm',
            'Bayesian optimization', 'constraint optimization'
        ]

        tasks = [
            'classify', 'predict', 'cluster', 'segment', 'recognize', 'detect',
            'analyze', 'optimize', 'forecast', 'recommend', 'translate', 'generate',
            'segment', 'identify', 'categorize', 'extract', 'navigate', 'balance',
            'play', 'achieve', 'perform', 'train', 'build', 'implement', 'design',
            'develop', 'create', 'construct', 'apply', 'use'
        ]

        # Select technique and task based on action
        technique = techniques[action % len(techniques)]
        task = tasks[action % len(tasks)]

        # Generate prompt incorporating user's keywords
        generated_prompt = f"Create a {technique} to {task} {keyword_str} based on relevant features and data"

        return generated_prompt

    def learn(self, action, reward, prompt_text=None, response_text=None):
        """
        Learn from reward feedback with meta-learning integration.

        Args:
            action: The action taken
            reward: Either a single float reward (backward compatibility)
                    or a dict with 'weighted_reward' key for multi-objective rewards
            prompt_text: Original prompt for meta-learning analysis
            response_text: Generated response for meta-learning analysis
        """
        # Handle both single reward and multi-objective reward formats
        if isinstance(reward, dict):
            # Multi-objective reward - use weighted reward
            actual_reward = reward.get('weighted_reward', 0.5)
            # Store additional reward info for analysis
            self.last_multi_objective_reward = reward
        else:
            # Single reward (backward compatibility)
            actual_reward = float(reward)
            self.last_multi_objective_reward = None

        # Store reward and terminal flag
        self.memory.rewards.append(actual_reward)
        self.memory.is_terminals.append(False)  # Assume not terminal for now

        self.episode_rewards.append(actual_reward)

        # Meta-learning integration: monitor performance if service is available
        if self.meta_learning_service and prompt_text and response_text:
            try:
                performance_analysis = self.meta_learning_service.monitor_performance(
                    prompt_text, response_text, action, reward, self.learning_context
                )
                # Use meta-learning insights to adjust learning
                self._apply_meta_learning_insights(performance_analysis)
            except Exception as e:
                print(f"Meta-learning monitoring failed: {e}")

        # Update PPO with dynamic learning rate
        current_alpha = self.dynamic_parameters.get('alpha', self.alpha)
        if hasattr(self.ppo_agent, 'optimizer'):
            for param_group in self.ppo_agent.optimizer.param_groups:
                param_group['lr'] = current_alpha

        # Update PPO if enough experiences
        if len(self.memory.rewards) >= 32:  # Mini-batch size
            self.ppo_agent.update(self.memory)
            self.memory.clear_memory()

    def adapt_from_history(self):
        # Adapt based on historical interactions
        success_rate = self.history.get_success_rate()
        # Could adjust learning rate or other params based on success_rate
        pass

    def get_average_reward(self):
        return np.mean(self.episode_rewards) if self.episode_rewards else 0.0

    def get_last_multi_objective_reward(self):
        """
        Get details of the last multi-objective reward received.

        Returns:
            Dict with multi-objective reward details or None if no multi-objective reward received
        """
        return self.last_multi_objective_reward

    def _apply_dynamic_parameters(self):
        """Apply dynamic parameters from meta-learning service"""
        if self.meta_learning_service:
            meta_params = self.meta_learning_service.meta_learner.current_params
            self.dynamic_parameters.update({
                'alpha': meta_params.get('alpha', self.alpha),
                'epsilon': meta_params.get('epsilon', self.epsilon),
                'exploration_weight': meta_params.get('strategy_weights', {}).get('exploration', 0.5),
                'exploitation_weight': meta_params.get('strategy_weights', {}).get('exploitation', 0.5)
            })

    def _strategy_based_action_selection(self, state, strategy, epsilon):
        """Select action based on current learning strategy"""
        if strategy == LearningStrategy.EXPLORATION_FOCUSED:
            # Increase exploration
            epsilon = min(epsilon * 1.5, 0.9)
        elif strategy == LearningStrategy.EXPLOITATION_FOCUSED:
            # Decrease exploration
            epsilon = max(epsilon * 0.5, 0.01)
        elif strategy == LearningStrategy.BALANCED:
            # Use configured epsilon
            pass
        elif strategy == LearningStrategy.ADAPTIVE:
            # Dynamic epsilon based on recent performance
            recent_rewards = list(self.episode_rewards)[-10:]
            if recent_rewards:
                avg_recent = sum(recent_rewards) / len(recent_rewards)
                if avg_recent < 0.4:
                    epsilon = min(epsilon * 1.3, 0.8)  # Increase exploration when struggling
                elif avg_recent > 0.7:
                    epsilon = max(epsilon * 0.7, 0.05)  # Decrease exploration when doing well

        # Select action with potentially modified epsilon
        if np.random.random() < epsilon:
            # Random exploration
            action = np.random.randint(self.num_actions)
            # Calculate log probability for random action (uniform distribution)
            logprob = np.log(1.0 / self.num_actions)
        else:
            # Use policy
            action, logprob = self.ppo_agent.select_action(state)

        return action, logprob

    def _apply_meta_learning_insights(self, performance_analysis):
        """Apply insights from meta-learning performance analysis"""
        if not performance_analysis.get('adaptation_needed', False):
            return

        # Adjust learning behavior based on analysis
        if performance_analysis.get('performance_metrics', {}).get('learning_progress', 0) < 0:
            # Learning is declining, increase adaptation frequency
            if self.meta_learning_service:
                self.meta_learning_service.adaptation_interval = max(
                    15, self.meta_learning_service.adaptation_interval * 0.8
                )

        # Update learning context with performance insights
        success_indicators = performance_analysis.get('performance_metrics', {}).get('success_indicators', {})
        if success_indicators.get('high_reward', False):
            self.learning_context['learner_state'] = 'performing_well'
        elif success_indicators.get('learning_progressing', False):
            self.learning_context['learner_state'] = 'improving'
        else:
            self.learning_context['learner_state'] = 'needs_adaptation'

    def update_meta_learning_context(self, context: dict):
        """Update the learning context for meta-learning"""
        self.learning_context.update(context)

    def get_meta_learning_status(self):
        """Get current meta-learning related status"""
        return {
            'dynamic_parameters': self.dynamic_parameters.copy(),
            'learning_context': self.learning_context.copy(),
            'meta_learning_enabled': self.meta_learning_service is not None,
            'current_strategy': self.meta_learning_service.meta_learner.current_strategy.value if self.meta_learning_service else None
        }