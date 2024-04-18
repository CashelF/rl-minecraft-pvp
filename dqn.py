import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(9216, 512)  # This line might need adjustment depending on the final output size of the conv layers
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def train_dqn(env, model, transform, episodes=500, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64, max_t=1000):
    memory = deque(maxlen=10000)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    epsilon = epsilon_start
    action_size = env.action_space.n

    for episode in range(episodes):
        state = env.reset()
        state = transform(state).unsqueeze(0)  # Apply transformations and add batch dimension
        total_reward = 0

        for t in range(max_t):
            if random.random() > epsilon:
                with torch.no_grad():
                    action = model(state).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[random.randrange(action_size)]], dtype=torch.long)

            next_state, reward, done, _ = env.step(action.item())
            next_state = transform(next_state).unsqueeze(0)
            
            state = torch.tensor(state, dtype=torch.float32)
            action = torch.tensor([action], dtype=torch.long)
            reward = torch.tensor([reward], dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            done = torch.tensor([done], dtype=torch.float32)
            
            memory.append((state, action, reward, next_state, done))
            
            state = next_state
            total_reward += reward

            if len(memory) > batch_size:
                transitions = random.sample(memory, batch_size)
                batch = tuple(zip(*transitions))
                states, actions, rewards, next_states, dones = [torch.cat(batch_component) for batch_component in batch]

                current_q_values = model(states).gather(1, actions)
                next_q_values = model(next_states).max(1)[0].detach()
                expected_q_values = torch.tensor(rewards) + (gamma * next_q_values * (1 - torch.tensor(dones)))

                loss = loss_fn(current_q_values.squeeze(), expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(epsilon_end, epsilon_decay * epsilon)  # Decrease epsilon
        print(f"Episode {episode+1}/{episodes}, Total reward: {total_reward}, Epsilon: {epsilon}")

