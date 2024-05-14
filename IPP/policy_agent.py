import numpy as np
import random


class Policy_Agent:
    
    def __init__(self, env_shape, n_flags, actions, 
                 learning_rate=0.1, lr_scheduler= lambda x: x,
                 discount_factor=0.95, 
                 epsilon=0.9, epsilon_decay=0.999, epsilon_min=0.1, 
                 temp=0.1):
        
        self.logits = np.zeros(shape=(*env_shape, *([2]*n_flags), len(actions)))
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.actions = actions

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(range(len(self.actions)))
        else:
            action_index = sample_with_softmax(self.logits[state])
#             action_index = np.argmax(self.q_values[state])
            return action_index # self.actions[action_index]

    def update_policy(self, states, actions, returns):
        
        # Loop over steps in episode
        for state, action, G in zip(states, actions, returns):
            probs = softmax(self.logits[state])
#             print('Before', state, self.logits[state])
            for a in range(len(probs)):  # Update every action logit for the given state
                if a == action:
                    # For the taken action: increase logit proportionally to (1 - probability of the action)
                    self.logits[state][a] += self.learning_rate * G * (1 - probs[a])
                else:
                    # For all other actions: decrease logit proportionally to the probability of the action
                    self.logits[state][a] -= self.learning_rate * G * probs[a]
#             print('After ', state, self.logits[state])
            
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
            
            
    def update_learning_rate(self, **kwargs):
        self.learning_rate = self.lr_scheduler(self.learning_rate, **kwargs)
            
            
    def __repr__(self):
        return f'''Policy Agent Object 
    Current Epsilon = {self.epsilon}
    Learning Rate   = {self.learning_rate}
    Discount Factor = {self.discount_factor}
    
    Policy Logits:
{self.logits}''' 


def softmax(x, temp=1):
    """ Returns softmax probabilities with temperature tau
        Input:  x -- 1-dimensional array
        Output: s -- 1-dimensional array
        
        Note: The inclusion of the -x.max() is my own little trick to avoid overflow errors!
        Overflow errors are harmful but underflow errors are inconsequential here.
    """
    e_x = np.exp((x-x.max()) / temp)
    return e_x / e_x.sum()


def sample_with_softmax(arr, temp=1):
    # Can recover simple argmax behavior while avoiding ZeroDiv errors
    if temp==0: return np.argmax(arr)
    # Compute softmax and produce a weighted random sample
    weights = softmax(np.array(arr), temp)
#     print(weights)
    return random.choices([0,1,2,3], weights=weights)[0]