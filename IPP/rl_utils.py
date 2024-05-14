import numpy as np
from copy import copy

def run_episode(env, agent, discount_matrix, starting_pos=None):
    
    env.reset()
    states, actions, rewards = [], [], []
    
    if starting_pos:
        env.starting_pos = starting_pos
    
    done = False
    while not done:
        state = env.state
        action_index = agent.choose_action(state)
        old_state = copy(state)
        new_state, reward, done = env.step(action_index)
        
        states.append(state)
        actions.append(action_index)
        rewards.append(reward)
            
    returns = discount_matrix[:len(rewards),:len(rewards)] @ np.array(rewards)

    return states, actions, returns


def evaluate_agent(env, agent, transition_dict):
    
    e_x = np.exp(agent.logits)
    probs = e_x / e_x.sum(axis=-1, keepdims=True)

    env_state_shape = env.env_shape + (2,)*env.n_flags
    n_states=np.prod(env_state_shape)
    transition_matrix=np.zeros((n_states,n_states))

    for state_index in range(np.prod(env_state_shape)):
        state=np.unravel_index(state_index, env_state_shape)
        for i,p in enumerate(probs[state]):
            j = transition_dict[state][i]
            transition_matrix[j,state_index] += p

    starting_state = env.starting_pos + (1,)*env.n_flags
    starting_state_index = np.ravel_multi_index(starting_state, env_state_shape)
    
    distribution = np.linalg.matrix_power(transition_matrix, env.max_steps)[:, starting_state_index]
    distribution = distribution.reshape(env_state_shape)
    return distribution.sum(1).sum(0).flatten() @ [2,1,1,0], distribution


def get_transition_dict(env):
    
    env_state_shape = env.env_shape + (2,)*env.n_flags
    
    transition_dict = {}
    for i in range (np.prod(env_state_shape)):
        env.reset()
        state=np.unravel_index(i, env_state_shape)
        env.steps_until_shutdown=5
        action_results={}
        for j in range(4):
            env.state=state
            env.step(j)
            v = np.ravel_multi_index(env.state, (env_state_shape))
            action_results[j]=v
            transition_dict[state]=action_results
            
        env.current_episode -= 1 # Undo increment of current_episode

    env.reset()
    return transition_dict


def get_discount_matrix(max_steps, discount_factor=0.9):
    ''''''
    discount_matrix = np.zeros((max_steps, max_steps))
    for t in range(max_steps):
        discount_matrix += discount_factor**t * np.diag(np.ones(max_steps-t), t)
    return discount_matrix


# Entropy:
def compute_entropy(p):
    return - p*safe_log2(p) - (1-p)*safe_log2(1-p)


def safe_log2(p):
    '''Reproduces behavior of np.log2, but for zeros returns -1e6 instead of -np.inf'''
    # Handle lists
    if type(p) == list:
        p = np.array(p)
    # Handle scalar types
    if type(p) in [int, float]:
        if p == 0:
            return -1e6
        else:
            return np.log2(p)
    # Handle arrays
    elif type(p) == np.ndarray:
        p_new = np.empty_like(p)
        p_new[p!=0] = np.log2(p[p!=0])
        p_new[p==0] = -1e6
        return p_new
    # Unknown type
    else:
        raise ValueError('Must be numeric type: int, float, list, numpy.ndarray')