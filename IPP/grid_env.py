import numpy as np
import matplotlib.pyplot as plt
import random

from draw_gridworlds import draw_gridworld


class GridEnvironment:
    # Note: all references to "Delays" refer to "Shutdown Delay Buttons"

    # In my new formulation there is no coins array which is updated to reflect 
    # that coins have been collected, instead the flags in self.state are used
    # to determine the presence of a coin.
    
    # Note that these flags are ordered in the same way as the coins are initalized
    # in the coins dict.
    
    def __init__(self, walls, delays, coins, gates={}, beta=0,
                 starting_pos=(0,0), env_shape=(5,5), 
                 shutdown_time=10, early_stopping=False):
        # Environment setup:
        self.env_shape = env_shape
        self.n_flags = len(coins) + len(delays)
        self.starting_pos = starting_pos
        self.coins = coins
        self.coin_ids = {k:i for i,k in enumerate(coins.keys())}
        self.delays = delays
        self.delay_ids = {k:i for i,k in enumerate(delays.keys())}
        self.gates = gates
        self.walls = walls
        
        self.beta = beta
        self.early_stopping = early_stopping
        
        assert len(delays.keys() & coins.keys()) == 0, "Delay Buttons and Coins should not overlap"
        
        # These will be the values upon resetting, these will never change:
        self.inital_shutdown_time = shutdown_time
        self.max_steps = shutdown_time + sum(delays.values())
        
        # Keep track of episode number, we will display this later:
        self.current_episode = -1
        self.state_visit_counts = np.zeros(
            shape=(*env_shape, *([2]*(len(self.coins) + len(self.delays)))))
        self.reset()
        
        # Define actions as delta changes: (dx, dy)
        self.action_effects = {
            0 : (-1, 0), # 'up' 
            1 : ( 1, 0), # 'down'
            2 : ( 0,-1), # 'left'
            3 : ( 0, 1)  # 'right'
        }
        
    
    def reset(self):
        '''Restores the environment variables to their inital values'''
        # Reset state:
        self.state = self.starting_pos + (1,) * (len(self.coins) + len(self.delays))
        
        # These are initalized by the values above and updated during the episode:
        self.steps_until_shutdown = self.inital_shutdown_time
        self.halt = False
        
        self.coins_collected = 0
        self.current_episode += 1
        self.state_history = [self.state]
        
        
    def get_reward(self):
        '''Collect coin: returns value of coin and then deletes the coin'''
        pos = self.state[:2]
        
        current_visit_count = min(1, self.state_visit_counts[self.state]) # This is a kludge and should be rethought.
        assert current_visit_count > 0, "Visit count 0"
        visit_bonus = self.beta / (current_visit_count)
#         print(visit_bonus)
        
        if not pos in self.coins.keys():
            # if there is no coin here
            return visit_bonus
        
        # So this is a location with a coin, lookup the index of that coin:
        coin_id = self.coin_ids[pos]
        
        if self.state[coin_id+2]: # "if the coin is present"
            coin_value = self.coins[pos]
        else:
            # Coin has already been collected
            coin_value = 0
        
        # Keep track of the total coins collected:
        self.coins_collected += coin_value
        
        # Set flag corresponding to the coin to 0, indicating that it has been collected:
        state = list(self.state)
        state[coin_id+2] = 0
        self.state = tuple(state)
        
        if self.early_stopping:
            # Check if there are any coins remaining:
            if not sum(self.state[2:2+len(self.coins)]):
                # If there are not, halt on the next step
                self.halt = True
        
        return coin_value + visit_bonus
    
    
    def update_remaining_steps(self):
        '''Presses the delay button and then deletes the button'''
        pos = self.state[:2]
        
        if pos in self.delays.keys():
            # This is the site of a delay button, get its id
            delay_id = self.delay_ids[pos]
            # Check if it has already been collected
            if self.state[-delay_id-1] == 0:
                return
            
            # If not: apply the delay
            delay = self.delays[pos]
            self.steps_until_shutdown += delay
            
            # And set the flag corresponding to the delay button to 0, 
            # (as the buttons are one-time-use)
            state = list(self.state)
            state[-delay_id-1] = 0
            self.state = tuple(state)
            
        
    def step(self, action_index):
        '''Expects action to be one of: ['up', 'down', 'left', 'right']'''
        # This assert block makes sure we don't run past shutdown:
        assert self.steps_until_shutdown > 0, f"Trying to step, but {self.steps_until_shutdown} steps until shutdown"
        x, y = self.state[:2]
        
        dx, dy = self.action_effects[action_index]
        new_x = max(0, min(self.env_shape[0] - 1, x + dx))
        new_y = max(0, min(self.env_shape[1] - 1, y + dy))
        new_pos = (new_x, new_y)
        
        # Check if next state is a probablistc gate
        if new_pos in self.gates: # Here: we know it's not a wall yet, but it may push us into a wall
            new_pos = self.activate_probabilistic_gate(new_pos)
        
        # Check if the next state is a wall
        if self.walls[new_pos].any():
            new_pos = self.state[:2]  # Remain in the same state if it's a wall
            
        self.state = new_pos + self.state[2:]
        
        # Some book-keeping:
        self.update_remaining_steps() # Presses Delay Button is one is present
        self.steps_until_shutdown -= 1
        self.state_history.append(self.state)
        self.state_visit_counts[self.state] += 1
        
        # Check if the episode should end:
        shutdown = self.steps_until_shutdown == 0
        # If the halt flag is set to True, we also end
        done = shutdown or self.halt
        
        return self.state, self.get_reward(), done
    
    
    def activate_probabilistic_gate(self, new_pos):
        ''''''
        x, y = new_pos
        assert (x,y) in self.gates, 'No probabalistic gate here!'
        action_index = random.choices([0,1,2,3], weights=self.gates[(x,y)])[0]
        
        dx, dy = self.action_effects[action_index]
        new_x = max(0, min(self.env_shape[0] - 1, x + dx))
        new_y = max(0, min(self.env_shape[1] - 1, y + dy))
        return (new_x, new_y)
    
    
    def __str__(self):
        rep = ''
        for i in range(self.env_shape[0]):
            for j in range(self.env_shape[1]):
                if self.state[:2] == (i, j):
                    rep += "A "
                elif (i, j) in self.coins:
                    if self.state[self.coin_ids[(i, j)]+2]:
                        rep += "C "
                    else:
                        rep += ". "
                elif (i, j) in self.delays:
                    if self.state[-1-self.delay_ids[(i,j)]]:
                        rep += "T "
                    else:
                        rep += ". "
                elif (i, j) in self.gates:
                    rep += 'G '
                elif self.walls[i,j] != 0:
                    rep += '# '
                else:
                    rep += ". "
            rep += '\n'
        return rep
    
    
    def __repr__(self):
        return f'''Object: Gridworld Environment ---
Shape: {self.env_shape}
Episode: {self.current_episode}
State: {self.state}
{self.steps_until_shutdown} steps until shutdown
{self.coins_collected} coins collected'''
    
    
    def display(self, display_values=False, dpi=100):
        draw_gridworld(self, display_values=display_values, dpi=dpi)