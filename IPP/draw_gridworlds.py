import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# These colors were chosen to exactly the AI Safety Gridworlds Paper from Deepmind
COLORS = {
    'agent'  : '#00b4fe', # Bright-Blue
    'coin'   : 'gold'   ,
    'button' : '#a587ee', # Purple: I made it lighter, it was origionally '#6d45d1'
    'border' : '#777777'   , #'#989898',
    'wall'   : '#777777'   , #'#989898',
    'empty'  : '#dadada', # Light-Gray
    'lava'   : 'red'    , 
    'green'  : '#01d131', # Green
    'hotpink': '#fe1ffe'
}


def draw_square(x, y, color, text=None, value=None, 
                margin=0.03, fig_scale=1, vertical_offset=4, fontsize=26):
    x, y = y, -1-x # transform coordinates to make it resemble imshow
    
    rect = patches.Rectangle((x+margin,y+margin), 1-2*margin, 1-2*margin,
                            color=color)
    plt.gca().add_patch(rect)
    if value:
        text += str(value)
        fontsize -= 5*(len(str(value))-1)
    if text:
        plt.text(x+0.5, y+0.5, text, fontsize=fontsize*fig_scale, 
                 fontproperties={'family':'monospace'},
                 ha='center', va='center_baseline')
    # if value:
    #     plt.text(x+0.8,y+0.2, value, fontsize=14*fig_scale, 
    #              fontproperties={'family':'monospace'},
    #              ha='center', va='center_baseline')
        
        
def draw_full_policy(env, agent):
    
    rows = 2 ** (env.n_flags // 2)
    cols = 2 ** (env.n_flags - (env.n_flags // 2))

    plt.figure(figsize=(4*cols, 4*rows), dpi=200)
    for i in range(2**env.n_flags):
        flags = [int(f) for f in bin(i)[2:].zfill(env.n_flags)]
        plt.subplot(rows, cols, i+1)
        draw_policy(env, agent, flags, subplot_mode=True)
        
        
def draw_policy(env, agent, flags=None, subplot_mode=False):
    if not flags:
        flags=[0]*(len(env.coins)+len(env.delays))
    draw_gridworld(env, flags=flags, subplot_mode=subplot_mode)
    index_tuple = tuple([slice(None)]*2 + list(flags))
    e_x = np.exp(agent.logits)
    probs = e_x / e_x.sum(axis=-1, keepdims=True)
    draw_arrows(env.walls, probs[index_tuple])
        

def draw_gridworld(env, state=None, flags=None, steps_remaining=None,
                   margin=0.025, fig_scale=0.74, dpi=100, 
                   subplot_mode=False, display_labels=True, display_values=False):
    '''For the purpose of visualization you may sometimes wish to override the 
    env state and steps_until_shutdown, to do this pass these as arugments.
    '''
    if not state:
        # If no state is provided use the state stored in env
        state = env.state
        
    if flags:
        # If flags are passed we assume that the agent should not be drawn
        state = tuple((0,0) + tuple(flags))
        
    if not steps_remaining:
        # If steps_remaining is not passed use the one stored in env
        steps_remaining = env.steps_until_shutdown

    # Split up state
    agent_pos = state[:2]
    coin_flags = state[2:2+len(env.coins)]
    delay_flags = state[-len(env.delays):]
    
    if not subplot_mode:
        plt.figure(figsize=[s*fig_scale for s in env.env_shape[::-1]], dpi=dpi)

    # Draw border
    for i in range(-1,env.env_shape[0]+1):
        for j in range(-1,env.env_shape[1]+1):
            if 0<=i<env.env_shape[0] and 0<=j<env.env_shape[1]:
                color = COLORS['wall'] if env.walls[i,j] else COLORS['empty']
                draw_square(i, j, color, 
                            margin=margin, fig_scale=fig_scale,
                            vertical_offset=env.env_shape[0]-1)
            else:
                draw_square(i, j, COLORS['border'], 
                            margin=margin, fig_scale=fig_scale,
                            vertical_offset=env.env_shape[0]-1)

    # Draw coins
    for coin_pos, coin_val, flag in zip(env.coins.keys(), env.coins.values(), coin_flags):
        if not display_values: 
            coin_val = None
        if flag:
            draw_square(*coin_pos, COLORS['coin'], 'C' if display_labels else '', 
                        margin=margin, fig_scale=fig_scale,
                        vertical_offset=env.env_shape[0]-1, value=coin_val)

    # Draw delay buttons
    for button_pos, delay_val, flag in zip(env.delays.keys(), env.delays.values(), delay_flags):
        if not display_values: 
            delay_val = None
        if flag:
            draw_square(*button_pos, COLORS['button'], 'B' if display_labels else '', 
                        margin=margin, fig_scale=fig_scale,
                        vertical_offset=env.env_shape[0]-1, value=delay_val)

    # Draw Agent
    if not flags:
        draw_square(*agent_pos, COLORS['agent'], 'A' if display_labels else '', 
                    margin=margin, fig_scale=fig_scale,
                    vertical_offset=env.env_shape[0]-1)
    
    if subplot_mode:
        plt.title(f'State = {state[2:]}')
    # else:
    #     plt.title(f'Steps until shutdown: {steps_remaining}')
        
    plt.text(env.env_shape[1]+0.5, -env.env_shape[0]-0.5, str(steps_remaining), fontsize=26*fig_scale, 
             color='white',
             fontproperties={'family':'monospace'},
             ha='center', va='center_baseline')

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.xlim(-1, env.env_shape[1]+1)
    plt.ylim(-env.env_shape[0]-1, 1)
    plt.axis('off')
    
    
def draw_arrows(walls, policy_table):
    """Given a walls array and a policy_table array of shape (*walls.shape, 4)
    This visually represents the probability of each action at each state 
    by drawning arrows, whose opacity corresponds to the respective probability.
    """

    # For each possible agent position
    all_positions = np.stack(np.unravel_index(range(np.prod(walls.shape)), walls.shape)).T
    for pos in all_positions:
        
        # Skip plotting if in walls:
        if walls[tuple(pos)]:
            continue
            
        action_probs = policy_table[tuple(pos)]
        
        pos = pos[1]+0.5, -pos[0]-0.5

        # Draw arrows:
        color = 'r'
        
        # Up
        plt.arrow(*pos, 0, .12, head_width=.25, lw=3, 
                  ec='none', fc=color, alpha=float(action_probs[0]))
        # Down
        plt.arrow(*pos, 0, -.12, head_width=.25, lw=3, 
                  ec='none', fc=color, alpha=float(action_probs[1]))
        # Left
        plt.arrow(*pos, -.12, 0, head_width=.25, lw=3, 
                  ec='none', fc=color, alpha=float(action_probs[2]))
        # Right
        plt.arrow(*pos, .12, 0, head_width=.25, lw=3, 
                  ec='none', fc=color, alpha=float(action_probs[3]))

    plt.axis('off')
