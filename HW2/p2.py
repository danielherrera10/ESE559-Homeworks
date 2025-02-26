import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
# initialize 4x4 grid
def init_grid():
    grid = []
    width = 4
    for i in range(1, width+1):
        for j in range(1, width+1):
            grid.append((i, j))
    return grid

#boundary check for state s
def in_bounds(s):
    x,y = s
    if x > 4 or x < 1 or y > 4 or y < 1:
        return False
    return True

#compute neighbors of state s
def neighbors(s):
    x,y = s
    S = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    return [s for s in S if in_bounds(s) and s not in obstacles]

#returns the action required to reach s_prime from s
def action(s,s_prime):
    x,y = s
    x_p, y_p = s_prime
    dx = x_p - x
    dy = y_p - y
    if dx == 1 and dy == 0:
        return 'right'
    elif dx == -1 and dy == 0:
        return 'left'
    elif dx == 0 and dy == 1:
        return 'up'
    elif dx == 0 and dy == -1:
        return 'down'
    elif dx == 0 and dy == 0:
        return 'idle'
    else:
        return None

def available_actions(s):
    available_a = []
    nbs = neighbors(s)
    for n in nbs:
        available_a.append(action(s,n))
    return available_a
    

#transition probability function for taking action a from state s to s_prime
def prob(s, a, s_prime, obstacles):
    if in_bounds(s_prime) == False or s_prime in obstacles or s in terminal_states:
        return 0
    elif a == 'idle':
        return 1
    # probability of going to correct state
    elif a == action(s, s_prime):
        return 0.7
    else:
        return 0.3/len(neighbors(s))
    
#MDP reward function calcates the reward for arriving at state s
#just returns -1 for idle action and any nonterminal state
def reward(s, terminal_states, obstacles):
    if s in terminal_states:
        return 10
    elif s in obstacles:
        return -10
    else:
        return -1

def policy_evaluation(states, obstacles, terminal_states, policy, V, theta, gamma):
    while True:
        delta = 0  # Convergence check

        for s in states:
            if s in obstacles:
                continue
            elif s in terminal_states:
                V[s] = reward(s, terminal_states, obstacles)
                continue
            
            u = V[s]  # Store old value for convergence check
            e_val = 0  

            # Sum over actions using stochastic policy pi(s, a)
            for a, pi_sa in policy[s].items():
                action_val = 0
                for s_prime in neighbors(s) + [s]:  # Include idle state
                    p = prob(s, a, s_prime, obstacles)
                    r = reward(s_prime, terminal_states, obstacles)
                    action_val += p * (r + gamma * V[s_prime])

                e_val += pi_sa * action_val  # Weighted by policy probability

            V[s] = e_val
            delta = max(delta, abs(u - V[s]))  # Track max change

        if delta < theta:
            break  # Stop if change is small

    return V


def policy_iteration(states, terminal_states, obstacles, gamma=0.9, theta=1e-3):
    # Initialize value function and random stochastic policy
    V = {s: 0 for s in states}
    policy = {s: {a: 1/len(available_actions(s)) for a in available_actions(s)}
              for s in states if s not in terminal_states and s not in obstacles}

    while True:
        V = policy_evaluation(states, obstacles, terminal_states, policy, V, theta, gamma)

        # Policy Improvement Step
        stable = True  # Check if policy changed

        for s in states:
            if s in obstacles or s in terminal_states:
                continue

            old_action = max(policy[s], key=policy[s].get)  # Get highest-probability action
            best_action = None
            best_value = float('-inf')

            # Find the best action using Bellman equation
            for a in available_actions(s):
                action_val = 0
                for s_prime in neighbors(s) + [s]:
                    p = prob(s, a, s_prime, obstacles)
                    r = reward(s_prime, terminal_states, obstacles)
                    action_val += p * (r + gamma * V[s_prime])

                if action_val > best_value:
                    best_value = action_val
                    best_action = a

            # Update policy to be fully deterministic
            policy[s] = {a: 1.0 if a == best_action else 0.0 for a in available_actions(s)}

            if old_action != best_action:
                stable = False  # Policy changed, must keep iterating

        if stable:
            break  # Stop when policy stops changing

    return policy



states = init_grid()
terminal_states = [(1,4),(4, 4)]  # Example terminal state
obstacles = [(3, 3)]  # Example obstacles
gamma = 0.8
theta = 1e-4

start = time.time()
optimal_policy = policy_iteration(states,terminal_states, obstacles, gamma, theta)
end = time.time()
runtime = end-start
print("Runtime: " + str(runtime) + " seconds")
# print(optimal_policy)

fig = plt.figure(1)
ax = fig.add_subplot(111)
term_green = Rectangle((4,4), 1, 1, color = 'green')
term_blue = Rectangle((1,4), 1, 1, color = 'blue')
term_red = Rectangle((3,3), 1,1,color='red')
ax.add_patch(term_blue)
ax.add_patch(term_green)
ax.add_patch(term_red)
plt.xlim(1,5)
plt.ylim(1,5)
ax.set_xticks(range(1,5))  # Ensure grid lines at 0,1,2,3,4
ax.set_yticks(range(1,5))
ax.set_aspect('equal')  # Keep squares properly scaled
plt.grid(which='major')

#plot policy arrows
plot_policy = []
for s, actions in optimal_policy.items():
    for action, prob in actions.items():
        if prob == 1.0:
            x,y = s
            if action == 'up':
                ax.arrow(x+0.5, y+0.2, 0, 0.6, head_width=0.1, head_length=0.1, fc='black', ec='black')
            elif action == 'down':
                ax.arrow(x+0.5, y+0.8, 0, -0.4, head_width=0.1, head_length=0.1, fc='black', ec='black')
            elif action == 'left':
                ax.arrow(x+0.8, y+0.5, -0.6, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
            elif action == 'right':
                ax.arrow(x+0.2, y+0.5, 0.6, 0, head_width=0.1, head_length=0.1, fc='black', ec='black') 
             
plt.title("Optimal policy for each state")
plt.show()






    


