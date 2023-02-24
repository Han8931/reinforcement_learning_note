import numpy as np
import gym

import pdb

class FoodTruck(gym.Env):
    def __init__(self):
        # demand: the amount of sold burgers
        self.v_demand = [100, 200, 300, 400]
        self.p_demand = [0.3, 0.4, 0.2, 0.1] # Mon-Fri
        self.capacity = self.v_demand[-1] 
        self.days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', "Weekend"]

        # We want to maximize fn=(net_revenue-unit_cost)
        self.unit_cost = 4 # Cost of food
        self.net_revenue = 7 # Per burger
        self.action_space = [0, 100, 200, 300, 400] # Number of patties to purchase
        # Monday inventory is 0
        # Generate all possible states to construct a fully observable state space
        # Possible inventory levels are [0-300]
        # ('Mon', 0): 0, ('Tue', 0): 0, ('Tue', 100): 0, ('Tue', 200): 0, ('Tue', 300): 0, ('Wed', 0): 0, ('Wed', 100)...
        self.state_space = [("Mon", 0)] + [(d, i) for d in self.days[1:] for i in [0, 100, 200, 300]]

    def get_next_state_reward(self, state, action, demand):
        day, inventory = state

        result = {}
        result['next_day'] = self.days[self.days.index(day) + 1]

        # Maximum is self.capacity
        result['starting_inventory'] = min(self.capacity, inventory + action) 
        result['cost'] = self.unit_cost * action # Total Price

        # Cannot sell more than the amount of inventory
        result['sales'] = min(result['starting_inventory'], demand) 

        result['revenue'] = self.net_revenue * result['sales']
        result['next_inventory'] = result['starting_inventory'] - result['sales']
        result['reward'] = result['revenue'] - result['cost']

        return result

    def get_transition_prob(self, state, action):
        "= p[s',r|s,a]"
        next_s_r_prob = {}
        for ix, demand in enumerate(self.v_demand):
            result = self.get_next_state_reward(state, action, demand)
            next_s = (result['next_day'], result['next_inventory'])
            reward = result['reward']
            prob = self.p_demand[ix] # I am not sure why this one is used as a transition prob here
            if (next_s, reward) not in next_s_r_prob:
                next_s_r_prob[next_s, reward] = prob
            else:
                next_s_r_prob[next_s, reward] += prob # ??
        return next_s_r_prob

    def reset(self):
        self.day = "Mon"
        self.inventory = 0
        state = (self.day, self.inventory)
        return state

    def is_terminal(self, state):
        day, inventory = state
        if day == "Weekend":
            return True
        else:
            return False

    def step(self, action):
        demand = np.random.choice(self.v_demand, p=self.p_demand)
        result = self.get_next_state_reward((self.day, self.inventory), action, demand)
        self.day = result['next_day']
        self.inventory = result['next_inventory']
        state = (self.day, self.inventory)
        reward = result['reward']
        done = self.is_terminal(state)
        info = {'demand': demand, 'sales': result['sales']}
        return state, reward, done, info


if __name__ =="__main__":

    # Simulating an arbitrary policy
    np.random.seed(0)
    foodtruck = FoodTruck()
    rewards = []
    for i_episode in range(10000):
        state = foodtruck.reset()
        done = False
        ep_reward = 0
        while not done:
            day, inventory = state
            action = max(0, 300 - inventory)
            state, reward, done, info = foodtruck.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    print(f"Avg Rewards: {np.mean(rewards)}") 

    # Single day expected reward
    ucost = 4
    uprice = 7
    v_demand = [100, 200, 300, 400]
    p_demand = [0.3, 0.4, 0.2, 0.1]
    inv = 400

    # uprice*np.sum([p_demand[i]*min(v_demand[i], inv) for i in range(4)]): profit by selling patties
    # inv*ucost: cost of purchasing patties
    profit = uprice*np.sum([p_demand[i]*min(v_demand[i], inv) for i in range(4)]) - inv*ucost
    print(f"Profit: {profit}") 


