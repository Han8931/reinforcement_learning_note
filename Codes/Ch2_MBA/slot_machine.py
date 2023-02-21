import numpy as np

class GaussianBandit:
    def __init__(self, mean=0, stdev=1):
        self.mean = mean
        self.stdev = stdev

    def pull_lever(self):
        reward = np.random.normal(self.mean, self.stdev)
        return np.round(reward,1)

class GaussianBanditGame:
    def __init__(self, bandits):
        self.bandits = bandits
        np.random.shuffle(self.bandits)
        self.reset_game()

    def play(self, choice):
        reward = self.bandits[choice-1].pull_lever()
        self.rewards.append(reward)
        self.total_reward+=reward
        self.n_played += 1
        return reward

    def user_play(self):
        self.reset_game()
        print("Game Started. Enter 0 as input to end the game.")

        while True:
            print(f"\n -- Round {self.n_played}")
            choice = int(input(f"Choose a machine " +
                     f"from 1 to {len(self.bandits)}: "))
            if choice in range(1, len(self.bandits) + 1):
                reward = self.play(choice)
                print(f"Machine {choice} gave " +
                      f"a reward of {reward}.")
                avg_rew = self.total_reward/self.n_played
                print(f"Your average reward " +
                      f"so far is {avg_rew}.")
            else:
                break
        print("Game has ended.")
        if self.n_played > 0:
            print(f"Total reward is {self.total_reward}" +
                  f" after {self.n_played} round(s).")
            avg_rew = self.total_reward/self.n_played
            print(f"Average reward is {avg_rew}.")

    def reset_game(self):
        self.rewards = []
        self.total_reward = 0
        self.n_played = 0


if __name__ =="__main__":
    slotA = GaussianBandit(5, 3)
    slotB = GaussianBandit(6, 2)
    slotC = GaussianBandit(1, 5)
    game = GaussianBanditGame([slotA, slotB, slotC])
    game.user_play()

