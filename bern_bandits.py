import numpy as np
import matplotlib.pyplot as plt

"""Plotting Util"""
def plot_regret(instant_regret, plot_type):
    """
    Args
      instant_regret: (list) instant regret
      plot_type: (string) type of regret to be plotted
    Out
      Shows plt plot
    """
    if plot_type not in ['instant','cumulative','avg_cum']:
        raise ValueError('Regret plot type incorrect')
    timesteps = range(1,len(instant_regret)+1)
    if plot_type == 'instant':
        plt.plot(timesteps,instant_regret, '-bo')
        plt.xlabel('Timesteps')
        plt.ylabel('Instant regret')
    elif plot_type == 'cumulative':
        cum_regret = np.cumsum(instant_regret)
        log_cum_regret = np.log(cum_regret)
        plt.plot(timesteps, log_cum_regret, '-bo')
        plt.xlabel('Timesteps')
        plt.ylabel('Cumulative regret')
    elif plot_type == 'avg_cum':
        avg_cum_regret = np.cumsum(instant_regret)/np.array(timesteps)
        log_avg_cum_regret = np.log(avg_cum_regret)
        plt.plot(timesteps, log_avg_cum_regret, '-bo')
        plt.xlabel('Timesteps')
        plt.ylabel('Average Cumulative regret')

    plt.show()


class DiscreteBandits(object):
    def __init__(self, num_arms, prior_distr_type, init_pulls):
        self.num_arms = num_arms
        self.prior_distr_type = prior_distr_type
        self.query_history = []

        # true means of each arm
        self.true_arm_means = None
        # number of times each arm was pulled
        self.arm_pulls = np.zeros(self.num_arms)
        # cumulative rewards received from each arm
        self.arm_rewards = np.zeros(self.num_arms)

        # idx of true best arm
        self.best_arm = None
        # mean of true best arm
        self.best_arm_mean = None

        # minimum number of trials to pull each arm
        self.init_pulls = init_pulls

        self.instant_regret = []


class BernoulliBandits(DiscreteBandits):
    def __init__(self, num_arms, prior_distr_type, init_pulls=1):
        super(BernoulliBandits, self).__init__(num_arms, prior_distr_type,
                                               init_pulls)
        self._setup_priors()
        self._setup_arms()

    def _setup_arms(self):
        np.random.seed(1234)
        self.true_arm_means = np.random.rand(self.num_arms)
        self.best_arm = np.argmax(self.true_arm_means)
        self.best_arm_mean = np.max(self.true_arm_means)

    def _setup_priors(self):
        self.distr_params = {}
        if self.prior_distr_type == 'beta':
            self.distr_params['alpha'] = []
            self.distr_params['beta'] = []
            for each_arm in range(self.num_arms):
                default_alpha = 1
                default_beta = 1
                self.distr_params['alpha'].append(default_alpha)
                self.distr_params['beta'].append(default_beta)
        else:
            raise ValueError('Are you sure you want to use non-conjugate prior?')

    def _update_distr(self, arm_pulled, reward_from_arm):

        self.distr_params['alpha'][arm_pulled] += reward_from_arm
        self.distr_params['beta'][arm_pulled] += (1-reward_from_arm)

    def _post_arm_pull(self, arm_idx, current_reward):
        current_regret = self.best_arm_mean - self.true_arm_means[arm_idx]
        self.arm_pulls[arm_idx] += 1
        self.arm_rewards[arm_idx] += current_reward
        self.query_history.append({'arm':arm_idx, 'reward':current_reward})
        self.instant_regret.append(current_regret)

    def pull_arm_TS(self):
        if self.arm_pulls[0] < self.init_pulls:
            self._init_pulls()

        thompson_samples = np.random.beta(a=self.distr_params['alpha'],
                                          b=self.distr_params['beta'])
        arm_idx = np.argmax(thompson_samples)
        current_reward = np.random.binomial(1,self.true_arm_means[arm_idx],1)[0]

        self._post_arm_pull(arm_idx, current_reward)
        self._update_distr(arm_idx,current_reward)


    def _init_pulls(self):
        for _ in range(self.init_pulls):
            for arm_idx in range(self.num_arms):
                current_reward = np.random.binomial(1,self.true_arm_means[arm_idx],1)[0]

                # self._post_arm_pull(arm_idx, current_reward)
                current_regret = self.best_arm_mean - self.true_arm_means[arm_idx]
                self.arm_pulls[arm_idx] += 1
                self.arm_rewards[arm_idx] += current_reward

                self._update_distr(arm_idx, current_reward)


    def pull_arm_bayesian_greedy(self):
        if self.arm_pulls[0] < self.init_pulls:
            self._init_pulls()

        current_mean_est = np.array(self.distr_params['alpha'],dtype=np.float32)/ \
                           (np.array(self.distr_params['alpha'],dtype=np.float32)+
                           np.array(self.distr_params['beta'],dtype=np.float32))
        arm_idx = np.argmax(current_mean_est)
        current_reward = np.random.binomial(1, self.true_arm_means[arm_idx],1)[0]

        self._post_arm_pull(arm_idx, current_reward)
        self._update_distr(arm_idx, current_reward)


    def pull_arm_freq_greedy(self):
        if self.arm_pulls[0] < self.init_pulls:
            self._init_pulls()

        current_mean_est = np.array(self.arm_rewards)/np.array(self.arm_pulls)
        arm_idx = np.argmax(current_mean_est)
        current_reward = np.random.binomial(1,self.true_arm_means[arm_idx],1)[0]

        self._post_arm_pull(arm_idx, current_reward)
        self._update_distr(arm_idx,current_reward)

    def pull_arm_UCB1(self):
        if self.arm_pulls[0] < self.init_pulls:
            self._init_pulls()

        # import pdb; pdb.set_trace()

        current_timestep = np.sum(self.arm_pulls) + 1
        current_UCB = np.array(self.arm_rewards)/np.array(self.arm_pulls) + \
                      np.sqrt((2*np.log(current_timestep))/self.arm_pulls)
        arm_idx = np.argmax(current_UCB)
        current_reward = np.random.binomial(1,self.true_arm_means[arm_idx],1)[0]

        self._post_arm_pull(arm_idx, current_reward)
        self._update_distr(arm_idx,current_reward)

def main():
    ucb_bandit = BernoulliBandits(10,'beta',1)
    ts_bandit = BernoulliBandits(10,'beta',1)
    freq_greedy_bandit = BernoulliBandits(10,'beta',1)
    bayes_greedy_bandit = BernoulliBandits(10,'beta',1)
    for i in range(100):
        # test_bern_bandit.pull_arm_freq_greedy()
        # bayes_greedy_bandit.pull_arm_bayesian_greedy()
        # test_bern_bandit.pull_arm_TS()
        ucb_bandit.pull_arm_UCB1()
        ts_bandit.pull_arm_TS()
        freq_greedy_bandit.pull_arm_freq_greedy()
        bayes_greedy_bandit.pull_arm_bayesian_greedy()

    print(ucb_bandit.true_arm_means)
    # print(np.array(ucb_bandit.arm_pulls))
    plot_regret(ucb_bandit.instant_regret,'cumulative')
    plot_regret(ts_bandit.instant_regret,'cumulative')
    plot_regret(freq_greedy_bandit.instant_regret,'cumulative')
    plot_regret(bayes_greedy_bandit.instant_regret,'cumulative')


if __name__ == '__main__':
    main()
