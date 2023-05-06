import random

import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def calculate_PNE(mu, N):
    mu_tmp = mu.reshape(-1, 1)
    divide = 1.0 / np.arange(1, N + 1)
    mul = mu_tmp * divide
    ind = np.unravel_index(np.argsort(mul, axis=None), mul.shape)
    max_ = mu[ind[0][-N]] / (ind[1][-N] + 1)
    tmp = np.floor((mu + 1e-6) / max_).astype(int)
    if np.sum(tmp) == N:
        return tmp, max_, True
    else:
        s = np.sum(tmp) - N
        for i in range(tmp.shape[0]):
            if np.abs(max_ * tmp[i] - mu[i]) < 1e-6:
                tmp[i] = tmp[i] - 1
                s -= 1
            if s == 0:
                break
        return tmp, max_, False


class Loop:
    def __init__(self, N, K, T, dis="beta", seed=None):
        set_seed(seed)
        self.N = N
        self.K = K
        self.T = T
        self.dis = dis

        while True:
            if self.dis == "beta":
                self.alpha = np.random.rand(K) * 5
                self.beta = np.random.rand(K) * 5
                self.rewards = np.random.beta(self.alpha, self.beta, (T, K))
                self.mu = self.alpha / (self.alpha + self.beta)
            elif self.dis == "bernoulli":
                self.mu = np.random.rand(K)
                self.rewards = np.random.binomial(1, self.mu, (T, K))

            self.weights = np.random.rand(self.T, self.N)

            self.pne, _, flag = calculate_PNE(self.mu, self.N)
            if flag:
                break

    def pull(self, choices, t):
        m = np.zeros(self.K, dtype=int)
        weight = np.zeros(self.K)
        for i, choice in enumerate(choices):
            m[choice] += 1
            weight[choice] += self.weights[t][i]
        is_pne = np.all(m == self.pne).astype(int)
        arm_rewards = self.rewards[t][choices]

        tmp = self.mu / (m + 1)
        max_deviation = np.max(tmp)
        raw = self.mu[choices] / m[choices]
        regrets = np.maximum(0, max_deviation - raw)

        personal_rewards = (self.rewards[t] / (weight + 1e-6))[choices]
        personal_rewards = personal_rewards * self.weights[t]
        regrets = np.array(regrets)
        return arm_rewards, personal_rewards, is_pne, regrets
