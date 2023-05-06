import numpy as np

from model.data import calculate_PNE, set_seed


def kl_divergence(a, b):
    return a * np.log(a / b + 1e-10) + (1 - a) * np.log((1 - a) / (1 - b) + 1e-10)


class SMAA:
    def __init__(self, N, K, T, rank, loop, tolerance, beta, seed=0):
        set_seed(seed)
        self.N = N
        self.K = K
        self.T = T
        self.rank = rank
        self.rewards = np.zeros(self.K)
        self.count = np.zeros(self.K)
        self.tolerance = tolerance
        self.beta = beta
        self.pull_list = []
        self.loop = loop

        self.coin = np.random.rand(T) > 0.5
        self.coin_count = 0

        self.KK = int(np.ceil(self.K / self.N)) * self.N

    def pull(self, t):
        if t < self.K:
            k = t
        elif t < self.KK:
            k = np.random.randint(0, self.K)
        else:
            if t % self.N == 0:
                mu = self.rewards / self.count
                pne, z, _ = calculate_PNE(mu, self.N)
                pne_backup = pne.copy()
                self.pne_list = []
                i = 0
                while i < pne.shape[0]:
                    if pne[i] > 0:
                        self.pne_list.append(i)
                        pne[i] -= 1
                    else:
                        i += 1

                self.candidate = []
                tmp = self.count * kl_divergence(mu, z)
                target = self.beta * (np.log(t + 1) + 4 * np.log(np.log(t + 1)))
                for i in range(self.K):
                    if pne_backup[i] == 0 and tmp[i] <= target:
                        self.candidate.append(i)

                self.explore_idx = -1

                for i in range(self.N):
                    if (
                        np.abs(z * pne_backup[self.pne_list[i]] - mu[self.pne_list[i]])
                        < self.tolerance
                    ):
                        self.explore_idx = i

            idx = (self.rank + t) % self.N
            k = self.pne_list[idx]
            if idx == self.explore_idx and len(self.candidate) > 0:
                if self.coin[self.coin_count]:
                    k = np.random.choice(self.candidate)
                self.coin_count += 1
        self.pull_list.append(k)
        return k

    def update(self, arm_reward, personal_reward, choices):
        k = self.pull_list[-1]
        self.rewards[k] += arm_reward
        self.count[k] += 1


class ExploreThenCommit:
    def __init__(self, N, K, T, rank, loop, alpha, seed=0):
        set_seed(seed)
        self.N = N
        self.K = K
        self.T = T
        self.rank = rank
        self.alpha = alpha
        self.loop = loop

        self.rewards = np.zeros(self.K)
        self.count = np.zeros(self.K)
        self.T0 = self.alpha * np.log(T)
        self.pne_list = None
        self.pull_list = []

    def pull(self, t):
        if t < self.K:
            k = t
        elif t < self.T0:
            k = np.random.randint(0, self.K)
        else:
            if self.pne_list is None:
                pne, _, _ = calculate_PNE(self.rewards / self.count, self.N)
                self.pne_list = []
                i = 0
                while i < pne.shape[0]:
                    if pne[i] > 0:
                        self.pne_list.append(i)
                        pne[i] -= 1
                    else:
                        i += 1
            k = self.pne_list[(self.rank + t) % self.N]
        self.pull_list.append(k)
        return k

    def update(self, arm_reward, personal_reward, choices):
        k = self.pull_list[-1]
        self.rewards[k] += arm_reward
        self.count[k] += 1


class SelfishRobustMMAB:
    def __init__(self, N, K, T, rank, loop, beta, seed=0):
        set_seed(seed)
        self.N = N
        self.K = K
        self.T = T
        self.rank = rank
        self.rewards = np.zeros(self.K)
        self.count = np.zeros(self.K)
        self.beta = beta
        self.pull_list = []
        self.loop = loop

        self.coin = np.random.rand(T) > 0.5
        self.coin_count = 0

        self.KK = int(np.ceil(self.K / self.N)) * self.N

    def pull(self, t):
        if t < self.K:
            k = t
        elif t < self.KK:
            k = np.random.randint(0, self.K)
        else:
            if t % self.N == 0:
                mu = self.rewards / self.count
                idx = np.argsort(mu)[::-1]
                z = mu[idx[self.N - 1]]

                self.candidate = []
                tmp = self.count * kl_divergence(mu, z)
                target = self.beta * (np.log(t + 1) + 4 * np.log(np.log(t + 1)))
                for i in range(self.K):
                    if mu[i] < z and tmp[i] <= target:
                        self.candidate.append(i)

                self.pne_list = idx[: self.N]
                self.explore_idx = self.N - 1

            idx = (self.rank + t) % self.N
            k = self.pne_list[idx]
            if idx == self.explore_idx and len(self.candidate) > 0:
                if self.coin[self.coin_count]:
                    k = np.random.choice(self.candidate)
                self.coin_count += 1
        self.pull_list.append(k)
        return k

    def update(self, arm_reward, personal_reward, choices):
        k = self.pull_list[-1]
        self.rewards[k] += arm_reward
        self.count[k] += 1


class TotalReward:
    def __init__(self, N, K, T, rank, loop, alpha, seed=0):
        self.N = N
        self.K = K
        self.T = T
        self.rank = rank
        self.alpha = alpha
        self.loop = loop

        self.rewards = np.zeros(self.K)
        self.count = np.zeros(self.K)
        self.T0 = self.alpha * np.log(T)
        self.pne_list = None
        self.pull_list = []
        set_seed(seed)

    def pull(self, t):
        if t < self.K:
            k = t
        elif t < self.T0:
            k = np.random.randint(0, self.K)
        else:
            if self.pne_list is None:
                if self.K >= self.N:
                    mu = self.rewards / self.count
                    idx = np.argsort(mu)[::-1]
                    self.pne_list = idx[: self.N]
                else:
                    mu = self.rewards / self.count
                    pne = [1] * self.K
                    pne[0] += self.N - self.K
                    flag = True
                    while flag:
                        flag = False
                        for i in range(self.K):
                            if pne[i] == 1:
                                continue
                            for j in range(self.K):
                                if i == j:
                                    continue
                                if mu[j] / (pne[j] + 1) > mu[i] / pne[i]:
                                    pne[i] -= 1
                                    pne[j] += 1
                                    flag = True
                                    break
                            if flag:
                                break
                    self.pne_list = []
                    i = 0
                    while i < self.K:
                        if pne[i] > 0:
                            self.pne_list.append(i)
                            pne[i] -= 1
                        else:
                            i += 1
                print(self.pne_list)
            k = self.pne_list[(self.rank + t) % self.N]
        self.pull_list.append(k)
        return k

    def update(self, arm_reward, personal_reward, choices):
        k = self.pull_list[-1]
        self.rewards[k] += arm_reward
        self.count[k] += 1


class SMAARelaxed:
    def __init__(self, K, T, loop, tolerance, eta, beta, seed=0):
        set_seed(seed)
        self.K = K
        self.T = T
        self.rewards = np.zeros(self.K)
        self.count = np.zeros(self.K)
        self.tolerance = tolerance
        self.eta = eta
        self.beta = beta
        self.pull_list = []
        self.loop = loop
        self.t0 = int(np.ceil(self.eta * 50 * K * K * np.log(4 * T)))
        self.t1 = 0
        self.KK = 0

        self.rank = -1

        self.count_collision = 0

        self.coin = np.random.rand(T) > 0.5
        self.coin_count = 0

    def pull(self, t):
        if t == self.t0:
            self.N = int(
                np.round(
                    np.log((self.t0 - self.count_collision) / self.t0)
                    / np.log(1 - 1 / self.K)
                    + 1
                )
            )
            if self.N > self.K:
                self.N = self.K
            self.t1 = int(np.ceil(self.N * np.log(2 * self.T)))
            self.KK = int(np.ceil((self.K + self.t0 + self.t1) / self.N)) * self.N
            print("Estimated N: ", self.N)
        if t < self.t0:
            k = np.random.randint(0, self.K)
        elif t < self.t0 + self.t1:
            if self.rank == -1:
                k = np.random.randint(0, self.N)
            else:
                k = self.rank
        elif t < self.t0 + self.t1 + self.K:
            if t == self.t0 + self.t1:
                print("Rank: ", self.rank)
            if self.rank == -1:
                self.rank = 0
            k = t - self.t0 - self.t1
        elif t < self.KK:
            k = np.random.randint(0, self.K)
        else:
            if t % self.N == 0:
                mu = self.rewards / self.count
                pne, z, _ = calculate_PNE(mu, self.N)
                pne_backup = pne.copy()
                self.pne_list = []
                i = 0
                while i < pne.shape[0]:
                    if pne[i] > 0:
                        self.pne_list.append(i)
                        pne[i] -= 1
                    else:
                        i += 1

                self.candidate = []
                tmp = self.count * kl_divergence(mu, z)
                target = self.beta * (np.log(t + 1) + 4 * np.log(np.log(t + 1)))
                for i in range(self.K):
                    if pne_backup[i] == 0 and tmp[i] <= target:
                        self.candidate.append(i)

                self.explore_idx = -1

                for i in range(self.N):
                    if (
                        np.abs(z * pne_backup[self.pne_list[i]] - mu[self.pne_list[i]])
                        < self.tolerance
                    ):
                        self.explore_idx = i

            idx = (self.rank + t) % self.N
            k = self.pne_list[idx]
            if idx == self.explore_idx and len(self.candidate) > 0:
                if self.coin[self.coin_count]:
                    k = np.random.choice(self.candidate)
                self.coin_count += 1
        self.pull_list.append(k)
        self.timestamp = t
        return k

    def update(self, arm_reward, personal_reward, choices):
        k = self.pull_list[-1]
        choices = np.array(choices)
        if np.sum(choices == k) > 1:
            self.count_collision += 1
        else:
            if self.timestamp >= self.t0 and self.rank == -1:
                self.rank = k
        self.rewards[k] += arm_reward
        self.count[k] += 1
