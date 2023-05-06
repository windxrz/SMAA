import argparse
import os
import pickle as pkl

import nni
import numpy as np
from tqdm import tqdm

from model.data import Loop
from model.method import (
    SMAA,
    ExploreThenCommit,
    SelfishRobustMMAB,
    SMAARelaxed,
    TotalReward,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=8)
    parser.add_argument("-K", type=int, default=10)
    parser.add_argument("-T", type=int, default=500000)
    parser.add_argument(
        "--dis", type=str, choices=["bernoulli", "beta"], default="beta"
    )

    parser.add_argument(
        "--method",
        choices=[
            "SMAA",
            "ExploreThenCommit",
            "SelfishRobustMMAB",
            "TotalReward",
            "SMAARelaxed",
        ],
        default="SMAA",
    )

    # ExploreThenCommit / TotalReward
    parser.add_argument("--alpha", type=int, default=500)

    # SMAA / SMAARelaxed
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=1e-6)

    # SMAARelaxed
    parser.add_argument("--eta", type=int, default=0.02)

    parser.add_argument("--nni", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.nni:
        my_dict = vars(args)
        optimized_params = nni.get_next_parameter()
        my_dict.update(optimized_params)
        args = argparse.Namespace(**my_dict)

    print(args)

    N, K, T, dis, method = args.N, args.K, args.T, args.dis, args.method
    if method == "ExploreThenCommit":
        method_name = "{}_alpha_{}".format(method, args.alpha)
    elif method == "TotalReward":
        method_name = "{}_alpha_{}".format(method, args.alpha)
    elif method == "SMAA":
        method_name = "{}_beta_{}_tolerance_{}".format(
            method, args.beta, args.tolerance
        )
    elif method == "SMAARelaxed":
        method_name = "{}_eta_{}_beta_{}_tolerance_{}".format(
            method, args.eta, args.beta, args.tolerance
        )
    elif method == "SelfishRobustMMAB":
        method_name = "{}_beta_{}".format(method, args.beta)
    res_path_base = os.path.join(
        "results",
        "N_{}_K_{}_T_{}_dis_{}".format(N, K, T, dis),
        method_name,
    )
    if not os.path.exists(res_path_base):
        os.makedirs(res_path_base)

    total_runs = 100
    pne_nums = []
    regrets_sum = []
    for seed_data in range(total_runs):
        print("Running {}/{}".format(seed_data + 1, total_runs))

        res_file = os.path.join(res_path_base, "{}.pkl".format(seed_data))
        if not os.path.exists(res_file):
            loop = Loop(N, K, T, dis, seed_data)
            print(loop.mu)
            print(loop.pne)
            players = []
            for i in range(args.N):
                if method == "ExploreThenCommit":
                    player = ExploreThenCommit(
                        N, K, T, i, loop, alpha=args.alpha, seed=i
                    )
                elif method == "TotalReward":
                    player = TotalReward(N, K, T, i, loop, alpha=args.alpha, seed=i)
                elif method == "SMAA":
                    player = SMAA(
                        N,
                        K,
                        T,
                        i,
                        loop,
                        beta=args.beta,
                        tolerance=args.tolerance,
                        seed=i,
                    )
                elif method == "SMAARelaxed":
                    player = SMAARelaxed(
                        K,
                        T,
                        loop,
                        tolerance=args.tolerance,
                        eta=args.eta,
                        beta=args.beta,
                        seed=i,
                    )
                elif method == "SelfishRobustMMAB":
                    player = SelfishRobustMMAB(N, K, T, i, loop, beta=args.beta, seed=i)
                players.append(player)

            res_arm_rewards = []
            res_personal_rewards = []
            res_is_pne = []
            res_regrets = []
            for t in tqdm(range(T)):
                choices = []
                for i in range(N):
                    choices.append(players[i].pull(t))
                arm_rewards, personal_rewards, is_pne, regrets = loop.pull(choices, t)
                res_arm_rewards.append(arm_rewards.reshape(1, -1))
                res_personal_rewards.append(personal_rewards.reshape(1, -1))
                res_is_pne.append(is_pne)
                res_regrets.append(regrets.reshape(1, -1))
                for i in range(N):
                    players[i].update(arm_rewards[i], personal_rewards[i], choices)

            res_arm_rewards = np.concatenate(res_arm_rewards, axis=0)

            res_personal_rewards = np.concatenate(res_personal_rewards, axis=0)
            res_personal_rewards = np.cumsum(res_personal_rewards, axis=0)
            res_personal_rewards = res_personal_rewards[::100, :]

            res_is_pne = np.array(res_is_pne)
            res_is_pne = np.cumsum(res_is_pne)
            res_is_pne = res_is_pne[::100]

            res_regrets = np.concatenate(res_regrets, axis=0)
            res_regrets = np.cumsum(res_regrets, axis=0)
            res_regrets = res_regrets[::100]

            res = {
                "personal_rewards": res_personal_rewards,
                "is_pne": res_is_pne,
                "regrets": res_regrets,
            }

            with open(res_file, "wb") as f:
                f.write(pkl.dumps(res))
                f.close()

            for player in players:
                del player
            del loop
        else:
            with open(res_file, "rb") as f:
                res = pkl.loads(f.read())
                f.close()

        print("Is PNE: ", res["is_pne"][-1], "Regrets: ", res["regrets"][-1])

        pne_nums.append(res["is_pne"][-1])
        regrets_sum.append(np.mean(res["regrets"][-1]))
        del res

    report = {"default": T - np.mean(pne_nums), "regret": np.mean(regrets_sum)}
    print(report)

    if args.nni:
        nni.report_final_result(report)


if __name__ == "__main__":
    main()
