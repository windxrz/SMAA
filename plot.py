import os
import pickle as pkl

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc


rc("font", **{"family": "sans-serif", "sans-serif": ["Times New Roman"]})

# rc("text", usetex=True)

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42

COUNT = 100

LINEWIDTH = 3
MARKEREDGEWIDTH = 2
MS = 20
FONTSIZE = 22
LEGEND_FONSIZE = 20
LABEL_FONTSIZE = 25
MARKERSIZE = 10
TICKFONTSIZE = 18

COLOR_LIST = [
    "#C9625E",
    "#CBBD62",
    "#8ECB62",
    "#62CBC7",
    "#CD6ACA",
    "#000000",
    "#AAAAAA",
    "#9F10A3",
    "#BD2C14",
    "#0E5AD3",
    "#22AB16",
]

COLOR_DARK = {
    "SelfishRobustMMAB": "#DBAF63",
    "TotalReward": "#543466",
    "ExploreThenCommit": "#22613D",
    "SMAA": "#B03E3A",
}
COLOR = {
    "SelfishRobustMMAB": "#FB9649",
    "TotalReward": "#605CB8",
    "ExploreThenCommit": "#53C292",
    "SMAA": "#E64640",
}
MARKER = {
    "SelfishRobustMMAB": "^",
    "TotalReward": "s",
    "ExploreThenCommit": "p",
    "SMAA": "o",
}

COLOR_RELAXED = ["#4A0F0F", "#870300", "#D61010", "#E3784D", "#F6BDC0"]
MARKER_RELAXED = ["^", "s", "p", "o", "+", "H", "P", "."]


def setting_path(N, K, T, dis):
    return "N_{}_K_{}_T_{}_dis_{}".format(N, K, T, dis)


def analyze_method_run(setting, method):
    res_path = os.path.join("results", setting, method)

    if not os.path.exists(os.path.join(res_path, "final.pkl")):
        final = {"personal_rewards": None, "is_pne": None, "regrets": None}
        count = 0
        for filename in sorted(os.listdir(res_path)):
            with open(os.path.join(res_path, filename), "rb") as f:
                res = pkl.loads(f.read())
                f.close()
            count += 1
            if final["personal_rewards"] is None:
                final["personal_rewards"] = res["personal_rewards"][np.newaxis, :]
                final["is_pne"] = res["is_pne"][np.newaxis, :]
                final["regrets"] = res["regrets"][np.newaxis, :]
            else:
                final["personal_rewards"] = np.concatenate(
                    [final["personal_rewards"], res["personal_rewards"][np.newaxis, :]],
                    axis=0,
                )
                final["is_pne"] = np.concatenate(
                    [final["is_pne"], res["is_pne"][np.newaxis, :]],
                    axis=0,
                )
                final["regrets"] = np.concatenate(
                    [final["regrets"], res["regrets"][np.newaxis, :]],
                    axis=0,
                )
            if count == COUNT:
                break

        if count == 0:
            return 0, None

        final["regrets"] = np.mean(final["regrets"], axis=2)

        final["regrets_std"] = np.std(final["regrets"], axis=0)
        final["is_pne_std"] = np.std(final["is_pne"], axis=0)

        final["regrets"] = np.mean(final["regrets"], axis=0)
        final["is_pne"] = np.mean(final["is_pne"], axis=0)

        if count == COUNT:
            with open(os.path.join(res_path, "final.pkl"), "wb") as f:
                f.write(pkl.dumps(final))
                f.close()
    else:
        with open(os.path.join(res_path, "final.pkl"), "rb") as f:
            final = pkl.loads(f.read())
            f.close()
        count = COUNT

    print(setting, method, count, final["is_pne"][-1], final["regrets"][-1])

    return count, final


def analyze_method(setting, method):
    is_pne_max = 0
    regret_min = 1e9
    final = None
    for run in sorted(os.listdir(os.path.join("results", setting))):
        if method + "_" not in run:
            continue
        count, res = analyze_method_run(setting, run)
        if count == 0:
            continue
        if count < COUNT // 10 * 5:
            continue
        # if res["regrets"][-1] < regret_min:
        #     regret_min = res["regrets"][-1]
        #     final = res.copy()
        if res["is_pne"][-1] > is_pne_max:
            is_pne_max = res["is_pne"][-1]
            final = res.copy()
    if final is not None:
        print(setting, method, "best", final["is_pne"][-1], final["regrets"][-1])
    return final


def plot_part(N, K, T, dis, ax1, ax2):
    setting_name = setting_path(N, K, T, dis)

    step = 100

    if not os.path.exists(os.path.join("results", setting_name)):
        return

    for method in COLOR.keys():
        if "Relaxed" in method:
            continue
        res = analyze_method(setting_name, method)
        if res is None:
            continue

        method_name = method.split("_")[0]
        label_name = method_name.replace("Ours", "SMAA (Ours)")

        ax1.plot(
            range(1, T + 1, step),
            res["regrets"],
            label=label_name,
            color=COLOR[method_name],
            linewidth=LINEWIDTH,
            marker=MARKER[method_name],
            markevery=T // (5 * step) - 1,
            markersize=MARKERSIZE,
            markerfacecolor="None",
            markeredgewidth=MARKEREDGEWIDTH,
        )

        ax2.plot(
            range(1, T + 1, step),
            (np.arange(1, T + 1, step) - res["is_pne"]),  # / np.arange(1, T + 1, step),
            label=label_name,
            color=COLOR[method_name],
            linewidth=LINEWIDTH,
            marker=MARKER[method_name],
            markevery=T // (5 * step) - 1,
            markersize=MARKERSIZE,
            markerfacecolor="None",
            markeredgewidth=MARKEREDGEWIDTH,
        )

        ax1.ticklabel_format(style="sci", scilimits=(-3, 4), axis="both")
        ax2.ticklabel_format(style="sci", scilimits=(-3, 4), axis="both")

        ax1.set_xticks(np.arange(0, T + 1, T // 5))
        ax2.set_xticks(np.arange(0, T + 1, T // 5))
        ax1.tick_params(axis="both", which="major", labelsize=TICKFONTSIZE)
        ax2.tick_params(axis="both", which="major", labelsize=TICKFONTSIZE)

        ax1.set_title(
            "$N={}$, {} distribution".format(N, dis.capitalize()), size=FONTSIZE, pad=15
        )
        ax2.set_xlabel("Round", size=FONTSIZE)


def plot_all():
    plt.clf()
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.5))

    plot_part(8, 10, 500000, "beta", axes[0][0], axes[1][0])
    plot_part(8, 10, 500000, "bernoulli", axes[0][1], axes[1][1])
    plot_part(20, 10, 500000, "beta", axes[0][2], axes[1][2])
    plot_part(20, 10, 500000, "bernoulli", axes[0][3], axes[1][3])

    axes[0][0].set_ylabel("Regret", size=FONTSIZE)
    axes[1][0].set_ylabel("\# of Non-equilibrium", size=FONTSIZE)

    lines, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        lines,
        labels,
        prop={"size": LEGEND_FONSIZE},
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=5,
    )
    plt.tight_layout()
    plt.savefig("figs/results.png", bbox_inches="tight")
    plt.savefig("figs/results.pdf", bbox_inches="tight")


def plot_relaxed():
    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    N, T, dis, method = 8, 500000, "beta", "SMAARelaxed"
    step = 100

    delta = 0
    for i, K in enumerate(range(10, 26, 5)):
        setting_name = setting_path(N, K, T, dis)
        res = analyze_method(setting_name, method)
        if res is None:
            continue
        axes[0].plot(
            range(1, T + 1, step),
            res["regrets"],
            label="$K={}$".format(K),
            color=COLOR_RELAXED[i + delta],
            linewidth=LINEWIDTH,
            marker=MARKER_RELAXED[i],
            markevery=T // (5 * step) - 1,
            markersize=MARKERSIZE,
            markerfacecolor="None",
            markeredgewidth=MARKEREDGEWIDTH,
        )

        axes[1].plot(
            range(1, T + 1, step),
            (np.arange(1, T + 1, step) - res["is_pne"]),  # / np.arange(1, T + 1, step),
            label="$K={}$".format(K),
            color=COLOR_RELAXED[i + delta],
            linewidth=LINEWIDTH,
            marker=MARKER_RELAXED[i],
            markevery=T // (5 * step) - 1,
            markersize=MARKERSIZE,
            markerfacecolor="None",
            markeredgewidth=MARKEREDGEWIDTH,
        )

    axes[0].set_title("Regret", size=FONTSIZE, pad=15)
    axes[1].set_title("\# of Non-equilibrium", size=FONTSIZE, pad=15)
    axes[0].set_xlabel("Round", size=FONTSIZE)
    axes[1].set_xlabel("Round", size=FONTSIZE)
    for ax in axes:
        ax.ticklabel_format(style="sci", scilimits=(-3, 4), axis="both")
        ax.set_xticks(np.arange(0, T + 1, T // 5))
        ax.tick_params(axis="both", which="major", labelsize=TICKFONTSIZE)

    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        lines,
        labels,
        prop={"size": LEGEND_FONSIZE * 0.9},
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
    )
    plt.tight_layout()
    plt.savefig("figs/results_relaxed.png", bbox_inches="tight")
    plt.savefig("figs/results_relaxed.pdf", bbox_inches="tight")


def main():
    if not os.path.exists("figs"):
        os.mkdir("figs")
    plot_all()
    plot_relaxed()


if __name__ == "__main__":
    main()
