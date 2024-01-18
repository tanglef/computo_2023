import torch
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
import json
import matplotlib.ticker as mtick
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import itertools

sns.set_style("whitegrid")


def figure_3():
    nrow = 5
    ncol = 5
    fig, axs = plt.subplots(nrow, ncol, sharey="row", sharex="col", figsize=(12, 8))
    match_ = {
        0: "bird",
        1: "car",
        2: "cat",
        3: "deer",
        4: "dog",
        5: "frog",
        6: "horse",
        7: "plane",
        8: "ship",
        9: "truck",
    }
    path = Path.cwd() / "datasets" / "cifar10H" / "train"
    for i in range(nrow):
        img_folder = path / f"{match_[i]}"
        all_imgs = list(img_folder.glob("*"))[:ncol]
        for j in range(ncol):
            image = np.asarray(Image.open(path / all_imgs[j]))
            axs[i, j].imshow(image, aspect="equal")
            axs[i, j].grid(False)
            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
    rows = list(match_.values())[:ncol]
    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size="large")
    plt.subplots_adjust(wspace=-0.8, hspace=0.25)
    plt.tight_layout()
    plt.show()


def figure_4():
    nrow = 5
    ncol = 5
    fig, axs = plt.subplots(nrow, ncol, sharey="row", sharex="col", figsize=(12, 8))
    match_ = {
        0: "coast",
        1: "forest",
        2: "highway",
        3: "insidecity",
        4: "mountain",
        5: "opencountry",
        6: "street",
        7: "tallbuilding",
    }
    path = Path.cwd() / "datasets" / "labelme" / "train"
    for i in range(nrow):
        img_folder = path / f"{match_[i]}"
        all_imgs = list(img_folder.glob("*"))[:ncol]
        for j in range(ncol):
            image = np.asarray(Image.open(path / all_imgs[j]))
            axs[i, j].imshow(image, aspect="equal")
            axs[i, j].grid(False)
            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
    rows = list(match_.values())[:ncol]
    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size="large")
    plt.subplots_adjust(wspace=-0.8, hspace=0.25)
    plt.tight_layout()
    plt.show()


def figure_5():
    nrow = 2
    ncol = 5
    fig, axs = plt.subplots(nrow, ncol, sharey="row", figsize=(12, 5))
    match_ = {
        0: "bird",
        1: "car",
        2: "cat",
        3: "deer",
        4: "dog",
        5: "frog",
        6: "horse",
        7: "plane",
        8: "ship",
        9: "truck",
    }
    inv_match_ = {v: k for k, v in match_.items()}
    real_class_to_idx = {
        "plane": 0,
        "car": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9,
    }
    inv_real_class_to_idx = {v: k for k, v in real_class_to_idx.items()}
    path = Path.cwd() / "datasets" / "cifar10H" / "train"
    list_numbers = [231, 26, 766, 0, 34]
    names = []
    for j in range(ncol):
        img_folder = path / f"{match_[j]}"
        all_imgs = list(img_folder.glob("*"))
        i = 0
        id_ = list_numbers[j]
        if j == 3:
            image = np.asarray(Image.open(path / "deer" / "deer-8153.png"))
            names.append("deer-8153")
        else:
            image = np.asarray(Image.open(path / all_imgs[id_]))
            names.append((path / all_imgs[id_]).stem)
        axs[i, j].imshow(image)
        axs[i, j].axis("off")
        # axs[i, j].set_yticklabels([])
    with open(path / ".." / "answers.json", "r") as f:
        votes = json.load(f)
    for i, name in enumerate(names):
        taskid = str(name).split("-")[-1]
        worker_votes = votes[taskid]
        distrib = np.zeros(len(match_))
        for worker, vote in worker_votes.items():
            distrib[inv_match_[inv_real_class_to_idx[vote]]] += 1
        sns.barplot(
            data=pd.DataFrame(
                {"label": match_.values(), "voting distribution": distrib},
            ),
            x="label",
            y="voting distribution",
            ax=axs[1, i],
        )
        axs[1, i].set_xticklabels(match_.values(), rotation=90)
        if i > 0:
            # axs[1, i].set_yticklabels([])
            axs[1, i].set_ylim([0, 100])
            axs[1, i].set_ylabel("")
        else:
            axs[1, i].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    # cols = [rf"$y^\star=${match_[i]}" for i in range(5)]
    # for ax, col in zip(axs[0], cols):
    #     ax.set_title(col)
    for ax in axs.flatten():
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.xaxis.set_tick_params(labelsize=13)
        ax.yaxis.set_tick_params(labelsize=13)
    plt.show()


def figure_5_labelmeversion():
    nrow = 2
    ncol = 5
    match_ = {
        0: "coast",
        1: "forest",
        2: "highway",
        3: "insidecity",
        4: "mountain",
        5: "opencountry",
        6: "street",
        7: "tallbuilding",
    }

    fig, axs = plt.subplots(nrow, ncol, sharey="row", figsize=(12, 5))
    path = Path.cwd() / "datasets" / "labelme" / "train"
    list_numbers = [0, 50, 3, 4, 91]
    names = []
    for j in range(ncol):
        img_folder = path / f"{match_[j]}"
        all_imgs = list(img_folder.glob("*"))
        i = 0
        id_ = list_numbers[j]
        image = np.asarray(Image.open(path / all_imgs[id_]))
        names.append((path / all_imgs[id_]).stem)
        axs[i, j].imshow(image)
        axs[i, j].axis("off")
    with open(path / ".." / "answers.json", "r") as f:
        votes = json.load(f)
    for i, name in enumerate(names):
        taskid = str(name).split("-")[-1]
        worker_votes = votes[taskid]
        distrib = np.zeros(len(match_))
        for worker, vote in worker_votes.items():
            distrib[vote] += 1
        distrib = distrib / np.sum(distrib) * 100
        sns.barplot(
            data=pd.DataFrame(
                {"label": match_.values(), "voting distribution": distrib},
            ),
            x="label",
            y="voting distribution",
            ax=axs[1, i],
        )
        axs[1, i].set_xticklabels(match_.values(), rotation=90)
        if i > 0:
            # axs[1, i].set_yticklabels([])
            axs[1, i].set_ylim([0, 100])
            axs[1, i].set_ylabel("")
        else:
            axs[1, i].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    for ax in axs.flatten():
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.xaxis.set_tick_params(labelsize=13)
        ax.yaxis.set_tick_params(labelsize=13)
    plt.show()


def hinton(matrix, max_weight=None, ax=None, classes=None, my_title={}):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))
    ax.set_title(my_title, y=1.02)
    ax.patch.set_facecolor("white")
    ax.set_aspect("equal", "box")
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_ylabel("True label")
    ax.set_xlabel("Proposed label")
    ax.set_xlim(-1, len(classes))
    ax.set_ylim(-1, len(classes))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)

    blues = cm.Blues
    newcolors = blues(np.linspace(0, 1, 256))
    for (x, y), w in np.ndenumerate(matrix[classes, :][:, classes]):
        idx = np.searchsorted(np.linspace(0, 1, 256), w)
        color = newcolors[idx]
        size = (
            np.sqrt(abs(w) / max_weight) if w > 0 else np.sqrt(abs(1e-8) / max_weight)
        )
        x, y = y, x
        rect = plt.Rectangle(
            [x - size / 2.1, y - size / 2.1],
            size,
            size,
            facecolor=color,
            edgecolor="black",
        )
        ax.add_patch(rect)
    ax.margins(y=0.05, x=0.05)
    ax.invert_yaxis()


def figure_6(mats, mats_confu):
    fig, axs = plt.subplots(1, 3, sharey=True)
    hinton(mats[-1], 1, my_title="Spammer worker", ax=axs[0], classes=np.arange(5))
    hinton(mats[0], 1, my_title="Expert worker", ax=axs[2], classes=np.arange(5))
    hinton(
        mats_confu[8],
        1,
        my_title="Common worker",
        ax=axs[1],
        classes=np.arange(5),
    )
    axs[0].set_ylabel("True label")
    axs[1].set_ylabel("")
    axs[2].set_ylabel("")
    plt.tight_layout()
    plt.show()


def figure_simulations(workerload, feedback):
    nbins = 17
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    sns.histplot(workerload, stat="percent", bins=nbins, shrink=1, ax=ax[0])
    ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax[0].set_xlabel(r"$\vert\mathcal{T}(w_j)\vert$")
    sns.histplot(feedback, stat="percent", bins=nbins, shrink=1, ax=ax[1])
    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax[1].set_xlabel(r"$\vert\mathcal{A}(x_i)\vert$")
    ax[1].set_xlim(8, 12)
    for i in range(2):
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[i].xaxis.label.set_size(15)
        ax[i].yaxis.label.set_size(15)
        ax[i].xaxis.set_tick_params(labelsize=13)
        ax[i].yaxis.set_tick_params(labelsize=13)
        ax[i].title.set_size(18)
    plt.tight_layout()


def load_data(dataset, n_classes, classes):
    if "cifar" in dataset:
        mv = np.load(f"./datasets/{dataset}/labels/labels_cifar-10h_mv.npy")
    else:
        mv = np.load(f"./datasets/{dataset}/labels/labels_labelme_mv.npy")

    entrop = np.load(f"./datasets/{dataset}/identification/entropies.npy")
    path_train = Path(f"./datasets/{dataset}/train")
    glad = 1 / np.exp(
        np.load(f"./datasets/{dataset}/identification/glad/difficulties.npy")[:, 1]
    )
    dfwaum = (
        pd.read_csv(
            f"./datasets/{dataset}/identification/resnet34/waum_0.01_yang/waum.csv"
        )
        if dataset.startswith("cifar")
        else pd.read_csv(
            f"./datasets/{dataset}/identification/modellabelme/waum_0.01_yang/waum.csv"
        )
    )
    sorted_df = dfwaum.sort_values(by="waum")
    tasks = sorted_df["task"].values
    img_ns, img_glad, img_waum = [], [], []
    idxs_ns = np.argsort(entrop)[::-1]
    idxs_glad = np.argsort(glad)[::-1]
    idxs_waum = [Path(task).stem.split("-")[1] for task in tasks]
    # key_to_index = {v: k for k, v in zip(sorted_df["index"], idxs_waum)}
    for idxs, im_store in zip(
        [idxs_ns, idxs_glad, idxs_waum], [img_ns, img_glad, img_waum]
    ):
        imgs = []
        for k in range(n_classes):
            imgs.append([])
            flag = 0
            for id_, file in list(
                itertools.product(idxs, path_train.glob(f"{classes[k]}/*"))
            ):
                if file.stem.endswith(f"-{id_}") and mv[int(id_)] == k:
                    im = Image.open(file)
                    if dataset.startswith("cifar"):
                        im = im.resize((32, 32))
                    else:
                        im = im.resize((64, 64))  # memory saving
                    imgs[k].append(np.array(im))
                    flag += 1
                if flag == 12:
                    break
            image_k_row1 = np.hstack(imgs[k][:6])
            image_k_row2 = np.hstack(imgs[k][6:])
            image_k = np.vstack((image_k_row1, image_k_row2))
            im_store.append(image_k)
    all_images = [img_ns, img_glad, img_waum]
    return all_images


def get_visible_strat(strategy, n_classes):
    ll = [False] * (n_classes * 3)
    for k in range(n_classes * strategy, n_classes * (strategy + 1)):
        ll[k] = True
    return ll


def get_layer_strat(strategy, n_classes):
    ll_layer = ["below"] * (n_classes * 3)
    for k in range(n_classes * strategy, n_classes * (strategy + 1)):
        ll_layer[k] = "above"
    return ll_layer


def get_visible_class(lab, n_classes):
    ll = [False] * (n_classes * 3)
    for k in range(3):
        ll[n_classes * k + lab] = True
    return ll


def generate_plot(n_classes, all_images, classes):
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Entropy", "GLAD difficulty", "WAUM"),
        vertical_spacing=0.1,
    )
    button_classes = [
        dict(
            args=[{"visible": get_visible_class(k, n_classes)}],
            label=f"{classes[k]}",
            method="update",
        )
        for k in range(n_classes)
    ]
    buttons_method = [
        dict(
            args=[
                {
                    "visible": get_visible_strat(0, n_classes),
                    "layer": get_layer_strat(0, n_classes),
                }
            ],
            label="entropy",
            method="update",
        ),
        dict(
            args=[
                {
                    "visible": get_visible_strat(1, n_classes),
                    "layer": get_layer_strat(1, n_classes),
                }
            ],
            label="glad",
            method="update",
        ),
        dict(
            args=[
                {
                    "visible": get_visible_strat(2, n_classes),
                    "layer": get_layer_strat(1, n_classes),
                }
            ],
            label="waum",
            method="update",
        ),
    ]
    layout = go.Layout(
        updatemenus=[
            {
                "type": "buttons",
                "buttons": button_classes,
                "active": 3,
                "showactive": True,
                "direction": "down",  # Display buttons horizontally
                "x": 0.1,  # X-position of the buttons (0.5 centers them)
                "y": 1.0,  # Y-position   of the buttons
            },
        ]
    )
    data = [
        go.Image(
            z=all_images[method][k],
            visible=True if k == 3 else False,
            name=f"{classes[k]} - {method}",
        )
        for method in range(len(all_images))
        for k in range(n_classes)
    ]

    for k in range(n_classes * len(all_images)):
        fig.add_trace(
            data[k],
            row=1 + k // n_classes,
            col=1,
        )
    fig.update_layout(layout)
    fig.update_layout(
        width=800,
        height=700,
        margin=dict(l=0, r=0, t=30, b=0, autoexpand=True),
    )
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.show()
    fig, ax = plt.subplots(figsize=(0.1, 0.1), layout="constrained")
    ax.axis("off")
    plt.show()


def figure_bird(workerload, feedback):
    nbins = 40
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    sns.histplot(workerload, stat="percent", bins=nbins, shrink=1, ax=ax[0])
    ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax[0].set_xlabel(r"$\vert\mathcal{T}(w_j)\vert$")
    sns.histplot(feedback, stat="percent", bins=nbins, shrink=1, ax=ax[1])
    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax[0].tick_params(axis="x", rotation=45)
    ax[1].set_xlabel(r"$\vert\mathcal{A}(x_i)\vert$")
    # ax[1].set_xlim(8, 12)s
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    for i in range(2):
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[i].xaxis.label.set_size(15)
        ax[i].yaxis.label.set_size(15)
        ax[i].xaxis.set_tick_params(labelsize=13)
        ax[i].yaxis.set_tick_params(labelsize=13)
        ax[i].title.set_size(18)
    plt.tight_layout()
