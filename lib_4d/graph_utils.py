import matplotlib

matplotlib.use("Agg")

import networkx as nx
from matplotlib import pyplot as plt
import os, os.path as osp
import numpy as np
import imageio

def fig2nparray(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    h, w = fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0]
    data = data.reshape((h, w, 4))
    return data[:, :, :3]  # RGB only


def find_subgraph(view_list, pair_list, t):
    # find the first t views and their corresponding pairs
    assert t > 0 and t < len(view_list)
    sub_view_list = view_list[: t + 1]  # contain id=t frame
    sub_graph_edge_list = []
    sub_graph_new_edge_list = []  # any edge that contain the id=t frame
    sub_graph_old_edge_list = []
    for p in pair_list:
        if p[0] in sub_view_list and p[1] in sub_view_list:
            sub_graph_edge_list.append(p)
            if p[0] == view_list[t] or p[1] == view_list[t]:
                sub_graph_new_edge_list.append(p)
            else:
                sub_graph_old_edge_list.append(p)
    return sub_graph_edge_list, sub_graph_new_edge_list, sub_graph_old_edge_list


def filter_edges(view_list, pair_list, possible_intervals: list):
    valid_edges = []
    for p in pair_list:
        i = view_list.index(p[0])
        j = view_list.index(p[1])
        if abs(i - j) in possible_intervals:
            valid_edges.append(p)
    return valid_edges


def get_view_list(src, nerfies_flag=False):
    ret = [
        f
        for f in os.listdir(osp.join(src, "images"))
        if f.endswith(".jpg") or f.endswith(".png")
    ]
    if nerfies_flag:
        tid = np.array([int(f.split(".")[0].split("_")[1]) for f in ret])
        tid = np.argsort(tid)
        ret = [ret[i] for i in tid]
    else:
        ret.sort()
    return ret


def get_dense_sub_complete_graph_pairs(fn_list: list, intervals=[1]):
    # RCVD section 4.1, use 1,2,4,... interval view pairs
    # fn_list: list of file names
    # power: start from 0 and contains power
    # ! the dense here means every view have to find the interval neighbors
    assert isinstance(
        fn_list, list
    ), f"fn_list should be a list, but got {type(fn_list)}"
    fn_list.sort()
    pairs = []
    N = len(fn_list)
    for i in range(N):
        for j in range(i + 1, N):
            if j - i in intervals:
                pairs.append((fn_list[i], fn_list[j]))
    return pairs


def get_interval_sub_complete_graph_pairs(fn_list: list, intervals=[1]):
    # RCVD section 4.1, use 1,2,4,... interval view pairs
    # fn_list: list of file names
    # power: start from 0 and contains power
    # ! here the sparse means sub sequence, like the figure in sec.4.1 in RCVD paper
    assert isinstance(
        fn_list, list
    ), f"fn_list should be a list, but got {type(fn_list)}"
    # intervals = [2**i for i in range(power + 1)]
    fn_list.sort()
    pairs = []
    N = len(fn_list)
    for mod in intervals:
        sub = fn_list[::mod]
        for i in range(len(sub) - 1):
            pairs.append((sub[i], sub[i + 1]))
    return pairs


def save_pairs_to_txt(pair_list, save_path):
    with open(save_path, "w") as f:
        for pair in pair_list:
            f.write(f"{pair[0]} {pair[1]}\n")
    return


def read_pairs_from_txt(txt_path):
    pair_list = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            pair_list.append(tuple(line.strip().split(" ")))
    return pair_list


def viz_view_pairs(view_list, pair_list, save_path, show=False):
    # use networkx to visualize the view pairs as a graph and return plt figure
    # add nodes
    G = nx.Graph()
    G.add_nodes_from(view_list)
    # add edges
    G.add_edges_from(pair_list)

    fig = plt.figure(figsize=(6, 6))
    pos = nx.circular_layout(G)
    # pos = nx.spiral_layout(G)
    # posx = [i for i in range(len(view_list))]
    # pos = {key: [posx[i], 0] for i, key in enumerate(view_list)}

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="r", alpha=0.9)
    nx.draw_networkx_edges(
        G,
        pos,
        width=2.0,
        alpha=0.6,
        edge_color="g",
        # connectionstyle="arc3,rad=0.5",
        # arrows=True,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.title(f"{len(view_list)} views and {len(pair_list)} pairs")
    np_fig = fig2nparray(fig)
    if show:
        plt.show()
    plt.close()
    if save_path is not None:
        imageio.imwrite(save_path, np_fig)
    return np_fig
