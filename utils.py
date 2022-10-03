"""This module contains some auxiliary functions that are used in the notebooks."""
import pandas as pd
import networkx as nx


def get_test_edges(data, edges_threshold=6, n_smallest=5):
    """In the given graph finds the edges that appeared the latest.

    Args:
        data (pd.DataFrame): a dataframe with columns 'u', 'v', 't', 'h'.
        edges_threshold (int): the minimum number of samples for each unique entry in 'u'
            for it to be selected, corresponds to the minimum amount of edges for a node
        n_smallest (int): the number of edges to be selected from each selected node
    Returns:
        A list with tuples (u, v).
    """
    ids = data.u.value_counts() > edges_threshold
    ids = ids[ids.values].index
    multiple_edges_df = data[data["u"].isin(ids)]
    df1 = (
        multiple_edges_df.set_index("v")
        .groupby("u", sort=False)["t"]
        .apply(lambda x: pd.Series(x.nsmallest(n_smallest).index))
        .unstack()
    )
    df1.columns = df1.columns + 1
    df1 = df1.add_prefix("v_").reset_index()
    df1["v"] = df1[df1.columns[1:]].values.tolist()
    df1_short = df1[["u", "v"]]
    test_edges = df1_short.explode("v").reset_index(drop=True)
    return test_edges


def get_graph(df):
    """Creates a graph from the given dataframe.

    Args:
        data (pd.DataFrame): a dataframe with columns 'u', 'v', 'h'.
    Returns:
        A networkx graph.
    """
    edge_list = df.values.tolist()
    G = nx.Graph()
    for i in range(len(edge_list)):
        G.add_edge(edge_list[i][0], edge_list[i][1], weight=edge_list[i][2])
    return G


def get_links(edges_data):
    """This function creates a dictionary with all edges for each user in the given data.

    Args:
        edges_data (pd.DataFrame): a dataframe with columns 'u', 'v' which correspond to the edges.
    Returns:
        A dictionary of the form {user_id: {friend_1, friend_2, ..., friend_n}}
    """
    dict_actual = {}
    for id in edges_data["u"].unique():
        dict_actual[id] = set(edges_data[edges_data["u"] == id].v.values)
    return dict_actual


def recall_at_k(dict_predicted, dict_actual):
    """Computes the 'recall at k' metric.

    Args:
        dict_predicted (dict): a dictionary of the form {user_id: {pred_1, pred_2, ..., pred_k}}
        dict_actual (dict): a dictionary of the same form as dict_predicted
    """
    total_relevant = 0
    total_actual = 0
    for key, predicted in dict_predicted.items():
        actual = dict_actual[key]
        num_relevant = len(predicted & actual)
        num_actual = len(actual) if len(actual) <= 10 else 10
        total_relevant += num_relevant
        total_actual += num_actual
    return total_relevant / total_actual
