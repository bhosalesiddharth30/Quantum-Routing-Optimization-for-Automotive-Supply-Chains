import pandas as pd
import networkx as nx
import numpy as np

def build_graph(csv_path):
    df = pd.read_csv(csv_path)
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['id'], pos=(row['x'], row['y']), type=row['type'], demand=row['demand'], cost=row['cost'])
    for i in G.nodes():
        for j in G.nodes():
            if i < j:
                xi, yi = G.nodes[i]['pos']
                xj, yj = G.nodes[j]['pos']
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                G.add_edge(i, j, weight=dist)
    return G

def adjacency_matrix(G):
    return nx.to_numpy_array(G, weight='weight')

def save_adjacency_matrix(matrix, filename):
    np.savetxt(filename, matrix, delimiter=",")
