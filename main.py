# main.py
# Quantum Routing Optimization for Automotive Supply Chains
#
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from qiskit_optimization.applications import Tsp
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit import Aer

from utils.graph_tools import build_graph, adjacency_matrix

def main():
    # Step 1: Build graph from CSV
    G = build_graph('data/supply_nodes.csv')
    pos = {n: G.nodes[n]['pos'] for n in G.nodes()}

    # Step 2: Visualize supply chain graph
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Supply Chain Graph")
    plt.show()

    # Step 3: Create adjacency matrix & formulate QUBO for TSP
    adj_matrix = adjacency_matrix(G)
    tsp = Tsp(distance_matrix=adj_matrix)
    qp = tsp.to_quadratic_program()

    # Step 4: QAOA solver setup
    qaoa = QAOA(reps=2, optimizer=COBYLA(), quantum_instance=Aer.get_backend('aer_simulator'))
    meo = MinimumEigenOptimizer(qaoa)
    result = meo.solve(qp)
    solution = tsp.interpret(result.x)
    cost = tsp.tsp_value(solution, adj_matrix)
    print("QAOA route cost:", cost)
    print("QAOA route order:", solution)

    # Step 5: Classical Dijkstra baseline (sum of shortest paths from node 0)
    total_classical_cost = 0
    for node in G.nodes():
        if node != 0:
            length = nx.shortest_path_length(G, source=0, target=node, weight='weight')
            total_classical_cost += length
    print("Classical (Dijkstra) total cost:", total_classical_cost)

    # Step 6: Visualize QAOA-optimized route
    route_edges = [(solution[i], solution[i + 1]) for i in range(len(solution) - 1)]
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='r', width=2)
    plt.title("QAOA Optimized Route")
    plt.show()

if __name__ == "__main__":
    main()
