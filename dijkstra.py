#!/usr/bin/env python3
"""
Dijkstra's Shortest Path Algorithm for Sioux Falls Network with Visualization
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple


class Graph:
    """A class representing an undirected weighted graph."""

    def __init__(self, nodes: List[str], edges_with_weights: List[Tuple[str, str, int]]) -> None:
        """
        Initialize the Graph.

        Args:
            nodes: A list of node identifiers.
            edges_with_weights: A list of tuples containing (node1, node2, weight).
        """
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, edges_with_weights)
        self.edges_with_weights = edges_with_weights  # Store edges for visualization

    def construct_graph(
        self, nodes: List[str], edges_with_weights: List[Tuple[str, str, int]]
    ) -> Dict[str, Dict[str, int]]:
        """
        Construct an undirected graph ensuring both directions are stored.

        Args:
            nodes: A list of node identifiers.
            edges_with_weights: A list of tuples containing (node1, node2, weight).

        Returns:
            A dictionary representing the symmetrical graph.
        """
        graph: Dict[str, Dict[str, int]] = {node: {} for node in nodes}

        for node1, node2, weight in edges_with_weights:
            graph[node1][node2] = weight
            graph[node2][node1] = weight  # Ensure undirected behavior

        return graph

    def get_nodes(self) -> List[str]:
        """Return the list of nodes in the graph."""
        return self.nodes

    def get_outgoing_edges(self, node: str) -> List[str]:
        """Return the neighbors of the given node."""
        return list(self.graph[node].keys())

    def value(self, node1: str, node2: str) -> int:
        """Return the weight of the edge between node1 and node2."""
        return self.graph[node1][node2]


def dijkstra_algorithm(graph: Graph, start_node: str) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Perform Dijkstra's algorithm to compute the shortest path from the start node to all other nodes.

    Args:
        graph: An instance of Graph.
        start_node: The starting node for the search.

    Returns:
        A tuple containing:
            - A dictionary mapping each node to its previous node on the shortest path.
            - A dictionary mapping each node to the total cost of the shortest path from the start node.
    """
    unvisited_nodes = set(graph.get_nodes())
    shortest_path: Dict[str, int] = {node: sys.maxsize for node in unvisited_nodes}
    previous_nodes: Dict[str, str] = {}

    shortest_path[start_node] = 0

    while unvisited_nodes:
        current_min_node = min(unvisited_nodes, key=lambda node: shortest_path[node])
        unvisited_nodes.remove(current_min_node)

        for neighbor in graph.get_outgoing_edges(current_min_node):
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                previous_nodes[neighbor] = current_min_node

    return previous_nodes, shortest_path


def visualize_graph(graph: Graph, shortest_path: List[str]) -> None:
    """
    Visualize the graph using NetworkX and Matplotlib.

    Args:
        graph: An instance of Graph.
        shortest_path: List of nodes representing the shortest path.
    """
    G = nx.Graph()

    # Add edges with weights
    for node1, node2, weight in graph.edges_with_weights:
        G.add_edge(node1, node2, weight=weight)

    pos = nx.spring_layout(G, seed=42)  # Generate layout for positioning nodes

    # Increase figure size for better visualization
    plt.figure(figsize=(18, 18))  # Adjust width and height as needed

    # Draw the entire graph in gray
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", edge_color="gray", font_size=10)

    # Highlight the shortest path in red
    path_edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path) - 1)]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=3)

    # Draw edge labels (weights)
    edge_labels = {(node1, node2): f"{d['weight']}" for node1, node2, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Sioux Falls Network - Shortest Path Highlighted")
    plt.show()


def print_result(
    previous_nodes: Dict[str, str], shortest_path: Dict[str, int], start_node: str, target_node: str
) -> List[str]:
    """
    Print and return the shortest path.

    Args:
        previous_nodes: A dictionary mapping each node to its previous node in the shortest path.
        shortest_path: A dictionary mapping each node to the cost from the start node.
        start_node: The starting node.
        target_node: The target node.

    Returns:
        The shortest path as a list of node identifiers.
    """
    path = []
    node = target_node

    while node != start_node:
        path.append(node)
        node = previous_nodes.get(node)
        if node is None:
            print(f"No path found from {start_node} to {target_node}.")
            return []
    path.append(start_node)
    path.reverse()

    print(f"Best path from {start_node} to {target_node} (cost {shortest_path[target_node]}):")
    print(" -> ".join(path))

    return path


def load_graph_from_excel(file_path: str) -> Graph:
    """
    Load graph data (nodes and weighted edges) from an Excel file.

    Args:
        file_path: Path to the Excel file.

    Returns:
        An instance of Graph.
    """
    df_links = pd.read_excel(file_path, sheet_name="Positive")  # Load weights

    # Extract edges with weights
    edges_with_weights = []
    for _, row in df_links.iterrows():
        link_id = row["Link number"]
        weight = row["Time (minute)"]
        if link_id in corrected_edges:
            start, end = corrected_edges[link_id]
            edges_with_weights.append((start, end, weight))

    nodes = list(set(node for edge in edges_with_weights for node in edge[:2]))

    return Graph(nodes, edges_with_weights)


def main():
    """Main function to execute the algorithm and display results."""
    file_path = "Link Cost.xlsx"  # Update with actual path if needed
    graph = load_graph_from_excel(file_path)

    start_node = "1"
    target_node = "20"

    previous_nodes, shortest_path = dijkstra_algorithm(graph, start_node)
    shortest_path_nodes = print_result(previous_nodes, shortest_path, start_node, target_node)

    # Visualize the graph and highlight the shortest path
    visualize_graph(graph, shortest_path_nodes)


if __name__ == "__main__":
    main()
