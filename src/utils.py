from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_reg(df):
    reg = LinearRegression()
    reg.fit(df.iloc[:,[0]],df.iloc[:,[1]])
    x = df.iloc[:,[0]]
    y = reg.predict(x)
    plt.plot(x,y,lw=1.6)



def simulate_data(seed=99):
    np.random.seed(seed)
    
    severity_levels = [
        "Mild",
        "Moderate",
        "Severe",
    ]
    n_per_group = 100
    data = []

    for i, sev in enumerate(severity_levels):
        
        # dosage increases with severity
        dosage = np.random.normal(loc=i+6, scale=1.2, size=n_per_group)

        # higher dosage -> faster recovery
        recovery_time = np.random.normal(7 + 2.5*i, 1.1, n_per_group) - 0.35*dosage

        # ensure reasonable bounds
        dosage = np.clip(dosage, 0.2, 10)
        recovery_time = np.clip(recovery_time, 0.0, None)
        
        for x, y in zip(dosage, recovery_time):
            data.append([x, y, sev])

    return pd.DataFrame(data, columns=["dosage", "recovery_time", "severity"])


######################################################################################################
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
import networkx as nx
from causallearn.graph.Endpoint import Endpoint

def nx_to_causallearn_graph(nx_dag):
    # Convert a NetworkX directed graph (DAG) into a causal-learn GeneralGraph

    # Create GraphNode objects for each node in the NetworkX graph snd Initialise causal-learn graph with these nodes
    nodes = {n: GraphNode(str(n)) for n in nx_dag.nodes()}
    graph = GeneralGraph(list(nodes.values()))
    
    # Add directed edges to the causal-learn graph
    for u, v in nx_dag.edges():
        graph.add_directed_edge(nodes[u], nodes[v])
    
    return graph



def cpdag_to_nx(cpdag):
    """
    Convert a causal-learn CPDAG into two NetworkX graphs:
    one directed and one undirected
    """
    
    G_dir = nx.DiGraph()
    G_undir = nx.Graph()
    
    # Iterate through CPDAG edges and classify them
    for edge in cpdag.get_graph_edges():
        n1 = edge.get_node1().get_name()
        n2 = edge.get_node2().get_name()
        
        end1 = edge.get_endpoint1()
        end2 = edge.get_endpoint2()
        
        # Case 1: n1 → n2
        if end1 == Endpoint.TAIL and end2 == Endpoint.ARROW:
            G_dir.add_edge(n1, n2)
        
        # Case 2: n2 → n1
        elif end1 == Endpoint.ARROW and end2 == Endpoint.TAIL:
            G_dir.add_edge(n2, n1)
        
        # Case 3: undirected edge (no clear orientation)
        else:
            G_undir.add_edge(n1, n2)
    
    return G_dir, G_undir

######################################################################################################
def show_assoc(X, Y, Z, title):
    print("\033[1m" + f"--------- {title} ---------" + "\033[0m")
    
    # Marginal relationship
    print(f"P(Y=1 | X=1)      = {Y[X==1].mean():.3f}    P(Y=1 | X=0) = {Y[X==0].mean():.3f}")
    print(f"P(Y=1)            = {Y.mean():.3f}")
    print()

    # Conditional on Z
    for zval in (0, 1):
        mask = (Z == zval)
        
        y_x1 = Y[(X == 1) & mask]
        y_x0 = Y[(X == 0) & mask]
        
        p_y_x1 = y_x1.mean()
        p_y_x0 = y_x0.mean()

        y_z = Y[mask]
        p_y_z = y_z.mean()
        
        print(f"P(Y=1 | X=1, Z={zval}) = {p_y_x1:.3f}    P(Y=1 | X=0, Z={zval}) = {p_y_x0:.3f}")
        print(f"P(Y=1 | Z={zval})      = {p_y_z:.3f}")
        print()
    print()