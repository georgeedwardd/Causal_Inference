from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def plot_reg(df):
    # Initialise a simple linear regression model
    reg = LinearRegression()
    
    # Fit the model using the first column as the predictor (X)
    # and the second column as the target (y)
    reg.fit(df.iloc[:, [0]], df.iloc[:, [1]])
    
    # Extract predictor values for prediction (same X used for fitting)
    x = df.iloc[:, [0]]
    
    # Generate predicted values from the fitted regression model
    y = reg.predict(x)
    
    # Plot the regression line on the current figure
    plt.plot(x, y, lw=1.6)


######################################################################################################
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
import networkx as nx
from causallearn.graph.Endpoint import Endpoint

def nx_to_causallearn_graph(nx_dag):
    # Convert a NetworkX directed graph (DAG) into a causal-learn GeneralGraph

    # Create GraphNode objects for each node in the NetworkX graph
    # Initialise a causal-learn graph using these wrapped nodes
    nodes = {n: GraphNode(str(n)) for n in nx_dag.nodes()}
    graph = GeneralGraph(list(nodes.values()))
    
    # Add directed edges from the NetworkX DAG into the causal-learn graph
    for u, v in nx_dag.edges():
        graph.add_directed_edge(nodes[u], nodes[v])
    
    # Return the converted causal-learn representation
    return graph


def cpdag_to_nx(cpdag):
    """
    Convert a causal-learn CPDAG into two NetworkX graphs:
    one directed graph (identified orientations) and one undirected graph (ambiguous edges)
    """
    
    # Initialise empty graphs for directed and undirected structures
    G_dir = nx.DiGraph()
    G_undir = nx.Graph()
    
    # Iterate over all edges in the CPDAG structure
    for edge in cpdag.get_graph_edges():
        
        # Extract node names from causal-learn node objects
        n1 = edge.get_node1().get_name()
        n2 = edge.get_node2().get_name()
        
        # Extract edge endpoint types (arrow vs tail)
        end1 = edge.get_endpoint1()
        end2 = edge.get_endpoint2()
        
        # Case 1: n1 → n2 (fully directed edge)
        if end1 == Endpoint.TAIL and end2 == Endpoint.ARROW:
            G_dir.add_edge(n1, n2)
        
        # Case 2: n2 → n1 (reverse direction)
        elif end1 == Endpoint.ARROW and end2 == Endpoint.TAIL:
            G_dir.add_edge(n2, n1)
        
        # Case 3: undirected edge (orientation not identifiable)
        else:
            G_undir.add_edge(n1, n2)
    
    # Return both representations for downstream analysis
    return G_dir, G_undir


######################################################################################################
def show_assoc(X, Y, Z, title):
    # Print a bold section header for clarity in output
    print("\033[1m" + f"--------- {title} ---------" + "\033[0m")
    
    # --- Marginal (unconditional) relationships ---
    # Compare outcome Y between X=1 and X=0 without conditioning on Z
    print(f"P(Y=1 | X=1)      = {Y[X==1].mean():.3f}    P(Y=1 | X=0) = {Y[X==0].mean():.3f}")
    
    # Overall probability of Y=1 in the full dataset
    print(f"P(Y=1)            = {Y.mean():.3f}")
    print()

    # --- Conditional relationships given Z ---
    # Examine how the relationship between X and Y changes when conditioning on Z
    for zval in (0, 1):
        
        # Create mask for subgroup Z = zval
        mask = (Z == zval)
        
        # Subgroup outcomes for X = 1 and X = 0 within Z = zval
        y_x1 = Y[(X == 1) & mask]
        y_x0 = Y[(X == 0) & mask]
        
        # Conditional probabilities within subgroup
        p_y_x1 = y_x1.mean()
        p_y_x0 = y_x0.mean()

        # Overall probability of Y=1 within this Z subgroup
        y_z = Y[mask]
        p_y_z = y_z.mean()
        
        # Print conditional comparisons
        print(f"P(Y=1 | X=1, Z={zval}) = {p_y_x1:.3f}    P(Y=1 | X=0, Z={zval}) = {p_y_x0:.3f}")
        print(f"P(Y=1 | Z={zval})      = {p_y_z:.3f}")
        print()
    
    # Final spacing for readability between sections
    print()