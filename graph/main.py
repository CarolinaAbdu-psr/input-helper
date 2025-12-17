import networkx as nx
from data_loader import data_loader
from get_data_properties import extract_node_properties

STUDY_PATH = r"D:\\01-Repositories\\factory-graphs\\Bolivia"
G, load_times = data_loader(STUDY_PATH)
G, node_properties = extract_node_properties(G)
#print(G.nodes(data = 'object'))

nx.write_graphml(G, "grafo.graphml")
