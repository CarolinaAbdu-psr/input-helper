import psr.factory
import networkx as nx
import time

profiling_times = {}

def create_nodes(G, study):
    # Add every object as a Graph node
    all = study.get_all_objects()

    for item in all:
        code = (item.type + "_" + str(item.code)).replace(" ","")
        G.add_node(code, type=item.type ,object=item) #Add node with code and object

def create_edges(G):
    # Add edges between connected nodes
    for node in G.nodes:
        obj = G.nodes[node]["object"] #get object
        conneced_objects = obj.referenced_by()
        for item in conneced_objects:
            edge_title = f"Ref_{obj.type}" 
            item_code = (item.type + "_" + str(item.code)).replace(" ","")
            G.add_edge(item_code,node,title=edge_title)


# --- Load Data ---
def data_loader(study_path):
    try:
        start_load = time.time()
        study = psr.factory.load_study(study_path)
        end_load = time.time()
        profiling_times['load_study_time'] = end_load - start_load

        G = nx.DiGraph()
        start_build = time.time()
        create_nodes(G,study)
        create_edges(G)
        G.add_node("Context",object=study.context, type="Contex")
        end_build = time.time()
        profiling_times['build_graph_time'] = end_build - start_build

    except Exception as e:
        print(f"Erro ao carregar o estudo: {e}")

    return G, profiling_times

