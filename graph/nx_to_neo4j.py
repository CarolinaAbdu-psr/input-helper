import networkx as nx
#from graph.data_loader import data_loader
from data_loader import data_loader
#from graph.get_data_properties import extract_node_properties
from get_data_properties import extract_node_properties
from neo4j import GraphDatabase
from typing import Dict, Any


def load_networkx_to_neo4j(
    G: nx.DiGraph, 
    node_properties: Dict[Any, Dict[str, Any]], 
    uri: str = "neo4j://127.0.0.1:7687",
    auth: tuple = ("neo4j", "sua_senha_aqui"),
    default_node_label: str = "Entity",
    default_rel_type: str = "LINKS_TO",
    clear_existing_data: bool = False
) -> str:
    """
    Carrega um grafo NetworkX (nx.DiGraph) e suas propriedades de nó associadas 
    em um banco de dados Neo4j.
    """
    # 1. Conexão com o Neo4j
    try:
        driver = GraphDatabase.driver(uri, auth=auth)
        driver.verify_connectivity()
    except Exception as e:
        return f"Erro ao conectar ao Neo4j: {e}"

    # Funções de transação para escrita no Neo4j

    def clear_data_tx(tx):
        """Transação para apagar todos os nós e relacionamentos."""
        print("ATENÇÃO: Apagando todos os dados existentes no Neo4j...")
        # DETACH DELETE remove primeiro os relacionamentos, depois os nós
        tx.run("MATCH (n) DETACH DELETE n")
        print("✅ Dados antigos apagados com sucesso.")
    
    def create_node_tx(tx, node_id, properties):
        """Cria ou atualiza um nó."""
        
        # Define o Label (Rótulo) e remove do dicionário de propriedades se presente
        
        label = properties.pop('ObjType', default_node_label)
        

        if not isinstance(label, str) or not label:
            label = default_node_label
        
        # Adiciona a ID do NetworkX como propriedade para referência
        properties['id_netx'] = node_id 
        
        # Cypher: MERGE garante que o nó só é criado se não existir, 
        # usando 'id_netx' como chave única. SET atualiza todas as propriedades.
        merge_query = (
            f"MERGE (n:{label} {{id_netx: $id_netx}}) "
            f"SET n += $props "
            "RETURN n"
        )
        tx.run(merge_query, id_netx=node_id, props=properties)
        

    def create_relationship_tx(tx, source_id, target_id, rel_attrs):
        """Cria uma relação direcionada (para DiGraph)."""
        
        # Define o Type (Tipo de Relacionamento) usando 'title' ou o padrão
        rel_type = rel_attrs.pop('title', default_rel_type)

        if not isinstance(rel_type, str) or not rel_type:
            rel_type = default_rel_type
        
        # Cypher: MATCH encontra os nós pelo 'id_netx' e MERGE cria a relação direcionada
        query = (
            "MATCH (a), (b) "
            "WHERE a.id_netx = $source_id AND b.id_netx = $target_id "
            f"MERGE (a)-[r:{rel_type}]->(b) " # -> indica direção
            "SET r += $props"
        )
        tx.run(query, source_id=source_id, target_id=target_id, props=rel_attrs)


    # 2. Execução das Transações
    with driver.session() as session:
        if clear_existing_data:
            session.execute_write(clear_data_tx)

        print("Iniciando carregamento de nós...")
        
        # Cria os Nós (usando o node_properties fornecido)
        nodes_entities= []
        for n in G.nodes():
            # Obtém as propriedades do nó, garantindo um dicionário 
            data = node_properties.get(n, {})
            label = data.get('ObjType', default_node_label)
            if label not in nodes_entities:
                nodes_entities.append(label)
            entities = session.execute_write(create_node_tx, n, dict(data))
 
        print("Iniciando carregamento de relacionamentos...")

        # Cria os Relacionamentos (usando os atributos de aresta do G)
        edges_labels = []
        for u, v, attrs in G.edges(data=True):
            rel_type = attrs.get('title', default_rel_type)
            if rel_type not in edges_labels:
                edges_labels.append(rel_type)
            session.execute_write(create_relationship_tx, u, v, dict(attrs))

    driver.close()
    print("✅ Grafo carregado no Neo4j com sucesso!")
    return nodes_entities, edges_labels


if __name__ == "__main__":

    neo4j_uri = "neo4j://127.0.0.1:7687"
    neo4j_auth = ("neo4j", "psr-2025")

    STUDY_PATH = r"D:\\01-Repositories\\factory-graphs\\Bolivia"
    G, load_times = data_loader(STUDY_PATH)
    G, node_properties = extract_node_properties(G)
    print(G.nodes())

    nodes,edges = load_networkx_to_neo4j(G, node_properties, uri=neo4j_uri, auth=neo4j_auth,clear_existing_data=True)

