from neo4j import GraphDatabase

class Neo4jConnector:
    def __init__(self, uri, user, password):
        # Inicializa a conexão com o Neo4j
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Fecha a conexão
        self._driver.close()

    def run_query(self, cypher_query, parameters=None):
        # Executa uma consulta Cypher e retorna os resultados
        with self._driver.session() as session:
            try:
                # Usa um bloco de transação para garantir a execução
                result = session.run(cypher_query, parameters if parameters else {})
                return [record for record in result]
            except Exception as e:
                print(f"Erro ao executar a consulta Cypher: {e}")
                return None