import json
import argparse
import yaml

import datetime as dt
import os
from typing import (
    Tuple,
    List,
    Annotated,
    Dict,
    Any
)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
import chromadb.config
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from graph.data_loader import data_loader
from graph.get_data_properties import extract_node_properties
from graph.nx_to_neo4j import load_networkx_to_neo4j
from Neo4jConnector import Neo4jConnector

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

REQUEST_TIMEOUT = 120
MAX_TOKENS = 4096

AGENTS_DIR = "helper"
CYPHER_AGENT_FILENAME = "agent.yaml"

def load_cypher_agent_config(filepath: str) -> bool:
    """Carrega os templates do YAML para variáveis globais."""
    global SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_CYPHER_GENERATION, USER_PROMPT_TEXTUAL_RESPONSE
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        SYSTEM_PROMPT_TEMPLATE = config['system_prompt_template']
        USER_PROMPT_CYPHER_GENERATION = config['translation_task']['user_prompt_cypher_generation']
        USER_PROMPT_TEXTUAL_RESPONSE = config['formatting_task']['user_prompt_textual_response']
        
        logger.info(f"Templates do agente Cypher carregados com sucesso de {filepath}")
        return True
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logger.error(f"Erro ao carregar ou analisar o agente Cypher em {filepath}: {e}")
        return False

# Load agents configuration from individual files
def load_agents_config() -> Dict[str, Any]:
    """Carrega a configuração do agente Text-to-Cypher."""
    filepath = os.path.join(AGENTS_DIR, CYPHER_AGENT_FILENAME)
    if load_cypher_agent_config(filepath):
        return {'status': 'loaded'}
    return {'status': 'failed'}

# Load configuration once
_AGENTS_CONFIG = load_agents_config()


def get_schema(study_path):

    neo4j_uri = "neo4j://127.0.0.1:7687"
    neo4j_auth = ("neo4j", "psr-2025")

    G, load_times = data_loader(study_path)
    node_properties = extract_node_properties(G)
    nodes, edges = load_networkx_to_neo4j(G, node_properties, uri=neo4j_uri, auth=neo4j_auth,clear_existing_data=True)
    names = []
    for obj in node_properties.values(): 
        name = obj.get('name')
        names.append(name)
    SCHEMA_DATA = f"""Nodes:{nodes}, 
    Relationships: {edges}, 
    Names: {names}"""

    return SCHEMA_DATA

rag_data = """pergunta_natural": "Qual a soma da capacidade instalada das térmicas que usam Gás como combustível?",
    "cypher_query": "MATCH (t:ThermalPlant)-[:Ref_Fuel]->(c:Fuel {name: 'Gas'}), (t)-[:HAS_PROPERTY]->(p:Property {nome_propriedade: 'InstalledCapacity'}) RETURN SUM(p.valor) AS total_potencia"

    "pergunta_natural: Qual a soma da capacidade instalada das plantas térmicas?"
    "cypher_query: MATCH (t:ThermalPlant)-[:Ref_Fuel]->(c:Fuel {name: 'Gas'}), (t)-[:HAS_PROPERTY]->(p:Property {nome_propriedade:
  
    "pergunta_natural": "Liste o nome das restrições ligadas à usina de Belo Monte.",
    "cypher_query": "MATCH (u:FactoryElement {name: 'Belo Monte'})-[:LINKED_TO]->(r:Restricao) RETURN r.nome"
"""

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    input: str # A pergunta do usuário
    context: str # Contexto RAG (Exemplos Cypher)
    schema: str # Esquema do Grafo
    cypher_query: str # Query Cypher gerada
    query_result: str # Resultado da Query do Neo4j
    chat_language: str
    agent_type: str

def retrieve_context(state: GraphState) -> Dict[str, Any]:
    """
    1. Injeta o esquema do grafo.
    2. Realiza a busca RAG e injeta os exemplos de Cypher.
    """
    logger.info("Etapa: Recuperação de Contexto (RAG)")
    
    # Na implementação real, você faria a busca de similaridade aqui:
    # rag_data = vector_db.retrieve(state["input"]) 
    
    return {
        "context": rag_data, # Contexto RAG (Exemplos Cypher)
        "schema": SCHEMA_DATA # Esquema do Grafo (Preenchido por get_schema)
    }


def generate_query(state: GraphState, llm: BaseChatOpenAI) -> Dict[str, Any]:
    """
    2. Primeira etapa da LLM: Traduz a pergunta para Cypher.
    """
    logger.info("Etapa: Geração da Query Cypher")
    
    # 1. Monta o System Prompt (inclui SCHEMA e RAG_EXAMPLES)
    system_prompt_content = SYSTEM_PROMPT_TEMPLATE.format(
        graph_schema=state["schema"],
        rag_examples=state["context"]
    )
    system_message = SystemMessage(content=system_prompt_content)
    
    # 2. Monta o User Prompt (inclui a PERGUNTA do usuário)
    user_prompt_content = USER_PROMPT_CYPHER_GENERATION.format(
        user_input=state["input"]
    )
    human_message = HumanMessage(content=user_prompt_content)

    # 3. Envia para a LLM
    response = llm.invoke([system_message, human_message])
    
    # A resposta da LLM deve ser PURAMENTE a query Cypher
    cypher_query = response.content.strip()
    
    logger.info(f"Query Cypher gerada: {cypher_query[:200]}...")
    return {"cypher_query": cypher_query}

def execute_query(state: GraphState) -> Dict[str, Any]:
    """
    3. Executa a query Cypher no banco de dados e obtém o resultado.
    """
    logger.info("Etapa: Execução da Query")
    
    cypher_query = state["cypher_query"]
    
    # Executa a query usando a instância simulada ou real
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "psr-2025"
    NEO4J_CONNECTOR = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    db_results = NEO4J_CONNECTOR.run_query(cypher_query) 
    
    # Converte o resultado para uma string formatada para passar à LLM (JSON ou String)
    if db_results is not None:
        query_result_str = json.dumps([dict(record) for record in db_results])
    else:
        query_result_str = "ERROR: Could not execute query or connection failed."
        
    logger.info(f"Resultado do DB: {query_result_str[:100]}...")
    return {"query_result": query_result_str}

def generate_textual_response(state: GraphState, llm: BaseChatOpenAI) -> Dict[str, Any]:
    """
    4. Segunda etapa da LLM: Formata o resultado da query em uma resposta textual.
    """
    logger.info("Etapa: Geração da Resposta Textual")
    
    # 1. Monta o System Prompt (Instruções de Formatação)
    # Reutiliza o template base, mas a instrução principal está no User Prompt
    system_message = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(
        graph_schema="", # Não precisa de schema/rag aqui
        rag_examples=""
    ))
    
    # 2. Monta o User Prompt (inclui RESULTADO, QUERY EXECUTADA e PERGUNTA)
    user_prompt_content = USER_PROMPT_TEXTUAL_RESPONSE.format(
        user_input=state["input"],
        cypher_query_executed=state["cypher_query"],
        query_result=state["query_result"]
    )
    human_message = HumanMessage(content=user_prompt_content)

    # 3. Envia para a LLM
    response = llm.invoke([system_message, human_message])
    
    return {"messages": [AIMessage(content=response.content)]}


# --- FLUXO DO LANGGRAPH ---


def create_langgraph_workflow(llm: BaseChatOpenAI):
    """Cria o workflow do LangGraph com os novos nós de 4 etapas."""
    
    # Funções parciais para injetar o LLM nos nós
    def generate_query_partial(state: GraphState):
        return generate_query(state, llm)
        
    def generate_textual_response_partial(state: GraphState):
        return generate_textual_response(state, llm)
    
    workflow = StateGraph(GraphState)
    
    # 1. Recupera contexto (RAG + Schema)
    workflow.add_node("retrieve_context", retrieve_context) 
    # 2. Gera a query Cypher (LLM 1)
    workflow.add_node("generate_query", generate_query_partial)
    # 3. Executa a query no DB
    workflow.add_node("execute_query", execute_query)
    # 4. Gera a resposta textual (LLM 2)
    workflow.add_node("generate_textual_response", generate_textual_response_partial)
    
    # Definindo o fluxo
    workflow.add_edge(START, "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_query")
    workflow.add_edge("generate_query", "execute_query")
    workflow.add_edge("execute_query", "generate_textual_response")
    workflow.add_edge("generate_textual_response", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app, memory


def initialize(model: str) -> Tuple[StateGraph, MemorySaver]:
    try:
        if model == "gpt-5-2025-08-07":
            llm = ChatOpenAI(
                model_name="gpt-5-2025-08-07",
                request_timeout=REQUEST_TIMEOUT
            )
        elif model == "gpt-4.1":
            llm = ChatOpenAI(
                model_name="gpt-4.1",
                temperature=0.7,
                max_tokens=MAX_TOKENS,
                request_timeout=REQUEST_TIMEOUT
            )
        elif model == "gpt-4.1-mini":
            llm = ChatOpenAI(
                model_name="gpt-4.1-mini",
                temperature=0.7,
                max_tokens=MAX_TOKENS,
                request_timeout=REQUEST_TIMEOUT
            )
        elif model == "o3":
            llm = ChatOpenAI(
                model_name="o3",
                request_timeout=REQUEST_TIMEOUT
            )
        elif model == "claude-4-sonnet":
            llm = ChatAnthropic(
                model='claude-sonnet-4-20250514',
                anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
                temperature=0.7,
                max_tokens=MAX_TOKENS,
                timeout=REQUEST_TIMEOUT
            )
        elif model == "deepseek-reasoner":
            llm = BaseChatOpenAI(
                model='deepseek-reasoner',
                openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
                openai_api_base='https://api.deepseek.com',
                temperature=0.7,
                max_tokens=MAX_TOKENS,
                request_timeout=REQUEST_TIMEOUT
            )
        else:
            llm = ChatOpenAI(
                model_name="gpt-4.1",
                temperature=0.7,
                max_tokens=MAX_TOKENS,
                request_timeout=REQUEST_TIMEOUT
            )

        app, memory = create_langgraph_workflow(llm)
        
        return app, memory

    except Exception as e:
        print(f"Error initializing RAG: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="SDDP Graph RAG Agent - Traduz perguntas em Cypher e responde.")
    parser.add_argument("-m", "--model", default="gpt-4.1", 
                        help="Modelo de LLM a ser usado (ex: gpt-4.1, claude-4-sonnet). Padrão: gpt-4.1")
    parser.add_argument("-s", "--study_path", required=True, 
                        help="Caminho para os arquivos de estudo SDDP necessários para carregar o esquema do grafo no Neo4j.")
    parser.add_argument("-q", "--query", required=True, 
                        help="A pergunta em linguagem natural a ser processada pelo agente.")
    
    args = parser.parse_args()

    model = args.model
    study_path = args.study_path
    user_input = args.query
    
    logger.info(f"--- Iniciando Agente RAG para SDDP ---")
    logger.info(f"Modelo selecionado: {model}")
    logger.info(f"Caminho do estudo: {study_path}")
    logger.info(f"Pergunta do usuário: '{user_input}'")

    try:
        global SCHEMA_DATA 
        SCHEMA_DATA = get_schema(study_path) 
        logger.info(f"✅ Esquema do Grafo carregado e Neo4j inicializado/atualizado.")
        
        # 3. Inicialização do LangGraph
        # A função 'initialize' cria o LLM e constrói o workflow de 4 etapas
        chain, memory = initialize(model)
        logger.info(f"✅ Workflow LangGraph e LLM inicializados.")

        # 4. Configuração da Execução (Gerenciamento de Conversação/Thread)
        # O 'thread_id' é crucial para persistir o histórico da conversa (checkpointing)
        thread_id = 1 # Use um ID estático ou gere um dinamicamente
        config = {"configurable": {"thread_id": thread_id}}

        # 5. Execução do Workflow (Invocação)
        logger.info("\n--- EXECUTANDO O WORKFLOW DO AGENTE (4 ETAPAS) ---")
        
        # A entrada inicial define o 'input' e zera as 'messages' para uma nova pergunta
        result = chain.invoke({
            "input": user_input,
            "messages": [] 
        }, config=config)

        # 6. Processamento da Resposta
        if "messages" in result and result["messages"]:
            # A resposta final está na última mensagem gerada pelo nó 'generate_textual_response'
            final_response = result["messages"][-1].content
            
            print("\n==============================================")
            print("✅ RESPOSTA FINAL DO AGENTE:")
            print(final_response)
            print("==============================================\n")
            
        else:
            logger.warning("O workflow foi executado, mas nenhuma mensagem de resposta foi gerada.")
            
    except Exception as e:
        logger.error(f"\n--- ERRO CRÍTICO NO PIPELINE ---")
        logger.error(f"Detalhe do erro: {type(e).__name__}: {str(e)}")
        sys.exit(1) # Finaliza o programa com erro