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

logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# Load environment variables from .env file
load_dotenv()


REQUEST_TIMEOUT = 120
MAX_TOKENS = 4096

# -------------------------------------------------------
# Load agent template
#--------------------------------------------------------

AGENTS_DIR = "helper"
CYPHER_AGENT_FILENAME = "agent.yaml"

def load_cypher_agent_config(filepath: str) -> bool:
    """Load templates to global variables"""
    global SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_CYPHER_GENERATION, USER_PROMPT_TEXTUAL_RESPONSE
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        SYSTEM_PROMPT_TEMPLATE = config['system_prompt_template']
        USER_PROMPT_CYPHER_GENERATION = config['translation_task']['user_prompt_cypher_generation']
        USER_PROMPT_TEXTUAL_RESPONSE = config['formatting_task']['user_prompt_textual_response']
        
        logger.info(f"Cyphr agent template loaded with succes from {filepath}")
        return True
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logger.error(f"Error loading Cypher agente from {filepath}: {e}")
        return False


def load_agents_config() -> Dict[str, Any]:
    """Loads Text-to-Cypher agent configuration."""
    filepath = os.path.join(AGENTS_DIR, CYPHER_AGENT_FILENAME)
    if load_cypher_agent_config(filepath):
        return {'status': 'loaded'}
    return {'status': 'failed'}

# Load configuration once
_AGENTS_CONFIG = load_agents_config()

# -------------------------------------------------------
# Functions to create State Nodes using LangGraph  
#--------------------------------------------------------

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    input: str 
    context: str # Cypher Examples
    schema: str # Graph Entities, names 
    cypher_query: str 
    query_result: str 
    chat_language: str
    agent_type: str

# -------------------------------------------------------
# Step 1 : Retrive Context 
#--------------------------------------------------------

def get_schema(study_path):

    """1.1. Create a Neo4j Database to the case and save important informations
    at SCHEMA_DATA"""

    neo4j_uri = "neo4j://127.0.0.1:7687"
    neo4j_auth = ("neo4j", "psr-2025")

    #Load Graph 
    G, load_times = data_loader(study_path)
    node_properties = extract_node_properties(G)
    nodes, edges = load_networkx_to_neo4j(G, node_properties, uri=neo4j_uri, auth=neo4j_auth,clear_existing_data=True)
    
    # Get Entities Names (nodes), Relatioships Names (edges) and Objects Names (names)
    names = []
    for obj in node_properties.values(): 
        name = obj.get('name')
        names.append(name)
    SCHEMA_DATA = f"""Nodes:{nodes}, 
    Relationships: {edges}, 
    Names: {names}"""

    return SCHEMA_DATA

# 1.2. Examples that will be substitute by an file of example to be used by the retriver 
rag_data = """pergunta_natural": "Qual a soma da capacidade instalada das térmicas que usam Gás como combustível?",
    "cypher_query": MATCH (tp:ThermalPlant)-[:Ref_Fuel]->(fuel:Fuel) WHERE fuel.name = "gas"
    RETURN sum(tp.InstCap) AS total_installed_capacity_gas;

    "pergunta_natural: Qual a soma da capacidade instalada das plantas térmicas?"
    "cypher_query: MATCH (tp:ThermalPlant)
    RETURN sum(tp.InstCap) AS total_installed_capacity_gas;
  
    "pergunta_natural": "Liste o nome das restrições ligadas à usina de Belo Monte.",
    "cypher_query": "MATCH (u:FactoryElement {name: 'Belo Monte'})-[:LINKED_TO]->(r:Restricao) RETURN r.nome"
"""

def retrieve_context(state: GraphState) -> Dict[str, Any]:
    """
    1. Get Cypher Examples (rag data) and get Graph Schema
    """
    logger.info("Step: Retrive Context")
    
    
    return {
        "context": rag_data,
        "schema": SCHEMA_DATA 
    }


# -------------------------------------------------------
# Step 2 : Generate Cypher Query 
#--------------------------------------------------------

def generate_query(state: GraphState, llm: BaseChatOpenAI) -> Dict[str, Any]:
    """
    2. LLM First Step: Traslate Question to Cypher Query 
    """
    logger.info("Step: Cypher Query Generation")
    
    # 2.1. Create System Prompt 
    system_prompt_content = SYSTEM_PROMPT_TEMPLATE.format(
        graph_schema=state["schema"],
        rag_examples=state["context"]
    )
    system_message = SystemMessage(content=system_prompt_content)
    
    # 2.2. Create User Prompt 
    user_prompt_content = USER_PROMPT_CYPHER_GENERATION.format(
        user_input=state["input"]
    )
    human_message = HumanMessage(content=user_prompt_content)

    # 2.3. Send to LLM 
    response = llm.invoke([system_message, human_message])
    
    # 2.4. Cypher Query Response 
    cypher_query = response.content.strip()
    
    logger.info(f"Query Cypher generated: {cypher_query[:200]}...")
    return {"cypher_query": cypher_query}

# -------------------------------------------------------
# Step 3: Run query   
#--------------------------------------------------------

def execute_query(state: GraphState) -> Dict[str, Any]:
    """
    3. Executes Cypher Query on Neo4j Database
    """
    logger.info("Step: Running query")
    
    # 3.1 Get query generated by step 2
    cypher_query = state["cypher_query"]
    
    # 3.2. Run query on sddp graph database
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "psr-2025"
    NEO4J_CONNECTOR = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    db_results = NEO4J_CONNECTOR.run_query(cypher_query) 
    
    # 3.3. Convert results to str 
    if db_results is not None:
        query_result_str = json.dumps([dict(record) for record in db_results])
    else:
        query_result_str = "ERROR: Could not execute query or connection failed."
        
    logger.info(f"Query Result: {query_result_str[:100]}...")
    return {"query_result": query_result_str}

# -------------------------------------------------------
# Step 4: Generate Textual Response  
#--------------------------------------------------------

def generate_textual_response(state: GraphState, llm: BaseChatOpenAI) -> Dict[str, Any]:
    """
    4. LLM Second Step : Use query result to generate textual response
    """
    logger.info("Etapa: Geração da Resposta Textual")
    
    # 4.1. Create Sytem Prompt 
    system_message = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(
        graph_schema="", 
        rag_examples=""
    ))
    
    # 4.2. Crete user prompt with question, cypher query and query result
    user_prompt_content = USER_PROMPT_TEXTUAL_RESPONSE.format(
        user_input=state["input"],
        cypher_query_executed=state["cypher_query"],
        query_result=state["query_result"]
    )
    human_message = HumanMessage(content=user_prompt_content)

    # 4.3. Invoke LLM
    response = llm.invoke([system_message, human_message])
    
    return {"messages": [AIMessage(content=response.content)]}


# -------------------------------------------------------
# Run steps with LangGraph and selected LLM
#--------------------------------------------------------

def create_langgraph_workflow(llm: BaseChatOpenAI):
    """Create LangGraph workflow with 4 steps"""
    
    # Get LLM results to inject on nodes
    def generate_query_partial(state: GraphState):
        return generate_query(state, llm)
        
    def generate_textual_response_partial(state: GraphState):
        return generate_textual_response(state, llm)
    
    workflow = StateGraph(GraphState)
    
    # 1. Retrive Context (RAG + Schema)
    workflow.add_node("retrieve_context", retrieve_context) 
    # 2. Generate cypher query 
    workflow.add_node("generate_query", generate_query_partial)
    # 3. Run query on Neo4j database 
    workflow.add_node("execute_query", execute_query)
    # 4. Generate textual response
    workflow.add_node("generate_textual_response", generate_textual_response_partial)
    
    # Workflow
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

# -------------------------------------------------------
# Main workflow
#--------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="SDDP Graph RAG Agent - Translates questions into Cypher and answers them.")
    parser.add_argument("-m", "--model", default="gpt-4.1", 
                         help="LLM model to be used (e.g., gpt-4.1, claude-4-sonnet). Default: gpt-4.1")
    parser.add_argument("-s", "--study_path", required=True, 
                         help="Path to the SDDP study files required to load the graph schema into Neo4j.")
    parser.add_argument("-q", "--query", required=True, 
                         help="The natural language question to be processed by the agent.")
    
    args = parser.parse_args()

    model = args.model
    study_path = args.study_path
    user_input = args.query
    
    logger.info(f"--- Starting SDPP Agent RAG---")
    logger.info(f"Selected LLM model: {model}")
    logger.info(f"SDDP Study Path: {study_path}")
    logger.info(f"User question: '{user_input}'")

    try:
        # 1. Get Schema and create neo4j database 
        global SCHEMA_DATA 
        SCHEMA_DATA = get_schema(study_path) 
        logger.info(f"✅ Graph Schema loaded and Neo4j initialized/updated.")
        
        # 2. Initialize LangGraph Workflow
        chain, memory = initialize(model)
        logger.info(f"✅ Workflow LangGraph e LLM inicializados.")

        # 3. Configuration 
        thread_id = 1 # Use um ID estático ou gere um dinamicamente
        config = {"configurable": {"thread_id": thread_id}}

        # 4. Run Workflow
        logger.info("\n--- EXECUTANDO O WORKFLOW DO AGENTE (4 ETAPAS) ---")
        
        # Get input and cleaning previous messages
        result = chain.invoke({
            "input": user_input,
            "messages": [] 
        }, config=config)

        # 5. Process Response
        if "messages" in result and result["messages"]:
            # A resposta final está na última mensagem gerada pelo nó 'generate_textual_response'
            final_response = result["messages"][-1].content
            
            print("\n==============================================")
            print("✅ AGENT FINAL RESPONSE:")
            print(final_response)
            print("==============================================\n")
            
        else:
            logger.warning("Workflow executed, but no answer generated .")
            
    except Exception as e:
        logger.error(f"\n--- CRITICAL ERROR ---")
        logger.error(f"Erro Detail: {type(e).__name__}: {str(e)}")
        sys.exit(1) 