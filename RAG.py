import argparse
import yaml
import os
import networkx as nx
from typing import Tuple, List, Annotated, Dict, Any
from typing_extensions import TypedDict

import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.tools import tool

from langchain_openai import ChatOpenAI 

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from graph.data_loader import data_loader
from graph.get_data_properties import extract_node_properties


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# Load environment variables from .env file
load_dotenv()

REQUEST_TIMEOUT = 120
MAX_TOKENS = 4096

# Agent template settings
AGENTS_DIR = "agent"
CYPHER_AGENT_FILENAME = "agent.yaml"

# Global graph variable
G = None

# -----------------------------
# Load agent configuration
# -----------------------------

def load_cypher_agent_config(filepath: str) -> bool:
    """Load prompts and templates from the agent YAML configuration."""
    global SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_CYPHER_GENERATION, USER_PROMPT_TEXTUAL_RESPONSE
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        SYSTEM_PROMPT_TEMPLATE = config['system_prompt_template']
        USER_PROMPT_CYPHER_GENERATION = config['translation_task']['user_prompt_cypher_generation']
        USER_PROMPT_TEXTUAL_RESPONSE = config['formatting_task']['user_prompt_textual_response']

        logger.info(f"Cypher agent template loaded successfully from {filepath}")
        return True
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logger.error(f"Error loading Cypher agent from {filepath}: {e}")
        return False


def load_agents_config() -> Dict[str, Any]:
    """Load agent configuration file."""
    filepath = os.path.join(AGENTS_DIR, CYPHER_AGENT_FILENAME)
    if load_cypher_agent_config(filepath):
        return {'status': 'loaded'}
    return {'status': 'failed'}


# Load configuration once
_AGENTS_CONFIG = load_agents_config()

# -----------------------------
# Create GraphState
# -----------------------------

class GraphState(TypedDict):
    """Represents the state of our graph traversal."""
    messages: Annotated[List[BaseMessage], add_messages]
    input: str
    current_node_id: str | None # The Full_Id of the node currently being inspected
    visited_nodes: List[str] # List of Full_Ids already visited (to avoid cycles)
    retrieved_context: List[Dict[str, Any]] # Collected data from inspected nodes
    schema: str
    chat_language: str
    agent_type: str


# -----------------------------
# Get study schema 
# -----------------------------

def create_schema(study_path: str, schema_path:str):
    "Create a schema and save it at the study folder"

    # Load graph data
    global G
    G = data_loader(study_path)
    G, node_properties = extract_node_properties(G)
    nx.write_graphml(G, "grafo.graphml")

    entities = G.nodes(data='type')
    names = G.nodes(data='name')

    SCHEMA_DATA = f"Entities available : {entities} \n Nodes with their possible names: {names}."

    logger.info(f"Saving graph schema to {schema_path}...")
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write(SCHEMA_DATA)

    return SCHEMA_DATA

def load_existent_schema(schema_path: str):
    "Load the schema from an existent file"
    if not os.path.isfile(schema_path):
        logger.error(f"Schema file not found: {schema_path}. Please create the study.")
        raise FileNotFoundError(schema_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        global G
        G = nx.read_graphml("grafo.graphml")
        SCHEMA_DATA = f.read()
    print(SCHEMA_DATA)
    return SCHEMA_DATA

def get_schema(study_path: str, recreate_study: bool) -> str:
    """Create or load a schema summary file for the study.

    If `recreate_study` is True this will load the networkx study, push to Neo4j
    and extract a simple text representation saved to `schema.txt`.
    """
    schema_path = os.path.join(study_path, "schema.txt")

    if recreate_study:
        logger.info("Recreating graph database and extracting schema from study...")
        SCHEMA_DATA = create_schema(study_path, schema_path)
        
    else:
        SCHEMA_DATA = load_existent_schema(schema_path)

    return SCHEMA_DATA


# -----------------------------
# Define tools 
# -----------------------------

def get_entity(type: str) -> str:
    """Auxiliary function to find the Full_Id from the type."""
    global G
    ids = []
    if G is None:
        return None
    for full_id, data in G.nodes(data=True):
        if data.get("type", "").lower() == type.lower(): 
            ids.append(full_id)
    return ids 

def get_full_id_by_name(name: str) -> str:
    """Auxiliary function to find the Full_Id from the Name."""
    global G
    if G is None:
        return None
    for full_id, data in G.nodes(data=True):
        if data.get("name", "").lower() == name.lower(): 
            return full_id
    return None

def get_node_name_by_id(full_id: str) -> str:
    """Auxiliary function to find the Name from the Full_Id."""
    global G
    if G is None:
        return full_id
    return G.nodes[full_id].get("name", full_id) if full_id in G else full_id

@tool 
def search_by_entity(query:str)->str:
    """
    Searches nodes in the graph by their 'type'. 
    Returns a list of Full_Ids (the unique identifiers) of the nodes found.
    Use this tool as the first step to find entities of interest by type.
    """
    logger.info(f"Searching for entities of type: {query}")
    ids = get_entity(query)
    if ids:
        logger.info(f"Names found for type '{query}': {ids}")
        return f"Names found for type '{query}': {ids}"
    else:
        logger.info(f"No entities found for type '{query}'")
        return f"No entities found for type '{query}'"

@tool
def search_node(query: str) -> str:
    """
    Searches a node in the graph by its name or term. 
    Returns the node's Full_Id (the unique identifier) for use in other tools.
    Use this tool as the first step to find the entity of interest.
    """
    node_id = get_full_id_by_name(query)
    if node_id:
        logger.info(f"Node found: {node_id} for query '{query}'")
        return f"Node found! Full_Id: '{node_id}'. This node's 'Name' is '{get_node_name_by_id(node_id)}'."
    else:
        logger.info(f"Node not found for query '{query}'")
        return f"Node with the name '{query}' not found in the graph. Try broader search terms or exact entity names."

@tool
def inspect_node(full_id: str) -> str:
    """
    Queries the Type, Description, and all Properties/Attributes
    of a specific node using its Full_Id. The LLM uses Description for evaluation 
    and Properties for final extraction.
    """
    logger.info(f"Inspecting node with Full_Id: {full_id}")
    global G
    if G is None:
        return "Error: Graph not initialized."
    
    if full_id not in G:
        return f"Error: Full_Id '{full_id}' not found in the graph."
    
    node_data = G.nodes[full_id]
    
    result = {
        "Type": node_data.get("type", "Unknown"), 
        "Name": node_data.get("name", full_id), 
        "Description": node_data.get("Description", "No description available."), 
        "Properties": {k: v for k, v in node_data.items() if k not in ["type", "name", "Description"]} 
    }
    return str(result)

@tool
def traverse_graph(full_id: str) -> str:
    """
    Lists all neighbor nodes (out-going and in-coming) of a specific Full_Id, 
    along with the label of the edge connecting them. 
    The LLM uses this information for visit decision and navigation.
    """
    logger.info(f"Traversing graph for Full_Id: {full_id}")
    global G
    if G is None:
        return "Error: Graph not initialized."
    
    if full_id not in G:
        return f"Error: Full_Id '{full_id}' not found in the graph."

    neighbors_info = []
    
    # Outgoing edges
    for edge in list(G.out_edges(full_id, data=True)):
        label = edge[2].get("title", "NO_LABEL") 
        neighbors_info.append({
            "direction": "OUTGOING (Neighbor is the Object)", 
            "edge_label": label,
            "neighbor_full_id": edge[1], 
            "neighbor_name": get_node_name_by_id(edge[1]) 
        })

    # Incoming edges
    for edge in list(G.in_edges(full_id, data=True)):
        label = edge[2].get("title", "NO_LABEL") 
        neighbors_info.append({
            "direction": "INCOMING (Neighbor is the Subject)", 
            "edge_label": label,
            "neighbor_full_id": edge[0], 
            "neighbor_name": get_node_name_by_id(edge[0]) 
        })

    return str(neighbors_info)


# Define the list of tools
tools = [search_node, search_by_entity, inspect_node, traverse_graph]


# -----------------------------
# Decision nodes 
# ---------------------------

def initial_search_or_next_action(state: GraphState):
    """
    Initial node: The LLM decides the first action (usually calling 'search_node') 
    or decides the next step in the loop (usually calling 'inspect_node' or 'traverse_graph').
    """
    messages = state["messages"]
    
    # Add system message with instructions if this is the first call
    if len(messages) == 1:  # Only the user's question
        system_msg = SystemMessage(content=f"""You are a graph traversal assistant. Your goal is to answer the user's question by exploring a knowledge graph.

Available tools:
1. search_node(query): Find a node by name - use this first to locate entities
2. inspect_node(full_id): Get detailed information about a node
3. traverse_graph(full_id): List neighbors of a node to explore relationships

Graph schema represents the available entities : {state.get('schema', 'No schema available')}

Strategy:
1. First, use search_node to find relevant entities
2. Then use inspect_node to get their details
3. Use traverse_graph to explore relationships if needed
4. When you have enough information, provide a final answer without calling more tools

Current context collected: {state.get('retrieved_context', [])}
""")
        messages = [system_msg] + messages
    
    # The LLM outputs an AIMessage with a tool call
    response = llm_with_tools.invoke(messages) 
    
    return {"messages": [response]}


def tools_condition(state: GraphState) -> str:
    """
    Check if the last message contains tool calls.
    Routes to 'tools' if tool calls exist, otherwise END.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the AI message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END


def check_sufficiency(state: GraphState) -> str:
    """
    Implements Step 2 (Judgment): Decide whether to stop, inspect a new node, 
    or continue exploring neighbors.

    Returns the next node name: 'generate_answer' or 'search_or_action'.
    """
    # Get the latest tool result
    messages = state["messages"]
    last_message = messages[-1]
    
    # Update retrieved context with the tool result
    retrieved_context = state.get("retrieved_context", [])
    if hasattr(last_message, 'content'):
        retrieved_context.append({
            "content": last_message.content,
            "timestamp": len(retrieved_context)
        })
    
    # Compile the full context
    context_str = "\n".join([f"- {item['content']}" for item in retrieved_context])
    
    # Get the original question
    original_question = state["messages"][0].content if state["messages"] else state.get("input", "")
    
    # Ask the LLM to judge sufficiency
    judgment_prompt = f"""Original Question: {original_question}

Context Collected So Far:
{context_str}

Is the collected context sufficient to answer the original question completely and accurately?

Respond with EXACTLY one word:
- SUFFICIENT: if you can now answer the question with confidence
- TRAVERSE: if you need to explore more nodes or relationships

Your response:"""
    
    judgment_response = llm_judgment.invoke([HumanMessage(content=judgment_prompt)])
    judgment = judgment_response.content.strip().upper()
    
    logger.info(f"Sufficiency judgment: {judgment}")
    
    if "SUFFICIENT" in judgment:
        return "generate_answer"
    else:
        return "search_or_action"


def generate_answer(state: GraphState) -> Dict[str, Any]:
    """
    Generate the final answer based on collected context.
    """
    retrieved_context = state.get("retrieved_context", [])
    context_str = "\n".join([f"- {item['content']}" for item in retrieved_context])
    
    original_question = state["messages"][0].content if state["messages"] else state.get("input", "")
    
    answer_prompt = f"""Based on the following context from the knowledge graph, answer the user's question.

Original Question: {original_question}

Context:
{context_str}

Provide a clear, concise answer based solely on the information gathered. If the context doesn't fully answer the question, acknowledge what information is missing.

Your answer:"""
    
    response = llm_judgment.invoke([HumanMessage(content=answer_prompt)])
    
    return {
        "messages": [AIMessage(content=response.content)],
        "context": context_str
    }


# -----------------------------
# Workflow
# ---------------------------

def create_langgraph_workflow(llm: BaseChatOpenAI):
    """Create and compile the LangGraph workflow."""
    global llm_with_tools, llm_judgment
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create a separate LLM for judgment (no tools)
    llm_judgment = llm
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    workflow = StateGraph(GraphState)

    # --- Define Nodes ---
    workflow.add_node("search_or_action", initial_search_or_next_action)
    workflow.add_node("call_tool", tool_node)
    workflow.add_node("generate_answer", generate_answer)

    # --- Define Edges ---

    # 1. START: Initial Search
    workflow.add_edge(START, "search_or_action")

    # 2. Decision on Tool Call
    workflow.add_conditional_edges(
        "search_or_action",
        tools_condition,
        {
            "tools": "call_tool",
            END: END,
        },
    )

    # 3. After Tool Execution: Check sufficiency
    workflow.add_conditional_edges(
        "call_tool",
        check_sufficiency,
        {
            "generate_answer": "generate_answer",
            "search_or_action": "search_or_action",
        },
    )

    # 4. Final Edge
    workflow.add_edge("generate_answer", END)

    # Compile
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app, memory


def initialize(model: str) -> Tuple[StateGraph, MemorySaver]:
    """Initialize the LLM and return the compiled LangGraph workflow and memory."""

    try:
        if model == "gpt-5-2025-08-07":
            llm = ChatOpenAI(model_name="gpt-5-2025-08-07", request_timeout=REQUEST_TIMEOUT)
        elif model == "gpt-4.1":
            llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)
        elif model == "gpt-4.1-mini":
            llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)
        elif model == "o3":
            llm = ChatOpenAI(model_name="o3", request_timeout=REQUEST_TIMEOUT)
        elif model == "claude-4-sonnet":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model='claude-sonnet-4-20250514', anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'), temperature=0.7, max_tokens=MAX_TOKENS, timeout=REQUEST_TIMEOUT)
        elif model == "deepseek-reasoner":
            llm = BaseChatOpenAI(model='deepseek-reasoner', openai_api_key=os.getenv('DEEPSEEK_API_KEY'), openai_api_base='https://api.deepseek.com', temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)
        elif model == "local_land":
            llm = ChatOpenAI(model_name="qwen3:14b", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT, base_url="http://10.246.47.184:10000/v1")
        else:
            llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)

        app, memory = create_langgraph_workflow(llm)
        return app, memory
    
    except Exception as e:
        logger.error(f"Error initializing RAG: {e}")
        raise


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="RAG Agent - Graph traversal with tools to answer questions")
    parser.add_argument("-m", "--model", default="gpt-4.1", help="LLM model to be used. Default: gpt-4.1")
    parser.add_argument("-s", "--study_path", required=True, help="Path to the study files required to load the graph schema.")
    parser.add_argument("-q", "--query", required=True, help="Natural language question to be processed by the agent.")
    parser.add_argument("-r", "--recreate_study", default="False", help="If the database needs recreation, set True")
    args = parser.parse_args()
    model = args.model
    study_path = args.study_path
    user_input = args.query
    recreate_study = args.recreate_study.lower() == "true"

    logger.info("--- Starting RAG Agent ---")
    logger.info(f"Selected model: {model}")
    logger.info(f"Study path: {study_path}")
    logger.info(f"User question: {user_input}")
    
    try:
        # Load schema and initialize graph
        SCHEMA_DATA = get_schema(study_path, recreate_study)
        logger.info("Graph schema loaded successfully.")
        
        # Initialize workflow
        chain, memory = initialize(model)
        logger.info("Workflow and LLM initialized.")
        
        # Run workflow
        thread_id = "1"
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info("--- RUNNING WORKFLOW ---")
        initial_state = {
            "input": user_input,
            "messages": [HumanMessage(content=user_input)],
            "schema": SCHEMA_DATA,
            "retrieved_context": [],
            "visited_nodes": []
        }
        
        result = chain.invoke(initial_state, config=config)
        
        if "messages" in result and result["messages"]:
            final_response = result["messages"][-1].content
            print("\n==============================================")
            print("AGENT FINAL RESPONSE:")
            print(final_response)
            print("==============================================\n")
        else:
            logger.warning("Workflow executed but no answer generated.")
            
    except Exception as e:
        logger.error("--- CRITICAL ERROR ---")
        logger.error(f"Error Detail: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)