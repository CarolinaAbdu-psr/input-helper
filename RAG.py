import json
import argparse
import yaml
import datetime as dt
import os
import re
from typing import Tuple, List, Annotated, Dict, Any

import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
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

from tenacity import retry, stop_after_attempt, wait_exponential

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
# Creata GraphState
# -----------------------------

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    input: str
    context: str
    schema: str
    cypher_query: List[str]
    query_result: str
    chat_language: str
    agent_type: str


# -----------------------------
# Get study schema 
# -----------------------------

def create_schema(study_path: str, schema_path:str):
    "Create a schema and save it at the study folder"
    neo4j_uri = "neo4j://127.0.0.1:7687"
    neo4j_auth = ("neo4j", "psr-2025")

    # Load graph data
    G, load_times = data_loader(study_path)
    G, node_properties = extract_node_properties(G)
    nodes, edges = load_networkx_to_neo4j(
        G, node_properties, uri=neo4j_uri, auth=neo4j_auth, clear_existing_data=True
    )

    names = {}
    for obj in node_properties.values():
        name = obj.get('name')
        obj_type = obj.get('ObjType')
        if obj_type not in names:
            names[obj_type] = []
        names[obj_type].append(name)

    SCHEMA_DATA = f"Nodes with their possible names: {names}, Relationships: {edges}.  "

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
        SCHEMA_DATA = f.read()
    
    return SCHEMA_DATA

def get_schema(study_path: str, recreate_study: bool) -> str:
    """Create or load a schema summary file for the study.

    If `recreate_study` is True this will load the networkx study, push to Neo4j
    and extract a simple text representation saved to `schema.txt`.
    """
    schema_path = os.path.join(study_path, "schema.txt")

    if recreate_study:
        logger.info("Recreating Neo4j database and extracting schema from study...")
        SCHEMA_DATA = create_schema(study_path, schema_path)
        
    else:
        SCHEMA_DATA = load_existent_schema(schema_path)

    return SCHEMA_DATA

# -----------------------------
# Retrive context
# -----------------------------

def load_vectorstore() -> Chroma:
    """Load a Chroma vectorstore persisted in `vectorstore` directory."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = "vectorstore"
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
    raise ValueError(f"Vectorstore not found: {persist_directory}")


def format_contrastive_examples(docs: List) -> str:
    """Format contrastive RAG examples with correct and incorrect Cypher queries."""
    formatted_blocks = []
    for i, doc in enumerate(docs):
        metadata = getattr(doc, 'metadata', {})
        block = f"""
        ### Example {i + 1}
        Question: {doc.page_content}

        CORRECT SYNTAX (DO): ({metadata.get('correct_cypher_inst', 'N/A')})
        ```cypher
        {metadata.get('correct_cypher_query', 'N/A')}
        ```

        INCORRECT SYNTAX (DON'T DO): ({metadata.get('incorrect_cypher_inst', 'N/A')})
        ```cypher
        {metadata.get('incorrect_cypher_query','N/A')}
        ```
        """
        formatted_blocks.append(block.strip())

    return "\n\n" + "\n\n".join(formatted_blocks)


def retrieve_context(state: GraphState) -> Dict[str, Any]:
    """Retrieve RAG examples and attach schema context."""
    logger.info("Step: Retrieve Context")

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["input"])

    context_str = format_contrastive_examples(docs)

    logger.info("Examples context generated")

    return {
        "context": context_str,
        "schema": SCHEMA_DATA
    }


# -----------------------------
# Cypher generation helpers
# -----------------------------

def parse_multiple_queries(response) -> List[str]:
    """Parse a model response containing one or more Cypher queries.

    The function treats lines starting with '//' as separators and collects query blocks.
    """
    queries = []
    current_query = []
    content = getattr(response, 'content', str(response))
    print(content)
    for line in content.split('\n'):
        if line.strip().startswith('//') and current_query:
            queries.append('\n'.join(current_query).strip())
            current_query = []
        elif not line.strip().startswith('//'):
            current_query.append(line)
    if current_query:
        queries.append('\n'.join(current_query).strip())
    return [q for q in queries if q]


def generate_query(state: GraphState, llm: BaseChatOpenAI) -> Dict[str, Any]:
    """Generate Cypher queries from user input using the configured prompts and LLM."""

    logger.info("Step: Cypher Query Generation")
    system_prompt_content = SYSTEM_PROMPT_TEMPLATE.format(
        graph_schema=state["schema"],
        rag_examples=state["context"]
    )

    system_message = SystemMessage(content=system_prompt_content)
    user_prompt_content = USER_PROMPT_CYPHER_GENERATION.format(
        user_input=state["input"]
    )

    human_message = HumanMessage(content=user_prompt_content)
    response = llm.invoke([system_message, human_message])

    #Format multiple queries
    cypher_queries = parse_multiple_queries(response)
    logger.info(f"Generated {len(cypher_queries)} cypher queries: {cypher_queries}")

    return {"cypher_query": cypher_queries}


# -----------------------------
# Execution tools and auto-fix
# -----------------------------

def validate_cypher_query(query: str, schema: str) -> Dict[str, Any]:
    """Basic validation for Cypher queries.

    Checks balanced brackets, presence of MATCH/CREATE/MERGE and flags dangerous operations.
    """
    errors = []
    warnings = []
    if not query.strip():
        errors.append("Empty query")
        return {"is_valid": False, "errors": errors, "warnings": warnings}
    if query.count('(') != query.count(')'):
        errors.append("Unbalanced parentheses")
    if query.count('[') != query.count(']'):
        errors.append("Unbalanced square brackets")
    if query.count('{') != query.count('}'):
        errors.append("Unbalanced braces")

    query_upper = query.upper()
    has_match = 'MATCH' in query_upper
    has_return = 'RETURN' in query_upper
    has_create = 'CREATE' in query_upper
    has_merge = 'MERGE' in query_upper
    
    if not (has_match or has_create or has_merge):
        errors.append("Query must contain MATCH, CREATE or MERGE")
    if has_match and not has_return:
        warnings.append("MATCH without RETURN may not return data")

    dangerous_patterns = ['DELETE', 'DETACH DELETE', 'DROP', 'REMOVE']
    for pattern in dangerous_patterns:
        if pattern in query_upper:
            warnings.append(f"Query contains potentially dangerous operation: {pattern}")

    is_valid = len(errors) == 0

    return {"is_valid": is_valid, "errors": errors, "warnings": warnings}


class Neo4jExecutorWithRetry:
    """Executor that runs Cypher queries against Neo4j with retry logic."""

    def __init__(self, uri: str, username: str, password: str, max_retries: int = 3):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.max_retries = max_retries

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    def execute_with_retry(self, query: str) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def execute_query_safe(self, query: str, validate_first: bool = True) -> Dict[str, Any]:
        execution_log = []
        validation = None
        try:
            if validate_first:
                validation = validate_cypher_query(query, "")
                execution_log.append(f"Validation: {validation}")

                #Not valid 
                if not validation["is_valid"]:
                    return {
                        "success": False,
                        "data": None,
                        "records_count": 0,
                        "error": f"Validation failed: {validation['errors']}",
                        "validation": validation,
                        "execution_log": execution_log
                    }
                
                # Valid with warnings 
                if validation["warnings"]:
                    logger.warning(f"Validation warnings: {validation['warnings']}")

            # Execute query with retry 
            execution_log.append("Starting execution...")
            records = self.execute_with_retry(query)
            execution_log.append(f"Success: {len(records)} records returned")
            return {
                "success": True,
                "data": records,
                "records_count": len(records),
                "error": None,
                "validation": validation if validate_first else None,
                "execution_log": execution_log
            }
        
        except Exception as e:
            error_msg = str(e)
            execution_log.append(f"Error: {error_msg}")
            return {
                "success": False,
                "data": None,
                "records_count": 0,
                "error": error_msg,
                "validation": validation if validate_first else None,
                "execution_log": execution_log
            }

    def close(self):
        self.driver.close()


def auto_fix_cypher_query(query: str, error: str, schema: str, llm: ChatOpenAI) -> str:
    """Attempt to auto-fix a Cypher query given an error message using the LLM."""
    system_prompt = f"""You are a Neo4j/Cypher expert. A query failed with the following error.
    Please produce a corrected query only.

    GRAPH SCHEMA:
    {schema}

    ORIGINAL QUERY:
    {query}

    ERROR:
    {error}

    INSTRUCTIONS:
    - Fix only what is necessary for the query to run as intended.
    - Preserve the original logic if possible.
    - Return ONLY the corrected query, without additional text.
    """
    system_message = SystemMessage(content=system_prompt)
    human_message = HumanMessage(content="Please correct the query above.")
    response = llm.invoke([system_message, human_message])
    fixed_query = response.content.strip()
    if "```" in fixed_query:
        m = re.search(r'```(?:cypher)?\n(.*?)\n```', fixed_query, re.DOTALL)
        if m:
            fixed_query = m.group(1)
    return fixed_query


def execute_queries_with_auto_fix(
    queries: List[str],
    executor: Neo4jExecutorWithRetry,
    llm: ChatOpenAI,
    schema: str,
    max_fix_attempts: int = 2
) -> Dict[str, Any]:
    """Execute all queries, attempting automatic fixes if they fail.
    
    Runs all queries and stores their results. Returns a comprehensive result
    with all attempts and query results only after processing all queries.
    """
    all_attempts = []
    successful_queries = []
    failed_queries = []
    
    for q_idx, query in enumerate(queries, start=1): # For each query generated 
        logger.info(f"Attempting query {q_idx}/{len(queries)}")
        attempts_for_this_query = []
        current_query = query
        query_success = False
        
        for attempt in range(max_fix_attempts + 1): # Try until reachs maximum attempts 
            logger.info(f"  Try {attempt + 1}/{max_fix_attempts + 1}")
            result = executor.execute_query_safe(current_query, validate_first=True)
            attempt_info = {"attempt_number": attempt + 1, "query": current_query, "result": result}
            attempts_for_this_query.append(attempt_info)
            
            # Check if query succeeded
            if result["success"] and result["records_count"] > 0:
                logger.info(f"  Success with {result['records_count']} records")
                successful_queries.append({
                    "query_index": q_idx,
                    "final_query": current_query,
                    "result": result,
                    "attempts": attempts_for_this_query
                })
                query_success = True
                break
            
            # Try automatic fix if not on last attempt
            elif attempt < max_fix_attempts:
                logger.warning(f"Failed: {result.get('error', 'unknown')}")
                logger.info("  → Trying automatic correction...")
                try:
                    current_query = auto_fix_cypher_query(current_query, result.get('error',''), schema, llm)
                    logger.info("  → Corrected query generated")
                except Exception as e:
                    logger.error(f"  → Auto-fix error: {e}")
                    break
            else:
                logger.warning("Attempts exhausted for this query")
        
        # Store this query's attempts
        all_attempts.append({
            "query_index": q_idx,
            "original_query": query,
            "success": query_success,
            "attempts": attempts_for_this_query
        })
        
        # If query failed after all attempts, add to failed list
        if not query_success:
            failed_queries.append({
                "query_index": q_idx,
                "query": query,
                "attempts": attempts_for_this_query
            })
    
    # Return comprehensive result with all queries processed
    return {
        "success": len(successful_queries) > 0,
        "successful_queries": successful_queries,
        "failed_queries": failed_queries,
        "total_queries": len(queries),
        "successful_count": len(successful_queries),
        "failed_count": len(failed_queries),
        "all_attempts": all_attempts
    }


def execute_all_queries(state: GraphState, llm: BaseChatOpenAI) -> Dict[str, Any]:
    """Execute the generated Cypher queries using the executor with auto-fix support."""
    logger.info("Step: Running query")
    queries = state["cypher_query"]
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USER = os.getenv('NEO4J_USER')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    executor = Neo4jExecutorWithRetry(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        result = execute_queries_with_auto_fix(queries, executor, llm, state.get('schema', ''), max_fix_attempts=2)
        
        # Build comprehensive result summary
        summary = {
            "total_queries": result['total_queries'],
            "successful_count": result['successful_count'],
            "failed_count": result['failed_count'],
            "successful_queries": [],
            "failed_queries": []
        }
        
        # Collect data from successful queries
        for sq in result.get('successful_queries', []):
            summary["successful_queries"].append({
                "query_index": sq['query_index'],
                "query": sq['final_query'],
                "records": sq['result'].get('data', [])
            })
        
        # Collect info from failed queries
        for fq in result.get('failed_queries', []):
            summary["failed_queries"].append({
                "query_index": fq['query_index'],
                "query": fq['query'],
                "error": fq['attempts'][-1]['result'].get('error', 'Unknown error') if fq['attempts'] else 'No attempts'
            })
        
        query_result_str = json.dumps(summary, indent=2)
        logger.info(f"Query execution completed: {result['successful_count']} successful, {result['failed_count']} failed")
        return {"query_result": query_result_str}
    finally:
        executor.close()


# -----------------------------
# Generate textual final response
# -----------------------------

def generate_textual_response(state: GraphState, llm: BaseChatOpenAI) -> Dict[str, Any]:
    """Generate a user-facing answer using the executed query result."""
    logger.info("Step: Generate textual response")
    system_message = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(graph_schema="", rag_examples=""))
    user_prompt_content = USER_PROMPT_TEXTUAL_RESPONSE.format(
        user_input=state["input"],
        cypher_query_executed=state["cypher_query"],
        query_result=state["query_result"]
    )
    human_message = HumanMessage(content=user_prompt_content)
    response = llm.invoke([system_message, human_message])
    return {"messages": [AIMessage(content=response.content)]}


# -----------------------------
# Build LangGraph workflow
# -----------------------------

def create_langgraph_workflow(llm: BaseChatOpenAI):
    """Create LangGraph workflow that runs RAG -> generate cypher -> execute -> format answer."""
    def generate_query_partial(state: GraphState):
        return generate_query(state, llm)

    def generate_textual_response_partial(state: GraphState):
        return generate_textual_response(state, llm)

    def execute_query_partial(state: GraphState):
        return execute_all_queries(state, llm)

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_query", generate_query_partial)
    workflow.add_node("execute_query", execute_query_partial)
    workflow.add_node("generate_textual_response", generate_textual_response_partial)

    workflow.add_edge(START, "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_query")
    workflow.add_edge("generate_query", "execute_query")
    workflow.add_edge("execute_query", "generate_textual_response")
    workflow.add_edge("generate_textual_response", END)
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
            llm = ChatOpenAI(model_name="qwen3:14b", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT, base_url= "http://10.246.47.184:10000/v1")
        else:
            llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)

        app, memory = create_langgraph_workflow(llm)
        return app, memory
    
    except Exception as e:
        logger.error(f"Error initializing RAG: {e}")
        raise


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="RAG Agent - Translate questions to Cypher and answer them with RAG + auto-fix tools")
    parser.add_argument("-m", "--model", default="gpt-4.1", help="LLM model to be used. Default: gpt-4.1")
    parser.add_argument("-s", "--study_path", required=True, help="Path to the study files required to load the graph schema into Neo4j.")
    parser.add_argument("-q", "--query", required=True, help="Natural language question to be processed by the agent.")
    parser.add_argument("-r", "--recreate_study", default=False, help="If the neo4j database needs recreation, set True")
    args = parser.parse_args()
    model = args.model
    study_path = args.study_path
    user_input = args.query
    recreate_study = False if args.recreate_study=="False" else True

    logger.info("--- Starting RAG Agent ---")
    logger.info(f"Selected model: {model}")
    logger.info(f"Study path: {study_path}")
    logger.info(f"User question: {user_input}")
    try:
        global SCHEMA_DATA
        SCHEMA_DATA = get_schema(study_path, recreate_study)
        logger.info("Graph schema loaded and Neo4j initialized/updated.")
        chain, memory = initialize(model)
        logger.info("Workflow and LLM initialized.")
        thread_id = 1
        config = {"configurable": {"thread_id": thread_id}}
        logger.info("--- RUNNING WORKFLOW (4 STEPS) ---")
        result = chain.invoke({"input": user_input, "messages": []}, config=config)
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
        sys.exit(1)
