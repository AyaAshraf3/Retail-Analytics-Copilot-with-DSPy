"""
LangGraph Agent Implementation: Retail Analytics Copilot

7 Nodes: Router → Retriever → Planner → NL2SQL → Executor → Synthesizer → Repair

State management and graph orchestration
"""

import json
import logging
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime

import dspy
from langgraph.graph import StateGraph, END

from .dspy_signatures import RouterModule, NLToSQLModule, SynthesizerModule, PlannerModule
from .rag.retrieval import DocumentRetriever
from .tools.sqlite_tool import SQLiteTool

logger = logging.getLogger(__name__)

def setup_agent_logging():
    """Setup logging for agent."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"agent_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )



@dataclass
class AgentState:
    """Complete state management for the agent."""
    question: str = ""
    format_hint: str = "str"
    route: Optional[str] = None
    rag_results: List[Dict[str, Any]] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    sql_query: str = ""
    sql_results: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Any = None
    explanation: str = ""
    citations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    error: Optional[str] = None
    repair_count: int = 0
    trace: List[str] = field(default_factory=list)
    needs_sql: bool = False
    sql_valid: bool = False
    output_valid: bool = False


class RetailAnalyticsCopilot:
    """Main agent orchestrator using LangGraph."""

    def __init__(
        self,
        db_path: str = "data/northwind.sqlite",
        doc_dir: str = "docs",
        model: str = "phi3.5:3.8b-mini-instruct-q4_K_M",
        max_tokens: int = 1000,
        temperature: float = 0.1
    ):
        """
        Initialize the copilot agent.

        Args:
            db_path: Path to SQLite database
            doc_dir: Directory containing documentation
            model: Ollama model name
            max_tokens: Maximum tokens for LLM
            temperature: Temperature for LLM
        """
        self.db_path = db_path
        self.doc_dir = doc_dir

        # Initialize DSPy with Ollama
        self._setup_dspy(model, max_tokens, temperature)

        # Initialize components
        self.retriever = DocumentRetriever(doc_dir=doc_dir)
        self.db = SQLiteTool(db_path)
        self.db_schema = self.db.get_schema()

        # Initialize DSPy modules
        self.router = RouterModule()
        self.planner = PlannerModule()
        self.nltosql = NLToSQLModule()
        self.synthesizer = SynthesizerModule()

        # Build graph
        self.graph = self._build_graph()

    def _setup_dspy(self, model: str, max_tokens: int, temperature: float):
        """Configure DSPy with Ollama."""
        try:
            lm = dspy.LM(
                model=f"ollama/{model}",
                api_base='http://localhost:11434',
                api_key='',
                max_tokens=max_tokens,
                temperature=temperature
            )
            dspy.settings.configure(lm=lm)
            logger.info(f"DSPy configured with Ollama model: {model}")
        except Exception as e:
            logger.warning(f"Could not configure Ollama: {e}. Continuing with default LM.")


    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("router", self.node_router)
        workflow.add_node("retriever", self.node_retriever)
        workflow.add_node("planner", self.node_planner)
        workflow.add_node("nltosql", self.node_nltosql)
        workflow.add_node("executor", self.node_executor)
        workflow.add_node("synthesizer", self.node_synthesizer)
        workflow.add_node("repair", self.node_repair)

        # Add edges
        workflow.set_entry_point("router")

        # Router → Retriever or Planner based on route
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "retriever": "retriever",
                "planner": "planner",
            }
        )

        # Retriever → Planner
        workflow.add_edge("retriever", "planner")

        # Planner → NL2SQL or Synthesizer
        workflow.add_conditional_edges(
            "planner",
            self._needs_sql,
            {
                "nltosql": "nltosql",
                "synthesizer": "synthesizer",
            }
        )

        # NL2SQL → Executor
        workflow.add_edge("nltosql", "executor")

        # Executor → Synthesizer or Repair
        workflow.add_conditional_edges(
            "executor",
            self._executor_decision,
            {
                "synthesizer": "synthesizer",
                "repair": "repair",
            }
        )

        # Repair → NL2SQL or Synthesizer
        workflow.add_conditional_edges(
            "repair",
            self._repair_decision,
            {
                "nltosql": "nltosql",
                "synthesizer": "synthesizer",
            }
        )

        # Synthesizer → Output validation
        workflow.add_conditional_edges(
            "synthesizer",
            self._output_valid,
            {
                "end": END,
                "repair": "repair",
            }
        )

        return workflow.compile()

    # ==================== NODE IMPLEMENTATIONS ====================

    def node_router(self, state: AgentState) -> AgentState:
        """Node 1: Route query to RAG, SQL, or hybrid."""
        logger.info(f"[ROUTER] Processing: {state.question[:60]}...")
        route = self.router(state.question)
        state.route = route
        state.trace.append(f"Route: {route}")
        logger.info(f"[ROUTER] Route decision: {route}")
        return state

    def node_retriever(self, state: AgentState) -> AgentState:
        """Node 2: Retrieve relevant documents using BM25."""
        logger.info("[RETRIEVER] Retrieving documents...")
        results = self.retriever.retrieve(state.question, topk=3)
        state.rag_results = results
        state.citations.extend([r["id"] for r in results])
        state.trace.append(f"Retrieved {len(results)} documents")
        logger.info(f"[RETRIEVER] Retrieved {len(results)} chunks")
        return state

    def node_planner(self, state: AgentState) -> AgentState:
        """Node 3: Extract constraints from question and documents."""
        logger.info("[PLANNER] Extracting constraints...")
        constraints = self.planner(state.question, state.rag_results)
        state.constraints = constraints
        state.trace.append("Constraints extracted")
        logger.info(f"[PLANNER] Constraints: {constraints}")
        return state

    def node_nltosql(self, state: AgentState) -> AgentState:
        """Node 4: Generate SQL query using DSPy with validation."""
        logger.info("[NL2SQL] Generating SQL...")
        sql = self.nltosql(
            question=state.question,
            schema=self.db_schema,
            constraints=state.constraints,
            validator=self.db.validate_query  
        )
        state.sql_query = sql
        state.trace.append(f"Generated SQL: {sql[:80]}...")
        logger.info(f"[NL2SQL] Generated: {sql}")
        return state

    def node_executor(self, state: AgentState) -> AgentState:
        """Node 5: Execute SQL and capture results or errors."""
        logger.info("[EXECUTOR] Executing SQL...")
        is_valid, error = self.db.validate_query(state.sql_query)
        state.sql_valid = is_valid

        if is_valid:
            result = self.db.execute_query(state.sql_query)
            if result["error"]:
                state.error = result["error"]
                state.trace.append(f"SQL error: {state.error}")
                logger.error(f"[EXECUTOR] Execution error: {state.error}")
            else:
                state.sql_results = result["rows"]
                for table in self._extract_table_names(state.sql_query):
                    if table not in state.citations:
                        state.citations.append(table)
                state.trace.append(f"Executed, got {len(result['rows'])} rows")
                logger.info(f"[EXECUTOR] Success: {len(result['rows'])} rows")
        else:
            state.error = error
            state.sql_valid = False
            state.trace.append(f"SQL validation error: {error}")
            logger.error(f"[EXECUTOR] Validation error: {error}")

        return state

    def node_synthesizer(self, state: AgentState) -> AgentState:
        """Node 6: Synthesize typed answer with citations."""
        logger.info("[SYNTHESIZER] Synthesizing answer...")
        try:
            result = self.synthesizer(
                question=state.question,
                sql_results=state.sql_results,
                rag_results=state.rag_results,
                format_hint=state.format_hint,
                citations=state.citations
            )

        
            if isinstance(result, dict):
                state.final_answer = result.get("answer") or result.get("final_answer")
                state.explanation = result.get("explanation", "")
                state.confidence = result.get("confidence", 0.0)
            else:
                logger.error(f"[SYNTHESIZER] Unexpected result type: {type(result)}")
                state.final_answer = None
                state.explanation = "Error: Unexpected result format"
                state.confidence = 0.0
                return state

            state.trace.append("Answer synthesized")
            state.output_valid = self._validate_output_type(state.final_answer, state.format_hint)
            logger.info(f"[SYNTHESIZER] Answer: {str(state.final_answer)[:60]}...")
        except Exception as e:
            logger.error(f"[SYNTHESIZER] Error: {e}", exc_info=True)
            state.final_answer = None
            state.explanation = f"Error: {str(e)}"
            state.confidence = 0.0
            state.error = str(e)

        return state


    def node_repair(self, state: AgentState) -> AgentState:
        """Node 7: Repair logic with retry budget (≤2 attempts)."""
        logger.info(f"[REPAIR] Attempt {state.repair_count + 1}...")
        state.repair_count += 1

        if state.repair_count >= 2:
            state.trace.append("Max repair attempts reached")
            logger.info("[REPAIR] Max attempts reached")
            return state

        # Repair strategies
        if state.error and not state.sql_valid:
            # SQL validation/execution failed → try simpler query
            state.sql_query = "SELECT 1 as result;"  # Fallback
            state.trace.append("Repair: Using fallback query")
            logger.info("[REPAIR] Fallback to simple query")

        elif not state.output_valid:
            # Output format wrong → re-synthesize with stronger prompt
            state.final_answer = None
            state.trace.append("Repair: Re-synthesizing output")
            logger.info("[REPAIR] Re-synthesizing")

        return state

    # ==================== CONDITIONAL EDGES ====================

    def _route_decision(self, state: AgentState) -> str:
        """Decide whether to use retriever or go straight to planner."""
        if state.route in ["rag", "hybrid"]:
            return "retriever"
        else:  # "sql"
            return "planner"

    def _needs_sql(self, state: AgentState) -> str:
        """Decide if SQL is needed."""
        if state.route in ["sql", "hybrid"]:
            state.needs_sql = True
            return "nltosql"
        else:  # "rag"
            state.needs_sql = False
            return "synthesizer"

    def _executor_decision(self, state: AgentState) -> str:
        """Decide if executor succeeded or needs repair."""
        if state.error or not state.sql_valid:
            return "repair"
        else:
            return "synthesizer"

    def _repair_decision(self, state: AgentState) -> str:
        """Decide if repair should retry SQL or go to synthesizer."""
        if state.repair_count < 2 and not state.sql_valid:
            return "nltosql"
        else:
            return "synthesizer"

    def _output_valid(self, state: AgentState) -> str:
        """Decide if output format is valid."""
        if state.output_valid or state.repair_count >= 2:
            return "end"
        else:
            return "repair"

    # ==================== UTILITIES ====================

    def _extract_table_names(self, sql: str) -> List[str]:
        """Extract table names from SQL query."""
        import re
        # Simple extraction: FROM/JOIN table_name
        tables = re.findall(r'(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE)
        return list(set(tables))

    def _validate_output_type(self, value: Any, format_hint: str) -> bool:
        """Validate if output matches expected type."""
        format_hint = format_hint.lower().strip()
        if value is None:
            return False

        try:
            if "int" in format_hint:
                return isinstance(value, int)
            elif "float" in format_hint:
                return isinstance(value, float)
            elif "list" in format_hint or "[" in format_hint:
                return isinstance(value, list)
            elif "dict" in format_hint or "{" in format_hint:
                return isinstance(value, dict)
            else:
                return isinstance(value, str)
        except:
            return False

    def run(self, question: str, format_hint: str = "str") -> AgentState:
        """Run the agent on a question."""
        initial_state = AgentState(question=question, format_hint=format_hint)
        
        logger.info(f"Starting agent for: {question}")
        
        try:
            # graph.invoke() returns a DICT
            result_dict = self.graph.invoke(initial_state)
            
            # Convert dict back to AgentState object
            if isinstance(result_dict, dict):
                final_state = AgentState(
                    question=result_dict.get("question", ""),
                    format_hint=result_dict.get("format_hint", "str"),
                    route=result_dict.get("route"),
                    rag_results=result_dict.get("rag_results", []),
                    constraints=result_dict.get("constraints", {}),
                    sql_query=result_dict.get("sql_query", ""),
                    sql_results=result_dict.get("sql_results", []),
                    final_answer=result_dict.get("final_answer"),
                    explanation=result_dict.get("explanation", ""),
                    citations=result_dict.get("citations", []),
                    confidence=result_dict.get("confidence", 0.0),
                    error=result_dict.get("error"),
                    repair_count=result_dict.get("repair_count", 0),
                    trace=result_dict.get("trace", []),
                    needs_sql=result_dict.get("needs_sql", False),
                    sql_valid=result_dict.get("sql_valid", False),
                    output_valid=result_dict.get("output_valid", False),
                )
            else:
                final_state = result_dict
            
            logger.info(f"Finished. Answer: {final_state.final_answer}")
            return final_state 
            
        except Exception as e:
            logger.error(f"Agent failed: {e}", exc_info=True)
            initial_state.error = str(e)
            return initial_state


    def close(self):
        """Clean up resources."""
        self.db.close()
        logger.info("Agent closed")
