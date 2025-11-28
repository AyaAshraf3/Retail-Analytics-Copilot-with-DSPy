"""
Main Entrypoint: Retail Analytics Copilot CLI
Process batch of questions and generate evaluation output
"""

import click
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict
from datetime import datetime

from agent.agent_graph import RetailAnalyticsCopilot, AgentState
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
console = Console()

def setup_logging():
    """Setup logging to console and file."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"batch_evaluation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def load_evaluation_file(filepath: str) -> List[Dict[str, Any]]:
    """Load evaluation questions from JSONL file."""
    questions = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        logger.info(f"Loaded {len(questions)} questions from {filepath}")
        return questions
    except Exception as e:
        console.print(f"[red]Error loading evaluation file: {e}[/red]")
        return []


def format_answer(answer: Any, format_hint: str) -> Any:
    """Format answer for JSON output."""
    try:
        if answer is None:
            return None

        format_hint = format_hint.lower().strip()

        if "int" in format_hint:
            return int(float(answer)) if answer is not None else None
        elif "float" in format_hint:
            return round(float(answer), 2) if answer is not None else None
        elif "list" in format_hint or "[" in format_hint:
            if isinstance(answer, list):
                return answer
            return [answer] if answer else []
        elif "dict" in format_hint or "{" in format_hint:
            if isinstance(answer, dict):
                return answer
            return {"value": answer} if answer else {}
        else:
            return str(answer) if answer else ""

    except Exception as e:
        logger.warning(f"Error formatting answer: {e}")
        return answer


def generate_output_row(question_id: str, question_obj: Dict, state: AgentState) -> Dict[str, Any]:
    """Generate output row matching the Output Contract."""
    
    # Safely get final_answer, handling None and conversion
    final_answer = getattr(state, 'final_answer', None)
    
    return {
        "id": question_id,
        "finalanswer": format_answer(final_answer, state.format_hint),
        "sql": state.sql_query if state.needs_sql else "",
        "confidence": round(state.confidence, 2) if state.confidence else 0.0,
        "explanation": state.explanation[:200] if state.explanation else "",
        "citations": state.citations if state.citations else [],
        "trace": state.trace if state.trace else []
    }


@click.command()
@click.option(
    '--batch',
    type=click.Path(exists=True),
    required=True,
    help='Input JSONL file with evaluation questions'
)
@click.option(
    '--out',
    type=click.Path(),
    required=True,
    help='Output JSONL file for results'
)
@click.option(
    '--db',
    type=click.Path(exists=True),
    default='data/northwind.sqlite',
    help='Path to SQLite database'
)
@click.option(
    '--docs',
    type=click.Path(exists=True),
    default='docs',
    help='Path to documents directory'
)
@click.option(
    '--model',
    default='phi3.5',
    help='Ollama model name'
)
def run_agent(batch: str, out: str, db: str, docs: str, model: str):
    """
    Run Retail Analytics Copilot on batch of questions.

    Example:
        python main.py --batch sample_questions_hybrid_eval.jsonl --out outputs/hybrid.jsonl
    """
    # Setup logging 
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Batch evaluation started. Log: {log_file}")

    console.print("[bold cyan]Retail Analytics Copilot - Batch Evaluation[/bold cyan]")
    console.print(f"Input:  {batch}")
    console.print(f"Output: {out}")
    console.print(f"DB:     {db}")
    console.print(f"Docs:   {docs}\n")


    # Load questions
    questions = load_evaluation_file(batch)
    if not questions:
        console.print("[red]No questions loaded[/red]")
        return

    # Initialize agent
    console.print("[yellow]Initializing agent...[/yellow]")
    try:
        agent = RetailAnalyticsCopilot(
            db_path=db,
            doc_dir=docs,
            model=model
        )
    except Exception as e:
        console.print(f"[red]Failed to initialize agent: {e}[/red]")
        return

    # Process questions
    results = []
    console.print(f"[green]Processing {len(questions)} questions...[/green]\n")

    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(questions))

        for question_obj in questions:
            try:
                question_id = question_obj.get("id", "unknown")
                question_text = question_obj.get("question", "")
                format_hint = question_obj.get("formathint", "str")

                # Run agent
                state = agent.run(question_text, format_hint)

                # Generate output
                output_row = generate_output_row(question_id, question_obj, state)
                results.append(output_row)

                # Log result
                final_answer = getattr(state, 'final_answer', None)
                logger.info(f"✓ {question_id}: {final_answer}")

            except Exception as e:
                logger.error(f"✗ {question_obj.get('id', 'unknown')}: {e}")
                results.append({
                    "id": question_obj.get("id", "unknown"),
                    "finalanswer": None,
                    "sql": "",
                    "confidence": 0.0,
                    "explanation": f"Error: {str(e)}",
                    "citations": [],
                    "trace": []
                })

            progress.update(task, advance=1)

    # Write output
    console.print("\n[yellow]Writing results...[/yellow]")
    try:
        with open(out, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        console.print(f"[green]✓ Results written to {out}[/green]\n")
    except Exception as e:
        console.print(f"[red]Error writing output: {e}[/red]")
        return

    # Summary
    passed = sum(1 for r in results if r.get("finalanswer") is not None)
    console.print("[bold cyan]Summary[/bold cyan]")
    console.print(f"Total: {len(results)}")
    console.print(f"Passed: {passed}/{len(results)} ({100*passed//len(results) if results else 0}%)")
    console.print(f"Output: {out}\n")

    # Close agent
    agent.close()


if __name__ == "__main__":
    run_agent()