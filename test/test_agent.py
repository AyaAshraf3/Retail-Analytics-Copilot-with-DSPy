"""
Test Suite for Individual Agent Components
Verify retriever, database, DSPy modules, and router functionality
"""

import logging
from pathlib import Path

import dspy
from rich.console import Console
from rich.table import Table

from agent.rag.retrieval import DocumentRetriever
from agent.tools.sqlite_tool import SQLiteTool
from agent.dspy_signatures import RouterModule, NLToSQLModule

console = Console()


def setup_dspy() -> bool:
    """Configure DSPy with Ollama."""
    try:
     
        lm = dspy.LM(
            model=f"ollama/phi3.5",
            api_base='http://localhost:11434',
            api_key='',
            max_tokens=500,
            temperature=0.1
        )
        dspy.settings.configure(lm=lm)
        console.print("[green]✓ DSPy configured with Ollama[/green]")
        return True
    except Exception as e:
        console.print(f"[yellow]⚠ Warning: Could not configure Ollama: {e}[/yellow]")
        return False


def test_retriever():
    """Test BM25 document retriever."""
    console.print("\n[bold cyan]Testing Document Retriever (BM25)[/bold cyan]")
    console.print("─" * 50)
    
    try:
        retriever = DocumentRetriever("docs")
        console.print(f"[green]✓ Loaded {len(retriever.chunks)} document chunks[/green]")
        
        # Test queries
        test_queries = [
            "What is the return policy for beverages?",
            "Summer marketing campaign dates",
            "Average order value formula"
        ]
        
        table = Table(title="Retrieval Results")
        table.add_column("Query", style="cyan")
        table.add_column("Top Result", style="green")
        table.add_column("Score", style="yellow")
        
        for query in test_queries:
            results = retriever.retrieve(query, topk=1)
            if results:
                result = results[0]
                table.add_row(
                    query[:30] + "...",
                    result["id"],
                    f"{result['score']:.3f}"
                )
                console.print(f"  Result: {result['content'][:80]}...")
        
        console.print(table)
        return True
    except Exception as e:
        console.print(f"[red]✗ Retriever test failed: {e}[/red]")
        return False


def test_database():
    """Test SQLite database access."""
    console.print("\n[bold cyan]Testing Database Access[/bold cyan]")
    console.print("─" * 50)
    
    try:
        db = SQLiteTool("data/northwind.sqlite")
        console.print("[green]✓ Connected to database[/green]")
        
        # Get schema
        schema = db.get_schema()
        schema_lines = len(schema.split('\n'))
        console.print(f"[green]✓ Retrieved schema ({schema_lines} lines)[/green]")
        
        # Test queries
        test_queries = [
            ("SELECT COUNT(*) as total_orders FROM Orders", "Total orders"),
            ("SELECT COUNT(*) as total_products FROM Products", "Total products"),
            ("SELECT COUNT(DISTINCT CategoryID) as categories FROM Products", "Total categories"),
        ]
        
        table = Table(title="Query Results")
        table.add_column("Description", style="cyan")
        table.add_column("Result", style="green")
        
        for query, desc in test_queries:
            result = db.execute_query(query)
            if not result["error"]:
                value = result["rows"][0] if result["rows"] else {}
                first_val = next(iter(value.values())) if value else "N/A"
                table.add_row(desc, str(first_val))
            else:
                table.add_row(desc, f"[red]Error[/red]")
        
        console.print(table)
        db.close()
        return True
    except Exception as e:
        console.print(f"[red]✗ Database test failed: {e}[/red]")
        return False


def test_router():
    """Test DSPy Router module."""
    console.print("\n[bold cyan]Testing DSPy Router[/bold cyan]")
    console.print("─" * 50)
    
    if not setup_dspy():
        console.print("[yellow]Skipping Router test (Ollama not available)[/yellow]")
        return True
    
    try:
        router = RouterModule()
        
        test_cases = [
            ("What is the return policy for beverages?", "rag"),
            ("Top 3 products by revenue", "sql"),
            ("Revenue during Summer Beverages 1997 campaign", "hybrid"),
        ]
        
        table = Table(title="Router Classification")
        table.add_column("Question", style="cyan")
        table.add_column("Expected", style="yellow")
        table.add_column("Predicted", style="green")
        table.add_column("Match", style="bold")
        
        correct = 0
        for question, expected in test_cases:
            predicted = router(question)
            match = "✓" if predicted == expected else "✗"
            if predicted == expected:
                correct += 1
            table.add_row(
                question[:35] + "...",
                expected,
                predicted,
                match
            )
        
        console.print(table)
        console.print(f"[green]Accuracy: {correct}/{len(test_cases)} ({100*correct//len(test_cases)}%)[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Router test failed: {e}[/red]")
        return False


def test_nltosql():
    """Test DSPy NL2SQL module."""
    console.print("\n[bold cyan]Testing DSPy NL2SQL[/bold cyan]")
    console.print("─" * 50)
    
    if not setup_dspy():
        console.print("[yellow]Skipping NL2SQL test (Ollama not available)[/yellow]")
        return True
    
    try:
        db = SQLiteTool("data/northwind.sqlite")
        schema = db.get_schema()
        nltosql = NLToSQLModule()
        
        test_cases = [
            ("What is the total revenue from all orders?", {}),
            ("Top 3 products by revenue", {}),
            ("Revenue in June 1997", {"dates": ["1997-06-01", "1997-06-30"]}),
        ]
        
        table = Table(title="SQL Generation Results")
        table.add_column("Question", style="cyan")
        table.add_column("Generated SQL", style="green")
        table.add_column("Valid", style="yellow")
        
        valid_count = 0
        for question, constraints in test_cases:
            sql = nltosql(question, schema, constraints)
            is_valid, error = db.validate_query(sql)
            if is_valid:
                valid_count += 1
            
            status = "[green]✓[/green]" if is_valid else "[red]✗[/red]"
            table.add_row(
                question[:30] + "...",
                sql,
                status
            )
        
        console.print(table)
        console.print(f"[green]Valid SQL: {valid_count}/{len(test_cases)} ({100*valid_count//len(test_cases)}%)[/green]")
        db.close()
        return True
    except Exception as e:
        console.print(f"[red]✗ NL2SQL test failed: {e}[/red]")
        return False


def main():
    """Run all component tests."""
    console.print("\n[bold blue]Retail Analytics Copilot - Component Tests[/bold blue]")
    console.print("=" * 50)
    
    results = {
        "Retriever": test_retriever(),
        "Database": test_database(),
        "Router": test_router(),
        "NL2SQL": test_nltosql(),
    }
    
    # Summary
    console.print("\n[bold cyan]Test Summary[/bold cyan]")
    console.print("=" * 50)
    
    table = Table(title="Component Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    
    for component, passed in results.items():
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(component, status)
    
    console.print(table)
    
    total = len(results)
    passed = sum(results.values())
    console.print(f"\nTotal: {passed}/{total} components working")
    console.print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
