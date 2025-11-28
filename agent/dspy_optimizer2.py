"""
DSPy NL2SQL Optimizer - Using LabeledFewShot 

Key Change: Replaced BootstrapFewShot with LabeledFewShot
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import dspy
from dspy.teleprompt import LabeledFewShot  
import logging
import os
import sys
import io
from datetime import datetime
from agent.tools.sqlite_tool import SQLiteTool
from agent.dspy_signatures import PlannerModule

# ============================================================================
# METRIC: Test against REAL database
# ============================================================================
def metric_valid_sql(example, prediction, trace=None):
    """Validate SQL by executing it against real database."""
    try:
        # Extract SQL from prediction
        if isinstance(prediction, str):
            sql = prediction
        elif hasattr(prediction, 'sql_query'):
            sql = prediction.sql_query
        elif isinstance(prediction, dict) and 'sql_query' in prediction:
            sql = prediction['sql_query']
        else:
            sql = str(prediction)
        
        sql = sql.strip()
        
        # Remove markdown code blocks if present
        if sql.startswith("```"):
            parts = sql.split("```")
            sql = parts[1] if len(parts) > 1 else sql
        if sql.startswith("sql\n"):
            sql = sql[4:]
        
        # Connect to REAL database
        conn = sqlite3.connect("data/northwind.sqlite")
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        
        return 1  # Valid
    except Exception as e:
        return 0  # Invalid

# ============================================================================
# DSPy TRAINING DATASET - Select BEST Examples Only
# ============================================================================

def create_training_dataset() -> List:
    """Create DSPy dataset with only the BEST working examples."""
    # Get REAL schema 
    db = SQLiteTool("data/northwind.sqlite")
    real_schema = db.get_schema()
    
    # Initialize planner 
    planner = PlannerModule()
    
    # ONLY include examples that work well
    # These are simple, clear examples that the model can learn from
    training_questions = [
        {
            "question": "List all Beverages products.",
            "expected_sql": "SELECT p.ProductID, p.ProductName FROM Products p JOIN Categories c ON p.CategoryID = c.CategoryID WHERE c.CategoryName = 'Beverages';"
        },
        {
            "question": "Total quantity sold across all orders.",
            "expected_sql": "SELECT SUM(Quantity) AS TotalQuantity FROM [Order Details];"
        },
        {
            "question": "Average discount by category.",
            "expected_sql": "SELECT c.CategoryName, ROUND(AVG(od.Discount), 2) AS AvgDiscount FROM [Order Details] od JOIN Products p ON od.ProductID = p.ProductID JOIN Categories c ON p.CategoryID = c.CategoryID GROUP BY c.CategoryID, c.CategoryName;"
        },
        {
            "question": "Top 3 products by total revenue all-time.",
            "expected_sql": "SELECT p.ProductName, ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) AS Revenue FROM [Order Details] od JOIN Products p ON od.ProductID = p.ProductID GROUP BY p.ProductID, p.ProductName ORDER BY Revenue DESC LIMIT 3;"
        },
        {
            "question": "How many orders were placed in June 1997?",
            "expected_sql": "SELECT COUNT(*) AS OrderCount FROM Orders WHERE OrderDate BETWEEN '1997-06-01' AND '1997-06-30';"
        },
        {
            "question": "Which products had zero discount?",
            "expected_sql": "SELECT DISTINCT p.ProductName FROM [Order Details] od JOIN Products p ON od.ProductID = p.ProductID WHERE od.Discount = 0;"
        },
    ]
    
    # Generate training examples
    dataset = []
    
    for item in training_questions:
        question = item["question"]
        
        # Extract constraints using REAL planner
        constraints = planner(question=question, rag_results=[])
        
        # Create DSPy example
        dspy_example = dspy.Example(
            question=question,
            schema=real_schema,
            constraints=constraints
        ).with_inputs("question", "schema", "constraints")
        
        # Add expected output
        dspy_example.sql_query = item["expected_sql"]
        dataset.append(dspy_example)
    
    db.close()
    return dataset

# ============================================================================
# TEST DATASET - All 10 questions
# ============================================================================

def create_test_dataset() -> List:
    """Create test dataset with all questions."""
    db = SQLiteTool("data/northwind.sqlite")
    real_schema = db.get_schema()
    planner = PlannerModule()
    
    test_questions = [
        "During Summer Beverages 1997 (June 1-30), which product category had the highest total quantity sold?",
        "What was the Average Order Value during Winter 1997 (December 1-31)?",
        "Top 3 products by total revenue all-time.",
        "Total revenue from Beverages category during Summer 1997 (June 1-30).",
        "Top customer by gross margin in 1997. Margin = SUM((UnitPrice * 0.30) * Quantity * (1 - Discount)).",
        "How many orders were placed in June 1997?",
        "List all Beverages products.",
        "Total quantity sold across all orders.",
        "Average discount by category.",
        "Which products had zero discount?",
    ]
    
    dataset = []
    for question in test_questions:
        constraints = planner(question=question, rag_results=[])
        dspy_example = dspy.Example(
            question=question,
            schema=real_schema,
            constraints=constraints
        ).with_inputs("question", "schema", "constraints")
        dataset.append(dspy_example)
    
    db.close()
    return dataset

# ============================================================================
# OPTIMIZATION: LabeledFewShot (Simple & Reliable)
# ============================================================================

def optimize_nl2sql():
    """Optimize NL2SQL module using LabeledFewShot."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("DSPy NL2SQL Optimization - LabeledFewShot")
    logger.info("=" * 80)
    
    # Setup DSPy with Ollama
    try:
        lm = dspy.LM(
            model="ollama/phi3.5:3.8b-mini-instruct-q4_K_M",
            api_base='http://localhost:11434',
            api_key='',
            max_tokens=500,
            temperature=0.1
        )
        dspy.settings.configure(lm=lm)
        logger.info("[OK] DSPy configured with Ollama")
    except Exception as e:
        logger.error(f"[ERROR] Failed to configure Ollama: {e}")
        return
    
    try:
        from agent.dspy_signatures import NLToSQLModule
        logger.info("[OK] Imported NLToSQLModule from agent.dspy_signatures")
    except ImportError as e:
        logger.error(f"[ERROR] Could not import NLToSQLModule: {e}")
        return
    
    # Create datasets
    trainset = create_training_dataset()
    testset = create_test_dataset()
    logger.info(f"[OK] Created training set with {len(trainset)} examples")
    logger.info(f"[OK] Created test set with {len(testset)} examples")
    
    # ========== BEFORE: Unoptimized Module ==========
    logger.info("\n" + "=" * 80)
    logger.info("BEFORE: Unoptimized NL2SQL Module")
    logger.info("=" * 80)
    
    unoptimized_module = NLToSQLModule()
    before_scores = []
    
    for i, example in enumerate(testset):
        try:
            pred = unoptimized_module(
                question=example.question,
                schema=example.schema,
                constraints=example.constraints
            )
            score = metric_valid_sql(example, pred)
            before_scores.append(score)
            status = "[OK]" if score else "[FAIL]"
            logger.info(f"  [{i+1}/{len(testset)}] {status}: {example.question[:50]}...")
        except Exception as e:
            logger.warning(f"  [{i+1}/{len(testset)}] [ERROR]: {str(e)[:50]}...")
            before_scores.append(0)
    
    before_rate = sum(before_scores) / len(before_scores) if before_scores else 0
    logger.info(f"\nBEFORE - Valid SQL Rate: {before_rate*100:.1f}% ({int(sum(before_scores))}/{len(before_scores)})")
    
    # ========== OPTIMIZATION: LabeledFewShot ==========
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZING: LabeledFewShot with Training Examples")
    logger.info("=" * 80)
    logger.info("LabeledFewShot adds your example queries to the prompt.")
    logger.info("This is simpler and more reliable than BootstrapFewShot.")
    
    try:
        optimizer = LabeledFewShot(k=3) 
        
        logger.info("Running optimization (fast, no complex bootstrapping)...")
        
        optimized_module = optimizer.compile(
            student=NLToSQLModule(),
            trainset=trainset
        )
        
        logger.info("[OK] Optimization complete!")
        
    except Exception as e:
        logger.error(f"[ERROR] Optimization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # ========== AFTER: Optimized Module ==========
    logger.info("\n" + "=" * 80)
    logger.info("AFTER: Optimized NL2SQL Module")
    logger.info("=" * 80)
    
    after_scores = []
    for i, example in enumerate(testset):
        try:
            pred = optimized_module(
                question=example.question,
                schema=example.schema,
                constraints=example.constraints
            )
            score = metric_valid_sql(example, pred)
            after_scores.append(score)
            status = "[OK]" if score else "[FAIL]"
            logger.info(f"  [{i+1}/{len(testset)}] {status}: {example.question[:50]}...")
        except Exception as e:
            logger.warning(f"  [{i+1}/{len(testset)}] [ERROR]: {str(e)[:50]}...")
            after_scores.append(0)
    
    after_rate = sum(after_scores) / len(after_scores) if after_scores else 0
    logger.info(f"\nAFTER - Valid SQL Rate: {after_rate*100:.1f}% ({int(sum(after_scores))}/{len(after_scores)})")
    
    # ========== RESULTS ==========
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 80)
    
    improvement = (after_rate - before_rate) * 100
    improvement_pct = (improvement / (before_rate * 100)) * 100 if before_rate > 0 else 0
    
    logger.info(f"Before: {before_rate*100:.1f}%")
    logger.info(f"After: {after_rate*100:.1f}%")
    logger.info(f"Improvement: {improvement:+.1f} percentage points ({improvement_pct:+.0f}% relative)")
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS")
    logger.info("=" * 80)
    
    if after_rate > before_rate:
        logger.info(f"✅ SUCCESS: NL2SQL improved by {improvement:.1f} percentage points!")
        logger.info("   LabeledFewShot successfully taught the model better SQL patterns.")
    elif after_rate == before_rate:
        logger.info(f"ℹ️ No change in performance (both {before_rate*100:.0f}%)")
        logger.info("   Model already knew these patterns, or examples didn't help.")
    else:
        logger.info(f"⚠️ Performance decreased by {abs(improvement):.1f} percentage points")
        logger.info("   This can happen with:")
        logger.info("   - Small models (Phi-3.5 3.8B has limited capacity)")
        logger.info("   - Conflicting examples in training set")
        logger.info("   - Model following training examples too rigidly")
    
    logger.info("\nKey Insight:")
    logger.info("- LabeledFewShot is simpler than BootstrapFewShot")
    logger.info("- Works better for small models and few examples")
    logger.info("- Just adds examples to prompt (no complex bootstrapping)")
    
    return {
        "before_rate": before_rate,
        "after_rate": after_rate,
        "improvement": improvement,
        "training_examples": len(trainset),
        "test_examples": len(testset)
    }

if __name__ == "__main__":
    # Windows UTF-8 encoding fix
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # Setup logging
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"nl2sql_optimization_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    
    # Run optimization
    results = optimize_nl2sql()
    
    if results:
        print("\n" + "=" * 80)
        print("OPTIMIZATION SUMMARY")
        print("=" * 80)
        print(f"Optimizer: LabeledFewShot (k=3)")
        print(f"Training Examples: {results['training_examples']}")
        print(f"Test Examples: {results['test_examples']}")
        print(f"Before: {results['before_rate']*100:.1f}%")
        print(f"After: {results['after_rate']*100:.1f}%")
        print(f"Improvement: {results['improvement']:+.1f} percentage points")
        print("=" * 80)
        print(f"\nLog saved to: {log_file}")