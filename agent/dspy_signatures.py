"""
DSPy Signatures 

This file has:
1. Router 
2. NLToSQL with bracket notation fallbacks
3. Synthesizer and Planner
"""

import dspy
from typing import List, Dict, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


class RouterSignature(dspy.Signature):
    """Route query to RAG, SQL, or hybrid processing."""
    question: str = dspy.InputField(desc="User query")
    route: str = dspy.OutputField(
        desc="Route decision: 'rag' (document lookup), 'sql' (database query), or 'hybrid' (both). "
        "Use 'rag' for policy/definition questions like 'what is the return policy' or 'define KPI'. "
        "Use 'sql' for pure data queries like 'top 3 products', 'total revenue', without time periods. "
        "Use 'hybrid' when asking about specific time periods like 'Summer 1997' or 'June 1997' combined with calculations."
    )


class RouterModule(dspy.Module):
    """DSPy module for query routing."""
    def __init__(self):
        super().__init__()
        self.router = dspy.ChainOfThought(RouterSignature)
        self.using_fallback = False

    def forward(self, question: str) -> str:
        """Classify query into route."""
        self.using_fallback = False
        try:
            result = self.router(question=question)
            route = result.route.lower().strip()

            # Extract just the route word if there's extra text
            if 'rag' in route and 'hybrid' not in route:
                route = 'rag'
            elif 'sql' in route and 'hybrid' not in route:
                route = 'sql'
            elif 'hybrid' in route:
                route = 'hybrid'

            # Ensure valid route
            if route not in ['rag', 'sql', 'hybrid']:
                logger.warning(f"DSPy returned invalid route '{route}', using fallback")
                route = self._fallback_routing(question)
                self.using_fallback = True
            else:
                logger.info(f"✓ DSPy routing successful: {route}")
            return route

        except Exception as e:
            logger.warning(f"DSPy routing failed: {e}, using fallback")
            self.using_fallback = True
            return self._fallback_routing(question)

    def _fallback_routing(self, question: str) -> str:
        """Rule-based fallback routing when DSPy fails."""
        q_lower = question.lower()

        # RAG indicators: policy, definition, calendar
        rag_keywords = ['policy', 'return', 'definition', 'kpi', 'formula', 'what is', 'explain', 'calendar']

        # SQL indicators: numbers, aggregations, top N, revenue
        sql_keywords = ['top', 'total', 'sum', 'count', 'average', 'revenue', 'quantity', 'all-time', 'by revenue']

        # Hybrid indicators: specific time periods + calculations
        hybrid_keywords = ['during', '1997', 'summer', 'winter', 'spring', 'fall', 'june', 'july', 'august']

        # Count matches
        rag_score = sum(1 for kw in rag_keywords if kw in q_lower)
        sql_score = sum(1 for kw in sql_keywords if kw in q_lower)
        hybrid_score = sum(1 for kw in hybrid_keywords if kw in q_lower)

        # Decision logic
        if rag_score > 0 and sql_score == 0 and hybrid_score == 0:
            logger.info(f"[FALLBACK] Routing to 'rag'")
            return 'rag'
        elif sql_score > 0 and hybrid_score == 0:
            logger.info(f"[FALLBACK] Routing to 'sql'")
            return 'sql'
        elif hybrid_score > 0:
            logger.info(f"[FALLBACK] Routing to 'hybrid'")
            return 'hybrid'
        else:
            logger.info(f"[FALLBACK] Default routing to 'hybrid'")
            return 'hybrid'

    def get_routing_method(self) -> str:
        """Return whether DSPy or fallback was used."""
        return "fallback" if self.using_fallback else "dspy"



class NLToSQLSignature(dspy.Signature):
    """Generate SQL from natural language query using database schema."""
    
    question: str = dspy.InputField(
        desc="Natural language question about the database"
    )
    database_schema: str = dspy.InputField(
        desc="Complete database schema with tables, columns, and types. "
        "IMPORTANT: Table names with spaces must be enclosed in brackets, e.g., [Order Details]"
    )
    constraints: str = dspy.InputField(
        desc="Extracted constraints as JSON: dates, categories, filters"
    )
    sql_query: str = dspy.OutputField(
        desc="Valid SQLite query. "
        "CRITICAL RULES: "
        "1. Use [Order Details] with brackets (not OrderDetails) "
        "2. Use od alias for [Order Details] table: JOIN [Order Details] od "
        "3. Include discount calculation: (1 - od.Discount) "
        "4. Use BETWEEN for date ranges "
        "5. Use GROUP BY with aggregations "
        "6. Only SELECT valid columns "
        "7. Return executable SQL that works on SQLite Northwind database"
    )


class NLToSQLModule(dspy.Module):
    """DSPy module for SQL generation from natural language."""
    
    def __init__(self, max_retries: int = 3):
        """
        Initialize NL2SQL module.
        
        Args:
            max_retries: Number of retry attempts if validation fails
        """
        super().__init__()
        self.generator = dspy.ChainOfThought(NLToSQLSignature)
        self.max_retries = max_retries
        self.using_fallback = False

    def forward(
        self,
        question: str,
        schema: str,
        constraints: Dict[str, Any] = None,
        validator=None
    ) -> str:
        """
        Generate SQL query from question and schema.
        
        Args:
            question: Natural language question
            schema: Database schema from PRAGMA
            constraints: Optional constraints (dates, categories, etc.)
            validator: Optional validator function (is_valid, error_msg) = validator(sql)
        
        Returns:
            SQL query string
        """
        self.using_fallback = False
        
        if constraints is None:
            constraints = {}
        
        constraints_str = str(constraints) if constraints else "{}"
        
        # generate valid SQL with retries
        for attempt in range(self.max_retries):
            try:
                # Call DSPy generator
                result = self.generator(
                    question=question,
                    database_schema=schema,
                    constraints=constraints_str
                )
                
                sql = result.sql_query.strip() if result.sql_query else ""
                
                # Clean markdown if present
                sql = self._clean_sql(sql)
                
                logger.info(f"[Attempt {attempt+1}/{self.max_retries}] Generated SQL: {sql[:100]}")
                
                # Validate with provided validator
                if validator:
                    is_valid, error = validator(sql)
                    
                    if is_valid:
                        logger.info(f"✓ SQL validation passed: {sql[:80]}")
                        return sql
                    else:
                        logger.warning(f"✗ SQL validation failed (attempt {attempt+1}): {error}")
                        
                        # On last attempt, return anyway
                        if attempt == self.max_retries - 1:
                            logger.warning(f"Max retries reached, returning anyway")
                            return sql
                else:
                    # No validator provided - just return the SQL
                    return sql
                
            except Exception as e:
                logger.error(f"[Attempt {attempt+1}] DSPy error: {e}")
                
                if attempt == self.max_retries - 1:
                    raise
        
     
        raise RuntimeError(f"Failed to generate valid SQL after {self.max_retries} attempts")

    def _clean_sql(self, sql: str) -> str:
        """Remove markdown wrapping from generated SQL."""
        
        # Remove markdown code blocks
        if '```sql' in sql.lower():
            parts = sql.split('```')
            for part in parts:
                if 'SELECT' in part.upper():
                    sql = part.strip()
                    break
        elif '```' in sql:
            parts = sql.split('```')
            if len(parts) >= 2:
                sql = parts[1].strip()
        
        # Remove "sql:" or "SQL:" prefix
        sql = re.sub(r'^(sql|SQL)[:\s]+', '', sql, flags=re.IGNORECASE)
        
        # Single line
        sql = ' '.join(sql.split())
        
        return sql.strip()

    def get_generation_method(self) -> str:
        """Return generation method used."""
        return "fallback" if self.using_fallback else "dspy"


class SynthesizerSignature(dspy.Signature):
    """Synthesize answer from query results."""
    question: str = dspy.InputField(desc="Original user question")
    sql_results: str = dspy.InputField(desc="SQL query results as JSON")
    rag_results: str = dspy.InputField(desc="Retrieved documents as JSON")
    format_hint: str = dspy.InputField(desc="Expected output type: int, float, str, list, or dict")
    citations: str = dspy.InputField(desc="Comma-separated list of sources")
    
    final_answer: str = dspy.OutputField(desc="The answer value as a string")
    explanation: str = dspy.OutputField(desc="Brief explanation of the answer")
    confidence: float = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")


class SynthesizerModule(dspy.Module):
    """DSPy module for answer synthesis."""
    def __init__(self):
        super().__init__()
        self.synthesizer = dspy.ChainOfThought(SynthesizerSignature)

    def forward(
        self,
        question: str,
        sql_results: List[Any] = None,
        rag_results: List[Dict[str, Any]] = None,
        format_hint: str = "str",
        citations: List[str] = None
    ) -> Dict[str, Any]:
        """Synthesize final answer from all sources."""
        try:
            import json

            sql_str = json.dumps(sql_results if sql_results else [], default=str)
            rag_str = json.dumps(rag_results if rag_results else [], default=str)
            citations_str = ",".join(citations) if citations else ""

            result = self.synthesizer(
                question=question,
                sql_results=sql_str,
                rag_results=rag_str,
                format_hint=format_hint,
                citations=citations_str
            )

            answer_str = result.final_answer.strip()
            explanation = result.explanation.strip()

            try:
                confidence = float(result.confidence)
                confidence = max(0.0, min(1.0, confidence))
            except:
                confidence = 0.5

            answer = self._convert_to_type(answer_str, format_hint)

            return {
                "answer": answer,
                "explanation": explanation,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Synthesizer error: {e}")
            return {
                "answer": None,
                "explanation": f"Error synthesizing answer: {str(e)}",
                "confidence": 0.0
            }

    def _convert_to_type(self, value: str, format_hint: str) -> Any:
        """Convert string value to target type."""
        format_hint = format_hint.lower().strip()
        try:
            if "int" in format_hint:
                return int(float(value))
            elif "float" in format_hint:
                return float(value)
            elif "list" in format_hint or "[" in format_hint:
                import json
                return json.loads(value)
            elif "dict" in format_hint or "{" in format_hint:
                import json
                return json.loads(value)
            else:
                return str(value)
        except:
            return value


class PlannerModule(dspy.Module):
    """Extract constraints from question and documents."""
    def __init__(self):
        super().__init__()

    def forward(
        self,
        question: str,
        rag_results: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract dates, categories, KPIs from question and documents."""
        constraints = {
            "dates": self._extract_dates(question),
            "categories": self._extract_categories(question),
            "kpis": self._extract_kpis(question),
        }

        if rag_results:
            for result in rag_results:
                content = result.get("content", "")
                constraints["dates"].extend(self._extract_dates(content))
                constraints["categories"].extend(self._extract_categories(content))
                constraints["kpis"].extend(self._extract_kpis(content))

        # Remove duplicates
        for key in constraints:
            constraints[key] = list(set(constraints[key]))

        return constraints

    
    def _extract_dates(self, text: str) -> List[str]:
        """
        Extract date patterns - both explicit and natural language.
        
        Handles:
        - YYYY-MM-DD format: "1997-06-01"
        - Natural language: "June 1997" or "June 1-30" -> converts to date range
        - Seasonal references: "Summer 1997"
        """
        import re
        dates = []
        
        # Pattern 1: Explicit YYYY-MM-DD dates
        pattern = r'\d{4}-\d{2}-\d{2}'
        explicit_dates = re.findall(pattern, text)
        dates.extend(explicit_dates)
        
        # Pattern 2: "June 1-30" format (month with day range)
        month_day_range_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})-(\d{1,2})'
        month_day_matches = re.findall(month_day_range_pattern, text, re.IGNORECASE)
        

        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        years_in_text = re.findall(year_pattern, text)
        default_year = years_in_text[0] if years_in_text else '1997'
        
        month_to_num = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        for month_name, start_day, end_day in month_day_matches:
            month_num = month_to_num.get(month_name.lower(), '01')
            start_day = start_day.zfill(2)
            end_day = end_day.zfill(2)
            dates.append(f"{default_year}-{month_num}-{start_day}")
            dates.append(f"{default_year}-{month_num}-{end_day}")
        
        # Pattern 3: Month + Year (e.g., "June 1997", "December 1997")
        month_year_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})'
        month_year_matches = re.findall(month_year_pattern, text, re.IGNORECASE)
        
        for month_name, year in month_year_matches:
            month_num = month_to_num.get(month_name.lower(), '01')
            # Add first and last day of month
            dates.append(f"{year}-{month_num}-01")
            
            # Determine last day of month
            if month_num in ['01', '03', '05', '07', '08', '10', '12']:
                last_day = '31'
            elif month_num in ['04', '06', '09', '11']:
                last_day = '30'
            else:  # February
                # Simple leap year check
                year_int = int(year)
                if year_int % 4 == 0 and (year_int % 100 != 0 or year_int % 400 == 0):
                    last_day = '29'
                else:
                    last_day = '28'
            
            dates.append(f"{year}-{month_num}-{last_day}")
        
        # Pattern 4: Seasonal references (Summer, Winter, etc.)
        seasonal_pattern = r'(Summer|Winter|Spring|Fall|Autumn)\s+(?:.*?\s+)?(\d{4})'
        seasonal_matches = re.findall(seasonal_pattern, text, re.IGNORECASE)
        
        for season, year in seasonal_matches:
            season_lower = season.lower()
            if 'summer' in season_lower:
                # Look for specific month mentions nearby
                if 'june' in text.lower():
                    dates.extend([f"{year}-06-01", f"{year}-06-30"])
                else:
                    dates.extend([f"{year}-06-01", f"{year}-08-31"])  # Full summer
            elif 'winter' in season_lower:
                if 'december' in text.lower():
                    dates.extend([f"{year}-12-01", f"{year}-12-31"])
                else:
                    dates.extend([f"{year}-12-01", f"{year}-02-28"])  # Full winter
            elif 'spring' in season_lower:
                dates.extend([f"{year}-03-01", f"{year}-05-31"])
            elif 'fall' in season_lower or 'autumn' in season_lower:
                dates.extend([f"{year}-09-01", f"{year}-11-30"])
        
        return dates

    def _extract_categories(self, text: str) -> List[str]:
        """Extract product categories."""
        categories = [
            "Beverages", "Condiments", "Confections", "Dairy Products",
            "Grains/Cereals", "Meat/Poultry", "Produce", "Seafood"
        ]
        found = [cat for cat in categories if cat.lower() in text.lower()]
        return found

    def _extract_kpis(self, text: str) -> List[str]:
        """Extract KPI mentions."""
        kpis = ["AOV", "Gross Margin", "Revenue", "Quantity", "Discount"]
        found = [kpi for kpi in kpis if kpi.lower() in text.lower()]
        return found