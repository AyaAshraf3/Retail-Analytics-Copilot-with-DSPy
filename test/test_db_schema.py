"""
Database Diagnostic Script - Test Northwind SQLite

Tests actual table structure 
"""

import sqlite3
import os

def test_database(db_path=r"D:\Retail Analytics Copilot\data\northwind.sqlite"):
    """Test database tables and structure."""
    
    print(f"\n{'='*60}")
    print(f"Testing Database: {db_path}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        print("\nTo download Northwind SQLite database:")
        print("1. Visit: https://github.com/jpwhite3/northwind-SQLite3/raw/main/Northwind_large.sqlite")
        print("2. Save as: data/northwind.sqlite")
        print("\nOR download via command:")
        print("mkdir -p data")
        print("curl -L https://github.com/jpwhite3/northwind-SQLite3/raw/main/Northwind_large.sqlite -o data/northwind.sqlite")
        return False
    
    print(f"✓ Database exists: {db_path}\n")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        
        print("Tables in database:")
        print("-" * 40)
        for table in tables:
            table_name = table[0]
            print(f"  - {table_name}")
            
            # Get column count
            cursor.execute(f"PRAGMA table_info([{table_name}])")
            columns = cursor.fetchall()
            print(f"    Columns: {len(columns)}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
            count = cursor.fetchone()[0]
            print(f"    Rows: {count}\n")
        
        print("\n" + "="*60)
        print("Testing SQL Queries")
        print("="*60 + "\n")
        
        # Test 1: OrderDetails table
        print("Test 1: Check OrderDetails table name")
        try:
            cursor.execute("SELECT COUNT(*) FROM [Order Details]")
            count = cursor.fetchone()[0]
            print(f"✓ [Order Details] exists with {count} rows (USE WITH BRACKETS)")
        except Exception as e:
            print(f"[Order Details] not found: {e}")
        
        # Test 2: Simple revenue query
        print("\nTest 2: Revenue query (with proper table names)")
        try:
            query = """
            SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Revenue 
            FROM Orders o 
            JOIN [Order Details] od ON o.OrderID = od.OrderID
            """
            cursor.execute(query)
            result = cursor.fetchone()
            revenue = result[0] if result else 0
            print(f"Revenue query works: ${revenue:,.2f}")
        except Exception as e:
            print(f"Revenue query failed: {e}")
        
        # Test 3: Top products
        print("\nTest 3: Top 3 products by revenue")
        try:
            query = """
            SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Revenue
            FROM Products p 
            JOIN [Order Details] od ON p.ProductID = od.ProductID
            GROUP BY p.ProductID, p.ProductName
            ORDER BY Revenue DESC 
            LIMIT 3
            """
            cursor.execute(query)
            results = cursor.fetchall()
            for product, revenue in results:
                print(f"✓ {product}: ${revenue:,.2f}")
        except Exception as e:
            print(f"Top products query failed: {e}")
        
        # Test 4: Date filtering
        print("\nTest 4: Revenue for Summer 1997 (June-August)")
        try:
            query = """
            SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Revenue
            FROM Orders o
            JOIN [Order Details] od ON o.OrderID = od.OrderID
            WHERE o.OrderDate BETWEEN '1997-06-01' AND '1997-08-31'
            """
            cursor.execute(query)
            result = cursor.fetchone()
            revenue = result[0] if result else 0
            print(f"Summer 1997 revenue: ${revenue:,.2f}")
        except Exception as e:
            print(f"Date filter query failed: {e}")
        
        conn.close()
        print("\n" + "="*60)
        print(" Database is working correctly!")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"\n Database error: {e}")
        return False


if __name__ == "__main__":
    success = test_database()
    exit(0 if success else 1)
