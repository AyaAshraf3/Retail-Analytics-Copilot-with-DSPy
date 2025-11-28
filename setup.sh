#!/bin/bash
# Setup script for Retail Analytics Copilot

set -e

echo "Retail Analytics Copilot - Setup"
echo "===================================="
echo

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"

# Create directories
echo "[2/5] Creating directories..."
mkdir -p data outputs logs
echo "✓ Created: data/, outputs/, logs/"

# Check Ollama
echo "[3/5] Checking Ollama connection..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠ Ollama not running on localhost:11434"
    echo "  Start Ollama with: ollama serve"
    echo "  Then pull model: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M"
else
    echo "✓ Ollama is running"
    ollama list | grep phi3.5 || echo "⚠ phi3.5:3.8b-mini-instruct-q4_K_M not found. Run: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M"
fi


# Download database
echo "[5/5] Checking Northwind database..."
if [ -f "data/northwind.sqlite" ]; then
    size=$(du -h data/northwind.sqlite | cut -f1)
    echo "✓ Database found ($size)"
else
    echo "Downloading Northwind database..."
    curl -L -o data/northwind.sqlite https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db 2>/dev/null
    echo "✓ Database downloaded"
fi


# Create views
echo "Creating SQLite views..."
sqlite3 data/northwind.sqlite << 'EOF' 2>/dev/null || true
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM Orders;
CREATE VIEW IF NOT EXISTS orderitems AS SELECT * FROM [Order Details];
CREATE VIEW IF NOT EXISTS products AS SELECT * FROM Products;
CREATE VIEW IF NOT EXISTS customers AS SELECT * FROM Customers;
EOF
echo "✓ Views created"

echo
echo "===================================="
echo "✓ Setup complete!"
echo
echo "Next steps:"
echo "1. Start Ollama: ollama serve"
echo "2. Run agent: python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs/hybrid.jsonl"
echo
