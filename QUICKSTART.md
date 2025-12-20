# Quick Start Guide

Get your RAG MCP server running in 5 minutes!

## Option 1: Use the Example Implementation (Fastest)

### 1. Install Dependencies

```bash
uv sync
```

This will create a `.venv` and install all dependencies from `pyproject.toml`.

### 2. Update the Server File

Replace the `RAGBackend` class in `rag_knowledge_mcp.py` with the one from `example_chroma_backend.py`:

```bash
# Open rag_knowledge_mcp.py
# Delete the placeholder RAGBackend class (lines ~70-180)
# Copy the RAGBackend class from example_chroma_backend.py
# Paste it in place of the old one
```

Or use this one-liner (if you're comfortable):

```bash
python3 << 'EOF'
# Read the example backend
with open('example_chroma_backend.py', 'r') as f:
    example_content = f.read()

# Extract just the RAGBackend class
backend_start = example_content.find('class RAGBackend:')
backend_code = example_content[backend_start:]

# Read the main server
with open('rag_knowledge_mcp.py', 'r') as f:
    server_content = f.read()

# Replace the placeholder backend
import_end = server_content.find('# ============================================================================\n# RAG Backend Interface')
backend_end = server_content.find('# ============================================================================\n# Lifespan Management')

new_content = (
    server_content[:import_end] +
    '# ============================================================================\n# RAG Backend Implementation\n# ============================================================================\n\n' +
    backend_code +
    '\n\n' +
    server_content[backend_end:]
)

# Write updated server
with open('rag_knowledge_mcp.py', 'w') as f:
    f.write(new_content)

print("✓ Backend updated successfully!")
EOF
```

### 3. Test It

```bash
uv run python test_rag_backend.py
```

You should see all tests passing ✓

### 4. Configure Claude Desktop

**macOS**: Edit `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows**: Edit `%APPDATA%\Claude\claude_desktop_config.json`

Add this configuration (update the path!):

```json
{
  "mcpServers": {
    "rag-knowledge": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/FULL/PATH/TO/rag-mcp-server",
        "python",
        "rag_knowledge_mcp.py"
      ]
    }
  }
}
```

### 5. Restart Claude Desktop

Quit and reopen Claude Desktop. Look for the 🔌 icon indicating MCP servers are connected.

### 6. Test in Claude

Try these queries:

```
Add this to my knowledge base:
"Retrieval-Augmented Generation (RAG) combines the power of large language 
models with the ability to retrieve relevant information from external knowledge 
sources. This approach significantly improves accuracy and reduces hallucinations."

Source: RAG Overview
Tags: ai, rag, machine-learning
```

Then search:

```
Search my knowledge base for information about RAG
```

## Option 2: Build Your Own Implementation

### 1. Choose Your Stack

Pick your vector database and embedding model:

**Vector Databases:**

- Chroma (easiest, local-first)
- Pinecone (managed, scalable)
- Qdrant (fast, feature-rich)
- FAISS (Facebook's library)

**Embedding Models:**

- Sentence Transformers (free, local)
- OpenAI (paid, high quality)
- Cohere (paid, specialized)

### 2. Install Dependencies

Edit `pyproject.toml` to add your chosen dependencies, then:

```bash
uv sync
```

### 3. Implement RAGBackend Methods

Edit `rag_knowledge_mcp.py` and fill in the TODO sections in the `RAGBackend` class:

```python
class RAGBackend:
    async def initialize(self):
        # TODO: Connect to your vector DB
        # TODO: Load your embedding model
        pass
    
    async def search(self, query, top_k, score_threshold, filters):
        # TODO: Encode query
        # TODO: Query vector DB
        # TODO: Return results
        pass
    
    async def add_document(self, content, metadata, chunk_size, chunk_overlap):
        # TODO: Chunk document
        # TODO: Generate embeddings
        # TODO: Store in vector DB
        pass
    
    # ... implement other methods
```

Use the example implementation as a reference!

### 4. Test Your Implementation

```bash
uv run python test_rag_backend.py
```

Fix any errors until all tests pass.

### 5. Configure and Use

Same as Option 1, steps 4-6.

## Troubleshooting

### "ModuleNotFoundError: No module named 'mcp'"

```bash
uv sync
```

### "Server won't start" or "Connection failed"

1. Test the server directly:

   ```bash
   uv run python rag_knowledge_mcp.py
   ```

2. Check logs in Claude Desktop (Help > Show Logs)

3. Verify your config file uses the full path to your project directory

### "No results found" when searching

1. Make sure you've added documents first
2. Try lowering score_threshold to 0.0
3. Check that your embedding model loaded correctly

### Import errors with Chroma

Edit `pyproject.toml` to pin a specific version:

```toml
dependencies = [
    "chromadb==0.4.22",
    # ...
]
```

Then run `uv sync`

## Next Steps

Once working:

1. **Add real documents**: Use `rag_add_document` to build your knowledge base
2. **Tune chunking**: Experiment with chunk_size and chunk_overlap
3. **Try different models**: Test various embedding models for quality
4. **Add reranking**: Improve search quality with cross-encoders
5. **Implement filtering**: Add metadata filters for targeted search

## Advanced Usage

### Custom Chunking Strategy

Replace the `_chunk_text` method with something more sophisticated:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def _chunk_text(self, text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)
```

### Hybrid Search (Vector + Keyword)

Combine semantic and keyword search for better results:

```python
async def search(self, query, top_k=5, hybrid_alpha=0.5):
    # Get vector search results
    vector_results = await self._vector_search(query, top_k * 2)
    
    # Get keyword results (implement BM25 or similar)
    keyword_results = await self._keyword_search(query, top_k * 2)
    
    # Merge with weighted scoring
    merged = self._merge_results(vector_results, keyword_results, alpha=hybrid_alpha)
    return merged[:top_k]
```

### Document Preprocessing

Add support for PDF, DOCX, HTML:

```python
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup

async def add_document_from_file(self, filepath, metadata):
    # Extract text based on file type
    if filepath.endswith('.pdf'):
        reader = PdfReader(filepath)
        content = "\n".join([page.extract_text() for page in reader.pages])
    elif filepath.endswith('.docx'):
        doc = Document(filepath)
        content = "\n".join([para.text for para in doc.paragraphs])
    elif filepath.endswith('.html'):
        with open(filepath, 'r') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            content = soup.get_text()
    else:
        with open(filepath, 'r') as f:
            content = f.read()
    
    return await self.add_document(content, metadata)
```

## Resources

- [MCP Documentation](https://modelcontextprotocol.io/)
- [Chroma Docs](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- Example implementations in `example_chroma_backend.py`

Happy building! 🚀
