# RAG Knowledge Base MCP Server

A Model Context Protocol (MCP) server that provides semantic search capabilities over a RAG (Retrieval-Augmented Generation) knowledge base. This server enables Claude and other LLMs to query, retrieve, and manage documents using vector similarity search.

## Overview

This MCP server provides a clean interface for RAG functionality while keeping the actual implementation (chunking strategies, embedding models, vector databases) completely pluggable. You can integrate any RAG stack you prefer.

### Features

- **Semantic Search**: Query your knowledge base using natural language
- **Document Management**: Add, delete, list, and retrieve documents
- **Flexible Formats**: Returns results in Markdown or JSON format
- **Pagination**: Efficient handling of large document collections
- **Progress Reporting**: Visual feedback for long-running operations
- **Pluggable Backend**: Easy integration with your existing RAG infrastructure

### Tools Provided

1. **rag_search_knowledge** - Semantic search over knowledge base
2. **rag_add_document** - Ingest new documents with chunking
3. **rag_delete_document** - Remove documents from index
4. **rag_list_documents** - Browse available documents
5. **rag_get_document** - Retrieve specific documents by ID
6. **rag_get_stats** - View knowledge base statistics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Client (Claude)                      │
│                 (Desktop App / Claude Code)                  │
└────────────────────────┬────────────────────────────────────┘
                         │ stdio transport
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastMCP Server Layer                        │
│  - Tool registration & validation (Pydantic)                 │
│  - Request/response formatting                               │
│  - Error handling & logging                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   RAGBackend Interface                       │
│              (YOUR IMPLEMENTATION GOES HERE)                 │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Chunking   │  │  Embeddings  │  │  Vector DB   │      │
│  │   Strategy   │  │    Model     │  │   (Chroma,   │      │
│  │  (Your code) │  │ (Your code)  │  │  Pinecone,   │      │
│  │              │  │              │  │   FAISS...)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### 1. Install Dependencies

```bash
uv sync
```

### 2. Add Your RAG Dependencies

Edit `pyproject.toml` and add dependencies for your chosen stack. For example:

```toml
dependencies = [
    "mcp>=1.1.0",
    "chromadb>=0.4.0",
    "sentence-transformers>=2.3.0",
    # Add your stack here
    # "pinecone-client>=3.0.0",
    # "openai>=1.0.0",
]
```

Then run `uv sync` to install.

### 3. Implement the RAG Backend

The `RAGBackend` class in `rag_knowledge_mcp.py` is your implementation layer. Replace the placeholder methods with your actual RAG logic:

```python
class RAGBackend:
    async def initialize(self):
        # Connect to your vector database
        # Load your embedding model
        
    async def search(self, query, top_k, score_threshold, filters):
        # 1. Encode query with embedding model
        # 2. Query vector database
        # 3. Return ranked results
        
    async def add_document(self, content, metadata, chunk_size, chunk_overlap):
        # 1. Chunk the document
        # 2. Generate embeddings
        # 3. Store in vector database
        
    # ... implement other methods
```

## Configuration

### Configure Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "rag-knowledge": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/your/project",
        "python",
        "rag_knowledge_mcp.py"
      ],
      "env": {
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Configure Claude Code

Create or update `~/.config/claude-code/mcp_config.json`:

```json
{
  "mcpServers": {
    "rag-knowledge": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/your/project",
        "python",
        "rag_knowledge_mcp.py"
      ],
      "env": {
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Environment Variables (Optional)

You can configure your RAG backend using environment variables:

```bash
export VECTOR_DB_URL="http://localhost:6333"  # Qdrant
export EMBEDDING_MODEL="all-MiniLM-L6-v2"     # Sentence Transformers
export OPENAI_API_KEY="sk-..."                # OpenAI embeddings
```

## Usage Examples

### Searching the Knowledge Base

```python
# Claude can now use these tools directly:
"""
Search for information about machine learning optimization techniques
"""

# Calls: rag_search_knowledge(
#   query="machine learning optimization techniques",
#   top_k=5,
#   score_threshold=0.7
# )
```

### Adding Documents

```python
# Add a document to the knowledge base
"""
Add this research paper to my knowledge base:
[paste paper content]
Source: "Neural Networks for NLP (2024)"
Tags: machine-learning, nlp, neural-networks
"""

# Calls: rag_add_document(
#   content="...",
#   source="Neural Networks for NLP (2024)",
#   tags=["machine-learning", "nlp", "neural-networks"]
# )
```

### Listing Documents

```python
# Browse available documents
"""
Show me all documents in the knowledge base
"""

# Calls: rag_list_documents(limit=20, offset=0)
```

## Implementation Guide

### Example: Chroma + Sentence Transformers

Here's a complete example implementation using Chroma and Sentence Transformers:

```python
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

class RAGBackend:
    def __init__(self):
        self.client = None
        self.collection = None
        self.model = None
    
    async def initialize(self):
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info(f"RAG backend initialized with {self.collection.count()} documents")
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # Encode query
        query_embedding = self.model.encode(query).tolist()
        
        # Query Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            score = 1 - results['distances'][0][i]  # Convert distance to similarity
            if score >= score_threshold:
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "score": score,
                    "metadata": results['metadatas'][0][i]
                })
        
        return formatted_results
    
    async def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> Dict[str, Any]:
        # Simple chunking (replace with your preferred method)
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)
        
        # Generate embeddings
        embeddings = self.model.encode(chunks).tolist()
        
        # Generate IDs
        doc_id = metadata.get('source', 'doc') + f"_{int(datetime.now().timestamp())}"
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Add chunk metadata
        chunk_metadatas = [
            {**metadata, "chunk_index": i, "parent_doc": doc_id}
            for i in range(len(chunks))
        ]
        
        # Add to Chroma
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=chunk_metadatas
        )
        
        return {
            "document_id": doc_id,
            "chunks_created": len(chunks),
            "metadata": metadata
        }
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple word-based chunking"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    # Implement other methods similarly...
```

### Example: Pinecone + OpenAI

```python
import pinecone
from openai import AsyncOpenAI

class RAGBackend:
    async def initialize(self):
        # Initialize Pinecone
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV")
        )
        
        # Connect to index
        self.index = pinecone.Index("knowledge-base")
        
        # Initialize OpenAI client
        self.openai = AsyncOpenAI()
    
    async def search(self, query: str, top_k: int = 5, **kwargs):
        # Generate embedding with OpenAI
        response = await self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format and return results...
```

## Testing

### Test Server Startup

```bash
uv run python rag_knowledge_mcp.py --help
```

### Test with MCP Inspector

```bash
npx @modelcontextprotocol/inspector python rag_knowledge_mcp.py
```

This opens a web interface where you can test all tools interactively.

### Test Individual Tools

Use the inspector to call tools manually:

```json
// Test search
{
  "query": "machine learning",
  "top_k": 3,
  "score_threshold": 0.5
}

// Test add document
{
  "content": "This is a test document about AI.",
  "source": "test.txt",
  "tags": ["test", "ai"]
}
```

## Best Practices

### Chunking Strategies

- **Fixed-size chunks**: Simple but may break context
- **Recursive chunking**: Split by paragraphs, then sentences
- **Semantic chunking**: Use embeddings to find natural breakpoints
- **Consider your use case**: Technical docs vs narrative text

### Embedding Models

- **Lightweight**: `all-MiniLM-L6-v2` (384 dim) - fast, good for general use
- **Balanced**: `all-mpnet-base-v2` (768 dim) - better quality
- **Domain-specific**: Fine-tune on your data
- **Commercial**: OpenAI `text-embedding-3-small/large`, Cohere

### Vector Databases

- **Chroma**: Easy to use, good for prototypes, local-first
- **Pinecone**: Managed, scalable, good for production
- **Qdrant**: Fast, feature-rich, self-hostable
- **FAISS**: Facebook's library, very fast, lower-level

### Performance Tips

1. **Batch operations**: Add multiple documents at once
2. **Async all the way**: Use async/await for I/O
3. **Connection pooling**: Reuse database connections (via lifespan)
4. **Caching**: Cache embeddings for common queries
5. **Index optimization**: Tune your vector DB parameters

## Troubleshooting

### Server won't start

```bash
# Check Python version (3.10+)
python --version

# Verify all dependencies installed
uv sync

# Check for import errors
uv run python -c "from mcp.server.fastmcp import FastMCP"
```

### Claude can't connect

1. Check config file path and JSON syntax
2. Verify Python path in config
3. Check Claude logs (Help > Show Logs)
4. Test server directly: `python rag_knowledge_mcp.py`

### Search returns no results

1. Verify documents are indexed: `rag_list_documents`
2. Check score threshold (try 0.0 first)
3. Test embedding model separately
4. Check vector DB connection

### Memory issues with large documents

1. Reduce `chunk_size` parameter
2. Process documents in batches
3. Use streaming for large files
4. Consider document filtering/sampling

## Advanced Features

### Adding Reranking

Improve search quality by reranking initial results:

```python
from sentence_transformers import CrossEncoder

async def search(self, query, top_k=5, ...):
    # Initial retrieval (get more candidates)
    candidates = await self._initial_search(query, top_k * 3)
    
    # Rerank with cross-encoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, doc['content']] for doc in candidates]
    scores = reranker.predict(pairs)
    
    # Sort by reranker scores and return top_k
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in reranked[:top_k]]
```

### Adding Metadata Filtering

```python
# Allow filtering by source, date, tags, etc.
await rag_backend.search(
    query="machine learning",
    filters={
        "tags": {"$contains": "neural-networks"},
        "created_at": {"$gte": "2024-01-01"}
    }
)
```

### Hybrid Search (Vector + Keyword)

Combine semantic search with traditional keyword search:

```python
async def search(self, query, top_k=5, hybrid_alpha=0.5):
    # Vector search
    vector_results = await self._vector_search(query, top_k * 2)
    
    # Keyword search (BM25)
    keyword_results = await self._keyword_search(query, top_k * 2)
    
    # Combine scores with weighted average
    combined = self._merge_results(
        vector_results,
        keyword_results,
        alpha=hybrid_alpha
    )
    
    return combined[:top_k]
```

## Contributing

This is a template server designed to be customized for your needs. Common enhancements:

- [ ] Document preprocessing (PDF, DOCX, HTML parsing)
- [ ] Metadata extraction (title, author, date)
- [ ] Multi-modal support (images, tables)
- [ ] Query expansion and refinement
- [ ] Caching layer for frequent queries
- [ ] Monitoring and analytics

## License

MIT License - feel free to use and modify for your projects.

## Resources

- [MCP Documentation](https://modelcontextprotocol.io/)
- [FastMCP GitHub](https://github.com/modelcontextprotocol/python-sdk)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Pinecone Documentation](https://docs.pinecone.io/)

## Support

For issues with:

- **MCP protocol**: See [MCP docs](https://modelcontextprotocol.io/)
- **This template**: File an issue or adapt as needed
- **Your RAG implementation**: Consult your vector DB/embedding model docs
