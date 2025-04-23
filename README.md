Dynamic Website RAG System
A comprehensive RAG (Retrieval-Augmented Generation) system designed specifically for dynamic websites and Single Page Applications (SPAs). This system leverages advanced web crawling techniques, BM25 ranking algorithms, and LangGraph for orchestration to provide accurate, up-to-date information retrieval.
Features

Dynamic Content Handling: Built specifically to handle JavaScript-rendered content in modern web applications
Interactive Element Exploration: Automatically discovers and interacts with buttons, tabs, and expandable sections to reveal hidden content
BM25 Content Filtering: Uses the BM25 algorithm for more accurate document ranking and retrieval
Real-time Web Data: Integrates with web scraping tools to bring in live data from the internet
LangGraph Orchestration: Employs a graph-based workflow approach for modular, reactive operations
Multiple Crawling Strategies:

Dynamic Website Crawler: For SPAs and JavaScript-heavy sites
Crawl4AI Integration: Efficient crawling of standard websites with markdown conversion


Hybrid Search: Combines semantic vector search with BM25 lexical search for better results
Knowledge Base Management: Save and load knowledge bases for persistent usage

Architecture
The system consists of several interconnected components:

DynamicWebsiteRAG: Core component for handling JavaScript-rendered content using Playwright
Crawl4AIRagAdapter: Integration with Crawl4AI for efficient web scraping
LangGraphRAG: Orchestration layer for complex workflows using the LangGraph framework
RAG2System: High-level interface implementing RAG 2.0 concepts

Installation
bash# Clone the repository
git clone https://github.com/your-username/dynamic-website-rag.git
cd dynamic-website-rag

# Install Node.js dependencies
npm install

# Install Python dependencies for Crawl4AI
pip install crawl4ai
crawl4ai setup
Configuration
Create a .env file with your API keys:
OPENAI_API_KEY=your_openai_key_here
Usage
Basic Usage
javascriptconst { RAG2System } = require('./rag2-system');

async function main() {
  const rag = new RAG2System({
    llmModel: 'gpt-4o-mini',
    useRealTimeData: true,
    useBM25: true
  });
  
  try {
    // Process a query with automatic website selection
    const result = await rag.processQuery(
      What are the latest features in RAG 2.0 systems?
    );
    
    console.log(Answer:, result.answer);
    console.log(Sources:, result.sources);
    
    // Save knowledge base for future use
    rag.saveKnowledgeBase();
  } finally {
    await rag.close();
  }
}

main().catch(console.error);
Custom Website Crawling
javascriptconst { RAG2System } = require('./rag2-system');

async function main() {
  const rag = new RAG2System();
  
  try {
    // Process a query with specific websites
    const result = await rag.processQuery(
      What are the benefits of using LangGraph for RAG workflows?,
      [https://python.langchain.com/docs/langgraph]
    );
    
    console.log(Answer:, result.answer);
    console.log(Sources:, result.sources);
  } finally {
    await rag.close();
  }
}

main().catch(console.error);
Using LangGraph Orchestration
javascriptconst { RAG2System } = require('./rag2-system');

async function main() {
  const rag = new RAG2System();
  
  try {
    // Process with LangGraph orchestration
    const result = await rag.processWithLangGraph(
      How can I implement real-time web scraping for my RAG application?
    );
    
    console.log(Answer:, result.answer);
    console.log(Sources:, result.sources);
    console.log(Logs:, result.logs);
  } finally {
    await rag.close();
  }
}

main().catch(console.error);
Components
DynamicWebsiteRAG
This component is responsible for handling dynamic websites and SPAs:

Browser Automation: Uses Playwright to render JavaScript and interact with the page
Element Interaction: Clicks on buttons, expands sections, and navigates tabs
Content Extraction: Converts HTML to a markdown-like format for better LLM processing
Chunking: Breaks content into manageable pieces that fit within token limits
Vector Search: Creates embeddings and performs similarity search

Crawl4AIRagAdapter
This adapter integrates with the Crawl4AI library:

Sitemap Extraction: Automatically finds and parses XML sitemaps
Parallel Crawling: Processes multiple URLs in batches for efficiency
Markdown Conversion: Converts HTML to clean markdown format
Memory Efficiency: Optimized for handling large websites

LangGraphRAG
This component provides orchestration using the LangGraph framework:

Graph-based Workflow: Represents the RAG pipeline as a series of connected nodes
State Management: Tracks the state of the processing pipeline
Error Handling: Gracefully handles failures at each step
Parallel Processing: Executes independent tasks concurrently

RAG2System
High-level interface implementing RAG 2.0 concepts:

Smart Crawling: Automatically selects the appropriate crawler for each website
Caching: Caches web data to reduce unnecessary recrawling
Hybrid Search: Combines semantic and lexical search methods
Website Suggestion: Uses LLMs to identify relevant websites for queries

BM25 Implementation
The BM25 implementation provides lexical search capabilities:
javascript// Example of BM25 usage
const { BM25Ranker } = require('./dynamic-rag-system');

// Create a ranker with documents and field weights
const ranker = new BM25Ranker(documents, { title: 2, content: 1 });

// Search for relevant documents
const results = ranker.search(RAG systems for dynamic websites, 5);
Advanced Configuration
The system supports various configuration options:
javascriptconst rag = new RAG2System({
  // LLM settings
  embeddingModel: 'text-embedding-3-small',
  llmModel: 'gpt-4o-mini',
  temperature: 0.2,
  
  // Crawling settings
  maxInteractions: 8,
  interactionDelay: 1000,
  scrollDepth: 3,
  
  // Retrieval settings
  useRealTimeData: true,
  useBM25: true,
  
  // Cache settings
  cacheExpiration: 3600000, // 1 hour in milliseconds
  
  // Storage settings
  dataDir: './rag2_data'
});
Handling Interactive Elements
The system can interact with various UI elements on websites:

Buttons and clickable controls
Expandable content (accordions, dropdowns)
Tabs and navigation elements
Form fields and inputs
Modal and dialog triggers
Tree views and hierarchical navigation

Error Handling
The system includes robust error handling:

Connection failures
Timeout management
Invalid HTML handling
JavaScript execution errors
Rate limiting detection

Performance Optimization
Several techniques are employed to optimize performance:

Batch Processing: Process URLs in batches
Caching: Cache crawled content to avoid redundant requests
Selective Crawling: Focus on content-rich elements
Content Filtering: Skip irrelevant page sections
Efficient Vector Storage: Optimize vector search for large datasets

References
This system is based on research and technologies discussed in:

RAG 2.0: Supercharging LLMs with Real-Time Web Data and LangGraph
LangGraph: The Orchestration Engine
Web Scraping Integration
Dynamic Retrieval: Turning Scraped Data into Graph-Ready Nodes
Latency Optimization: Speeding Up Real-Time RAG in Production

License
MIT License
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
