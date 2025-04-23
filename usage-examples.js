// Example 1: Basic RAG query with automatic website detection
const { RAG2System } = require('./langgraph-implementation');

async function basicExample() {
  console.log("Example 1: Basic RAG query with automatic website detection");
  
  const rag = new RAG2System({
    llmModel: 'gpt-4o-mini',
    useRealTimeData: true,
    useBM25: true
  });
  
  try {
    // Process a query without specifying websites
    // System will automatically detect relevant sites
    const result = await rag.processQuery(
      "What are the key features of RAG 2.0 systems?"
    );
    
    console.log("\n=== ANSWER ===");
    console.log(result.answer);
    
    console.log("\n=== SOURCES ===");
    result.sources.forEach((source, i) => {
      console.log(`${i+1}. ${source.title || 'Untitled'}: ${source.url}`);
    });
    
    // Save knowledge base for future use
    rag.saveKnowledgeBase();
  } finally {
    // Always close the browser when done
    await rag.close();
  }
}

// Example 2: Crawling a dynamic SPA
const { DynamicWebsiteRAG } = require('./dynamic-rag-system');

async function dynamicSpaExample() {
  console.log("Example 2: Crawling a dynamic SPA");
  
  const dynamicRag = new DynamicWebsiteRAG({
    maxInteractions: 10,
    interactionDelay: 1500,
    scrollDepth: 2,
    maxTokens: 4000
  });
  
  try {
    // Crawl a dynamic SPA
    console.log("Crawling a dynamic single-page application...");
    await dynamicRag.crawlDynamicSite('https://example-spa.com/dashboard', 1);
    
    console.log(`Crawled ${dynamicRag.documents.length} document chunks`);
    
    // Generate a response to a query
    const query = "What features does this dashboard offer?";
    console.log(`\nQuerying: "${query}"`);
    
    const response = await dynamicRag.generateResponse(query);
    
    console.log("\n=== ANSWER ===");
    console.log(response.answer);
    
    console.log("\n=== SOURCES ===");
    response.sources.forEach((source, i) => {
      console.log(`${i+1}. ${source.title || 'Untitled'}: ${source.url}`);
    });
    
    // Export the knowledge base
    dynamicRag.exportKnowledgeBase('./spa_knowledge_base.json');
  } finally {
    await dynamicRag.close();
  }
}

// Example 3: Using Crawl4AI for efficient website crawling
const { Crawl4AIRagAdapter } = require('./crawl-for-ai-integration');

async function crawl4aiExample() {
  console.log("Example 3: Using Crawl4AI for efficient website crawling");
  
  const crawler = new Crawl4AIRagAdapter({
    batchSize: 5,
    maxPages: 20,
    outputDir: './scraped_data'
  });
  
  try {
    // Crawl a documentation website efficiently
    console.log("Crawling website using Crawl4AI...");
    await crawler.crawlWebsite('https://docs.example.com', 1);
    
    console.log(`Processed ${crawler.documents.length} document chunks`);
    
    // Save the knowledge base
    crawler.saveKnowledgeBase('./crawl4ai_knowledge_base.json');
    
    // Generate a response
    const query = "How do I install this framework?";
    console.log(`\nQuerying: "${query}"`);
    
    const response = await crawler.generateResponse(query);
    
    console.log("\n=== ANSWER ===");
    console.log(response.answer);
    
    console.log("\n=== SOURCES ===");
    response.sources.forEach((source, i) => {
      console.log(`${i+1}. ${source.title || 'Untitled'}: ${source.url}`);
    });
  } catch (error) {
    console.error('Error in Crawl4AI example:', error);
  }
}

// Example 4: LangGraph orchestration for complex workflows
const { LangGraphRAG } = require('./spa-rag-integration');

async function langGraphExample() {
  console.log("Example 4: LangGraph orchestration for complex workflows");
  
  const langGraphRAG = new LangGraphRAG({
    maxInteractions: 8,
    interactionDelay: 1500,
    llmModel: 'gpt-4o-mini'
  });
  
  try {
    // Run a query through the LangGraph workflow
    console.log("Running query through LangGraph workflow...");
    
    const result = await langGraphRAG.run(
      "What are the advantages of using LangGraph for RAG orchestration?"
    );
    
    console.log("\n=== ANSWER ===");
    console.log(result.answer);
    
    console.log("\n=== SOURCES ===");
    result.sources.forEach((source, i) => {
      console.log(`${i+1}. ${source.title || 'Untitled'}: ${source.url}`);
    });
    
    console.log("\n=== WORKFLOW LOGS ===");
    result.logs.forEach((log, i) => {
      console.log(`${i+1}. ${log}`);
    });
    
    // Save for future use
    langGraphRAG.saveKnowledgeBase('./langgraph_knowledge_base.json');
  } catch (error) {
    console.error('Error in LangGraph example:', error);
  }
}

// Example 5: Loading from a saved knowledge base
const fs = require('fs');

async function loadKnowledgeBaseExample() {
  console.log("Example 5: Loading from a saved knowledge base");
  
  const knowledgeBasePath = './spa_knowledge_base.json';
  
  // Check if knowledge base exists
  if (!fs.existsSync(knowledgeBasePath)) {
    console.log(`Knowledge base ${knowledgeBasePath} doesn't exist yet. Run example 2 first.`);
    return;
  }
  
  const rag = new DynamicWebsiteRAG();
  
  try {
    // Load the knowledge base
    console.log(`Loading knowledge base from ${knowledgeBasePath}...`);
    const success = rag.importKnowledgeBase(knowledgeBasePath);
    
    if (!success) {
      console.log("Failed to load knowledge base.");
      return;
    }
    
    console.log(`Loaded ${rag.documents.length} documents from knowledge base`);
    
    // Query the loaded knowledge base
    const query = "What are the main sections of this website?";
    console.log(`\nQuerying: "${query}"`);
    
    const response = await rag.generateResponse(query);
    
    console.log("\n=== ANSWER ===");
    console.log(response.answer);
    
    console.log("\n=== SOURCES ===");
    response.sources.forEach((source, i) => {
      console.log(`${i+1}. ${source.title || 'Untitled'}: ${source.url}`);
    });
  } finally {
    await rag.close();
  }
}

// Example 6: Hybrid search with BM25 and vector search
async function hybridSearchExample() {
  console.log("Example 6: Hybrid search with BM25 and vector search");
  
  const rag = new DynamicWebsiteRAG();
  const knowledgeBasePath = './spa_knowledge_base.json';
  
  // Check if knowledge base exists
  if (!fs.existsSync(knowledgeBasePath)) {
    console.log(`Knowledge base ${knowledgeBasePath} doesn't exist yet. Run example 2 first.`);
    return;
  }
  
  try {
    // Load the knowledge base
    console.log(`Loading knowledge base from ${knowledgeBasePath}...`);
    rag.importKnowledgeBase(knowledgeBasePath);
    
    // Perform a hybrid search
    const query = "authentication methods";
    console.log(`\nPerforming hybrid search for: "${query}"`);
    
    const docs = await rag.hybridSearch(query, 5);
    
    console.log(`\nFound ${docs.length} relevant documents:`);
    docs.forEach((doc, i) => {
      console.log(`\n--- Document ${i+1} ---`);
      console.log(`URL: ${doc.url}`);
      console.log(`Title: ${doc.title || 'Untitled'}`);
      console.log(`Content: ${doc.content.substring(0, 150)}...`);
    });
  } finally {
    await rag.close();
  }
}

// Main function to run all examples
async function runAllExamples() {
  try {
    await basicExample();
    console.log("\n" + "=".repeat(50) + "\n");
    
    await dynamicSpaExample();
    console.log("\n" + "=".repeat(50) + "\n");
    
    await crawl4aiExample();
    console.log("\n" + "=".repeat(50) + "\n");
    
    await langGraphExample();
    console.log("\n" + "=".repeat(50) + "\n");
    
    await loadKnowledgeBaseExample();
    console.log("\n" + "=".repeat(50) + "\n");
    
    await hybridSearchExample();
  } catch (error) {
    console.error("Error running examples:", error);
  }
}

// To run a specific example, uncomment the corresponding line:
// basicExample().catch(console.error);
// dynamicSpaExample().catch(console.error);
// crawl4aiExample().catch(console.error);
// langGraphExample().catch(console.error);
// loadKnowledgeBaseExample().catch(console.error);
// hybridSearchExample().catch(console.error);

// Or run all examples:
// runAllExamples().catch(console.error);

module.exports = {
  basicExample,
  dynamicSpaExample,
  crawl4aiExample,
  langGraphExample,
  loadKnowledgeBaseExample,
  hybridSearchExample,
  runAllExamples
};