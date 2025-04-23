// Implementation of RAG 2.0 with LangGraph
// Based on RAG 2.0 concepts from the documentation in paste.txt

const { OpenAI } = require('openai');
const { DynamicWebsiteRAG } = require('./dynamic-rag-system');
const { Crawl4AIRagAdapter } = require('./crawl-for-ai-integration');
const { LangGraphRAG } = require('./spa-rag-integration');
const fs = require('fs');
const path = require('path');

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * RAG 2.0 implementation using LangGraph
 * This class implements the advanced RAG concepts from the documentation:
 * 1. Real-time data from web crawling
 * 2. Dynamic retrieval with BM25
 * 3. Orchestration using LangGraph
 */
class RAG2System {
  constructor(options = {}) {
    this.options = {
      embeddingModel: 'text-embedding-3-small',
      llmModel: 'gpt-4o-mini',
      useRealTimeData: true,
      useBM25: true,
      cacheExpiration: 3600000, // 1 hour in milliseconds
      dataDir: './rag2_data',
      ...options
    };
    
    // Create data directory if it doesn't exist
    if (!fs.existsSync(this.options.dataDir)) {
      fs.mkdirSync(this.options.dataDir, { recursive: true });
    }
    
    // Initialize components
    this.dynamicRAG = new DynamicWebsiteRAG({
      ...this.options,
      maxInteractions: 8,
      interactionDelay: 1000
    });
    
    this.crawl4aiAdapter = new Crawl4AIRagAdapter({
      ...this.options,
      batchSize: 10,
      maxPages: 50,
      outputDir: path.join(this.options.dataDir, 'crawled')
    });
    
    this.langGraphRAG = new LangGraphRAG({
      ...this.options,
      maxInteractions: 5,
      interactionDelay: 1500
    });
    
    // Cache for web data
    this.dataCache = new Map();
  }
  
  /**
   * Crawl a website or load from cache if recent
   * @param {string} url URL to crawl
   * @param {boolean} forceRefresh Force refresh even if cached data exists
   * @returns {Promise<Array>} Crawled documents
   */
  async getWebsiteData(url, forceRefresh = false) {
    const cacheKey = `website_${url}`;
    const cachedData = this.dataCache.get(cacheKey);
    
    // Check if we have fresh cached data
    if (!forceRefresh && cachedData && 
        (Date.now() - cachedData.timestamp) < this.options.cacheExpiration) {
      console.log(`Using cached data for ${url} (${cachedData.documents.length} documents)`);
      return cachedData.documents;
    }
    
    console.log(`Crawling website: ${url}`);
    
    // Decide which crawler to use based on URL complexity
    let documents = [];
    try {
      // Determine if site likely needs dynamic crawling
      const dynamicSitePatterns = [
        'angular', 'react', 'vue', 'spa', 'app', 'dashboard',
        'platform', 'portal', '#', 'javascript:', '?_escaped_fragment_=',
        'client', 'single-page'
      ];
      
      const needsDynamicCrawling = dynamicSitePatterns.some(pattern => 
        url.toLowerCase().includes(pattern)
      );
      
      if (needsDynamicCrawling) {
        // Use dynamic website crawler for SPAs
        await this.dynamicRAG.crawlDynamicSite(url, 1);
        documents = this.dynamicRAG.documents;
      } else {
        // Use Crawl4AI for more standard websites
        await this.crawl4aiAdapter.crawlWebsite(url, 1);
        documents = this.crawl4aiAdapter.documents;
      }
      
      // Cache the results
      this.dataCache.set(cacheKey, {
        documents,
        timestamp: Date.now()
      });
      
      return documents;
    } catch (error) {
      console.error(`Error crawling ${url}:`, error.message);
      
      // If we have cached data, return it even if expired
      if (cachedData) {
        console.log(`Falling back to cached data for ${url}`);
        return cachedData.documents;
      }
      
      return [];
    }
  }
  
  /**
   * BM25 ranking implementation for document filtering
   * @param {Array} documents Documents to rank
   * @param {string} query User query
   * @param {number} topK Number of documents to return
   * @returns {Array} Ranked documents
   */
  rankWithBM25(documents, query, topK = 5) {
    const { BM25Ranker } = require('./dynamic-rag-system');
    const ranker = new BM25Ranker(documents, { title: 2, content: 1 });
    return ranker.search(query, topK);
  }
  
  /**
   * Process a user query using RAG 2.0 approach
   * @param {string} query User query
   * @param {Array} websiteUrls Optional specific website URLs to search
   * @returns {Promise<Object>} Response with answer and sources
   */
  async processQuery(query, websiteUrls = []) {
    try {
      console.log(`Processing query: "${query}"`);
      
      // Step 1: Analyze query to identify relevant websites
      let sitesToCrawl = [...websiteUrls];
      
      if (this.options.useRealTimeData && sitesToCrawl.length === 0) {
        // Use LLM to identify relevant websites if none provided
        const siteSuggestionResponse = await openai.chat.completions.create({
          model: this.options.llmModel,
          messages: [
            { 
              role: 'system', 
              content: 'You are an AI assistant that analyzes user queries and determines which websites would contain the most relevant information. Output ONLY a JSON array of up to 3 specific website URLs without any explanation.'
            },
            { 
              role: 'user', 
              content: `Based on this user query, provide a JSON array of up to 3 specific website URLs that would likely contain the most relevant information: "${query}"` 
            }
          ],
          response_format: { type: "json_object" }
        });
        
        try {
          const suggestedSites = JSON.parse(siteSuggestionResponse.choices[0].message.content).urls || [];
          sitesToCrawl = [...suggestedSites];
          console.log(`Identified ${sitesToCrawl.length} relevant websites:`, sitesToCrawl);
        } catch (jsonError) {
          console.warn('Error parsing suggested sites JSON:', jsonError.message);
        }
      }
      
      // Step 2: Crawl websites or load from cache
      let allDocuments = [];
      for (const url of sitesToCrawl) {
        const documents = await this.getWebsiteData(url);
        allDocuments = [...allDocuments, ...documents];
      }
      
      console.log(`Collected ${allDocuments.length} documents from ${sitesToCrawl.length} websites`);
      
      // Step 3: Filter documents using BM25 if enabled
      let relevantDocuments = allDocuments;
      if (this.options.useBM25 && allDocuments.length > 0) {
        relevantDocuments = this.rankWithBM25(allDocuments, query, 10);
        console.log(`Filtered to ${relevantDocuments.length} documents using BM25`);
      }
      
      // Step 4: Generate response with LLM
      if (relevantDocuments.length === 0) {
        return {
          answer: "I couldn't find relevant information to answer your question. Please try a different query or specify websites to search.",
          sources: []
        };
      }
      
      // Prepare context for the LLM
      const context = relevantDocuments.map((doc, i) => 
        `[${i+1}] From ${doc.url}:\n${doc.content}`
      ).join('\n\n');
      
      // Generate an answer
      const response = await openai.chat.completions.create({
        model: this.options.llmModel,
        messages: [
          { 
            role: 'system', 
            content: 'You are a helpful AI assistant that specializes in real-time information retrieval. Answer questions based ONLY on the provided context. If the context doesn\'t contain enough information, acknowledge that and explain what might help. Cite your sources with [1], [2], etc.'
          },
          { 
            role: 'user', 
            content: `Context from web pages crawled in real-time:\n\n${context}\n\nQuestion: ${query}\n\nProvide a detailed answer based only on the information in the context above. Cite your sources.` 
          }
        ],
        temperature: 0.2
      });
      
      return {
        answer: response.choices[0].message.content,
        sources: relevantDocuments.map(doc => ({
          url: doc.url,
          title: doc.title || doc.url
        }))
      };
      
    } catch (error) {
      console.error('Error processing query:', error);
      return {
        answer: "I encountered an error while processing your query. Please try again later.",
        sources: [],
        error: error.message
      };
    }
  }
  
  /**
   * Use LangGraph orchestration for more complex workflows
   * @param {string} query User query
   * @returns {Promise<Object>} Response with answer and sources
   */
  async processWithLangGraph(query) {
    return this.langGraphRAG.run(query);
  }
  
  /**
   * Save the current knowledge base
   * @param {string} filename Filename to save to
   */
  saveKnowledgeBase(filename = 'rag2_knowledge_base.json') {
    const filePath = path.join(this.options.dataDir, filename);
    this.dynamicRAG.exportKnowledgeBase(filePath);
    console.log(`Knowledge base saved to ${filePath}`);
  }
  
  /**
   * Load a previously saved knowledge base
   * @param {string} filename Filename to load from
   * @returns {boolean} Success or failure
   */
  loadKnowledgeBase(filename = 'rag2_knowledge_base.json') {
    const filePath = path.join(this.options.dataDir, filename);
    const result = this.dynamicRAG.importKnowledgeBase(filePath);
    
    if (result) {
      console.log(`Knowledge base loaded from ${filePath}`);
    }
    
    return result;
  }
  
  /**
   * Clean up resources
   */
  async close() {
    await this.dynamicRAG.close();
  }
}

// Example implementation to showcase usage
async function example() {
  const rag2 = new RAG2System({
    llmModel: 'gpt-4o-mini',  // Use appropriate model
    useRealTimeData: true,
    useBM25: true
  });
  
  try {
    // Option 1: Process query with specific websites
    const result1 = await rag2.processQuery(
      "What are the benefits of using LangGraph for orchestrating RAG workflows?",
      ["https://python.langchain.com/docs/langgraph"]
    );
    
    console.log("\nOption 1: Query with specified websites");
    console.log("Answer:", result1.answer);
    console.log("Sources:", result1.sources.map(s => s.url).join(", "));
    
    // Option 2: Process query with automatic website selection
    const result2 = await rag2.processQuery(
      "What are the latest features in RAG 2.0 systems?"
    );
    
    console.log("\nOption 2: Query with automatic website selection");
    console.log("Answer:", result2.answer);
    console.log("Sources:", result2.sources.map(s => s.url).join(", "));
    
    // Option 3: Process with LangGraph orchestration
    const result3 = await rag2.processWithLangGraph(
      "How can I implement real-time web scraping for my RAG application?"
    );
    
    console.log("\nOption 3: Query with LangGraph orchestration");
    console.log("Answer:", result3.answer);
    console.log("Sources:", result3.sources.map(s => s.url).join(", "));
    
    // Save knowledge base for future use
    rag2.saveKnowledgeBase();
  } finally {
    // Clean up
    await rag2.close();
  }
}

// Uncomment to run the example
// example().catch(console.error);

module.exports = { RAG2System };