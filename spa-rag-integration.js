// SPA Integration with LangGraph
// This module integrates our Dynamic Website RAG system with LangGraph for orchestration

const { DynamicWebsiteRAG } = require('./dynamic-rag-system');
const { OpenAI } = require('openai');
const fs = require('fs');
const path = require('path');
const { defineConfig } = require('langgraph');
const { StateGraph, step } = require('langgraph/state');

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Define the state schema for our LangGraph workflow
const stateSchema = {
  query: "string",
  crawl_urls: ["string"],
  documents: ["object"],
  filtered_documents: ["object"],
  response: "string",
  logs: ["string"],
  error: "string?"
};

// LangGraph integration class for our RAG system
class LangGraphRAG {
  constructor(options = {}) {
    this.options = {
      maxInteractions: 5,
      interactionDelay: 1000,
      maxTokens: 8000,
      embeddingModel: 'text-embedding-3-small',
      llmModel: 'gpt-4-turbo',
      ...options
    };
    
    this.ragSystem = new DynamicWebsiteRAG(this.options);
    this.graph = this.buildWorkflow();
  }
  
  // Create the LangGraph workflow
  buildWorkflow() {
    const builder = new StateGraph({ schema: stateSchema });
    
    // Step 1: Process the query and determine URLs to crawl
    builder.addNode("process_query", step(async (state) => {
      try {
        console.log(`Processing query: ${state.query}`);
        
        // Use LLM to analyze the query and determine which URLs to crawl
        const response = await openai.chat.completions.create({
          model: this.options.llmModel,
          messages: [
            { 
              role: 'system', 
              content: 'You are an AI assistant that analyzes user queries and determines which web pages need to be crawled to answer the question. Output only a JSON array of URLs relevant to the query.'
            },
            { 
              role: 'user', 
              content: `User query: "${state.query}"\n\nBased on this query, provide a JSON array of up to 3 specific URLs that would be most relevant to crawl. If the query doesn't specify websites, use default websites related to the topic.` 
            }
          ],
          response_format: { type: "json_object" }
        });
        
        const urls = JSON.parse(response.choices[0].message.content).urls || [];
        
        return {
          crawl_urls: urls,
          logs: [...state.logs, `Determined ${urls.length} URLs to crawl`]
        };
      } catch (error) {
        console.error('Error in process_query:', error);
        return { 
          error: `Error processing query: ${error.message}`,
          logs: [...state.logs, `Error: ${error.message}`]
        };
      }
    }));
    
    // Step 2: Crawl the relevant websites
    builder.addNode("crawl_websites", step(async (state) => {
      try {
        console.log(`Crawling ${state.crawl_urls.length} URLs`);
        
        // Initialize browser if needed
        if (!this.ragSystem.pageBrowser) {
          await this.ragSystem.initialize();
        }
        
        // Crawl each URL
        const visitedUrls = new Set();
        for (const url of state.crawl_urls) {
          await this.ragSystem.crawlDynamicSite(url, 1, visitedUrls);
        }
        
        return { 
          documents: this.ragSystem.documents,
          logs: [...state.logs, `Crawled ${state.crawl_urls.length} URLs, found ${this.ragSystem.documents.length} documents`]
        };
      } catch (error) {
        console.error('Error in crawl_websites:', error);
        return { 
          error: `Error crawling websites: ${error.message}`,
          logs: [...state.logs, `Error: ${error.message}`]
        };
      }
    }));
    
    // Step 3: Filter documents using BM25
    builder.addNode("filter_documents", step(async (state) => {
      try {
        console.log(`Filtering ${state.documents.length} documents`);
        
        // Use BM25 ranking to filter documents
        const { BM25Ranker } = require('./dynamic-rag-system');
        const ranker = new BM25Ranker(state.documents, { title: 2, content: 1 });
        const filteredDocs = ranker.search(state.query, 5);
        
        return { 
          filtered_documents: filteredDocs,
          logs: [...state.logs, `Filtered to ${filteredDocs.length} most relevant documents`]
        };
      } catch (error) {
        console.error('Error in filter_documents:', error);
        return { 
          error: `Error filtering documents: ${error.message}`,
          logs: [...state.logs, `Error: ${error.message}`]
        };
      }
    }));
    
    // Step 4: Generate response
    builder.addNode("generate_response", step(async (state) => {
      try {
        console.log(`Generating response from ${state.filtered_documents.length} documents`);
        
        // Prepare context for the LLM
        const context = state.filtered_documents.map((doc, i) => 
          `[${i+1}] From ${doc.url}:\n${doc.content}`
        ).join('\n\n');
        
        // Generate an answer
        const response = await openai.chat.completions.create({
          model: this.options.llmModel,
          messages: [
            { 
              role: 'system', 
              content: 'You are a helpful AI assistant that answers questions based on web content. Provide comprehensive, accurate answers based only on the provided context. Cite sources using [1], [2], etc.'
            },
            { 
              role: 'user', 
              content: `Context:\n${context}\n\nQuestion: ${state.query}\n\nProvide a detailed answer based only on the information in the context above. Cite your sources.` 
            }
          ],
          temperature: 0.2
        });
        
        const answer = response.choices[0].message.content;
        
        return { 
          response: answer,
          logs: [...state.logs, `Generated response (${answer.length} chars)`]
        };
      } catch (error) {
        console.error('Error in generate_response:', error);
        return { 
          error: `Error generating response: ${error.message}`,
          response: "I encountered an error while trying to answer your question based on the web content.",
          logs: [...state.logs, `Error: ${error.message}`]
        };
      }
    }));
    
    // Define workflow edges
    builder.addEdge("process_query", "crawl_websites");
    builder.addEdge("crawl_websites", "filter_documents");
    builder.addEdge("filter_documents", "generate_response");
    
    // Handle errors
    builder.addConditionalEdges(
      "process_query",
      (state) => state.error ? "end" : "crawl_websites"
    );
    
    builder.addConditionalEdges(
      "crawl_websites",
      (state) => state.error ? "end" : "filter_documents"
    );
    
    builder.addConditionalEdges(
      "filter_documents",
      (state) => state.error ? "end" : "generate_response"
    );
    
    // Set entry point
    builder.setEntryPoint("process_query");
    
    // Compile the graph
    return builder.compile();
  }
  
  // Run the workflow
  async run(query) {
    try {
      // Initialize the state
      const initialState = {
        query,
        crawl_urls: [],
        documents: [],
        filtered_documents: [],
        response: "",
        logs: [`Started processing query: ${query}`],
        error: null
      };
      
      // Execute the graph
      const result = await this.graph.invoke(initialState);
      
      // Clean up resources
      await this.ragSystem.close();
      
      return {
        answer: result.response,
        logs: result.logs,
        error: result.error,
        sources: result.filtered_documents.map(doc => ({
          url: doc.url,
          title: doc.title || doc.url
        }))
      };
    } catch (error) {
      console.error('Error running workflow:', error);
      await this.ragSystem.close();
      
      return {
        answer: "An error occurred while processing your query.",
        logs: [`Error: ${error.message}`],
        error: error.message,
        sources: []
      };
    }
  }
  
  // Save knowledge base for future use
  saveKnowledgeBase(filePath) {
    this.ragSystem.exportKnowledgeBase(filePath);
  }
  
  // Load a previously saved knowledge base
  loadKnowledgeBase(filePath) {
    return this.ragSystem.importKnowledgeBase(filePath);
  }
}

// Example usage
async function main() {
  const langGraphRAG = new LangGraphRAG({
    maxInteractions: 8,
    interactionDelay: 1500,
    llmModel: 'gpt-4o-mini'
  });
  
  try {
    // Either load an existing knowledge base
    // langGraphRAG.loadKnowledgeBase('./spa_knowledge_base.json');
    
    // Or run a query that will crawl websites
    const result = await langGraphRAG.run(
      "What are the key features of RAG systems for dynamic websites?"
    );
    
    console.log("ANSWER:");
    console.log(result.answer);
    
    console.log("\nSOURCES:");
    result.sources.forEach(source => console.log(`- ${source.url}`));
    
    // Save for future use
    langGraphRAG.saveKnowledgeBase('./spa_knowledge_base.json');
    
    console.log("\nLOGS:");
    result.logs.forEach(log => console.log(`- ${log}`));
  } catch (error) {
    console.error("Error in main function:", error);
  }
}

// Uncomment to run the example
// main().catch(console.error);

module.exports = { LangGraphRAG };