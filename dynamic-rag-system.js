// Dynamic Website and SPA RAG System
// This system combines web scraping for dynamic content with RAG capabilities
// and incorporates BM25 for content filtering

const { chromium } = require('playwright');
const { encode } = require('gpt-3-encoder');
const { OpenAI } = require('openai');
const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs');
const path = require('path');
const { BM25 } = require('search-query-parser');

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Interactive element selectors
const interactiveSelectors = {
  // Buttons and clickable controls
  buttons: [
    'button:visible',
    'input[type="button"]:visible', 
    'input[type="submit"]:visible',
    'a[role="button"]:visible',
    '.btn:visible',
    '.button:visible',
    '[class*="btn-"]:visible',
    // ...and more from the provided list
  ],
  // Other selector categories from the provided list
  expandables: [
    '[aria-expanded="false"]:visible',
    '.expand:visible',
    '.collapse:visible',
    // ...and more
  ],
  // Additional selectors can be included as needed
};

// Class for BM25-based content filtering
class BM25Ranker {
  constructor(documents, fieldWeights = { title: 2, content: 1 }) {
    this.fieldWeights = fieldWeights;
    this.documents = documents;
    this.bm25Models = {};
    
    // Create a BM25 model for each field
    Object.keys(this.fieldWeights).forEach(field => {
      const fieldCorpus = documents.map(doc => this.tokenize(doc[field] || ''));
      this.bm25Models[field] = new BM25(fieldCorpus);
    });
  }

  tokenize(text) {
    return text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(token => token.length > 2);
  }

  search(query, topK = 5) {
    const queryTokens = this.tokenize(query);
    
    // Calculate scores for each document across all fields
    const scores = this.documents.map((doc, idx) => {
      let score = 0;
      
      // Sum weighted scores across all fields
      Object.keys(this.fieldWeights).forEach(field => {
        const fieldScore = this.bm25Models[field].score(queryTokens, idx);
        score += fieldScore * this.fieldWeights[field];
      });
      
      return { doc, score };
    });
    
    // Sort by score and return top K
    return scores
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .filter(item => item.score > 0)
      .map(item => item.doc);
  }
}

// Main RAG system for dynamic websites and SPAs
class DynamicWebsiteRAG {
  constructor(options = {}) {
    this.options = {
      maxInteractions: 5,
      interactionDelay: 1000,
      scrollDepth: 3,
      maxTokens: 8000,
      embeddingModel: 'text-embedding-3-small',
      llmModel: 'gpt-3.5-turbo',
      temperature: 0.0,
      ...options
    };
    
    this.pageBrowser = null;
    this.vectorStore = [];
    this.documents = [];
    this.interactiveElementsMap = new Map();
  }

  // Initialize browser
  async initialize() {
    this.pageBrowser = await chromium.launch({
      headless: false // Use headful mode for better dynamic content detection
    });
    console.log('Browser initialized');
  }

  // Clean up resources
  async close() {
    if (this.pageBrowser) {
      await this.pageBrowser.close();
      this.pageBrowser = null;
    }
  }

  // Convert HTML to markdown-like format for better LLM processing
  htmlToMarkdown(html) {
    const $ = cheerio.load(html);
    
    // Remove script and style tags
    $('script, style, iframe, noscript').remove();
    
    // Extract meaningful content
    let markdown = '';
    
    // Extract title
    const title = $('title').text().trim();
    if (title) markdown += `# ${title}\n\n`;
    
    // Extract headings and paragraphs
    $('h1, h2, h3, h4, h5, h6, p, li, .content, article, [role="main"]').each((_, el) => {
      const $el = $(el);
      const tagName = el.tagName.toLowerCase();
      const text = $el.text().trim();
      
      if (!text) return;
      
      if (tagName.startsWith('h')) {
        const level = parseInt(tagName.charAt(1));
        markdown += `${'#'.repeat(level)} ${text}\n\n`;
      } else if (tagName === 'li') {
        markdown += `* ${text}\n`;
      } else {
        markdown += `${text}\n\n`;
      }
    });
    
    // Clean up the markdown (remove extra spaces, newlines, etc.)
    return markdown
      .replace(/\n{3,}/g, '\n\n')
      .trim();
  }

  // Create embeddings for text using OpenAI
  async createEmbedding(text) {
    try {
      const response = await openai.embeddings.create({
        model: this.options.embeddingModel,
        input: text
      });
      return response.data[0].embedding;
    } catch (error) {
      console.error('Error creating embedding:', error);
      return null;
    }
  }

  // Calculate cosine similarity between two vectors
  cosineSimilarity(vecA, vecB) {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    return dotProduct / (magA * magB);
  }

  // Search the vector store for similar documents
  async semanticSearch(query, topK = 5) {
    try {
      const queryEmbedding = await this.createEmbedding(query);
      if (!queryEmbedding) return [];
      
      // Calculate similarity scores
      const similarities = this.vectorStore.map((item, idx) => ({
        index: idx,
        score: this.cosineSimilarity(queryEmbedding, item.embedding)
      }));
      
      // Sort by similarity and return top results
      return similarities
        .sort((a, b) => b.score - a.score)
        .slice(0, topK)
        .map(item => this.documents[item.index]);
    } catch (error) {
      console.error('Error in semantic search:', error);
      return [];
    }
  }

  // Find and interact with dynamic elements on a page
  async exploreInteractiveElements(page) {
    const interactiveElements = [];
    
    // Find all interactive elements using the selectors
    for (const [category, selectors] of Object.entries(interactiveSelectors)) {
      for (const selector of selectors) {
        try {
          const elements = await page.$$(selector);
          for (const element of elements) {
            const text = await element.textContent();
            const tagName = await element.evaluate(el => el.tagName);
            const rect = await element.boundingBox();
            
            if (rect && rect.width > 0 && rect.height > 0) {
              interactiveElements.push({
                element,
                text: text.trim(),
                category,
                tagName,
                selector
              });
            }
          }
        } catch (error) {
          console.warn(`Error finding elements with selector "${selector}":`, error.message);
        }
      }
    }
    
    return interactiveElements;
  }

  // Handle JavaScript-rendered content and SPA navigation
  async captureAfterJSExecution(page, url, maxWaitTime = 5000) {
    try {
      // Wait for network to be idle (no requests for 500ms)
      await page.waitForLoadState('networkidle', { timeout: maxWaitTime });
      
      // Execute any additional wait logic if needed for specific sites
      await page.waitForTimeout(1000); // Additional time for JS execution
      
      return await page.content();
    } catch (error) {
      console.warn(`Warning: ${error.message} when waiting for page load. Proceeding with current content.`);
      return await page.content();
    }
  }

  // Chunking text into manageable pieces
  chunkText(text, maxTokens = 1000) {
    const tokens = encode(text);
    
    if (tokens.length <= maxTokens) {
      return [text];
    }
    
    const chunks = [];
    let currentChunk = [];
    let currentLength = 0;
    
    // Split by paragraphs first
    const paragraphs = text.split('\n\n');
    
    for (const paragraph of paragraphs) {
      const paragraphTokens = encode(paragraph);
      
      if (currentLength + paragraphTokens.length <= maxTokens) {
        currentChunk.push(paragraph);
        currentLength += paragraphTokens.length;
      } else {
        if (currentChunk.length > 0) {
          chunks.push(currentChunk.join('\n\n'));
          currentChunk = [paragraph];
          currentLength = paragraphTokens.length;
        } else {
          // Handle case where a single paragraph exceeds max tokens
          const words = paragraph.split(' ');
          let tempChunk = [];
          let tempLength = 0;
          
          for (const word of words) {
            const wordTokens = encode(word + ' ').length;
            
            if (tempLength + wordTokens <= maxTokens) {
              tempChunk.push(word);
              tempLength += wordTokens;
            } else {
              chunks.push(tempChunk.join(' '));
              tempChunk = [word];
              tempLength = wordTokens;
            }
          }
          
          if (tempChunk.length > 0) {
            chunks.push(tempChunk.join(' '));
          }
        }
      }
    }
    
    if (currentChunk.length > 0) {
      chunks.push(currentChunk.join('\n\n'));
    }
    
    return chunks;
  }

  // Crawl a dynamic website or SPA
  async crawlDynamicSite(url, depth = 0, visitedUrls = new Set()) {
    if (visitedUrls.has(url) || depth < 0) {
      return;
    }
    
    console.log(`Crawling: ${url} (depth: ${depth})`);
    visitedUrls.add(url);
    
    if (!this.pageBrowser) {
      await this.initialize();
    }
    
    const context = await this.pageBrowser.newContext({
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
      viewport: { width: 1280, height: 800 }
    });
    
    const page = await context.newPage();
    await page.setDefaultTimeout(30000);
    
    try {
      await page.goto(url, { waitUntil: 'domcontentloaded' });
      let html = await this.captureAfterJSExecution(page, url);
      
      // Extract initial content
      let markdown = this.htmlToMarkdown(html);
      let documentChunks = this.chunkText(markdown, this.options.maxTokens);
      
      for (const chunk of documentChunks) {
        const embedding = await this.createEmbedding(chunk);
        if (embedding) {
          this.vectorStore.push({
            url,
            embedding
          });
          
          this.documents.push({
            url,
            title: url,
            content: chunk,
            source: 'initial'
          });
        }
      }
      
      // Explore interactive elements
      const interactiveElements = await this.exploreInteractiveElements(page);
      
      // Store the mapping of interactive elements for this page
      this.interactiveElementsMap.set(url, interactiveElements);
      
      // Interact with elements to reveal more content (up to maxInteractions)
      let interactionsCount = 0;
      for (const { element, text, category } of interactiveElements) {
        if (interactionsCount >= this.options.maxInteractions) break;
        
        try {
          // Scroll to the element
          await element.scrollIntoViewIfNeeded();
          await page.waitForTimeout(300);
          
          // Click and wait for possible content changes
          await element.click();
          await page.waitForTimeout(this.options.interactionDelay);
          
          // Check if URL changed (SPA navigation)
          const newUrl = page.url();
          const contentChanged = newUrl !== url;
          
          // Capture new content
          const newHtml = await this.captureAfterJSExecution(page, newUrl);
          const newMarkdown = this.htmlToMarkdown(newHtml);
          
          // Skip if markdown is very similar to previous
          if (newMarkdown === markdown) continue;
          
          markdown = newMarkdown;
          documentChunks = this.chunkText(markdown, this.options.maxTokens);
          
          for (const chunk of documentChunks) {
            const embedding = await this.createEmbedding(chunk);
            if (embedding) {
              this.vectorStore.push({
                url: newUrl,
                embedding
              });
              
              this.documents.push({
                url: newUrl,
                title: newUrl,
                content: chunk,
                source: `interaction:${category}:${text}`
              });
            }
          }
          
          // If navigated to a new URL in SPA, add to visited and optionally recurse
          if (contentChanged && depth > 0) {
            visitedUrls.add(newUrl);
            // Navigate back to continue exploring the original page
            await page.goBack();
            await this.captureAfterJSExecution(page, url);
          }
          
          interactionsCount++;
        } catch (error) {
          console.warn(`Error interacting with element (${text}):`, error.message);
        }
      }
      
      // Check for and follow links at current depth
      if (depth > 0) {
        const links = await page.$$eval('a[href]', links => 
          links.map(link => {
            const href = link.getAttribute('href');
            if (href && !href.startsWith('#') && !href.startsWith('javascript:')) {
              return new URL(href, link.baseURI).href;
            }
            return null;
          }).filter(Boolean)
        );
        
        // Follow unique links on the same domain
        const currentDomain = new URL(url).hostname;
        const sameDomainLinks = links.filter(link => {
          try {
            return new URL(link).hostname === currentDomain && !visitedUrls.has(link);
          } catch {
            return false;
          }
        }).slice(0, 5); // Limit number of links to follow
        
        for (const link of sameDomainLinks) {
          await this.crawlDynamicSite(link, depth - 1, visitedUrls);
        }
      }
    } catch (error) {
      console.error(`Error crawling ${url}:`, error);
    } finally {
      await context.close();
    }
  }

  // Generate hybrid search (semantic + BM25)
  async hybridSearch(query, topK = 5) {
    // Get semantic search results
    const semanticResults = await this.semanticSearch(query, topK * 2);
    
    // Apply BM25 filtering
    const bm25Ranker = new BM25Ranker(this.documents, { title: 2, content: 1 });
    const bm25Results = bm25Ranker.search(query, topK * 2);
    
    // Combine results (giving priority to documents that appear in both)
    const combinedResults = new Map();
    
    // Add all semantic results with their scores
    semanticResults.forEach(doc => {
      combinedResults.set(doc.url + doc.content.substring(0, 50), { 
        doc, 
        score: 1.0, 
        inBoth: false 
      });
    });
    
    // Add BM25 results, with higher scores for those in both sets
    bm25Results.forEach(doc => {
      const key = doc.url + doc.content.substring(0, 50);
      if (combinedResults.has(key)) {
        const existing = combinedResults.get(key);
        existing.score += 1.0;
        existing.inBoth = true;
      } else {
        combinedResults.set(key, { doc, score: 0.5, inBoth: false });
      }
    });
    
    // Sort results by score
    const sortedResults = Array.from(combinedResults.values())
      .sort((a, b) => {
        // First prioritize items in both results
        if (a.inBoth !== b.inBoth) return a.inBoth ? -1 : 1;
        // Then by score
        return b.score - a.score;
      })
      .map(item => item.doc)
      .slice(0, topK);
    
    return sortedResults;
  }

  // Generate a response using the RAG approach
  async generateResponse(query) {
    try {
      // Retrieve relevant documents using hybrid search
      const relevantDocs = await this.hybridSearch(query, 3);
      
      if (relevantDocs.length === 0) {
        return {
          answer: "I don't have enough information to answer this question accurately.",
          sources: []
        };
      }
      
      // Prepare context for the LLM
      const context = relevantDocs.map((doc, i) => 
        `[${i+1}] From ${doc.url}:\n${doc.content}`
      ).join('\n\n');
      
      // Prepare the prompt
      const prompt = `You are a helpful AI assistant with access to information about websites. Answer the following question based ONLY on the provided context. If the context doesn't contain enough information to answer fully, acknowledge that and explain what additional information would help.

Context:
${context}

Question: ${query}

Provide a concise, accurate answer. If you reference specific information, indicate which numbered source ([1], [2], etc.) it came from.`;

      // Generate answer with LLM
      const response = await openai.chat.completions.create({
        model: this.options.llmModel,
        messages: [{ role: 'user', content: prompt }],
        temperature: this.options.temperature,
        max_tokens: 1000
      });
      
      return {
        answer: response.choices[0].message.content,
        sources: relevantDocs.map(doc => ({ url: doc.url, title: doc.title }))
      };
    } catch (error) {
      console.error('Error generating response:', error);
      return {
        answer: "I encountered an error while processing your question.",
        sources: []
      };
    }
  }

  // Export the knowledge base for reuse
  exportKnowledgeBase(filePath) {
    const data = {
      documents: this.documents,
      vectorStore: this.vectorStore,
      timestamp: new Date().toISOString()
    };
    
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
    console.log(`Knowledge base exported to ${filePath}`);
  }

  // Import a previously saved knowledge base
  importKnowledgeBase(filePath) {
    try {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      this.documents = data.documents || [];
      this.vectorStore = data.vectorStore || [];
      console.log(`Knowledge base imported from ${filePath}`);
      return true;
    } catch (error) {
      console.error('Error importing knowledge base:', error);
      return false;
    }
  }
}

// Example usage
async function main() {
  const rag = new DynamicWebsiteRAG({
    maxInteractions: 10,
    interactionDelay: 1500,
    scrollDepth: 2,
    maxTokens: 4000
  });
  
  try {
    // Either crawl a new site
    await rag.crawlDynamicSite('https://example.com', 2);
    rag.exportKnowledgeBase('./knowledge_base.json');
    
    // Or import existing knowledge base
    // rag.importKnowledgeBase('./knowledge_base.json');
    
    // Generate a response
    const response = await rag.generateResponse('What features does this site offer?');
    console.log('Response:', response.answer);
    console.log('Sources:', response.sources);
  } finally {
    await rag.close();
  }
}

// Uncomment to run the example
// main().catch(console.error);

module.exports = { DynamicWebsiteRAG, BM25Ranker };