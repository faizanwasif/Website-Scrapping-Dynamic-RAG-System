// Crawl4AI Integration Module
// This module integrates Crawl4AI with our Dynamic Website RAG system for efficient site crawling

const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const util = require('util');
const execPromise = util.promisify(exec);
const { DynamicWebsiteRAG } = require('./dynamic-rag-system');
const { encode } = require('gpt-3-encoder');
const { OpenAI } = require('openai');

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Class for Crawl4AI integration
class Crawl4AIRagAdapter {
  constructor(options = {}) {
    this.options = {
      batchSize: 10,
      maxPages: 100,
      outputDir: './scraped_data',
      embeddingModel: 'text-embedding-3-small',
      maxTokens: 4000,
      ...options
    };
    
    this.documents = [];
    this.vectorStore = [];
    this.ragSystem = new DynamicWebsiteRAG(this.options);
    
    // Create output directory if it doesn't exist
    if (!fs.existsSync(this.options.outputDir)) {
      fs.mkdirSync(this.options.outputDir, { recursive: true });
    }
  }
  
  // Install Crawl4AI if not already installed
  async installCrawl4AI() {
    try {
      // Check if crawl4ai is installed
      await execPromise('pip show crawl4ai');
      console.log('Crawl4AI is already installed.');
    } catch (error) {
      console.log('Installing Crawl4AI...');
      try {
        await execPromise('pip install crawl4ai');
        await execPromise('crawl4ai setup');
        console.log('Crawl4AI installed successfully.');
      } catch (installError) {
        console.error('Error installing Crawl4AI:', installError.message);
        throw new Error('Failed to install Crawl4AI');
      }
    }
  }
  
  // Get sitemap URLs from a website
  async getSitemapUrls(url) {
    try {
      // Create a temporary Python script to extract sitemap URLs
      const scriptPath = path.join(this.options.outputDir, 'get_sitemap.py');
      
      const pythonScript = `
import requests
import xml.etree.ElementTree as ET
import json
import sys

def get_sitemap_urls(sitemap_url):
    try:
        # Ensure the URL ends with sitemap.xml if not provided
        if not sitemap_url.endswith('sitemap.xml'):
            if sitemap_url.endswith('/'):
                sitemap_url = sitemap_url + 'sitemap.xml'
            else:
                sitemap_url = sitemap_url + '/sitemap.xml'
        
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        
        # Parse the XML
        root = ET.fromstring(response.content)
        
        # Define namespace if it exists
        ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Extract all URLs
        urls = []
        for url in root.findall('.//sm:url/sm:loc', ns):
            urls.append(url.text)
        
        # If no URLs found with namespace, try without
        if not urls:
            for url in root.findall('.//loc'):
                urls.append(url.text)
        
        return urls
    except Exception as e:
        print(f"Error getting sitemap: {str(e)}", file=sys.stderr)
        return []

if __name__ == "__main__":
    sitemap_url = "${url}"
    urls = get_sitemap_urls(sitemap_url)
    print(json.dumps(urls))
      `;
      
      fs.writeFileSync(scriptPath, pythonScript);
      
      // Execute the Python script
      const { stdout, stderr } = await execPromise(`python ${scriptPath}`);
      
      if (stderr) {
        console.warn('Warnings when getting sitemap:', stderr);
      }
      
      // Parse the output
      const urls = JSON.parse(stdout.trim());
      console.log(`Found ${urls.length} URLs in sitemap for ${url}`);
      
      return urls;
    } catch (error) {
      console.error('Error getting sitemap URLs:', error.message);
      return [];
    }
  }
  
  // Crawl a website using Crawl4AI
  async crawlWebsite(url, depth = 1) {
    try {
      await this.installCrawl4AI();
      
      // Create a temporary Python script for crawling
      const scriptPath = path.join(this.options.outputDir, 'crawl_website.py');
      const outputPath = path.join(this.options.outputDir, 'crawled_data.json');
      
      let urls = [url];
      
      // If depth > 0, try to get sitemap URLs
      if (depth > 0) {
        const sitemapUrls = await this.getSitemapUrls(url);
        if (sitemapUrls.length > 0) {
          urls = [...new Set([...urls, ...sitemapUrls])];
          
          // Limit the number of URLs
          if (urls.length > this.options.maxPages) {
            console.log(`Limiting to ${this.options.maxPages} URLs from the original ${urls.length}`);
            urls = urls.slice(0, this.options.maxPages);
          }
        }
      }
      
      // Create a Python script that uses Crawl4AI to crawl the URLs
      const pythonScript = `
import asyncio
import json
from typing import List, Dict, Any
from crawl4ai import BrowserConfig, CrawlerConfig, crawl_url, crawl_urls_parallel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

async def crawl_sites_parallel(urls: List[str], batch_size: int):
    crawler_config = CrawlerConfig()
    browser_config = BrowserConfig(headless=True)
    
    results = []
    
    # Process URLs in batches
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch)} URLs")
        
        batch_results = await crawl_urls_parallel(batch, crawler_config, browser_config, batch_size)
        
        for url, result in zip(batch, batch_results):
            if result.get("success", False):
                results.append({
                    "url": url,
                    "content": result.get("markdown", ""),
                    "title": result.get("title", url),
                    "success": True
                })
            else:
                print(f"Failed to crawl: {url}")
                results.append({
                    "url": url,
                    "content": "",
                    "title": url,
                    "success": False
                })
    
    return results

async def main():
    urls = ${JSON.stringify(urls)}
    batch_size = ${this.options.batchSize}
    
    results = await crawl_sites_parallel(urls, batch_size)
    
    # Save results to file
    with open("${outputPath}", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Crawled {len(results)} URLs. Saved to ${outputPath}")

if __name__ == "__main__":
    asyncio.run(main())
      `;
      
      fs.writeFileSync(scriptPath, pythonScript);
      
      // Execute the Python script
      console.log(`Starting to crawl ${urls.length} URLs...`);
      const { stdout, stderr } = await execPromise(`python ${scriptPath}`);
      
      console.log(stdout);
      if (stderr) {
        console.warn('Warnings during crawl:', stderr);
      }
      
      // Process the crawled data
      return await this.processCrawledData(outputPath);
    } catch (error) {
      console.error('Error crawling website:', error.message);
      throw error;
    }
  }
  
  // Process the crawled data
  async processCrawledData(dataPath) {
    try {
      // Read the crawled data
      const rawData = fs.readFileSync(dataPath, 'utf8');
      const crawledData = JSON.parse(rawData);
      
      console.log(`Processing ${crawledData.length} crawled pages...`);
      
      // Process each crawled page
      for (const item of crawledData) {
        if (!item.success || !item.content) continue;
        
        // Chunk the content
        const chunks = this.chunkText(item.content, this.options.maxTokens);
        
        for (const [index, chunk] of chunks.entries()) {
          try {
            // Create embedding
            const embedding = await this.createEmbedding(chunk);
            
            if (embedding) {
              this.vectorStore.push({
                url: item.url,
                embedding
              });
              
              this.documents.push({
                url: item.url,
                title: item.title || item.url,
                content: chunk,
                chunkIndex: index,
                source: 'crawl4ai'
              });
            }
          } catch (embeddingError) {
            console.warn(`Error creating embedding for ${item.url}:`, embeddingError.message);
          }
        }
      }
      
      console.log(`Processed ${this.documents.length} document chunks with embeddings`);
      
      return {
        documents: this.documents,
        vectorStore: this.vectorStore
      };
    } catch (error) {
      console.error('Error processing crawled data:', error.message);
      throw error;
    }
  }
  
  // Create embeddings using OpenAI
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
  
  // Chunk text into manageable pieces
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
  
  // Import data into RAG system
  importToRAG() {
    this.ragSystem.documents = [...this.documents];
    this.ragSystem.vectorStore = [...this.vectorStore];
    
    return this.ragSystem;
  }
  
  // Search for relevant documents using both RAG and BM25
  async hybridSearch(query, topK = 5) {
    return this.ragSystem.hybridSearch(query, topK);
  }
  
  // Generate a response to a query
  async generateResponse(query) {
    return this.ragSystem.generateResponse(query);
  }
  
  // Save the knowledge base
  saveKnowledgeBase(filePath) {
    this.ragSystem.exportKnowledgeBase(filePath);
  }
  
  // Load a previously saved knowledge base
  loadKnowledgeBase(filePath) {
    const result = this.ragSystem.importKnowledgeBase(filePath);
    
    if (result) {
      this.documents = [...this.ragSystem.documents];
      this.vectorStore = [...this.ragSystem.vectorStore];
    }
    
    return result;
  }
}

// Example usage
async function main() {
  const crawler = new Crawl4AIRagAdapter({
    batchSize: 5,
    maxPages: 20,
    outputDir: './scraped_data'
  });
  
  try {
    // Either crawl a new site
    await crawler.crawlWebsite('https://example.com', 1);
    crawler.saveKnowledgeBase('./crawl4ai_knowledge_base.json');
    
    // Or load existing knowledge base
    // crawler.loadKnowledgeBase('./crawl4ai_knowledge_base.json');
    
    // Use the RAG system
    const response = await crawler.generateResponse('What is the main topic of this website?');
    console.log('Response:', response.answer);
    console.log('Sources:', response.sources);
  } catch (error) {
    console.error('Error in main function:', error);
  }
}

// Uncomment to run the example
// main().catch(console.error);

module.exports = { Crawl4AIRagAdapter };