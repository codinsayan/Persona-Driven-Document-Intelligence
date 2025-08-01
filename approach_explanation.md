**Approach of 1b** 

**1\. Introduction & Problem Statement:**  
Challenge 1B, "Connecting the Dots," requires creating an intelligent offline CPU-based system (under 1GB model size, \<60 seconds processing) to extract relevant sections from PDFs based on persona and job-to-be-done. Standard search methods fail to understand complex logical constraints. Our multi-stage AI pipeline mimics expert workflow for speed and deep reasoning.

**2\. Solution Overview:**   
**A Multi-Stage** **"Filter-Retrieve-Rerank"** **Architecture**  
To meet the competing demands of speed and high-accuracy reasoning, we designed a **multi-stage "funnel"** architecture. *This approach intelligently manages computational resources by using fast, lightweight models for broad initial filtering and reserving the most powerful (and computationally expensive) Large Language Model (LLM) for the final, critical judgment.*

Our pipeline consists of four distinct stages:

**Stage 0: LLM-Powered Document Filtering:** An initial "triage" to identify the most relevant PDFs before any parsing occurs.

**Stage 1: Focused Section Parsing:** Deep parsing is performed only on the pre-filtered, relevant documents.

**Stage 2: High-Recall Candidate Retrieval:** A fast semantic search to create a shortlist of potentially relevant sections.

**Stage 3:** **LLM-as-a-Judge Re-ranking:** A final, deep-reasoning step to score and rank the shortlist for the final output.

**3\. Detailed Architectural Stages**  
**Stage 0: LLM-Powered Document Filtering**  
An initial screening uses a small, fast LLM to score document relevance based on title against the user's full query (persona \+ JTBD). This optimizes the process, preventing wasted time on irrelevant files and ensuring subsequent intensive stages focus only on promising data.

**Stage 1: Focused Section Parsing**  
Once the top 4 most relevant documents are selected, they are passed to our custom document\_parser.py script. This script uses a **pre-trained machine learning model (LightGBM)** *that analyzes a variety of linguistic and positional features (font size, position on page, text content) to classify each line of text as a specific heading level (H1, H2, etc.) or as body text*. It then intelligently groups these lines into structured sections, each with a title, content, and page number.

**Stage 2: High-Recall Candidate Retrieval (Bi-Encoder)**  
A bi-encoder model (BAAI/bge-base-en) uses semantic search to find broad answers. It converts queries and parsed sections into numerical vectors, then calculates cosine similarity to retrieve the top 15-25 candidate sections most semantically similar to the query, acting as a high-recall "funnel" for precise evaluation.

**Stage 3: LLM-as-a-Judge Re-ranking**  
This is the core of our solution's intelligence and what sets it apart from simpler search systems. The 15-25 candidates are passed to our primary reasoning engine: a small, quantized, and **instruction-tuned LLM (tinyllama-1.1b-chat-v1.0.Q4\_K\_M.gguf).**

The LLM is prompted to act as an "expert judge." For each candidate section, it is asked to provide a relevance score from 1 to 10 and a brief justification, based on the section's content and the user's original query. This "direct scoring" forces the LLM to perform deep reasoning and strictly adhere to all constraints. It can correctly identify that a recipe containing "beef" is not "vegetarian" and assign a low score, a task where the previous stages would fail. The final output is then ranked according to these intelligent scores, ensuring the highest relevance and correctness.

**4\. Model Selection & Rationale**  
The hackathon's strict constraints dictated our model choices:

**Semantic Search Model (BAAI/bge-base-en-v1.5):** Chosen for its excellent balance of speed and high performance on semantic similarity tasks. It is essential for a high-quality candidate retrieval stage.

**LLM (tinyllama-1.1b-chat-v1.0.Q4\_K\_M.gguf):** This model was specifically chosen for its small size (\~669MB) and impressive instruction-following capabilities. As a quantized GGUF model, it is highly optimized for fast inference on a CPU, making it the perfect "expert judge" that can operate within the 60-second time limit.

**5\. Conclusion**  
This multi-stage architecture provides a robust and efficient solution to the challenge. By intelligently layering fast retrieval methods with the deep reasoning of a small LLM, our system can accurately analyze diverse collections of documents and extract precisely what the user needs. It successfully filters, understands, and ranks content, moving beyond simple search to deliver true, persona-driven document intelligence.  
