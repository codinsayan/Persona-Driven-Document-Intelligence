# **Persona-Driven Document Intelligence**

This repository contains the complete solution for Challenge 1b. It uses a multi-stage AI pipeline, including a fast semantic search model and a powerful Large Language Model (LLM), to analyze a collection of PDFs. The system extracts and ranks the most relevant sections based on a specific user persona and their job-to-be-done, running entirely offline within a Docker container.

## **How to Build and Run**

The entire solution is containerized for easy and consistent execution.

### **1\. Clone Repository**
Clone this Repository, by running the following command:
```Bash  
git clone https://github.com/codinsayan/Persona-Driven-Document-Intelligence
```

Navigate to the root directory
```Bash  
cd Hybrid-Machine-Learning-Solution-for-PDF-Structure-Extraction
```

### **2\. Build the Docker Image**

In the root directory, build the docker container
```Bash  
docker build --platform linux/amd64 -t smart-searcher:latest .
```

### **3\. Run the Container**

Place your collection of PDFs and the `challenge1b_input.json` inside the `main/` directory. Then, from the root `Submission/` directory, run the command below.

```Bash  
docker run --rm -v "$(pwd)/MAIN:/app/MAIN" --network none smart-searcher:latest
```

This command will process the collection inside the `MAIN/` folder and generate the `challenge1b_output.json` file in that same directory.

## **Solution Overview**

### **Approach**

Our solution uses a sophisticated, multi-stage **"Filter-Retrieve-Rerank" architecture** to balance the competing needs for speed and high-accuracy reasoning, ensuring we meet the strict 60-second time limit.

1. **LLM Document Filtering:** To avoid wasting time parsing irrelevant files, the pipeline begins by using a small, fast LLM to score each document based solely on its title against the user's query. This immediately prunes the search space to the top 4 most relevant PDFs.  
2. **Focused Section Parsing:** Only the selected documents are processed by our custom parser, which uses a pre-trained `LightGBM` model to classify text lines and group them into structured, meaningful sections.  
3. **Candidate Retrieval:** A fast semantic search model (`BAAI/bge-base-en-v1.5`) is used to retrieve the top 15 candidate sections from the parsed content. This acts as a high-recall "funnel" for the final, most computationally expensive step.  
4. **LLM Re-ranking:** Finally, a powerful instruction-tuned LLM (`tinyllama-1.1b-chat-v1.0`) acts as an expert judge. It scores each candidate for its relevance and strict adherence to all task constraints (e.g., "vegetarian," "gluten-free"). The final output is ranked according to these intelligent scores.

### **Models and Key Libraries**

* **Models**:  
  * **LLM Judge & Filter**: **`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`**, a small, fast, and instruction-tuned LLM, quantized for optimal CPU performance. It serves as our primary reasoning engine.  
  * **Semantic Retrieval Model**: **`BAAI/bge-base-en-v1.5`**, a high-performance sentence-transformer model used for fast candidate retrieval.  
  * **Heading Classifier**: A custom-trained **`LightGBM`** model used by the document parser to identify structural elements.  
* **Libraries**:  
  * `PyMuPDF` for robust PDF text and feature extraction.  
  * `pandas`, `scikit-learn`, & `lightgbm` for the document structure classification model.  
  * `sentence-transformers` for semantic vector embeddings.  
  * `ctransformers` for efficient, CPU-based inference of the GGUF-quantized LLM.  
  * `Docker` for creating a reproducible execution environment.

### **Role of Each Script**

* **main.py**: The main entry point for the Docker container. It locates the `main` data directory and orchestrates the end-to-end pipeline.  
* **optimized\_orchestrator.py**: Contains the core multi-stage pipeline logic, including document filtering, candidate retrieval, and LLM re-ranking.  
* **document\_parser.py**: Responsible for parsing a single PDF into a structured list of sections using the trained `LightGBM` model.  
* **feature\_extractor.py**: A utility script that extracts the detailed feature vectors for each line of text from a PDF, which are then fed to the parser's classifier.  
* **semantic\_search.py**: A helper script containing the `SemanticSearch` class, used by the orchestrator for the candidate retrieval stage.





