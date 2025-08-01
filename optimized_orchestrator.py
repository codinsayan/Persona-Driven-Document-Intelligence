import json
import time
import argparse
import os
import re
from datetime import datetime
from ctransformers import AutoModelForCausalLM
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util

# Import our previously built components
from document_parser import parse_document_to_sections

class OptimizedOrchestrator:
    """
    Orchestrates a highly optimized, multi-stage search and ranking pipeline:
    0. Ranks the input documents by title and selects the most relevant ones.
    1. Parses only the selected documents into sections.
    2. Retrieves candidate sections using fast semantic search.
    3. Uses a local LLM to score and re-rank the candidates for final output.
    """

    def __init__(self, llm_model_path, semantic_model_path):
        print("Initializing models...")
        # Keep default context_length (512) as requested
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_path, model_type='llama', context_length=512)
        self.semantic_model = SentenceTransformer(semantic_model_path)
        # --- BUG FIX: Correctly store the semantic model path as a class attribute ---
        self.semantic_model_path = semantic_model_path
        print("Models Initialized.")

    def _rank_and_filter_documents(self, documents, query, top_k=4):
        """
        Ranks documents based on their titles' semantic similarity to the query.
        """
        print(f"\nRanking {len(documents)} documents to select top {top_k}...")
        
        doc_titles = [doc['title'] for doc in documents]
        
        query_embedding = self.semantic_model.encode(query, convert_to_tensor=True)
        title_embeddings = self.semantic_model.encode(doc_titles, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(query_embedding, title_embeddings)[0]
        
        top_results = torch.topk(cosine_scores, k=min(top_k, len(documents)))
        
        top_docs = [documents[i] for i in top_results[1]]
        
        print("Selected Documents:")
        for doc in top_docs:
            print(f"  - {doc['filename']}")
            
        return top_docs

    def _run_parsing_pipeline(self, pdf_info_list, input_dir, model_path, encoder_path):
        """Processes only the selected PDFs using the custom parser."""
        all_sections = []
        pdf_filenames = [doc['filename'] for doc in pdf_info_list]
        for filename in pdf_filenames:
            pdf_path = os.path.join(input_dir, filename)
            if os.path.exists(pdf_path):
                sections = parse_document_to_sections(pdf_path, model_path, encoder_path)
                if sections:
                    all_sections.extend(sections)
        
        all_sections_path = "all_parsed_sections.json"
        with open(all_sections_path, 'w', encoding='utf-8') as f:
            json.dump(all_sections, f, indent=4)
        print(f"\nSaved {len(all_sections)} parsed sections from selected PDFs to '{all_sections_path}'.")
        
        return all_sections_path, all_sections

    def _get_llm_judgment(self, query, content):
        """Generates a score and justification from the LLM."""
        prompt = f"""[INST]
You are an expert evaluator. Your task is to score a document section's relevance to a user's task on a scale of 1 to 10. You must strictly follow all constraints in the task.

Here is an example:
Task: "Food Contractor Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items."
Content: "Title: Beef Stroganoff. Content: Ingredients: 1 pound beef sirloin..."
Output:
```json
{{
  "relevance_score": 1,
  "justification": "This is irrelevant because the recipe contains beef, which is not vegetarian."
}}
```

Now, evaluate the following:
Task: "{query}"
Content: "{content[:50]}"
Output:
[/INST]
"""
        default_response = {"relevance_score": 1, "justification": "LLM failed to produce a valid JSON output."}
        try:
            response_text = self.llm(prompt, max_new_tokens=150, temperature=0.01)
            match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if match:
                json_part = match.group(1)
                response_json = json.loads(json_part)
                response_json["relevance_score"] = int(response_json.get("relevance_score", 1))
                return response_json
            return default_response
        except Exception:
            return default_response

    def run(self, args):
        start_time = time.time()
        
        print("1. Reading input file...")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        query = f"{input_data['persona']['role']} {input_data['job_to_be_done']['task']}"
        print(f"\n2. Formulated Query: '{query}'")

        top_documents = self._rank_and_filter_documents(input_data['documents'], query, top_k=4)
        
        parsed_sections_path, all_sections = self._run_parsing_pipeline(
            top_documents, args.input_dir, args.classifier_model, args.label_encoder
        )
        
        if not all_sections:
            print("No sections were parsed from the selected documents. Exiting.")
            return

        print(f"\n3. Retrieving top {args.candidates_to_retrieve} candidates with Semantic Search...")
        semantic_engine = SemanticSearch(parsed_sections_path, model_name=self.semantic_model_path)
        candidate_results = semantic_engine.search(query, top_n=args.candidates_to_retrieve)
        
        print(f"\n4. Re-ranking top {len(candidate_results)} candidates with LLM...")
        llm_judgments = []
        scored_candidates = []
        for doc_info, semantic_score in tqdm(candidate_results, desc="LLM Scoring"):
            # --- FIX: Truncate content to prevent context overflow ---
            content_snippet = doc_info.get('content', '')[:1000] # Truncate to ~200 tokens
            content_to_score = f"Title: {doc_info.get('section_title', '')}. Content: {content_snippet}"
            judgment = self._get_llm_judgment(query, content_to_score)
            llm_judgments.append({"document": doc_info.get("document_name"), "section_title": doc_info.get("section_title"), "llm_score": judgment.get("relevance_score"), "llm_justification": judgment.get("justification")})
            scored_candidates.append((doc_info, judgment.get("relevance_score", 1)))

        with open("llm_judgments.json", 'w', encoding='utf-8') as f:
            json.dump(llm_judgments, f, indent=4)
        print("\nSaved detailed LLM judgments to 'llm_judgments.json'")

        final_ranked_list = sorted(scored_candidates, key=lambda item: item[1], reverse=True)

        print("\n5. Generating final output...")
        final_output = { "metadata": { "input_documents": [d['filename'] for d in input_data['documents']], "persona": input_data['persona']['role'], "job_to_be_done": input_data['job_to_be_done']['task'], "processing_timestamp": datetime.now().isoformat() } }
        extracted_sections = []
        for i, (doc_info, score) in enumerate(final_ranked_list[:args.top_n]):
            extracted_sections.append({ "document": doc_info.get("document_name"), "section_title": doc_info.get("section_title"), "importance_rank": i + 1, "page_number": doc_info.get("page_number") })
        final_output["extracted_sections"] = extracted_sections
        subsection_analysis = []
        all_docs_map = {f"{doc.get('document_name')}_{doc.get('page_number')}_{doc.get('section_title')}": doc for doc in all_sections}
        for section in extracted_sections:
            doc_id = f"{section['document']}_{section['page_number']}_{section['section_title']}"
            doc_info = all_docs_map.get(doc_id)
            if doc_info:
                subsection_analysis.append({ "document": doc_info.get("document_name"), "refined_text": doc_info.get("content"), "page_number": doc_info.get("page_number") })
        final_output["subsection_analysis"] = subsection_analysis
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4)
        
        total_time = time.time() - start_time
        print(f"\n✅ Success! Output written to {args.output_file} in {total_time:.2f} seconds.")

# This is a helper class for the orchestrator to use. It should be in a separate file
# but is included here for completeness if the user doesn't have it.
class SemanticSearch:
    def __init__(self, json_file_path, model_name='all-MiniLM-L6-v2'):
        self.corpus_data = self._load_corpus(json_file_path)
        if not self.corpus_data:
            raise ValueError("Corpus is empty for SemanticSearch.")
        self.model = SentenceTransformer(model_name)
        self.corpus_content = [doc.get("content", "") for doc in self.corpus_data]
        self.corpus_embeddings = self.model.encode(self.corpus_content, show_progress_bar=True, convert_to_tensor=True)

    def _load_corpus(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def search(self, query, top_n=5):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = torch.topk(cosine_scores, k=min(top_n, len(self.corpus_data)))
        results_with_scores = []
        for score, idx in zip(top_results[0], top_results[1]):
            results_with_scores.append((self.corpus_data[idx], float(score)))
        return results_with_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Optimized Search Orchestrator.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="output.json")
    parser.add_argument("--semantic_model_path", type=str, default="all_mini_model")
    parser.add_argument("--llm_model_path", type=str, required=True)
    parser.add_argument("--classifier_model", default="heading_classifier_model.joblib")
    parser.add_argument("--label_encoder", default="label_encoder.joblib")
    parser.add_argument("--candidates_to_retrieve", type=int, default=25)
    parser.add_argument("--top_n", type=int, default=5)
    
    args = parser.parse_args()

    orchestrator = OptimizedOrchestrator(llm_model_path=args.llm_model_path, semantic_model_path=args.semantic_model_path)
    orchestrator.run(args)
