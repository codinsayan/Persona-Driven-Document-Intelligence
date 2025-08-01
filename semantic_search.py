import json
import time
import argparse
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

class SemanticSearch:
    """
    A class to perform semantic (dense retrieval) search using sentence-transformer models.

    This class loads data from a JSON file, generates vector embeddings for the
    'content' of each JSON object, and retrieves the most semantically similar
    documents based on a query's embedding.
    """

    def __init__(self, json_file_path, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the SemanticSearch object.

        Args:
            json_file_path (str): The file path to the JSON data.
            model_name (str): The name of the sentence-transformer model to use.
        """
        self.corpus_data = self._load_corpus(json_file_path)
        if not self.corpus_data:
            raise ValueError("Corpus is empty. Please check the JSON file path and content.")

        print(f"Loading sentence-transformer model: {model_name}...")
        # This will download the model on the first run, or load from a local path
        self.model = SentenceTransformer(model_name)
        print("Model loaded.")

        # We will create embeddings for the 'content' of each document
        self.corpus_content = [doc.get("content", "") for doc in self.corpus_data]
        self.corpus_embeddings = self._embed_corpus()

    def _load_corpus(self, file_path):
        """
        Loads the corpus from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            list of dict: A list of document dictionaries.
        """
        print(f"Loading corpus from {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return []
        except json.JSONDecodeError:
            print(f"Error: The file {file_path} is not a valid JSON file.")
            return []

    def _embed_corpus(self):
        """
        Generates embeddings for the entire corpus.
        This is a one-time operation for a given corpus.
        """
        print("\nGenerating embeddings for the corpus. This may take a while...")
        # The model.encode method handles the batching and progress bar for us.
        # convert_to_tensor=True is recommended for performance with util.cos_sim
        embeddings = self.model.encode(
            self.corpus_content,
            show_progress_bar=True,
            convert_to_tensor=True
        )
        print("Corpus embedding complete.")
        return embeddings

    def search(self, query, top_n=5):
        """
        Performs a semantic search for the given query.

        Args:
            query (str): The search query.
            top_n (int): The number of top results to return.

        Returns:
            list of tuple: A list of tuples, where each tuple contains the
                           full document dictionary and its cosine similarity score.
        """
        print(f"\nSearching for query: '{query}'")
        
        # 1. Generate the embedding for the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # 2. Compute cosine similarity between the query and all document embeddings
        # util.cos_sim returns a 2D tensor, so we get the first row.
        cosine_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]

        # 3. Rank the documents based on the scores using torch.topk
        top_results = torch.topk(cosine_scores, k=min(top_n, len(self.corpus_data)))

        # 4. Format the results
        results_with_scores = []
        # torch.topk returns a tuple of (values, indices)
        for score, idx in zip(top_results[0], top_results[1]):
            # The score is a tensor, convert it to a standard float
            # The idx is the index in the original corpus
            results_with_scores.append((self.corpus_data[idx], float(score)))
            
        return results_with_scores

if __name__ == '__main__':
    # --- Setup Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Perform Semantic (Dense) search on a JSON file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to the parsed JSON file containing the document corpus."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The search query string."
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="The number of top results to return (default: 5)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default='all-MiniLM-L6-v2',
        help="The sentence-transformer model to use (default: 'all-MiniLM-L6-v2').\nCan be a model name or a local path."
    )

    args = parser.parse_args()

    # --- Main Execution ---

    # 1. Create an instance of the SemanticSearch class
    start_time = time.time()
    semantic_search = SemanticSearch(args.json_file, model_name=args.model)
    end_time = time.time()
    print(f"\nTotal setup and embedding time: {end_time - start_time:.4f} seconds")

    # 2. Perform a search
    results = semantic_search.search(query=args.query, top_n=args.top_n)

    # 3. Print the results
    print("\n--- Search Results ---")
    if not results:
        print("No results found.")
    else:
        for doc, score in results:
            print(f"Score: {score:.4f} | Document: {doc.get('document_name')} | Section: {doc.get('section_title')}")
            content_snippet = doc.get('content', '')[:150]
            print(f"Content: {content_snippet}...")
            print("-" * 20)

    # 4. Prepare results for RRF
    print("\n--- Preparing for RRF ---")
    rrf_input_semantic = []
    for rank, (doc, score) in enumerate(results, 1):
        unique_id = f"{doc.get('document_name')}_{doc.get('page_number')}_{doc.get('section_title')}"
        rrf_input_semantic.append({'id': unique_id, 'rank': rank, 'score': score})

    print("Semantic search results ready for RRF (JSON format):")
    print(json.dumps(rrf_input_semantic, indent=2))
