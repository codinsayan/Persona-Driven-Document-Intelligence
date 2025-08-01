import os
import argparse
from optimized_orchestrator import OptimizedOrchestrator, SemanticSearch

def run_solution(collection_path):
    """
    This is the main entry point for the hackathon solution.
    It takes a path to a collection directory and runs the
    optimized orchestration pipeline on it.
    """
    # Define paths relative to the collection directory
    input_file_path = os.path.join(collection_path, "challenge1b_input.json")
    input_dir_path = os.path.join(collection_path, "PDFs")
    output_file_path = os.path.join(collection_path, "challenge1b_output.json")
    
    # Define paths relative to the script's root location
    script_root = os.path.dirname(os.path.abspath(__file__))
    semantic_model_path = os.path.join(script_root, "semantic_model")
    llm_model_path = os.path.join(script_root, "llm_model/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    classifier_model_path = os.path.join(script_root, "models/heading_classifier_model.joblib")
    label_encoder_path = os.path.join(script_root, "models/label_encoder.joblib")

    # --- Argument Simulation ---
    # We create an "args" object to pass to the orchestrator's run method
    args = argparse.Namespace(
        input_file=input_file_path,
        input_dir=input_dir_path,
        output_file=output_file_path,
        semantic_model_path=semantic_model_path,
        llm_model_path=llm_model_path,
        classifier_model=classifier_model_path,
        label_encoder=label_encoder_path,
        candidates_to_retrieve=15,
        top_n=5
    )

    # --- Execution ---
    print(f"--- Starting Persona-Driven Document Intelligence Solution for '{collection_path}' ---")
    
    if not os.path.exists(args.input_file):
        print(f"FATAL ERROR: Input file not found at {args.input_file}")
        return

    orchestrator = OptimizedOrchestrator(
        llm_model_path=args.llm_model_path,
        semantic_model_path=args.semantic_model_path
    )
    orchestrator.run(args)
    
    print(f"\n--- Solution Finished for '{collection_path}' ---")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Run the main solution pipeline on a collection directory.")
    # parser.add_argument("collection_directory", help="The path to the collection directory (e.g., 'main').", default="./MAIN")
    main_folder = './MAIN' 
    
    run_solution(main_folder)