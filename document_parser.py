import os
import json
import argparse
import joblib
import pandas as pd

# ==============================================================================
# PART 1: EXTERNAL DEPENDENCY
# The feature extraction logic is imported from your trusted script.
# ==============================================================================
from feature_extractor import extract_features_from_pdf


# ==============================================================================
# PART 2: SECTION GROUPING LOGIC
# ==============================================================================

HEADING_HIERARCHY = {
    "Title": 0, "H1": 1, "H2": 2, "H3": 3, "H4": 4, "H5": 5, "H6": 6
}

def group_text_into_sections(labeled_lines, pdf_filename):
    """
    Groups a flat list of labeled text lines into logical, hierarchical sections.
    """
    sections = []
    current_section = None
    active_heading_stack = []

    for item in labeled_lines:
        label = item.get('label')
        text = item.get('text', '').strip()
        page_num = item.get('page')

        if not text:
            continue

        is_heading = label in HEADING_HIERARCHY

        if is_heading:
            # If a section is already being built, finish it and add to the list.
            if current_section:
                # Consolidate content and add the completed section
                current_section['content'] = ' '.join(current_section['content'].split())
                sections.append(current_section)

            heading_level = HEADING_HIERARCHY[label]
            
            # Manage the hierarchy stack for nested headings.
            while active_heading_stack and active_heading_stack[-1]['level'] >= heading_level:
                active_heading_stack.pop()
            
            # Create the new section.
            current_section = {
                "document_name": pdf_filename,
                "page_number": page_num,
                "section_title": text,
                "content": "", # This will be populated by subsequent 'Body' text.
                "hierarchy_level": heading_level,
                "full_path": [h['title'] for h in active_heading_stack] + [text]
            }
            active_heading_stack.append({'title': text, 'level': heading_level})

        elif label == 'Body' and current_section:
            # If it's body text and we are inside a section, append it.
            current_section['content'] += f" {text}"
        
    # Add the very last section to the list after the loop finishes.
    if current_section:
        current_section['content'] = ' '.join(current_section['content'].split())
        sections.append(current_section)

    return sections

# ==============================================================================
# PART 3: MAIN ORCHESTRATOR
# ==============================================================================

def parse_document_to_sections(pdf_path, model_path, encoder_path):
    """
    Orchestrates the entire process: feature extraction, prediction, and section grouping.
    """
    pdf_filename = os.path.basename(pdf_path)
    print(f"Processing '{pdf_filename}'...")

    # Step 1: Load Model and Encoder
    try:
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
    except Exception as e:
        print(f"Error loading model/encoder: {e}. Please ensure files exist.")
        return None

    # Step 2: Extract features from the PDF using the imported function
    features_list = extract_features_from_pdf(pdf_path)
    if not features_list:
        print("Could not extract any features from the PDF.")
        return None
    
    df = pd.DataFrame(features_list)
    
    # Step 3: Prepare DataFrame for prediction, ensuring features match the model
    model_features = model.feature_name_
    
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
            
    X_predict = df[model_features]

    # Step 4: Predict labels for each text line
    predictions_encoded = model.predict(X_predict)
    predictions_labels = label_encoder.inverse_transform(predictions_encoded)
    df['predicted_label'] = predictions_labels

    # Step 5: Create a flat list of labeled text lines to be grouped
    labeled_lines = []
    for _, row in df.iterrows():
        if row['predicted_label'] != 'Other':
            labeled_lines.append({
                "label": row['predicted_label'],
                "text": row['text'],
                "page": int(row['page_num'])
            })
    
    # Step 6: Group the flat list into structured sections
    structured_sections = group_text_into_sections(labeled_lines, pdf_filename)
    print(f"Successfully parsed into {len(structured_sections)} sections.")
    
    return structured_sections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A script to parse a PDF into structured, logical sections."
    )
    parser.add_argument("pdf_file", help="The path to the PDF file to process.")
    parser.add_argument("--model", default="heading_classifier_model.joblib", help="Path to the classifier model file.")
    parser.add_argument("--encoder", default="label_encoder.joblib", help="Path to the label encoder file.")
    
    args = parser.parse_args()

    if not os.path.exists(args.pdf_file):
        print(f"Error: Input PDF file not found at '{args.pdf_file}'")
    else:
        # This function now returns the final, grouped sections
        parsed_sections = parse_document_to_sections(args.pdf_file, args.model, args.encoder)
        
        if parsed_sections:
            # Save the final structured output
            base_name = os.path.splitext(os.path.basename(args.pdf_file))[0]
            output_json_path = f"{base_name}_sections.json"
            
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_sections, f, indent=4, ensure_ascii=False)
            
            print(f"\nSuccessfully created structured sections file at: {output_json_path}")

