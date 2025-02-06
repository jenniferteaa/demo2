import csv
import json

INPUT_CSV = "output.csv"
OUTPUT_JSONL = "statement.json"

def csv_to_ndjson(input_csv: str, output_jsonl: str, claim_column: str = "revise") -> None:
    """
    Reads `input_csv`, extracts the `claim_column`,
    and writes each row's value as NDJSON to `output_jsonl`.
    
    Each line looks like:
      {"input_info": {"claim": "..."}}
    """
    with open(input_csv, "r", encoding="utf-8") as csv_file, open(output_jsonl, "w", encoding="utf-8") as ndjson_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            claim_text = row[claim_column].strip()  # e.g. "revise" or "revise_statements"
            
            # Build a single JSON object for NDJSON
            obj = {
                "input_info": {
                    "claim": claim_text
                }
            }
            # Write it as one line
            ndjson_file.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Customize as needed
    csv_to_ndjson(INPUT_CSV, OUTPUT_JSONL, claim_column="revise")
