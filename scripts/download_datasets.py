import json
import random
from pathlib import Path
import sys
from tqdm import tqdm

# Add strider to path if not installed
sys.path.insert(0, str(Path(__file__).parent.parent))
from strider.structure.mfe import fold_mfe
def main():
    print("Downloading ArchiveII dataset from Hugging Face...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        return
        
    # Load ArchiveII
    dataset = load_dataset('multimolecule/archiveii', split='test')
    
    dataset_dir = Path(__file__).parent.parent / "data" / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    parsed_data = []
    
    for item in tqdm(dataset, desc="Processing ArchiveII"):
        seq = item['sequence'].upper().replace('T', 'U')
        struct = item['secondary_structure']

        if len(seq) <= 100 and "(" in struct and ")" in struct:
            # Calculate true energy using Strider's native exact tables
            _, energy, _ = fold_mfe(seq, celsius=37.0, material="rna")
            
            parsed_data.append({
                "id": item['id'],
                "family": item['family'],
                "sequence": seq,
                "structure": struct,
                "energy": energy
            })
            
    random.shuffle(parsed_data)
    train_size = int(len(parsed_data) * 0.8)
    train_set = parsed_data[:train_size]
    val_set = parsed_data[train_size:]

    train_file = dataset_dir / "archiveII_train.json"
    val_file = dataset_dir / "archiveII_val.json"

    with open(train_file, "w") as f:
        json.dump(train_set, f, indent=2)
    with open(val_file, "w") as f:
        json.dump(val_set, f, indent=2)

    print(f"Train: {len(train_set)} examples → {train_file}")
    print(f"Val:   {len(val_set)} examples → {val_file}")

if __name__ == "__main__":
    main()
