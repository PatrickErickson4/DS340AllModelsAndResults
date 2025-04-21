import os
import json
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

"""
This script generates train/validation/test CSVs by splitting all images in
PlantVillage-Tomato/All-Tomato into 70/15/15, stratified by class.
It does not move or delete any images.
"""

def create_dataset(data_dir="PlantVillage-Tomato", output_dir=None,
                   train_frac=0.60, valid_frac=0.20, test_frac=0.20, random_state=100):
    # Use All-Tomato as the source
    all_dir = os.path.join(data_dir, 'All-Tomato')
    # Gather all image paths
    image_paths = glob.glob(os.path.join(all_dir, '*', '*'))
    if not image_paths:
        print(f"No images found in {all_dir}")
        return
    # Extract labels from folder names: e.g. Tomato__Label___...
    labels = [os.path.basename(os.path.dirname(p)).split('___')[1] for p in image_paths]
    # Build DataFrame
    df = pd.DataFrame({'filepath': image_paths, 'label_tag': labels})
    # Create label-to-index mapping
    classes = sorted(df['label_tag'].unique())
    mapping = {lab: idx for idx, lab in enumerate(classes)}
    df['label'] = df['label_tag'].map(mapping)
    # Stratified split: train vs temp
    train_df, temp_df = train_test_split(
        df, train_size=train_frac, stratify=df['label'], random_state=random_state)
    # Then split temp into validation and test
    valid_rel = valid_frac / (valid_frac + test_frac)
    valid_df, test_df = train_test_split(
        temp_df, train_size=valid_rel, stratify=temp_df['label'], random_state=random_state)
    # Determine output dir
    out_dir = output_dir or data_dir
    os.makedirs(out_dir, exist_ok=True)
    # Write CSVs
    train_csv = os.path.join(out_dir, 'train.csv')
    valid_csv = os.path.join(out_dir, 'valid.csv')
    test_csv  = os.path.join(out_dir, 'test.csv')
    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Save class mapping
    with open(os.path.join(out_dir, 'class_mapping.json'), 'w') as f:
        json.dump(mapping, f)
    # Report
    print(f"Wrote {len(train_df)} entries to {train_csv}")
    print(f"Wrote {len(valid_df)} entries to {valid_csv}")
    print(f"Wrote {len(test_df)} entries to {test_csv}")
    print(f"Classes: {classes}")

if __name__ == '__main__':
    create_dataset("Tomato-Merged")
