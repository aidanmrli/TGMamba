import os
import torch
from tqdm import tqdm

def combine_dataset_files(input_dir, output_file):
    data_list = []
    file_names = sorted([f for f in os.listdir(input_dir) if f.startswith('data_')])
    
    for file_name in tqdm(file_names, desc="Combining files"):
        data = torch.load(os.path.join(input_dir, file_name))
        data_list.append(data)
    
    combined_data = {
        'data': data_list,
        'num_samples': len(data_list)
    }
    
    torch.save(combined_data, output_file)
    print(f"Combined dataset saved to {output_file}")

# Run this for each split
for split in ['train', 'val', 'test']:
    input_dir = f'/home/amli/TGMamba/data/preprocessed_dodh_{split}'
    output_file = f'/home/amli/TGMamba/data/dodh_{split}.pt'
    combine_dataset_files(input_dir, output_file)