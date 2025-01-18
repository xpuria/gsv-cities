# experiments/evaluate.py
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataloaders.val.SFXSDataset import SFXSDataset
from dataloaders.val.TokyoXSDataset import TokyoXSDataset
from utils.validation import get_validation_recalls
from main import VPRModel
import json
from pathlib import Path
import numpy as np
from prettytable import PrettyTable

class Evaluator:
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
        self.transform = T.Compose([
            T.Resize((320, 320)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self):
        """Load the model with specified configuration"""
        model = VPRModel(**self.config)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model.cuda()
    
    def evaluate_dataset(self, dataset, dataset_name):
        """Evaluate model on a specific dataset"""
        model = self.load_model()
        loader = DataLoader(dataset, batch_size=32, num_workers=2)
        
        # Get all descriptors
        all_descriptors = []
        with torch.no_grad():
            for batch in loader:
                images, _ = batch
                descriptors = model(images.cuda()).cpu()
                all_descriptors.append(descriptors)
        
        all_descriptors = torch.cat(all_descriptors, dim=0)
        
        # Split into references and queries
        r_list = all_descriptors[:dataset.num_references]
        q_list = all_descriptors[dataset.num_references:]
        
        # Calculate recalls
        recalls_dict, _ = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10],
            gt=dataset.ground_truth,
            print_results=True,
            dataset_name=dataset_name
        )
        
        return recalls_dict
    
    def evaluate_all(self, save_results=True):
        """Evaluate on all test datasets"""
        results = {}
        
        # Evaluate on SF-XS test
        sfxs_dataset = SFXSDataset(subset_name="sfxs_test", transform=self.transform)
        results['sfxs'] = self.evaluate_dataset(sfxs_dataset, "SF-XS Test")
        
        # Evaluate on Tokyo-XS
        tokyo_dataset = TokyoXSDataset(transform=self.transform)
        results['tokyo'] = self.evaluate_dataset(tokyo_dataset, "Tokyo-XS")
        
        if save_results:
            save_path = Path("/content/drive/MyDrive/gsv_cities/results")
            save_path.mkdir(parents=True, exist_ok=True)
            
            with open(save_path / f"{Path(self.model_path).stem}_results.json", 'w') as f:
                json.dump(results, f, indent=4)
        
        return results

def print_comparison_table(results_dict):
    """Print a comparison table of results"""
    table = PrettyTable()
    table.field_names = ["Configuration", "SF-XS R@1", "SF-XS R@5", "Tokyo R@1", "Tokyo R@5"]
    
    for config_name, results in results_dict.items():
        row = [
            config_name,
            f"{results['sfxs'][1]*100:.2f}%",
            f"{results['sfxs'][5]*100:.2f}%",
            f"{results['tokyo'][1]*100:.2f}%",
            f"{results['tokyo'][5]*100:.2f}%"
        ]
        table.add_row(row)
    
    print("\nResults Comparison:")
    print(table)