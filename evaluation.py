import torch
import torch.amp as amp
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
import faiss
import faiss.contrib.torch_utils

def get_validation_recalls(r_list,
                         q_list,
                         q_list_indexes,
                         r_list_indexes,
                         k_values,
                         gt,
                         db_size,
                         query_size,
                         verbose=False,
                         faiss_gpu=False,
                         dataset_name='dataset'):
    """
    Calculate validation recalls - modified to work with provided dataset formats
    """
    embed_size = r_list.shape[1]
    
    # Setup FAISS index
    if faiss_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True
        flat_config.device = 0
        faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
    else:
        faiss_index = faiss.IndexFlatL2(embed_size)
    
    # Add reference embeddings to index
    faiss_index.add(r_list)
    
    # Search for nearest neighbors
    _, predictions = faiss_index.search(q_list, max(k_values))
    
    # Calculate recalls
    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break
    
    correct_at_k = correct_at_k / len(predictions)
    recalls_dict = {k:v for k,v in zip(k_values, correct_at_k)}
    
    # Print results if requested
    if verbose:
        print('\n')
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in k_values]
        table.add_row(['Recall@K'] + [f'{100*v:.2f}' for v in correct_at_k])
        print(table.get_string(title=f"Performance on {dataset_name}"))
    
    return recalls_dict, predictions

def eval_model(**kwargs):
    """
    Evaluate model on test dataset - handles both SF-XS and Tokyo-XS formats
    """
    model = kwargs['model']
    test_loader = kwargs['test_dataloader']
    test_dataset = kwargs['test_dataset']
    verbose = kwargs.get('verbose', True)
    
    # Compute descriptors
    all_descriptors = torch.tensor([])
    all_indexes = torch.tensor([])
    
    model.eval()
    with torch.no_grad(), amp.autocast(device_type='cuda'):
        for batch in tqdm(test_loader, desc="Computing descriptors"):
            images, indexes = batch
            descriptors = model(images.cuda()).cpu()
            all_descriptors = torch.cat((all_descriptors, descriptors), dim=0)
            all_indexes = torch.cat((all_indexes, indexes), dim=0)
    
    # Split into database and queries
    query_size = test_dataset.num_queries
    database = all_descriptors[query_size:]
    database_indexes = all_indexes[query_size:]
    queries = all_descriptors[:query_size]
    queries_indexes = all_indexes[:query_size]
    
    # Compute recalls
    recalls_dict, predictions = get_validation_recalls(
        r_list=database,
        q_list=queries,
        q_list_indexes=queries_indexes,
        r_list_indexes=database_indexes,
        k_values=[1, 5],
        gt=test_dataset.ground_truth,
        db_size=len(database),
        query_size=query_size,
        verbose=verbose,
        dataset_name=test_dataset.__class__.__name__
    )
    
    if verbose:
        print(f'R@1: {recalls_dict[1]:.4f}')
        print(f'R@5: {recalls_dict[5]:.4f}')
    
    return queries, database, predictions

def visualize_predictions(model, dataset, k=5):
    """
    Optional: Visualize model predictions
    """
    model.eval()
    # Add visualization code here if needed
    pass