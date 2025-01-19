import torch
import torch.amp as amp
import numpy as np
from tqdm import tqdm
import os
from evaluation import get_validation_recalls

def train_step(batch, model, loss_fn, miner, optimizer, scaler):
    """
    Single training step - handles the GSVCities format where batch contains
    multiple images per place
    """
    images, labels = batch  # images shape: [B, K, C, H, W]
    B, K, ch, h, w = images.shape
    
    # Reshape images to [B*K, C, H, W] and move to GPU
    images = images.view(-1, ch, h, w).cuda()
    labels = labels.view(-1).cuda()
    
    acc = 0.0
    
    with amp.autocast(device_type='cuda'):
        # Forward pass
        descriptors = model(images)
        
        # Mining and loss calculation
        if miner is not None:
            miner_outputs = miner(descriptors, labels)
            loss = loss_fn(descriptors, labels, miner_outputs)
            
            unique_mined = torch.unique(miner_outputs[0])
            n_mined = unique_mined.numel()
            n_samples = descriptors.size(0)
            acc = 1.0 - (n_mined / n_samples)
        else:
            loss = loss_fn(descriptors, labels)
            if isinstance(loss, tuple):
                loss, acc = loss

    # Backward pass
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # Cleanup
    del images, labels, descriptors
    torch.cuda.empty_cache()
    
    return loss.item(), acc

def train_model(**kwargs):
    # Extract arguments
    model = kwargs.get('model')
    loss_fn = kwargs.get('loss_fn')
    miner = kwargs.get('miner', None)
    optimizer = kwargs.get('optimizer')
    train_loader = kwargs.get('train_dataloader')
    val_loader = kwargs.get('test_dataloader')
    test_dataset = kwargs.get('test_dataset')
    num_epochs = kwargs.get('num_epochs', 30)
    verbose = kwargs.get('verbose', True)
    experiment_name = kwargs.get('job_name', 'experiment')
    scheduler = kwargs.get('scheduler', None)
    
    # Setup
    scaler = amp.GradScaler('cuda')
    best_r1 = 0
    best_r5 = 0
    epochs_no_improve = 0
    patience = kwargs.get('patience', 5)
    
    # Create weights directory
    os.makedirs('./weights', exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_losses = []
        train_accs = []
        
        for batch in tqdm(train_loader, desc="Training"):
            loss, acc = train_step(batch, model, loss_fn, miner, optimizer, scaler)
            train_losses.append(loss)
            train_accs.append(acc)
        
        # Validation phase
        model.eval()
        all_descriptors = torch.tensor([])
        all_indexes = torch.tensor([])
        
        with torch.no_grad(), amp.autocast(device_type='cuda'):
            for batch in tqdm(val_loader, desc="Validating"):
                images, indexes = batch
                descriptors = model(images.cuda()).cpu()
                all_descriptors = torch.cat((all_descriptors, descriptors), dim=0)
                all_indexes = torch.cat((all_indexes, indexes), dim=0)
        
        # Split into queries and database
        query_size = test_dataset.num_queries
        database = all_descriptors[query_size:]
        database_indexes = all_indexes[query_size:]
        queries = all_descriptors[:query_size]
        queries_indexes = all_indexes[:query_size]
        
        # Compute recalls
        recalls_dict, _ = get_validation_recalls(
            r_list=database,
            q_list=queries,
            q_list_indexes=queries_indexes,
            r_list_indexes=database_indexes,
            k_values=[1, 5],
            gt=test_dataset.ground_truth,
            db_size=len(database),
            query_size=query_size,
            verbose=verbose,
            dataset_name='val_loader'
        )
        
        # Schedule learning rate if needed
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(recalls_dict[1])
            else:
                scheduler.step()
        
        # Save best model
        if recalls_dict[1] + recalls_dict[5] > best_r1 + best_r5:
            best_r1 = recalls_dict[1]
            best_r5 = recalls_dict[5]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'recalls': recalls_dict,
            }, f'./weights/{experiment_name}.pth')
            
            epochs_no_improve = 0
            if verbose:
                print("New best model saved!")
        else:
            epochs_no_improve += 1
            
        # Early stopping check
        if epochs_no_improve >= patience:
            if verbose:
                print("Early stopping triggered!")
            break
        
        # Print epoch summary
        if verbose:
            print(f"Loss: {np.mean(train_losses):.4f}")
            print(f"Mining Acc: {np.mean(train_accs):.4f}")
            print(f"R@1: {recalls_dict[1]:.4f}")
            print(f"R@5: {recalls_dict[5]:.4f}")
        
        # Cleanup
        torch.cuda.empty_cache()
    
    return model, best_r1, best_r5