import torch
import torch.amp as amp
import numpy as np
from tqdm.notebook import tqdm
import os

from models.helper import get_backbone, get_aggregator
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule 
from utils import get_loss, get_miner
from utils.validation import get_validation_recalls

class VPRModel(torch.nn.Module):
    """Combined backbone and aggregator model"""
    def __init__(self, backbone, aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

def train_step(batch, model, loss_fn, miner, optimizer, scaler):
    """Perform one training step"""
    places, labels = batch
    BS, N, ch, h, w = places.shape
    
    # Reshape places and labels
    images = places.view(BS*N, ch, h, w).cuda()
    labels = labels.view(-1).cuda()
    
    acc = 0.0
    
    # Forward pass with automatic mixed precision
    with amp.autocast(device_type='cuda'):
        descriptors = model(images)
        
        if miner is not None:
            miner_outputs = miner(descriptors, labels)
            loss = loss_fn(descriptors, labels, miner_outputs)
            
            # Calculate batch accuracy 
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
    
    # Clean up
    del images, labels, descriptors
    torch.cuda.empty_cache()
    
    return loss, acc

def train_model(
    # Model parameters
    backbone_arch='resnet50',
    pretrained=True,
    layers_to_freeze=2,
    layers_to_crop=[],
    
    # Aggregator parameters
    agg_arch='ConvAP',
    agg_config={'in_channels': 2048,
                'out_channels': 512,
                's1': 2,
                's2': 2},
    
    # Training parameters
    batch_size=100,
    img_per_place=4,
    lr=0.0002,
    optimizer_name='adam',
    weight_decay=0,
    momentum=0.9,
    warmup_steps=600,
    num_epochs=30,
    milestones=[5, 10, 15, 25],
    lr_mult=0.3,
    
    # Loss parameters
    loss_name='MultiSimilarityLoss',
    miner_name='MultiSimilarityMiner',
    miner_margin=0.1,
    
    # Dataset parameters
    image_size=(320, 320),
    num_workers=8,
    val_set_names=['pitts30k_val', 'msls_val'],
    
    # Output parameters
    output_dir='./weights',
    experiment_name=None
):
    """Main training function with all parameters as function arguments"""
    
    if experiment_name is None:
        experiment_name = f"{backbone_arch}_{agg_arch}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    backbone = get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
    aggregator = get_aggregator(agg_arch, agg_config)
    model = VPRModel(backbone, aggregator).cuda()
    
    # Setup loss and mining functions
    loss_fn = get_loss(loss_name)
    miner = get_miner(miner_name, miner_margin)
    
    # Setup optimizer
    if optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                  lr=lr,
                                  momentum=momentum,
                                  weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                   lr=lr,
                                   weight_decay=weight_decay)
    
    # Setup data module
    datamodule = GSVCitiesDataModule(
        batch_size=batch_size,
        img_per_place=img_per_place,
        image_size=image_size,
        num_workers=num_workers,
        val_set_names=val_set_names
    )
    
    # Setup grad scaler for mixed precision training
    scaler = amp.GradScaler()
    
    # Training loop
    best_r1 = 0
    best_r5 = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_losses = []
        train_accs = []
        
        train_loader = datamodule.train_dataloader()
        for batch in tqdm(train_loader, desc="Training"):
            loss, acc = train_step(batch, model, loss_fn, miner, optimizer, scaler)
            train_losses.append(loss.item())
            train_accs.append(acc)
            
        avg_loss = np.mean(train_losses)
        avg_acc = np.mean(train_accs)
        print(f"Training - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_loaders = datamodule.val_dataloader()
        
        with torch.no_grad():
            for i, (val_name, val_loader) in enumerate(zip(val_set_names, val_loaders)):
                print(f"\nValidating on {val_name}")
                
                all_descriptors = []
                for batch in tqdm(val_loader, desc="Computing descriptors"):
                    places, _ = batch
                    descriptors = model(places.cuda()).cpu()
                    all_descriptors.append(descriptors)
                
                descriptors = torch.cat(all_descriptors, dim=0)
                
                # Split into references and queries
                num_references = val_loader.dataset.num_references
                r_list = descriptors[:num_references]
                q_list = descriptors[num_references:]
                
                # Calculate recalls
                recalls, _ = get_validation_recalls(
                    r_list=r_list,
                    q_list=q_list,
                    k_values=[1, 5, 10],
                    gt=val_loader.dataset.ground_truth,
                    print_results=True,
                    dataset_name=val_name
                )
                
                # Track best model
                if val_name == val_set_names[0]:  # Track on first validation set
                    if recalls[1] > best_r1:
                        best_r1 = recalls[1]
                        best_r5 = recalls[5]
                        # Save best model
                        save_path = os.path.join(output_dir, f'best_model_{experiment_name}.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'recall@1': best_r1,
                            'recall@5': best_r5,
                        }, save_path)
                        print(f"Saved best model to {save_path}")
                        
        # Clean up after each epoch
        torch.cuda.empty_cache()
    
    print(f"\nTraining completed! Best R@1: {best_r1:.4f}, Best R@5: {best_r5:.4f}")
    return model, best_r1, best_r5

# Example usage in Colab:
if __name__ == '__main__':
    # You can modify these parameters as needed
    model, r1, r5 = train_model(
        backbone_arch='resnet50',
        agg_arch='ConvAP',
        batch_size=100,
        num_epochs=30,
        experiment_name='resnet50_convap_test'
    )