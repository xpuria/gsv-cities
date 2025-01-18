# experiments/train.py
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
from pathlib import Path
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from main import VPRModel

def train_model(config_name, model_config, save_dir="/content/drive/MyDrive/gsv_cities/weights"):
    """
    Train a model with specified configuration
    Args:
        config_name: Name of the configuration (for saving)
        model_config: Dictionary of model parameters
        save_dir: Directory to save model weights
    """
    # Create save directory
    save_path = Path(save_dir) / config_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    datamodule = GSVCitiesDataModule(
        batch_size=32,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False,
        image_size=(320, 320),
        num_workers=2,
        show_data_stats=True,
        val_set_names=['sfxs_val']  # Validate on SF-XS
    )
    
    # Initialize model
    model = VPRModel(**model_config)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_path),
        filename=f'{config_name}_'+'{epoch:02d}_{sfxs_val/R1:.4f}',
        save_top_k=3,
        mode='max',
        monitor='sfxs_val/R1'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='sfxs_val/R1',
        patience=5,
        mode='max',
        verbose=True
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=30,
        precision=16,
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root_dir=str(save_path),
        check_val_every_n_epoch=1,
    )
    
    # Train model
    trainer.fit(model=model, datamodule=datamodule)
    
    # Save final model
    final_path = save_path / 'final_model.pth'
    torch.save(model.state_dict(), str(final_path))
    
    return model, checkpoint_callback.best_model_path