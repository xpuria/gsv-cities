# main.py
#import pytorch_lightning as pl
import torch
import torch.amp as amp
#from pytorch_lightning.callbacks import ModelCheckpoint
import utils
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from models import helper

class VPRModel(pl.LightningModule):
    def __init__(self,
                backbone_arch='resnet18',
                pretrained=True,
                layers_to_freeze=1,
                layers_to_crop=[4, 5],    # Crop after conv3
                agg_arch='GeM',           # Default to GeM pooling
                agg_config={'p': 3},      # GeM pooling parameter
                lr=0.03,
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                loss_name='MultiSimilarityLoss',
                miner_name='MultiSimilarityMiner',
                miner_margin=0.1,
                faiss_gpu=False
                ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model components
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        
        # Training parameters
        self.lr = lr
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult
        
        # Loss setup
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = []
        self.faiss_gpu = faiss_gpu
        
        # Mixed precision
        self.scaler = amp.GradScaler()

    def forward(self, x):
        with amp.autocast(device_type='cuda'):
            x = self.backbone(x)
            x = self.aggregator(x)
        return x

    def configure_optimizers(self):
        # Setup optimizer
        if self.optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum
            )
        elif self.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )

        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.milestones,
            gamma=self.lr_mult
        )
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        places, labels = batch
        BS, N, ch, h, w = places.shape
        
        # Reshape input
        images = places.view(BS*N, ch, h, w).cuda()
        labels = labels.view(-1).cuda()
        
        # Forward pass with mixed precision
        with amp.autocast(device_type='cuda'):
            descriptors = self(images)
            
            if self.miner is not None:
                miner_outputs = self.miner(descriptors, labels)
                loss = self.loss_fn(descriptors, labels, miner_outputs)
                
                # Calculate mining statistics
                unique_mined = torch.unique(miner_outputs[0])
                n_mined = unique_mined.numel()
                n_samples = descriptors.size(0)
                batch_acc = 1.0 - (n_mined / n_samples)
            else:
                loss = self.loss_fn(descriptors, labels)
                batch_acc = 0.0
                if isinstance(loss, tuple):
                    loss, batch_acc = loss
        
        self.batch_acc.append(batch_acc)
        self.log('b_acc', sum(self.batch_acc) / len(self.batch_acc), prog_bar=True)
        self.log('loss', loss, prog_bar=True)
        
        return loss

    def on_train_epoch_end(self):
        self.batch_acc = []

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        with amp.autocast(device_type='cuda'):
            descriptors = self(places)
        return descriptors.detach().cpu()

    def validation_epoch_end(self, val_step_outputs):
        if not isinstance(val_step_outputs[0], list):
            val_step_outputs = [val_step_outputs]
        
        for i, (val_name, val_dataset) in enumerate(zip(self.trainer.datamodule.val_set_names, 
                                                      self.trainer.datamodule.val_datasets)):
            feats = torch.cat(val_step_outputs[i], dim=0)
            
            # Split references and queries
            r_list = feats[:val_dataset.num_references]
            q_list = feats[val_dataset.num_references:]
            
            # Calculate recalls
            recalls_dict, _ = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10],
                gt=val_dataset.ground_truth,
                print_results=True,
                dataset_name=val_name,
                faiss_gpu=self.faiss_gpu
            )
            
            # Log metrics
            self.log(f'{val_name}/R1', recalls_dict[1], prog_bar=True)
            self.log(f'{val_name}/R5', recalls_dict[5], prog_bar=True)
            
            # Clear memory
            torch.cuda.empty_cache()