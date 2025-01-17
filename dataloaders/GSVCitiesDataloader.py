import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from prettytable import PrettyTable

# 1) Import your custom SF-XS and Tokyo-XS datasets
from dataloaders.val.SFXSDataset import SFXSDataset
from dataloaders.val.TokyoXSDataset import TokyoXSDataset

# 2) If you still need to train on GSV-cities for the "train" part:
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

# 3) Optional: if you want to keep the same naming
IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}

TRAIN_CITIES = [
    'Bangkok', 'BuenosAires', 'LosAngeles',
    'MexicoCity', 'OSL', 'Rome', 'Barcelona',
    'Chicago', 'Madrid', 'Miami', 'Phoenix',
    'TRT', 'Boston', 'Lisbon', 'Medellin',
    'Minneapolis', 'PRG', 'WashingtonDC',
    'Brussels', 'London', 'Melbourne',
    'Osaka', 'PRS'
]

class GSVCitiesDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=32,
                 img_per_place=4,
                 min_img_per_place=4,
                 shuffle_all=False,
                 image_size=(480, 640),
                 num_workers=4,
                 show_data_stats=True,
                 cities=TRAIN_CITIES,
                 mean_std=IMAGENET_MEAN_STD,
                 batch_sampler=None,
                 random_sample_from_each_place=True,
                 val_set_names=['sfxs_val', 'tokyoxs']
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.show_data_stats = show_data_stats
        self.cities = cities
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.random_sample_from_each_place = random_sample_from_each_place
        self.val_set_names = val_set_names
        self.save_hyperparameters() 


        self.train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])


        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)
        ])


        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.shuffle_all
        }
        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers // 2,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False
        }

    def setup(self, stage):
        """Called by Lightning at the beginning of fit/test."""

        if stage == 'fit':
            # 1) Load training dataset (GSVCities)
            self.reload() 

            # 2) Build validation sets
            self.val_datasets = []
            for val_name in self.val_set_names:
                val_name_lower = val_name.lower()
                if 'sfxs_val' in val_name_lower:

                    self.val_datasets.append(
                        SFXSDataset(which_ds='sfxs_val', input_transform=self.valid_transform)
                    )
                elif 'sfxs_test' in val_name_lower:

                    self.val_datasets.append(
                        SFXSDataset(which_ds='sfxs_test', input_transform=self.valid_transform)
                    )
                elif 'tokyo' in val_name_lower:

                    self.val_datasets.append(
                        TokyoXSDataset(input_transform=self.valid_transform)
                    )
                else:
                    raise NotImplementedError(f"Val set {val_name} not recognized")

            if self.show_data_stats:
                self.print_stats()

        elif stage == 'test':
            pass

    def reload(self):
        """Rebuild the training dataset each time we call train_dataloader()."""
        self.train_dataset = GSVCitiesDataset(
            cities=self.cities,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform
        )

    def train_dataloader(self):
        self.reload()  
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dls = []
        for val_dataset in self.val_datasets:
            val_dls.append(DataLoader(val_dataset, **self.valid_loader_config))
        return val_dls

    def print_stats(self):
        """Just prints out dataset stats with PrettyTable."""
        print()
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        # Show # of cities
        table.add_row(["# of cities", f"{len(self.cities)}"])
        # Show # of places & # of images in train set
        table.add_row(["# of places", f'{len(self.train_dataset.places_ids)}'])
        table.add_row(["# of images", f'{self.train_dataset.total_nb_images}'])
        print(table.get_string(title="Training Dataset"))

        print()
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        for i, val_name in enumerate(self.val_set_names):
            table.add_row([f"Validation set {i+1}", f"{val_name}"])
        print(table.get_string(title="Validation Datasets"))

        print()
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(["# of iterations",
                       f"{len(self.train_dataset)//self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))