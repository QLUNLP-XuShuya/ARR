from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from dataset.data_helper import create_datasets
import torch


class DataModule(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_alignment = getattr(args, 'use_alignment', True)
        self.use_cross_attention = getattr(args, 'use_cross_attention', True)
        self.use_cot = getattr(args, 'use_cot', True)

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        train_dataset, dev_dataset, test_dataset = create_datasets(self.args)
        self.dataset = {
            "train": train_dataset, "validation": dev_dataset, "test": test_dataset
        }

        if getattr(self.args, 'local_rank', 0) == 0:
            print("Dataset loaded successfully:")
            print(f"  - Train samples: {len(train_dataset)}")
            print(f"  - Validation samples: {len(dev_dataset)}")
            print(f"  - Test samples: {len(test_dataset)}")
            if self.use_alignment:
                print("  - Alignment data: enabled")
            if self.use_cross_attention:
                print("  - Cross-attention finding labels: enabled")
            if self.use_cot:
                print("  - CoT findings/impression splitting: enabled")

    def collate_fn(self, batch):
        """
        DDP-safe custom collate. Keeps images / texts as Python lists
        (since Swin processes them per-sample) and stacks tensorized fields.
        """
        out = {
            'image': [item['image'] for item in batch],
            'input_text': [item['input_text'] for item in batch],
            'id': [item['id'] for item in batch],
        }

        if self.use_alignment:
            out['extracted_phrases'] = [item.get('extracted_phrases', []) for item in batch]
            out['alignment_data'] = [item.get('alignment_data', None) for item in batch]

        if self.use_cross_attention and 'finding_labels' in batch[0]:
            out['finding_labels'] = torch.stack(
                [item['finding_labels'] for item in batch], dim=0
            )

        if self.use_cot:
            out['findings_text'] = [item.get('findings_text', item['input_text']) for item in batch]
            out['impression_text'] = [item.get('impression_text', item['input_text']) for item in batch]

        return out

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.args.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=self.args.num_workers,
            prefetch_factor=self.args.prefetch_factor,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.args.val_batch_size,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args.num_workers,
            prefetch_factor=self.args.prefetch_factor,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.args.test_batch_size,
            drop_last=False,
            pin_memory=False,
            num_workers=self.args.num_workers,
            prefetch_factor=self.args.prefetch_factor,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

