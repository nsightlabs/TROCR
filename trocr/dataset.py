from __future__ import annotations

from typing import List, Tuple
import torch
import pandas as pd
from abc import ABC, abstractmethod
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor

class HTRDataset(Dataset):
    def __init__(self, data: List, processor: TrOCRProcessor, max_target_length: int = 128):
        self.data = data
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, text = self.data[idx]
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids

        # Replace padding token id with -100 so it's ignored in loss computation
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}    

class DatasetLoader(ABC):
    def __init__(self):
        self.train_data = None
        self.val_data = None
        self.test_data = None

    @abstractmethod
    def _prepare(self):
        """Prepare dataset."""
        pass

    def get(self, split: str):
        if split == "train":
            return self.train_data
        elif split == "val":
            return self.val_data
        elif split == "test":
            return self.test_data
        else:
            raise ValueError(f"Unknown split: {split}. Expected: train, val, test.")
        
class BarbadosDatasetLoader(DatasetLoader):
    def __init__(self, path: str):
        self.path = Path(path)
        self._prepare()

    def _prepare(self):
        df = pd.read_csv(self.path /  "Train.csv")
        tmp_df, test_df = train_test_split(df, test_size=0.1, random_state=0)
        train_df, val_df = train_test_split(tmp_df, test_size=0.1, random_state=0)

        transcriptions = df.set_index('ID')['Target'].to_dict()

        self.train_data = [(str(self.path / "images/images" / f"{row.ID}.jpg"), transcriptions.get(row.ID)) for row in train_df.itertuples()]
        self.val_data = [(str(self.path / "images/images" / f"{row.ID}.jpg"), transcriptions.get(row.ID)) for row in val_df.itertuples()]
        self.test_data = [(str(self.path / "images/images" / f"{row.ID}.jpg"), transcriptions.get(row.ID)) for row in test_df.itertuples()]
        self.transcriptions = transcriptions

class BenthamDatasetLoader(DatasetLoader):
    def __init__(self, path: str):
        self.path = Path(path)
        self._prepare()

    def _prepare(self):
        train_file = self.path / "Partitions" / "TrainLines.lst"
        val_file = self.path / "Partitions" / "ValidationLines.lst"
        test_file = self.path / "Partitions" / "TestLines.lst"        

        train_filenames = train_file.read_text().strip().split("\n")
        val_filenames = val_file.read_text().strip().split("\n")
        test_filenames = test_file.read_text().strip().split("\n")

        transcriptions = {}
        all_filenames = train_filenames + val_filenames + test_filenames
        for filename in tqdm(all_filenames, desc='loading transcriptions'):
            transcription_file = self.path / "Transcriptions" / f"{filename}.txt"
            transcriptions[filename] = transcription_file.read_text().strip()

        self.train_data = [(str(self.path / "Images/Lines" / f"{filename}.png"), transcriptions.get(filename)) for filename in train_filenames]
        self.val_data = [(str(self.path / "Images/Lines" / f"{filename}.png"), transcriptions.get(filename)) for filename in val_filenames]
        self.test_data = [(str(self.path / "Images/Lines" / f"{filename}.png"), transcriptions.get(filename)) for filename in test_filenames]
        self.transcriptions = transcriptions
        
# class WashingtonDatasetLoader(DatasetLoader): 
#     def __init__(self, path: str, cv_fold: str = "cv1"):
#         self.path = Path(path)
#         self.cv_fold = cv_fold
#         self._prepare()
 
#     def _prepare(self):
#         sets_dir = self.path / "sets" / self.cv_fold
#         train_file = sets_dir / "train.txt"
#         val_file = sets_dir / "valid.txt"
#         test_file = sets_dir / "test.txt"
 
#         train_ids = train_file.read_text().strip().split("\n")
#         val_ids = val_file.read_text().strip().split("\n")
#         test_ids = test_file.read_text().strip().split("\n")
 
#         transcriptions = self._load_transcriptions()
 
#         images_dir = self.path / "data" / "line_images_normalized"
#         self.train_data = [(str(images_dir / f"{line_id}.png"), transcriptions.get(line_id)) for line_id in train_ids]
#         self.val_data = [(str(images_dir / f"{line_id}.png"), transcriptions.get(line_id)) for line_id in val_ids]
#         self.test_data = [(str(images_dir / f"{line_id}.png"), transcriptions.get(line_id)) for line_id in test_ids]
#         self.transcriptions = transcriptions
 
#     def _load_transcriptions(self):
#         transcription_file = self.path / "ground_truth" / "transcription.txt"
#         transcriptions = {}
#         lines = transcription_file.read_text().strip().split("\n")
#         for line in tqdm(lines, desc="loading transcriptions"):
#             line_id, spelling_field = line.strip().split(" ")
#             if not encoded:
#                 continue
#             transcriptions[line_id] = decode_line(encoded)
#         return transcriptions
        
class RimesDatasetLoader(DatasetLoader):
    def __init__(self, path: str):
        self.path = Path(path)
        self._prepare()

    def _prepare(self):
        train_file = self.path / "Sets" / "TrainLines.txt"
        val_file = self.path / "Sets" / "ValidationLines.txt"
        test_file = self.path / "Sets" / "TestLines.txt"        

        train_filenames = train_file.read_text().strip().split("\n")
        val_filenames = val_file.read_text().strip().split("\n")
        test_filenames = test_file.read_text().strip().split("\n")

        transcriptions = {}
        all_filenames = train_filenames + val_filenames + test_filenames
        for filename in tqdm(all_filenames, desc='loading transcriptions'):
            transcription_file = self.path / "Transcriptions" / f"{filename}.txt"
            transcriptions[filename] = transcription_file.read_text().strip()

        self.train_data = [(str(self.path / "Images" / f"{filename}.jpg"), transcriptions.get(filename)) for filename in train_filenames]
        self.val_data = [(str(self.path / "Images" / f"{filename}.jpg"), transcriptions.get(filename)) for filename in val_filenames]
        self.test_data = [(str(self.path / "Images" / f"{filename}.jpg"), transcriptions.get(filename)) for filename in test_filenames]
        self.transcriptions = transcriptions
        
class SaintGallDatasetLoader(DatasetLoader):
    def __init__(self, path: str, normalized_images: bool = False):
        self.path = Path(path)
        self.normalized_images = normalized_images
        self._prepare()

    def _prepare(self):
        train_file = self.path / "sets" / "train.txt"
        val_file = self.path / "sets" / "valid.txt"
        test_file = self.path / "sets" / "test.txt"        

        train_base_filenames = train_file.read_text().strip().split("\n")
        val_base_filenames = val_file.read_text().strip().split("\n")
        test_base_filenames = test_file.read_text().strip().split("\n")

        images_folder_name = ("line_images" if self.normalized_images else "line_images_normalized" )
        images_folder = self.path / f"data/{images_folder_name}"
        train_filenames = [image_path.stem for base_filename in train_base_filenames for image_path in images_folder.rglob(f"{base_filename}*")]
        val_filenames = [image_path.stem for base_filename in val_base_filenames for image_path in images_folder.rglob(f"{base_filename}*")]
        test_filenames = [image_path.stem for base_filename in test_base_filenames for image_path in images_folder.rglob(f"{base_filename}*")]

        transcriptions = {}
        transcription_file = self.path / "ground_truth/transcription.txt"
        text = transcription_file.read_text().strip()
        lines = text.split('\n')
        for line in tqdm(lines, desc="Loading transcriptions"):
            line = line.strip()
            if not line:
                continue
            filename, spelling_field, _ = line.split(' ', 2)
            words = spelling_field.split('|')
            words = [''.join('&' if tok == 'et' else '.' if tok == 'pt' else tok for tok in word.split('-')) for word in words]
            transcription = ' '.join(words)   
            transcriptions[filename] = transcription        
        
        self.train_data = [(str(images_folder / f"{filename}.png"), transcriptions.get(filename)) for filename in train_filenames]
        self.val_data = [(str(images_folder / f"{filename}.png"), transcriptions.get(filename)) for filename in val_filenames]
        self.test_data = [(str(images_folder / f"{filename}.png"), transcriptions.get(filename)) for filename in test_filenames]
        self.transcriptions = transcriptions
        

class IAMDatasetLoader(DatasetLoader):
    def __init__(self, path: str, validation_set: Tuple[int] = (1,)):
        self.path = Path(path)
        self.validation_set = validation_set
        self._prepare()

    def _get_image_folder(self, filename: str) -> str:
        part1, part2, _ = filename.split('-')
        return f"{part1}/{part1}-{part2}"

    def _prepare(self):
        train_file = self.path / "trainset.txt"
        val_files = [self.path / f"validationset{i}.txt" for i in self.validation_set]
        test_file = self.path / "testset.txt"        

        train_filenames = train_file.read_text().strip().split("\n")
        val_filenames = [filename for val_file in val_files for filename in val_file.read_text().strip().split("\n")]
        test_filenames = test_file.read_text().strip().split("\n")

        transcriptions = {}
        transcription_file = self.path / "lines.txt"
        text = transcription_file.read_text().strip()
        lines = text.split('\n')
        for line in tqdm(lines, desc='Loading transcriptions'):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=8)
            words = parts[-1].split('|')
            transcription = ' '.join(words)
            filename = parts[0]
            transcriptions[filename] = transcription

        self.train_data = [(str(self.path / f"{self._get_image_folder(filename)}" / f"{filename}.png"), transcriptions.get(filename)) for filename in train_filenames]
        self.val_data = [(str(self.path / f"{self._get_image_folder(filename)}" / f"{filename}.png"), transcriptions.get(filename)) for filename in val_filenames]
        self.test_data = [(str(self.path / f"{self._get_image_folder(filename)}" / f"{filename}.png"), transcriptions.get(filename)) for filename in test_filenames]
        self.transcriptions = transcriptions