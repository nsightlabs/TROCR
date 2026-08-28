from __future__ import annotations

import os
import torch
import argparse
import pandas as pd
import torch.multiprocessing as mp
from tqdm import tqdm
from uuid import uuid4
from omegaconf import OmegaConf
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)
from dataset import (
    DatasetLoader,
    BarbadosDatasetLoader
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", default="configs/default.yaml", help="YAML config file with training parameters")
    return p.parse_args()


def main(rank: int, world_size: int, cfg: OmegaConf.load, submission_list: list):    
    dataset = "barbados"
    cls = globals()[OmegaConf.select(cfg, f"data.datasets.{dataset}.loader_class")]
    loader: DatasetLoader = cls(OmegaConf.select(cfg, f"data.datasets.{dataset}.path"))
    predict_data = loader.get('predict')
    sample_data = predict_data[rank::world_size]

    device = f"cuda:{rank}"
    processor = TrOCRProcessor.from_pretrained(cfg.load_from)
    model = VisionEncoderDecoderModel.from_pretrained(cfg.load_from).to(device)
    model = model.eval()
    
    for item in tqdm(sample_data, desc=f'Inference: [{rank}]'):
        image_path, ID = item
        images = [Image.open(image_path).convert("RGB")]       
        pixel_values = processor(images, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        submission_list.append({'ID': ID, 'Target': generated_text}) 


if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load(args.config_file)
    print(OmegaConf.to_yaml(cfg))
    
    world_size = torch.cuda.device_count()    
    manager = mp.Manager()
    submission_list = manager.list()
 
    if world_size > 1:
        mp.spawn(
            main,
            args=(world_size, cfg, submission_list),
            nprocs=world_size,
            join=True,
        )
    else:
        main(0, world_size, cfg, submission_list)
 
    results = list(submission_list)
    df = pd.DataFrame(results)
    os.makedirs(cfg.output_dir, exist_ok=True)
    output_csv = os.path.join(cfg.output_dir, "".join(str(uuid4()).split('-')) + '.csv')
    df.to_csv(output_csv, index=False)    
    print(f"Saved {len(df)} predictions to {output_csv}")