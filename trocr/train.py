from __future__ import annotations

import os
import argparse
from tabulate import tabulate
from omegaconf import OmegaConf
from metrics import make_compute
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from trocr.dataset import (
    DatasetLoader,
    HTRDataset
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", default="configs/default.yaml", help="YAML config file with training parameters")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config_file)
    print(OmegaConf.to_yaml(cfg))
    
    train_data = []
    val_data = []
    rows = []
    
    for dataset in cfg.data.datasets:
        cls = globals()[OmegaConf.select(cfg, f"data.datasets.{dataset}.loader_class")]
        if dataset == "iam":
            loader: DatasetLoader = cls(OmegaConf.select(cfg, f"data.datasets.{dataset}.path"), validation_set=OmegaConf.select(cfg, f"data.datasets.{dataset}.validation_set"))
        else:
            loader: DatasetLoader = cls(OmegaConf.select(cfg, f"data.datasets.{dataset}.path"))
        
        val = loader.get('val')
        train = loader.get('train')
        test = loader.get('test')
                
        if dataset == "barbados":
            train_data.extend(train + val)
            val_data.extend(test)
            rows.append([OmegaConf.select(cfg, f"data.datasets.{dataset}.name"), len(train + val), len(test)])
        else:     
            train_data.extend(train + val + test)            
            rows.append([OmegaConf.select(cfg, f"data.datasets.{dataset}.name"), len(train + val + test), 0])
            
    rows.append(["Combined", len(train_data), len(val_data)])
    print(tabulate(rows, headers=["Dataset", "Train", "Val"], tablefmt="grid"))

    processor = TrOCRProcessor.from_pretrained(cfg.load_from)
    model = VisionEncoderDecoderModel.from_pretrained(cfg.load_from)
    
    # Required config for generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Generation parameters go on generation_config, NOT config
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.max_length = cfg.model.generation_config.max_length
    model.generation_config.early_stopping = cfg.model.generation_config.early_stopping
    model.generation_config.no_repeat_ngram_size = cfg.model.generation_config.no_repeat_ngram_size
    model.generation_config.length_penalty = cfg.model.generation_config.length_penalty
    model.generation_config.num_beams = cfg.model.generation_config.num_beams
    
    train_dataset = HTRDataset(train_data, processor, max_target_length=cfg.model.generation_config.max_target_length)
    eval_dataset = HTRDataset(val_data, processor, max_target_length=cfg.model.generation_config.max_target_length)
    
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy=cfg.train.save_strategy,
        save_strategy=cfg.train.save_strategy,
        eval_steps=cfg.train.save_steps,
        save_steps=cfg.train.save_steps,
        per_device_train_batch_size=cfg.train.train_batch_size,
        per_device_eval_batch_size=cfg.train.eval_batch_size,
        fp16=cfg.train.fp16,
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.train.num_train_epochs,
        max_epochs=cfg.train.max_epochs,
        learning_rate=cfg.train.learning_rate,
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=cfg.train.load_best_model_at_end,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to=cfg.reporting.report_to,  # change to "wandb" or "tensorboard" if you want logging
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor.image_processor,
        args=training_args,
        compute_metrics=make_compute(processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    
    trainer.train()
    model.save_pretrained(os.path.join(cfg.train.output_dir, "checkpoint-final"))
    processor.save_pretrained(os.path.join(cfg.train.output_dir, "checkpoint-final"))


if __name__ == "__main__":
    main()