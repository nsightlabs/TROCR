from __future__ import annotations

import os
import torch
import argparse
from tabulate import tabulate
from omegaconf import OmegaConf
from metrics import make_compute
from data_aug import build_data_aug, OptForDataAugment, DataAugment
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback
)
from dataset import (
    DatasetLoader,
    HTRDataset,
    IAMDatasetLoader,
    RimesDatasetLoader,
    BarbadosDatasetLoader,
    BenthamDatasetLoader,
    SaintGallDatasetLoader
)

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    frozen_params = all_params - trainable_params
    print(
        f"trainable params: {trainable_params:,} || "
        f"frozen params: {frozen_params:,} || "
        f"total params: {all_params:,} || "
        f"trainable%: {100 * trainable_params / all_params:.4f}%"
    )
    

def make_collator(model):
    def collate_fn(features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        # Precompute decoder_input_ids so it survives label_smoothing's popping of `labels`
        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=labels)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }
    return collate_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", default="configs/default.yaml", help="YAML config file with training parameters")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config_file)
    print(OmegaConf.to_yaml(cfg))
    
    data = {'train':[], 'val':[]}
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
            data['train'].extend(train + val)
            data['val'].extend(test)
            rows.append([OmegaConf.select(cfg, f"data.datasets.{dataset}.name"), len(train + val), len(test)])
        else:     
            data['train'].extend(train + val + test)            
            rows.append([OmegaConf.select(cfg, f"data.datasets.{dataset}.name"), len(train + val + test), 0])
            
    rows.append(["Combined", len(data['train']), len(data['val'])])
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
    
    if cfg.train.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    
    use_lora = "lora" in cfg.train
    if use_lora:
        lora_config = LoraConfig(
            r=cfg.train.lora.r,                         
            lora_alpha=cfg.train.lora.alpha,               
            lora_dropout=cfg.train.lora.dropout,
            bias=cfg.train.lora.bias,
            target_modules=cfg.train.lora.target_modules,  
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        _original_save_pretrained = model.save_pretrained
        def _patched_save_pretrained(*args, **kwargs):
            kwargs.setdefault("save_embedding_layers", False)
            return _original_save_pretrained(*args, **kwargs)
        model.save_pretrained = _patched_save_pretrained
        
    print_trainable_parameters(model)
    
    dataset = {}
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=cfg.train.early_stopping_patience),
    ]
    
    input_size = cfg.train.preprocess.input_size        
    if isinstance(input_size, list):
        if len(input_size) == 1:
            input_size = (input_size[0], input_size[0])
        else:
            input_size = tuple(input_size)
    elif isinstance(input_size, int):
        input_size = (input_size, input_size)
    
    for split, split_data in data.items():        
        if cfg.train.preprocess.type == 'DA2':            
            tfm = build_data_aug(input_size, mode=split)     
        elif cfg.train.preprocess.type == 'RandAugment':
            opt = OptForDataAugment(eval= (split != 'train'), isrand_aug=True, imgW=input_size[1], imgH=input_size[0], intact_prob=0.5, augs_num=3, augs_mag=None)
            tfm = DataAugment(opt)  
        elif cfg.train.preprocess.type is None: 
            tfm = None
        else:
            raise Exception('Undeined image preprocess method.')   
        dataset[split] = HTRDataset(split_data, processor, transform=tfm, max_target_length=cfg.model.generation_config.max_target_length)
    
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy=cfg.train.save_strategy,
        save_strategy=cfg.train.save_strategy,
        logging_strategy=cfg.train.save_strategy,
        eval_steps=cfg.train.save_steps,
        save_steps=cfg.train.save_steps,
        logging_steps=cfg.train.save_steps,
        per_device_train_batch_size=cfg.train.train_batch_size,
        per_device_eval_batch_size=cfg.train.eval_batch_size,
        fp16=cfg.train.fp16,
        output_dir=cfg.output_dir,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        num_train_epochs=cfg.train.num_train_epochs,
        max_steps=cfg.train.max_steps,
        learning_rate=cfg.train.learning_rate,
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=cfg.train.load_best_model_at_end,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to=cfg.reporting.report_to,
        seed=cfg.train.seed,   
        optim=cfg.train.optimizer,  
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        label_smoothing_factor=cfg.train.label_smoothing_factor
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=make_compute(processor),
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        data_collator=make_collator(model),
        callbacks=callbacks
    )
    
    trainer.train()
    model.save_pretrained(os.path.join(cfg.output_dir, "checkpoint-final"))
    processor.save_pretrained(os.path.join(cfg.output_dir, "checkpoint-final"))


if __name__ == "__main__":
    main()