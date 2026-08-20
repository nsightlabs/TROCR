from __future__ import annotations

import os
import argparse
from tabulate import tabulate
from omegaconf import OmegaConf
from metrics import make_compute
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

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
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
    
    train_dataset = HTRDataset(train_data, processor, max_target_length=cfg.model.generation_config.max_target_length)
    eval_dataset = HTRDataset(val_data, processor, max_target_length=cfg.model.generation_config.max_target_length)
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=cfg.train.early_stopping_patience),
    ]
    
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
        dataloader_num_workers=cfg.train.dataloader_num_workers  
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor.image_processor,
        args=training_args,
        compute_metrics=make_compute(processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=callbacks
    )
    
    trainer.train()
    model.save_pretrained(os.path.join(cfg.output_dir, "checkpoint-final"))
    processor.save_pretrained(os.path.join(cfg.output_dir, "checkpoint-final"))


if __name__ == "__main__":
    main()