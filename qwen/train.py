from unsloth import FastVisionModel 
from unsloth.trainer import UnslothVisionDataCollator
import os
import torch
import argparse
import pandas as pd
from PIL import Image
from omegaconf import OmegaConf
from jiwer import process_characters
from trl import SFTTrainer, SFTConfig

def load_model(model_name):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 16,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 16,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )
    return model, tokenizer


def prepare_dataset(dataset_csv, IMAGE_DIR):
    df = pd.read_csv(dataset_csv)
    references = df['Target'].tolist()
    hypotheses = df['Prediction'].tolist()
    output = process_characters(reference=references, hypothesis=hypotheses)
    
    dataset = []
    for ID, ref, hyp, alignments in zip(df['ID'].tolist(), output.references, output.hypotheses, output.alignments):
        edits = {}
        for alignment in alignments:
            if alignment.type == "substitute":
                ref_start_idx, ref_end_idx = alignment.ref_start_idx, alignment.ref_end_idx
                hyp_start_idx, hyp_end_idx = alignment.hyp_start_idx, alignment.hyp_end_idx
                for hyp_idx, ref_idx in zip(range(hyp_start_idx, hyp_end_idx), range(ref_start_idx, ref_end_idx)):
                    edits.setdefault('substitute', {})[hyp_idx] = f"{hyp[hyp_idx]}->{ref[ref_idx]}"
            
        if 'substitute' in edits:
            OCR_TEXT = ''.join(hyp)        
            PROMPT = ("You are provided the OCR text from TROCR model\n"
                        "Identify the characters that should be substituted and which characters should replace them.\n"
                        f"OCR TEXT:\n\t{OCR_TEXT}")
            IMAGE = Image.open(os.path.join(IMAGE_DIR, f"{ID}.jpg"))
                    
            SUBSTITUTES = '\n'.join([f"\t{k}: {v}" for k, v in edits.get('substitute').items()])
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image", "image": IMAGE}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": SUBSTITUTES}
                ]}
            ]
            dataset.append({"messages": messages})
    return dataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", default="configs/default.yaml", help="YAML config file with training parameters")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config_file)
    print(OmegaConf.to_yaml(cfg))
    
    dataset = {
        'train': prepare_dataset(cfg.train_csv, cfg.IMAGE_DIR),
        'test': prepare_dataset(cfg.test_csv, cfg.IMAGE_DIR)
    }
    
    model, tokenizer = load_model(cfg.model_name)
    FastVisionModel.for_training(model) # Enable for training!
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
        train_dataset = dataset['train'],
        eval_dataset = dataset['test'],
        args = SFTConfig(
            per_device_train_batch_size = cfg.per_device_train_batch_size,
            per_device_eval_batch_size = cfg.per_device_eval_batch_size,
            gradient_accumulation_steps = cfg.gradient_accumulation_steps,
            warmup_steps = cfg.warmup_steps,
            max_steps = -1,
            num_train_epochs = cfg.num_train_epochs, # Set this instead of max_steps for full training runs
            eval_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate = 2e-4,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",     # For Weights and Biases

            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = 2048,
        ),
    )
    trainer_stats = trainer.train()
    print(trainer_stats)
    
if __name__ == "__main__":
    main()