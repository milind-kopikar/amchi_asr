import os
import yaml
from transformers import AutoProcessor, TrainingArguments, Trainer
from datasets import load_from_disk
import torch

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    model_name = config['model']['base_model']
    adapter_lang = config['model']['adapter_lang']
    freeze_base = config['model']['freeze_base_model']
    output_dir = config['project']['output_dir']
    dataset_dir = os.path.join(output_dir, 'hf_dataset')

    # Load dataset
    dataset = load_from_disk(dataset_dir)

    # Load processor and model
    from transformers import MmsForCTC
    processor = AutoProcessor.from_pretrained(model_name, target_lang=adapter_lang)
    model = MmsForCTC.from_pretrained(model_name)

    # Adapter setup
    model.init_adapter_layers()
    model.load_adapter(adapter_lang)
    if freeze_base:
        model.freeze_base_model()

    # Training args
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'checkpoints'),
        per_device_train_batch_size=config['training']['batch_size'],
        evaluation_strategy="epoch",
        num_train_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        save_total_limit=2,
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, 'logs'),
        report_to="none"
    )

    def preprocess(batch):
        audio = batch["audio"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        with processor.as_target_processor():
            labels = processor(batch["transcript"]).input_ids
        batch["input_values"] = inputs.input_values[0]
        batch["labels"] = torch.tensor(labels)
        return batch

    dataset = dataset.map(preprocess)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor,
    )
    trainer.train()
    model.save_pretrained(os.path.join(output_dir, 'final_model'))
    processor.save_pretrained(os.path.join(output_dir, 'final_model'))
    print(f"Model and processor saved to {os.path.join(output_dir, 'final_model')}")

if __name__ == "__main__":
    main()
