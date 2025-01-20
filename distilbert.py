from transformers import (
    logging, set_seed, DataCollatorWithPadding, Trainer, TrainerCallback, TrainerState,TrainerControl,
    TrainingArguments, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import DatasetDict, load_dataset
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score
import wandb
import torch
import os
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()

class Args:
    def __init__(self):
        self.model_name_or_path = "distilbert/distilbert-base-uncased"
        self.max_seq_length = 512
        self.learning_rate = 1e-4
        self.num_epochs = 8
        self.per_gpu_batch_size = 16
        self.seed = 42
        self.output_dir = "./fine-tune-results"
        self.gradient_accumulation_steps = 16
        self.freeze = False
        self.lr_scheduler_type = "cosine"
        self.num_warmup_steps = 10
        self.weight_decay = 0.01
        self.push_to_hub = False
        self.model_hub_name = "readability-assesments"
        self.warmup_ratio = 0.1
        self.dropout_rate = 0.3
        self.max_grad_norm = 1.0
        self.layer_lr_decay = 0.95


def calculate_metrics(predictions, labels):
    mse = mean_squared_error(labels, predictions)
    rmse = root_mean_squared_error(labels, predictions, squared=False)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    smape = np.mean(2 * np.abs(predictions - labels) / (np.abs(labels) + np.abs(predictions)) * 100)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metrics(predictions, labels)

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_copy
class WandbPredictionCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero and "eval_predictions" in kwargs:
            predictions = kwargs["eval_predictions"].predictions
            labels = kwargs["eval_predictions"].label_ids

            if predictions.shape[-1] > 1:
                predictions = np.argmax(predictions, axis=-1)

            metrics = calculate_metrics(predictions, labels)
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, Any], **kwargs):
        if state.is_world_process_zero:
            wandb.log({
                "predict/mse": metrics["test_mse"],
                "predict/rmse": metrics["test_rmse"],
                "predict/mae": metrics["test_mae"],
                "predict/r2": metrics["test_r2"],
                "predict/smape": metrics["test_smape"],
                "global_step": state.global_step
            })


def main():
    args = Args()
    set_seed(args.seed)


    wandb.login(key="ded9e04f43e8c7f6ed6f4a2e262c502cde7562cf")
    wandb.init(
        project="CodeSage",
        name=args.model_name_or_path,
        settings=wandb.Settings(code_dir="."),
    )

    train_dataset = load_dataset('csv', data_files="/Users/vera.kudrevskaia/Desktop/Jetbrains/Model_3/data_tune_train.csv", split="train")
    test_dataset = load_dataset('csv', data_files="/Users/vera.kudrevskaia/Desktop/Jetbrains/Model_3/data_tune_test.csv", split="train")

    train_validation_split = train_dataset.train_test_split(test_size=0.15)
    train_validation = DatasetDict(
        {
            "train": train_validation_split["train"],
            "valid": train_validation_split["test"]
        }
    )

    train_test_validation = DatasetDict(
        {
            "train": train_validation["train"],
            "valid": train_validation["valid"],
            "test": test_dataset
        }
    )

    print("Train dataset size:", len(train_test_validation["train"]))
    print("Validation dataset size:", len(train_test_validation["valid"]))
    print("Test dataset size:", len(train_test_validation["test"]))

    print("Loading config, model, and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config.problem_type = "regression"
    config.num_labels = 1
    config.classifier_dropout = None
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config, trust_remote_code=True
    )

    if args.freeze:
        print("Freezing model parameters")
        for param in model.encoder.parameters():
            param.requires_grad = False

    model.classifier.dropout = torch.nn.Dropout(args.dropout_rate)

    def convert_examples_to_features(example):
        inputs = tokenizer(example["file_content"], truncation=True, max_length=args.max_seq_length)
        label = example["score"]
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "label": label,
        }

    tokenized_datasets = train_test_validation.map(
        convert_examples_to_features,
        batched=True,
        remove_columns=["score", "file_content"],
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=args.per_gpu_batch_size,
        per_device_eval_batch_size=args.per_gpu_batch_size,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        save_total_limit=2,
        metric_for_best_model='rmse',
        load_best_model_at_end=True,
        run_name="readability-java",
        report_to="wandb",
        max_grad_norm=args.max_grad_norm,
    )

    # Implement layer-wise learning rates
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []

    model_layers = [(name, module) for name, module in model.named_children() if list(module.parameters())]
    num_layers = len(model_layers)

    for idx, (name, layer) in enumerate(model_layers):
        layer_params = list(layer.named_parameters())
        layer_lr = args.learning_rate * (args.layer_lr_decay ** (num_layers - idx - 1))

        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in layer_params if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
                "lr": layer_lr,
            },
            {
                "params": [p for n, p in layer_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": layer_lr,
            },
        ])

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    num_training_steps = len(tokenized_datasets["train"]) * args.num_epochs // (args.per_gpu_batch_size * args.gradient_accumulation_steps)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    class EarlyStoppingCallback(TrainerCallback):
        def __init__(self, early_stopping_patience=3, early_stopping_threshold=0.0):
            self.early_stopping_patience = early_stopping_patience
            self.early_stopping_threshold = early_stopping_threshold
            self.best_metric = float('inf')
            self.no_improve_count = 0

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            eval_metric = metrics.get("eval_rmse")
            if eval_metric is not None:
                if eval_metric < self.best_metric - self.early_stopping_threshold:
                    self.best_metric = eval_metric
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1

                if self.no_improve_count >= self.early_stopping_patience:
                    print(f"Early stopping triggered. Best metric: {self.best_metric:.4f}")
                    control.should_training_stop = True

            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[WandbPredictionCallback, EarlyStoppingCallback(early_stopping_patience=3)],
        optimizers=(optimizer, lr_scheduler),
    )


    print("Training...")
    trainer.add_callback(CustomCallback(trainer))
    trainer.train()

    result = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    result_str = (f"Evaluation results on the test set: rmse :{result['rmse']}, mae :{result['mae']}, r2:{result['r2']}")
    print(result_str)

    with open(f"{args.output_dir}/seed{args.seed}_result.txt", "w") as fw:
        fw.write(result_str + "\n")

    if args.push_to_hub:
        model.push_to_hub(args.model_hub_name)


if __name__ == "__main__":
    main()
