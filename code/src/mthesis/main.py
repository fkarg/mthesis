import os
import sys
import typer
import torch
import logging
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline,
)
from pathlib import Path

from mthesis.models import JsonformerModel
from mthesis.utils import load_yaml, save_yaml
from mthesis.dataloader import MOFDataset, LabeledMOFDataset

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def evaluate(
    settings: str = None,
    stats: str = None,
    device: str = None,
):
    if settings is None:
        settings = "settings.yml"
    if stats is None:
        stats_path = "stats.yml"
    else:
        stats_path = stats
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Loading settings and stats")

    settings = load_yaml(settings)
    stats = load_yaml(stats_path)
    # run evaluation of models

    evaluated = frozenset(map(lambda s: (s["paragraph_id"], s["model_name"]), stats))

    log.info(f"Loading Dataset")
    dataset = MOFDataset(settings["dataset_path"])

    for model_settings in settings["models"]:
        model_path = model_settings["model_path"]
        model_name = model_settings["model_name"]
        model = None

        progress_bar = tqdm(dataset, file=open(os.devnull, "w"))
        progress_bar.set_postfix_str(model_name)

        diff = -(len(evaluated) % len(dataset))
        first = True
        count = 0

        for item in progress_bar:
            log.info(str(progress_bar))
            print(str(progress_bar))
            paragraph_id = item["paragraph_id"]
            if (paragraph_id, model_name) in evaluated:
                log.debug(f"Skipping {paragraph_id}, as it has been processed before.")
                progress_bar.update()
                continue

            if first:
                progress_bar.update(diff)
                first = False
                log.info(f"Loading Model [{model_name}]")
                model = JsonformerModel(**model_settings)
                model.eval()  # set model to eval mode

            count += 1

            entry = {
                "paragraph_id": paragraph_id,
                "model_name": model_name,
            }

            entry["answer"] = model(item["text"])  # forward the dataset text
            stats.append(entry)
            if count >= 20:
                log.info("Saving progress to stats")
                save_yaml(stats, stats_path)
                count = 0
        save_yaml(stats, stats_path)


@app.command()
def test(
    settings: str = None,
    stats: str = None,
    device: str = None,
):
    """ Test information.
    """
    if settings is None:
        settings = "settings.yml"
    if stats is None:
        stats_path = "stats.yml"
    else:
        stats_path = stats
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Loading settings and stats")

    settings = load_yaml(settings)
    stats = load_yaml(stats_path)

    for model_settings in settings["models"]:
        model_name = model_settings["model_name"]
        model_path = model_settings["model_path"]
        log.info(f"Loading Tokenizer [{model_name}]")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        log.info(f"Loading Model [{model_name}] as QA")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
            )
            log.info("Succeeded loading. Attempting forward")

            question = "How many programming languages does BLOOM support?"
            context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
            pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
            result = pipe(question=question, context=context)
            log.info(f"QA Result: {result}")

        except Exception as e:
            log.error(f"{e} during loading of {model_name}.")



@app.command()
def train(
    settings: str = None,
    stats: str = None,
    device: str = None,
):
    if settings is None:
        settings = "settings.yml"
    if stats is None:
        stats_path = "stats.yml"
    else:
        stats_path = stats
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Loading settings and stats")

    settings = load_yaml(settings)
    stats = load_yaml(stats_path)

    try:
        label_cols = {
            param: settings["extract_config"][param]["dataset_cols"][0]
            for param in settings["extract_config"]
        }
    except KeyError as e:
        log.critical(
            f"KeyError: Invalid SETTINGSFILE. Missing global key `extract_config` or param-key `dataset_cols`."
        )
        sys.exit(1)

    # TODO: FOR loop
    model_settings = settings["models"][0]

    model_path = model_settings["model_path"]
    model_name = model_settings["model_name"]

    log.info(f"Loading Tokenizer [{model_name}]")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    log.info("Loading Dataset")
    dataset = LabeledMOFDatasetTokens(tokenizer, True, settings["dataset_path"], label_cols, from_csv="mof_dataset_labeled.csv")

    # dataset.to_csv("mof_dataset_labeled.csv")

    generator = torch.Generator().manual_seed(42)
    train_ds, eval_ds = torch.utils.data.random_split(dataset, [0.05, 0.95], generator=generator)

    log.info(f"Loading Model [{model_name}]")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
    )
    # model = JsonformerHFModel(model_path)

    OUTPUT_MODEL_PATH = f"/home/kit/iti/lz5921/checkpoints/{model_name}/"
    Path(OUTPUT_MODEL_PATH).mkdir(parents=True, exist_ok=True)


    # DataCollator: https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling
    # mlm (bool, optional, defaults to True) — Whether or not to use masked
    # language modeling. If set to False, the labels are the same as the inputs
    # with the padding tokens ignored (by setting them to -100). Otherwise, the
    # labels are -100 for non-masked tokens and the value to predict for the
    # masked token.
    data_collator = DataCollatorForLanguageModeling(tokenizer)  # mlm=False

    ## Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_PATH,
        overwrite_output_dir=True,
        num_train_epochs=1,
        weight_decay=0.005,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        logging_dir="./logs",
        logging_steps=1,
        logging_strategy="epoch",
        optim="adamw_torch",
        learning_rate=1e-4,
        evaluation_strategy="epoch",  # alternatively: "no",
        fp16=True,
        save_strategy="steps",
        save_steps=50,
    )

    ## Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # automatically restores model, epoch, step, LR schedulers, etc from checkpoint
    # model.config.use_cache = False
    model.train()  # put model in training mode
    trainer.train()
    model.save_pretrained(OUTPUT_MODEL_PATH)


def main():
    app()


if __name__ == "__main__":
    main()
