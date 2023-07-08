import os
import typer
import torch
import logging
from tqdm import tqdm

from mthesis.models import JsonformerModel
from mthesis.utils import load_yaml, save_yaml
from mthesis.dataloader import MOFDataset

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

    for model_settings in settings["models"]:
        model_path = model_settings["model_path"]
        model_name = model_settings["model_name"]

        log.info(f"Loading Model [{model_name}]")
        model = JsonformerModel(model_path)
        model.eval()  # set model to eval mode

        log.info(f"Loading Dataset")

        dataset = MOFDataset(settings["dataset_path"])

        progress_bar = tqdm(dataset, file=open(os.devnull, "w"))

        diff = -(len(evaluated) % len(dataset))
        first = True

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

            entry = {
                "paragraph_id": paragraph_id,
                "model_name": model_name,
            }

            entry["answer"] = model(item["text"])
            stats.append(entry)
            save_yaml(stats, stats_path)

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
    # run evaluation of models

    evaluated = frozenset(map(lambda s: (s["paragraph_id"], s["model_name"]), stats))

    # model training run
    raise NotImplementedError



def main():
    app()


if __name__ == "__main__":
    main()
