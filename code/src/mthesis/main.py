import typer
import logging
from tqdm import tqdm

# from mthesis.models import JsonformerModel
from mthesis.utils import load_yaml
from mthesis.dataloader import MOFDataset

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
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
        stats: str = "stats.yml"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    settings = load_yaml(settings)
    stats = load_yaml(stats)
    # run evaluation of models

    model_path = settings["models"][0]["model_path"]

    log.info(f"Loading [{model_path}]")
    model = JsonformerModel(model_path)
    model.eval()  # set model to eval mode

    progress_bar = tqdm(MOFDataset("~/mof_synthesis"), file=open(os.devnull, "w"))

    for item in progress_bar:
        print(str(progress_bar))
        print(item)

    # print(model(x))
    # {'additive': 'water', 'solvent': 'water', 'temperature': 90.0, 'temperature_unit': 'C', 'time': 40.0, 'time_unit': 'h'}


@app.command()
def train(
    settings: str = None,
    stats: str = None,
    device: str = None,
):
    if settings is None:
        settings = "settings.yml"
    if stats is None:
        stats: str = "stats.yml"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    settings = load_yaml(settings)
    stats = load_yaml(stats)

    # model training run
    raise NotImplementedError


def main():
    app()


if __name__ == "__main__":
    main()
