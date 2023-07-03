import typer
import logging

from mthesis.models import JsonformerModel
from mthesis.utils import load_yaml

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
):
    if settings is None:
        settings = "settings.yml"
    if stats is None:
        stats: str = "stats.yml"

    settings = load_yaml(settings)
    stats = load_yaml(stats)
    # run evaluation of models

    log.info(f"Loading [{settings['models'][0]['model_name']}]")

    model = JsonformerModel(settings["models"][0]["model_path"])
    # model.eval()
    # print(model(x))
    # {'additive': 'water', 'solvent': 'water', 'temperature': 90.0, 'temperature_unit': 'C', 'time': 40.0, 'time_unit': 'h'}


@app.command()
def train(
    stats: str = None,
):
    # model training run
    raise NotImplementedError


def main():
    app()


if __name__ == "__main__":
    main()
