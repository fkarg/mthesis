import typer

from mthesis.models import JsonformerModel
from mthesis.utils import load_yaml

app = typer.Typer()


@app.command()
def evaluate(
    settings: str = None,
    stats: str = None,
):
    if settings is None:
        settings = "settings.yml"
    if stats is None:
        stats: str = "stats.yml",

    settings = load_yaml(settings)
    stats = load_yaml(stats)
    # run evaluation of models

    x = "Crystals were grown from a solution of Zn(NO3)2·4H2O (885 mg, 3.39 mmol) and 1 (100 mg, 0.282 mmol) in DEF (11.3 mL), and additional water (1.04 g, 57.63 mmol) kept at 90 °C for 40 h, resulting in 196 mg (73%) colourless crystals. Anal. Calcd. for Zn12C150H166O56N12; C, 47.2; H, 4.38; N, 4.40 (DEF 31.8). Found: C, 48.6; H, 5.89, N, 6.26%. The difference is explained by the presence of nine free, uncoordinated DEF molecules, which lead to a theoretical total N of 6.22 (DEF 44.9). IR (KBr): 3441 (O–H, br), 3088, 2979, 2939, 2878, 1638, 1586, 1364, 1302, 1264, 1213, 1104, 777, 723."

    model = JsonformerModel(settings["models"][0]["model_path"])
    model.eval()
    print(model(x))


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
