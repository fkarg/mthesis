import typer

app = typer.Typer()

@app.command()
def evaluate(
        stats_name: str = "stats.yml",
):
    # run evaluation of models
    pass

@app.command()
def train(
        stats_name: str = "stats.yml",
):
    # model training run
    pass

def main():
    app()

if __name__ == "__main__":
    main()
