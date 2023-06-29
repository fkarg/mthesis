import lightning.pytorch as pl
from jsonformer import Jsonformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

class JsonformerModel(pl.LightningModule):
    """Provides interface for a LLM wrapped with JsonFormer for material Science applications.

    PARAMS:
        model_path [str]: path to both model and tokenizer in pytorch / huggingface format.
        load_params [bool]: if the model should load parameters. if they will be restored from a checkpoint anyways.
    """

    def __init__(self, model_path: str | None = None, load_params: bool = True, checkpoint_path: str = None):

        if checkpoint_path:
            return JsonformerModel.load_from_checkpoint(checkpoint_path)

        super().__init__()
        if load or model_path is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        else:
            self.model = None
            self.tokenizer = None

        self.prompt = "Generate the information of used parameters for the reaction based on the following schema:"
        self.schema = {
            "type": "object",
            "properties": {
                "additive": {"type": "string"},
                "solvent": {"type": "string"},
                "temperature": {"type": "number"},
                "time": {"type": "number"},
            },
        }

    def forward(self, text):
        # TODO: ensure that the prompt is added somewhere before?
        # input = text + "\n" + self.prompt

        generated = Jsonformer(self.model, self.tokenizer, self.schema, text)()

        return generated

    def training_step(self):
        pass
