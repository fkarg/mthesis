import lightning.pytorch as pl
import torch
from jsonformer import Jsonformer
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)


class JsonformerHFModel(PreTrainedModel):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)


class JsonformerModel(pl.LightningModule):
    """Provides interface for a LLM wrapped with JsonFormer for material Science applications.

    PARAMS:
        model_path [str]: path to both model and tokenizer in pytorch / huggingface format.
        load_params [bool]: if the model should load parameters. if they will be restored from a checkpoint anyways.
    """

    def __init__(
        self,
        model_path: str | None = None,
        load_params: bool = True,
        checkpoint_path: str = None,
    ):
        if checkpoint_path:
            return JsonformerModel.load_from_checkpoint(checkpoint_path)

        super().__init__()
        if load_params and model_path is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                # torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        else:
            self.model = None
            self.tokenizer = None
            log.warning(
                "JsonformerModel is initialized without underlying generative model or tokenizer."
            )

        # self.prompt = "Generate the information of used parameters for the reaction based on the following schema:"
        self.schema = {
            "type": "object",
            "properties": {
                "additive": {"type": "string"},
                "solvent": {"type": "string"},
                "temperature": {"type": "number"},
                "temperature_unit": {"type": "string"},
                "time": {"type": "number"},
                "time_unit": {"type": "string"},
            },
        }

    def forward(self, text: str) -> dict:
        # https://github.com/1rgs/jsonformer/blob/main/jsonformer/main.py#L240C13-L240C13
        return Jsonformer(self.model, self.tokenizer, self.schema, text)()

    def training_step(self):
        raise NotImplementedError
