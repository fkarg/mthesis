from pl_bolts.callbacks import ORTCallback
from pytorch_lightning import Trainer, seed_everything
from lightning_transformers.plugins.checkpoint import HFSaveCheckpoint
from torch.optim import AdamW

from mthesis.models import JsonformerModel

seed_everything(42, workers=True)

model = JsonformerModel("falcon")
optimizer = AdamW(get_grouped_params(model), lr=5e-4)
# Epochs: 3-5
# Training examples: 100, 300, 500
trainer = Trainer(
    # devices="auto", accelerator="auto", deterministic=False,  # all default
    # precision="16-mixed",
    callbacks=ORTCallback(),
    max_epochs=1,
    accumulate_grad_batches=4,
    fast_dev_run=True,
    # plugins=HFSaveCheckpoint(model=model),
)  # set deterministic=True for reproducible results later on

# automatically restores model, epoch, step, LR schedulers, etc from checkpoint
trainer.fit(model, ckpt_path="")
