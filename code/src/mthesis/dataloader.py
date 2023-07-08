import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader
from collections import OrderedDict
import torch

import os
import logging

from mthesis.utils import read_paragraph

log = logging.getLogger(__name__)


class ConceptDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = (
            ">>ABSTRACT<<"  # = Token_id of 1
            + row["abstract"]
            + ">>SUMMARY<<"  # = Token_id of 3
            + str(row["tags"].split(","))
            + ">>INTRODUCTION<<"  # = Token_id of 2
        )

        # Tokenize text and tags separately
        text_encodings = tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer_max_length,
        )

        return {
            "input_ids": text_encodings["input_ids"].flatten().to(device),
            "attention_mask": text_encodings["attention_mask"].flatten().to(device),
            "labels": text_encodings["input_ids"].flatten().to(device),
        }


class MOFDataset(torch.utils.data.Dataset):
    """Dataset of MOF synthesis paragraphs, without labels."""

    def __init__(self, dataset_path: str = None):
        # TODO: hardcoded location: `dataset_path`
        if not dataset_path:
            dataset_path = "~/mof_synthesis/synthesis_paragraphs"
        log.info(f"Loading MOFDataset from '{dataset_path}'")
        self.paragraph_list = os.listdir(dataset_path)
        self.paragraphs = dict()
        for p_id in self.paragraph_list:
            try:
                self.paragraphs[p_id] = read_paragraph(p_id, dataset_path)
            except Exception as e:
                log.warning(e)
                continue
        self.paragraphs = OrderedDict(self.paragraphs)
        self.paragraph_list = list(self.paragraphs.items())

    def __len__(self):
        # return length of full dataset
        log.info(f"len: {len(self.paragraphs)}")
        return len(self.paragraphs)

    def __getitem__(self, idx: int | str) -> dict:
        # return item at position idx.
        paragraph = ""
        try:
            paragraph = self.paragraphs[idx]
        except KeyError:
            idx, paragraph = self.paragraph_list[idx]

        # since Jsonformer is doing tokenization later on,
        # we're not doing that yet
        return {
            "paragraph_id": idx,
            "text": paragraph,
        }


class MOFDatasetWithLabels(MOFDataset):
    """Dataset of MOF synthesis paragraphs, with labels for ."""

    def __init__(self, dataset_path: str = None, label_cols: dict = None):
        # load paragraphs first
        super().__init__(tokenizer, dataset_path)

        if not labels_for:
            labels_for = {
                "additive": "additive1",
                "solvent": "solvent1",
                "temperature": "temperature_Celsius",
                "time": "time_h",
            }

        # now load label data
        labels_path_A = os.path.join(self.dataset_path, "results", "SynMOF_A_out.csv")
        labels_path_M = os.path.join(self.dataset_path, "results", "SynMOF_M_out.csv")

        labels_A = pd.read_csv(labels_path_A, sep=";")
        labels_M = pd.read_csv(labels_path_M, sep=";")

        for paragraph_id in self.paragraph_list:
            self.labels[paragraph_id] = dict()
            for parameter, config in self.extract_config.items():
                try:
                    col_name = config["dataset_cols"][0]
                    answer_a = labels_A.loc[labels_A["filename"] == paragraph_id][
                        col_name
                    ].values[0]
                    answer_a = int(answer_a) if not np.isnan(answer_a) else None
                    answer_m = (
                        None
                        if paragraph_id not in labels_M["filename"].values
                        else labels_M.loc[labels_M["filename"] == paragraph_id][
                            col_name
                        ].values[0]
                    )
                    if answer_m is not None:
                        answer_m = int(answer_m) if not np.isnan(answer_m) else None
                    self.labels[paragraph_id][parameter] = {
                        "actual_a": answer_a,
                        "actual_m": answer_m,
                    }
                except KeyError as e:
                    log.critical(
                        "KeyError: Invalid SETTINGSFILE. Requested `dataset_cols` don't exist. Failed when loading actuals."
                    )
                    sys.exit(1)

        # TODO: convert additive, solvent to cid
        # TODO: convert cid (from label) to string
        # resulting output from Jsonformer:
        # {'additive': 'water', 'solvent': 'water', 'temperature': 90.0, 'temperature_unit': 'C', 'time': 40.0, 'time_unit': 'h'}

    def __getitem__(self, idx: int | str) -> dict:
        # return item at position idx
        raise NotImplementedError


class MOFDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        # download, tokenize, save data to disk.
        # called from main process only. local (object) variables won't be available
        # when doing distributed training. Last step should always be to save to disk.
        pass

    def setup(self, stage: str):
        # how to split, define dataset, etc
        # `stage` is one of "fit" / "train"
        # first step is to load from disk. this is executed for each
        # distributed shard.
        pass

    def predict_dataloader(self):
        pass
