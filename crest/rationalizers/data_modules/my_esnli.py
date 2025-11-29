from functools import partial
from itertools import chain
import datasets as hf_datasets
import nltk
import numpy as np
import torch
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor
from torchnlp.utils import collate_tensors
from transformers import PreTrainedTokenizerBase

from rationalizers import constants
from rationalizers.data_modules.snli import SNLIDataModule
from rationalizers.data_modules.utils import token_type_ids_from_input_ids


class MyESNLIDataModule(SNLIDataModule):
    """DataModule for custom e-SNLI dataset from eraserbenchmark-master/esnli_dataset_builder."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        super().__init__(d_params, tokenizer)
        # path to your custom dataset (absolute path or relative to crest/ directory)
        self.path = "../eraserbenchmark-master/esnli_dataset_builder/my_dataset/my_dataset.py"
        self.is_multilabel = True
        self.nb_classes = 2  # entailment (1), not_entailment (0)
        # Dataset already has binary labels, no need to filter neutrals

    def setup(self, stage: str = None):
        # Load your custom dataset
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )

        # Rename columns to match expected format
        # sentence1 is hypothesis, sentence2 is premise
        self.dataset = self.dataset.rename_column("sentence1", "hypothesis")
        self.dataset = self.dataset.rename_column("sentence2", "premise")

        # Debug: print unique label values right after loading
        import numpy as np
        print("DEBUG: Unique labels after loading and renaming:")
        print("Train:", np.unique(self.dataset["train"]["label"]))
        print("Val:", np.unique(self.dataset["validation"]["label"]))
        print("Test:", np.unique(self.dataset["test"]["label"]))

        # cap dataset size - useful for quick testing
        if self.max_dataset_size is not None:
            self.dataset["train"] = self.dataset["train"].select(range(min(self.max_dataset_size, len(self.dataset["train"]))))
            self.dataset["validation"] = self.dataset["validation"].select(range(min(self.max_dataset_size, len(self.dataset["validation"]))))
            self.dataset["test"] = self.dataset["test"].select(range(min(self.max_dataset_size, len(self.dataset["test"]))))

        # build tokenizer if not provided
        if self.tokenizer is None:
            # build tokenizer info (vocab + special tokens) based on train and validation set
            tok_samples = chain(
                self.dataset["train"]["premise"],
                self.dataset["train"]["hypothesis"],
                self.dataset["validation"]["premise"],
                self.dataset["validation"]["hypothesis"],
            )
            self.tokenizer = self.tokenizer_cls(tok_samples)

        # map strings to ids
        def _encode(ex: dict):
            if self.concat_inputs:
                if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                    if not self.swap_pair:
                        input_enc = self.tokenizer(
                            ex["premise"].strip(),
                            ex["hypothesis"].strip(),
                            padding=False,  # do not pad, padding will be done later
                            truncation=True,  # truncate to max length accepted by the model
                        )
                    else:
                        input_enc = self.tokenizer(
                            ex["hypothesis"].strip(),
                            ex["premise"].strip(),
                            padding=False,  # do not pad, padding will be done later
                            truncation=True,  # truncate to max length accepted by the model
                        )
                    ex["input_ids"] = input_enc["input_ids"]
                    ex["token_type_ids"] = token_type_ids_from_input_ids(ex["input_ids"], self.sep_token_id)
                else:
                    if not self.swap_pair:
                        ex["input_ids"] = self.tokenizer.encode(
                            ex["premise"].strip() + ' ' + self.sep_token + ' ' + ex["hypothesis"].strip()
                        )
                    else:
                        ex["input_ids"] = self.tokenizer.encode(
                            ex["hypothesis"].strip() + ' ' + self.sep_token + ' ' + ex["premise"].strip()
                        )
                    ex["token_type_ids"] = token_type_ids_from_input_ids(ex["input_ids"], self.sep_token_id)
            else:
                if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                    ex["prem_ids"] = self.tokenizer(
                        ex["premise"].strip(),
                        padding=False,  # do not pad, padding will be done later
                        truncation=True,  # truncate to max length accepted by the model
                    )["input_ids"]
                    ex["hyp_ids"] = self.tokenizer(
                        ex["hypothesis"].strip(),
                        padding=False,  # do not pad, padding will be done later
                        truncation=True,  # truncate to max length accepted by the model
                    )["input_ids"]
                else:
                    ex["prem_ids"] = self.tokenizer.encode(ex["premise"].strip())
                    ex["hyp_ids"] = self.tokenizer.encode(ex["hypothesis"].strip())
            return ex

        self.dataset = self.dataset.map(_encode)

        if self.concat_inputs:
            self.dataset = self.dataset.filter(lambda ex: len(ex["input_ids"]) <= self.max_seq_len)
        else:
            self.dataset = self.dataset.filter(lambda ex: len(ex["prem_ids"]) <= self.max_seq_len)
            self.dataset = self.dataset.filter(lambda ex: len(ex["hyp_ids"]) <= self.max_seq_len)

        def get_dist(y):
            vals, counts = np.unique(y, return_counts=True)
            return dict(zip(vals, counts / counts.sum()))

        print("Label distribution:")
        print("Train:", get_dist(self.dataset["train"]["label"]))
        print("Val:", get_dist(self.dataset["validation"]["label"]))
        print("Test:", get_dist(self.dataset["test"]["label"]))
        print(f"Total samples - Train: {len(self.dataset['train'])}, Val: {len(self.dataset['validation'])}, Test: {len(self.dataset['test'])}")

        # convert `columns` to pytorch tensors and keep un-formatted columns
        if self.concat_inputs:
            self.dataset.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "label",],
                output_all_columns=True,
            )
        else:
            self.dataset.set_format(
                type="torch",
                columns=["prem_ids", "hyp_ids", "label",],
                output_all_columns=True,
            )
