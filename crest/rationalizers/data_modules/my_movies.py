from itertools import chain
import datasets as hf_datasets
from transformers import PreTrainedTokenizerBase

from rationalizers.data_modules.imdb import ImdbDataModule


class MyMoviesDataModule(ImdbDataModule):
    """DataModule for custom Movies dataset from eraserbenchmark-master/movies_dataset_builder."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        super().__init__(d_params, tokenizer)
        # path to your custom dataset (relative to repo root)
        self.path = "../../eraserbenchmark-master/movies_dataset_builder/my_dataset/my_dataset.py"

    def setup(self, stage: str = None):
        # Load your custom dataset
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )

        # cap dataset size - useful for quick testing
        if self.max_dataset_size is not None:
            self.dataset["train"] = self.dataset["train"].select(range(min(self.max_dataset_size, len(self.dataset["train"]))))
            self.dataset["validation"] = self.dataset["validation"].select(range(min(self.max_dataset_size, len(self.dataset["validation"]))))
            self.dataset["test"] = self.dataset["test"].select(range(min(self.max_dataset_size, len(self.dataset["test"]))))

        # build tokenizer if not provided
        if self.tokenizer is None:
            # build tokenizer info (vocab + special tokens) based on train and validation set
            tok_samples = chain(
                self.dataset["train"]["text"],
                self.dataset["validation"]["text"]
            )
            self.tokenizer = self.tokenizer_cls(tok_samples)

        # function to map strings to ids
        def _encode(example: dict):
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                example["input_ids"] = self.tokenizer(
                    example["text"].strip(),
                    padding=False,  # do not pad, padding will be done later
                    truncation=True,  # truncate to max length accepted by the model
                )["input_ids"]
            else:
                example["input_ids"] = self.tokenizer.encode(example["text"].strip())
            return example

        # function to filter out examples longer than max_seq_len
        def _filter(example: dict):
            return len(example["input_ids"]) <= self.max_seq_len

        # apply encode and filter
        self.dataset = self.dataset.map(_encode)
        self.dataset = self.dataset.filter(_filter)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "label"],
            output_all_columns=True,
        )
