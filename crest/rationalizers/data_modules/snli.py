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
from rationalizers.data_modules.base import BaseDataModule
from rationalizers.data_modules.utils import token_type_ids_from_input_ids


class SNLIDataModule(BaseDataModule):
    """DataModule for SNLI Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        """
        :param d_params: hyperparams dict. See docs for more info.
        """
        super().__init__(d_params)
        # hard-coded stuff
        self.path = "stanfordnlp/snli"  # hf_datasets will handle everything
        self.is_multilabel = True
        self.nb_classes = 2  # entailment (0), contradiction (1) - removing neutral

        # hyperparams
        self.batch_size = d_params.get("batch_size", 64)
        self.num_workers = d_params.get("num_workers", 0)
        self.vocab_min_occurrences = d_params.get("vocab_min_occurrences", 1)
        self.max_seq_len = d_params.get("max_seq_len", 99999999)
        self.max_dataset_size = d_params.get("max_dataset_size", None)
        self.concat_inputs = d_params.get("concat_inputs", True)
        self.swap_pair = d_params.get("swap_pair", False)
        # Force binary classification by always filtering neutrals
        self.filter_neutrals = True  # Force this to True for binary classification
        self.ignore_neutrals = False
        self.use_revised_snli_val = d_params.get("use_revised_snli_val", False)
        # Always set nb_classes to 2 for binary classification
        self.nb_classes = 2

        # objects
        self.dataset = None
        self.label_encoder = None
        self.tokenizer = tokenizer
        self.tokenizer_cls = partial(
            # WhitespaceEncoder,
            # TreebankEncoder,
            StaticTokenizerEncoder,
            tokenize=nltk.wordpunct_tokenize,
            min_occurrences=self.vocab_min_occurrences,
            reserved_tokens=[
                constants.PAD,
                constants.UNK,
                constants.EOS,
                constants.SOS,
                "<copy>",
            ],
            padding_index=constants.PAD_ID,
            unknown_index=constants.UNK_ID,
            eos_index=constants.EOS_ID,
            sos_index=constants.SOS_ID,
            append_sos=False,
            append_eos=False,
        )
        self.sep_token = self.tokenizer.sep_token or self.tokenizer.eos_token
        self.sep_token_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id

    def _collate_fn(self, samples: list, are_samples_batched: bool = False):
        """
        :param samples: a list of dicts
        :param are_samples_batched: in case a batch/bucket sampler are being used
        :return: dict of features, label (Tensor)
        """
        if are_samples_batched:
            # dataloader batch size is 1 -> the sampler is responsible for batching
            samples = samples[0]

        # convert list of dicts to dict of lists
        collated_samples = collate_tensors(samples, stack_tensors=list)

        # pad and stack input ids
        def pad_and_stack_ids(x, pad_id=constants.PAD_ID):
            x_ids, x_lengths = stack_and_pad_tensors(x, padding_index=pad_id)
            return x_ids, x_lengths

        def stack_labels(y):
            if isinstance(y, list):
                return torch.stack(y, dim=0)
            return y

        if self.concat_inputs:
            input_ids, lengths = pad_and_stack_ids(collated_samples["input_ids"])
            token_type_ids, _ = pad_and_stack_ids(collated_samples["token_type_ids"], pad_id=2)

            # stack labels
            labels = stack_labels(collated_samples["label"])

            # keep tokens in raw format
            prem_tokens = collated_samples["premise"]
            hyp_tokens = collated_samples["hypothesis"]

            if not self.swap_pair:
                tokens = [p.strip() + ' ' + self.sep_token + ' ' + h.strip() for p, h in zip(prem_tokens, hyp_tokens)]
            else:
                tokens = [p.strip() + ' ' + self.sep_token + ' ' + h.strip() for p, h in zip(hyp_tokens, prem_tokens)]

            batch = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "lengths": lengths,
                "labels": labels,
                "tokens": tokens,
            }

        else:
            prem_ids, prem_lengths = pad_and_stack_ids(collated_samples["prem_ids"])
            hyp_ids, hyp_lengths = pad_and_stack_ids(collated_samples["hyp_ids"])

            # stack labels
            labels = stack_labels(collated_samples["label"])

            # keep tokens in raw format
            prem_tokens = collated_samples["premise"]
            hyp_tokens = collated_samples["hypothesis"]

            # return batch to the data loader
            batch = {
                "prem_ids": prem_ids,
                "hyp_ids": hyp_ids,
                "prem_lengths": prem_lengths,
                "hyp_lengths": hyp_lengths,
                "labels": labels,
                "prem_tokens": prem_tokens,
                "hyp_tokens": hyp_tokens,
            }
        return batch

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        print("=" * 80)
        print("SNLI SETUP METHOD CALLED - CUSTOM CODE IS RUNNING!")
        print("=" * 80)
        # Assign train/val/test datasets for use in dataloaders
        if isinstance(self.path, (list, tuple)):
            self.dataset = hf_datasets.load_dataset(
                *self.path,
                download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
            )
        else:
            self.dataset = hf_datasets.load_dataset(
                "stanfordnlp/snli",
                download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
            )

        if self.use_revised_snli_val:
            print('Using revised SNLI val set')
            ds_rev = hf_datasets.load_dataset(
                path="./rationalizers/custom_hf_datasets/revised_snli.py",
                download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
                side='both',
            )
            ds_rev = ds_rev.filter(lambda x: x['is_original'])
            ds_rev = ds_rev.rename_column("prem", "premise")
            ds_rev = ds_rev.rename_column("hyp", "hypothesis")
            ds_rev = ds_rev.remove_columns(["batch_id", "is_original"])
            self.dataset['validation'] = ds_rev['test']

        # filter out invalid samples
        self.dataset = self.dataset.filter(lambda ex: ex["label"] != -1)
        print("=== DEBUG: Starting label processing ===")
        print("Original label distribution before any processing:")
        def get_dist(y):
            vals, counts = np.unique(y, return_counts=True)
            return dict(zip(vals, counts / counts.sum()))
        
        print("TRAIN:", get_dist(self.dataset["train"]["label"]))
        print("VAL:", get_dist(self.dataset["validation"]["label"]))
        print("TEST:", get_dist(self.dataset["test"]["label"]))
        
        # Remove neutral labels (label == 1) and remap to binary classification
        def process_labels(example):
            if example["label"] == 1:  # neutral - remove this sample
                return None  # This will be filtered out
            elif example["label"] == 0:  # entailment stays 0
                example["label"] = 0
                return example
            elif example["label"] == 2:  # contradiction becomes 1
                example["label"] = 1
                return example
            else:
                print(f"WARNING: Unexpected label value: {example['label']}")
                return None  # Filter out unexpected values
            
        # Filter out neutrals first
        print("=== DEBUG: Filtering out neutrals ===")
        self.dataset = self.dataset.filter(lambda ex: ex["label"] != 1)
        print("After filtering neutrals - label distribution:")
        print("TRAIN:", get_dist(self.dataset["train"]["label"]))
        
        # Then remap labels using a simpler approach
        print("=== DEBUG: Remapping contradiction labels (2->1) ===")
        def remap_contradiction_to_1(example):
            if example["label"] == 2:
                example["label"] = 1
            return example
            
        self.dataset = self.dataset.map(remap_contradiction_to_1)
        print("After first remapping - label distribution:")
        print("TRAIN:", get_dist(self.dataset["train"]["label"]))
        
        # Double-check by forcing the label conversion
        print("=== DEBUG: Ensuring binary labels (batch processing) ===")
        def ensure_binary_labels(batch):
            # Convert all 2s to 1s in the batch
            labels = []
            for label in batch["label"]:
                if label == 2:
                    labels.append(1)
                else:
                    labels.append(label)
            batch["label"] = labels
            return batch
            
        self.dataset = self.dataset.map(ensure_binary_labels, batched=True)
        print("After ensuring binary - label distribution:")
        print("TRAIN:", get_dist(self.dataset["train"]["label"]))
        print("=== DEBUG: Label processing complete ===")
        
        print("FINAL LABEL CHECK - should be 0s and 1s only:")
        train_labels = self.dataset["train"]["label"]
        unique_labels = set(train_labels)
        print("Unique labels in training set:", unique_labels)

        # cap dataset size - useful for quick testing
        if self.max_dataset_size is not None:
            self.dataset["train"] = self.dataset["train"].select(range(self.max_dataset_size))
            self.dataset["validation"] = self.dataset["validation"].select(range(self.max_dataset_size))
            self.dataset["test"] = self.dataset["test"].select(range(self.max_dataset_size))

        # build tokenize rand label encoder
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

        # Note: Neutrals have already been filtered and labels remapped above
        # No additional filtering needed here

        print("=== FINAL DISTRIBUTION CHECK ===")
        print(get_dist(self.dataset["train"]["label"]))
        print(get_dist(self.dataset["validation"]["label"]))
        print(get_dist(self.dataset["test"]["label"]))

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
