import random
import re
from os import PathLike
from typing import Any, List, Optional, Union

from transformers import BatchEncoding, T5Tokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy

from beignet.transforms import Transform


class PT5TeacherForcingTransform(Transform):
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, PathLike] = "ProstT5",
        mask_percentage: float = 0.125,
        padding: Union[bool, str, PaddingStrategy] = "max_length",
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: int = 512,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: bool = True,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ):
        super().__init__()

        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._padding = padding
        self._mask_percentage = mask_percentage
        self._truncation = truncation
        self._max_length = max_length
        self._return_token_type_ids = return_token_type_ids
        self._return_attention_mask = return_attention_mask
        self._return_overflowing_tokens = return_overflowing_tokens
        self._return_special_tokens_mask = return_special_tokens_mask
        self._return_offsets_mapping = return_offsets_mapping
        self._return_length = return_length
        self._verbose = verbose

        self._auto_tokenizer = T5Tokenizer.from_pretrained(
            f"Rostlab/{pretrained_model_name_or_path}", do_lower_case=False
        )

    def transform(self, text: str | List[str] | List[int], **_) -> BatchEncoding:
        sequence = " ".join(list(re.sub(r"[UZOB]", "X", text)))  # replace UNK AAs
        num_toks = len(sequence.split())
        num_to_mask = max(1, int(self._mask_percentage * num_toks))
        mask_idxs = random.sample(range(num_toks), num_to_mask)
        input_sequence = []
        label_sequence = []
        mask_idx = 0
        for idx, char in enumerate(sequence.split()):
            if idx in mask_idxs:
                input_sequence.append(f"<extra_id_{mask_idx}>")
                label_sequence.append(char)
                mask_idx += 1
            else:
                input_sequence.append(char)
                if len(label_sequence) == 0:
                    label_sequence.append("<extra_id_0>")
                if len(label_sequence) > 0 and not label_sequence[-1].startswith(
                    "<extra_id_"
                ):
                    label_sequence.append(f"<extra_id_{mask_idx}>")

        input_sequence = " ".join(input_sequence)
        label_sequence = " ".join(label_sequence)

        # upper case AAs or lower case 3Di
        if self._pretrained_model_name_or_path == "ProstT5":
            input_sequence = (
                "<AA2fold>" + " " + input_sequence
                if sequence.isupper()  # uppercase original input -> AA seq
                else "<fold2AA>" + " " + input_sequence
            )
            label_sequence = (
                "<AA2fold>" + " " + label_sequence
                if sequence.isupper()  # uppercase original input -> AA seq
                else "<fold2AA>" + " " + label_sequence
            )

        input_ids = self._auto_tokenizer(
            input_sequence,
            add_special_tokens=True,
            padding=self._padding,
            truncation=self._truncation,
            max_length=self._max_length,
            return_tensors="pt",
            return_token_type_ids=self._return_token_type_ids,
            return_attention_mask=self._return_attention_mask,
            return_overflowing_tokens=self._return_overflowing_tokens,
            return_special_tokens_mask=self._return_special_tokens_mask,
            return_offsets_mapping=self._return_offsets_mapping,
            return_length=self._return_length,
            verbose=self._verbose,
        )

        label_ids = self._auto_tokenizer(
            label_sequence,
            add_special_tokens=True,
            padding=self._padding,
            truncation=self._truncation,
            max_length=self._max_length,
            return_tensors="pt",
            return_token_type_ids=self._return_token_type_ids,
            return_attention_mask=self._return_attention_mask,
            return_overflowing_tokens=self._return_overflowing_tokens,
            return_special_tokens_mask=self._return_special_tokens_mask,
            return_offsets_mapping=self._return_offsets_mapping,
            return_length=self._return_length,
            verbose=self._verbose,
        )

        label_ids["labels"] = label_ids["input_ids"].clone()
        label_ids["labels"][
            label_ids["labels"] == self._auto_tokenizer.pad_token_id
        ] = -100  # ignore in loss

        tokenized = {
            "input_ids": input_ids["input_ids"],
            "attention_mask": input_ids["attention_mask"],
            "label_ids": label_ids["labels"],
        }
        return tokenized

    def validate(self, flat_inputs: list[Any]) -> None:
        pass
