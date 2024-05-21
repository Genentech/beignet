import re
from os import PathLike
from typing import Any, List, Optional, Union

from transformers import BatchEncoding, T5Tokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy

from beignet.transforms import Transform


class PT5TokenizerTransform(Transform):
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, PathLike] = "ProstT5",
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

    def transform(
        self,
        text: Union[str, List[str], List[int]],
        parameters: dict[str, Any],
    ) -> BatchEncoding:
        sequence = " ".join(list(re.sub(r"[UZOB]", "X", text)))  # replace UNK AAs

        # upper case AAs or lower case 3Di
        if self._pretrained_model_name_or_path == "ProstT5":
            sequence = (
                "<AA2fold>" + " " + sequence
                if sequence.isupper()
                else "<fold2AA>" + " " + sequence
            )

        tokenized = self._auto_tokenizer(
            sequence,
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

        return tokenized

    def validate(self, flat_inputs: list[Any]) -> None:
        pass
