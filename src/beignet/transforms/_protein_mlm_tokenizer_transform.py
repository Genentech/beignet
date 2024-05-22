import importlib.resources
from os import PathLike
from typing import Any, Dict, List, Union

from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
)

from beignet.tokenizers import ProteinMLMTokenizer

from ._transform import Transform


class ProteinMLMTokenizerTransform(Transform):
    def __init__(
        self,
        path: str | PathLike = None,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy = False,
        max_length: int | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        tokenizer_dir: str | None = "pmlm_tokenizer",
        mlm: bool = True,
        reversal_augmentation: bool = False,
    ):
        super().__init__()

        self._pretrained_model_name_or_path = path
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
        self._tokenizer_dir = tokenizer_dir
        self._mlm = mlm

        if self._pretrained_model_name_or_path is not None:
            self._auto_tokenizer = ProteinMLMTokenizer.from_pretrained(
                self._pretrained_model_name_or_path,
                do_lower_case=False,
                use_fast=True,
            )
        elif self._tokenizer_dir is not None:
            path = importlib.resources.files("lobster") / "assets" / self._tokenizer_dir
            self._auto_tokenizer = ProteinMLMTokenizer.from_pretrained(
                path,
                do_lower_case=False,
                use_fast=True,
            )

    def transform(
        self,
        text: Union[str, List[str], List[int]],
        parameters: dict[str, Any],
    ) -> BatchEncoding:
        tokenized = self._auto_tokenizer(
            text,
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

        if self._mlm:
            tokenized["labels"] = tokenized["input_ids"].clone()
            tokenized["labels"][:-1] = tokenized["input_ids"][1:]
            tokenized["labels"][-1] = -100
        else:
            labels = tokenized["input_ids"].clone()
            if self._auto_tokenizer.pad_token_id is not None:
                labels[
                    labels == self._auto_tokenizer.pad_token_id
                ] = -100  # ignore in loss
            tokenized["labels"] = labels

        return tokenized

    def _reverse_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(text, str):
            return text[::-1]
        elif isinstance(text, list):
            return [t[::-1] for t in text]

    def _transform(self, input: Any, parameters: Dict[str, Any]) -> Any:
        return self.transform(input, parameters)

    def validate(self, flat_inputs: list[Any]) -> None:
        pass

    def _check_inputs(self, inputs: List[Any]) -> None:
        pass
