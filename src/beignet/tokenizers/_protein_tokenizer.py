import importlib.resources
import os
from typing import Dict, List, Optional

import transformers.utils.logging
from transformers.tokenization_utils import PreTrainedTokenizer, Trie
from transformers.tokenization_utils_base import AddedToken

logger = transformers.utils.logging.get_logger(__name__)

TOKENIZERS_DIRECTORY = importlib.resources.files("beignet") / "data" / "tokenizers"

VOCAB_PATH = TOKENIZERS_DIRECTORY / "vocab.txt"

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/esm2_t6_8M_UR50D": 1024,
    "facebook/esm2_t12_35M_UR50D": 1024,
}


def load_vocab_file(vocab_file):
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
    return [ll.strip() for ll in lines]


class ProteinTokenizer(PreTrainedTokenizer):
    vocab_files_names: Dict[str, str] = {
        "vocab_file": "vocab.txt",
    }

    pretrained_vocab_files_map: Dict[str, Dict[str, str]] = {
        "vocab_file": {
            "facebook/esm2_t6_8M_UR50D": "https://huggingface.co/facebook/esm2_t6_8M_UR50D/resolve/main/vocab.txt",
            "facebook/esm2_t12_35M_UR50D": "https://huggingface.co/facebook/esm2_t12_35M_UR50D/resolve/main/vocab.txt",
        },
    }

    model_input_names: List[str] = [
        "attention_mask",
        "input_ids",
    ]

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file=VOCAB_PATH,
        bos_token: str | AddedToken | None = None,
        eos_token: str | AddedToken | None = "<eos>",
        unk_token: str | AddedToken | None = "<unk>",
        sep_token: str | AddedToken | None = None,
        pad_token: str | AddedToken | None = "<pad>",
        cls_token: str | AddedToken | None = "<cls>",
        mask_token: str | AddedToken | None = "<mask>",
        **kwargs,
    ):
        self.all_tokens = load_vocab_file(vocab_file)
        self._id_to_token = dict(enumerate(self.all_tokens))
        super().__init__(**kwargs)

        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        self.unique_no_split_tokens = self.all_tokens

        self._create_trie(self.unique_no_split_tokens)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def _tokenize(self, text, **kwargs):
        return text.split()

    def get_vocab_size(self, with_added_tokens=False):
        return len(self._id_to_token)

    def get_vocab(self):
        return {token: i for i, token in enumerate(self.all_tokens)}

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.eos_token_id]  # No sep token in ESM vocabulary
        if token_ids_1 is None:
            if self.eos_token_id is None:
                return cls + token_ids_0
            else:
                return cls + token_ids_0 + sep
        elif self.eos_token_id is None:
            raise ValueError(
                "Cannot tokenize multiple sequences when EOS token is not set!"
            )
        return (
            cls + token_ids_0 + sep + token_ids_1 + sep
        )  # Multiple inputs always have an EOS token

    def get_special_tokens_mask(
        self,
        token_ids_0: List,
        token_ids_1: List | None = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens
        added. This method is called when adding special tokens using the
        tokenizer `prepare_for_model` or `encode_plus` methods.
        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special
                tokens for the model.
        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0
            for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided "
                    "sequence of ids is already formatted with special tokens "
                    "for the model."
                )

            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        mask = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]
        return mask

    def save_vocabulary(self, save_directory, filename_prefix):
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.txt",
        )
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.all_tokens))
        return (vocab_file,)

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size(with_added_tokens=False)

    def _add_tokens(
        self,
        new_tokens: List[str] | List[AddedToken],
        special_tokens: bool = False,
    ) -> int:
        return super()._add_tokens(new_tokens, special_tokens=True)

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()

        for token in unique_no_split_tokens:
            if (
                hasattr(self, "do_lower_case")
                and self.do_lower_case
                and token not in self.all_special_tokens
            ):
                trie.add(token.lower())
            else:
                trie.add(token)

        self.tokens_trie = trie