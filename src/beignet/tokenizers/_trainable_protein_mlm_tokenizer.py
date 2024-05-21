import itertools
import os

import tokenizers
from datasets import load_dataset

from beignet.tokenizers import ProteinMLMTokenizer


class TrainableProteinMLMTokenizer(ProteinMLMTokenizer):
    def __init__(self, **kwargs):
        self._tokenizer, self._trainer = self._build_tokenizer(**kwargs)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def _build_tokenizer(self, **kwargs):
        pad_token = kwargs.get("pad_token", "<pad>")
        unk_token = kwargs.get("unk_token", "<unk>")
        max_vocab_size = kwargs.get("max_vocab_size", 1280)

        tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token=unk_token))
        tokenizer.normalizer = tokenizers.normalizers.NFKC()
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel()
        trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=max_vocab_size,
            initial_alphabet=[
                "A",
                "R",
                "N",
                "D",
                "C",
                "E",
                "Q",
                "G",
                "H",
                "I",
                "L",
                "K",
                "M",
                "F",
                "P",
                "S",
                "T",
                "W",
                "Y",
                "V",
                ".",
                "-",
            ],
            special_tokens=["<pad>", "<cls>", "<eos>", "<unk>", "<mask>"],
        )

        tokenizer.special_tokens_map = {"pad_token": pad_token, "unk_token": unk_token}
        return tokenizer, trainer

    @staticmethod
    def _batch_iterator(hf_dataset, batch_size, text_column):
        for i in range(0, len(hf_dataset), batch_size):
            yield hf_dataset[i : i + batch_size][text_column]

    @staticmethod
    def _batch_txt_to_hf_iterator(txt_file, batch_size, text_column="text"):
        hf_dataset = load_dataset(text_column, data_files=[txt_file])
        for i in range(0, len(hf_dataset["train"]), batch_size):
            yield hf_dataset["train"][i : i + batch_size][text_column]

    @staticmethod
    def _batch_txt_iterator(txt_file, num_lines):
        with open(txt_file, "r") as f:
            return list(itertools.islice(f, num_lines))

    def fit(self, txt_file, num_lines=100):
        self._tokenizer.train_from_iterator(
            self._batch_txt_iterator(txt_file, num_lines),
            trainer=self._trainer,
            # length=len(hf_dataset),
        )
        super().__init__(tokenizer_object=self._tokenizer)
        # setattr(self, "model_input_names", ["input_ids"])
        self.model_input_names = ["input_ids"]
        for k, v in self._tokenizer.special_tokens_map.items():
            setattr(self, k, v)
