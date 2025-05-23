import json
from dataclasses import dataclass
import re
from typing import List, Union

from utils.stop_words import load_stop_words


@dataclass
class PreprocessorConfig:
    lowercase: bool = True
    remove_punctuation: bool = True
    stopword_path: str = ""
    order_types_to_ignore: set = None

    @classmethod
    def from_json(cls, config: dict) -> "PreprocessorConfig":
        return cls(**config)

    @classmethod
    def from_json_path(cls, path: str) -> "PreprocessorConfig":
        if path[-4:] != "json":
            raise ValueError("Extension of PreprocessorConfig path not json.")
        with open(path, "r") as fp:
            config = json.load(fp)
        return cls(**config)


@dataclass
class Preprocessor(PreprocessorConfig):
    stopwords: Union[List[str], None] = None
    punctuations: str = ".,?!"

    def __post_init__(self):
        if self.stopword_path:
            self.stopwords = load_stop_words(self.stopword_path)
        if self.remove_punctuation:
            self.table = str.maketrans("","", self.punctuations)
        
        self.number_dash_pattern = re.compile(r'(?<=\d)-')

    def _lowercasing(self, text: str) -> str:
        if self.lowercase:
            return text.lower()
        else:
            return text

    def _stopword_removal(self, text: str) -> str:
        stopwords = self.stopwords
        if stopwords:
            return " ".join(t for t in text.split() if t not in stopwords)
        else:
            return text

    def _remove_punctuation(self, text: str) -> str:
        return text.translate(self.table)
    
    def _remove_number_dash(self, text: str) -> str:
        # replace a dash following a number by a space
        text = self.number_dash_pattern.sub(' ', text)
        return text

    def _process(self, text: str) -> str:
        new_t = text
        if self.lowercase:
            new_t = self._lowercasing(new_t)
        if self.remove_punctuation:
            new_t = self._remove_punctuation(new_t)
        if self.stopwords:
            new_t = self._stopword_removal(new_t)
        
        new_t = self._remove_number_dash(new_t)
        return new_t

    def _batch_process(self, texts: List[str]) -> List[str]:
        output = []
        for t in texts:
            if not isinstance(t, str):
                t = str(t)
            output.append(self._process(t))
        return output

    def __call__(self, text):
        if isinstance(text, list):
            return self._batch_process(text)
        
        if not isinstance(text, str):
            text = str(text)
        
        return self._process(text)

    @classmethod
    def from_config(cls, config: PreprocessorConfig) -> "Preprocessor":
        return cls(lowercase=config.lowercase, stopword_path=config.stopword_path)

    @classmethod
    def from_json_path(cls, path: str) -> "Preprocessor":
        config = PreprocessorConfig.from_json_path(path)
        return Preprocessor.from_config(config)
