from pathlib import Path
from langchain_core.documents import Document
from abc import ABC, abstractmethod

class BaseLoader(ABC):
    """a base class for all loaders"""
    @abstractmethod
    def load(self, path: Path | str)-> Document:
        raise NotImplementedError()