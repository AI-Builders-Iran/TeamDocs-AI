from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from .base import BaseLoader

class TextLoader(BaseLoader):

    def load(self, path: Path | str) -> list[Document]:
        loader = TextLoader()
        return loader.load(path=path)
