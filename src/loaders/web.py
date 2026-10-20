import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from .base import BaseLoader


class WebLoader(BaseLoader):

    def load(self, url: str) -> list[Document]:
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs={"parse_only": bs4.SoupStrainer(["article", "main"])},
            continue_on_failure=True,
        )
        return loader.load()
