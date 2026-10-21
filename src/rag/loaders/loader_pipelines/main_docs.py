# rag/main_loader.py

import os

os.environ.setdefault("USER_AGENT", "TeamDocs-AI/1.0")  # must be set before WebBaseLoader is imported

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from rag.loaders import DocsLoader, PDFLoader, MarkdownLoader, TXTLoader, PPTXLoader
from rag.loaders.web_loader.web import WebLoader


class MainDocsLoader:
    """
    Single entry point for loading documents.

    ``load(source)`` auto-detects what it was given:
    a file, a directory, a ZIP archive, a URL (http/https),
    or a list/tuple/set of any of these.
    English documents only.
    """

    _SUPPORTED: dict[str, type] = {
        ".pdf": PDFLoader,
        ".txt": TXTLoader,
        ".md": MarkdownLoader,
        ".doc": DocsLoader,
        ".docx": DocsLoader,
        ".docs": DocsLoader,
        ".pptx": PPTXLoader,
    }

    _ARCHIVE_SUFFIXES: set[str] = {".zip"}
    _URL_PREFIXES: tuple[str, ...] = ("http://", "https://")
    _MAX_ZIP_DEPTH: int = 3

    def __init__(
            self,
            default_loader_kwargs: dict | None = None,
            per_suffix_kwargs: dict[str, dict] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        default_loader_kwargs:
            Kwargs shared by every loader (e.g. ``{"mode": "elements"}``).
        per_suffix_kwargs:
            Per-extension kwargs that override the shared ones
            (e.g. ``{".pptx": {"mode": "elements"}}``).
        """
        self.default_kwargs: dict = default_loader_kwargs or {}
        self.per_suffix_kwargs: dict[str, dict] = {
            s.lower(): kw for s, kw in (per_suffix_kwargs or {}).items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, source) -> list[Document]:
        """
        Load anything:

            loader = MainDocsLoader()
            loader.load("report.pdf")
            loader.load("data/")
            loader.load("archive.zip")
            loader.load("https://example.com/docs")
            loader.load(["a.pdf", "b.pptx", "data/", "https://example.com"])
        """
        if isinstance(source, (list, tuple, set)):
            return self.load_many(source)

        if self._is_url(source):
            return self.load_url(str(source))

        path = Path(source)
        if path.is_dir():
            return self.load_directory(path)

        return self.load_file(path)

    def get_loader(self, file_path: str | Path):
        """Return an instantiated loader for the given file's suffix."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        loader_cls = self._SUPPORTED.get(suffix)
        if loader_cls is None:
            raise ValueError(
                f"Unsupported file format: '{suffix}'. "
                f"Allowed formats: {list(self._SUPPORTED.keys())}"
            )

        kwargs = {**self.default_kwargs, **self.per_suffix_kwargs.get(suffix, {})}
        return loader_cls(str(path), **kwargs)

    def load_file(self, file_path: str | Path) -> list[Document]:
        """Load a single file. ZIP archives are extracted and loaded recursively."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if path.suffix.lower() in self._ARCHIVE_SUFFIXES:
            return self._load_zip(path, depth=0)

        return self._load_single_file(path)

    def load_url(self, url: str) -> list[Document]:
        """Load a web page with WebBaseLoader."""

        docs = WebLoader().load(url=url)

        for doc in docs:
            doc.metadata.setdefault("source", url)
            doc.metadata.setdefault("file_type", "web")

        return docs

    def load_many(self, sources: Iterable[str | Path]) -> list[Document]:
        """Load multiple sources; a failure on one does not stop the rest."""
        all_docs: list[Document] = []

        for s in sources:
            try:
                all_docs.extend(self.load(s))
            except Exception as exc:  # noqa: BLE001
                print(f"Skipped {s}: {exc}")

        return all_docs

    def load_directory(
            self,
            directory: str | Path,
            recursive: bool = True,
    ) -> list[Document]:
        """Load every supported file (and archive) inside a directory."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Invalid directory: {directory}")

        allowed = set(self._SUPPORTED) | self._ARCHIVE_SUFFIXES
        pattern = "**/*" if recursive else "*"
        paths = [
            p for p in directory.glob(pattern)
            if p.is_file() and p.suffix.lower() in allowed
        ]
        return self.load_many(paths)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_url(self, source) -> bool:
        return isinstance(source, str) and source.lower().startswith(self._URL_PREFIXES)

    def _load_single_file(self, path: Path) -> list[Document]:
        """Load one supported file and normalize its metadata."""
        docs = self.get_loader(path).load()

        for doc in docs:
            doc.metadata.setdefault("source", str(path))
            doc.metadata.setdefault("file_type", path.suffix.lower().lstrip("."))

        return docs

    def _load_zip(self, zip_path: Path, depth: int) -> list[Document]:
        """
        Extract a ZIP archive to a temporary directory and load its
        supported entries. Nested archives are handled up to ``_MAX_ZIP_DEPTH``.
        Guards against Zip Slip.
        """
        if depth > self._MAX_ZIP_DEPTH:
            return []

        all_docs: list[Document] = []
        tmp_dir = Path(tempfile.mkdtemp(prefix="mainloader_zip_"))

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                root = tmp_dir.resolve()
                for member in zf.infolist():
                    target = (tmp_dir / member.filename).resolve()
                    if not target.is_relative_to(root):
                        continue  # Zip Slip attempt
                    zf.extract(member, tmp_dir)

            for file in tmp_dir.rglob("*"):
                if not file.is_file():
                    continue

                suffix = file.suffix.lower()

                if suffix in self._ARCHIVE_SUFFIXES:
                    all_docs.extend(self._load_zip(file, depth + 1))
                elif suffix in self._SUPPORTED:
                    docs = self._load_single_file(file)
                    for doc in docs:
                        doc.metadata["archive_source"] = str(zip_path)
                        doc.metadata["archive_member"] = str(file.relative_to(tmp_dir))
                    all_docs.extend(docs)

        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid ZIP archive: {zip_path}") from exc
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return all_docs
