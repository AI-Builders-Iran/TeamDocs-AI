from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .schema import SemanticMetadata
from .prompts import METADATA_EXTRACTION_PROMPT


def build_metadata_context(
        text: str,
        max_chars: int = 12_000,
) -> str:
    """Build a bounded context for document-level metadata extraction."""

    text = text.strip()

    if len(text) <= max_chars:
        return text

    part_size = max_chars // 3

    beginning = text[:part_size]
    middle_start = (len(text) - part_size) // 2
    middle = text[middle_start:middle_start + part_size]
    ending = text[-part_size:]

    return (
        "[BEGINNING OF DOCUMENT]\n"
        f"{beginning}\n\n"
        "[MIDDLE OF DOCUMENT]\n"
        f"{middle}\n\n"
        "[END OF DOCUMENT]\n"
        f"{ending}"
    )


class MetadataExtractor:
    """Extract and attach semantic metadata to LangChain documents."""

    def __init__(self, llm: BaseChatModel):
        self.prompt = ChatPromptTemplate.from_template(
            METADATA_EXTRACTION_PROMPT
        )

        self.structured_llm = llm.with_structured_output(
            SemanticMetadata
        )

        self.chain = self.prompt | self.structured_llm

    def extract_document(self, document: Document) -> Document:
        """Extract semantic metadata from a single document."""

        if not document.page_content.strip():
            return document

        semantic_metadata = self.chain.invoke(
            {
                "document_content": build_metadata_context(document.page_content)
            }
        )

        if not isinstance(semantic_metadata, SemanticMetadata):
            semantic_metadata = SemanticMetadata.model_validate(
                semantic_metadata
            )

        metadata = document.metadata.copy()

        metadata.update(
            semantic_metadata.model_dump(exclude_none=True)
        )

        return Document(
            page_content=document.page_content,
            metadata=metadata,
        )

    def extract(
            self,
            documents: list[Document],
    ) -> list[Document]:
        """Extract metadata from multiple documents."""

        return [
            self.extract_document(document)
            for document in documents
        ]
