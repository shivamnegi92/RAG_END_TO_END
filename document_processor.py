from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config


class DocumentProcessor:
    "Handles PDF loading and text chunking for embedding and Retrival."

    def __init__(
        self, chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

    def load_and_split(self, file_path):
        "Handles PDF file load and splits to text chunks using splitter"

        loader = PDFPlumberLoader(file_path)
        document = loader.load()
        return self.text_splitter.split_documents(document)


if __name__ == "__main__":

    processor = DocumentProcessor()
    chunks = processor.load_and_split("uploads/Draft_2.docx.pdf")
    for chunk in chunks[:3]:
        print(chunk.page_content)
