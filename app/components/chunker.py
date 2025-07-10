from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed

class TextChunker(RecursiveCharacterTextSplitter):
    def __init__(
        self,
        chunk_size=150,
        chunk_overlap=0,
        separators=["\n\n", "\n", "?", "!", ".", ";"],
        max_workers: int = 4
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.length_function,
            separators=separators,
        )
        self.max_workers = max_workers

    def split_texts(self, texts):
        """
        Параллельно разбивает список текстов на чанки с использованием воркеров.
        :param texts: список строк для разбиения
        :param max_workers: количество параллельных процессов (по умолчанию 4)
        :return: список списков чанков
        """
        # results = [self.split_text(text) for text in texts]
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.split_text, text): text for text in texts}
            for future in as_completed(futures):
                results.append(future.result())
        return results

    @staticmethod
    def length_function(text):
        return len(text.split())
