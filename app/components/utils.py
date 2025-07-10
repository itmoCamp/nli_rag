import numpy as np
import re
from typing import Optional, Tuple

def parse_response(response_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Извлекает рассуждение, метку и цитату из ответа модели.
    
    :param response_text: Ответ модели в текстовом формате
    :return: (reasoning, label, quote) - кортеж с извлеченными данными
    """
    # Извлечение блока рассуждений
    reasoning_match = re.search(
        r'РАССУЖДЕНИЕ:\s*(.*?)(?=\nОТВЕТ:)', 
        response_text, 
        re.DOTALL
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    
    # Извлечение метки
    label_match = re.search(
        r'ОТВЕТ:\s*(Entailment|Contradiction|Neutral|None of the above)\b', 
        response_text, 
        re.IGNORECASE
    )
    label = label_match.group(1).capitalize() if label_match else None
    
    # Извлечение цитаты (с поддержкой многострочных цитат)
    quote_match = re.search(
        r'SOURCES:\s*«((?:.|\n)*?)»', 
        response_text, 
        re.DOTALL
    )
    quote = quote_match.group(1).strip() if quote_match else None
    
    return reasoning, label, quote


def find_quote_span(full_text: str, quote: str) -> Optional[Tuple[int, int]]:
    """
    Ищет точное вхождение цитаты в полном тексте и возвращает span.

    :param full_text: Полный текст контракта.
    :param quote: Цитата из SOURCES.
    :return: (start_idx, end_idx) или None, если не найдено.
    """
    if not quote:
        return None

    start = full_text.find(quote)
    if start == -1:
        return None
    return (start, start + len(quote))

def find_top2_closest_chunks(chunks_embeddings: np.ndarray,
                             hypotheses_embeddings: np.ndarray) -> np.ndarray:
    """
    Быстро находит для каждой гипотезы два наиболее близких чанка по косинусному сходству.
    
    :param chunks_embeddings: np.ndarray формы (n_chunks, dim)
    :param hypotheses_embeddings: np.ndarray формы (n_hypotheses, dim)
    :return: np.ndarray формы (n_hypotheses, 2) — индексы двух ближайших чанков
    """
    # L2-нормализация векторов
    c_norm = chunks_embeddings / np.linalg.norm(chunks_embeddings, axis=1, keepdims=True)
    h_norm = hypotheses_embeddings / np.linalg.norm(hypotheses_embeddings, axis=1, keepdims=True)
    # similarity: (n_hypotheses, n_chunks)
    sims = h_norm @ c_norm.T
    # берём два максимума по строке (две наиболее близкие)
    top2_idx = np.argsort(-sims, axis=1)[:, :2]
    return top2_idx
