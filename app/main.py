from dotenv import load_dotenv
import os
import yaml
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
import numpy as np

from yandex_cloud_ml_sdk import YCloudML
from concurrent.futures import ThreadPoolExecutor, as_completed

from .components.answer import answer_question
from .components.embeder import embed_doc
from .components.chunker import TextChunker
from .components.utils import parse_response, find_quote_span, find_top2_closest_chunks

# Загрузка переменных из .env
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Загрузка токена и ID папки
IAM_TOKEN = os.getenv("YC_IAM_TOKEN") or os.getenv("IAM_TOKEN")
FOLDER_ID = os.getenv("FOLDER_ID")
if not IAM_TOKEN or not FOLDER_ID:
    logger.error("Не заданы переменные окружения YC_IAM_TOKEN и/или FOLDER_ID")
    raise EnvironmentError("Требуются YC_IAM_TOKEN и FOLDER_ID")
logger.info("IAM_TOKEN и FOLDER_ID загружены")

# Загрузка конфигурации
with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
logger.info("Конфигурация загружена из config.yml")

LIST_OF_HYPOTHESIS_TOS = config.get("list_of_hypothesis_tos") or []
LIST_OF_HYPOTHESIS_PP  = config.get("list_of_hypothesis_pp") or []
LIST_OF_HYPOTHESIS_NDA = config.get("list_of_hypothesis_nda") or []
PROMPT = config.get("prompt", "")

# Инициализация Yandex Cloud ML SDK и моделей
sdk = YCloudML(folder_id=FOLDER_ID, auth=IAM_TOKEN)
chunker   = TextChunker(chunk_size=200, chunk_overlap=0, max_workers=2)
gen_model = sdk.models.completions("yandexgpt-lite", model_version="rc")
doc_model = sdk.models.text_embeddings("doc")
logger.info("Модели и SDK готовы")

# Предвычисляем эмбеддинги гипотез
tos_embeddings = np.array([embed_doc(doc_model, h) for h in LIST_OF_HYPOTHESIS_TOS])
pp_embeddings  = np.array([embed_doc(doc_model, h) for h in LIST_OF_HYPOTHESIS_PP])
nda_embeddings = np.array([embed_doc(doc_model, h) for h in LIST_OF_HYPOTHESIS_NDA])
logger.info("Эмбеддинги гипотез готовы")

# Pydantic-модели
class AnswerRequest(BaseModel):
    text: str
    task: str

class AnswerItem(BaseModel):
    hypothesis: str
    reasoning: str
    quote: str
    chunks: List[str]    # <- теперь список строк
    label: str
    label_num: int
    start: int
    end: int

app = FastAPI(title="Contract NLI Service")

def split_and_embed(text: str, max_workers: int = None) -> Tuple[List[str], np.ndarray]:
    chunks = chunker.split_texts([text])[0]
    logger.info(f"Split text into {len(chunks)} chunks.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(embed_doc, doc_model, c) for c in chunks]
        embeddings = [f.result() for f in as_completed(futures)]
    return chunks, np.array(embeddings)

def get_hypotheses(task: str) -> Tuple[np.ndarray, List[str]]:
    if task == "tos":
        return tos_embeddings, LIST_OF_HYPOTHESIS_TOS
    elif task == "pp":
        return pp_embeddings, LIST_OF_HYPOTHESIS_PP
    elif task == "nda":
        return nda_embeddings, LIST_OF_HYPOTHESIS_NDA
    else:
        logger.error(f"Unknown task type: {task}")
        raise ValueError("Unknown task type. Use 'tos', 'pp' or 'nda'.")

def find_relevant_chunks(chunks_emb: np.ndarray, hyp_emb: np.ndarray) -> List[List[int]]:
    return find_top2_closest_chunks(chunks_emb, hyp_emb)

def build_prompt(concept: str, chunk_list: List[str]) -> str:
    formatted = "\n".join(f"- «{c}»" for c in chunk_list)
    return PROMPT.format(concept=concept, chunks=formatted)

def process_hypothesis(args: Tuple[str, List[str]]) -> Optional[Dict]:
    concept, chunk_list = args
    prompt_text = build_prompt(concept, chunk_list)
    tag_and_seq = answer_question(gen_model, text=prompt_text)
    reasoning, label, quote = parse_response(tag_and_seq)
    span = find_quote_span(prompt_text, quote)
    label_num = 1 if label == "Entailment" else 0 if label == "Neutral" else -1

    logger.info(f"Hypothesis '{concept}': label={label}, quote='{quote}'")

    if label and span and label != "None of the above" and quote != 'Нет релевантных фрагментов':
        return {
            "hypothesis": concept,
            "reasoning": reasoning,
            "quote": quote,
            "chunks": chunk_list,     # <- список строк, а не единая строка
            "label": label,
            "label_num": label_num,
            "start": span[0],
            "end": span[1],
        }
    return None

def get_answer(text: str, task: str, max_workers: int = None) -> List[Dict]:
    logger.info(f"get_answer called with task='{task}'.")
    chunks, chunks_emb = split_and_embed(text, max_workers)
    hyp_emb, hypotheses = get_hypotheses(task)
    relevant = find_relevant_chunks(chunks_emb, hyp_emb)

    results: List[Optional[Dict]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_hypothesis, (hypotheses[i], [chunks[j] for j in relevant[i]]))
            for i in range(len(hypotheses))
        ]
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)

    logger.info(f"get_answer produced {len(results)} result(s).")
    return results

@app.post("/answer", response_model=List[AnswerItem])
async def get_answer_api(req: AnswerRequest):
    logger.info("Received request: task=%s, text_len=%d", req.task, len(req.text))
    try:
        answer = get_answer(req.text, req.task)
        logger.info("Returning %d items", len(answer))
        return answer
    except ValueError as e:
        logger.error("Error in get_answer: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
