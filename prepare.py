#!/usr/bin/env python3
"""
Разовая подготовка окружения. Запускается ДО оценки, сеть разрешена
(это явно допускает задание: "до запуска разрешается однократно скачать
зависимости и необходимые модели").

    pip install -r requirements.txt
    python prepare.py

Скачивает и кладёт ЛОКАЛЬНО:
  - models/qwen2.5-7b-instruct-q4_k_m.gguf   (LLM-реранкер, ~4.7 ГБ)
  - models/multilingual-e5-small/            (эмбеддинг-модель, ~450 МБ)

После этого run.py работает полностью офлайн, читая модели по тем же
локальным путям (src/config.py, DEFAULT_LLM_GGUF_PATH /
DEFAULT_EMBEDDING_MODEL_DIR).

Если у вас уже есть эти веса, скачанные иначе - просто положите их по
тем же путям (или передайте свои пути через --llm-model-path /
--embedding-model-dir в run.py) и запускать prepare.py не обязательно.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config


def download_llm(dest_path: str, repo_id: str, filename: str) -> None:
    if os.path.exists(dest_path):
        print(f"[skip] LLM уже есть: {dest_path}")
        return
    from huggingface_hub import hf_hub_download

    print(f"Скачиваю {repo_id}/{filename} ...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
    # hf_hub_download кладёт файл в свой кеш; копируем/симлинкуем в наш
    # предсказуемый путь, чтобы run.py не зависел от кеша huggingface.
    if os.path.abspath(downloaded) != os.path.abspath(dest_path):
        os.replace(downloaded, dest_path) if os.path.dirname(downloaded) == os.path.dirname(dest_path) \
            else _copy(downloaded, dest_path)
    print(f"Готово: {dest_path}")


def _copy(src: str, dst: str) -> None:
    import shutil

    shutil.copyfile(src, dst)


def download_embedding_model(dest_dir: str, repo_id: str) -> None:
    if os.path.isdir(dest_dir) and os.listdir(dest_dir):
        print(f"[skip] Эмбеддинг-модель уже есть: {dest_dir}")
        return
    from sentence_transformers import SentenceTransformer

    print(f"Скачиваю {repo_id} ...")
    model = SentenceTransformer(repo_id)
    os.makedirs(dest_dir, exist_ok=True)
    model.save(dest_dir)
    print(f"Готово: {dest_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm-repo", default=config.LLM_HF_REPO)
    ap.add_argument("--llm-filename", default=config.LLM_HF_FILENAME)
    ap.add_argument("--llm-dest", default=config.DEFAULT_LLM_GGUF_PATH)
    ap.add_argument("--embedding-repo", default=config.EMBEDDING_HF_REPO)
    ap.add_argument("--embedding-dest", default=config.DEFAULT_EMBEDDING_MODEL_DIR)
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-embeddings", action="store_true")
    args = ap.parse_args()

    os.makedirs(config.DEFAULT_MODELS_DIR, exist_ok=True)

    if not args.skip_llm:
        download_llm(args.llm_dest, args.llm_repo, args.llm_filename)
    if not args.skip_embeddings:
        download_embedding_model(args.embedding_dest, args.embedding_repo)

    print("\nПодготовка окружения завершена. Дальше run.py работает без сети:")
    print("    python run.py --out ./out")


if __name__ == "__main__":
    main()
