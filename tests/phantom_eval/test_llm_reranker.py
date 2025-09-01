import asyncio
import logging
import os

import nest_asyncio
import pandas as pd

from phantom_eval.agents import get_agent
from phantom_eval.agents.common import parse_llm_reranker_response
from phantom_eval.llm import (
    InferenceGenerationConfig,
    get_llm,
)
from phantom_eval.prompts import get_llm_prompt
from phantom_eval.utils import load_data, setup_logging

CORPUS_PATH = "/Users/kanelisa/Documents/phantom-wiki/out/dataset/phantom-wiki-v1/depth_20_size_50_seed_1.jsonl"
INDEX_PATH = "/Users/kanelisa/Documents/phantom-wiki/out/indexes/bm25"
def test_llm_reranker():
    # Load data
    dataset = load_data(
        "kilian-group/phantom-wiki-v1",
        "depth_20_size_500_seed_1",
        from_local=False,
    )
    # test one question
    df_qa_pairs = pd.DataFrame(dataset["qa_pairs"])
    df_text = pd.DataFrame(dataset["text"])
    question = df_qa_pairs.iloc[0]["question"]

    # Get LLM
    llm_chat = get_llm("gemini", "gemini-2.0-flash", model_kwargs={})

    # Get LLM prompt
    llm_prompt = get_llm_prompt("llm-reranker", "gemini-2.0-flash")
    agent_kwargs = {}

    # Get agent
    agent = get_agent(
        "llm-reranker",
        text_corpus=df_text,
        llm_prompt=llm_prompt,
        agent_kwargs=agent_kwargs,
    )
    # Run agent
    inf_gen_config = InferenceGenerationConfig(
        max_tokens=4096,
        temperature=0.0,
        top_k=-1,
        top_p=0.7,
        repetition_penalty=1.0,
        max_retries=3,
        wait_seconds=2)
    response = asyncio.run(agent.run(llm_chat, question, inf_gen_config))
    reranking_list = parse_llm_reranker_response(response.pred)
    print(reranking_list)
    print(f"\nlen: {len(reranking_list)}")


def test_llm_rag_reranker():
    # Load data
    dataset = load_data(
        "kilian-group/phantom-wiki-v1",
        "depth_20_size_50_seed_1",
        from_local=False,
    )
    # test one question
    df_qa_pairs = pd.DataFrame(dataset["qa_pairs"])
    df_text = pd.DataFrame(dataset["text"])
    question = df_qa_pairs.iloc[0]["question"]

    # Get LLM
    llm_chat = get_llm("gemini", "gemini-2.0-flash", model_kwargs={})

    # Get LLM prompt
    llm_prompt = get_llm_prompt("llm-reranker", "gemini-2.0-flash")
    agent_kwargs = {}

    # Get agent
    agent_kwargs = dict(
        embedding_model_name="",  # not used for bm25
        retrieval_method="bm25",
        index_path=INDEX_PATH,
        corpus_path=CORPUS_PATH,
    )
    agent = get_agent(
        "llm-rag-reranker",
        text_corpus=df_text,
        llm_prompt=llm_prompt,
        agent_kwargs=agent_kwargs,
    )
    # Run agent
    inf_gen_config = InferenceGenerationConfig(
        max_tokens=4096,
        temperature=0.0,
        top_k=-1,
        top_p=0.7,
        repetition_penalty=1.0,
        max_retries=3,
        wait_seconds=2)
    response = asyncio.run(agent.run(llm_chat, question, inf_gen_config))
    reranking_list = parse_llm_reranker_response(response.pred)
    print(reranking_list)
    print(f"\nlen: {len(reranking_list)}")

if __name__ == "__main__":
    # test_llm_reranker()
    test_llm_rag_reranker()