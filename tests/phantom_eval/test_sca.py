import asyncio
import logging
import os

import nest_asyncio
import pandas as pd

from phantom_eval.agents import get_agent
from phantom_eval.llm import (
    InferenceGenerationConfig,
    get_llm,
)
from phantom_eval.prompts import get_llm_prompt
from phantom_eval.utils import load_data, setup_logging

nest_asyncio.apply()

# To run this test, you need to first generate the corpus and index files.
# See https://github.com/kilian-group/phantom-wiki/wiki/RAG for instructions.
# Then set the environment variables before running pytest.
# NOTE: The INDEX_PATH should point to the index file/prefix, not just the directory.
CORPUS_PATH = "/Users/kanelisa/Documents/phantom-wiki/out/dataset/phantom-wiki-v1/depth_20_size_50_seed_1.jsonl"
INDEX_PATH = "/Users/kanelisa/Documents/phantom-wiki/out/indexes/bm25"


def test_sufficient_context_autorater():
    # Set up logging to see debug messages from the agent
    setup_logging("DEBUG")

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
    llm_prompt = get_llm_prompt("sufficient-context-autorater", "gemini-2.0-flash")

    # Get agent
    agent_kwargs = dict(
        embedding_model_name="",  # not used for bm25
        retriever_num_documents=4,
        retrieval_method="bm25",
        index_path=INDEX_PATH,
        corpus_path=CORPUS_PATH,
        sca_max_steps=2,
    )
    agent = get_agent(
        "sufficient-context-autorater",
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

    # assert response.error is None
    # assert "Sufficient Context" in response.pred


if __name__ == "__main__":
    if not CORPUS_PATH or not INDEX_PATH or not os.path.exists(CORPUS_PATH) or not os.path.exists(INDEX_PATH):
        print("CORPUS_PATH and INDEX_PATH must be set and point to the correct files.")
    else:
        test_sufficient_context_autorater()