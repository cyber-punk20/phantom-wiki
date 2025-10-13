import asyncio
import logging
import os

import nest_asyncio
import pandas as pd
import vertexai

from phantom_eval.agents import get_agent
from phantom_eval import get_parser

from phantom_eval.llm import (
    InferenceGenerationConfig,
    get_llm,
)
from phantom_eval.prompts import get_llm_prompt
from phantom_eval.utils import load_data, setup_logging

nest_asyncio.apply()

def test_sca_vertexai():
    setup_logging("DEBUG")
    parser = get_parser()
    args = parser.parse_args()

    assert (
        args.corpus_name is not None
    ), "corpus_name must be specified when retrieval_method is vertexai"
    assert (
        args.vertexai_project_id is not None
    ), "vertexai_project_id must be specified when retrieval_method is vertexai"
    assert (
        args.vertexai_location is not None
    ), "vertexai_location must be specified when retrieval_method is vertexai"
    vertexai.init(project=args.vertexai_project_id, location=args.vertexai_location)
    # Load data
    dataset = load_data(
        "kilian-group/phantom-wiki-v1",
        "depth_20_size_50_seed_1",
        from_local=False,
    )
    # test one question
    df_qa_pairs = pd.DataFrame(dataset["qa_pairs"])
    question = df_qa_pairs.iloc[10]["question"]

    # Get LLM
    llm_chat = get_llm("gemini", "gemini-2.5-pro", model_kwargs={})

    # Get LLM prompt
    llm_prompt = get_llm_prompt("sca-qr", "gemini-2.5-pro")

    # Get agent
    agent_kwargs = dict(
        retriever_num_documents=2,
        corpus_name=args.corpus_name,
        retrieval_method="vertexai",
        sca_max_steps=10,
    )

    agent = get_agent(
        "sca-qr",
        text_corpus=[],
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
    print("response: \n")
    print(response)
    print("agent interactions: \n")
    print(agent.agent_interactions)

if __name__ == "__main__":
    test_sca_vertexai()