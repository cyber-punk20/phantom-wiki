import pandas as pd

from phantom_eval.agents.common import Agent, SufficientContextAutorater, LLMReranker, LLMRAGReranker
from phantom_eval.agents.cot import CoTAgent, CoTRAGAgent, CoTSCAgent, CoTWithSCAAgent, CoTWithLLMRerankerAgent
from phantom_eval.agents.nshot import NshotAgent, NshotRAGAgent,  NshotSCAgent, NshotWithSCAAgent, NshotWithLLMRankerAgent
from phantom_eval.agents.react import ActAgent, CoTSC_ReactAgent, React_CoTSCAgent, ReactAgent
from phantom_eval.prompts import LLMPrompt

SUPPORTED_METHOD_NAMES: list[str] = [
    "zeroshot",
    "fewshot",
    "zeroshot-sca",
    "fewshot-sca",
    "zeroshot-sc",
    "fewshot-sc",
    "cot",
    "cot-sc",
    "cot-sca",
    "react",
    "act",
    "react->cot-sc",
    "cot-sc->react",
    "zeroshot-rag",
    "fewshot-rag",
    "cot-rag",
    "sufficient-context-autorater",
]


def get_agent(
    method: str,
    text_corpus: pd.DataFrame,
    llm_prompt: LLMPrompt,
    agent_kwargs: dict,
) -> Agent:
    """
    Returns an `Agent` object based on the specified method.
    """
    match method:
        case "zeroshot" | "fewshot":
            return NshotAgent(text_corpus, llm_prompt, **agent_kwargs)
        case "zeroshot-sc" | "fewshot-sc":
            return NshotSCAgent(text_corpus, llm_prompt, **agent_kwargs)
        case "zeroshot-sca" | "fewshot-sca":
            return NshotWithSCAAgent(text_corpus, llm_prompt, **agent_kwargs)
        case "zeroshot-reranker" | "fewshot-reranker" | "zeroshot-rag-reranker" | "fewshot-rag-reranker":
            return NshotWithLLMRankerAgent(text_corpus, llm_prompt, **agent_kwargs)
        case "cot":
            return CoTAgent(text_corpus, llm_prompt, **agent_kwargs)
        case "cot-sc":
            return CoTSCAgent(text_corpus, llm_prompt, **agent_kwargs)
        case "cot-sca":
            return CoTWithSCAAgent(text_corpus, llm_prompt, **agent_kwargs)
        case "cot-reranker" | "cot-rag-reranker":
            return CoTWithLLMRerankerAgent(text_corpus, llm_prompt, **agent_kwargs)
        case "react":
            return ReactAgent(text_corpus, llm_prompt, **agent_kwargs)
        case "act":
            return ActAgent(text_corpus, llm_prompt, **agent_kwargs)
        case "react->cot-sc":
            return React_CoTSCAgent(text_corpus, llm_prompt, **agent_kwargs)
        case "cot-sc->react":
            return CoTSC_ReactAgent(text_corpus, llm_prompt, **agent_kwargs)
        case "zeroshot-rag" | "fewshot-rag":
            return NshotRAGAgent(text_corpus, llm_prompt, **agent_kwargs)
        case "cot-rag":
            return CoTRAGAgent(text_corpus, llm_prompt, **agent_kwargs)
        case "sufficient-context-autorater":
            return SufficientContextAutorater(text_corpus, llm_prompt, **agent_kwargs)
        case "llm-reranker":
            return LLMReranker(text_corpus, llm_prompt, **agent_kwargs)
        case "llm-rag-reranker":
            return LLMRAGReranker(text_corpus, llm_prompt, **agent_kwargs)
        case _:
            raise ValueError(f"Invalid method: {method}")
