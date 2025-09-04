"""
This module provides common agent components for phantom_eval including:

- Abstract `Agent` class for implementing evaluation methods (zeroshot, cot, react)
- `SCMixin` for self-consistency voting across multiple predictions
- `CustomEmbeddings` wrapper for model embeddings via local OpenAI API
- `RAGMixin` for retrieval-augmented generation
- Utility functions for evidence retrieval and Reasoning LLM names

The agents derived from `Agent` class can run evaluations on single question at a time or batches
using different LLM prompts and chat interfaces.
"""

import abc
import json
import logging
import subprocess
from collections import Counter
import traceback
from pprint import pformat

import openai
import pandas as pd
from flashrag.retriever import BM25Retriever, DenseRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from phantom_eval._types import ContentTextMessage, Conversation, LLMChatResponse, Message
from phantom_eval.gpu_utils import get_gpu_count
from phantom_eval.llm import InferenceGenerationConfig, LLMChat, aggregate_usage
from phantom_eval.prompts import LLMPrompt, SUFFICIENT_CONTEXT_AUTORATER_EXAMPLES, RERANKER_LLM_EXAMPLE, CONTEXT_FILTER_LLM_EXAMPLE
from phantom_eval.score import normalize_pred

logger = logging.getLogger(__name__)


class Agent(abc.ABC):
    """
    Abstract class for an agent that implements an evaluation method (e.g. zeroshot, cot, react)
    using the specified `LLMPrompt` and `LLMChat` objects.

    The agent can be run on a single question or on a batch of questions.
    """

    def __init__(self, text_corpus: pd.DataFrame, llm_prompt: LLMPrompt):
        """
        Args:
            text_corpus (pd.DataFrame): The text corpus to search for answers.
                Must contain two columns: 'title' and 'article'.
            llm_prompt (LLMPrompt): The prompt to be used by the agent.
        """
        self.text_corpus = text_corpus
        self.llm_prompt = llm_prompt
        self.agent_interactions: Conversation | list[Conversation] = None

    @abc.abstractmethod
    async def run(
        self, llm_chat: LLMChat, question: str, inf_gen_config: InferenceGenerationConfig, *args, **kwargs
    ) -> LLMChatResponse:
        """
        Run the agent with an `LLMChat` on a given question.

        Args:
            llm_chat (LLMChat): The LLMChat object to use for generating responses.
            question (str): The question to ask the agent.
            inf_gen_config (InferenceGenerationConfig): The inference generation config to use
                for generating responses.
        """

    @abc.abstractmethod
    async def batch_run(
        self,
        llm_chat: LLMChat,
        questions: list[str],
        inf_gen_config: InferenceGenerationConfig,
        *args,
        **kwargs,
    ) -> list[LLMChatResponse]:
        """
        Asynchronously run the agent with an `LLMChat` on a list of questions.

        Args:
            llm_chat (LLMChat): The LLMChat object to use for generating responses.
            questions (list[str]): The list of questions to ask the agent.
            inf_gen_config (InferenceGenerationConfig): The inference generation config to use
                for generating responses.
        """

    @abc.abstractmethod
    def _build_agent_prompt(self, question: str) -> str:
        """
        Builds and returns the agent prompt with the given question.
        The prompt may depend on the agent's internal state.
        """

    def reset(self) -> None:
        """
        Reset the agent to its initial state.
        """

class SCMixin:
    """
    Mixin class to implement self-consistency, i.e. take a majority vote over multiple predictions.

    Combine this with an agent class to implement self-consistency evaluation.
    """

    def __init__(self, num_votes: int, sep: str):
        """
        Args:
            num_votes (int): The number of votes to take for the majority vote.
            sep (str): The separator used to split the prediction.
        """
        self.num_votes = num_votes
        self.sep = sep

    def take_majority_vote(self, responses: list[LLMChatResponse], sep: str) -> LLMChatResponse:
        """
        Take the majority vote over all answers from the response predictions.

        Args:
            responses (list[LLMChatResponse]): List of response predictions.
                Each response pred may contain multiple answers e.g. A, B, C.
                So response preds can be like [[A<sep>B], [A<sep>B<sep>C]] where [A<sep>B] is the first
                response pred and [A<sep>B<sep>C] is the second response pred.
            sep (str): The separator used to split the prediction.

        Returns:
            LLMChatResponse: the majority vote as a single string of answers separated by <sep>
                (the output string is in `LLMChatResponse.pred`).
                If no answer has a majority vote, an error message is returned in `LLMChatResponse.error`.
        """
        n_preds = len(responses)
        preds: list[set[str]] = [normalize_pred(response.pred, sep) for response in responses]
        total_usage: dict = aggregate_usage([response.usage for response in responses])

        # Flatten the list of sets to a single list, e.g. becomes [A, B, A, B, C]
        all_answers: list[str] = [answer for pred in preds for answer in pred]
        vote_counts = Counter(all_answers)

        # Select all answers that have more than n_preds / 2 counts
        majority_responses = [answer for answer, count in vote_counts.items() if count > n_preds / 2]
        error = (
            None
            if len(majority_responses) > 0
            else f"<agent_error>No majority vote found in {vote_counts=}.</agent_error>"
        )

        majority_responses_str = sep.join(majority_responses)
        return LLMChatResponse(pred=majority_responses_str, usage=total_usage, error=error)


class CustomEmbeddings(Embeddings):
    """
    Wrapper class for model embeddings, accessed on vLLM via the local OpenAI API.
    """

    def __init__(self, client: openai.OpenAI):
        self.client = client
        self.model = self.client.models.list().data[0].id

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Return the embeddings of the input document texts. Each text is embedded as a list of floats.
        """
        return [obj.embedding for obj in self.client.embeddings.create(input=texts, model=self.model).data]

    def embed_query(self, text: str) -> list[float]:
        """
        Return the embedding of the input query text. The text is embedded as a list of floats.
        """
        return self.embed_documents([text])[0]


class RAGMixin:
    """
    Mixin class to implement RAG evaluation with a retriever.

    Combine this with an agent class to implement RAG evaluation with prompting techniques like zeroshot, cot.
    """

    # class variable to store the indices for each text corpus to avoid re-indexing across multiple instances
    # implemented as a dict of text corpus id -> (retriever, tokenizer)
    _indices = {}

    def __getstate__(self):
        """Custom getstate to handle unpickleable retriever object."""
        state = self.__dict__.copy()
        if "retriever" in state:
            del state["retriever"]
        return state

    def __setstate__(self, state):
        """Custom setstate to handle unpickleable retriever object."""
        self.__dict__.update(state)
        # Re-initialize retriever. The logic in __init__ will handle caching.
        RAGMixin.__init__(
            self,
            self.text_corpus,
            self.embedding_model_name,
            self.retriever_num_documents,
            self.port,
            self.retrieval_method,
            self.index_path,
            self.corpus_path,
        )

    def __init__(
        self,
        text_corpus: pd.DataFrame,
        embedding_model_name: str = None,
        retriever_num_documents: int = 4,
        port: int = 8001,
        retrieval_method: str = "bm25",
        index_path: str = None,
        corpus_path: str = None,
    ):
        """
        Args:
            text_corpus (pd.DataFrame): The text corpus containing documents in the "article" column.
            embedding_model_name (str): The embedding model name for dense and faiss retrieval methods.
                All embedding models available through huggingface and loadable by vLLM are supported.
                Defaults to None.
                NOTE: BM25 does not require passing an embedding model name.
            retriever_num_documents (int): Number of documents retrieved.
                Defaults to 4.
            retrieval_method (str): The retrieval method to use. Can be "bm25", "dense", or "faiss".
                Defaults to "bm25".
            retrieval_method (str): The retrieval method to use. Can be "bm25", "dense", or "faiss".
                Defaults to "bm25".
                https://github.com/kilian-group/phantom-wiki/wiki/RAG
                To build the index, please follow the instructions at
                https://github.com/kilian-group/phantom-wiki/wiki/RAG
                Defaults to None.
            corpus_path (str): The path to the corpus file for the BM25 or dense retriever.
                To build the corpus, please follow the instructions at
                https://github.com/kilian-group/phantom-wiki/wiki/RAG
                To build the corpus, please follow the instructions at
                https://github.com/kilian-group/phantom-wiki/wiki/RAG
                To build the corpus, please follow the instructions at
                https://github.com/kilian-group/phantom-wiki/wiki/RAG
                To build the corpus, please follow the instructions at
                https://github.com/kilian-group/phantom-wiki/wiki/RAG
                Defaults to None.
        """
        self.embedding_model_name = embedding_model_name
        self.retriever_num_documents = retriever_num_documents
        self.retrieval_method = retrieval_method
        self.text_corpus = text_corpus
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.port = port

        # TODO: deprecate the text_corpus argument. The new workflow is to index the text corpus separately,
        # then pass the index path to the constructor.
        # Use the following arguments to check for existing retriever objects.
        if self.retrieval_method in ["bm25", "dense"]:
            key = (
                self.retrieval_method,
                self.embedding_model_name,
                index_path,
                corpus_path,
            )
            if key in self._indices:
                logger.debug("Using an existing retriever object...")
                self.retriever = self._indices[key]
                return

            match self.retrieval_method:
                case "bm25":
                    bm25_config = {
                        "retrieval_method": "bm25",
                        "retrieval_topk": retriever_num_documents,
                        "index_path": index_path,
                        "corpus_path": corpus_path,
                        "silent_retrieval": True,
                        "bm25_backend": "bm25s",
                        # Additional retriever features
                        # See https://github.com/bogoliubon/FlashRAG/blob/
                        # 5f6eeafbf86c959475c4989b699666e5ccaa1a21/docs/
                        # original_docs/basic_usage.md#additional-features-of-the-retriever
                        "save_retrieval_cache": False,
                        "retrieval_cache_path": "~",
                        "use_retrieval_cache": False,
                        "use_reranker": False,
                    }
                    logger.info("Initializing BM25 retriever...")
                    self.retriever = BM25Retriever(config=bm25_config)
                    logger.info(f"Retriever config: {pformat(self.retriever.config)}")
                    # Store the retriever object in the _indices dict for reuse across instances
                    self._indices[key] = self.retriever

                case "dense":
                    # Default: https://github.com/bogoliubon/FlashRAG/blob/
                    # 5f6eeafbf86c959475c4989b699666e5ccaa1a21/flashrag/config/basic_config.yaml#L49
                    dense_config = {
                        "retrieval_method": "dense",
                        "retrieval_topk": retriever_num_documents,
                        "index_path": index_path,
                        "corpus_path": corpus_path,
                        "retrieval_model_path": embedding_model_name,
                        "retrieval_query_max_length": 128,
                        "retrieval_pooling_method": "mean",
                        "retrieval_use_fp16": True,
                        "retrieval_batch_size": 16,
                        "use_sentence_transformer": True,
                        "faiss_gpu": False,
                        "silent_retrieval": True,
                        # Additional retriever features
                        # See https://github.com/bogoliubon/FlashRAG/blob/
                        # 5f6eeafbf86c959475c4989b699666e5ccaa1a21/docs/
                        # original_docs/basic_usage.md#additional-features-of-the-retriever
                        "save_retrieval_cache": False,
                        "retrieval_cache_path": "~",
                        "use_retrieval_cache": False,
                        "use_reranker": False,
                        "instruction": "~",
                    }
                    self.retriever = DenseRetriever(config=dense_config)
                    logger.info(f"Retriever config: {pformat(self.retriever.config)}")
                    # Store the retriever object in the _indices dict for reuse across instances
                    self._indices[key] = self.retriever

        else:
            texts = text_corpus["article"].tolist()

            # Launch server on the last GPU
            subprocess.call(
                [
                    "./src/phantom_eval/launch_embedding_server.sh",
                    embedding_model_name,
                    str(port),
                    str(get_gpu_count() - 1),
                ]
            )

            # Embed documents and build retriever
            BASE_URL = f"http://0.0.0.0:{port}/v1"
            API_KEY = "token-abc123"
            client = openai.OpenAI(
                base_url=BASE_URL,
                api_key=API_KEY,
            )
            embeddings = CustomEmbeddings(client)
            vectorstore = FAISS.from_texts(texts, embeddings)
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_num_documents})
    
    def get_RAG_evidence(self, question: str, step_round: int = 1) -> str:
        return self._get_RAG_evidence(question, self.retriever_num_documents * step_round)

    def _get_RAG_evidence(self, question: str, retriever_num_documents: int) -> str:
        """
        Returns retrieved articles given the question from the text corpus.
        The retrieved articles are concatenated as a string.
        """
        if self.retrieval_method in ["bm25", "dense"]:
            docs = self.retriever._search(question, num=retriever_num_documents, return_score=False)
            docs = [doc["contents"] for doc in docs]
        else:
            docs = [doc.page_content for doc in self.retriever.invoke(question)]
        return "\n================\n\n".join(docs)


def get_all_evidence(text_corpus: pd.DataFrame) -> str:
    """
    Return all articles in the text corpus concatenated as a string.
    """
    return "\n================\n\n".join(text_corpus["article"])


def parse_prolog_query(pred: str) -> str:
    """
    Parse the prolog query from the prediction.
    """
    return pred.replace("`", "").strip().split("\n")[-1]


class SufficientContextAutorater(Agent, RAGMixin):
    """
    SufficientContextAutorater will keep retrieving documents until receiving a sufficient context signal from the LLM.
    """
    def __init__(
        self,
        text_corpus: pd.DataFrame,
        llm_prompt: LLMPrompt,
        sufficient_context_examples: str = SUFFICIENT_CONTEXT_AUTORATER_EXAMPLES,
        sca_max_steps: int = 5,
        embedding_model_name: str = "",
        port: int = 8001,
        retriever_num_documents: int = 4,
        retrieval_method: str = "bm25",
        index_path: str = None,
        corpus_path: str = None,
    ):
        """
        Args:
            text_corpus (pd.DataFrame): The text corpus to search for answers.
                Must contain two columns: 'title' and 'article'.
            llm_prompt (LLMPrompt): The prompt to be used by the agent.
            sufficient_context_example (str): Prompt examples to include in agent prompt.
                Defaults to "".
            embedding_model_name (str): The name of the embedding model to use for retrieval.
                Defaults to "".
            retriever_num_documents (int): The number of documents to retrieve.
                Defaults to 4.
            port (int): The port to use for the retriever.
                Defaults to 8001.
            retrieval_method (str): The retrieval method to use. Can be "faiss", "bm25" or "dense".
                Defaults to "bm25".
            index_path (str): The path to the index file for the BM25 or dense retriever.
                Defaults to None.
            corpus_path (str): The path to the corpus file for the BM25 or dense retriever.
                Defaults to None.

        """
        super().__init__(text_corpus, llm_prompt)
        RAGMixin.__init__(
            self,
            text_corpus,
            embedding_model_name,
            retriever_num_documents,
            port,
            retrieval_method,
            index_path,
            corpus_path,
        )
        self.sufficient_context_examples = sufficient_context_examples
        self.sca_max_steps = sca_max_steps
        self.reset()

    def reset(self) -> None:
        self.step_round = 1
        self.finished = False
        self.evidence: str = ""
        self.agent_interactions: Conversation = Conversation(messages=[])
  
    def _parse_response(self, response_text: str) -> bool:
        """
        Parses the LLM's response to check for context sufficiency.

        Args:
            response_text: The text generated by the language model.

        Returns:
            True if the context is sufficient, False otherwise.
        """
        self.agent_interactions.messages.append(
            Message(role="assistant", content=[ContentTextMessage(text=f"SAC round {self.step_round}"),
                                               ContentTextMessage(text=response_text)])
        )
        try:
            # Find the JSON part of the response
            json_str = response_text.split("### JSON")[-1].strip()
            # Parse the JSON
            json_obj = json.loads(json_str)
            # Check for the sufficiency signal
            return json_obj.get("Sufficient Context") == 1
        except (json.JSONDecodeError, IndexError):
            # If parsing fails or the format is wrong, assume context is not sufficient
            return False
    
    async def batch_run(
        self,
        llm_chat: LLMChat,
        questions: list[str],
        inf_gen_config: InferenceGenerationConfig,
        *args,
        **kwargs,
    ) -> list[LLMChatResponse]:
        raise NotImplementedError("Batch run is not supported for SufficientContextAutorater.")
    
    def _build_agent_prompt(self, question: str) -> str:
        # Retrieve relevant context
        self.evidence = self.get_RAG_evidence(question, self.step_round)
        return self.llm_prompt.get_prompt().format(
            evidence=self.evidence, examples=self.sufficient_context_examples, question=question
        )

    async def _prompt_agent(
        self,
        llm_chat: LLMChat,
        question: str,
        inf_gen_config: InferenceGenerationConfig,
    ) -> LLMChatResponse:
        """
        Prompts the LLM with the agent's current prompt (created from question, scratchpad,
        and `leading_llm_prompt`). The `leading_llm_prompt` is not part of the scratchpad,
        but is used to indicate the current step. For example, "Action 2: ".

        Args:
            llm_chat (LLMChat): The LLMChat object to use for generating responses.
            question (str): The question to ask the agent.
            inf_gen_config (InferenceGenerationConfig): The inference generation config to use
                for generating responses.
        """
        # Put the full scratchpad in the prompt and ask the LLM to generate.
        # All of the back and forth conversation so far becomes the user prompt.
        user_message: str = self._build_agent_prompt(question)
        self.agent_interactions.messages.append(
            Message(role="user", content=[ContentTextMessage(text=user_message)])
        )
        conv: Conversation = Conversation(
            messages=[
                Message(role="user", content=[ContentTextMessage(text=user_message)])
            ]
        )
        response: LLMChatResponse = await llm_chat.generate_response(conv, inf_gen_config)
        return response

    async def run(
        self,
        llm_chat: LLMChat,
        question: str,
        inf_gen_config: InferenceGenerationConfig,
        *args,
        **kwargs,
    ) -> LLMChatResponse:
        logger.debug(f"\n\t>>> question: {question}\n")
        logger.debug(f"\n\t>>> sca_max_steps: {self.sca_max_steps}\n")

        total_usage: dict = {}
        sca_signals = []
        while (self.step_round <= self.sca_max_steps) and (not self.finished):
            logger.debug(f"\n\t>>> step_round: {self.step_round}\n")
            try:
                response = await self._prompt_agent(llm_chat, question, inf_gen_config)
                total_usage = aggregate_usage([total_usage, response.usage])
                # Check if the context is sufficient
                if self._parse_response(response.pred):
                  self.finished = True
                self.step_round += 1
                sca_signals.append(self.finished)
            except Exception:
                response = LLMChatResponse(
                    pred="", usage=total_usage, error=f"<agent_error>{traceback.format_exc()}</agent_error>"
                )
                break
        logger.debug(f"\n\t>>> evidence:\n{self.evidence}\n")
        logger.debug(f"\n\t>>> finished: {self.finished}\n")
        logger.debug(f"\n\t>>> agent_interactions: {self.agent_interactions}\n")
        logger.debug(f"\n\t>>> total_usage: {total_usage}\n")
        logger.debug(f"\n\t>>> response: {response}\n")

        return LLMChatResponse(pred=self.evidence, usage=total_usage, sca_signal=self.finished, sca_signals=sca_signals)

def parse_llm_reranker_response(response_text: str) -> list[str]:
    """
    Parses the LLM's response to extract the ranked list of titles.
    Args:
        response_text: The text generated by the language model.

    Returns:
        A list of article titles in ranked order.
    """
    try:
        # Find the JSON part of the response
        json_start = response_text.index("{")
        json_end = response_text.rfind("}") + 1
        json_str = response_text[json_start: json_end]
        # Parse the JSON
        json_obj = json.loads(json_str)
        # Check for the 'rankings' key
        return json_obj.get("rankings", [])
    except (json.JSONDecodeError, IndexError, ValueError):
        # If parsing fails or the format is wrong, return an empty list
        logger.warning(f"Could not parse LLM reranker response: {response_text}")
        return []

def rerank_evidence(text_corpus: pd.DataFrame, ranking_list: list[str]) -> str:
    """
    Reranks articles from the text_corpus based on the provided ranking list of titles.

    Args:
        text_corpus (pd.DataFrame): The text corpus containing articles.
        ranking_list (list[str]): A list of article titles in the desired order.

    Returns:
        str: The reranked evidence as a single string, with articles joined by a delimiter.
    """
    # Set 'title' as the index to easily select and reorder rows.
    corpus_by_title = text_corpus.set_index("title")

    # Filter ranking_list to only include titles present in the corpus.
    valid_titles = [title for title in ranking_list if title in corpus_by_title.index]

    # Reorder the DataFrame based on the valid_titles.
    reranked_corpus = corpus_by_title.loc[valid_titles]

    # Join the 'article' column of the reranked DataFrame.
    return "\n================\n\n".join(reranked_corpus["article"])

def filter_evidence(text_corpus: pd.DataFrame, relevant_references: list[str]) -> str:
    """
    Filter articles from the text_corpus that does not have titles in relevant_references.

    Args:
        text_corpus (pd.DataFrame): The text corpus containing articles.
        relevant_references (list[str]): A list of relevant article titles.

    Returns:
        str: The relevant evidences as a single string, with articles joined by a delimiter.
    """
    # Set 'title' as the index to easily select and reorder rows.
    corpus_by_title = text_corpus.set_index("title")

    # Filter relevant_references to only include titles present in the corpus.
    valid_titles = [title for title in relevant_references if title in corpus_by_title.index]

    final_corpus = corpus_by_title.loc[valid_titles]

    # Join the 'article' column of the reranked DataFrame.
    return "\n================\n\n".join(final_corpus["article"])


def parse_context_filter_response(response_text: str) -> list[str]:
    """
    Parses the LLM's response to extract the relevant list of titles.
    Args:
        response_text: The text generated by the language model.

    Returns:
        A list of relevant article titles.
    """
    try:
        # Find the JSON part of the response
        json_start = response_text.index("{")
        json_end = response_text.rfind("}") + 1
        json_str = response_text[json_start: json_end]
        # Parse the JSON
        json_obj = json.loads(json_str)
        # Check for the 'rankings' key
        return json_obj.get("relevant_references", [])
    except (json.JSONDecodeError, IndexError, ValueError):
        # If parsing fails or the format is wrong, return an empty list
        logger.warning(f"Could not parse LLM context filter response: {response_text}")
        return []

class LLMRAGReranker(RAGMixin):
    def __init__(
        self,
        text_corpus: pd.DataFrame,
        llm_prompt: LLMPrompt,
        fewshot_examples: str = "",
        embedding_model_name: str = "",
        retriever_num_documents: int = 32,
        port: int = 8001,
        retrieval_method: str = "bm25",
        index_path: str = None,
        corpus_path: str = None,
        reranker_llm_example: str = RERANKER_LLM_EXAMPLE,
    ):
        RAGMixin.__init__(
            self,
            text_corpus,
            embedding_model_name,
            retriever_num_documents,
            port,
            retrieval_method,
            index_path,
            corpus_path,
        )
        self.reranker_llm_example = reranker_llm_example
        self.reranker_llm_prompt = llm_prompt
    
    def combine_llm_reranker_evidence_and_question(self, evidence: str, question: str) -> str:
        return self.reranker_llm_prompt.get_prompt().format(evidence=evidence, question=question, example=self.reranker_llm_example)
    
    def build_llm_reranker_agent_prompt(self, question: str) -> str:
        # Retrieve relevant context
        evidence = self.get_RAG_evidence(question)
        return self.combine_llm_reranker_evidence_and_question(evidence, question)
    
    async def run(
        self,
        llm_chat: LLMChat,
        question: str,
        inf_gen_config: InferenceGenerationConfig,
        *args,
        **kwargs,
    ) -> LLMChatResponse:
        logger.debug(f"\n\t>>> question: {question}\n")
        prompt = self.build_llm_reranker_agent_prompt(question)
        conv = Conversation(messages=[Message(role="user", content=[ContentTextMessage(text=prompt)])])
        self.agent_interactions = conv

        # Generate response
        inf_gen_config = inf_gen_config.model_copy(update=dict(stop_sequences=[]), deep=True)
        response = await llm_chat.generate_response(conv, inf_gen_config)
        logger.debug(f"\n\t>>>LLMReranker response: {response}\n")
        ranking_list = parse_llm_reranker_response(response.pred)
        logger.debug(f"\n\t>>>ranking_list: {ranking_list}\n")
        if len(ranking_list) == 0:
            reranked_evidence = self.get_RAG_evidence(question)
            has_valid_reranker_result = False
        else:
            reranked_evidence = rerank_evidence(self.text_corpus, ranking_list)
            has_valid_reranker_result = True


        # Update agent's conversation
        self.agent_interactions.messages.append(
            Message(role="assistant", content=[ContentTextMessage(text=str(response.pred))])
        )
        return LLMChatResponse(pred=reranked_evidence, usage=response.usage, has_valid_reranker_result=has_valid_reranker_result)
    
    async def batch_run(
        self,
        llm_chat: LLMChat,
        questions: list[str],
        inf_gen_config: InferenceGenerationConfig,
        *args,
        **kwargs,
    ) -> list[LLMChatResponse]:
        logger.debug(f"\n\t>>> questions: {questions}\n")

        # Create a conversation for each user prompt, and initialize agent interactions
        prompts: list[str] = [self.build_llm_reranker_agent_prompt(question) for question in questions]
        convs = [
            Conversation(messages=[Message(role="user", content=[ContentTextMessage(text=prompt)])])
            for prompt in prompts
        ]
        self.agent_interactions = convs

        # Generate response
        inf_gen_config = inf_gen_config.model_copy(update=dict(stop_sequences=[]), deep=True)
        responses = await llm_chat.batch_generate_response(convs, inf_gen_config)

        # Add the responses to the agent's conversations
        for i, response in enumerate(responses):
            self.agent_interactions[i].messages.append(
                Message(role="assistant", content=[ContentTextMessage(text=response.pred)])
            )
            ranking_list = parse_llm_reranker_response(response.pred)
            if len(ranking_list) == 0:
                reranked_evidence = self.get_RAG_evidence(questions[i])
                has_valid_reranker_result = False
            else:
                reranked_evidence = rerank_evidence(self.text_corpus, ranking_list)
                has_valid_reranker_result = True
        return [LLMChatResponse(pred=reranked_evidence, usage=response.usage, has_valid_reranker_result=has_valid_reranker_result) for response in responses]


class LLMReranker():
    def __init__(
            self, 
            text_corpus: pd.DataFrame, 
            llm_prompt: LLMPrompt,
            reranker_llm_example: str = RERANKER_LLM_EXAMPLE,
        ):
        """
        Args:
            text_corpus (pd.DataFrame): The text corpus to search for answers.
                Must contain two columns: 'title' and 'article'.
            llm_prompt (LLMPrompt): The prompt to be used by the agent.
            reranker_llm_example (str): Prompt examples to include in agent prompt.
        """
        self.reranker_text_corpus = text_corpus
        self.reranker_llm_prompt = llm_prompt
        self.reranker_llm_example = reranker_llm_example
    
    def combine_llm_reranker_evidence_and_question(self, evidence: str, question: str) -> str:
        return self.reranker_llm_prompt.get_prompt().format(evidence=evidence, question=question, example=self.reranker_llm_example)
    
    def build_llm_reranker_agent_prompt(self, question: str) -> str:
        # Retrieve relevant context
        evidence = get_all_evidence(self.reranker_text_corpus)
        return self.combine_llm_reranker_evidence_and_question(evidence, question)
    
    async def run(
        self,
        llm_chat: LLMChat,
        question: str,
        inf_gen_config: InferenceGenerationConfig,
        *args,
        **kwargs,
    ) -> LLMChatResponse:
        logger.debug(f"\n\t>>> question: {question}\n")
        prompt = self.build_llm_reranker_agent_prompt(question)
        conv = Conversation(messages=[Message(role="user", content=[ContentTextMessage(text=prompt)])])
        self.agent_interactions = conv

        # Generate response
        inf_gen_config = inf_gen_config.model_copy(update=dict(stop_sequences=[]), deep=True)
        response = await llm_chat.generate_response(conv, inf_gen_config)
        ranking_list = parse_llm_reranker_response(response.pred)
        logger.debug(f"\n\t>>>ranking_list: {ranking_list}\n")
        if len(ranking_list) == 0:
            reranked_evidence = get_all_evidence(self.reranker_text_corpus)
            has_valid_reranker_result = False
        else:
            reranked_evidence = rerank_evidence(self.reranker_text_corpus, ranking_list)
            has_valid_reranker_result = True




        # Update agent's conversation
        self.agent_interactions.messages.append(
            Message(role="assistant", content=[ContentTextMessage(text=str(response.pred))])
        )
        return LLMChatResponse(pred=reranked_evidence, usage=response.usage, has_valid_reranker_result=has_valid_reranker_result)
    
    async def batch_run(
        self,
        llm_chat: LLMChat,
        questions: list[str],
        inf_gen_config: InferenceGenerationConfig,
        *args,
        **kwargs,
    ) -> list[LLMChatResponse]:
        logger.debug(f"\n\t>>> questions: {questions}\n")

        # Create a conversation for each user prompt, and initialize agent interactions
        prompts: list[str] = [self.build_llm_reranker_agent_prompt(question) for question in questions]
        convs = [
            Conversation(messages=[Message(role="user", content=[ContentTextMessage(text=prompt)])])
            for prompt in prompts
        ]
        self.agent_interactions = convs

        # Generate response
        inf_gen_config = inf_gen_config.model_copy(update=dict(stop_sequences=[]), deep=True)
        responses = await llm_chat.batch_generate_response(convs, inf_gen_config)

        # Add the responses to the agent's conversations
        llm_responses = []
        for i, response in enumerate(responses):
            self.agent_interactions[i].messages.append(
                Message(role="assistant", content=[ContentTextMessage(text=response.pred)])
            )
            ranking_list = parse_llm_reranker_response(response.pred)
            if len(ranking_list) == 0:
                reranked_evidence = get_all_evidence(self.reranker_text_corpus)
                has_valid_reranker_result = False
            else:
                reranked_evidence = rerank_evidence(self.reranker_text_corpus, ranking_list)
                has_valid_reranker_result = True
            llm_responses.append(LLMChatResponse(pred=reranked_evidence, usage=response.usage, convs=self.agent_interactions[i], has_valid_reranker_result=has_valid_reranker_result))
        return llm_responses
    

class LLMFilter(Agent):
    def __init__(
            self, 
            text_corpus: pd.DataFrame, 
            llm_prompt: LLMPrompt,
            context_filter_llm_example: str = CONTEXT_FILTER_LLM_EXAMPLE,
        ):
        """
        Args:
            text_corpus (pd.DataFrame): The text corpus to search for answers.
                Must contain two columns: 'title' and 'article'.
            llm_prompt (LLMPrompt): The prompt to be used by the agent.
            context_filter_llm_example (str): Prompt examples to include in agent prompt.
        """
        super().__init__(text_corpus, llm_prompt)
        self.context_filter_llm_example = context_filter_llm_example
    
    def combine_evidence_and_question(self, evidence: str, question: str) -> str:
        return self.llm_prompt.get_prompt().format(evidence=evidence, question=question, example=self.context_filter_llm_example)
    
    def _build_agent_prompt(self, question: str) -> str:
        # Retrieve relevant context
        evidence = get_all_evidence(self.text_corpus)
        return self.combine_evidence_and_question(evidence, question)
    
    async def run(
        self,
        llm_chat: LLMChat,
        question: str,
        inf_gen_config: InferenceGenerationConfig,
        *args,
        **kwargs,
    ) -> LLMChatResponse:
        logger.debug(f"\n\t>>> question: {question}\n")
        prompt = self._build_agent_prompt(question)
        conv = Conversation(messages=[Message(role="user", content=[ContentTextMessage(text=prompt)])])
        self.agent_interactions = conv

        # Generate response
        inf_gen_config = inf_gen_config.model_copy(update=dict(stop_sequences=[]), deep=True)
        response = await llm_chat.generate_response(conv, inf_gen_config)
        relevant_evidences = parse_context_filter_response(response.pred)
        logger.debug(f"\n\t>>>relevant_evidences: {relevant_evidences}\n")
        if len(relevant_evidences) == 0:
            filtered_evidence = get_all_evidence(self.text_corpus)
            has_valid_filter_result = False
        else:
            filtered_evidence = filter_evidence(self.text_corpus, relevant_evidences)
            has_valid_filter_result = True




        # Update agent's conversation
        self.agent_interactions.messages.append(
            Message(role="assistant", content=[ContentTextMessage(text=str(response.pred))])
        )
        return LLMChatResponse(pred=filtered_evidence, usage=response.usage, has_valid_filter_result=has_valid_filter_result)
    
    async def batch_run(
        self,
        llm_chat: LLMChat,
        questions: list[str],
        inf_gen_config: InferenceGenerationConfig,
        *args,
        **kwargs,
    ) -> list[LLMChatResponse]:
        logger.debug(f"\n\t>>> questions: {questions}\n")

        # Create a conversation for each user prompt, and initialize agent interactions
        prompts: list[str] = [self._build_agent_prompt(question) for question in questions]
        convs = [
            Conversation(messages=[Message(role="user", content=[ContentTextMessage(text=prompt)])])
            for prompt in prompts
        ]
        self.agent_interactions = convs

        # Generate response
        inf_gen_config = inf_gen_config.model_copy(update=dict(stop_sequences=[]), deep=True)
        responses = await llm_chat.batch_generate_response(convs, inf_gen_config)

        # Add the responses to the agent's conversations
        llm_responses = []
        for i, response in enumerate(responses):
            self.agent_interactions[i].messages.append(
                Message(role="assistant", content=[ContentTextMessage(text=response.pred)])
            )
            relevant_evidences = parse_context_filter_response(response.pred)
            if len(relevant_evidences) == 0:
                filtered_evidence = get_all_evidence(self.text_corpus)
                has_valid_filter_result = False
            else:
                filtered_evidence = filter_evidence(self.text_corpus, relevant_evidences)
                has_valid_filter_result = True
            llm_responses.append(LLMChatResponse(pred=filtered_evidence, usage=response.usage, convs=self.agent_interactions[i], has_valid_filter_result=has_valid_filter_result))
        return llm_responses
    

class LLMRAGFilter(Agent, RAGMixin):
    def __init__(
        self,
        text_corpus: pd.DataFrame,
        llm_prompt: LLMPrompt,
        fewshot_examples: str = "",
        embedding_model_name: str = "",
        retriever_num_documents: int = 32,
        port: int = 8001,
        retrieval_method: str = "bm25",
        index_path: str = None,
        corpus_path: str = None,
        reranker_llm_example: str = RERANKER_LLM_EXAMPLE,
    ):
        RAGMixin.__init__(
            self,
            text_corpus,
            embedding_model_name,
            retriever_num_documents,
            port,
            retrieval_method,
            index_path,
            corpus_path,
        )
        Agent.__init__(self, text_corpus, llm_prompt)
        self.context_filter_llm_example = CONTEXT_FILTER_LLM_EXAMPLE
    
    def combine_evidence_and_question(self, evidence: str, question: str) -> str:
        return self.llm_prompt.get_prompt().format(evidence=evidence, question=question, example=self.context_filter_llm_example)
    
    def _build_agent_prompt(self, question: str) -> str:
        # Retrieve relevant context
        evidence = self.get_RAG_evidence(question)
        return self.combine_evidence_and_question(evidence, question)
    
    async def run(
        self,
        llm_chat: LLMChat,
        question: str,
        inf_gen_config: InferenceGenerationConfig,
        *args,
        **kwargs,
    ) -> LLMChatResponse:
        logger.debug(f"\n\t>>> question: {question}\n")
        prompt = self._build_agent_prompt(question)
        conv = Conversation(messages=[Message(role="user", content=[ContentTextMessage(text=prompt)])])
        self.agent_interactions = conv

        # Generate response
        inf_gen_config = inf_gen_config.model_copy(update=dict(stop_sequences=[]), deep=True)
        response = await llm_chat.generate_response(conv, inf_gen_config)
        relevant_evidences = parse_context_filter_response(response.pred)
        logger.debug(f"\n\t>>>relevant_evidences: {relevant_evidences}\n")
        if len(relevant_evidences) == 0:
            filtered_evidence = self.get_RAG_evidence(question)
            has_valid_filter_result = False
        else:
            filtered_evidence = filter_evidence(self.text_corpus, relevant_evidences)
            has_valid_filter_result = True




        # Update agent's conversation
        self.agent_interactions.messages.append(
            Message(role="assistant", content=[ContentTextMessage(text=str(response.pred))])
        )
        return LLMChatResponse(pred=filtered_evidence, usage=response.usage, has_valid_filter_result=has_valid_filter_result)
    
    async def batch_run(
        self,
        llm_chat: LLMChat,
        questions: list[str],
        inf_gen_config: InferenceGenerationConfig,
        *args,
        **kwargs,
    ) -> list[LLMChatResponse]:
        logger.debug(f"\n\t>>> questions: {questions}\n")

        # Create a conversation for each user prompt, and initialize agent interactions
        prompts: list[str] = [self._build_agent_prompt(question) for question in questions]
        convs = [
            Conversation(messages=[Message(role="user", content=[ContentTextMessage(text=prompt)])])
            for prompt in prompts
        ]
        self.agent_interactions = convs

        # Generate response
        inf_gen_config = inf_gen_config.model_copy(update=dict(stop_sequences=[]), deep=True)
        responses = await llm_chat.batch_generate_response(convs, inf_gen_config)

        # Add the responses to the agent's conversations
        llm_responses = []
        for i, response in enumerate(responses):
            self.agent_interactions[i].messages.append(
                Message(role="assistant", content=[ContentTextMessage(text=response.pred)])
            )
            relevant_evidences = parse_context_filter_response(response.pred)
            if len(relevant_evidences) == 0:
                filtered_evidence = self.get_RAG_evidence(questions[i])
                has_valid_filter_result = False
            else:
                filtered_evidence = filter_evidence(self.text_corpus, relevant_evidences)
                has_valid_filter_result = True
            llm_responses.append(LLMChatResponse(pred=filtered_evidence, usage=response.usage, convs=self.agent_interactions[i], has_valid_filter_result=has_valid_filter_result))
        return llm_responses