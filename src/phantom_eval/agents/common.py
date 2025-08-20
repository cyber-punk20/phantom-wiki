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
from phantom_eval.prompts import LLMPrompt, SUFFICIENT_CONTEXT_AUTORATOR_EXAMPLE
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
    
    def get_RAG_evidence(self, question: str, step_round: int) -> str:
        return self._get_RAG_evidence(question, self.retriever_num_documents * step_round) 
    
    def get_RAG_evidence(self, question: str) -> str:
        return self._get_RAG_evidence(question, self.retriever_num_documents)

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
        sufficient_context_example: str = SUFFICIENT_CONTEXT_AUTORATOR_EXAMPLE,
        max_steps: int = 5,
        embedding_model_name: str = "",
        port: int = 8001,
        retriever_num_documents: int = 4,
        retrieval_method: str = "bm25"
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
        )
        self.sufficient_context_example = sufficient_context_example
        self.max_steps = max_steps
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
            Message(role="assistant", content=[ContentTextMessage(text=response_text)])
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
            evidence=self.evidence, example=self.sufficient_context_example, question=question
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
        total_usage: dict = {}
        while (self.step_round <= self.max_steps) and (not self.finished):
            try:
                response = await self._prompt_agent(llm_chat, question, inf_gen_config)
                total_usage = aggregate_usage([total_usage, response.usage])
                # Check if the context is sufficient
                if self._parse_response(response.pred):
                  self.finished = True
                self.step_round += 1
            except Exception:
                response = LLMChatResponse(
                    pred="", usage=total_usage, error=f"<agent_error>{traceback.format_exc()}</agent_error>"
                )
                break

        if (self.step_round > self.max_steps) and (not self.finished):
            response = LLMChatResponse(
                pred="",
                usage=total_usage,
                error=f"<agent_error>SufficientContextAutorater: max act steps ({self.max_steps})"
                "reached without finishing.</agent_error>",
            )

        return LLMChatResponse(pred=response.pred, usage=total_usage, error=response.error)




