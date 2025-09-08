"""
This module implements N-shot agents for phantom_eval, such as zeroshot and fewshot methods.
The module contains three main agent classes:

- `NshotAgent`: Base agent implementing zero-shot and few-shot evaluation based on provided examples
- `NshotSCAgent`: Extends NshotAgent with self-consistency through majority voting
- `NshotRAGAgent`: Extends NshotAgent with Retrieval Augmented Generation (RAG)
"""

import asyncio
import logging
import re
import traceback

import pandas as pd

import phantom_eval.constants as constants
from phantom_eval._types import ContentTextMessage, Conversation, LLMChatResponse, Message
from phantom_eval.agents.common import Agent, RAGMixin, SCMixin, get_all_evidence, get_evidence_from_sca_context_corpus, parse_prolog_query
from phantom_eval.llm import InferenceGenerationConfig, LLMChat
from phantom_eval.prompts import LLMPrompt
from phantom_eval.utils import load_sca_context_corpus

logger = logging.getLogger(__name__)


class NshotAgent(Agent):
    """
    Agent that implements zero-shot and few-shot evaluation, depending on whether
    `fewshot_examples` is provided.
    """

    def __init__(
        self,
        text_corpus: pd.DataFrame,
        llm_prompt: LLMPrompt,
        fewshot_examples: str = "",
        prolog_query: bool = False,
    ):
        """
        Args:
            text_corpus (pd.DataFrame): The text corpus to search for answers.
                Must contain two columns: 'title' and 'article'.
            llm_prompt (LLMPrompt): The prompt to be used by the agent.
            fewshot_examples (str): Few-shot prompt examples to include in agent prompt.
                If "", the agent is zero-shot. Defaults to "".
            prolog_query (bool): Whether to use the prompt for eliciting prolog queries from LLMs.
                Passed on to `LLMPrompt.get_prompt()`. Defaults to False.
        """
        super().__init__(text_corpus, llm_prompt)
        self.fewshot_examples = fewshot_examples
        self.prolog_query = prolog_query

    def combine_evidence_and_question(self, evidence: str, question: str) -> str:
        """
        Combine the evidence and question to form the agent prompt using `self.llm_prompt`.
        """
        if self.fewshot_examples:  # Few-shot
            return self.llm_prompt.get_prompt(self.prolog_query).format(
                evidence=evidence, examples=self.fewshot_examples, question=question
            )
        else:  # Zero-shot
            return self.llm_prompt.get_prompt(self.prolog_query).format(evidence=evidence, question=question)

    def _build_agent_prompt(self, question: str) -> str:
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

        # Create a conversation with 1 user prompt and initialize agent interactions
        prompt = self._build_agent_prompt(question)
        conv = Conversation(messages=[Message(role="user", content=[ContentTextMessage(text=prompt)])])
        self.agent_interactions = conv

        # Generate response
        inf_gen_config = inf_gen_config.model_copy(update=dict(stop_sequences=[]), deep=True)
        response = await llm_chat.generate_response(conv, inf_gen_config)

        # Update agent's conversation
        self.agent_interactions.messages.append(
            Message(role="assistant", content=[ContentTextMessage(text=response.pred)])
        )

        if self.prolog_query:
            response.pred = parse_prolog_query(response.pred)
            return response

        try:
            pred = NshotAgent.parse_answer(response.pred)
            error = None
        except Exception as e:
            pred = ""
            error = f"<agent_error>{traceback.format_exc()}</agent_error>"
            error = f"<agent_error>{e}</agent_error>"
        return LLMChatResponse(pred=pred, usage=response.usage, error=error)

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
        for i, response in enumerate(responses):
            self.agent_interactions[i].messages.append(
                Message(role="assistant", content=[ContentTextMessage(text=response.pred)])
            )
            if self.prolog_query:
                responses[i].pred = parse_prolog_query(response.pred)

        parsed_responses: list[LLMChatResponse] = []
        for response in responses:
            try:
                pred = NshotAgent.parse_answer(response.pred)
                error = None
            except Exception as e:
                pred = ""
                error = f"<agent_error>{traceback.format_exc()}</agent_error>"
                error = f"<agent_error>{e}</agent_error>"
            parsed_responses.append(LLMChatResponse(pred=pred, usage=response.usage, error=error))
        return parsed_responses

    @classmethod
    def parse_answer(cls, pred: str) -> str:
        """
        Parse the response to extract the answer using regex.
        First try to parse the response as a thinking answer (of the form: "</think> ...").
        If it fails, then return the original response.
        """
        pattern = r"</think>\s*(.+)"
        m = re.search(pattern, pred)
        if m:
            return m.group(1)
        else:
            return pred


class NshotSCAgent(NshotAgent, SCMixin):
    """
    Agent to implement zeroshot-sc and fewshot-sc evaluation with majority vote (self-consistency).
    """

    def __init__(
        self,
        text_corpus: pd.DataFrame,
        llm_prompt: LLMPrompt,
        fewshot_examples: str = "",
        num_votes: int = 3,
        sep: str = constants.answer_sep,
    ):
        """
        Args:
            text_corpus (pd.DataFrame): The text corpus to search for answers.
                Must contain two columns: 'title' and 'article'.
            llm_prompt (LLMPrompt): The prompt to be used by the agent.
            fewshot_examples (str): Few-shot prompt examples to include in agent prompt.
                If "", the agent is zero-shot. Defaults to "".
            num_votes (int): The number of votes to take for the majority vote.
                Defaults to 3.
            sep (str): The separator used to split the prediction.
                Defaults to `constants.answer_sep`.
        """
        NshotAgent.__init__(self, text_corpus, llm_prompt, fewshot_examples)
        SCMixin.__init__(self, num_votes, sep)

    async def run(
        self, llm_chat: LLMChat, question: str, inf_gen_config: InferenceGenerationConfig, *args, **kwargs
    ) -> LLMChatResponse:
        # Relies on the implementation of run in the subclass
        responses: list[LLMChatResponse] = asyncio.gather(
            *[super().run(llm_chat, question, inf_gen_config, *args, **kwargs) for _ in range(self.num_votes)]
        )
        return self.take_majority_vote(responses, self.sep)

    async def batch_run(
        self,
        llm_chat: LLMChat,
        questions: list[str],
        inf_gen_config: InferenceGenerationConfig,
        *args,
        **kwargs,
    ) -> list[LLMChatResponse]:
        # Relies on the implementation of batch_run in the subclass
        responses: list[list[LLMChatResponse]] = [
            await super().batch_run(llm_chat, questions, inf_gen_config, *args, **kwargs)
            for _ in range(self.num_votes)
        ]  # shape (num_votes, num_questions)
        # Take majority vote for each question, so transpose the responses list
        transposed_responses = [list(responses_each_question) for responses_each_question in zip(*responses)]
        return [
            self.take_majority_vote(responses_each_question, self.sep)
            for responses_each_question in transposed_responses
        ]


class NshotRAGAgent(NshotAgent, RAGMixin):
    """
    Agent to implement zeroshot-rag and fewshot-rag evaluation where the
    agent uses retriever to create evidence.
    """

    def __init__(
        self,
        text_corpus: pd.DataFrame,
        llm_prompt: LLMPrompt,
        fewshot_examples: str = "",
        embedding_model_name: str = "whereisai/uae-large-v1",
        retriever_num_documents: int = 4,
        port: int = 8001,
        retrieval_method: str = "bm25",
        index_path: str = None,
        corpus_path: str = None,
        corpus_name: str = None,
        vector_distance_threshold: float = None,
    ):
        """
        Args:
            text_corpus (pd.DataFrame): The text corpus to search for answers.
                Must contain two columns: 'title' and 'article'.
            llm_prompt (LLMPrompt): The prompt to be used by the agent.
            fewshot_examples (str): Prompt examples to include in agent prompt.
                If "", the agent is zero-shot. Defaults to "".
            embedding_model_name (str): The name of the embedding model to use for retrieval.
                Defaults to "whereisai/uae-large-v1".
            retriever_num_documents (int): The number of documents to retrieve.
                Defaults to 4.
            port (int): The port to use for the retriever.
                Defaults to 8001.
            retrieval_method (str): The retrieval method to use. Can be "bm25", "dense" or "faiss".
                Defaults to "bm25".
            index_path (str): The path to the index file for the BM25 or dense retriever.
                Defaults to None.
            corpus_path (str): The path to the corpus file for the BM25 or dense retriever.
                Defaults to None.
            corpus_name (str): The name of the corpus to use for the vertexai retriever.
                Defaults to None.
            vector_distance_threshold (float): The maximum distance for vertexai retrieved vectors to be considered relevant.
                Defaults to None.
        """
        NshotAgent.__init__(self, text_corpus, llm_prompt, fewshot_examples)
        RAGMixin.__init__(
            self,
            text_corpus,
            embedding_model_name,
            retriever_num_documents,
            port,
            retrieval_method,
            index_path,
            corpus_path,
            corpus_name,
            vector_distance_threshold,
        )

    def _build_agent_prompt(self, question: str) -> str:
        """
        Override the method in NshotAgent to use RAG to create evidence.
        """
        evidence = self.get_RAG_evidence(question)
        return self.combine_evidence_and_question(evidence, question)

class NshotSCAAgent(Agent):
    def __init__(
        self,
        llm_prompt: LLMPrompt,
        fewshot_examples: str = "",
        prolog_query: bool = False,
        sca_context_corpus_path: str = None,
    ):
        """
        Args:
            llm_prompt (LLMPrompt): The prompt to be used by the agent.
            fewshot_examples (str): Few-shot prompt examples to include in agent prompt.
                If "", the agent is zero-shot. Defaults to "".
            prolog_query (bool): Whether to use the prompt for eliciting prolog queries from LLMs.
                Passed on to `LLMPrompt.get_prompt()`. Defaults to False.
            sca_context_corpus_path (path): The path context corpus to search for answers.
        """
        super().__init__([], llm_prompt)
        self.fewshot_examples = fewshot_examples
        self.prolog_query = prolog_query
        self.sca_context_corpus = load_sca_context_corpus(sca_context_corpus_path)


    def combine_evidence_and_question(self, evidence: str, question: str) -> str:
        """
        Combine the evidence and question to form the agent prompt using `self.llm_prompt`.
        """
        if self.fewshot_examples:  # Few-shot
            return self.llm_prompt.get_prompt(self.prolog_query).format(
                evidence=evidence, examples=self.fewshot_examples, question=question
            )
        else:  # Zero-shot
            return self.llm_prompt.get_prompt(self.prolog_query).format(evidence=evidence, question=question)

    def _build_agent_prompt(self, question: str, question_id: str) -> str:
        evidence = get_evidence_from_sca_context_corpus(self.sca_context_corpus, question_id)
        return self.combine_evidence_and_question(evidence, question)

    async def run(
        self,
        llm_chat: LLMChat,
        question: str,
        inf_gen_config: InferenceGenerationConfig,
        *args,
        **kwargs,
    ) -> LLMChatResponse:
        question_id = kwargs.get("question_id")
        if not question_id:
            raise ValueError("`question_id` must be provided in kwargs for NshotSCAAgent.run()")
        logger.debug(f"\n\t>>> question: {question}\n")

        # Create a conversation with 1 user prompt and initialize agent interactions
        prompt = self._build_agent_prompt(question, question_id)
        conv = Conversation(messages=[Message(role="user", content=[ContentTextMessage(text=prompt)])])
        self.agent_interactions = conv

        # Generate response
        inf_gen_config = inf_gen_config.model_copy(update=dict(stop_sequences=[]), deep=True)
        response = await llm_chat.generate_response(conv, inf_gen_config)

        # Update agent's conversation
        self.agent_interactions.messages.append(
            Message(role="assistant", content=[ContentTextMessage(text=response.pred)])
        )

        if self.prolog_query:
            response.pred = parse_prolog_query(response.pred)
            return response

        try:
            pred = NshotAgent.parse_answer(response.pred)
            error = None
        except Exception as e:
            pred = ""
            error = f"<agent_error>{traceback.format_exc()}</agent_error>"
            error = f"<agent_error>{e}</agent_error>"
        return LLMChatResponse(pred=pred, usage=response.usage, error=error)

    async def batch_run(
        self,
        llm_chat: LLMChat,
        questions: list[str],
        inf_gen_config: InferenceGenerationConfig,
        *args,
        **kwargs,
    ) -> list[LLMChatResponse]:
        question_ids = kwargs.get("question_ids")
        if not question_ids:
            raise ValueError("`question_ids` must be provided in kwargs for NshotSCAAgent.batch_run()")
        logger.debug(f"\n\t>>> questions: {questions}\n")

        # Create a conversation for each user prompt, and initialize agent interactions
        prompts: list[str] = []
        for i in range(len(questions)):
            prompts.append(self._build_agent_prompt(questions[i], question_ids[i]))
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
            if self.prolog_query:
                responses[i].pred = parse_prolog_query(response.pred)

        parsed_responses: list[LLMChatResponse] = []
        for response in responses:
            try:
                pred = NshotAgent.parse_answer(response.pred)
                error = None
            except Exception as e:
                pred = ""
                error = f"<agent_error>{traceback.format_exc()}</agent_error>"
                error = f"<agent_error>{e}</agent_error>"
            parsed_responses.append(LLMChatResponse(pred=pred, usage=response.usage, error=error))
        return parsed_responses