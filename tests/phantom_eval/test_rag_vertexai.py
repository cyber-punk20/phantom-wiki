import vertexai

from phantom_eval import get_parser
from phantom_eval.agents.common import RAGMixin


def test_rag_vertexai():
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

    rag = RAGMixin(
        text_corpus=[],
        retriever_num_documents=args.retriever_num_documents,
        corpus_name=args.corpus_name,
        retrieval_method="vertexai",
    )
    print(
        rag.get_RAG_evidence(
            "Who is the friend of the sister of the daughter-in-law of the grandparent of the grandparent of the son of the person whose hobby is meteorology?"
        )
    )

if __name__ == "__main__":
    test_rag_vertexai()