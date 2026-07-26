# Reference implementation co-developed with Claude (Anthropic) for AI engineering practice.

"""
================================================================================
SESSION 4 — BETTER RAG: RETRIEVAL THAT ACTUALLY WORKS
================================================================================
Study plan: LLM Agents + Fine-tuning (10 sessions, 2hrs/weekend)
GitHub notes — fully commented reference file

What this session covers:
  - Upgrade 1: Multi-query retriever  (fixes: one phrasing misses chunks)
  - Upgrade 2: Hybrid search BM25     (fixes: vector misses exact terms)
  - Upgrade 3: Cohere re-ranker       (fixes: top-k isn't always most relevant)
  - Upgrade 4: RAGAS evals            (measures whether upgrades actually helped)

Prerequisites:
  - Session 3 complete (basic RAG pipeline + ChromaDB built)
  - ./chroma_db/ folder exists with your embedded chunks
  - ./documents/ folder with your PDFs

Install:
  pip install langchain langchain-anthropic langchain-chroma langchain-community
  pip install langchain-cohere cohere rank-bm25 ragas datasets
================================================================================
"""

import os
import logging
from langchain_anthropic import ChatAnthropic, AnthropicEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_cohere import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
os.environ["COHERE_API_KEY"]    = "your-cohere-key"     # free tier: cohere.com


# ================================================================================
# SETUP — models, vector store, raw chunks
# ================================================================================

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,       # always 0 for RAG — deterministic answers
    max_tokens=1024
)

embeddings = AnthropicEmbeddings(
    model="voyage-3"     # Anthropic's embedding model — converts text → vectors
)

# Load the vector store built in session 3
# This does NOT re-embed — just loads what's already on disk
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Load raw chunks for BM25 (keyword search)
# BM25 works on raw text — it can't use the vector store
# So we reload the docs and split them the same way as session 3
loader = DirectoryLoader(
    "./documents",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # same settings as session 3 — must match
    chunk_overlap=200
)
chunks = splitter.split_documents(pages)
print(f"Loaded {len(chunks)} chunks for BM25")


# ================================================================================
# UPGRADE 1 — MULTI-QUERY RETRIEVER
# ================================================================================
# Problem it solves:
#   Your question "What is RAG?" might not semantically match a chunk that says
#   "Retrieval Augmented Generation is a technique..." — the phrasing is different.
#   Vector search only sees meaning, not all possible phrasings.
#
# How it works:
#   The LLM generates 3 alternative phrasings of your question.
#   Each phrasing is used for a separate vector search.
#   All results are combined and deduplicated.
#   You go from 4 chunks to 9-12 unique chunks — higher recall.

# Turn on logging to see the generated queries — very helpful for debugging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Base retriever — standard vector similarity search
base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}   # return top 4 per query
)

# Wrap with multi-query
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
    # LangChain has a default prompt that asks Claude to generate
    # 3 alternative phrasings — you can customise it if needed
)

# Example usage
# docs = multi_query_retriever.invoke("What is RAG?")
# Console will show:
#   Generated queries: [
#     "What is RAG?",
#     "How does Retrieval Augmented Generation work?",
#     "What is the purpose of retrieval in LLM systems?"
#   ]


# ================================================================================
# UPGRADE 2 — HYBRID SEARCH (BM25 + VECTOR)
# ================================================================================
# Problem it solves:
#   Vector search is great at meaning/concepts but bad at exact terms.
#   "What did Lewis et al. say?" — "Lewis" is a proper noun.
#   Vector search might not rank chunks with "Lewis" highly because
#   the vector for "Lewis" doesn't cluster near your question vector.
#   BM25 (keyword search) catches exact matches that vectors miss.
#
# How it works:
#   BM25: classic keyword search — scores chunks by term frequency
#         great for: names, acronyms, technical terms, exact phrases
#   Vector: semantic search — scores chunks by meaning similarity
#            great for: concepts, paraphrasing, related ideas
#   EnsembleRetriever: merges both rankings using Reciprocal Rank Fusion (RRF)
#                      weighted by the weights you provide

# BM25 retriever — pure keyword search, no embeddings needed
bm25_retriever = BM25Retriever.from_documents(
    chunks,   # needs the raw chunk text, not the vector store
    k=6       # return top 6 keyword matches
)

# Vector retriever — pure semantic search
vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 6}   # return top 6 semantic matches
)

# Combine both into a hybrid retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
    # 40% weight to BM25 results, 60% to vector results
    # Tune based on your content:
    #   More technical/named entities → increase BM25 weight (e.g. 0.5, 0.5)
    #   More conceptual questions → increase vector weight (e.g. 0.3, 0.7)
    # EnsembleRetriever uses Reciprocal Rank Fusion (RRF) to merge rankings:
    #   A chunk that ranks high in BOTH gets a very high final score
    #   A chunk that only appears in one gets a moderate score
)

# Example usage
# docs = hybrid_retriever.invoke("What did Lewis et al. say about RAG?")
# BM25 will find chunks with "Lewis" — vector might have missed them


# ================================================================================
# UPGRADE 3 — COHERE RE-RANKER
# ================================================================================
# Problem it solves:
#   After retrieval you have 8-12 chunks. They're ranked by similarity score
#   but similarity != relevance. A chunk can be semantically close to your
#   question but not actually answer it.
#   Example:
#     Question: "What is the refund policy?"
#     Rank 3 by vector: "Policy updates are sent via email" — mentions "policy"
#     Rank 4 by vector: "Refunds take 5-10 business days" — directly answers it
#   The re-ranker fixes this ordering.
#
# How it works:
#   Vector search: encodes question and chunks SEPARATELY → compares vectors
#   Re-ranker (cross-encoder): reads question AND chunk TOGETHER as raw text
#                              → scores actual relevance, not just similarity
#   This is slower but much more accurate — used as a second pass on a small set
#
# Cost: ~$1 per 1000 calls on Cohere paid tier
#       Free tier: 1000 calls/month (enough for learning)
#       Alternative: FlashRank (free, local) — see bottom of file

reranker = CohereRerank(
    model="rerank-english-v3.0",
    top_n=4    # from however many chunks retriever returns → keep best 4
               # this is how you control how many chunks Claude sees
)

# Wrap any retriever with the re-ranker
# ContextualCompressionRetriever = "retrieve then compress/filter"
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever   # re-ranks the hybrid results
    # You can wrap ANY retriever here:
    #   base_retriever=base_retriever         → re-rank basic vector results
    #   base_retriever=multi_query_retriever  → re-rank multi-query results
    #   base_retriever=hybrid_retriever       → re-rank hybrid results (best)
)

# After re-ranking, each doc has a relevance_score in its metadata:
# docs = rerank_retriever.invoke("What is the refund policy?")
# for doc in docs:
#     print(doc.metadata["relevance_score"])  # 0.0 to 1.0
#     print(doc.page_content[:80])


# ================================================================================
# THE FULL UPGRADED CHAIN
# ================================================================================
# Layers:
#   1. Hybrid search (BM25 + vector) — casts a wide net, catches exact + semantic
#   2. Cohere re-ranker — reads question + chunk together, picks truly relevant ones
#
# Note: multi-query is commented out here to keep latency reasonable.
# Uncomment it if recall is more important than speed for your use case.

# Uncomment to add multi-query on top of hybrid:
# final_retriever = MultiQueryRetriever.from_llm(
#     retriever=ContextualCompressionRetriever(
#         base_compressor=reranker,
#         base_retriever=hybrid_retriever
#     ),
#     llm=llm
# )

final_retriever = rerank_retriever   # hybrid + re-rank


# Format retrieved docs into a single context string for the prompt
def format_docs(docs):
    """
    Joins retrieved Document objects into one string.
    Includes page number and source filename for traceability.
    Chunks separated by --- so Claude can see boundaries.
    """
    if not docs:
        return "No relevant documentation found."

    formatted = []
    for doc in docs:
        page     = doc.metadata.get("page", "?")
        source   = doc.metadata.get("source", "unknown")
        score    = doc.metadata.get("relevance_score", None)
        score_str = f", relevance: {score:.2f}" if score else ""
        formatted.append(
            f"[Page {page}, source: {source}{score_str}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


# Prompt — tells Claude to use ONLY the retrieved context
# This prevents hallucination — Claude can't answer from training data
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say:
"I don't have that information in the provided documents."

Context:
{context}

Question:
{question}
""")


# The full RAG chain using LCEL (LangChain Expression Language)
# Each | pipes the output of the left into the input of the right
chain = (
    {
        # "context" key: question → retriever → format into string
        # The | chains: retriever takes the question string, returns docs,
        # format_docs takes docs, returns a single formatted string
        "context":  final_retriever | format_docs,

        # "question" key: question passes through unchanged
        # RunnablePassthrough() = "don't transform this, just forward it"
        # Needed because the question goes to TWO places:
        #   (1) the retriever to find relevant chunks
        #   (2) the prompt as the {question} placeholder
        "question": RunnablePassthrough()
    }
    | prompt           # fills {context} and {question} into the template
    | llm              # sends the formatted prompt to Claude
    | StrOutputParser() # unwraps Claude's AIMessage object → plain string
)


def ask(question: str):
    """Run a question through the full RAG pipeline."""
    print(f"\nQ: {question}")
    answer = chain.invoke(question)
    print(f"A: {answer}")
    return answer


def ask_with_sources(question: str):
    """
    Run a question and show which chunks were used.
    Calls the retriever separately to get the raw source docs.
    """
    print(f"\nQ: {question}")

    # Get the source chunks (separate retriever call)
    source_docs = final_retriever.invoke(question)

    # Get the answer (full chain call)
    answer = chain.invoke(question)

    print(f"A: {answer}")
    print(f"\nSources used ({len(source_docs)} chunks):")
    for doc in source_docs:
        page  = doc.metadata.get("page", "?")
        src   = doc.metadata.get("source", "unknown")
        score = doc.metadata.get("relevance_score", None)
        snippet = doc.page_content[:100].replace("\n", " ")
        score_str = f" [score: {score:.2f}]" if score else ""
        print(f"  Page {page} — {src}{score_str}")
        print(f"  '{snippet}...'")

    return answer, source_docs


# ================================================================================
# UPGRADE 4 — RAGAS EVALS
# ================================================================================
# RAGAS measures your RAG pipeline quality automatically.
# It uses another LLM to score your outputs against ground truth.
#
# Four metrics:
#   faithfulness      — is the answer grounded in the retrieved context?
#                       low score = Claude is hallucinating beyond the chunks
#   answer_relevancy  — does the answer actually address the question?
#                       low score = answered something else / too vague
#   context_recall    — did retrieval find ALL the relevant chunks?
#                       low score = multi-query retriever would help
#   context_precision — are retrieved chunks actually useful?
#                       low score = re-ranker would help, too much noise
#
# Use this to:
#   1. Establish a baseline (session 3 basic retriever)
#   2. Measure each upgrade's improvement
#   3. Know when retrieval is good enough to move on

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset


def run_ragas_eval(retriever_to_test, questions, ground_truths, chain_to_test=None):
    """
    Run RAGAS evaluation on a retriever + chain combination.

    Args:
        retriever_to_test: any LangChain retriever
        questions: list of test questions
        ground_truths: list of expected correct answers
        chain_to_test: the full chain (uses global chain if None)

    Returns:
        RAGAS evaluation scores dict
    """
    use_chain = chain_to_test or chain

    # Collect inputs and outputs for each question
    results = {
        "question":     [],
        "answer":       [],
        "contexts":     [],    # list of lists — the retrieved chunks per question
        "ground_truth": []
    }

    print(f"Running eval on {len(questions)} questions...")

    for i, (question, truth) in enumerate(zip(questions, ground_truths)):
        print(f"  [{i+1}/{len(questions)}] {question[:50]}...")

        # Get retrieved chunks for this question
        docs = retriever_to_test.invoke(question)
        contexts = [doc.page_content for doc in docs]

        # Get the answer from the chain
        answer = use_chain.invoke(question)

        results["question"].append(question)
        results["answer"].append(answer)
        results["contexts"].append(contexts)
        results["ground_truth"].append(truth)

    # Convert to HuggingFace Dataset format (what RAGAS expects)
    dataset = Dataset.from_dict(results)

    # Wrap LLM and embeddings for RAGAS
    # RAGAS uses these to score your answers
    ragas_llm        = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Run evaluation
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    return scores


def compare_retrievers(questions, ground_truths):
    """
    Compare baseline (session 3) vs upgraded (session 4) retriever.
    Prints a side-by-side comparison of RAGAS scores.
    """
    print("\n=== Evaluating baseline retriever (session 3) ===")
    baseline_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Need a chain that uses the baseline retriever
    baseline_chain = (
        {"context": baseline_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    baseline_scores = run_ragas_eval(
        retriever_to_test=baseline_retriever,
        questions=questions,
        ground_truths=ground_truths,
        chain_to_test=baseline_chain
    )

    print("\n=== Evaluating upgraded retriever (session 4) ===")
    upgraded_scores = run_ragas_eval(
        retriever_to_test=final_retriever,
        questions=questions,
        ground_truths=ground_truths
    )

    # Print comparison
    print("\n" + "="*60)
    print(f"{'Metric':<25} {'Baseline':>10} {'Upgraded':>10} {'Delta':>10}")
    print("="*60)

    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    for metric in metrics:
        base = baseline_scores[metric]
        upgr = upgraded_scores[metric]
        delta = upgr - base
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"{metric:<25} {base:>10.3f} {upgr:>10.3f} {arrow} {abs(delta):>7.3f}")

    print("="*60)
    return baseline_scores, upgraded_scores


# ================================================================================
# ALTERNATIVE RE-RANKERS (free options)
# ================================================================================

def get_flashrank_retriever(base):
    """
    FlashRank: free re-ranker that runs locally — no API key needed.
    Slightly lower quality than Cohere but zero cost.
    Install: pip install flashrank
    """
    from langchain_community.document_compressors import FlashrankRerank
    reranker = FlashrankRerank(top_n=4)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base
    )


def get_huggingface_reranker(base):
    """
    HuggingFace cross-encoder: free, runs locally, good quality.
    Needs GPU for speed at scale, fine on CPU for small workloads.
    Install: pip install sentence-transformers
    """
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    model    = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=model, top_n=4)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base
    )

# Usage — plug in instead of Cohere:
# final_retriever = get_flashrank_retriever(hybrid_retriever)
# final_retriever = get_huggingface_reranker(hybrid_retriever)


# ================================================================================
# RETRIEVER DECISION GUIDE
# ================================================================================
#
# Which retriever to use when:
#
# BASIC (session 3)
#   vectorstore.as_retriever(search_kwargs={"k": 4})
#   Use when: prototyping, small doc set, speed is priority
#   Quality: ~60-70% on RAGAS
#
# + MULTI-QUERY
#   MultiQueryRetriever.from_llm(retriever=base, llm=llm)
#   Use when: users ask questions in many different ways
#             context_recall score is low (missing relevant chunks)
#   Adds: 1 extra LLM call per query, higher latency
#   Quality boost: +10-15% context_recall
#
# + HYBRID (BM25 + vector)
#   EnsembleRetriever([bm25, vector], weights=[0.4, 0.6])
#   Use when: docs have proper nouns, technical terms, acronyms
#             users ask about specific named things
#   Adds: BM25 index in memory (fast, no API cost)
#   Quality boost: +5-10% on named entity questions
#
# + RE-RANKER (Cohere / FlashRank / HuggingFace)
#   ContextualCompressionRetriever(reranker, base_retriever)
#   Use when: production app, precision matters, budget allows
#             context_precision score is low (noisy chunks)
#   Adds: 1 Cohere API call per query (~$0.001)
#   Quality boost: +15-20% context_precision
#
# FULL STACK (what we built in this session)
#   hybrid → re-rank (+ optionally multi-query on top)
#   Use when: production customer-facing RAG app
#   Quality: ~85-90% on RAGAS
#
# SELF-QUERYING (covered in session 3 extra)
#   SelfQueryRetriever.from_llm(...)
#   Use when: multiple document types with rich metadata
#             users ask questions that imply a filter ("show Q3 finance docs")
#   Adds: 1 extra LLM call to generate the filter


# ================================================================================
# WHAT EACH LAYER DOES — VISUAL SUMMARY
# ================================================================================
#
#   Your question
#         │
#         ▼
#   [OPTIONAL] Multi-query: generates 3 phrasings
#         │ 3 separate searches
#         ▼
#   Hybrid search:
#     BM25 (keyword) ──┐
#                       ├── EnsembleRetriever (RRF merge) → 8-15 chunks
#     Vector (semantic) ┘
#         │
#         ▼
#   Cohere re-ranker:
#     reads question + each chunk TOGETHER
#     scores true relevance (not just similarity)
#     keeps top 4
#         │
#         ▼
#   format_docs() → single context string
#         │
#         ▼
#   Prompt template fills {context} + {question}
#         │
#         ▼
#   Claude reads context → generates answer
#         │
#         ▼
#   StrOutputParser() → plain string


# ================================================================================
# MAIN — run examples
# ================================================================================

if __name__ == "__main__":

    # Basic ask
    ask("What is RAG?")
    ask("Who invented RAG?")

    # Ask with sources
    ask_with_sources("What are the key conclusions?")

    # Run RAGAS eval comparison (replace with your real questions + answers)
    TEST_QUESTIONS = [
        "What is RAG?",
        "How does vector search work?",
        "What problem does RAG solve?",
        "What is the difference between RAG and fine-tuning?",
        "How does chunking work?",
    ]

    GROUND_TRUTHS = [
        "RAG stands for Retrieval Augmented Generation. It retrieves relevant context from a knowledge base and injects it into the LLM prompt.",
        "Vector search converts text into numerical vectors using an embedding model, then finds the most similar vectors using cosine similarity.",
        "RAG solves the knowledge cutoff problem — LLMs have stale training data, RAG lets them answer from live/private documents.",
        "RAG retrieves external knowledge at inference time without changing model weights. Fine-tuning permanently updates model weights during training.",
        "Chunking splits documents into smaller pieces (typically 500-1500 characters) with overlap so context is not lost at boundaries.",
    ]

    # Uncomment to run the full evaluation comparison
    # baseline_scores, upgraded_scores = compare_retrievers(TEST_QUESTIONS, GROUND_TRUTHS)
