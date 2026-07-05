# Reference implementation co-developed with Claude (Anthropic) for AI engineering practice.

# ── IMPORTS ───────────────────────────────────────────────────────────────────

from langchain_anthropic import ChatAnthropic
# The LangChain wrapper around Anthropic's API.
# Lets you use Claude inside LangChain chains instead of calling the API directly.

from langchain_chroma import Chroma
# LangChain's wrapper around ChromaDB.
# Handles creating the database, storing vectors, and querying them.

from langchain_anthropic import AnthropicEmbeddings
# Converts text → vectors using Anthropic's Voyage embedding model.
# A vector is just a list of ~1500 numbers that captures the "meaning" of text.
# Similar meaning = similar numbers = close together in vector space.

from langchain_community.document_loaders import PyPDFLoader
# Reads a PDF file and gives you back a list of Document objects.
# One Document per page. Each Document has .page_content (text) and .metadata (page number etc).

from langchain.text_splitter import RecursiveCharacterTextSplitter
# Cuts long documents into smaller chunks.
# "Recursive" means it tries splitting on paragraphs first, then sentences,
# then words — whatever keeps chunks under your size limit without cutting mid-sentence.

from langchain.prompts import ChatPromptTemplate
# A reusable template for your prompt.
# You define it once with {placeholders}, fill them in at runtime.

from langchain_core.output_parsers import StrOutputParser
# Claude's response comes back as a complex object (AIMessage).
# This parser unwraps it to a plain Python string.

from langchain_core.runnables import RunnablePassthrough
# A no-op placeholder in a chain.
# "Pass this value through unchanged" — used to route the question
# to both the retriever AND the prompt without modifying it.

import os
os.environ["ANTHROPIC_API_KEY"] = "your-key"
# Sets the API key as an environment variable.
# Both ChatAnthropic and AnthropicEmbeddings read it from here automatically.
# Better practice: load from a .env file using python-dotenv.


# ── STEP 1: LOAD THE PDF ─────────────────────────────────────────────────────

loader = PyPDFLoader("your_document.pdf")
# Creates a loader pointed at your PDF file.
# Doesn't read it yet — just sets up the path.

pages = loader.load()
# Actually reads the PDF. Returns a list of Document objects, one per page.
# Each Document looks like:
#   Document(
#     page_content="The actual text from page 1...",
#     metadata={"source": "your_document.pdf", "page": 0}
#   )

print(f"Loaded {len(pages)} pages")
# Sanity check — make sure it read the right number of pages.
# If this is 0, the PDF might be scanned images (needs OCR) or password protected.


# ── STEP 2: SPLIT INTO CHUNKS ─────────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    # Maximum characters per chunk.
    # Why not just use whole pages? Because a page might be 3000 chars —
    # too long to embed meaningfully, and wastes context when only part is relevant.
    # 1000 chars ≈ ~200 words ≈ 2-3 paragraphs. Good default.

    chunk_overlap=200,
    # Each chunk shares 200 chars with the previous one.
    # Why? Because if a key sentence lands right at a chunk boundary,
    # overlap ensures it appears fully in at least one chunk.
    # Without overlap you'd miss context at the seams.

    separators=["\n\n", "\n", ".", " "]
    # The splitter tries these in order to find a clean cut point.
    # First tries to split on double newline (paragraph break) — cleanest.
    # If the chunk is still too big, tries single newline.
    # Then sentence end. Then word boundary.
    # Never cuts mid-word.
)

chunks = splitter.split_documents(pages)
# Takes the list of page Documents and splits each one into smaller chunks.
# Returns a new list of Documents — more of them, but smaller.
# Metadata (source, page number) is inherited by each chunk automatically.

print(f"Split into {len(chunks)} chunks")
# A 20-page PDF might give you 80-120 chunks depending on content density.


# ── STEP 3: EMBED AND STORE IN CHROMADB ──────────────────────────────────────

embeddings = AnthropicEmbeddings(model="voyage-3")
# Sets up the embedding model — Anthropic's Voyage 3.
# This model converts any text string into a vector of ~1024 numbers.
# Texts with similar meaning get similar vectors.
# e.g. "dog" and "puppy" will have vectors close together.
#      "dog" and "photosynthesis" will be far apart.
# Note: this costs tokens — you pay per chunk embedded.

vectorstore = Chroma.from_documents(
    documents=chunks,
    # The list of chunk Documents to store.

    embedding=embeddings,
    # The embedding model to use. ChromaDB calls this for each chunk
    # to get its vector, then stores both the vector AND the original text.

    persist_directory="./chroma_db"
    # Where to save the database on disk.
    # Without this, the database lives in memory and disappears when your script ends.
    # With this, it's saved to a folder and you can reload it next run without re-embedding.
)

print(f"Stored {vectorstore._collection.count()} chunks in ChromaDB")
# Confirms how many chunks were stored.
# Should match the number from step 2.


# ── STEP 4: CREATE A RETRIEVER ────────────────────────────────────────────────

retriever = vectorstore.as_retriever(
    search_type="similarity",
    # How to search. "similarity" = cosine similarity (most common).
    # Cosine similarity measures the angle between two vectors.
    # Angle of 0° = identical meaning. Angle of 90° = completely unrelated.
    # Other options: "mmr" (maximal marginal relevance — avoids returning
    # duplicate-ish chunks) or "similarity_score_threshold" (only return
    # chunks above a minimum relevance score).

    search_kwargs={"k": 4}
    # k = how many chunks to return per query.
    # 4 is a good default — enough context without flooding the prompt.
    # Each chunk is ~1000 chars, so 4 chunks ≈ 4000 chars of context.
    # If answers seem incomplete, try k=6. If the prompt gets too long, try k=2.
)
# The retriever is the object you call with a question.
# It embeds the question → searches ChromaDB → returns k Documents.


# ── STEP 5: BUILD THE PROMPT ──────────────────────────────────────────────────

template = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}
"""
# The prompt template with two placeholders:
# {context}  — will be filled with the retrieved chunks
# {question} — will be filled with the user's question
#
# "ONLY the context" instruction is important — stops Claude from answering
# from its training data when the answer isn't in your documents.
# Without it, Claude might give a plausible-sounding but wrong answer.

prompt = ChatPromptTemplate.from_template(template)
# Wraps the string template in a LangChain object that can be used in a chain.
# When invoked, it fills in the placeholders and returns a formatted prompt.


# ── STEP 6: SET UP THE LLM ───────────────────────────────────────────────────

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    # Which Claude model to use.

    temperature=0,
    # Deterministic output — same question always gets same answer.
    # Good for factual Q&A over documents.

    max_tokens=1024
    # Maximum length of Claude's response.
    # 1024 is fine for most answers. Increase if you expect long outputs.
)


# ── STEP 7: FORMAT FUNCTION ───────────────────────────────────────────────────

def format_docs(docs):
    # Takes a list of Document objects (from the retriever)
    # and joins them into one string to inject into the prompt.

    return "\n\n---\n\n".join(
        # Joins all chunks with a "---" separator between them
        # so Claude can clearly see where one chunk ends and another begins.

        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        # Formats each chunk with its page number.
        # doc.metadata.get('page', '?') safely gets the page number,
        # defaulting to '?' if metadata is missing.
        # Including the page number lets you (and Claude) trace answers back to source.

        for doc in docs
        # Loops over every Document the retriever returned.
    )

# Example output:
# [Page 3]
# The company was founded in 1995 by...
#
# ---
#
# [Page 7]
# Revenue grew 40% year over year...


# ── STEP 8: WIRE THE CHAIN TOGETHER ───────────────────────────────────────────

chain = (
    {
        "context":  retriever | format_docs,
        # The | symbol means "pipe" — pass output of left into input of right.
        # So: user's question → retriever (finds relevant chunks) → format_docs (joins to string)
        # The result fills {context} in the prompt.

        "question": RunnablePassthrough()
        # RunnablePassthrough just forwards the input unchanged.
        # The user's question passes through directly to fill {question} in the prompt.
        # We need this because the question goes to TWO places:
        # (1) the retriever to find relevant chunks, and (2) the prompt as {question}.
    }
    | prompt
    # Takes the dict {"context": "...", "question": "..."} and fills them
    # into the template, producing a complete formatted prompt string.

    | llm
    # Sends the formatted prompt to Claude.
    # Returns an AIMessage object containing Claude's response.

    | StrOutputParser()
    # Unwraps the AIMessage to a plain Python string.
    # Without this you'd get back an object, not a string.
)

# The full flow for one question:
# "Who founded the company?"
#   → retriever finds 4 relevant chunks about founding
#   → format_docs joins them with page numbers
#   → prompt fills in context + question
#   → Claude reads context and answers
#   → StrOutputParser returns plain string


# ── STEP 9: ASK QUESTIONS ─────────────────────────────────────────────────────

def ask(question: str):
    print(f"\nQ: {question}")
    answer = chain.invoke(question)
    # .invoke() runs the full chain end to end with the given input.
    # The input (question string) flows through every step above.
    print(f"A: {answer}")
    return answer

ask("What is the main topic of this document?")
ask("What are the key conclusions?")
ask("Who are the authors?")


# ── LOAD FROM DISK ON RESTART ─────────────────────────────────────────────────
# Don't re-embed the whole PDF every run — that costs money and takes time.
# Check if the database already exists first.

import os

CHROMA_PATH = "./chroma_db"

if os.path.exists(CHROMA_PATH):
    # Database folder already exists — load it directly.
    # No embedding happens here, just reading what's already stored.
    print("Loading existing vectorstore from disk...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
        # Still need the embedding model — used at QUERY time
        # to embed the user's question before searching.
        # Not used for the stored chunks (already embedded).
    )
else:
    # First run — build the database from scratch.
    print("Building vectorstore from PDF...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )


# ── RETURNING SOURCES ALONGSIDE THE ANSWER ────────────────────────────────────

def ask_with_sources(question: str):
    # Calls the retriever separately to get the raw source documents,
    # then calls the chain for the answer.
    # Two separate calls — simple and easy to understand.

    source_docs = retriever.invoke(question)
    # Returns the list of Document objects the retriever found.
    # Same documents that went into {context} — just unformatted.

    answer = chain.invoke(question)
    # Runs the full chain and gets Claude's answer as a string.

    print(f"\nQ: {question}")
    print(f"A: {answer}")
    print(f"\nSources used:")
    for doc in source_docs:
        page = doc.metadata.get('page', '?')
        snippet = doc.page_content[:100]
        # First 100 chars of each chunk — enough to see what it's about.
        print(f"  Page {page}: {snippet}...")

ask_with_sources("What are the key conclusions?")
# Output:
# Q: What are the key conclusions?
# A: The key conclusions are...
#
# Sources used:
#   Page 12: The study found that retrieval augmented generation significantly...
#   Page 13: Furthermore, the authors conclude that chunking strategy...
#   Page 4:  Earlier work established the baseline...
#   Page 14: In summary, the three main findings are...
