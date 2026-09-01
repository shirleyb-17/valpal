import os
import json
import re
import uuid

from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import (
    DocumentIntelligenceClient
)
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    AnalyzeDocumentRequest
)

from openai import AzureOpenAI
from pinecone import Pinecone


# ============================================================
# LOAD ENV
# ============================================================

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

PDF_PATH = "tata-340-361.pdf"

METADATA_PATH = "document_metadata.json"
MARKDOWN_PATH = "document_content.md"

VALIDATE_SCHEMA_PATH = "validate.json"
OUTPUT_PATH = "final_answers.json"

# This identifies one project/document space in Pinecone
PROJECT_ID = os.getenv("PROJECT_ID", "valpal_tata_001")

TOP_K = 5


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv(
    "DOCUMENT_INTELLIGENCE_ENDPOINT"
)

DOCUMENT_INTELLIGENCE_KEY = os.getenv(
    "DOCUMENT_INTELLIGENCE_KEY"
)

AZURE_FOUNDRY_ENDPOINT = os.getenv(
    "AZURE_FOUNDRY_ENDPOINT"
)

AZURE_FOUNDRY_API_KEY = os.getenv(
    "AZURE_FOUNDRY_API_KEY"
)

CHAT_DEPLOYMENT = os.getenv(
    "AZURE_FOUNDRY_DEPLOYMENT"
)

EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_EMBEDDING_DEPLOYMENT"
)

PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME"
)


# ============================================================
# VALIDATE CONFIG
# ============================================================

required_env = {
    "DOCUMENT_INTELLIGENCE_ENDPOINT":
        DOCUMENT_INTELLIGENCE_ENDPOINT,

    "DOCUMENT_INTELLIGENCE_KEY":
        DOCUMENT_INTELLIGENCE_KEY,

    "AZURE_FOUNDRY_ENDPOINT":
        AZURE_FOUNDRY_ENDPOINT,

    "AZURE_FOUNDRY_API_KEY":
        AZURE_FOUNDRY_API_KEY,

    "AZURE_FOUNDRY_DEPLOYMENT":
        CHAT_DEPLOYMENT,

    "AZURE_EMBEDDING_DEPLOYMENT":
        EMBEDDING_DEPLOYMENT,

    "PINECONE_API_KEY":
        PINECONE_API_KEY,

    "PINECONE_INDEX_NAME":
        PINECONE_INDEX_NAME,
}


missing = [
    name
    for name, value in required_env.items()
    if not value
]

if missing:
    raise ValueError(
        "Missing environment variables:\n"
        + "\n".join(missing)
    )


# ============================================================
# INITIALIZE AZURE OPENAI
# ============================================================

azure_client = AzureOpenAI(
    azure_endpoint=AZURE_FOUNDRY_ENDPOINT,
    api_key=AZURE_FOUNDRY_API_KEY,
    api_version="2024-10-21",
)


# ============================================================
# INITIALIZE DOCUMENT INTELLIGENCE
# ============================================================

document_client = DocumentIntelligenceClient(
    endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT,
    credential=AzureKeyCredential(
        DOCUMENT_INTELLIGENCE_KEY
    )
)


# ============================================================
# INITIALIZE PINECONE
# ============================================================

pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY
)

pinecone_index = pinecone_client.Index(
    PINECONE_INDEX_NAME
)


# ============================================================
# STEP 1
# DOCUMENT INTELLIGENCE
#
# PDF -> Markdown + paragraph metadata
# ============================================================

def extract_document():

    print("\n[1] Running Document Intelligence...")

    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    poller = document_client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=AnalyzeDocumentRequest(
            bytes_source=pdf_bytes
        ),
        output_content_format="markdown"
    )

    result: AnalyzeResult = poller.result()

    # --------------------------------------------------------
    # Markdown
    # --------------------------------------------------------

    markdown_text = result.content

    with open(
        MARKDOWN_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(markdown_text)

    # --------------------------------------------------------
    # Paragraph metadata
    # --------------------------------------------------------

    citations_metadata = []

    if result.paragraphs:

        for paragraph in result.paragraphs:

            page_number = None
            polygon = None

            if paragraph.bounding_regions:

                page_number = (
                    paragraph
                    .bounding_regions[0]
                    .page_number
                )

                polygon = (
                    paragraph
                    .bounding_regions[0]
                    .polygon
                )

            citations_metadata.append({

                "text_snippet":
                    paragraph.content,

                "role":
                    getattr(
                        paragraph,
                        "role",
                        None
                    ),

                "page_number":
                    page_number,

                "bounding_polygon":
                    polygon
            })

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            citations_metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"  Markdown saved -> {MARKDOWN_PATH}"
    )

    print(
        f"  Metadata saved -> {METADATA_PATH}"
    )

    print(
        f"  Paragraphs -> {len(citations_metadata)}"
    )

    return markdown_text, citations_metadata


# ============================================================
# STEP 2
# CREATE CHUNKS FROM document_metadata.json
#
# Each paragraph becomes a chunk.
# Page number is preserved.
# ============================================================

def build_chunks(metadata_path):

    print("\n[2] Building chunks...")

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:

        paragraphs = json.load(f)

    chunks = []

    for index, paragraph in enumerate(paragraphs):

        text = (
            paragraph.get("text_snippet")
            or ""
        ).strip()

        role = (
            paragraph.get("role")
            or ""
        ).lower()

        page_number = paragraph.get(
            "page_number"
        )

        # Skip empty paragraphs
        if not text:
            continue

        # Skip noisy Document Intelligence roles
        if role in {
            "pageheader",
            "pagefooter",
            "pagenumber"
        }:
            continue

        chunk_id = f"{PROJECT_ID}_chunk_{index}"

        chunks.append({

            "id":
                chunk_id,

            "text":
                text,

            "page":
                page_number,

            "role":
                paragraph.get("role"),

            "project_id":
                PROJECT_ID
        })

    print(
        f"  Created {len(chunks)} chunks"
    )

    return chunks


# ============================================================
# STEP 3
# EMBEDDING FUNCTION
# ============================================================

def create_embedding(text):

    response = azure_client.embeddings.create(

        model=EMBEDDING_DEPLOYMENT,

        input=text
    )

    return response.data[0].embedding


# ============================================================
# STEP 4
# CHUNK + EMBED + UPSERT TO PINECONE
# ============================================================

def chunk_and_upsert(chunks):

    print("\n[3] Embedding chunks and uploading to Pinecone...")

    vectors = []

    for i, chunk in enumerate(chunks):

        print(
            f"  Embedding {i + 1}/{len(chunks)}"
        )

        embedding = create_embedding(
            chunk["text"]
        )

        vectors.append({

            "id":
                chunk["id"],

            "values":
                embedding,

            "metadata": {

                "project_id":
                    PROJECT_ID,

                "text":
                    chunk["text"],

                "page":
                    chunk["page"],

                "role":
                    chunk["role"]
            }
        })

    # --------------------------------------------------------
    # Upsert into Pinecone
    #
    # Namespace = PROJECT_ID
    # --------------------------------------------------------

    pinecone_index.upsert(
        vectors=vectors,
        namespace=PROJECT_ID
    )

    print(
        f"  Upserted {len(vectors)} vectors"
    )

    print(
        f"  Namespace -> {PROJECT_ID}"
    )


# ============================================================
# STEP 5
# LOAD validate.json
# ============================================================

def load_questions():

    print("\n[4] Loading validate.json...")

    with open(
        VALIDATE_SCHEMA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        validate_schema = json.load(f)

    flat_questions = []

    field_lookup = {}

    for section_name, section_data in (
        validate_schema.items()
    ):

        for field in section_data.get(
            "fields",
            []
        ):

            field_name = field[
                "field_name"
            ]

            question = field[
                "question"
            ]

            flat_questions.append(
                (
                    field_name,
                    question
                )
            )

            field_lookup[field_name] = {

                "id":
                    field.get("id"),

                "section":
                    section_name,

                "question":
                    question
            }

    print(
        f"  Questions -> {len(flat_questions)}"
    )

    return flat_questions, field_lookup


# ============================================================
# STEP 6
# RETRIEVE FROM PINECONE
# ============================================================

def retrieve_top_k(question):

    print(
        f"\n  Retrieving context for:"
        f"\n  {question}"
    )

    # --------------------------------------------------------
    # Embed question
    # --------------------------------------------------------

    question_embedding = create_embedding(
        question
    )

    # --------------------------------------------------------
    # Query Pinecone
    # --------------------------------------------------------

    response = pinecone_index.query(

        namespace=PROJECT_ID,

        vector=question_embedding,

        top_k=TOP_K,

        include_metadata=True
    )

    matches = response.matches

    print(
        f"  Retrieved {len(matches)} chunks"
    )

    # --------------------------------------------------------
    # Convert Pinecone response
    # --------------------------------------------------------

    retrieved_chunks = []

    for match in matches:

        metadata = match.metadata or {}

        retrieved_chunks.append({

            "id":
                match.id,

            "score":
                match.score,

            "text":
                metadata.get("text"),

            "page":
                metadata.get("page"),

            "role":
                metadata.get("role")
        })

    return retrieved_chunks


# ============================================================
# STEP 7
# LLM PROMPT
# ============================================================

ANSWER_PROMPT = """
You are ValPal's Financial Document Intelligence Engine.

Answer the question using ONLY the retrieved excerpts.

IMPORTANT RULES:

1. Never use outside knowledge.
2. Never invent a value.
3. If the answer is not clearly supported by the
   retrieved excerpts, return found=false.
4. Preserve the original number, unit and currency.
5. The quote must be copied exactly from one of
   the retrieved excerpts.
6. The page must come DIRECTLY from the metadata
   associated with the excerpt containing the quote.
7. Do not calculate or guess a page number.
8. Do not search for the quote elsewhere.
9. If there is insufficient evidence, return found=false.

Return ONLY valid JSON.

Expected output:

{
  "found": true,
  "value": 123,
  "unit": "USD",
  "confidence": 0.9,
  "page": 4,
  "quote": "short exact excerpt",
  "reason": "brief explanation"
}

If not found:

{
  "found": false,
  "value": null,
  "unit": null,
  "confidence": 0,
  "page": null,
  "quote": null,
  "reason": "No supporting evidence found in retrieved excerpts"
}

==================================================
QUESTION
==================================================

{question}

==================================================
RETRIEVED EXCERPTS
==================================================

{excerpts}
"""


# ============================================================
# STEP 8
# CLEAN LLM JSON
# ============================================================

def clean_json_response(raw):

    raw = raw.strip()

    raw = re.sub(
        r"^```json\s*",
        "",
        raw,
        flags=re.IGNORECASE
    )

    raw = re.sub(
        r"^```\s*",
        "",
        raw
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw
    )

    try:

        return json.loads(raw)

    except json.JSONDecodeError:

        start = raw.find("{")
        end = raw.rfind("}")

        if start != -1 and end != -1:

            try:

                return json.loads(
                    raw[
                        start:end + 1
                    ]
                )

            except json.JSONDecodeError:
                pass

        raise ValueError(
            "LLM returned invalid JSON:\n\n"
            + raw
        )


# ============================================================
# STEP 9
# ASK LLM
# ============================================================

def answer_question(
    question,
    retrieved_chunks
):

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    excerpts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        excerpts.append(
            f"""
[Retrieved Chunk {i}]
[Page: {chunk['page']}]
[Similarity: {chunk['score']}]

{chunk['text']}
"""
        )

    excerpts_text = "\n".join(
        excerpts
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = ANSWER_PROMPT.format(

        question=question,

        excerpts=excerpts_text
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    response = azure_client.chat.completions.create(

        model=CHAT_DEPLOYMENT,

        max_tokens=500,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw = (
        response
        .choices[0]
        .message
        .content
    )

    return clean_json_response(
        raw
    )


# ============================================================
# STEP 10
# VALIDATE PAGE AGAINST RETRIEVED CHUNKS
# ============================================================

def validate_answer_page(
    answer,
    retrieved_chunks
):

    """
    Safety check.

    We do NOT fuzzy-match the quote against the
    original document.

    Instead, we check whether the LLM's returned
    page belongs to one of the retrieved chunks.

    """

    if not answer.get("found"):

        return answer

    answer_page = answer.get(
        "page"
    )

    quote = (
        answer.get("quote")
        or ""
    ).strip()

    if answer_page is None or not quote:

        answer["found"] = False
        answer["reason"] = (
            "Answer did not contain "
            "required page or quote evidence."
        )

        return answer

    # --------------------------------------------------------
    # Check retrieved evidence
    # --------------------------------------------------------

    valid_evidence = False

    for chunk in retrieved_chunks:

        chunk_page = chunk.get(
            "page"
        )

        chunk_text = (
            chunk.get("text")
            or ""
        )

        if (
            chunk_page == answer_page
            and quote in chunk_text
        ):

            valid_evidence = True

            break

    # --------------------------------------------------------
    # Reject unsupported answer
    # --------------------------------------------------------

    if not valid_evidence:

        answer["found"] = False

        answer["value"] = None

        answer["unit"] = None

        answer["confidence"] = 0

        answer["page"] = None

        answer["quote"] = None

        answer["reason"] = (
            "LLM answer could not be grounded "
            "in the retrieved Pinecone evidence."
        )

    return answer


# ============================================================
# STEP 11
# RUN COMPLETE RAG PIPELINE
# ============================================================

def run():

    print("\n========================================")
    print("VALPAL RAG PIPELINE")
    print("========================================")

    # --------------------------------------------------------
    # 1. Document Intelligence
    # --------------------------------------------------------

    extract_document()

    # --------------------------------------------------------
    # 2. Build chunks
    # --------------------------------------------------------

    chunks = build_chunks(
        METADATA_PATH
    )

    # --------------------------------------------------------
    # 3. Embed + Pinecone
    # --------------------------------------------------------

    chunk_and_upsert(
        chunks
    )

    # --------------------------------------------------------
    # 4. Load questions
    # --------------------------------------------------------

    flat_questions, field_lookup = (
        load_questions()
    )

    answers = {}

    # --------------------------------------------------------
    # 5. Process every question
    # --------------------------------------------------------

    for field_name, question in (
        flat_questions
    ):

        print("\n----------------------------------------")

        print(
            f"FIELD: {field_name}"
        )

        print(
            f"QUESTION: {question}"
        )

        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        retrieved_chunks = retrieve_top_k(
            question
        )

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        answer = answer_question(

            question,

            retrieved_chunks
        )

        # ----------------------------------------------------
        # Grounding validation
        # ----------------------------------------------------

        answer = validate_answer_page(

            answer,

            retrieved_chunks
        )

        # ----------------------------------------------------
        # Reattach schema metadata
        # ----------------------------------------------------

        answer["id"] = (
            field_lookup[field_name]
            ["id"]
        )

        answer["field_name"] = (
            field_name
        )

        answer["question"] = (
            question
        )

        answer["citation"] = (

            f"Page {answer['page']}: "
            f"\"{answer['quote']}\""

            if (
                answer.get("found")
                and answer.get("page")
                and answer.get("quote")
            )

            else None
        )

        answers[field_name] = answer

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            answers,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n========================================")

    print(
        f"Saved -> {OUTPUT_PATH}"
    )

    print("========================================\n")

    print(
        json.dumps(
            answers,
            indent=2,
            ensure_ascii=False
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run()
