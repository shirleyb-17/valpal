import os
import json
import re
import uuid

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, AnalyzeDocumentRequest
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
PROJECT_ID = os.getenv("PROJECT_ID", "valpal_tata_001")
TOP_K = 5

DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
DOCUMENT_INTELLIGENCE_KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
AZURE_FOUNDRY_ENDPOINT = os.getenv("AZURE_FOUNDRY_ENDPOINT")
AZURE_FOUNDRY_API_KEY = os.getenv("AZURE_FOUNDRY_API_KEY")
CHAT_DEPLOYMENT = os.getenv("AZURE_FOUNDRY_DEPLOYMENT")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

missing = [
    name for name, value in {
        "DOCUMENT_INTELLIGENCE_ENDPOINT": DOCUMENT_INTELLIGENCE_ENDPOINT,
        "DOCUMENT_INTELLIGENCE_KEY": DOCUMENT_INTELLIGENCE_KEY,
        "AZURE_FOUNDRY_ENDPOINT": AZURE_FOUNDRY_ENDPOINT,
        "AZURE_FOUNDRY_API_KEY": AZURE_FOUNDRY_API_KEY,
        "AZURE_FOUNDRY_DEPLOYMENT": CHAT_DEPLOYMENT,
        "AZURE_EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,
        "PINECONE_API_KEY": PINECONE_API_KEY,
        "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME,
    }.items() if not value
]

if missing:
    raise ValueError("Missing environment variables:\n" + "\n".join(missing))

# ============================================================
# INITIALIZE CLIENTS
# ============================================================

azure_client = AzureOpenAI(
    azure_endpoint=AZURE_FOUNDRY_ENDPOINT,
    api_key=AZURE_FOUNDRY_API_KEY,
    api_version="2024-10-21",
)

document_client = DocumentIntelligenceClient(
    endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT,
    credential=AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY)
)

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)


# ============================================================
# STEP 1: DOCUMENT INTELLIGENCE
# ============================================================

def extract_document():
    print("\n[1] Running Document Intelligence...")

    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    poller = document_client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=AnalyzeDocumentRequest(bytes_source=pdf_bytes),
        output_content_format="markdown"
    )
    result: AnalyzeResult = poller.result()

    markdown_text = result.content
    with open(MARKDOWN_PATH, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    citations_metadata = []
    if result.paragraphs:
        for paragraph in result.paragraphs:
            page_number = None
            polygon = None
            if paragraph.bounding_regions:
                page_number = paragraph.bounding_regions[0].page_number
                polygon = paragraph.bounding_regions[0].polygon
            citations_metadata.append({
                "text_snippet": paragraph.content,
                "role": getattr(paragraph, "role", None),
                "page_number": page_number,
                "bounding_polygon": polygon
            })

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(citations_metadata, f, indent=4, ensure_ascii=False)

    print(f"  Markdown -> {MARKDOWN_PATH}")
    print(f"  Metadata -> {METADATA_PATH} ({len(citations_metadata)} paragraphs)")
    return markdown_text, citations_metadata


# ============================================================
# STEP 2: BUILD CHUNKS
# ============================================================

def build_chunks(citations_metadata):
    print("\n[2] Building chunks...")
    chunks = []
    for index, paragraph in enumerate(citations_metadata):
        text = (paragraph.get("text_snippet") or "").strip()
        role = (paragraph.get("role") or "").lower()
        page_number = paragraph.get("page_number")

        if not text:
            continue
        if role in {"pageheader", "pagefooter", "pagenumber"}:
            continue

        chunks.append({
            "id": f"{PROJECT_ID}_chunk_{index}",
            "text": text,
            "page": page_number,
            "role": paragraph.get("role"),
            "project_id": PROJECT_ID
        })

    print(f"  Created {len(chunks)} chunks")
    return chunks


# ============================================================
# STEP 3: EMBED + UPSERT TO PINECONE
# ============================================================

def create_embedding(text):
    response = azure_client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=text)
    return response.data[0].embedding


def chunk_and_upsert(chunks):
    print("\n[3] Embedding and uploading to Pinecone...")
    vectors = []
    for i, chunk in enumerate(chunks):
        print(f"  Embedding {i + 1}/{len(chunks)}", end="\r")
        embedding = create_embedding(chunk["text"])
        vectors.append({
            "id": chunk["id"],
            "values": embedding,
            "metadata": {
                "project_id": PROJECT_ID,
                "text": chunk["text"],
                "page": chunk["page"],
                "role": chunk["role"]
            }
        })

    pinecone_index.upsert(vectors=vectors, namespace=PROJECT_ID)
    print(f"\n  Upserted {len(vectors)} vectors to namespace '{PROJECT_ID}'")


# ============================================================
# STEP 4: LOAD QUESTIONS
# ============================================================

def load_questions():
    print("\n[4] Loading validate.json...")
    with open(VALIDATE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        validate_schema = json.load(f)

    flat_questions = []
    field_lookup = {}
    for section_name, section_data in validate_schema.items():
        for field in section_data.get("fields", []):
            field_name = field["field_name"]
            flat_questions.append((field_name, field["question"]))
            field_lookup[field_name] = {
                "id": field.get("id"),
                "section": section_name,
                "question": field["question"]
            }

    print(f"  Loaded {len(flat_questions)} questions")
    return flat_questions, field_lookup


# ============================================================
# STEP 5: RETRIEVE FROM PINECONE
# ============================================================

def retrieve_top_k(question):
    question_embedding = create_embedding(question)
    response = pinecone_index.query(
        namespace=PROJECT_ID,
        vector=question_embedding,
        top_k=TOP_K,
        include_metadata=True
    )
    return [
        {
            "id": match.id,
            "score": match.score,
            "text": (match.metadata or {}).get("text"),
            "page": (match.metadata or {}).get("page"),
            "role": (match.metadata or {}).get("role")
        }
        for match in response.matches
    ]


# ============================================================
# STEP 6: LLM ANSWER
# ============================================================

ANSWER_PROMPT = """
You are ValPal's Financial Document Intelligence Engine.

Answer the question using ONLY the retrieved excerpts below.
Never invent values or use outside knowledge.
If the answer is not clearly supported, return found=false.
The quote must be copied exactly from one of the excerpts.
The page must come directly from the excerpt containing the quote.

Return ONLY valid JSON in this shape:

{{
  "found": true,
  "value": 123,
  "unit": "INR crore",
  "confidence": 0.9,
  "page": 4,
  "quote": "short exact excerpt under 20 words",
  "reason": "brief explanation"
}}

If not found:

{{
  "found": false,
  "value": null,
  "unit": null,
  "confidence": 0,
  "page": null,
  "quote": null,
  "reason": "No supporting evidence found in retrieved excerpts"
}}

==================================================
QUESTION: {question}
==================================================
RETRIEVED EXCERPTS:
{excerpts}
"""


def clean_json_response(raw):
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError("LLM returned invalid JSON:\n\n" + raw)


def answer_question(question, retrieved_chunks):
    excerpts = "\n".join(
        f"[Chunk {i}] [Page: {c['page']}] [Score: {c['score']:.2f}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks, 1)
    )
    prompt = ANSWER_PROMPT.format(question=question, excerpts=excerpts)
    response = azure_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return clean_json_response(response.choices[0].message.content)


def validate_answer_page(answer, retrieved_chunks):
    if not answer.get("found"):
        return answer

    answer_page = answer.get("page")
    quote = (answer.get("quote") or "").strip()

    if answer_page is None or not quote:
        answer.update({"found": False, "reason": "Missing page or quote evidence."})
        return answer

    valid = any(
        c.get("page") == answer_page and quote in (c.get("text") or "")
        for c in retrieved_chunks
    )

    if not valid:
        answer.update({
            "found": False, "value": None, "unit": None,
            "confidence": 0, "page": None, "quote": None,
            "reason": "LLM answer could not be grounded in retrieved Pinecone evidence."
        })
    return answer


# ============================================================
# MAIN PIPELINE
# ============================================================

def run():
    print("\n========================================")
    print("VALPAL RAG PIPELINE")
    print("========================================")

    _, citations_metadata = extract_document()
    chunks = build_chunks(citations_metadata)
    chunk_and_upsert(chunks)

    flat_questions, field_lookup = load_questions()
    answers = {}

    for field_name, question in flat_questions:
        print(f"\n--- {field_name} ---")
        retrieved_chunks = retrieve_top_k(question)
        answer = answer_question(question, retrieved_chunks)
        answer = validate_answer_page(answer, retrieved_chunks)

        answer["id"] = field_lookup[field_name]["id"]
        answer["field_name"] = field_name
        answer["question"] = question
        answer["citation"] = (
            f"Page {answer['page']}: \"{answer['quote']}\""
            if answer.get("found") and answer.get("page") and answer.get("quote")
            else None
        )
        answers[field_name] = answer

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)

    print(f"\n========================================")
    print(f"Saved -> {OUTPUT_PATH}")
    print("========================================\n")
    print(json.dumps(answers, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
