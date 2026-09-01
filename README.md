# 📊 ValPal Financial Document Intelligence — RAG Pipeline

A production-oriented **Financial Document Intelligence and Retrieval-Augmented Generation (RAG) pipeline** for extracting structured financial information from documents.

The pipeline combines **Azure AI Document Intelligence**, **Azure OpenAI**, and **Pinecone** to transform financial documents into searchable evidence and answer predefined validation questions with **page-level citations and grounding checks**.

---

## 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │   Financial PDF      │
                         │  tata-340-361.pdf    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌────────────────────────────┐
                    │ Azure Document Intelligence│
                    │      prebuilt-layout       │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
          ┌─────────────────┐          ┌────────────────────┐
          │ Markdown        │          │ Paragraph Metadata │
          │ document_content│          │ page / role /      │
          │ .md             │          │ bounding polygon   │
          └─────────────────┘          └──────────┬─────────┘
                                                  │
                                                  ▼
                                      ┌────────────────────┐
                                      │     Chunking       │
                                      │ One paragraph =    │
                                      │ one chunk           │
                                      └──────────┬─────────┘
                                                 │
                                                 ▼
                                      ┌────────────────────┐
                                      │ Azure OpenAI       │
                                      │ Embeddings         │
                                      └──────────┬─────────┘
                                                 │
                                                 ▼
                                      ┌────────────────────┐
                                      │     Pinecone       │
                                      │ Vector Database    │
                                      │ Namespace=Project  │
                                      └──────────┬─────────┘
                                                 │
                    ┌────────────────────────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │    validate.json    │
          │ Financial Questions │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Question Embedding  │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Pinecone Retrieval  │
          │       Top-K = 5     │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Azure OpenAI LLM    │
          │ Evidence-based      │
          │ Answer Generation   │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Grounding Validator │
          │ Page + Quote Check  │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ final_answers.json  │
          └─────────────────────┘
```

---

# 🎯 Objective

The objective of this pipeline is to answer financial-document questions while ensuring that every extracted answer is:

* **Grounded in the source document**
* **Supported by retrieved evidence**
* **Associated with the correct page**
* **Accompanied by an exact quote**
* **Returned in structured JSON**
* **Rejected when sufficient evidence is unavailable**

Instead of asking an LLM to directly read an entire financial document, the system follows:

```text
Document
   ↓
Parse
   ↓
Normalize
   ↓
Chunk
   ↓
Embed
   ↓
Vector Store
   ↓
Retrieve
   ↓
Generate
   ↓
Validate
   ↓
Structured Answer
```

---

# 🧰 Technology Stack

| Component        | Technology                     |
| ---------------- | ------------------------------ |
| Document Parsing | Azure AI Document Intelligence |
| Document Model   | `prebuilt-layout`              |
| Document Output  | Markdown                       |
| LLM              | Azure OpenAI                   |
| Embeddings       | Azure OpenAI Embeddings        |
| Vector Database  | Pinecone                       |
| Retrieval        | Similarity Search              |
| Configuration    | Python `dotenv`                |
| Input Schema     | JSON                           |
| Output           | JSON                           |
| Language         | Python                         |

---

# 📁 Project Structure

```text
valpal-rag/
│
├── tata-340-361.pdf
│
├── validate.json
│
├── document_metadata.json
├── document_content.md
├── final_answers.json
│
├── main.py
│
├── .env
├── .gitignore
└── README.md
```

### Input Files

#### `tata-340-361.pdf`

Financial document used as the source of truth.

---

#### `validate.json`

Contains the questions/fields that the system needs to extract.

Example:

```json
{
  "loans": {
    "fields": [
      {
        "id": "Q1.1",
        "field_name": "non_current_loans_total",
        "question": "What is the total non-current loans (unsecured, considered good) as at March 31, 2026?",
        "data_type": "currency",
        "source_type": "financial_document"
      }
    ]
  }
}
```

The schema is flattened internally into:

```text
field_name
question
id
section
```

---

# 🔄 Pipeline Workflow

## 1. Document Intelligence

The PDF is sent to Azure AI Document Intelligence using:

```python
model_id="prebuilt-layout"
```

The pipeline requests Markdown output:

```python
output_content_format="markdown"
```

This provides:

* Document text
* Paragraphs
* Page numbers
* Paragraph roles
* Bounding polygons

The Markdown representation is saved as:

```text
document_content.md
```

Paragraph-level metadata is saved as:

```text
document_metadata.json
```

Example metadata:

```json
{
  "text_snippet": "Total non-current loans...",
  "role": null,
  "page_number": 12,
  "bounding_polygon": [...]
}
```

---

# 2. Chunking

The system uses a simple **paragraph-based chunking strategy**.

```text
One Document Intelligence paragraph
                ↓
             One Chunk
```

Each chunk preserves:

```json
{
  "id": "valpal_tata_001_chunk_25",
  "text": "...",
  "page": 12,
  "role": null,
  "project_id": "valpal_tata_001"
}
```

### Noise Filtering

The following Document Intelligence roles are ignored:

```text
pageHeader
pageFooter
pageNumber
```

This prevents repeated headers, footers, and page numbers from polluting retrieval.

---

# 3. Embedding Generation

Every chunk is converted into a vector using the configured Azure OpenAI embedding deployment.

```python
response = azure_client.embeddings.create(
    model=EMBEDDING_DEPLOYMENT,
    input=text
)
```

Conceptually:

```text
Financial Text
      ↓
Embedding Model
      ↓
Vector
[0.012, -0.231, 0.552, ...]
```

---

# 4. Pinecone Vector Storage

The generated vectors are uploaded to Pinecone.

Each vector contains:

```json
{
  "id": "valpal_tata_001_chunk_25",
  "values": [...],
  "metadata": {
    "project_id": "valpal_tata_001",
    "text": "...",
    "page": 12,
    "role": null
  }
}
```

The Pinecone namespace is isolated by:

```text
PROJECT_ID
```

For example:

```text
valpal_tata_001
```

This provides logical separation between different documents/projects.

---

# 5. Question Loading

The pipeline reads:

```text
validate.json
```

and extracts all questions.

For example:

```text
Q1.1
    ↓
What is the total non-current loans...?
```

Internally:

```python
flat_questions = [
    (
        "non_current_loans_total",
        "What is the total non-current loans...?"
    )
]
```

A lookup table is also created so the final answer can be reattached to its schema metadata.

---

# 6. Query Embedding

Each validation question is converted into an embedding using the **same embedding model** used for document chunks.

```text
Question
   ↓
Embedding Model
   ↓
Question Vector
```

This is important because document and query vectors need to exist in the same embedding space.

---

# 7. Vector Retrieval

The question vector is sent to Pinecone.

```python
top_k=5
```

The system retrieves the five most similar chunks.

Example:

```text
Question
   ↓
Pinecone
   ↓
Top 5 Similar Chunks
```

Each retrieved result contains:

```json
{
  "id": "valpal_tata_001_chunk_25",
  "score": 0.91,
  "text": "...",
  "page": 12,
  "role": null
}
```

---

# 8. Evidence-Based LLM Generation

The retrieved chunks are passed to Azure OpenAI.

The LLM is explicitly instructed to:

* Use only retrieved excerpts
* Never use outside knowledge
* Never invent values
* Preserve original numbers
* Preserve currency/unit
* Return an exact quote
* Use the page from retrieved metadata
* Return `found=false` if evidence is insufficient

Expected response:

```json
{
  "found": true,
  "value": 123456,
  "unit": "USD",
  "confidence": 0.95,
  "page": 12,
  "quote": "Total non-current loans...",
  "reason": "The retrieved excerpt directly states the requested amount."
}
```

---

# 9. JSON Cleaning

LLMs sometimes return JSON wrapped in Markdown:

````text
```json
{
   ...
}
````

````

The `clean_json_response()` function removes Markdown fences before parsing.

It also attempts to recover JSON when additional text surrounds the JSON object.

This ensures the pipeline can convert the LLM response into a Python dictionary.

---

# 10. Grounding Validation

This is one of the most important parts of the pipeline.

The LLM-generated answer is **not automatically trusted**.

The system verifies:

```text
LLM page
    ↓
Retrieved chunk page
````

and:

```text
LLM quote
    ↓
Exact substring
    ↓
Retrieved chunk text
```

The answer is accepted only when both conditions are satisfied.

### Valid

```text
Answer page = 12

Retrieved chunk:
Page 12
"Total non-current loans were..."

Quote:
"Total non-current loans were..."
```

✅ Accepted

### Invalid

```text
Answer page = 15

Retrieved chunk:
Page 12
"Total non-current loans were..."
```

❌ Rejected

Likewise, if the quote does not exist inside the retrieved chunk:

```text
❌ Grounding failure
```

The answer is converted to:

```json
{
  "found": false,
  "value": null,
  "unit": null,
  "confidence": 0,
  "page": null,
  "quote": null
}
```

---

# 🛡️ Grounding Strategy

The pipeline follows a conservative principle:

```text
No evidence
     ↓
No answer
```

rather than:

```text
No evidence
     ↓
LLM guess
```

This is particularly important for financial documents.

The architecture can therefore be summarized as:

```text
       RETRIEVAL
          ↓
     Evidence Set
          ↓
      LLM Answer
          ↓
   Grounding Validator
       ↙       ↘
   Valid       Invalid
     ↓            ↓
  Accept       Reject
```

---

# 📄 Output

The final output is stored in:

```text
final_answers.json
```

Example:

```json
{
  "non_current_loans_total": {
    "found": true,
    "value": 123456,
    "unit": "USD",
    "confidence": 0.95,
    "page": 12,
    "quote": "Total non-current loans...",
    "reason": "The retrieved evidence directly supports the answer.",
    "id": "Q1.1",
    "field_name": "non_current_loans_total",
    "question": "What is the total non-current loans...?",
    "citation": "Page 12: \"Total non-current loans...\""
  }
}
```

---

# 🔐 Environment Configuration

Create a `.env` file:

```env
DOCUMENT_INTELLIGENCE_ENDPOINT=your_document_intelligence_endpoint
DOCUMENT_INTELLIGENCE_KEY=your_document_intelligence_key

AZURE_FOUNDRY_ENDPOINT=your_azure_openai_endpoint
AZURE_FOUNDRY_API_KEY=your_azure_openai_key

AZURE_FOUNDRY_DEPLOYMENT=your_chat_deployment
AZURE_EMBEDDING_DEPLOYMENT=your_embedding_deployment

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index

PROJECT_ID=valpal_tata_001
```

### ⚠️ Security

Never commit `.env` to GitHub.

Add:

```gitignore
.env
__pycache__/
*.pyc
```

---

# 📦 Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install python-dotenv
pip install azure-core
pip install azure-ai-documentintelligence
pip install openai
pip install pinecone
```

Or create a `requirements.txt`:

```text
python-dotenv
azure-core
azure-ai-documentintelligence
openai
pinecone
```

Then:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Pipeline

Place the financial PDF in the project directory:

```text
tata-340-361.pdf
```

Place the validation schema:

```text
validate.json
```

Configure `.env`.

Then run:

```bash
python main.py
```

The pipeline executes:

```text
[1] Document Intelligence
        ↓
[2] Build chunks
        ↓
[3] Embed + Pinecone
        ↓
[4] Load validate.json
        ↓
[5] Retrieve evidence
        ↓
[6] Generate answers
        ↓
[7] Validate grounding
        ↓
[8] Save final_answers.json
```

---

# 📊 Pipeline Components

| Step | Function                 | Responsibility             |
| ---- | ------------------------ | -------------------------- |
| 1    | `extract_document()`     | PDF → Markdown + metadata  |
| 2    | `build_chunks()`         | Paragraphs → chunks        |
| 3    | `create_embedding()`     | Text → embeddings          |
| 4    | `chunk_and_upsert()`     | Embeddings → Pinecone      |
| 5    | `load_questions()`       | Load validation questions  |
| 6    | `retrieve_top_k()`       | Retrieve relevant evidence |
| 7    | `answer_question()`      | LLM answer generation      |
| 8    | `clean_json_response()`  | Parse LLM JSON             |
| 9    | `validate_answer_page()` | Grounding validation       |
| 10   | `run()`                  | Execute complete pipeline  |

---

# 🧠 RAG Design

This implementation follows a standard RAG architecture:

### Indexing Pipeline

```text
PDF
 ↓
Azure Document Intelligence
 ↓
Paragraph Extraction
 ↓
Metadata Preservation
 ↓
Chunking
 ↓
Embedding
 ↓
Pinecone
```

### Query Pipeline

```text
Question
 ↓
Question Embedding
 ↓
Pinecone Similarity Search
 ↓
Top-K Evidence
 ↓
Azure OpenAI
 ↓
Structured Answer
 ↓
Grounding Validation
```

---

# 🔎 Why Metadata Matters

The pipeline does not store only the text.

It stores:

```text
Text
Page
Role
Project ID
```

This allows the system to generate citations such as:

```text
Page 12:
"Total non-current loans..."
```

Without page metadata, the LLM could potentially produce an answer but the application would not know where that answer came from.

---

# 🧩 Project Isolation

Pinecone namespaces are based on:

```python
PROJECT_ID
```

Example:

```text
valpal_tata_001
```

Another document could use:

```text
valpal_reliance_001
```

Result:

```text
Pinecone
│
├── valpal_tata_001
│      ├── chunk_1
│      ├── chunk_2
│      └── ...
│
└── valpal_reliance_001
       ├── chunk_1
       ├── chunk_2
       └── ...
```

This prevents retrieval across unrelated document spaces.

---

# ⚙️ Current Configuration

```text
Chunking Strategy : Paragraph-based
Vector Database   : Pinecone
Top-K Retrieval   : 5
Document Parser   : Azure Document Intelligence
LLM               : Azure OpenAI
Embedding Model   : Azure OpenAI Embeddings
Output Format     : JSON
Citation          : Page + Exact Quote
Grounding         : Enabled
```

---

# 🚀 Future Improvements

The current implementation provides a strong baseline, but the following improvements can make the system more robust.

## 1. Hybrid Retrieval

Combine:

```text
Dense Vector Search
        +
BM25 / Keyword Search
```

This is especially useful for financial documents containing:

* Account numbers
* Exact terminology
* Financial statement headings
* Company names
* Specific dates

---

## 2. Reranking

Current:

```text
Question
   ↓
Pinecone
   ↓
Top 5
   ↓
LLM
```

Improved:

```text
Question
   ↓
Pinecone Top 20
   ↓
Reranker
   ↓
Top 5
   ↓
LLM
```

This can improve retrieval precision.

---

## 3. Table-Aware Extraction

Financial statements contain many tables.

A future version can preserve:

```text
Table
 ├── Row
 ├── Column
 ├── Cell
 └── Page
```

instead of treating every paragraph independently.

---

## 4. Semantic Chunking

Instead of:

```text
1 paragraph = 1 chunk
```

the system can use:

```text
Financial Section
      ↓
Subsection
      ↓
Related paragraphs
      ↓
Semantic chunk
```

This can improve context for questions requiring multiple nearby values.

---

## 5. Citation Bounding Boxes

The current metadata already captures:

```python
bounding_polygon
```

This can later be used to provide document-level visual citations:

```text
Page 12
   ↓
Bounding Box
   ↓
Highlighted source text
```

---

## 6. Confidence Calibration

Currently confidence is generated by the LLM.

A stronger implementation could calculate confidence from multiple signals:

```text
Retrieval Score
+
Quote Match
+
Answer Validation
+
Schema Validation
+
LLM Confidence
```

Example:

```text
Final Confidence =
weighted evidence score
```

---

## 7. Structured Output / JSON Schema

Instead of relying only on prompt instructions, structured LLM output can enforce:

```json
{
  "found": "boolean",
  "value": "...",
  "unit": "...",
  "confidence": "number",
  "page": "integer|null",
  "quote": "string|null",
  "reason": "string"
}
```

This reduces malformed responses.

---

# ⚠️ Important Limitations

### Paragraph-level chunking

A paragraph may not contain enough context for questions requiring multiple rows or columns.

### Top-K retrieval

Using:

```python
TOP_K = 5
```

may miss relevant evidence when a question requires information distributed across multiple parts of the document.

### Exact quote validation

The grounding validator requires:

```python
quote in chunk_text
```

Therefore, minor formatting differences introduced by the LLM can cause an otherwise correct answer to be rejected.

### Embedding dependency

The Pinecone index dimension must match the configured embedding model.

### LLM-generated confidence

The current confidence value is generated by the LLM and should not be treated as a statistically calibrated probability.

---

# 🔁 End-to-End Data Flow

```text
                         FINANCIAL DOCUMENT
                                  │
                                  ▼
                    Azure Document Intelligence
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
                  Markdown                 Metadata
                     │                    page / role /
                     │                    coordinates
                     │                         │
                     └────────────┬────────────┘
                                  ▼
                             Chunking
                                  │
                                  ▼
                            Embeddings
                                  │
                                  ▼
                              Pinecone
                                  │
                                  │
                    ┌─────────────┘
                    │
                    ▼
             validate.json
                    │
                    ▼
               Questions
                    │
                    ▼
             Query Embedding
                    │
                    ▼
            Pinecone Top-K Search
                    │
                    ▼
             Retrieved Evidence
                    │
                    ▼
              Azure OpenAI LLM
                    │
                    ▼
             Structured JSON
                    │
                    ▼
          Grounding Validation
              │             │
           PASS            FAIL
              │             │
              ▼             ▼
       final_answers.json   found=false
```

---

# 🎯 Key Design Principle

The core design principle of this system is:

> **The LLM generates the answer, but the retrieved document evidence determines whether the answer is accepted.**

This creates a safer architecture for financial-document extraction:

```text
LLM
 ↓
Candidate Answer
 ↓
Evidence Validation
 ↓
Accepted / Rejected
```

rather than:

```text
Document → LLM → Trust the answer
```

---

# 📌 Example Use Cases

This architecture can be extended to:

* Financial statement extraction
* Annual report analysis
* Balance sheet extraction
* Profit & loss extraction
* Loan information extraction
* Asset and liability extraction
* Tax document processing
* Valuation-model data preparation
* Regulatory document extraction
* Audit support
* Financial due diligence
* Document validation

---

# 🏁 Summary

**ValPal Financial Document Intelligence** converts unstructured financial documents into structured, evidence-backed answers.

The system combines:

```text
Azure Document Intelligence
          +
Azure OpenAI
          +
Pinecone
          +
RAG
          +
Grounding Validation
          ↓
Structured Financial Intelligence
```

The most important characteristics of the pipeline are:

```text
✅ Document parsing
✅ Metadata preservation
✅ Paragraph-based chunking
✅ Vector embeddings
✅ Pinecone retrieval
✅ Evidence-based generation
✅ Structured JSON output
✅ Page-level citations
✅ Exact quote validation
✅ Hallucination rejection
```

---

## 👩‍💻 Author

**Shirley Deborah B**

AI Engineer

Built as part of the **ValPal Financial Document Intelligence** workflow.
