# 📊 ValPal Financial Document Intelligence — RAG Pipeline


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


AI Engineer

Built as part of the **ValPal Financial Document Intelligence** workflow.
