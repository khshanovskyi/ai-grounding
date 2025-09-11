import asyncio
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `vector_based_grounding.png` to see the flow of app


SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about user information.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}

##USER QUESTION: 
{query}"""


def format_user_document(user: dict[str, Any]) -> str:
    #TODO:
    # Prepare context from users JSONs in the same way as in `no_grounding.py` `join_context` method (collect as one string)
    raise NotImplementedError


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        #TODO:
        # 1. Get all users (use UserClient)
        # 2. Prepare array of Document with `page_content=format_user_document(user)` (you need to iterate through users)

        print(f"↗️ Creating embeddings and vectorstore for {len(documents)} documents...")
        #TODO:
        # call `_create_vectorstore_with_batching` (don't forget that its async) and setup it as obj var `vectorstore`
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        #TODO:
        # 1. Split all `documents` on batches (100 documents in 1 batch). We need it since Embedding models have limited context window
        # 2. Iterate through document batches and create array with tasks that will generate FAISS vector stores from documents:
        #    `FAISS.afrom_documents(batch, self.embeddings)`
        # 3. gather tasks with asyncio: `await asyncio.gather(*batch_tasks, return_exceptions=True)` as `batch_results`
        # 4. Create `final_vectorstore` variable with None value
        # 5. Iterate through `batch_results` and:
        #       - If vectorstore (iterable vectorstore from generated batch) is present:
        #           - if `final_vectorstore` is None then reference vectorstore to `final_vectorstore`
        #           - otherwise merge them: `final_vectorstore.merge_from(batch_vectorstore)`
        # 6. Make validation of the `final_vectorstore`, if None -> raise error
        # 6. Return `final_vectorstore`
        raise NotImplementedError

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        #TODO:
        # 1. Make similarity search (`similarity_search_with_relevance_scores` method)
        # 2. Create `context_parts` empty array (we will collect content here)
        # 3. Iterate through retrieved relevant docs (pay attention that its tuple (doc, relevance_score)) and:
        #       - add doc page content to `context_parts` and then `print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content}")`
        # 4. Return joined context from `context_parts` with `\n\n` spliterator (to enhance readability)
        raise NotImplementedError

    def augment_prompt(self, query: str, context: str) -> str:
        # TODO: Make augmentation for USER_PROMPT via `format` method
        raise NotImplementedError

    def generate_answer(self, augmented_prompt: str) -> str:
        #TODO:
        # 1. Create messages array with:
        #       - SystemMessage(content=SYSTEM_PROMPT)
        #       - HumanMessage(content=augmented_prompt)
        # 2. Generate response `llm_client.invoke(messages)`
        # 3. Return response content
        raise NotImplementedError


async def main():
    embeddings = AzureOpenAIEmbeddings(
        #TODO:
        # deployment='text-embedding-3-small-1'
        # azure_endpoint=DIAL_URL
        # api_key=SecretStr(API_KEY)
        # dimensions=384
    )

    llm_client = AzureChatOpenAI(
        #TODO:
        # temperature=0.0
        # azure_deployment='gpt-4o'
        # azure_endpoint=DIAL_URL
        # api_key=SecretStr(API_KEY)
        # api_version=""
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            #TODO:
            # 1. Retrieve context
            # 2. Make augmentation
            # 3. Generate answer and print it
            raise NotImplementedError


asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold
# Benefits are:
#   - Similarity search by context
#   - Any input can be used for search
#   - Costs reduce