import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain.schema import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Before implementation open the `flow.png` to see the flow of app.
# Searches users by hobbies and provides their full info in JSON format:
#   Input: In need to gather people that love to go to mountains
#   Output:
#     rock climbing: [{full user info JSON},...],
#     hiking: [{full user info JSON},...],
#     camping: [{full user info JSON},...]
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in system will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor,
#    we will remove deleted users and add new - it will also resolve the issue with consistency within this 2 services
#    and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.

SYSTEM_PROMPT = """You are a RAG-powered assistant that groups users by their hobbies.

## Flow:
Step 1: User will ask to search users by their hobbies etc.
Step 2: Will be performed search in the Vector store to find most relevant users.
Step 3: You will be provided with CONTEXT (most relevant users, there will be user ID and information about user), and 
        with USER QUESTION.
Step 4: You group by hobby users that have such hobby and return response according to Response Format

## Response Format:
{format_instructions}
"""

USER_PROMPT = """## CONTEXT:
{context}

## USER QUESTION: 
{query}"""


llm_client = AzureChatOpenAI(
    #TODO:
    # temperature=0.0
    # azure_deployment='gpt-4o'
    # azure_endpoint=DIAL_URL
    # api_key=SecretStr(API_KEY)
    # api_version=""
)


class GroupingResult(BaseModel):
    hobby: str = Field(description="Hobby. Example: football, painting, horsing, photography, bird watching...")
    user_ids: list[int] = Field(description="List of user IDs that have hobby requested by user.")


class GroupingResults(BaseModel):
    grouping_results: list[GroupingResult] = Field(description="List matching search results.")


def format_user_document(user: dict[str, Any]) -> str:
    #TODO:
    # Return user id and about_me info.
    # Sample:
    # User:
    #   id: {id}
    #   About user: {about_me}
    # ---
    # Th
    raise NotImplementedError


class InputGrounder:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.user_client = UserClient()
        self.vectorstore = None

    async def __aenter__(self):
        await self.initialize_vectorstore()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def initialize_vectorstore(self, batch_size: int = 50):
        """Initialize vectorstore with all current users."""
        print("🔍 Loading all users for initial vectorstore...")
        #TODO:
        # 1. Get all users (use UserClient)
        # 2. Iterate through users and prepare array of Document with: `[Document(id=user.get('id'), page_content=format_user_document(user)) for user in users]`
        #    Pay attention that we save user id in separate column, we will use it later for removal of deleted users.
        #    Also, we persint in `page_content` just user id and `about_me` content, not the whole JSON.
        # 3. Split all `documents` on batches (100 documents in 1 batch). We need it since Embedding models have limited context window
        # 4. Setup vectorstore:
        #       - create Chroma (FAISS doesn't support necessary functionality for further task) as `self.vectorstore` with
        #           - collection_name="users"
        #           - embedding_function=self.embeddings
        #       - Prepare tasks array: iterate through batches and call `self.vectorstore.aadd_documents(batch)`
        #       - Gather tasks: `await asyncio.gather(*tasks)`
        raise NotImplementedError

    async def retrieve_context(self, query: str, k: int = 100, score: float = 0.2) -> str:
        """Retrieve context, with optional automatic vectorstore update."""
        #TODO:
        # 1. Call `_update_vectorstore` to fetch new users to vectorstor and remove deleted
        # 2. Make similarity search (`similarity_search_with_relevance_scores` method)
        # 3. Create `context_parts` empty array (we will collect content here)
        # 4. Iterate through retrieved relevant docs (pay attention that its tuple (doc, relevance_score)) and:
        #       - add doc page content to `context_parts` and then `print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content}")`
        # 5. Return joined context from `context_parts` with `\n\n` spliterator (to enhance readability)
        raise NotImplementedError

    async def _update_vectorstore(self):
        #TODO:
        # 1. Get all users (use UserClient)
        # 2. Get all the data from the vectorstore: `self.vectorstore.get()`
        # 3. Get set of ids from the vectorstor: `set(str(user_id) for user_id in vectorstore_data.get("ids", []))`. We
        #    need it to compare ids from DB with ids that we get via latest API call to UserService
        # 4. Prepare dict from retrieved users (key is user id, value is full user info): `{str(user.get('id')): user for user in users}`
        # 5. Prepare set with users ids
        # 6. Find new user ids: `users_ids_set - vectorstore_ids_set`
        # 7. Find user ids that need to be deleted: `vectorstore_ids_set - users_ids_set`
        # 8. If `ids_to_delete` is not empty then delete from vectorstore all rows with collected `ids_to_delete`.
        #    Chroma has method `delete`, that applies list of ids
        # 9. Prepare new Documents:
        #       - Iterate through new user ids and create array with Documents: `Document(id=user_id, page_content=format_user_document(users_dict[user_id]))`
        # 10. Id new documents are present then save them to vectorstore
        raise NotImplementedError

    def augment_prompt(self, query: str, context: str) -> str:
        # TODO: Make augmentation for USER_PROMPT via `format` method
        raise NotImplementedError

    def generate_answer(self, augmented_prompt: str) -> GroupingResults:
        #TODO:
        # 1. Create PydanticOutputParser with `pydantic_object=GroupingResults` as `parser`
        # 2. Create messages array with:
        #       - SystemMessagePromptTemplate.from_template(template=SYSTEM_PROMPT)
        #       - HumanMessage(content=augmented_prompt)
        # 3. Generate `prompt`: `ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())`
        # 4. Invoke it: `(prompt | llm_client | parser).invoke({})` as `grouping_results: GroupingResults`
        # 5. return grouping_results
        raise NotImplementedError


class OutputGrounder:
    def __init__(self):
        self.user_client = UserClient()

    async def ground_response(self, grouping_results: GroupingResults):
        #TODO:
        # 1. Iterate through grouping results
        # 2. Print hobby
        # 3. Print fetched users: await self._find_users(grouping_result.user_ids)
        # ---
        # JFYI:
        # This is quite simple output grounding (just to verify that such user exist), in reality we would probably
        # ground the hobbies and some other actions.
        raise NotImplementedError

    async def _find_users(self, ids: list[int]) -> list[dict[str, Any]]:
        async def safe_get_user(user_id: int) -> Optional[dict[str, Any]]:
            try:
                #TODO:
                # Get user by id (it is async method)
                raise NotImplementedError
            except Exception as e:
                if "404" in str(e):
                    print(f"User with ID {user_id} is absent (404)")
                    return None
                raise  # Re-raise non-404 errors

        #TODO:
        # 1. Prepare task array to get users and gather results
        # 2. Filter results and provide users that is not None
        raise NotImplementedError


async def main():
    embeddings = AzureOpenAIEmbeddings(
        #TODO:
        # deployment='text-embedding-3-small-1'
        # azure_endpoint=DIAL_URL
        # api_key=SecretStr(API_KEY)
        # dimensions=384
        # check_embedding_ctx_length=False
    )
    output_grounder = OutputGrounder()

    async with InputGrounder(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need to gather people that love to go to mountains")
        print(" - Find people that love to watch stars and night sky")
        print(" - I need people to go to fishing together")

        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            #TODO:
            # 1. Retrieve context
            # 2. Make augmentation
            # 3. Generate answer
            # 4. Make output grounding
            raise NotImplementedError


if __name__ == "__main__":
    asyncio.run(main())
