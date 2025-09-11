from enum import StrEnum
from typing import Any
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from openai import BaseModel
from pydantic import SecretStr, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `api_based_grounding.png` to see the flow of app

QUERY_ANALYSIS_PROMPT = """You are a query analysis system that extracts search parameters from user questions about users.

## Available Search Fields:
- **name**: User's first name (e.g., "John", "Mary")
- **surname**: User's last name (e.g., "Smith", "Johnson") 
- **email**: User's email address (e.g., "john@example.com")

## Instructions:
1. Analyze the user's question and identify what they're looking for
2. Extract specific search values mentioned in the query
3. Map them to the appropriate search fields
4. If multiple search criteria are mentioned, include all of them
5. Only extract explicit values - don't infer or assume values not mentioned

## Examples:
- "Who is John?" → name: "John"
- "Find users with surname Smith" → surname: "Smith" 
- "Look for john@example.com" → email: "john@example.com"
- "Find John Smith" → name: "John", surname: "Smith"
- "I need user emails that filled with hiking" → No clear search parameters (return empty list)

## Response Format:
{format_instructions}
"""

SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about user information.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
- Be conversational and helpful in your responses.
- When presenting user information, format it clearly and include relevant details.
"""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION: 
{query}"""


class SearchField(StrEnum):
    NAME = "name"
    SURNAME = "surname"
    EMAIL = "email"


class SearchRequest(BaseModel):
    search_field: SearchField = Field(description="Search field")
    search_value: str = Field(description="Search value. Sample: Adam.")


class SearchRequests(BaseModel):
    search_request_parameters: list[SearchRequest] = Field(
        description="List of search parameters to execute",
        default_factory=list
    )


llm_client = AzureChatOpenAI(
    #TODO:
    # temperature=0.0
    # azure_deployment='gpt-4o'
    # azure_endpoint=DIAL_URL
    # api_key=SecretStr(API_KEY)
    # api_version=""
)

user_client = UserClient()


def retrieve_context(user_question: str) -> list[dict[str, Any]]:
    """Extract search parameters from user query and retrieve matching users."""
    #TODO:
    # 1. Create PydanticOutputParser with `pydantic_object=SearchRequests` as `parser`
    # 2. Create messages array with:
    #       - SystemMessagePromptTemplate.from_template(template=QUERY_ANALYSIS_PROMPT)
    #       - HumanMessage(content=user_question)
    # 3. Generate `prompt`: `ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())`
    # 4. Invoke it: `(prompt | llm_client | parser).invoke({})` as `search_requests: SearchRequests` (you are using LCEL)
    # 5. If `search_requests` has `search_request_parameters`:
    #       - create `requests_dict`
    #       - iterate through `search_requests.search_request_parameters` and:
    #           - add to `requests_dict` the `search_request.search_field.value` as key and `search_request.search_value` as value
    #       - print `requests_dict`
    #       - search users (**requests_dict) with `user_client`
    #       - return users that you found
    # 6. Otherwise print 'No specific search parameters found!' and return empty array
    raise NotImplementedError


def augment_prompt(user_question: str, context: list[dict[str, Any]]) -> str:
    """Combine user query with retrieved context into a formatted prompt."""
    #TODO:
    # 1. Prepare context from users JSONs in the same way as in `no_grounding.py` `join_context` method (collect as one string)
    # 2. Make augmentation: ` USER_PROMPT.format(context=context_str, query=user_question)`
    # 3. print augmented prompt
    # 3. return augmented prompt
    raise NotImplementedError


def generate_answer(augmented_prompt: str) -> str:
    """Generate final answer using the augmented prompt."""
    #TODO:
    # 1. Create messages array with:
    #       - SystemMessage(content=SYSTEM_PROMPT)
    #       - HumanMessage(content=augmented_prompt)
    # 2. Generate response `llm_client.invoke(messages)`
    # 3. Return response content
    raise NotImplementedError


def main():
    print("Query samples:")
    print(" - I need user emails that filled with hiking and psychology")
    print(" - Who is John?")
    print(" - Find users with surname Adams")
    print(" - Do we have smbd with name John that love painting?")

    while True:
        user_question = input("> ").strip()
        if user_question:
            if user_question.lower() in ['quit', 'exit']:
                break
            #TODO:
            # 1. retrieve context
            # 2. if context is present:
            #       - make augmentation
            #       - generate answer with augmented prompt
            # 3. Otherwise print `No relevant information found`
    raise NotImplementedError


if __name__ == "__main__":
    main()


# The problems with API based Grounding approach are:
#   - We need a Pre-Step to figure out what field should be used for search (Takes time)
#   - Values for search should be correct (✅ John -> ❌ Jonh)
#   - Is not so flexible
# Benefits are:
#   - We fetch actual data (new users added and deleted every 5 minutes)
#   - Costs reduce