import asyncio
from typing import Any
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }


llm_client = AzureChatOpenAI(
    #TODO:
    # temperature=0.0
    # azure_deployment='gpt-4o'
    # azure_endpoint=DIAL_URL
    # api_key=SecretStr(API_KEY)
    # api_version=""
)

token_tracker = TokenTracker()


def join_context(context: list[dict[str, Any]]) -> str:
    context_str = ""
    for user in context:
        context_str += f"User:\n"
        for key, value in user.items():
            context_str += f"  {key}: {value}\n"
        context_str += "\n"
    return context_str


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    #TODO:
    # 1. Create messages array with:
    #       - SystemMessage(content=system_prompt)
    #       - HumanMessage(content=user_message)
    # 2. Generate response (use `ainvoke`, don't forget to `await` the response)
    # 3. Get usage: response.response_metadata.get('token_usage', {}).get("total_tokens", 0)
    # 4. Add tokens to `token_tracker`
    # 5. Print `response.content` and `total_tokens`
    # 5. return `response.content`
    raise NotImplementedError


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        #TODO:
        # 1. Get all users (use UserClient)
        # 2. Split all users on batches (100 users in 1 batch). We need it since LLMs have its limited context window
        # 3. Prepare tasks for async run of response generation for users batches:
        #       - create array tasks
        #       - iterate through `user_batches` and call `generate_response` with these params:
        #           - system_prompt=BATCH_SYSTEM_PROMPT
        #           - user_message=USER_PROMPT.format(context=join_context(user_batch), query=user_question)
        # 4. Gather tasks: `await asyncio.gather(*tasks)`
        # 5. Filter results on 'NO_MATCHES_FOUND' (see instructions for BATCH_SYSTEM_PROMPT)
        # 5. If results after filtration are present:
        #       - combine filtered results `"\n\n".join(relevant_results)`
        #       - call `await generate_response` with such params:
        #           - system_prompt=FINAL_SYSTEM_PROMPT,
        #           - user_message=f"SEARCH RESULTS:\n{combined_results}\n\nORIGINAL QUERY: {user_question}"
        # 6. Otherwise prin the info that `No users found matching`
        # 7. In the end print info about usage, you will be impressed of how many tokens you have used. (imagine if we have 10k or 100k users 😅)
    raise NotImplementedError


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Wr added + one chain in flow where original user data can be changed by LLM
#     User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation