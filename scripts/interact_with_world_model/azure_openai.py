# uv pip install openai azure.identity

import random
from openai import AzureOpenAI
from azure.identity import get_bearer_token_provider, AzureCliCredential

def get_endpoints():
    gpt_4o = [
        {
            "endpoints": "https://conversationhubeastus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubsouthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubwestus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        # {
        #     "endpoints": "https://conversationhubwestus3.openai.azure.com/",
        #     "speed": 150,
        #     "model": "gpt-4o"
        # },
        {
            "endpoints": "https://readineastus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://readineastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://readinnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        # {
        #     "endpoints": "https://readinsouthcentralus.openai.azure.com/",
        #     "speed": 150,
        #     "model": "gpt-4o"
        # },
        {
            "endpoints": "https://readinwestus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        # {
        #     "endpoints": "https://readinwestus3.openai.azure.com/",
        #     "speed": 150,
        #     "model": "gpt-4o"
        # },
        {
            "endpoints": "https://conversationhubeastus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://conversationhubsouthcentralus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        # {
        #     "endpoints": "https://conversationhubwestus.openai.azure.com/",
        #     "speed": 450,
        #     "model": "gpt-4o-global"
        # },
        {
            "endpoints": "https://readineastus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://readineastus2.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://readinnorthcentralus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
        {
            "endpoints": "https://readinwestus.openai.azure.com/",
            "speed": 450,
            "model": "gpt-4o-global"
        },
    ]

    gpt_4o_mini = [
        # https://conversationhubeastus.openai.azure.com/
        # https://conversationhubeastus2.openai.azure.com/
        # https://conversationhubnorthcentralus.openai.azure.com/
        # https://conversationhubsouthcentralus.openai.azure.com/
        # https://conversationhubswedencentral.openai.azure.com/
        # https://conversationhubwestus.openai.azure.com/
        # https://readineastus.openai.azure.com/
        # https://readineastus2.openai.azure.com/
        # https://readinnorthcentralus.openai.azure.com/
        # https://readinwestus.openai.azure.com/
        # https://malicata-azure-ai-foundry.cognitiveservices.azure.com/
        {
            "endpoints": "https://conversationhubeastus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://conversationhubsouthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://conversationhubswedencentral.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://conversationhubwestus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://readineastus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://readineastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://readinnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://readinwestus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://malicata-azure-ai-foundry.cognitiveservices.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },

    ]

    gpt_4_turbo = [
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4-turbo"
        },
        {
            "endpoints": "https://conversationhubswedencentral.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4-turbo"
        },
        {
            "endpoints": "https://readineastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4-turbo"
        },
        {
            "endpoints": "https://readinswedencentral.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4-turbo"
        },
    ]

    gpt_4_1 = [
        {
            "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-DZS"
        },
        {
            "endpoints": "https://conversationhubsouthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-DZS"
        },
        {
            "endpoints": "https://conversationhubswedencentral.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-DZS"
        },
        {
            "endpoints": "https://readinnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-DZS"
        },
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-global"
        },
        {
            "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-global"
        },
        {
            "endpoints": "https://conversationhubswedencentral.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-global"
        },
        {
            "endpoints": "https://readinnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4.1-global"
        },
    ]

    gpt_5 = [
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-5-global"
        },
    ]

    return {
        "gpt-4o": gpt_4o,
        "gpt-4o-mini": gpt_4o_mini,
        "gpt-4-turbo": gpt_4_turbo,
        "gpt-4.1": gpt_4_1,
        "gpt-5": gpt_5,
    }


def select_endpoint(model_name: str) -> dict:
    azure_endpoints = get_endpoints()
    entries = azure_endpoints[model_name]
    candidates = [e for e in entries if e.get("speed", 0) > 0 and e.get("endpoints")]
    weights = [e["speed"] for e in candidates]
    chosen = random.choices(candidates, weights=weights, k=1)[0]
    return chosen

def get_client(
        model_name="gpt-4o",
        tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47",
        api_version="2024-12-01-preview",
        max_retries=5,
    ):
    azure_ad_token_provider = get_bearer_token_provider(
        AzureCliCredential(tenant_id=tenant_id),
        "https://cognitiveservices.azure.com/.default"
    )

    selected = select_endpoint(model_name)
    client = AzureOpenAI(
        azure_endpoint=selected["endpoints"],
        azure_ad_token_provider=azure_ad_token_provider,
        api_version=api_version,
        max_retries=max_retries,
    )
    return client, selected["model"]


if __name__ == "__main__":
    model = "gpt-4o"
    max_tokens = 1000
    temperature = 0.7

    client, resolved_model = get_client(model_name=model)

    messages = [{"role": "user", "content": "What is the capital of France?",}]
    response = client.chat.completions.create(
        model=resolved_model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    content = response.choices[0].message.content
    print("Response from GPT-4o:")
    print(content)
