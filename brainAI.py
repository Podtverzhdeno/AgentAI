import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI

load_dotenv()

def get_openrouter_llm(model="openai/gpt-5.2-chat"):
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=100
    )

if __name__ == "__main__":
    llm = get_openrouter_llm(model="openai/gpt-5.2-chat")
    response = llm.invoke("Кто ты?")
    print(response.content)