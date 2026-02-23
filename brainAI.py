import os
import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Загружаем переменные окружения
load_dotenv()  # путь к твоему .env

def get_openrouter_llm(model="openai/gpt-5.2-chat"):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY не найден в .env")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=500
    )

async def main():
    llm = get_openrouter_llm()

    # Формируем сообщение
    message = HumanMessage(content="Кто ты?")

    try:
        # Асинхронный вызов модели
        response = await llm.ainvoke([message])
        print("Ответ модели:", response.content)
    except Exception as e:
        print("Ошибка при вызове LLM:", e)

if __name__ == "__main__":
    asyncio.run(main())
