import os
import asyncio
from typing import Annotated, TypedDict, Sequence

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

# Инструменты LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver  # Для памяти [9]

load_dotenv()

# 1. Описываем локальный инструмент
@tool
async def get_weather(city: str):
    """Возвращает текущую погоду в указанном городе."""
    return f"В городе {city} сейчас солнечно, +22°C (данные с world-weather.ru)."

# 2. Определяем состояние (Memory/State) [14, 15]
class AgentState(TypedDict):
    # add_messages позволяет автоматически объединять новые сообщения с историей [16]
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 3. Фабрика инструментов и клиента MCP [2, 6]
async def get_all_tools():
    # Конфигурация MCP-серверов (нужен установленный Node.js и сервер filesystem) [17]
    mcp_config = {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                os.path.abspath("C:/Users/user/Desktop/AgentStudy1") # Укажи путь к своей рабочей папке [18]
            ],
            "transport": "stdio"
        }
    }

    # Инициализируем MCP клиент
    mcp_client = MultiServerMCPClient(mcp_config)
    mcp_tools = await mcp_client.get_tools()

    # Объединяем локальные инструменты и инструменты MCP [12]
    return [get_weather] + mcp_tools

# 4. Настройка узлов графа [19, 20]
async def call_model(state: AgentState, config):
    # Извлекаем инструменты из метаданных графа (передадим их при компиляции)
    tools = config["configurable"].get("tools", [])

    llm = ChatOpenAI(
        model="openai/gpt-4o", # Рекомендуется использовать модели с поддержкой tool-calling [21]
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=100
    ).bind_tools(tools)

    # Добавляем системную роль для управления поведением [11]
    system_prompt = SystemMessage(content=(
        "Ты — продвинутый AI-ассистент с доступом к файловой системе и погоде. "
        "Используй инструменты для выполнения задач пользователя."
    ))

    messages = [system_prompt] + state["messages"]
    response = await llm.ainvoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Логика выбора: закончить или вызвать инструмент? [22]"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 5. Сборка графа [23, 24]
async def create_agent():
    tools = await get_all_tools()

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools)) # Стандартный узел для вызова инструментов [25, 26]

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    # Добавляем Checkpointer для сохранения состояния (памяти) [7, 8]
    memory = InMemorySaver()

    return workflow.compile(checkpointer=memory), tools

async def main():
    app, tools = await create_agent()

    # Конфигурация для запуска (thread_id позволяет разделять диалоги) [27, 28]
    config = {"configurable": {"thread_id": "user_1", "tools": tools}}

    # Пример задачи: узнать погоду и записать её в файл
    query = "Какая погода в Москве? Запиши этот ответ в файл weather_report.txt"
    inputs = {"messages": [HumanMessage(content=query)]}

    print(f"--- Запуск задачи: {query} ---\n")
    async for event in app.astream(inputs, config=config, stream_mode="values"):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if last_msg.content:
                print(f"[{type(last_msg).__name__}]: {last_msg.content}")

if __name__ == "__main__":
    # Убедись, что папка существует [18]
    os.makedirs("C:/Users/user/Desktop/AgentStudy1", exist_ok=True)
    asyncio.run(main())