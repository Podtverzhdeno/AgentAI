import os
import asyncio
from typing import Annotated, TypedDict, Sequence

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
# Инструменты LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# 1. Описываем инструмент (то, что агент "может делать")
@tool
async def get_weather(city: str):
    """Возвращает текущую погоду в указанном городе."""
    # Здесь могла бы быть логика запроса к API
    return f"В городе {city} https://world-weather.ru/."

tools = [get_weather]

# 2. Определяем состояние (то, что агент "помнит") [11]
class AgentState(TypedDict):
    # add_messages позволяет автоматически объединять новые сообщения с историей [12]
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 3. Настройка модели с привязкой инструментов [13]
def get_model():
    llm = ChatOpenAI(
        model="openai/gpt-5.2-chat", # Твоя модель через OpenRouter
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=100
    )
    return llm.bind_tools(tools) # "Биндим" инструменты к модели [14]

# --- Узлы графа ---

async def call_model(state: AgentState):
    """Узел, который вызывает LLM"""
    llm = get_model()
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Логика выбора: закончить или вызвать инструмент? [15]"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- Сборка Графа [16, 17] ---

workflow = StateGraph(AgentState)

# Добавляем узлы
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools)) # Стандартный узел для вызова инструментов [18]

# Выстраиваем связи
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue) # Условный переход [10]
workflow.add_edge("tools", "agent") # Возврат к агенту после работы инструмента

app = workflow.compile()

async def main():
    # Запускаем агента с вопросом, требующим вызова инструмента
    inputs = {"messages": [HumanMessage(content="Какая погода в Москве?")]}

    async for event in app.astream(inputs, stream_mode="values"):
        for value in event.values():
            last_msg = value[-1]
            print(f"Роль: {type(last_msg).__name__}")
            print(f"Контент: {last_msg.content}\n")

if __name__ == "__main__":
    asyncio.run(main())