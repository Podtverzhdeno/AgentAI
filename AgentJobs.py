import os
import asyncio
from enum import Enum
from typing import TypedDict, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# -----------------------------
# Константы
# -----------------------------

CATEGORIES = [
    "2D-аниматор", "3D-аниматор", "3D-моделлер",
    "Бизнес-аналитик", "Блокчейн-разработчик", "Frontend-разработчик",
    "Backend-разработчик", "Fullstack-разработчик", "Маркетолог",
    "UX/UI дизайнер", "Системный администратор", "QA-инженер",
    "Python-разработчик", "Java-разработчик", "Data Scientist"
]


class JobType(str, Enum):
    PROJECT = "проектная работа"
    PERMANENT = "постоянная работа"


class SearchType(str, Enum):
    LOOKING_FOR_WORK = "поиск работы"
    LOOKING_FOR_PERFORMER = "поиск исполнителя"


# -----------------------------
# Pydantic схема ответа
# -----------------------------

class ClassificationResult(BaseModel):
    job_type: JobType
    category: str
    search_type: SearchType
    job_type_confidence: float = Field(ge=0.0, le=1.0)
    category_confidence: float = Field(ge=0.0, le=1.0)
    search_type_confidence: float = Field(ge=0.0, le=1.0)


# -----------------------------
# State для LangGraph
# -----------------------------

class State(TypedDict):
    description: str
    result: Optional[ClassificationResult]
    processed: bool


# -----------------------------
# LLM
# -----------------------------

def get_openrouter_llm(model="openai/gpt-5.2-chat"):
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=1000
    )


# -----------------------------
# Агент
# -----------------------------

class VacancyClassificationAgent:

    def __init__(self):
        self.llm = get_openrouter_llm().with_structured_output(
            ClassificationResult
        )
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        workflow = StateGraph(State)

        workflow.add_node("classify", self._classify)
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", END)

        return workflow.compile()

    async def _classify(self, state: State):
        categories_str = "\n".join(CATEGORIES)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Ты классификатор вакансий. "
                    "Определи тип работы, категорию, тип поиска и уверенность 0-1."
                ),
                (
                    "human",
                    f"""
Описание:
{state["description"]}

Доступные категории:
{categories_str}

Верни строго структурированный ответ.
"""
                )
            ]
        )

        chain = prompt | self.llm
        result: ClassificationResult = await chain.ainvoke({})

        return {
            "result": result,   # ВАЖНО: возвращаем объект, а не dict
            "processed": True
        }

    async def classify(self, description: str) -> Dict:
        initial_state: State = {
            "description": description,
            "result": None,
            "processed": False
        }

        final_state = await self.workflow.ainvoke(initial_state)

        result: ClassificationResult = final_state["result"]

        return {
            "job_type": result.job_type.value,
            "category": result.category,
            "search_type": result.search_type.value,
            "confidence_scores": {
                "job_type_confidence": result.job_type_confidence,
                "category_confidence": result.category_confidence,
                "search_type_confidence": result.search_type_confidence
            },
            "success": final_state["processed"]
        }


# -----------------------------
# Тестирование
# -----------------------------

async def main():
    agent = VacancyClassificationAgent()

    test_cases = [
        "Требуется Python разработчик для создания веб-приложения на Django. Постоянная работа.",
        "Ищу заказы на создание логотипов и фирменного стиля.",
        "Нужен 3D-аниматор для краткосрочного проекта."
    ]

    for i, description in enumerate(test_cases, 1):
        print(f"\nТест {i}: {description}")
        result = await agent.classify(description)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())