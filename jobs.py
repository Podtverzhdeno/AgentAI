import os
import asyncio
import json
import re
from typing import TypedDict, Dict, Any
from enum import Enum
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

load_dotenv()

# Категории профессий
CATEGORIES = [
    "2D-аниматор", "3D-аниматор", "3D-моделлер",
    "Бизнес-аналитик", "Блокчейн-разработчик", "Frontend-разработчик",
    "Backend-разработчик", "Fullstack-разработчик", "Маркетолог",
    "UX/UI дизайнер", "Системный администратор", "QA-инженер",
    "Python-разработчик", "Java-разработчик", "Data Scientist"
]

class JobType(Enum):
    PROJECT = "проектная работа"
    PERMANENT = "постоянная работа"

class SearchType(Enum):
    LOOKING_FOR_WORK = "поиск работы"
    LOOKING_FOR_PERFORMER = "поиск исполнителя"

class State(TypedDict):
    description: str
    job_type: str
    category: str
    search_type: str
    confidence_scores: Dict[str, float]
    processed: bool

def get_openrouter_llm(model="openai/gpt-5.2-chat"):
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=1000
    )

class VacancyClassificationAgent:
    """Асинхронный агент для классификации вакансий и услуг"""

    def __init__(self):
        self.llm = get_openrouter_llm()
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(State)
        workflow.add_node("job_type_classification", self._classify_job_type)
        workflow.add_node("category_classification", self._classify_category)
        workflow.add_node("search_type_classification", self._classify_search_type)
        workflow.add_node("confidence_calculation", self._calculate_confidence)

        workflow.set_entry_point("job_type_classification")
        workflow.add_edge("job_type_classification", "category_classification")
        workflow.add_edge("category_classification", "search_type_classification")
        workflow.add_edge("search_type_classification", "confidence_calculation")
        workflow.add_edge("confidence_calculation", END)

        return workflow.compile()

    async def _classify_job_type(self, state: State) -> Dict[str, Any]:
        prompt = PromptTemplate(
            input_variables=["description"],
            template="""
Проанализируй следующее описание и определи тип работы.

Описание: {description}

Ответь только одним из двух вариантов:
- "проектная работа"
- "постоянная работа"

Тип работы:
"""
        )
        message = HumanMessage(content=prompt.format(description=state["description"]))
        response = await self.llm.ainvoke([message])
        job_type = response.content.strip().lower()
        if "проект" in job_type or "фриланс" in job_type:
            job_type = JobType.PROJECT.value
        else:
            job_type = JobType.PERMANENT.value
        return {"job_type": job_type}

    async def _classify_category(self, state: State) -> Dict[str, Any]:
        categories_str = "\n".join([f"- {cat}" for cat in CATEGORIES])
        prompt = PromptTemplate(
            input_variables=["description", "categories"],
            template="""
Проанализируй описание вакансии/услуги и выбери одну категорию из списка.

Описание: {description}

Доступные категории:
{categories}

Ответь только названием категории.
"""
        )
        message = HumanMessage(content=prompt.format(
            description=state["description"],
            categories=categories_str
        ))
        response = await self.llm.ainvoke([message])
        category = response.content.strip()
        if category not in CATEGORIES:
            category = self._find_closest_category(category)
        return {"category": category}

    async def _classify_search_type(self, state: State) -> Dict[str, Any]:
        prompt = PromptTemplate(
            input_variables=["description"],
            template="""
Проанализируй описание и определи, кто и что ищет.

Описание: {description}

Ответь только одним из вариантов:
- "поиск работы"
- "поиск исполнителя"
"""
        )
        message = HumanMessage(content=prompt.format(description=state["description"]))
        response = await self.llm.ainvoke([message])
        search_type = response.content.strip().lower()
        if "работ" in search_type or "ищу работу" in search_type:
            search_type = SearchType.LOOKING_FOR_WORK.value
        else:
            search_type = SearchType.LOOKING_FOR_PERFORMER.value
        return {"search_type": search_type}

    async def _calculate_confidence(self, state: State) -> Dict[str, Any]:
        prompt = PromptTemplate(
            input_variables=["description", "job_type", "category", "search_type"],
            template="""
Оцени уверенность классификации по шкале 0.0–1.0 для каждого параметра.

Описание: {description}
Тип работы: {job_type}
Категория: {category}
Тип поиска: {search_type}

Ответь строго в формате JSON одной строкой, без дополнительных пояснений:
{{"job_type_confidence": 0.0-1.0,"category_confidence": 0.0-1.0,"search_type_confidence": 0.0-1.0}}
"""
        )
        message = HumanMessage(content=prompt.format(
            description=state["description"],
            job_type=state["job_type"],
            category=state["category"],
            search_type=state["search_type"]
        ))

        response = await self.llm.ainvoke([message])
        content = response.content.strip()

        # Пытаемся извлечь JSON через регулярку
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                confidence_scores = json.loads(match.group())
            except json.JSONDecodeError:
                confidence_scores = None
        else:
            confidence_scores = None

        if not confidence_scores:
            confidence_scores = {
                "job_type_confidence": 0.7,
                "category_confidence": 0.7,
                "search_type_confidence": 0.7
            }

        return {"confidence_scores": confidence_scores, "processed": True}

    def _find_closest_category(self, predicted_category: str) -> str:
        predicted_lower = predicted_category.lower()
        for category in CATEGORIES:
            if predicted_lower in category.lower() or category.lower() in predicted_lower:
                return category
        return CATEGORIES[0]

    async def classify(self, description: str) -> Dict[str, Any]:
        initial_state = {
            "description": description,
            "job_type": "",
            "category": "",
            "search_type": "",
            "confidence_scores": {},
            "processed": False
        }
        result = await self.workflow.ainvoke(initial_state)
        return {
            "job_type": result["job_type"],
            "category": result["category"],
            "search_type": result["search_type"],
            "confidence_scores": result["confidence_scores"],
            "success": result["processed"]
        }

async def main():
    agent = VacancyClassificationAgent()
    test_cases = [
        "Требуется Python разработчик для создания веб-приложения на Django. Постоянная работа, полный рабочий день.",
        "Ищу заказы на создание логотипов и фирменного стиля. Работаю в Adobe Illustrator.",
        "Нужен 3D-аниматор для краткосрочного проекта создания рекламного ролика."
    ]

    for i, description in enumerate(test_cases, 1):
        print(f"Тест {i}: {description}")
        try:
            result = await agent.classify(description)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Ошибка: {e}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())