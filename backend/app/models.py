from typing import Literal

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    stock_code: str | None = Field(default=None, max_length=20)


class AgentStep(BaseModel):
    agent: str
    title: str
    content: str


class Source(BaseModel):
    title: str
    category: str
    score: float


class AnalysisResponse(BaseModel):
    trace_id: str
    mode: Literal["demo", "llm"]
    research: AgentStep
    risk: AgentStep
    strategy: AgentStep
    report: str
    sources: list[Source]
    disclaimer: str = "内容由 AI 生成，仅供研究与教学，不构成投资建议。"


class KnowledgeCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=2, max_length=10000)
    category: str = Field(default="用户知识", max_length=50)

