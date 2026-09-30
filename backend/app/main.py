from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .agents import AgentOrchestrator
from .config import get_settings
from .llm import LLMClient
from .models import AnalysisRequest, AnalysisResponse, KnowledgeCreate
from .monitor import MetricsStore
from .rag import Document, KnowledgeBase

settings = get_settings()
knowledge = KnowledgeBase()
metrics = MetricsStore()
orchestrator = AgentOrchestrator(knowledge, LLMClient(settings), metrics)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "mode": "llm" if orchestrator.llm.enabled else "demo"}


@app.post(f"{settings.api_prefix}/agents/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest) -> AnalysisResponse:
    try:
        return await orchestrator.analyze(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="分析任务执行失败") from exc


@app.get(f"{settings.api_prefix}/knowledge")
async def list_knowledge() -> dict:
    return {"total": len(knowledge.documents), "items": knowledge.documents}


@app.post(f"{settings.api_prefix}/knowledge", status_code=201)
async def add_knowledge(item: KnowledgeCreate) -> dict:
    knowledge.add(Document(title=item.title, content=item.content, category=item.category))
    return {"message": "知识已添加", "total": len(knowledge.documents)}


@app.get(f"{settings.api_prefix}/llmops/metrics")
async def llmops_metrics() -> dict:
    return {**metrics.snapshot(), "mode": "llm" if orchestrator.llm.enabled else "demo"}


@app.get(f"{settings.api_prefix}/market/stocks")
async def market_stocks() -> dict:
    return {
        "as_of": "演示数据（非实时）",
        "items": [
            {"code": "600519", "name": "贵州茅台", "price": 1686.00, "change_pct": 1.25},
            {"code": "000858", "name": "五粮液", "price": 142.50, "change_pct": -0.83},
            {"code": "300750", "name": "宁德时代", "price": 196.30, "change_pct": 2.15},
            {"code": "601318", "name": "中国平安", "price": 45.80, "change_pct": 0.55},
        ],
    }

