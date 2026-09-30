import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.agents import AgentOrchestrator
from app.config import get_settings
from app.llm import LLMClient
from app.models import AnalysisRequest
from app.monitor import MetricsStore
from app.rag import KnowledgeBase


async def main() -> None:
    orchestrator = AgentOrchestrator(KnowledgeBase(), LLMClient(get_settings()), MetricsStore())
    result = await orchestrator.analyze(
        AnalysisRequest(query="分析贵州茅台的基本面、主要风险与配置策略", stock_code="600519")
    )
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

