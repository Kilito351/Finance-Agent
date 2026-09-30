import asyncio
import time
from uuid import uuid4

from .llm import LLMClient
from .models import AgentStep, AnalysisRequest, AnalysisResponse, Source
from .monitor import MetricsStore
from .rag import KnowledgeBase


class AgentOrchestrator:
    def __init__(self, knowledge: KnowledgeBase, llm: LLMClient, metrics: MetricsStore) -> None:
        self.knowledge = knowledge
        self.llm = llm
        self.metrics = metrics

    async def _run(self, role: str, query: str, context: str, fallback: str) -> str:
        if not self.llm.enabled:
            return fallback
        started = time.perf_counter()
        try:
            result = await self.llm.complete(
                f"你是严谨的金融{role}。区分事实、假设和不确定性，不编造实时数据；输出简洁中文。",
                f"问题：{query}\n参考资料：{context}",
            )
            self.metrics.record(True, result.prompt_tokens, result.completion_tokens, result.latency_ms)
            return result.text
        except Exception:
            self.metrics.record(False, latency_ms=(time.perf_counter() - started) * 1000)
            return fallback + "\n\n（模型接口暂不可用，已返回本地演示结果。）"

    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        matches = self.knowledge.search(request.query)
        context = "\n".join(f"- {item['title']}：{item['content']}" for item in matches)
        subject = request.stock_code or "目标标的"
        research_fallback = (
            f"围绕 {subject}，建议从收入与利润趋势、盈利质量、现金流、估值分位和行业竞争五个维度验证。"
            "当前 Demo 未连接实时财务数据库，因此不对具体指标数值作推断。"
        )
        risk_fallback = (
            "主要风险包括行业景气变化、估值波动、经营与政策不确定性，以及数据滞后和模型幻觉。"
            "在获得最新公告与行情前，风险等级暂定为中等。"
        )
        research, risk = await asyncio.gather(
            self._run("投研分析师", request.query, context, research_fallback),
            self._run("风险分析师", request.query, context, risk_fallback),
        )
        strategy_fallback = (
            "策略上保持观察并分批验证：先核验最新财报和公告，再比较同业估值；设置单一标的仓位上限，"
            "以基本面恶化或风险事件作为退出条件。"
        )
        strategy = await self._run("投资策略师", request.query, research + "\n" + risk, strategy_fallback)
        report = (
            f"## 分析结论\n\n{research}\n\n## 风险提示\n\n{risk}\n\n## 策略建议\n\n{strategy}"
        )
        self.metrics.record(True)
        return AnalysisResponse(
            trace_id=str(uuid4()),
            mode="llm" if self.llm.enabled else "demo",
            research=AgentStep(agent="researcher", title="投研分析", content=research),
            risk=AgentStep(agent="risk", title="风险评估", content=risk),
            strategy=AgentStep(agent="strategist", title="策略建议", content=strategy),
            report=report,
            sources=[Source(title=item["title"], category=item["category"], score=item["score"]) for item in matches],
        )

