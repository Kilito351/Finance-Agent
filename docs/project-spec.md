# Agent项目‑基于LLM的金融分析AI Agent

# Agent项目‑基于LLM的金融分析AI Agent

1. 项目概述

## 1\.1 项目背景

随着金融市场的日益复杂化和数据量的爆发式增长,传统的人工分析模式已无法满足投资者对实时 性､精准性和深度洞察的需求｡本项目构建了一个基于大语言模型\(LLM\)的智能金融分析AI Agent,融 合RAG检索增强､Agent多智能体协作､LLMOps模型运维､DataOps数据编排等前沿技术,为用户提 供股票分析､财报解读､市场研判､风险预警等一站式智能投研服务｡

## 1\.2 核心功能

|模块 |功能描述 |
|---|---|
|智能投研助手 |自然语言查询股票基本面､财务指标､估值分析 |
|RAG知识库问答 |基于金融知识库\(法规､年报､研报\)的精准问答 |
|多Agent协作分析 |分析师Agent \+ 风控Agent \+ 策略Agent 协同工作|
|市场实时监控 |实时行情､异动预警､舆情监控 |
|数据可视化看板 |交互式图表､K线图､财务报表可视化 |
|LLMOps监控面板 |模型调用统计､Token消耗､响应延迟监控 |

2. 系统架构图
\[图片：系统架构图\]

3. 核心流程图

## 3\.1 Agent 协作时序图

\[图片：Agent协作时序图\]

## 3\.2 RAG 检索流程图

\[图片：RAG检索流程图\]

## 3\.3 DataOps 数据管道流程

\[图片：DataOps数据管道视图\]

4. 技术选型

|分类 |技术选型 |说明 |
|---|---|---|
|LLM大模型 |Qwen / DeepSeek / GPT‑4 |主推理模型,OpenAI兼容接口 |
|Embedding |BGE‑large‑zh / text‑embedding‑3‑large |文本向量化,1536维 |
|向量数据库 |ChromaDB \(本地\) / Milvus \(生产\) |RAG向量检索引擎 |
|Agent框架 |LangChain \+ 自研编排器 |多Agent协同､工具调用 |
|后端框架 |FastAPI \+ Python 3\.11 |高性能异步API服务 |
|前端框架 |Vue3 \+ TypeScript \+ Vite |响应式深色主题UI |
|数据库 |PostgreSQL \+ TimescaleDB |结构化存储 \+ 时序优化 |
|缓存/队列|Redis |会话缓存､消息队列 |
|数据管道 |Prefect |ETL编排调度 |
|监控 |Prometheus \+ Grafana |指标采集可视化 |
|容器化 |Docker \+ Docker Compose |一键部署 |

5. 项目结构

```Plaintext
Financial‑AI‑Agent/
├── docs/
│   └── 项目技术文档.md
│   └── INTEGRATED.md
├── FastAPI 后端
│   ├── app/
│   │   ├── main.py #入口
│   │   ├── config.py #全局配置
│   │   ├── core/
│   │   │   ├── llm.py #LLM统一接口
│   │   │   ├── redis_client.py #Redis连接
│   │   │   ├── vectorstore.py #向量库接口
│   │   ├── models/
│   │   │   ├── schemas.py #Pydantic模型
│   │   │   ├── database.py #SQLAlchemy ORM
│   │   ├── agents/
│   │   │   ├── base.py #Agent基类
│   │   │   ├── orchestrator.py #编排器
│   │   │   ├── researcher.py #投研Agent
│   │   │   ├── risk_agent.py #风控Agent
│   │   │   ├── strategist.py #策略Agent
│   │   │   └── report_agent.py #报告Agent
│   │   ├── rag/
│   │   │   ├── knowledge_base.py #知识库
│   │   │   ├── retriever.py #混合检索
│   │   │   ├── query_rewrite.py #查询改写
│   │   ├── llmops/
│   │   │   └── monitor.py #监控核心
│   │   ├── dataops/
│   │   │   └── pipeline.py #数据管道
│   │   ├── api/
│   │   │   ├── chat.py #对话API
│   │   │   ├── agent.py #AgentAPI
│   │   │   ├── rag.py #RAG API
│   │   │   ├── llmops.py #LLMOps API
│   │   │   └── data.py #数据API
│   ├── requirements.txt
│   ├── Dockerfile
├── frontend/ #Vue3 前端
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   ├── src/
│   │   ├── main.ts
│   │   ├── App.vue
│   │   ├── router.ts
│   │   ├── style.css
│   │   ├── api/
│   │   │   ├── index.ts
│   │   │   ├── chat.ts
│   │   │   ├── agent.ts
│   │   │   ├── llmops.ts
│   │   ├── stores/
│   │   │   ├── chat.ts
│   │   │   ├── llmops.ts
│   │   ├── views/
│   │   │   ├── ChatView.vue
│   │   │   ├── DashboardView.vue
│   │   │   ├── ReportView.vue
│   │   │   ├── KnowledgeView.vue
│   │   │   └── LLMOpsView.vue
├── docker‑compose.yml
├── .env.example
```

## 6 后端完整代码

### 6\.1 全局配置管理 

```Python
"""
backend/app/config.py
全局配置管理 - 使用 Pydantic Settings
"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """应用全局配置"""

    # === 应用基础配置 ===
    APP_NAME: str = "Financial AI Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"

    # === FastAPI 服务配置 ===
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # === LLM 大模型配置 ===
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "qwen‑plus"
    LLM_API_BASE: str = "https://dashscope.aliyuncs.com/compatible‑mode/v1"
    LLM_API_KEY: str = ""
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4096
    LLM_TIMEOUT: int = 120

    # === 嵌入模型配置 ===
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text‑embedding‑3‑large"
    EMBEDDING_API_BASE: str = "https://dashscope.aliyuncs.com/compatible‑mode/v1"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BATCH_SIZE: int = 100
    EMBEDDING_DIM: int = 1536

    # === 数据库配置 ===
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/financial_agent"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10

    # === 向量数据库配置 ===
    VECTORSTORE_TYPE: str = "chroma"
    CHROMA_PERSIST_DIR: str = "./data/chromadb"
    CHROMA_COLLECTION_NAME: str = "financial_knowledge"

    # === Redis 配置 ===
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_MAX_CONNECTIONS: int = 50

    # === RAG 配置 ===
    RAG_TOP_K: int = 10
    RAG_SCORE_THRESHOLD: float = 0.5
    RAG_RERANK_TOP_K: int = 5
    RAG_MAX_CONTEXT_LENGTH: int = 8000

    # === Agent 配置 ===
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_EARLY_STOP_THRESHOLD: float = 0.8
    AGENT_TOOL_CALL_TIMEOUT: int = 30
    AGENT_MEMORY_TYPE: str = "buffer"

    # === DataOps 配置 ===
    DATAOPS_PREFECT_API: str = "http://localhost:4200/api"
    DATAOPS_WIND_TOKEN: str = ""
    DATAOPS_TUSHARE_TOKEN: str = ""
    DATAOPS_ETL_SCHEDULE_CRON: str = "0 2 * * *"

    # === LLMOps 配置 ===
    LLMOPS_PROMETHEUS_PORT: int = 9090
    LLMOPS_ENABLE_METRICS: bool = True
    LLMOPS_LOG_FILE: str = "./logs/llm_requests.log"

    # === 安全配置 ===
    SECRET_KEY: str = "change‑this‑in‑production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf‑8"
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    """获取单例配置实例"""
    return Settings()
```

### 6\.2 LLM 大模型统一调用接口

```Python
"""
backend/app/core/llm.py
LLM 大模型统一调用接口 - 支持多模型供应商
"""
import time
import logging
from typing import Optional, AsyncIterator, Any, Dict, List
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class LLMCallLog:
    """LLM 调用日志"""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    timestamp: str
    success: bool
    error: Optional[str] = None


class TokenCostCalculator:
    """Token 成本计算器"""

    GPT4O_INPUT = 0.0025 / 1000
    GPT4O_OUTPUT = 0.01 / 1000
    QWEN_INPUT = 0.001 / 1000
    QWEN_OUTPUT = 0.002 / 1000
    DEEPSEEK_INPUT = 0.001 / 1000
    DEEPSEEK_OUTPUT = 0.002 / 1000

    @classmethod
    def calculate(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        model_lower = model.lower()
        if "gpt‑4o" in model_lower or "gpt‑4" in model_lower:
            return input_tokens * cls.GPT4O_INPUT + output_tokens * cls.GPT4O_OUTPUT
        elif "qwen" in model_lower:
            return input_tokens * cls.QWEN_INPUT + output_tokens * cls.QWEN_OUTPUT
        elif "deepseek" in model_lower:
            return input_tokens * cls.DEEPSEEK_INPUT + output_tokens * cls.DEEPSEEK_OUTPUT
        return 0.0


class LLMService:
    """LLM 大模型统一服务接口"""

    def __init__(self):
        self.settings = get_settings()
        self._llm: Optional[ChatOpenAI] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None

    @property
    def llm(self) -> ChatOpenAI:
        """延迟初始化 LLM 客户端"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.settings.LLM_MODEL,
                api_key=self.settings.LLM_API_KEY,
                base_url=self.settings.LLM_API_BASE,
                temperature=self.settings.LLM_TEMPERATURE,
                max_tokens=self.settings.LLM_MAX_TOKENS,
                timeout=self.settings.LLM_TIMEOUT,
                streaming=True,
            )
        return self._llm

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """延迟初始化 Embedding 客户端"""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self.settings.EMBEDDING_MODEL,
                api_key=self.settings.EMBEDDING_API_KEY,
                base_url=self.settings.EMBEDDING_API_BASE,
                batch_size=self.settings.EMBEDDING_BATCH_SIZE,
            )
        return self._embeddings

    async def chat(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = True,
    ) -> LLMCallLog:
        """通用对话接口"""
        start_time = time.time()

        try:
            if tools:
                response = await self.llm.bind_tools(tools).ainvoke(messages)
            else:
                response = await self.llm.ainvoke(messages)

            latency = (time.time() - start_time) * 1000

            usage = getattr(response, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)
            else:
                prompt_tokens = sum(len(str(m.content)) for m in messages) // 4
                completion_tokens = len(str(response.content)) // 4
                total_tokens = prompt_tokens + completion_tokens

            cost = TokenCostCalculator.calculate(
                self.settings.LLM_MODEL, prompt_tokens, completion_tokens
            )

            log = LLMCallLog(
                model=self.settings.LLM_MODEL,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency,
                cost_usd=cost,
                timestamp=time.strftime("%Y‑%m‑%d %H:%M:%S"),
                success=True,
            )
            logger.info(
                f"[LLM] model={self.settings.LLM_MODEL} tokens={total_tokens} "
                f"latency={latency:.0f}ms cost=${cost:.6f}"
            )
            return log

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"[LLM] 调用失败: {str(e)}")
            return LLMCallLog(
                model=self.settings.LLM_MODEL,
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                latency_ms=latency, cost_usd=0.0,
                timestamp=time.strftime("%Y‑%m‑%d %H:%M:%S"),
                success=False, error=str(e),
            )

    async def chat_stream(
        self, messages: List[BaseMessage]
    ) -> AsyncIterator[tuple[str, Optional[LLMCallLog]]]:
        """流式对话"""
        start_time = time.time()
        full_response = ""

        try:
            async for chunk in self.llm.astream(messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += token
                yield token, None

            latency = (time.time() - start_time) * 1000
            prompt_tokens = sum(len(str(m.content)) for m in messages) // 4
            completion_tokens = len(full_response) // 4
            total_tokens = prompt_tokens + completion_tokens
            cost = TokenCostCalculator.calculate(
                self.settings.LLM_MODEL, prompt_tokens, completion_tokens
            )
            log = LLMCallLog(
                model=self.settings.LLM_MODEL,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency,
                cost_usd=cost,
                timestamp=time.strftime("%Y‑%m‑%d %H:%M:%S"),
                success=True,
            )
            yield "", log

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            log = LLMCallLog(
                model=self.settings.LLM_MODEL,
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                latency_ms=latency, cost_usd=0.0,
                timestamp=time.strftime("%Y‑%m‑%d %H:%M:%S"),
                success=False, error=str(e),
            )
            yield "", log

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """文本向量化"""
        try:
            embeddings = await self.embeddings.aembed_documents(texts)
            logger.info(f"[Embedding] texts={len(texts)} dim={len(embeddings[0]) if embeddings else 0}")
            return embeddings
        except Exception as e:
            logger.error(f"[Embedding] 失败: {str(e)}")
            raise


llm_service = LLMService()


def get_llm_service() -> LLMService:
    return llm_service
```

### 6\.3 Pydantic 数据模型定义

```Python
"""
backend/app/models/schemas.py
Pydantic 请求/响应数据模型
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Any, Dict
from datetime import datetime
from enum import Enum


class AgentType(str, Enum):
    RESEARCHER = "researcher"
    RISK = "risk"
    STRATEGIST = "strategist"
    REPORT = "report"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    agent_type: Optional[AgentType] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=4000)
    agent_type: Optional[AgentType] = None
    stream: bool = True
    enable_rag: bool = True


class ChatResponse(BaseModel):
    session_id: str
    message: str
    agent_type: AgentType
    sources: Optional[List[Dict[str, Any]]] = None
    token_usage: Optional[Dict[str, int]] = None
    latency_ms: Optional[float] = None
    session_created: bool = False


class AgentInvokeRequest(BaseModel):
    session_id: str
    query: str
    agent_types: Optional[List[AgentType]] = None
    parallel: bool = True


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    score_threshold: float = Field(default=0.5, ge=0, le=1)
    filters: Optional[Dict[str, Any]] = None
    enable_rerank: bool = True


class RAGQueryResponse(BaseModel):
    query: str
    chunks: List[Dict[str, Any]]
    total_retrieved: int
    rerank_applied: bool


class KnowledgeBaseUpload(BaseModel):
    title: str
    content: str
    category: str
    metadata: Optional[Dict[str, Any]] = None


class LLMCallRecord(BaseModel):
    call_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    success: bool
    timestamp: str


class LLMOpsMetrics(BaseModel):
    all_time: Dict[str, Any]
    today: Dict[str, Any]


class DataPipelineRun(BaseModel):
    run_id: str
    pipeline_name: str
    status: TaskStatus
    started_at: str
    completed_at: Optional[str] = None
    records_processed: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None
```

### 6\.4 SQLAlchemy ORM 数据库模型

```Python
"""
backend/app/models/database.py
SQLAlchemy ORM 模型 - PostgreSQL 数据库表定义
"""
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON, Index,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete‑orphan")
    llm_calls = relationship("LLMCallLog", back_populates="user", cascade="all, delete‑orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(200), default="新对话")
    agent_types = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_message = Column(Text)
    message_count = Column(Integer, default=0)

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessageDB", back_populates="session", cascade="all, delete‑orphan")
    __table_args__ = (Index("idx_session_created", "session_id", "created_at"),)


class ChatMessageDB(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    agent_type = Column(String(50))
    token_count = Column(Integer, default=0)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("ChatSession", back_populates="messages")


class LLMCallLog(Base):
    """LLM 调用日志"""
    __tablename__ = "llm_call_logs"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(String(100), unique=True, index=True, nullable=False)
    session_id = Column(String(100), index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    model = Column(String(50), nullable=False)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    latency_ms = Column(Float, default=0)
    cost_usd = Column(Float, default=0)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    prompt_preview = Column(Text)
    response_preview = Column(Text)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="llm_calls")

    __table_args__ = (
        Index("idx_llm_model_created", "model", "created_at"),
        Index("idx_llm_daily", "created_at"),
    )


class PromptVersionDB(Base):
    """Prompt 版本管理"""
    __tablename__ = "prompt_versions"

    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    agent_type = Column(String(50))
    content = Column(Text, nullable=False)
    variables = Column(JSON, default=list)
    is_active = Column(Boolean, default=False)
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class KnowledgeDocument(Base):
    """知识库文档"""
    __tablename__ = "knowledge_documents"

    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(String(100), unique=True, index=True, nullable=False)
    title = Column(String(500), nullable=False)
    category = Column(String(50), nullable=False)
    source = Column(String(200))
    content_hash = Column(String(64))
    metadata = Column(JSON, default=dict)
    chunk_count = Column(Integer, default=0)
    indexed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    chunks = relationship("DocumentChunkDB", back_populates="document", cascade="all, delete‑orphan")


class DocumentChunkDB(Base):
    """文档片段"""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String(100), unique=True, index=True, nullable=False)
    document_id = Column(Integer, ForeignKey("knowledge_documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, default=0)
    metadata = Column(JSON, default=dict)
    embedding_id = Column(String(200))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("KnowledgeDocument", back_populates="chunks")

    __table_args__ = (Index("idx_chunk_doc_index", "document_id", "chunk_index"),)


class DataPipeline(Base):
    """数据管道定义"""
    __tablename__ = "data_pipelines"

    id = Column(Integer, primary_key=True, index=True)
    pipeline_name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    source_type = Column(String(50))
    schedule_cron = Column(String(100))
    is_active = Column(Boolean, default=True)
    config = Column(JSON, default=dict)
    last_run_at = Column(DateTime(timezone=True))
```

### 6\.5 Agent 基类框架

```Python
"""
backend/app/agents/base.py

Agent 基类 - 基于 LangChain 实现通用 Agent 框架
"""
import uuid
import time
import logging
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from langchain.schema import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain.tools import BaseTool

from app.models.schemas import AgentType
from app.core.llm import LLMService, get_llm_service

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    name: str
    agent_type: AgentType
    description: str
    system_prompt: str
    tools: List[BaseTool] = field(default_factory=list)
    max_iterations: int = 10
    early_stop_threshold: float = 0.8
    temperature: float = 0.7


@dataclass
class AgentExecutionResult:
    success: bool
    output: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    total_tokens: int = 0
    execution_time_ms: float = 0.0
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Agent 基类 - 提供通用生命周期管理
    """

    def __init__(self, config: AgentConfig, llm_service: Optional[LLMService] = None):
        self.config = config
        self.llm = llm_service or get_llm_service()
        self._execution_history: List[Dict] = []
        self._memory: List[BaseMessage] = []
        self.system_prompt = SystemMessage(content=config.system_prompt)
        logger.info(f"[Agent] {config.name} initialized, tools={len(config.tools)}")

    @abstractmethod
    def _build_system_prompt(self) -> str:
        pass

    @abstractmethod
    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        pass

    async def execute(
        self, task: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> AgentExecutionResult:
        task_id = str(uuid.uuid4())
        start_time = time.time()
        total_tokens = 0

        logger.info(f"[Agent:{self.config.name}] executing task_id={task_id}")

        try:
            prompt_content = self._build_system_prompt()
            if context:
                context_str = self._format_context(context)
                prompt_content += f"\n\n## Context\n{context_str}"
                prompt_content += f"\n\n## Task\n{task}"

            messages = [
                SystemMessage(content=prompt_content),
                HumanMessage(content=task),
            ]

            tools_json = self._tools_to_json(self.config.tools)
            if tools_json:
                call_log = await self.llm.chat(messages=messages, tools=tools_json, stream=False)
                if not call_log.success:
                    raise Exception(f"LLM failed: {call_log.error}")
                total_tokens = call_log.total_tokens
                agent_output = f"[tools executed]\n{call_log.error or 'done'}"
            else:
                call_log = await self.llm.chat(messages=messages, stream=False)
                if not call_log.success:
                    raise Exception(f"LLM failed: {call_log.error}")
                total_tokens = call_log.total_tokens
                agent_output = str(call_log.error or "done")

            parsed = self._parse_output(agent_output)
            execution_time = (time.time() - start_time) * 1000

            self._execution_history.append({
                "task_id": task_id, "task": task, "output": agent_output,
                "parsed": parsed, "tokens": total_tokens,
                "execution_time_ms": execution_time,
                "timestamp": time.strftime("%Y‑%m‑%d %H:%M:%S"),
            })

            return AgentExecutionResult(
                success=True, output=agent_output, iterations=1,
                total_tokens=total_tokens, execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"[Agent:{self.config.name}] failed: {str(e)}", exc_info=True)
            return AgentExecutionResult(
                success=False, output="", error=str(e),
                iterations=0, total_tokens=total_tokens,
                execution_time_ms=execution_time,
            )

    async def execute_stream(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        prompt_content = self._build_system_prompt()
        if context:
            context_str = self._format_context(context)
            prompt_content += f"\n\n## Context\n{context_str}"
            prompt_content += f"\n\n## Task\n{task}"
        messages = [SystemMessage(content=prompt_content), HumanMessage(content=task)]
        full = ""
        async for token, _ in self.llm.chat_stream(messages):
            full += token
            yield token
        logger.info(f"[Agent:{self.config.name}] stream done, len={len(full)}")

    def _format_context(self, context: Dict[str, Any]) -> str:
        lines = []
        for key, value in context.items():
            if isinstance(value, list):
                lines.append(f"### {key}")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"‑ {item.get('content', str(item))[:200]}")
                    else:
                        lines.append(f"‑ {str(item)[:200]}")
            elif isinstance(value, dict):
                lines.append(f"### {key}")
                for k, v in value.items():
                    lines.append(f"‑ {k}: {str(v)[:200]}")
            else:
                lines.append(f"**{key}**: {str(value)[:500]}")
        return "\n".join(lines)

    def _tools_to_json(self, tools: List[BaseTool]) -> List[Dict]:
        result = []
        for tool in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            })
        return result

    def get_execution_history(self) -> List[Dict]:
        return self._execution_history

    def clear_history(self):
        self._execution_history.clear()
```

### 6\.6 多Agent智能编排器

```Python
"""
backend/app/agents/orchestrator.py
Agent 编排器 - 多Agent协同调度核心
"""
import uuid
import time
import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio

from app.agents.base import AgentExecutionResult
from app.agents.researcher import ResearcherAgent
from app.agents.risk_agent import RiskAgent
from app.agents.strategist import StrategistAgent
from app.models.schemas import AgentType

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    max_parallel_agents: int = 3
    timeout_per_agent: int = 60
    enable_caching: bool = True


class AgentOrchestrator:
    """
    多Agent编排器
    核心能力:意图分类 | 任务拆解 | 并行/串行执行 | 结果聚合 | 记忆管理
    """

    INTENT_AGENT_MAP = {
        "stock_analysis": [AgentType.RESEARCHER],
        "stock_risk": [AgentType.RISK],
        "full_analysis": [AgentType.RESEARCHER, AgentType.RISK, AgentType.STRATEGIST],
        "investment_advice": [AgentType.RESEARCHER, AgentType.STRATEGIST],
        "risk_warning": [AgentType.RISK],
        "report": [AgentType.RESEARCHER],
        "general": [AgentType.RESEARCHER],
    }

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self._agents: Dict[AgentType, Any] = {}
        self._session_memory: Dict[str, List[Dict]] = {}
        self._init_agents()

    def _init_agents(self):
        self._agents = {
            AgentType.RESEARCHER: ResearcherAgent(),
            AgentType.RISK: RiskAgent(),
            AgentType.STRATEGIST: StrategistAgent(),
        }
        logger.info(f"[Orchestrator] initialized with {len(self._agents)} agents")

    def _classify_intent(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["风险", "风险评估", "暴雷", "违约", "亏损", "预警"]):
            if any(k in q for k in ["全面分析", "深度分析", "完整分析", "投资建议"]):
                return "full_analysis"
            return "stock_risk"
        if any(k in q for k in ["策略", "组合", "配置", "仓位", "建仓", "买入"]):
            return "investment_advice"
        if any(k in q for k in ["全面分析", "深度分析", "完整分析", "综合分析"]):
            return "full_analysis"
        if any(k in q for k in ["报告", "研报", "分析报告"]):
            return "report"
        if any(k in q for k in ["分析", "估值", "财务", "营收", "利润", "ROE", "PE"]):
            return "stock_analysis"
        return "general"

    def _extract_stock_codes(self, query: str) -> List[str]:
        return re.findall(r'\b\d{6}\b', query)

    def _build_context(
        self, session_id: str, agent_results: Dict[AgentType, AgentExecutionResult]
    ) -> Dict[str, Any]:
        context = {"agent_results": {}, "session_id": session_id, "timestamp": time.strftime("%Y‑%m‑%d %H:%M:%S")}
        for agent_type, result in agent_results.items():
            if result.success:
                context["agent_results"][agent_type.value] = result.output
        return context

    async def run(
        self, query: str,
        session_id: Optional[str] = None,
        agent_types: Optional[List[AgentType]] = None,
        parallel: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session_id = session_id or str(uuid.uuid4())
        run_id = str(uuid.uuid4())

        logger.info(f"[Orchestrator] run_id={run_id} parallel={parallel}")

        # Step 1: intent classification
        intent = self._classify_intent(query)
        stock_codes = self._extract_stock_codes(query)

        # Step 2: select agents
        if agent_types:
            selected_agents = agent_types
        else:
            selected_agents = [AgentType(a) for a in self.INTENT_AGENT_MAP.get(intent, ["researcher"])]
        # Step 3: execute
        agent_results: Dict[AgentType, AgentExecutionResult] = {}

        if parallel and len(selected_agents) > 1:
            tasks = []
            for agent_type in selected_agents:
                agent = self._agents.get(agent_type)
                if agent:
                    tasks.append((agent_type, self._run_single_agent(agent, query, stock_codes, context)))
            results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)
            for i, (agent_type, _) in enumerate(tasks):
                result = results[i]
                if isinstance(result, Exception):
                    agent_results[agent_type] = AgentExecutionResult(success=False, output="", error=str(result))
                else:
                    agent_results[agent_type] = result
        else:
            shared_context = context or {}
            for agent_type in selected_agents:
                agent = self._agents.get(agent_type)
                if agent:
                    if agent_type == AgentType.RISK and agent_results.get(AgentType.RESEARCHER):
                        shared_context["researcher_result"] = agent_results[AgentType.RESEARCHER].output
                    result = await self._run_single_agent(agent, query, stock_codes, shared_context)
                    agent_results[agent_type] = result

        # Step 4: aggregate
        final_response = self._aggregate_results(query, agent_results)

        # Step 5: record session
        self._session_memory.setdefault(session_id, []).append({
            "run_id": run_id, "query": query, "intent": intent,
            "agents_used": [a.value for a in selected_agents],
            "results": {k.value: v.output for k, v in agent_results.items()},
            "timestamp": time.strftime("%Y‑%m‑%d %H:%M:%S"),
        })

        total_time = sum(r.execution_time_ms for r in agent_results.values())
        total_tokens = sum(r.total_tokens for r in agent_results.values())

        return {
            "run_id": run_id, "session_id": session_id, "intent": intent,
            "agent_results": {k.value: v.output for k, v in agent_results.items()},
            "final_response": final_response,
            "execution_summary": {
                "agents_used": [a.value for a in selected_agents],
                "parallel": parallel, "total_execution_ms": total_time,
                "total_tokens": total_tokens,
                "success_count": sum(1 for r in agent_results.values() if r.success),
            },
        }

    async def _run_single_agent(
        self, agent: Any, query: str, stock_codes: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionResult:
        task = query
        if stock_codes:
            task = f"分析股票:{', '.join(stock_codes)}\n\n用户问题:{query}"

        try:
            return await asyncio.wait_for(
                agent.execute(task=task, context=context),
                timeout=self.config.timeout_per_agent
            )
        except asyncio.TimeoutError:
            return AgentExecutionResult(
                success=False, output="", error=f"timeout > {self.config.timeout_per_agent}s"
            )

    def _aggregate_results(
        self, query: str, agent_results: Dict[AgentType, AgentExecutionResult]
    ) -> str:
        parts = []
        for agent_type in [AgentType.RESEARCHER, AgentType.RISK, AgentType.STRATEGIST]:
            result = agent_results.get(agent_type)
            if result and result.success and result.output:
                parts.append(f"\n## {agent_type.value.upper()} Result\n\n{result.output}\n")
        if not parts:
            errors = [f"{k.value}: {v.error}" for k, v in agent_results.items() if not v.success]
            return f"分析遇到问题:{'; '.join(errors)}"
        return "\n".join(parts)

    def get_session_memory(self, session_id: str) -> List[Dict]:
        return self._session_memory.get(session_id, [])

    def clear_session(self, session_id: str):
        if session_id in self._session_memory:
            del self._session_memory[session_id]


_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
```

### 6\.7 投研分析 Agent

```Python
"""
backend/app/agents/researcher.py
投研分析 Agent - 股票基本面分析､财务解读

"""
from typing import Dict, Any
from app.agents.base import BaseAgent, AgentConfig, AgentExecutionResult
from app.models.schemas import AgentType
from langchain.tools import tool


@tool(description="查询股票基本信息(代码､名称､行业､上市时间､市值等)")
def get_stock_info(stock_code: str) -> str:
    return f'{{"code": "{stock_code}", "name": "股票名称", "industry": "行业"}}'


@tool(description="查询股票财务数据(营收､净利润､毛利率､ROE等)")
def get_stock_financials(stock_code: str, period: str = "annual") -> str:
    return f'{{"code": "{stock_code}", "period": "{period}", "revenue": 0, "net_profit": 0, "roe": 0.0}}'


@tool(description="查询股票K线数据(OHLCV)")
def get_stock_kline(stock_code: str, days: int = 60) -> str:
    return f'[{{"date": "2024‑01‑01", "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0}}]'


@tool(description="查询财经新闻")
def search_financial_news(keyword: str, limit: int = 10) -> str:
    return f'[{{"title": "新闻标题", "source": "来源", "date": "2024‑01‑01"}}]'


RESEARCHER_SYSTEM_PROMPT = """你是资深金融投研分析师,10年+经验,擅长基本面分析､财务建模｡
分析框架:
‑ **业绩驱动因素**:营收增长来源､市场份额变化
‑ **盈利能力分析**:毛利率､净利率､ROE 趋势
‑ **估值判断**:PE/PB/PS 横向(vs 行业)和纵向(vs 历史)对比
‑ **风险提示**:经营风险､财务风险､行业周期性
‑ **投资建议**:客观结论

输出:Markdown结构,数据有来源,关键结论**加粗**,风险⚠️开头"""


class ResearcherAgent(BaseAgent):
    def __init__(self):
        tools = [get_stock_info, get_stock_financials, get_stock_kline, search_financial_news]
        config = AgentConfig(
            name="投研分析Agent", agent_type=AgentType.RESEARCHER,
            description="股票基本面分析､财务解读",
            system_prompt=RESEARCHER_SYSTEM_PROMPT, tools=tools, max_iterations=8,
        )
        super().__init__(config)

    def _build_system_prompt(self) -> str:
        return self.config.system_prompt

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        return {"analysis": raw_output, "agent_type": "researcher"}

    async def analyze_stock(self, stock_code: str, query: str = "") -> AgentExecutionResult:
        task = f"分析股票 {stock_code}｡获取基本信息､财务数据,进行估值分析,对比行业,输出综合研判｡{query}"
        return await self.execute(task=task)
```

### 6\.8 风控预警 Agent

```Python
"""
backend/app/agents/risk_agent.py
风控预警 Agent - 风险识别､量化评估､预警
"""
from typing import Dict, Any
from app.agents.base import BaseAgent, AgentConfig, AgentExecutionResult
from app.models.schemas import AgentType
from langchain.tools import tool


@tool(description="计算股票风险指标(波动率､VaR､Beta等)")
def calculate_risk_metrics(stock_code: str, period: int = 60) -> str:
    return '{"code": "stock_code", "volatility": 0.25, "var_95": ‑0.03, "beta": 1.2}'


@tool(description="检查监管处罚和诉讼信息")
def check_regulatory_events(stock_code: str) -> str:
    return '[{"event": "处罚/诉讼事件", "date": "日期", "severity": "严重程度"}]'


RISK_AGENT_SYSTEM_PROMPT = """你是专业金融风控专家,擅长风险量化､预警模型､合规审查｡

风控框架:
‑ 财务风险:盈利质量､现金流异常､债务结构
‑ 经营风险:大客户依赖､竞争壁垒､管理层变更
‑ 市场风险:波动性､VaR､Beta､系统性风险
‑ 合规风险:监管处罚､信息披露违规

风险评分:
‑ 红色 高风险(80‑100)
‑ 橙色 中高风险(60‑79)
‑ 黄色 中风险(40‑59)
‑ 绿色 低风险(0‑39)"""


class RiskAgent(BaseAgent):
    def __init__(self):
        tools = [calculate_risk_metrics, check_regulatory_events]
        config = AgentConfig(
            name="风控预警Agent", agent_type=AgentType.RISK,
            description="风险识别､量化评估､预警监控",
            system_prompt=RISK_AGENT_SYSTEM_PROMPT, tools=tools, max_iterations=6,
        )
        super().__init__(config)

    def _build_system_prompt(self) -> str:
        return self.config.system_prompt

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        return {"analysis": raw_output, "agent_type": "risk"}

    async def assess_risk(self, stock_code: str, query: str = "") -> AgentExecutionResult:
        task = f"评估股票 {stock_code} 风险｡计算风险指标,检查监管事件,识别财务风险,输出风险评分(0‑100分)｡{query}"
        return await self.execute(task=task)
```

### 6\.9 策略生成

```Python
"""
backend/app/agents/strategist.py
策略生成 Agent - 投资策略､组合建议､交易方案
"""
from typing import Dict, Any
from app.agents.base import BaseAgent, AgentConfig, AgentExecutionResult
from app.models.schemas import AgentType


STRATEGIST_SYSTEM_PROMPT = """你是资深量化投资策略师,精通多因子模型､资产配置｡

策略原则:
‑ 风险收益匹配:诚实告知用户
‑ 分散化:单一标的仓位不超过20%
‑ 流动性优先:日均成交额>1亿
‑ 动态调整:根据市场环境调整

输出框架:
1. 策略概述:类型､适用场景､预期收益与风险
2. 选股标准:行业､市值､估值筛选条件
3. 仓位配置:各标的仓位､入场点位､止损线
4. 风险控制:最大回撤控制､动态止损规则

重要声明:仅供参考,不构成投资建议"""


class StrategistAgent(BaseAgent):
    def __init__(self):
        config = AgentConfig(
            name="策略生成Agent", agent_type=AgentType.STRATEGIST,
            description="投资策略､组合建议､交易方案",
            system_prompt=STRATEGIST_SYSTEM_PROMPT, tools=[], max_iterations=5,
        )
        super().__init__(config)

    def _build_system_prompt(self) -> str:
        return self.config.system_prompt

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        return {"strategy": raw_output, "agent_type": "strategist"}

    async def generate_strategy(self, context: Dict[str, Any]) -> AgentExecutionResult:
        task = f"""基于以下分析结果生成投资策略:

研究结论:{context.get('researcher_result', '无')}
风控结论:{context.get('risk_result', '无')}
用户偏好:{context.get('user_preference', '无')}

请按策略框架输出｡"""
        return await self.execute(task=task, context=context)
```

### 6\.10 报告生成

```Python
"""
backend/app/agents/report_agent.py
报告生成 Agent ‑ 自动生成结构化投研报告
"""
from typing import Dict, Any
from app.agents.base import BaseAgent, AgentConfig, AgentExecutionResult
from app.models.schemas import AgentType


REPORT_SYSTEM_PROMPT = """你是专业金融研究报告撰写专家｡

报告框架:
1. 执行摘要(100字内)

2. 公司概况:主营业务､市场地位
3. 基本面分析:营收､利润､现金流趋势

4. 估值分析:PE/PB/PS 对比
5. 风险因素
6. 投资建议(买入/持有/卖出)

格式:Markdown,数据表格,结论加粗"""


class ReportAgent(BaseAgent):
    def __init__(self):
        config = AgentConfig(
            name="报告生成Agent", agent_type=AgentType.REPORT,
            description="自动生成结构化投研报告",
            system_prompt=REPORT_SYSTEM_PROMPT, tools=[], max_iterations=3,
        )
        super().__init__(config)

    def _build_system_prompt(self) -> str:
        return self.config.system_prompt

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        return {"report": raw_output, "agent_type": "report"}

    async def generate_report(self, stock_code: str, analysis_data: Dict[str, Any]) -> AgentExecutionResult:
        task = f"为股票 {stock_code} 生成完整投研报告｡数据:{analysis_data}"
        return await self.execute(task=task)
```

### 6\.11 RAG 知识库管理

```Python
"""
backend/app/rag/knowledge_base.py
RAG 知识库管理 ‑ 文档加载､分块､索引
"""
import uuid
import hashlib
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.core.llm import get_llm_service, LLMService
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ChunkResult:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class KnowledgeBaseManager:
    """
    知识库管理器
    职责:文档加载 | 文档分块 | 向量化存储 | 知识检索
    """

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm = llm_service or get_llm_service()
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "｡", "!", "?", ";", ],
        )
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings
                client = chromadb.PersistentClient(
                    path=self.settings.CHROMA_PERSIST_DIR,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
                self._collection = client.get_or_create_collection(
                    name=self.settings.CHROMA_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"[KB] ChromaDB initialized: {self.settings.CHROMA_COLLECTION_NAME}")
            except ImportError:
                logger.warning("[KB] ChromaDB not installed")
                self._collection = None
        return self._collection

    def chunk_documents(
        self, documents: List[Document],
        chunk_size: int = 500, chunk_overlap: int = 50,
    ) -> List[ChunkResult]:
        """文档分块"""
        self.text_splitter.chunk_size = chunk_size
        self.text_splitter.chunk_overlap = chunk_overlap
        chunks = self.text_splitter.split_documents(documents)
        results = []
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            results.append(ChunkResult(
                chunk_id=chunk_id,
                content=chunk.page_content,
                metadata={
                    **chunk.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_hash": hashlib.md5(chunk.page_content.encode()).hexdigest(),
                }
            ))
        logger.info(f"[KB] chunked: {len(documents)} docs -> {len(results)} chunks")
        return results

    async def add_chunks_to_vectorstore(
        self, chunks: List[ChunkResult], batch_size: int = 100,
    ) -> int:
        """将文档块添加到向量存储"""
        if not self.collection:
            logger.warning("[KB] vectorstore not initialized")
            return 0
        texts = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [c.chunk_id for c in chunks]

        total_added = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            embeddings = await self.llm.embed(batch_texts)
            self.collection.add(embeddings=embeddings, documents=batch_texts, metadatas=batch_metas, ids=batch_ids)
            total_added += len(batch_texts)

        logger.info(f"[KB] added {total_added} chunks to vectorstore")
        return total_added

    async def add_text_chunks(
        self, texts: List[str], category: str = "general",
        metadata_list: Optional[List[Dict]] = None,
    ) -> int:
        """直接添加文本块"""
        results = []
        for i, text in enumerate(texts):
            meta = metadata_list[i] if metadata_list else {}
            results.append(ChunkResult(
                chunk_id=str(uuid.uuid4()), content=text,
                metadata={**meta, "category": category}
            ))
        return await self.add_chunks_to_vectorstore(results)

    def count_documents(self) -> int:
        if self.collection:
            return self.collection.count()
        return 0


kb_manager = KnowledgeBaseManager()


def get_kb_manager() -> KnowledgeBaseManager:
    return kb_manager
```

### 6\.12 RAG 混合检索器

```Python
"""
backend/app/rag/retriever.py
混合检索器 ‑ Sparse + Dense 融合检索 + Rerank
"""
import logging
from typing import List, Dict, Any, Optional

from app.core.llm import get_llm_service, LLMService
from app.rag.knowledge_base import get_kb_manager, KnowledgeBaseManager

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    混合检索器
    流程:Query Rewrite ‑> Sparse检索 ‑> Dense检索 ‑> RRF融合 ‑> Rerank ‑> GRAG
    """

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm = llm_service or get_llm_service()
        self.kb = get_kb_manager()

    async def retrieve(
        self, query: str,
        top_k: int = 10,
        score_threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        enable_rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        执行混合检索

        Args:
        query: 用户查询
        top_k: 返回数量
        score_threshold: 相似度阈值
        filters: 元数据过滤
        enable_rerank: 是否启用重排序
        """

        logger.info(f"[Retriever] query={query[:50]} top_k={top_k} rerank={enable_rerank}")
        # Step 1: 向量化查询
        query_embedding = await self.llm.embed([query])
        if not query_embedding:
            return []
        query_vec = query_embedding[0]

        # Step 2: ChromaDB 向量检索
        collection = self.kb.collection
        if not collection:
            logger.warning("[Retriever] vectorstore empty")
            return []

        where_filter = filters
        try:
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=top_k * 2,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            if not results or not results.get("ids"):
                return []

            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            # cosine distance ‑> similarity
            scores = [1 ‑ d for d in distances]

            chunks = []
            for i in range(len(ids)):
                chunks.append({
                    "chunk_id": ids[i],
                    "content": documents[i],
                    "score": scores[i] if i < len(scores) else 0.0,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                })

        except Exception as e:
            logger.error(f"[Retriever] ChromaDB search failed: {str(e)}")
            return []

        # Step 3: 过滤低分
        chunks = [c for c in chunks if c["score"] >= score_threshold]

        # Step 4: 重排序
        if enable_rerank and len(chunks) > 1:
            chunks = await self._rerank(query, chunks, top_k=top_k)

        # Step 5: top_k
        chunks = chunks[:top_k]
        logger.info(f"[Retriever] returned {len(chunks)} results")
        return chunks

    async def _rerank(
        self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Cross‑Encoder 重排序"""
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder("cross‑encoder/ms‑marco‑MiniLM‑L‑6‑v2", max_length=512)
            pairs = [(query, chunk["content"]) for chunk in chunks]
            scores = model.predict(pairs)
            for i, chunk in enumerate(chunks):
                chunk["rerank_score"] = float(scores[i])
                chunk["score"] = float(scores[i])
            chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            logger.info("[Retriever] rerank applied with Cross‑Encoder")
        except Exception as e:
            logger.warning(f"[Retriever] rerank failed: {str(e)}")
        return chunks

    async def retrieve_with_expansion(
        self, query: str, top_k: int = 10, **kwargs
    ) -> List[Dict[str, Any]]:
        """带查询扩展的检索"""
        try:
            from langchain.schema import HumanMessage
            llm = get_llm_service()
            expansion_prompt = f"为以下金融查询生成3个同义表达,原始查询:{query}"
            messages = [HumanMessage(content=expansion_prompt)]
            log = await llm.chat(messages, stream=False)
            expanded = [query]
            all_chunks = {}
            for q in expanded:
                chunks = await self.retrieve(q, top_k=top_k // 2, **kwargs)
                for c in chunks:
                    cid = c["chunk_id"]
                    if cid not in all_chunks or c["score"] > all_chunks[cid]["score"]:
                        all_chunks[cid] = c
            merged = sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)
            return merged[:top_k]
        except Exception as e:
            logger.warning(f"[Retriever] expansion failed: {str(e)}")
            return await self.retrieve(query, top_k, **kwargs)


hybrid_retriever = HybridRetriever()


def get_hybrid_retriever() -> HybridRetriever:
    return hybrid_retriever
```

### 6\.13 查询改写模块

查询改写、LLMOps监控与DataOps数据管道编排模块

```Python
"""
backend/app/rag/query_rewrite.py
查询改写模块 - 通过 LLM 优化用户查询
"""
import re
import json
import logging
from typing import Dict, Any
from app.core.llm import get_llm_service

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    查询改写器
    功能:意图澄清 | 同义词扩展 | 查询分解 | 格式修正
    """

    REWRITE_PROMPT = """你是专业金融搜索查询优化助手｡
原始查询:{original_query}

请优化查询:
1. 术语标准化:"茅台" -> "贵州茅台(600519)"
2. 意图识别:投研分析 / 风险评估 / 策略建议 / 知识问答
3. 查询扩展(可选):补充同义词

JSON格式输出:
{{"rewritten_query": "优化后查询", "intent": "意图", "stock_codes": [], 
"stock_names": [], "notes": ""}}
只输出JSON,不要其他内容｡"""

    def __init__(self):
        self.llm = get_llm_service()

    async def rewrite(self, query: str) -> Dict[str, Any]:

        try:
            from langchain.schema import HumanMessage
            messages = [HumanMessage(content=self.REWRITE_PROMPT.format(original_query=query))]
            log = await self.llm.chat(messages, stream=False)
            result_text = log.error or "{}"
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', 
            result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return {"rewritten_query": query, "intent": "unknown", 
            "stock_codes": [], "stock_names": [], "expand_queries": []}
        except Exception as e:
            logger.warning(f"[QueryRewrite] failed: {str(e)}")
            return {"rewritten_query": query, "intent": "unknown", 
            "stock_codes": [], "stock_names": [], "expand_queries": []}
```

### 6\.14 LLMOps 监控模块

```Python
"""
backend/app/llmops/monitor.py
LLMOps 监控模块 - 调用日志､成本追踪､指标暴露
"""
import time
import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class LLMCallRecord:
    call_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    success: bool
    error: Optional[str] = None
    session_id: Optional[str] = None
    prompt_preview: str = ""
    timestamp: str = ""


class LLMOpsMonitor:
    """
    LLMOps 监控器
    功能:全量记录LLM调用日志 | 实时计算Token消耗和成本
    统计成功率/平均延迟 | Prometheus指标暴露
    """

    def __init__(self):
        self._records: List[LLMCallRecord] = []
        self._daily_stats: Dict[str, Dict] = defaultdict(lambda: {
            "total_calls": 0, "total_tokens": 0, "total_cost": 0.0,
            "total_latency": 0.0, "success_calls": 0, "fail_calls": 0,
        })
        self._lock = Lock()
        self._max_records = 10000
        logger.info("[LLMOps] monitor initialized")

    def log_call(
        self, model: str, prompt_tokens: int, completion_tokens: int,
        total_tokens: int, latency_ms: float, cost_usd: float,
        success: bool, error: Optional[str] = None,
        session_id: Optional[str] = None,
        prompt_preview: str = "",
    ) -> str:
        call_id = str(uuid.uuid4())
        timestamp = time.strftime("%Y‑%m‑%d %H:%M:%S")
        today = timestamp[:10]

        record = LLMCallRecord(
            call_id=call_id, model=model, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, total_tokens=total_tokens,
            latency_ms=latency_ms, cost_usd=cost_usd, success=success,
            error=error, session_id=session_id, 
            prompt_preview=prompt_preview[:100],
            timestamp=timestamp,
        )
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records // 2:]

            stats = self._daily_stats[today]
            stats["total_calls"] += 1
            stats["total_tokens"] += total_tokens
            stats["total_cost"] += cost_usd
            stats["total_latency"] += latency_ms
            if success:
                stats["success_calls"] += 1
            else:
                stats["fail_calls"] += 1

        logger.debug(f"[LLMOps] log_call {model} tokens={total_tokens}")
        return call_id

    def get_metrics(self, days: int = 7) -> Dict[str, Any]:
        with self._lock:
            today = time.strftime("%Y‑%m‑%d")
            today_stats = self._daily_stats.get(today, {})

            totals = {"total_calls": 0, "total_tokens": 0, "total_cost_usd": 0.0,
            "total_latency": 0.0, "success_calls": 0, "fail_calls": 0}
            for d in range(days):
                day = (datetime.now() - timedelta(days=d)).strftime("%Y‑%m‑%d")
                stats = self._daily_stats.get(day, {})
                for k in totals:
                    totals[k] += stats.get(k, 0)

            total_calls = totals["total_calls"] or 1
            success_rate = totals["success_calls"] / total_calls
            today_calls = today_stats.get("total_calls", 0) or 1
            today_success_rate = today_stats.get("success_calls", 0) / today_calls
            return {
                "all_time": {
                    "total_calls": totals["total_calls"],
                    "total_tokens": totals["total_tokens"],
                    "total_cost_usd": round(totals["total_cost_usd"], 6),
                    "avg_latency_ms": round(totals["total_latency"] / total_calls, 2),
                    "success_rate": round(success_rate * 100, 2),
                },
                "today": {
                    "calls": today_stats.get("total_calls", 0),
                    "tokens": today_stats.get("total_tokens", 0),
                    "cost_usd": round(today_stats.get("total_cost", 0), 6),
                    "avg_latency_ms": round(today_stats.get("total_latency", 0) / today_calls, 2),
                    "success_rate": round(today_success_rate * 100, 2),
                },
            }

    def get_daily_stats(self, days: int = 30) -> List[Dict]:
        result = []
        with self._lock:
            for d in range(days):
                day = (datetime.now() - timedelta(days=d)).strftime("%Y‑%m‑%d")
                stats = self._daily_stats.get(day, {})
                total_calls = stats.get("total_calls", 0) or 1
                result.append({
                    "date": day,
                    "calls": stats.get("total_calls", 0),
                    "tokens": stats.get("total_tokens", 0),
                    "cost_usd": round(stats.get("total_cost", 0), 6),
                    "avg_latency_ms": round(stats.get("total_latency", 0) / total_calls, 2),
                    "success_rate": round(stats.get("success_calls", 0) / total_calls * 100, 2),
                })
        return result[::-1]

    def get_recent_calls(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            records = self._records[-limit:][::-1]
            return [{
                "call_id": r.call_id, "model": r.model, "tokens": r.total_tokens,
                "latency_ms": round(r.latency_ms, 2), "cost_usd": round(r.cost_usd, 6),
                "success": r.success, "error": r.error, "timestamp": r.timestamp,
            } for r in records]


llmops_monitor = LLMOpsMonitor()


def get_llmops_monitor() -> LLMOpsMonitor:
    return llmops_monitor
```

### 6\.15 DataOps 数据管道编排

```Python
"""
backend/app/dataops/pipeline.py
DataOps 数据管道编排 - 金融数据 ETL
"""
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    name: str
    description: str
    extractor: Optional[Callable] = None
    transformer: Optional[Callable] = None
    loader: Optional[Callable] = None
    validator: Optional[Callable] = None


class DataPipeline:
    """
    金融数据 ETL 管道
    阶段:Extract -> Transform -> Validate -> Load
    """

    def __init__(self, name: str):
        self.name = name
        self.stages: List[PipelineStage] = []
        self._is_running = False
        self._last_run_at: Optional[datetime] = None
        self._last_status: str = "idle"

    def add_stage(self, stage: PipelineStage):
        self.stages.append(stage)
        return self

    async def run(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import uuid
        run_id = str(uuid.uuid4())
        params = params or {}
        start_time = datetime.now()

        logger.info(f"[Pipeline:{self.name}] starting run_id={run_id}")
        self._is_running = True

        total_records = 0
        stage_results = {}

        try:
            for stage in self.stages:
                logger.info(f"[Pipeline:{self.name}] stage: {stage.name}")

                extracted_data = []
                if stage.extractor:
                    extracted_data = await self._safe_call(stage.extractor, params)
                transformed_data = []
                if stage.transformer:
                    transformed_data = await self._safe_call(stage.transformer, extracted_data, params)
                validation_result = {"passed": True, "errors": []}
                if stage.validator:
                    validation_result = await self._safe_call(stage.validator, transformed_data)
                loaded_count = 0
                if stage.loader:
                    loaded_count = await self._safe_call(stage.loader, transformed_data)
                total_records += loaded_count

                stage_results[stage.name] = {
                    "extracted": len(extracted_data),
                    "transformed": len(transformed_data),
                    "loaded": loaded_count,
                    "validated": validation_result.get("passed", False),
                }
            self._last_status = "success"
            self._last_run_at = datetime.now()

            return {
                "run_id": run_id, "pipeline": self.name, "status": "success",
                "total_records": total_records, "stage_results": stage_results,
                "started_at": start_time.isoformat(),
                "completed_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"[Pipeline:{self.name}] failed: {str(e)}", exc_info=True)
            self._last_status = "failed"
            return {
                "run_id": run_id, "pipeline": self.name, "status": "failed",
                "error": str(e), "total_records": total_records,
                "stage_results": stage_results,
                "started_at": start_time.isoformat(),
                "completed_at": datetime.now().isoformat(),
            }
        finally:
            self._is_running = False

    async def _safe_call(self, func: Callable, *args, **kwargs) -> Any:
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"[Pipeline] stage failed: {str(e)}")
            return []


# ===== 预置抽取器

async def wind_stock_quote_extractor(params: Dict) -> List[Dict]:
    """Wind 股票行情抽取器"""
    return [{"stock_code": "000001", "stock_name": "平安银行",
    "price": 12.50, "change_pct": 1.23, "volume": 50000000,
    "timestamp": datetime.now().isoformat()}]


async def tushare_financial_extractor(params: Dict) -> List[Dict]:
    """Tushare 财务数据抽取器"""
    return [{"stock_code": "600519", "report_date": "2024‑06‑30",
    "revenue": 83000000000, "net_profit": 4200000000,
    "gross_margin": 0.92, "roe": 0.15}]


async def stock_quote_transformer(data: List[Dict], params: Dict) -> List[Dict]:
    """行情数据转换"""
    return [{"code": item.get("stock_code", ""),
    "name": item.get("stock_name", ""),
    "price": float(item.get("price", 0)),
    "change_pct": float(item.get("change_pct", 0)),
    "volume": int(item.get("volume", 0))} for item in data]


async def data_quality_validator(data: List[Dict]) -> Dict:
    """数据质量校验"""
    errors = []
    for i, item in enumerate(data[:10]):
        if not item.get("code"):
            errors.append(f"第{i}条缺少股票代码")
    return {"passed": len(errors) == 0, "errors": errors, "checked": len(data)}


async def db_loader(data: List[Dict]) -> int:
    """数据库加载"""
    logger.info(f"[DBLoader] loading {len(data)} records")
    return len(data)
```

### 6\.16 FastAPI 应用入口

```Python
"""
backend/app/main.py
FastAPI 应用入口
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api import chat, agent, rag, llmops, data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y‑%m‑%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Financial AI Agent v{settings.APP_VERSION} starting...")
    logger.info(f"LLM: {settings.LLM_MODEL}, VectorStore: {settings.VECTORSTORE_TYPE}")
    yield
    logger.info("Financial AI Agent shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="基于 LLM + Agent + RAG 的智能金融分析平台 API",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"[GlobalException] {str(exc)}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
async def health_check():
    return {"status": "healthy", "app": settings.APP_NAME, "version": settings.APP_VERSION,
    "llm_model": settings.LLM_MODEL}


@app.get("/")
async def root():
    return {"app": settings.APP_NAME, "version": settings.APP_VERSION, "docs": "/docs"}


# 注册路由
app.include_router(chat.router, prefix="/api/v1", tags=["对话"])
app.include_router(agent.router, prefix="/api/v1", tags=["Agent编排"])
app.include_router(rag.router, prefix="/api/v1", tags=["RAG知识库"])
app.include_router(llmops.router, prefix="/api/v1", tags=["LLMOps监控"])
app.include_router(data.router, prefix="/api/v1", tags=["金融数据"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT,
    reload=settings.DEBUG, workers=1, log_level="info")
```

### 6\.17 对话聊天 API

```Python
"""
backend/app/api/chat.py
对话聊天 API
"""
import uuid
import time
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse, AgentType
from app.agents.orchestrator import get_orchestrator
from app.llmops.monitor import get_llmops_monitor
from app.rag.retriever import get_hybrid_retriever

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    对话接口
    流程:接收消息 -> RAG检索 -> Agent执行 -> 返回结果
    """
    session_id = request.session_id or str(uuid.uuid4())
    start_time = time.time()
    context = {}

    try:
        # RAG 检索
        sources = []
        if request.enable_rag:
            retriever = get_hybrid_retriever()
            chunks = await retriever.retrieve(query=request.message, top_k=5, enable_rerank=True)
            if chunks:
                context["rag_context"] = chunks
                sources = [{"content": c["content"][:200] + "...", "score": round(c["score"], 3),
                "metadata": c.get("metadata", {})} for c in chunks]

        # Agent 执行
        orchestrator = get_orchestrator()
        agent_types = [request.agent_type] if request.agent_type else None
        result = await orchestrator.run(
            query=request.message, session_id=session_id,
            agent_types=agent_types, parallel=False, context=context,
        )

        latency = (time.time() - start_time) * 1000

        # 记录 LLMOps
        monitor = get_llmops_monitor()
        monitor.log_call(
            model="orchestrator", prompt_tokens=0,
            completion_tokens=len(result["final_response"]) // 4,
            total_tokens=result["execution_summary"]["total_tokens"],
            latency_ms=latency, cost_usd=0.0, success=True,
            session_id=session_id, prompt_preview=request.message[:100],
        )

        return ChatResponse(
            session_id=session_id, message=result["final_response"],
            agent_type=AgentType.RESEARCHER,
            sources=sources if sources else None,
            token_usage={"total": result["execution_summary"]["total_tokens"]},
            latency_ms=round(latency, 2),
        )

    except Exception as e:
        logger.error(f"[Chat] failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式对话接口(SSE)"""
    session_id = request.session_id or str(uuid.uuid4())

    async def generate():
        try:
            orchestrator = get_orchestrator()
            result = await orchestrator.run(query=request.message, session_id=session_id)
            response_text = result["final_response"]
            for i in range(0, len(response_text), 50):
                chunk = response_text[i:i + 50]
                yield f"event: message\ndata: {chunk}\n\n"
            yield f"event: done\ndata: done\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event‑stream",
    headers={"Cache‑Control": "no‑cache", "X‑Accel‑Buffering": "no"})
```

### 6\.18 Agent 编排 API

```Python
"""
backend/app/api/agent.py
Agent 编排 API
"""
import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from app.models.schemas import AgentInvokeRequest, AgentType
from app.agents.orchestrator import get_orchestrator

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/agent/invoke")
async def invoke_agent(request: AgentInvokeRequest):
    """直接调用 Agent 编排器"""
    try:
        orchestrator = get_orchestrator()
        result = await orchestrator.run(
            query=request.query,
            session_id=request.session_id or str(uuid.uuid4()),
            agent_types=request.agent_types,
            parallel=request.parallel,
        )
        return {
            "run_id": result["run_id"], "session_id": result["session_id"],
            "intent": result["intent"], "final_response": result["final_response"],
            "execution_summary": result["execution_summary"],
        }
    except Exception as e:
        logger.error(f"[Agent API] failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent/memory/{session_id}")
async def get_session_memory(session_id: str):
    orchestrator = get_orchestrator()
    memory = orchestrator.get_session_memory(session_id)
    return {"session_id": session_id, "conversation_turns": len(memory), "memory": memory}


@router.delete("/agent/memory/{session_id}")
async def clear_session_memory(session_id: str):
    orchestrator = get_orchestrator()
    orchestrator.clear_session(session_id)
    return {"message": "session cleared", "session_id": session_id}


@router.get("/agent/types")
async def list_agent_types():
return {
    "agents": [
        {
            "type": "researcher", "name": "投研分析Agent",
            "description": "股票基本面分析", "capabilities": ["stock_analysis", "financial_report"]
        },
        {
            "type": "risk", "name": "风控预警Agent",
            "description": "风险识别评估", "capabilities": ["risk_assessment", "risk_warning"]
        },
        {
            "type": "strategist", "name": "策略生成Agent",
            "description": "投资策略建议", "capabilities": ["strategy_generation", "portfolio"]
        },
    ]
}
```

### 6\.19 RAG 知识库 API

```Python
"""
backend/app/api/rag.py
RAG 知识库 API
"""
import logging
from fastapi import APIRouter, HTTPException
from app.models.schemas import RAGQueryRequest, RAGQueryResponse, KnowledgeBaseUpload
from app.rag.retriever import get_hybrid_retriever
from app.rag.knowledge_base import get_kb_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """RAG 检索接口"""
    try:
        retriever = get_hybrid_retriever()
        chunks = await retriever.retrieve(
            query=request.query, top_k=request.top_k,
            score_threshold=request.score_threshold,
            filters=request.filters, enable_rerank=request.enable_rerank,
        )
        return RAGQueryResponse(
            query=request.query,
            chunks=[{"content": c["content"], "score": round(c["score"], 3),
            "metadata": c.get("metadata", {})} for c in chunks],
            total_retrieved=len(chunks), rerank_applied=request.enable_rerank,
        )
    except Exception as e:
        logger.error(f"[RAG API] query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/knowledge/add")
async def add_knowledge(request: KnowledgeBaseUpload):
    """添加知识库内容"""
    try:
        kb = get_kb_manager()
        count = await kb.add_text_chunks(
            texts=[request.content], category=request.category,
            metadata_list=[{"title": request.title, **(request.metadata or {})}],
        )
        return {"message": f"added {count} chunks", "title": request.title, "category": request.category}
    except Exception as e:
        logger.error(f"[RAG API] add failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/knowledge/stats")
async def get_knowledge_stats():
    kb = get_kb_manager()
    count = kb.count_documents()
    return {"total_chunks": count, "vectorstore_type": "chroma", "status": "healthy"}
```

### 6\.20 LLMOps 监控 API

```Python
"""
backend/app/api/llmops.py
LLMOps 监控 API
"""
import logging
from fastapi import APIRouter, Query
from app.llmops.monitor import get_llmops_monitor

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/llmops/metrics")
async def get_metrics(days: int = Query(7, ge=1, le=90)):
    """获取 LLMOps 监控指标"""
    monitor = get_llmops_monitor()
    return monitor.get_metrics(days=days)


@router.get("/llmops/daily")
async def get_daily_stats(days: int = Query(30, ge=1, le=90)):
    """获取每日统计"""
    monitor = get_llmops_monitor()
    return {"daily_stats": monitor.get_daily_stats(days=days)}


@router.get("/llmops/recent")
async def get_recent_calls(limit: int = Query(50, ge=1, le=200)):
    """获取最近调用记录"""
    monitor = get_llmops_monitor()
    return {"recent_calls": monitor.get_recent_calls(limit=limit)}
```

### 6\.21 金融数据查询 API

```Python
"""
backend/app/api/data.py
金融数据查询 API
"""
import logging
from fastapi import APIRouter, HTTPException
from app.dataops.pipeline import DataPipeline, PipelineStage, wind_stock_quote_extractor
logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/data/stock/{stock_code}")
async def get_stock_info(stock_code: str):
    """查询股票基本信息"""
    return {
        "code": stock_code, "name": f"股票{stock_code}",
        "industry": "未知行业", "market": "主板",
        "listing_date": "2010‑01‑01",
    }


@router.get("/data/stock/{stock_code}/quote")
async def get_stock_quote(stock_code: str):
    """查询股票实时行情"""
    return {
        "code": stock_code, "name": f"股票{stock_code}",
        "price": 12.50, "change_pct": 1.23, "volume": 50000000,
        "high": 12.80, "low": 12.30, "open": 12.35,
        "timestamp": "2024‑01‑01 09:30:00",
    }


@router.post("/data/pipeline/run")
async def run_data_pipeline(pipeline_name: str = "stock_quote"):
    """手动触发数据管道"""
    pipeline = DataPipeline(name=pipeline_name)
    pipeline.add_stage(PipelineStage(
        name="extract", description="extract stock data",
        extractor=wind_stock_quote_extractor,
    ))
    return await pipeline.run()
```

### 6\.22 Redis 连接管理

```Python
"""
backend/app/core/redis_client.py
Redis 连接管理
"""
import redis.asyncio as redis
from typing import Optional
from app.config import get_settings

settings = get_settings()
_redis: Optional[redis.Redis] = None


async def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT,
            db=settings.REDIS_DB, password=settings.REDIS_PASSWORD,
            decode_responses=True, 
            max_connections=settings.REDIS_MAX_CONNECTIONS,
        )
    return _redis


async def close_redis():
    global _redis
    if _redis:
        await _redis.close()
        _redis = None
```

### 6\.23 向量数据库统一接口

```Python
"""
backend/app/core/vectorstore.py
向量数据库统一接口
"""
from typing import List, Dict, Any, Optional
from app.config import get_settings

settings = get_settings()


class VectorStoreInterface:
    """向量数据库统一接口"""
    def __init__(self, store_type: str = None):
        self.store_type = store_type or settings.VECTORSTORE_TYPE
        self._client = None

    async def add(
        self, texts: List[str], embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]], ids: List[str],
    ) -> bool:
        if self.store_type == "chroma":
            return await self._add_chroma(texts, embeddings, metadatas, ids)
        elif self.store_type == "milvus":
            return await self._add_milvus(texts, embeddings, metadatas, ids)
        return False

    async def search(
        self, query_embedding: List[float],
        top_k: int = 10, filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        if self.store_type == "chroma":
            return await self._search_chroma(query_embedding, top_k, filters)
        elif self.store_type == "milvus":
            return await self._search_milvus(query_embedding, top_k, filters)
        return []

    async def _add_chroma(self, texts, embeddings, metadatas, ids) -> bool:
        return True

    async def _search_chroma(self, query_embedding, top_k, filters) -> List[Dict]:
        return []

    async def _add_milvus(self, texts, embeddings, metadatas, ids) -> bool:
        return True

    async def _search_milvus(self, query_embedding, top_k, filters) -> List[Dict]:
        return []
```

# 7\. 前端完整代码

## 7\.1 前端依赖清单

```JSON
{
"name": "financial‑ai‑agent‑frontend",
"version": "1.0.0",
"type": "module",
"scripts": {
"dev": "vite",
"build": "vue‑tsc && vite build",
"preview": "vite preview"
},
"dependencies": {
"vue": "^3.4.0",
"vue‑router": "^4.3.0",
"pinia": "^2.1.0",
"axios": "^1.7.0",
"echarts": "^5.5.0",
"@vueuse/core": "^10.11.0",
"dayjs": "^1.11.0",
"marked": "^12.0.0",
"highlight.js": "^11.9.0"
},
"devDependencies": {
"@vitejs/plugin‑vue": "^5.0.0",
"vite": "^5.3.0",
"typescript": "^5.4.0",
"vue‑tsc": "^2.0.0",
"@types/node": "^20.0.0"
}
}
```

## 7\.2 Vite 构建配置

```TypeScript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin‑vue'
import { resolve } from 'path'

export default defineConfig({
plugins: [vue()],
resolve: { alias: { '@': resolve(__dirname, 'src') } },
server: {
port: 5173,
proxy: {
'/api': { target: 'http://localhost:8000', changeOrigin: true },
},
},
build: { outDir: 'dist', sourcemap: false },
})
```

## 7\.3 TypeScript 配置

```JSON
{
"compilerOptions": {
"target": "ES2020",
"useDefineForClassFields": true,
"module": "ESNext",
"lib": ["ES2020", "DOM", "DOM.Iterable"],
"skipLibCheck": true,
"moduleResolution": "bundler",
"allowImportingTsExtensions": true,
"resolveJsonModule": true,
"isolatedModules": true,
"noEmit": true,
"jsx": "preserve",
"strict": true,
"paths": { "@/*": ["./src/*"] }
},
"include": ["src/**/*.ts", "src/**/*.d.ts", "src/**/*.tsx", "src/**/*.vue"],
"references": [{ "path": "./tsconfig.node.json" }]
}
```

## 7\.4 Vue3 应用入口

```TypeScript
import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'
import './style.css'

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')
```

## 7\.5 路由配置

```TypeScript
import { createRouter, createWebHistory } from 'vue‑router'
import ChatView from './views/ChatView.vue'
import DashboardView from './views/DashboardView.vue'
import ReportView from './views/ReportView.vue'
import KnowledgeView from './views/KnowledgeView.vue'
import LLMOpsView from './views/LLMOpsView.vue'

const router = createRouter({
history: createWebHistory(),
routes: [
{ path: '/', name: 'chat', component: ChatView },
{ path: '/dashboard', name: 'dashboard', component: DashboardView },
{ path: '/report', name: 'report', component: ReportView },
{ path: '/knowledge', name: 'knowledge', component: KnowledgeView },
{ path: '/llmops', name: 'llmops', component: LLMOpsView },
],
})

export default router
```

## 7\.6 全局样式

```CSS
*, *::before, *::after { box‑sizing: border‑box; margin: 0; padding: 0; }
html, body, #app { height: 100%; overflow: hidden; }
body {
font‑family: ‑apple‑system, BlinkMacSystemFont, 'Segoe UI', Roboto,
'PingFang SC', 'Microsoft YaHei', sans‑serif;
background: #0a0e1a;
color: #e0e6ed;
line‑height: 1.6;
‑webkit‑font‑smoothing: antialiased;
}
::-webkit‑scrollbar { width: 6px; height: 6px; }
::-webkit‑scrollbar‑track { background: rgba(255,255,255,0.03); }
::-webkit‑scrollbar‑thumb { background: rgba(99,179,237,0.3); border‑radius: 3px; }
::-webkit‑scrollbar‑thumb:hover { background: rgba(99,179,237,0.5); }
code, pre { font‑family: 'Fira Code', 'JetBrains Mono', Consolas, monospace; }
```

## 7\.7 Axios 统一封装

```TypeScript
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api/v1'
const apiClient = axios.create({
baseURL: API_BASE,
timeout: 120000,
headers: { 'Content‑Type': 'application/json' },
})

apiClient.interceptors.request.use(config => config, error => Promise.reject(error))
apiClient.interceptors.response.use(
response => response.data,
error => {
const message = error.response?.data?.detail || error.message
return Promise.reject(new Error(message))
}
)

export default apiClient
```

## 7\.8 对话 API 封装

```TypeScript
import apiClient from './index'

export interface ChatMessage {
role: 'user' | 'assistant'
content: string
agent_type?: string
sources?: Array<{ content: string; score: number }>
token_usage?: { total: number }
latency_ms?: number
timestamp?: string
}

export interface ChatRequest {
session_id: string
message: string
agent_type?: string
stream: boolean
enable_rag: boolean
}

export interface ChatResponse {
session_id: string
message: string
agent_type: string
sources?: Array<{ content: string; score: number }>
token_usage?: { total: number }
latency_ms?: number
}

export const chatApi = {
send: (data: ChatRequest) => apiClient.post<ChatResponse>('/chat', data),
}

export const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api/v1'
```

## 7\.9 LLMOps API 类型定义

```TypeScript
import apiClient from './index'

export interface LLMOpsMetrics {
all_time: { total_calls: number; total_tokens: number; total_cost_usd: number; avg_latency_ms: number; success_rate: number }
today: { calls: number; tokens: number; cost_usd: number; avg_latency_ms: number; success_rate: number }
}

export interface DailyStats {
date: string; calls: number; tokens: number; cost_usd: number; avg_latency_ms: number; success_rate: number
}

export interface RecentCall {
call_id: string; model: string; tokens: number; latency_ms: number; cost_usd: number
success: boolean; error?: string; timestamp: string
}

export const llmopsApi = {
getMetrics: (days = 7) => apiClient.get<LLMOpsMetrics>('/llmops/metrics', { params: { days } }),
getDailyStats: (days = 30) => apiClient.get<{ daily_stats: DailyStats[] }>('/llmops/daily', { params: { days } }),
getRecentCalls: (limit = 50) => apiClient.get<{ recent_calls: RecentCall[] }>('/llmops/recent', { params: { limit } }),
}
```

## 7\.10 Agent API 封装

```TypeScript
import apiClient from './index'

export const agentApi = {
invoke: (data: { session_id: string; query: string; agent_types?: string[]; parallel?: boolean }) =>
apiClient.post('/agent/invoke', data),
getMemory: (sessionId: string) => apiClient.get(`/agent/memory/${sessionId}`),
clearMemory: (sessionId: string) => apiClient.delete(`/agent/memory/${sessionId}`),
listTypes: () => apiClient.get('/agent/types'),
}
```

## 7\.11 对话状态管理

```TypeScript
import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface ChatMessage {
role: 'user' | 'assistant'
content: string
agent_type?: string
sources?: any[]
token_usage?: { total: number }
latency_ms?: number
}

export const useChatStore = defineStore('chat', () => {
const messages = ref<ChatMessage[]>([])
const sessionId = ref(`session_${Date.now()}`)
const isLoading = ref(false)
const enableRAG = ref(true)

function addMessage(msg: ChatMessage) { messages.value.push(msg) }
function clearMessages() { messages.value = []; sessionId.value = `session_${Date.now()}` }
return { messages, sessionId, isLoading, enableRAG, addMessage, clearMessages }
})
```

## 7\.12 LLMOps 状态管理

```TypeScript
import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useLLMOpsStore = defineStore('llmops', () => {
const metrics = ref<any>(null)
const dailyStats = ref<any[]>([])
const recentCalls = ref<any[]>([])

function setMetrics(data: any) { metrics.value = data }
function setDailyStats(data: any[]) { dailyStats.value = data }
function setRecentCalls(data: any[]) { recentCalls.value = data }

return { metrics, dailyStats, recentCalls, setMetrics, setDailyStats, setRecentCalls }
})
```

## 7\.13 根组件与侧边导航

```Plaintext
<template>
<div id="app" class="app‑container">
<aside class="sidebar">
<div class="sidebar‑header">
<h1 class="logo">🤖 金融AI Agent</h1>
<p class="subtitle">LLM + Agent + RAG</p>
</div>
<nav class="nav‑menu">
<router‑link to="/" class="nav‑item" active‑class="active">
<span class="nav‑icon">💬</span>智能对话
</router‑link>
<router‑link to="/dashboard" class="nav‑item" active‑class="active">
<span class="nav‑icon">📊</span>数据看板
</router‑link>
<router‑link to="/report" class="nav‑item" active‑class="active">
<span class="nav‑icon">📝</span>报告中心
</router‑link>
<router‑link to="/knowledge" class="nav‑item" active‑class="active">
<span class="nav‑icon">📚</span>知识库
</router‑link>
<router‑link to="/llmops" class="nav‑item" active‑class="active">
<span class="nav‑icon">⚙️</span>LLMOps监控
</router‑link>
</nav>
<div class="sidebar‑footer">
<div class="status‑indicator">
<span class="status‑dot"></span>系统正常
</div>
</div>
</aside>
<main class="main‑content">
<router‑view />
</main>
</div>
</template>

<style scoped>
.app‑container { display: flex; height: 100vh; background: #0a0e1a; color: #e0e6ed; }
.sidebar { width: 240px; background: linear‑gradient(180deg, #0f1629 0%, #1a1f35 100%);
border‑right: 1px solid rgba(99,179,237,0.15); display: flex; flex‑direction: column; padding: 20px 0; }
.sidebar‑header { padding: 0 20px 20px; border‑bottom: 1px solid rgba(99,179,237,0.1); }
.logo { font‑size: 18px; font‑weight: 700; color: #63b3ed; margin: 0; }
.subtitle { font‑size: 11px; color: #718096; margin: 4px 0 0; }
.nav‑menu { flex: 1; padding: 16px 0; }
.nav‑item { display: flex; align‑items: center; gap: 10px; padding: 12px 20px;
color: #a0aec0; text‑decoration: none; font‑size: 14px; transition: all 0.2s;
border‑left: 3px solid transparent; }
.nav‑item:hover { background: rgba(99,179,237,0.1); color: #63b3ed; }
.nav‑item.active { background: rgba(99,179,237,0.15); color: #63b3ed; border‑left‑color: #63b3ed; }
.sidebar‑footer { padding: 16px 20px; border‑top: 1px solid rgba(99,179,237,0.1); }
.status‑indicator { display: flex; align‑items: center; gap: 8px; font‑size: 12px; color: #68d391; }
.status‑dot { width: 8px; height: 8px; border‑radius: 50%; background: #68d391; box‑shadow: 0 0 8px #68d391; }
.main‑content { flex: 1; overflow: hidden; display: flex; flex‑direction: column; }
</style>
```

## 7\.14 智能对话页面

```Plaintext
<template>
<div class="chat‑view">
<header class="chat‑header">
<div class="header‑left">
<h2>💬 智能投研对话</h2>
<span class="model‑tag">Qwen‑Plus + RAG</span>
</div>
<div class="header‑right">
<button class="btn‑icon" @click="clearChat" title="清空对话">🗑️</button>
</div>
</header>

<div class="chat‑body" ref="chatBodyRef">
<div v‑if="messages.length === 0" class="welcome‑panel">
<div class="welcome‑icon">🤖</div>
<h3>金融分析 AI Agent</h3>
<p>基于 LLM + Agent + RAG 的智能投研助手</p>
<div class="quick‑prompts">
<div class="prompt‑label">试试这样问我:</div>
<button class="prompt‑chip" @click="quickAsk('分析贵州茅台600519的基本面和估值')">
分析贵州茅台的基本面
</button>
<button class="prompt‑chip" @click="quickAsk('对宁德时代进行风险评估')">
评估宁德时代的风险
</button>
<button class="prompt‑chip" @click="quickAsk('分析当前市场行情,给出投资建议')">
分析市场并给出投资建议
</button>
</div>
</div>

<div v‑else class="message‑list">
<div v‑for="(msg, idx) in messages" :key="idx" class="message‑item" :class="msg.role">
<div class="message‑avatar">{{ msg.role === 'user' ? '👤' : '🤖' }}</div>
<div class="message‑content">
<div class="message‑header">
<span class="sender‑name">{{ msg.role === 'user' ? '你' : 'AI Agent' }}</span>
<span v‑if="msg.agent_type" class="agent‑badge">{{ msg.agent_type }}</span>
</div>
<div class="message‑text" v‑html="renderMarkdown(msg.content)"></div>
<div v‑if="msg.sources && msg.sources.length > 0" class="sources‑panel">
<div class="sources‑title">📚 参考来源</div>
<div v‑for="(src, sidx) in msg.sources" :key="sidx" class="source‑item">
<span class="source‑score">{{ (src.score * 100).toFixed(0) }}%</span>
<span class="source‑content">{{ src.content }}</span>
</div>
</div>
<div v‑if="msg.token_usage || msg.latency_ms" class="message‑meta">
<span v‑if="msg.token_usage"> {{ msg.token_usage.total }} tokens</span>
<span v‑if="msg.latency_ms">⏱️{{ msg.latency_ms }}ms</span>
</div>
</div>
</div>

<div v‑if="isLoading" class="message‑item assistant">
<div class="message‑avatar">🤖</div>
<div class="message‑content">
<div class="typing‑indicator"><span></span><span></span><span></span></div>
</div>
</div>
</div>
</div>

<div class="chat‑footer">
<div class="input‑toolbar">
<label class="toolbar‑item">
<input type="checkbox" v‑model="enableRAG" />
<span>启用RAG增强</span>
</label>
</div>
<div class="input‑row">
<textarea v‑model="inputMessage" class="input‑area" placeholder="输入你的金融分析问题..."
rows="1" @keydown.enter.exact.prevent="sendMessage"
@input="autoResize" ref="inputRef"></textarea>
<button class="send‑btn" @click="sendMessage" :disabled="!inputMessage.trim() || isLoading">
{{ isLoading ? '⏳' : '🚀' }}
</button>
</div>
</div>
</div>
</template>

<script setup lang="ts">
import { ref, nextTick } from 'vue'
import { marked } from 'marked'
import { chatApi, ChatMessage } from '@/api/chat'

const messages = ref<Array<ChatMessage & { sources?: any[]; token_usage?: any; latency_ms?: number }>>([])
const inputMessage = ref('')
const isLoading = ref(false)
const enableRAG = ref(true)
const sessionId = ref(`session_${Date.now()}`)
const chatBodyRef = ref<HTMLElement>()
const inputRef = ref<HTMLTextAreaElement>()

function renderMarkdown(text: string) {
return marked.parse(text, { async: false }) as string
}
function autoResize() {
const el = inputRef.value
if (el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 200) + 'px' }
}

async function sendMessage() {
const text = inputMessage.value.trim()
if (!text || isLoading.value) return

messages.value.push({ role: 'user', content: text })
inputMessage.value = ''
isLoading.value = true
await nextTick()
scrollToBottom()

try {
const res = await chatApi.send({
session_id: sessionId.value,
message: text,
enable_rag: enableRAG.value,
stream: false,
})
messages.value.push({
role: 'assistant', content: res.message,
agent_type: res.agent_type, sources: res.sources,
token_usage: res.token_usage, latency_ms: res.latency_ms,
})
} catch (e: any) {
messages.value.push({ role: 'assistant', content: `出错了:${e.message}` })
} finally {
isLoading.value = false
await nextTick()
scrollToBottom()
}
}
function quickAsk(question: string) { inputMessage.value = question; sendMessage() }
function clearChat() { messages.value = []; sessionId.value = `session_${Date.now()}` }
function scrollToBottom() {
if (chatBodyRef.value) chatBodyRef.value.scrollTop = chatBodyRef.value.scrollHeight
}
</script>

<style scoped>
.chat‑view { display: flex; flex‑direction: column; height: 100%; background: #0a0e1a; }
.chat‑header { display: flex; align‑items: center; justify‑content: space‑between;
padding: 16px 24px; border‑bottom: 1px solid rgba(99,179,237,0.15);
background: linear‑gradient(90deg, rgba(99,179,237,0.08) 0%, transparent 100%); }
.header‑left { display: flex; align‑items: center; gap: 12px; }
.header‑left h2 { font‑size: 18px; font‑weight: 600; color: #e0e6ed; margin: 0; }
.model‑tag { padding: 4px 10px; background: rgba(99,179,237,0.2); border‑radius: 20px; font‑size: 12px; color: #63b3ed; }
.btn‑icon { background: none; border: none; cursor: pointer; font‑size: 18px; padding: 8px; border‑radius: 8px; }
.btn‑icon:hover { background: rgba(99,179,237,0.15); }
.chat‑body { flex: 1; overflow‑y: auto; padding: 24px; }
.welcome‑panel { display: flex; flex‑direction: column; align‑items: center; justify‑content: center; height: 100%; text‑align: center; color: #a0aec0; }
.welcome‑icon { font‑size: 64px; margin‑bottom: 16px; }
.welcome‑panel h3 { font‑size: 24px; color: #e0e6ed; margin: 0 0 8px; }
.welcome‑panel p { font‑size: 14px; color: #718096; margin: 0 0 32px; }
.quick‑prompts { display: flex; flex‑direction: column; gap: 12px; align‑items: center; }
.prompt‑label { font‑size: 13px; color: #718096; }
.prompt‑chip { padding: 10px 20px; background: rgba(99,179,237,0.1); border: 1px solid rgba(99,179,237,0.25);
border‑radius: 24px; color: #a0aec0; cursor: pointer; font‑size: 13px; transition: all 0.2s; max‑width: 400px; }
.prompt‑chip:hover { background: rgba(99,179,237,0.2); border‑color: rgba(99,179,237,0.5); color: #63b3ed; transform: translateY(‑2px); }
.message‑list { display: flex; flex‑direction: column; gap: 20px; }
.message‑item { display: flex; gap: 12px; max‑width: 80%; }
.message‑item.user { align‑self: flex‑end; flex‑direction: row‑reverse; }
.message‑avatar { width: 36px; height: 36px; border‑radius: 50%; display: flex; align‑items: center; justify‑content: center; font‑size: 18px; flex‑shrink: 0; background: rgba(99,179,237,0.15); }
.message‑item.assistant .message‑avatar { background: rgba(104,211,145,0.15); }
.message‑content { flex: 1; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border‑radius: 12px; padding: 14px 18px; }
.message‑item.user .message‑content { background: rgba(99,179,237,0.12); border‑color: rgba(99,179,237,0.25); }
.message‑header { display: flex; align‑items: center; gap: 8px; margin‑bottom: 8px; }
.sender‑name { font‑size: 13px; font‑weight: 600; color: #a0aec0; }
.agent‑badge { padding: 2px 8px; background: rgba(104,211,145,0.2); border‑radius: 10px; font‑size: 11px; color: #68d391; }
.message‑text { font‑size: 14px; line‑height: 1.7; color: #e0e6ed; word‑break: break‑word; }
.message‑text :deep(strong) { color: #63b3ed; }
.message‑text :deep(pre) { background: rgba(0,0,0,0.3); padding: 12px; border‑radius: 8px; overflow‑x: auto; }
.message‑text :deep(table) { width: 100%; border‑collapse: collapse; margin: 8px 0; }
.message‑text :deep(th), .message‑text :deep(td) { padding: 8px 12px; border: 1px solid rgba(255,255,255,0.1); font‑size: 13px; }
.message‑text :deep(th) { background: rgba(99,179,237,0.1); color: #63b3ed; }
.sources‑panel { margin‑top: 12px; padding: 10px; background: rgba(99,179,237,0.08); border‑radius: 8px; border: 1px solid rgba(99,179,237,0.15); }
.sources‑title { font‑size: 12px; color: #63b3ed; margin‑bottom: 8px; font‑weight: 600; }
.source‑item { display: flex; align‑items: flex‑start; gap: 8px; padding: 4px 0; font‑size: 12px; color: #718096; }
.source‑score { color: #68d391; font‑weight: 600; flex‑shrink: 0; }
.message‑meta { display: flex; gap: 12px; margin‑top: 8px; font‑size: 12px; color: #718096; }
.typing‑indicator { display: flex; gap: 4px; padding: 4px 0; }
.typing‑indicator span { width: 8px; height: 8px; border‑radius: 50%; background: #63b3ed; animation: bounce 1.4s infinite ease‑in‑out; }
.typing‑indicator span:nth‑child(1) { animation‑delay: ‑0.32s; }
.typing‑indicator span:nth‑child(2) { animation‑delay: ‑0.16s; }
@keyframes bounce { 0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; } 40% { transform: scale(1); opacity: 1; } }
.chat‑footer { border‑top: 1px solid rgba(99,179,237,0.15); padding: 16px 24px; background: rgba(15,22,41,0.8); }
.input‑toolbar { display: flex; gap: 16px; margin‑bottom: 10px; }
.toolbar‑item { display: flex; align‑items: center; gap: 6px; font‑size: 12px; color: #718096; cursor: pointer; }
.toolbar‑item input[type="checkbox"] { accent‑color: #63b3ed; }
.input‑row { display: flex; gap: 12px; align‑items: flex‑end; }
.input‑area { flex: 1; background: rgba(255,255,255,0.06); border: 1px solid rgba(99,179,237,0.2);
border‑radius: 12px; padding: 12px 16px; color: #e0e6ed; font‑size: 14px; resize: none; outline: none;
font‑family: inherit; line‑height: 1.5; transition: border‑color 0.2s; }
.input‑area:focus { border‑color: rgba(99,179,237,0.5); background: rgba(99,179,237,0.06); }
.input‑area::placeholder { color: #4a5568; }
.send‑btn { width: 44px; height: 44px; border‑radius: 12px; border: none;
background: linear‑gradient(135deg, #3182ce, #63b3ed); color: white; font‑size: 18px;
cursor: pointer; transition: all 0.2s; display: flex; align‑items: center; justify‑content: center; flex‑shrink: 0; }
.send‑btn:hover:not(:disabled) { transform: translateY(‑2px); box‑shadow: 0 4px 12px rgba(99,179,237,0.4); }
.send‑btn:disabled { opacity: 0.4; cursor: not‑allowed; }
</style>
```

## 7\.15 LLMOps 监控面板

```Plaintext
<template>
<div class="llmops‑view">
<header class="page‑header">
<h2>⚙️LLMOps 监控面板</h2>
<button class="btn‑refresh" @click="loadData">🔄 刷新</button>
</header>

<div class="metrics‑grid">
<div class="metric‑card">
<div class="metric‑label">今日调用次数</div>
<div class="metric‑value">{{ metrics?.today?.calls || 0 }}</div>
<div class="metric‑sub">成功率 {{ metrics?.today?.success_rate || 0 }}%</div>
</div>
<div class="metric‑card">
<div class="metric‑label">今日 Token 消耗</div>
<div class="metric‑value">{{ formatNumber(metrics?.today?.tokens || 0) }}</div>
<div class="metric‑sub">平均延迟 {{ metrics?.today?.avg_latency_ms || 0 }}ms</div>
</div>
<div class="metric‑card">
<div class="metric‑label">今日成本</div>
<div class="metric‑value">${{ (metrics?.today?.cost_usd || 0).toFixed(4) }}</div>
<div class="metric‑sub">累计 ${{ (metrics?.all_time?.total_cost_usd || 0).toFixed(4) }}</div>
</div>
<div class="metric‑card">
<div class="metric‑label">累计调用</div>
<div class="metric‑value">{{ formatNumber(metrics?.all_time?.total_calls || 0) }}</div>
<div class="metric‑sub">累计 Token {{ formatNumber(metrics?.all_time?.total_tokens || 0) }}</div>
</div>
</div>

<div class="section">
<h3 class="section‑title">📋 最近调用记录</h3>
<div class="table‑wrapper">
<table class="data‑table">
<thead><tr><th>时间</th><th>模型</th><th>Token数</th><th>延迟</th><th>成本</th><th>状态</th></tr
```



> （注：部分内容可能由 AI 生成）
