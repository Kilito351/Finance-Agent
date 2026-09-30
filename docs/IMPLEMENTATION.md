# 实现说明

本仓库依据原始《Agent 项目-基于 LLM 的金融分析 AI Agent》方案做了最小可运行复现。

## 已实现

- FastAPI API 与交互式 OpenAPI 文档
- 投研、风控、策略、报告四阶段协作流程
- 可选 OpenAI 兼容 LLM 接口与无密钥 Demo 降级
- 轻量中文关键词 RAG、来源回传和知识添加 API
- 调用量、成功率、Token 与延迟指标
- Vue 3 响应式深色界面与四个业务页面
- pytest、Docker Compose、Conda 和本地启动说明

## 有意简化

原方案的 PostgreSQL、Redis、ChromaDB、Prefect、Prometheus 和 Grafana 属于生产扩展。本复现保留清晰模块边界，但默认以内存实现替代，避免评审者为了运行 Demo 安装六个外部服务。静态市场卡片有醒目的“非实时”标识，避免将样例误认为真实行情。

## 生产化建议

1. 将知识文档与指标存储切换为持久化数据库，并为向量检索增加 embedding 与 reranker。
2. 接入有授权的数据源，记录数据时间戳、版本和血缘。
3. 增加 JWT/RBAC、速率限制、审计、敏感信息脱敏与 Prompt 注入防护。
4. 使用任务队列处理长报告，使用 SSE/WebSocket 推送各 Agent 状态。
5. 建立离线评测集，持续评估事实性、引用准确性、风险覆盖率和成本。

