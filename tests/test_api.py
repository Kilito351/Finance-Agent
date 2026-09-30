import os
import sys
from pathlib import Path

os.environ.pop("LLM_API_KEY", None)
sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "demo"}


def test_analysis_has_all_agent_outputs() -> None:
    response = client.post(
        "/api/v1/agents/analyze",
        json={"query": "分析贵州茅台的基本面与风险", "stock_code": "600519"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "demo"
    assert {data["research"]["agent"], data["risk"]["agent"], data["strategy"]["agent"]} == {
        "researcher", "risk", "strategist"
    }
    assert data["sources"]


def test_knowledge_can_be_added() -> None:
    before = client.get("/api/v1/knowledge").json()["total"]
    response = client.post(
        "/api/v1/knowledge",
        json={"title": "测试资料", "content": "用于测试检索。", "category": "测试"},
    )
    assert response.status_code == 201
    assert response.json()["total"] == before + 1

