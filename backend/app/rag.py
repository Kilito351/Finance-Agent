import re
from dataclasses import asdict, dataclass


@dataclass
class Document:
    title: str
    category: str
    content: str


class KnowledgeBase:
    """Dependency-free retrieval for the reproducible demo."""

    def __init__(self) -> None:
        self.documents = [
            Document("价值评估基础", "投研方法", "估值分析应结合市盈率、市净率、现金流、盈利质量与行业周期，避免依赖单一指标。"),
            Document("风险识别框架", "风险管理", "风险评估应覆盖市场、信用、流动性、经营、政策与模型风险，并说明数据时效性。"),
            Document("投资组合原则", "策略研究", "策略建议需考虑分散化、仓位上限、止损纪律和投资期限，历史表现不代表未来收益。"),
            Document("财报分析要点", "财务分析", "关注收入和利润增速、毛利率、经营现金流、负债结构、应收账款及非经常性损益。"),
        ]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        latin = re.findall(r"[a-zA-Z0-9]+", text.lower())
        chinese = [text[i : i + 2] for i in range(len(text) - 1) if "\u4e00" <= text[i] <= "\u9fff"]
        return set(latin + chinese)

    def search(self, query: str, limit: int = 3) -> list[dict]:
        query_tokens = self._tokens(query)
        ranked = []
        for document in self.documents:
            doc_tokens = self._tokens(document.title + document.content)
            overlap = len(query_tokens & doc_tokens)
            score = overlap / max(len(query_tokens), 1)
            ranked.append((score, document))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:limit]
        return [{**asdict(doc), "score": round(score, 4)} for score, doc in selected]

    def add(self, document: Document) -> None:
        self.documents.append(document)

