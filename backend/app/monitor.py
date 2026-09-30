from dataclasses import dataclass, field
from threading import Lock


@dataclass
class MetricsStore:
    calls: int = 0
    successes: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: list[float] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, success: bool, prompt_tokens: int = 0, completion_tokens: int = 0, latency_ms: float = 0) -> None:
        with self._lock:
            self.calls += 1
            self.successes += int(success)
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.latency_ms.append(latency_ms)

    def snapshot(self) -> dict:
        with self._lock:
            average = sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0
            return {
                "calls": self.calls,
                "success_rate": round(self.successes / self.calls, 4) if self.calls else 1.0,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "average_latency_ms": round(average, 2),
            }

