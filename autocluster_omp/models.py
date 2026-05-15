from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class WorkloadConfig:
    algorithm: str
    n_points: int
    n_clusters: int
    n_features: int
    iterations: int
    fuzziness: float = 2.0

    @property
    def estimated_work(self) -> int:
        work = self.n_points * self.n_clusters * self.n_features * self.iterations
        if self.algorithm in {"fuzzy-cmeans", "fcm"}:
            work *= self.n_clusters
        return work


@dataclass
class LoopInfo:
    start_line: int
    end_line: int
    header: str
    body: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoopAnalysis:
    loop: LoopInfo
    classification: str
    decision: str
    reason: str
    suggested_pragma: Optional[str] = None
    safety: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop": self.loop.to_dict(),
            "classification": self.classification,
            "decision": self.decision,
            "reason": self.reason,
            "suggested_pragma": self.suggested_pragma,
            "safety": self.safety,
        }


@dataclass
class BenchmarkResult:
    algorithm: str
    version: str
    threads: int
    runtime_seconds: float
    speedup: float
    efficiency: float
    checksum: Optional[float]
    correctness: str
    schedule: str = "static"
    workload_name: str = "default"
    dataset: Optional[str] = None
    labels: Optional[str] = None
    clusters: Optional[int] = None
    features: Optional[int] = None
    iterations: Optional[int] = None
    repeat: int = 1
    fuzziness: Optional[float] = None
    objective: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
