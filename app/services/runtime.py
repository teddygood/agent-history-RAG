from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.retrieval.graph_retriever import GraphRetriever
from app.services.embedder import Embedder, create_embedder
from app.services.extractor import HeuristicExtractor
from app.services.history_loader import AgentHistoryLoader
from app.services.ingestion import IngestionService
from app.services.jsonl_loader import JSONLLoader
from app.services.neo4j_store import Neo4jStore


@dataclass
class AppRuntime:
    store: Neo4jStore
    extractor: HeuristicExtractor
    embedder: Embedder
    ingestion: IngestionService
    retriever: GraphRetriever
    jsonl_loader: JSONLLoader
    history_loader: AgentHistoryLoader

    @classmethod
    def create(cls) -> "AppRuntime":
        store = Neo4jStore()
        store.init_schema()
        extractor = HeuristicExtractor()
        embedder = create_embedder()
        ingestion = IngestionService(store=store, extractor=extractor, embedder=embedder)
        retriever = GraphRetriever(store=store, extractor=extractor, embedder=embedder)
        return cls(
            store=store,
            extractor=extractor,
            embedder=embedder,
            ingestion=ingestion,
            retriever=retriever,
            jsonl_loader=JSONLLoader(),
            history_loader=AgentHistoryLoader(),
        )

    def close(self) -> None:
        self.store.close()

    def runtime_profile(self) -> dict[str, object]:
        return {
            "embedding": {
                "provider": getattr(self.embedder, "provider", "unknown"),
                "model_name": getattr(self.embedder, "model_name", "unknown"),
                "dim": int(getattr(self.embedder, "dim", settings.embedding_dim)),
                "model_candidates": list(settings.embedding_model_candidates),
            },
            "chunking": self.ingestion.get_chunking_settings(),
        }
