from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.graph_retriever import GraphRetriever
from app.services.embedder import HashEmbedder
from app.services.extractor import HeuristicExtractor
from app.services.history_loader import AgentHistoryLoader
from app.services.ingestion import IngestionService
from app.services.jsonl_loader import JSONLLoader
from app.services.neo4j_store import Neo4jStore


@dataclass
class AppRuntime:
    store: Neo4jStore
    extractor: HeuristicExtractor
    embedder: HashEmbedder
    ingestion: IngestionService
    retriever: GraphRetriever
    jsonl_loader: JSONLLoader
    history_loader: AgentHistoryLoader

    @classmethod
    def create(cls) -> "AppRuntime":
        store = Neo4jStore()
        store.init_schema()
        extractor = HeuristicExtractor()
        embedder = HashEmbedder()
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
