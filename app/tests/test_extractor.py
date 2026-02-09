from app.services.extractor import HeuristicExtractor


def test_extract_implicit_technical_concepts() -> None:
    extractor = HeuristicExtractor()
    text = "continuous batching과 paged attention을 비교해줘"
    entities = extractor.extract_entities(text)
    names = {item.canonical_name for item in entities}

    assert "continuous batching" in names
    assert "paged attention" in names


def test_relation_type_detection() -> None:
    extractor = HeuristicExtractor()
    text = "vector rag vs graph rag 차이를 비교"
    entities = extractor.extract_entities(text)
    relations = extractor.extract_relations(text, entities)
    assert relations
    assert relations[0].relation_type == "COMPARES"
