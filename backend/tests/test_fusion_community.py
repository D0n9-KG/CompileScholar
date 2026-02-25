from app.fusion.community import detect_fusion_communities
from app.fusion.keywords import extract_fusion_keywords


def test_fusion_community_count_is_stable_within_tolerance() -> None:
    nodes = [
        {"id": "ls:m1", "text": "finite element method for vibration", "evidence_quote": "finite element method"},
        {"id": "ke:m1", "text": "finite element method", "evidence_quote": "numerical method"},
        {"id": "ke:m2", "text": "beam model equation", "evidence_quote": "beam equation"},
        {"id": "ls:r1", "text": "natural frequency increases with stiffness", "evidence_quote": "frequency increases"},
        {"id": "ke:r1", "text": "natural frequency theory", "evidence_quote": "natural frequency"},
        {"id": "ke:r2", "text": "stiffness principle", "evidence_quote": "stiffness principle"},
    ]
    edges = [
        {"source": "ls:m1", "target": "ke:m1", "type": "EXPLAINS", "weight": 0.92},
        {"source": "ke:m1", "target": "ke:m2", "type": "RELATES_TO", "weight": 0.83},
        {"source": "ls:r1", "target": "ke:r1", "type": "EXPLAINS", "weight": 0.91},
        {"source": "ke:r1", "target": "ke:r2", "type": "RELATES_TO", "weight": 0.86},
    ]

    communities = detect_fusion_communities(nodes, edges)
    assert 1 <= len(communities) <= 3
    assert len(communities) == 2


def test_fusion_community_contains_representative_evidence_and_keywords() -> None:
    nodes = [
        {"id": "ls:1", "text": "finite element method for dynamics", "evidence_quote": "finite element method"},
        {"id": "ke:1", "text": "finite element method", "evidence_quote": "numerical discretization"},
        {"id": "ke:2", "text": "dynamic equation", "evidence_quote": "dynamic equation"},
    ]
    edges = [
        {"source": "ls:1", "target": "ke:1", "type": "EXPLAINS", "weight": 0.90},
        {"source": "ke:1", "target": "ke:2", "type": "RELATES_TO", "weight": 0.80},
    ]

    communities = detect_fusion_communities(nodes, edges)
    assert communities
    for c in communities:
        assert c.get("representative_evidence")

    keywords = extract_fusion_keywords(communities, nodes, top_k=3)
    assert keywords
    assert all(k.get("keyword") for k in keywords)
