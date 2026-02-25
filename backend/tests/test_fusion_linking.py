from app.fusion.linking import generate_explains_links


def test_fusion_linking_suppresses_false_positive_on_type_mismatch() -> None:
    logic_steps = [
        {
            "logic_step_id": "p1:Method",
            "paper_id": "p1",
            "step_type": "Method",
            "summary": "We propose a finite element method for beam vibration.",
            "evidence_chunk_ids": ["pc:1"],
        }
    ]
    entities = [
        {
            "entity_id": "ke:method",
            "name": "Finite Element Method",
            "entity_type": "method",
            "description": "A numerical method for mechanics simulations.",
            "source_chapter_id": "tb:1:ch1",
        },
        {
            "entity_id": "ke:history",
            "name": "Industrial Revolution",
            "entity_type": "history",
            "description": "A historical period in world history.",
            "source_chapter_id": "tb:1:ch1",
        },
    ]

    links = generate_explains_links(logic_steps, entities, min_score=0.45)
    targets = {link["entity_id"] for link in links}

    assert "ke:method" in targets
    assert "ke:history" not in targets


def test_fusion_linking_is_deterministic_on_fixed_fixture() -> None:
    logic_steps = [
        {
            "logic_step_id": "p1:Result",
            "paper_id": "p1",
            "step_type": "Result",
            "summary": "Natural frequency increases with stiffness in experiments.",
            "evidence_chunk_ids": ["pc:2"],
        }
    ]
    entities = [
        {
            "entity_id": "ke:theory",
            "name": "Natural Frequency",
            "entity_type": "theory",
            "description": "Vibration characteristic frequency of structures.",
            "source_chapter_id": "tb:2:ch3",
        },
        {
            "entity_id": "ke:method",
            "name": "Finite Element Method",
            "entity_type": "method",
            "description": "Numerical method for solving PDEs.",
            "source_chapter_id": "tb:2:ch3",
        },
    ]

    first = generate_explains_links(logic_steps, entities, min_score=0.35)
    second = generate_explains_links(logic_steps, entities, min_score=0.35)
    assert first == second
