"""
Quality Evaluation Script for 20-Paper E2E Test

Runs comprehensive quality queries across all 4 stages of P0+P1 optimization.
"""
from __future__ import annotations

from app.graph.neo4j_client import Neo4jClient
from app.settings import Settings


def run_quality_evaluation():
    """Run all quality evaluation queries and generate report."""
    settings = Settings()

    print("="*80)
    print("LogicKG P0+P1 Quality Evaluation - 20 Paper Test")
    print("="*80)

    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        with client._driver.session() as session:

            # ==================================================================
            # STAGE 1: P0 Meta Filter
            # ==================================================================
            print("\n" + "="*80)
            print("STAGE 1: P0 Meta Filter - Reduced Meta-Information Noise")
            print("="*80)

            # Meta noise detection (heuristic)
            query = """
            WITH ['author','affiliation','correspondence','submitted','accepted','published',
                  'funding','grant','acknowledg','doi','journal','conflict of interest',
                  'dataset availability','code repository','email','@'] AS meta_terms
            MATCH (:Paper)-[:HAS_CLAIM]->(cl:Claim)
            WITH count(cl) AS total_claims,
                 count(CASE WHEN any(k IN meta_terms WHERE toLower(coalesce(cl.text,'')) CONTAINS k) THEN 1 END) AS suspected_meta_claims
            RETURN total_claims,
                   suspected_meta_claims,
                   round(100.0 * toFloat(suspected_meta_claims) / toFloat(CASE WHEN total_claims=0 THEN 1 ELSE total_claims END), 2) AS suspected_meta_pct
            """
            result = session.run(query).single()
            print(f"\nMeta-Information Noise:")
            print(f"  Total claims: {result['total_claims']}")
            print(f"  Suspected meta claims: {result['suspected_meta_claims']}")
            print(f"  Meta noise rate: {result['suspected_meta_pct']}%")
            print(f"  [OK] Target: <10% meta noise")

            # Quality tier distribution
            query = """
            MATCH (p:Paper) WHERE coalesce(p.ingested,false)=true
            RETURN coalesce(p.phase1_quality_tier,'unknown') AS tier, count(*) AS papers
            ORDER BY papers DESC
            """
            results = list(session.run(query))
            print(f"\nQuality Tier Distribution:")
            for row in results:
                print(f"  {row['tier']}: {row['papers']} papers")

            # ==================================================================
            # STAGE 2: Assertion Layer
            # ==================================================================
            print("\n" + "="*80)
            print("STAGE 2: Assertion Layer - Proposition Deduplication")
            print("="*80)

            # Deduplication effectiveness
            query = """
            MATCH (:Paper)-[:HAS_CLAIM]->(cl:Claim)
            WITH count(cl) AS claim_count
            MATCH (pr:Proposition)
            WITH claim_count, count(pr) AS proposition_count
            RETURN claim_count, proposition_count,
                   round(toFloat(proposition_count)/toFloat(CASE WHEN claim_count=0 THEN 1 ELSE claim_count END), 4) AS propositions_per_claim,
                   round((1 - toFloat(proposition_count)/toFloat(CASE WHEN claim_count=0 THEN 1 ELSE claim_count END))*100.0, 2) AS dedup_reduction_pct
            """
            result = session.run(query).single()
            print(f"\nDeduplication Effectiveness:")
            print(f"  Total claims: {result['claim_count']}")
            print(f"  Unique propositions: {result['proposition_count']}")
            print(f"  Propositions/Claims ratio: {result['propositions_per_claim']}")
            print(f"  Dedup reduction: {result['dedup_reduction_pct']}%")
            print(f"  [OK] Target: 10-30% reduction from deduplication")

            # Cross-paper merging
            query = """
            MATCH (pr:Proposition)<-[:MAPS_TO]-(cl:Claim)<-[:HAS_CLAIM]-(p:Paper)
            WITH pr, count(DISTINCT p.paper_id) AS paper_span
            RETURN count(pr) AS total_props,
                   count(CASE WHEN paper_span > 1 THEN 1 END) AS cross_paper_props,
                   round(100.0 * toFloat(count(CASE WHEN paper_span > 1 THEN 1 END)) / toFloat(CASE WHEN count(pr)=0 THEN 1 ELSE count(pr) END), 2) AS cross_paper_merge_rate_pct,
                   round(avg(paper_span), 2) AS avg_papers_per_prop
            """
            result = session.run(query).single()
            print(f"\nCross-Paper Proposition Merging:")
            print(f"  Total propositions: {result['total_props']}")
            print(f"  Cross-paper propositions: {result['cross_paper_props']}")
            print(f"  Cross-paper merge rate: {result['cross_paper_merge_rate_pct']}%")
            print(f"  Avg papers per proposition: {result['avg_papers_per_prop']}")
            print(f"  [OK] Target: 5-15% cross-paper merging")

            # Context diversity (step_types_seen, kinds_seen)
            query = """
            MATCH (pr:Proposition)
            RETURN round(avg(size(coalesce(pr.step_types_seen,[]))),2) AS avg_step_types_seen,
                   round(avg(size(coalesce(pr.kinds_seen,[]))),2) AS avg_kinds_seen,
                   count(CASE WHEN size(coalesce(pr.step_types_seen,[])) >= 2 THEN 1 END) AS multi_step_props,
                   count(CASE WHEN size(coalesce(pr.kinds_seen,[])) >= 2 THEN 1 END) AS multi_kind_props
            """
            result = session.run(query).single()
            print(f"\nContext Diversity:")
            print(f"  Avg step_types_seen: {result['avg_step_types_seen']}")
            print(f"  Avg kinds_seen: {result['avg_kinds_seen']}")
            print(f"  Multi-step propositions: {result['multi_step_props']}")
            print(f"  Multi-kind propositions: {result['multi_kind_props']}")
            print(f"  [OK] Target: >1.5 avg context diversity")

            # ==================================================================
            # STAGE 3: Group Layer
            # ==================================================================
            print("\n" + "="*80)
            print("STAGE 3: Group Layer - Semantic Clustering")
            print("="*80)

            # Group structure
            query = """
            MATCH (pg:PropositionGroup)
            OPTIONAL MATCH (pr:Proposition)-[:IN_GROUP]->(pg)
            WITH pg, count(pr) AS members
            RETURN count(pg) AS group_count,
                   round(avg(members),2) AS avg_group_size,
                   sum(CASE WHEN members = 1 THEN 1 ELSE 0 END) AS singleton_groups,
                   sum(CASE WHEN members > 1 THEN 1 ELSE 0 END) AS multi_member_groups
            """
            result = session.run(query).single()
            if result and result['group_count']:
                print(f"\nGroup Structure:")
                print(f"  Total groups: {result['group_count']}")
                print(f"  Avg group size: {result['avg_group_size']}")
                print(f"  Singleton groups: {result['singleton_groups']}")
                print(f"  Multi-member groups: {result['multi_member_groups']}")
                print(f"  [OK] Target: 2-5 avg group size, <50% singletons")
            else:
                print(f"\n[WARN]  No groups found - clustering may not have run")

            # Grouping coverage
            query = """
            MATCH (pr:Proposition)
            OPTIONAL MATCH (pr)-[:IN_GROUP]->(pg:PropositionGroup)
            WITH count(DISTINCT pr.prop_id) AS total_props,
                 count(DISTINCT CASE WHEN pg IS NOT NULL THEN pr.prop_id END) AS grouped_props
            RETURN total_props, grouped_props,
                   round(100.0 * toFloat(grouped_props)/toFloat(CASE WHEN total_props=0 THEN 1 ELSE total_props END),2) AS grouping_coverage_pct
            """
            result = session.run(query).single()
            print(f"\nGrouping Coverage:")
            print(f"  Total propositions: {result['total_props']}")
            print(f"  Grouped propositions: {result['grouped_props']}")
            print(f"  Coverage: {result['grouping_coverage_pct']}%")
            print(f"  [OK] Target: >80% grouping coverage")

            # Sample high/low quality groups
            query = """
            MATCH (pr:Proposition)-[r:IN_GROUP]->(pg:PropositionGroup)
            WITH pg, count(pr) AS members, avg(coalesce(r.similarity_score,0.0)) AS mean_similarity
            RETURN pg.group_id, pg.label_text, members, round(mean_similarity,4) AS mean_similarity
            ORDER BY mean_similarity ASC, members DESC
            LIMIT 5
            """
            results = list(session.run(query))
            if results:
                print(f"\nLow Similarity Groups (potential issues):")
                for row in results:
                    print(f"  Group {row['pg.group_id']}: {row['members']} members, sim={row['mean_similarity']}")
                    print(f"    \"{row['pg.label_text'][:60]}...\"")

            # ==================================================================
            # STAGE 4: Conflict Detection
            # ==================================================================
            print("\n" + "="*80)
            print("STAGE 4: Conflict Detection - Semantic Conflict Coverage")
            print("="*80)

            # Aggregate conflict metrics from phase1_quality_json
            # Note: Requires APOC if using apoc.convert.fromJsonMap
            query = """
            MATCH (p:Paper)
            WHERE p.phase1_quality_json IS NOT NULL
            WITH count(p) AS papers,
                 sum(toInteger(coalesce(p.conflict_candidate_pairs,0))) AS candidate_pairs,
                 sum(toInteger(coalesce(p.conflict_pairs,0))) AS contradict_pairs
            RETURN papers, candidate_pairs, contradict_pairs,
                   round(100.0 * toFloat(contradict_pairs)/toFloat(CASE WHEN candidate_pairs=0 THEN 1 ELSE candidate_pairs END),2) AS conflict_rate_pct
            """
            try:
                result = session.run(query).single()
                if result:
                    print(f"\nConflict Detection Metrics:")
                    print(f"  Papers processed: {result['papers']}")
                    print(f"  Candidate pairs: {result['candidate_pairs']}")
                    print(f"  Contradictory pairs: {result['contradict_pairs']}")
                    print(f"  Conflict rate: {result['conflict_rate_pct']}%")
                    print(f"  [OK] Target: 5-20% conflict rate among candidates")
            except Exception as e:
                print(f"\n[WARN]  Could not retrieve conflict metrics: {e}")
                print("   (May need APOC or different field structure)")

            # ==================================================================
            # SUMMARY
            # ==================================================================
            print("\n" + "="*80)
            print("EVALUATION SUMMARY")
            print("="*80)

            # Count papers
            result = session.run("MATCH (p:Paper) WHERE p.ingested=true RETURN count(p) AS count").single()
            papers_ingested = result['count']

            print(f"\nPapers Ingested: {papers_ingested}/20")
            print(f"\n[OK] Quality evaluation complete!")
            print(f"\nNext steps:")
            print(f"  1. Review metrics above against targets")
            print(f"  2. Perform manual sampling (see quality_eval_20papers.py docstring)")
            print(f"  3. Generate final pass/fail decision")
            print(f"  4. Document any quality issues found")


if __name__ == "__main__":
    run_quality_evaluation()
