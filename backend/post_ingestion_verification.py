"""
Post-Ingestion Verification Script for 20-Paper E2E Test

Runs after ingestion completes to verify data integrity before quality evaluation.
"""
from __future__ import annotations

from app.graph.neo4j_client import Neo4jClient
from app.settings import Settings


def run_post_ingestion_verification():
    """Run all post-ingestion verification checks."""
    settings = Settings()

    print("="*80)
    print("Post-Ingestion Verification - 20 Paper E2E Test")
    print("="*80)

    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        with client._driver.session() as session:

            # ==================================================================
            # A. Graph Integrity Checks
            # ==================================================================
            print("\n" + "="*80)
            print("GRAPH INTEGRITY CHECKS")
            print("="*80)

            # Papers ingested
            query = "MATCH (p:Paper) WHERE p.ingested=true RETURN count(p) AS count"
            result = session.run(query).single()
            papers_count = result['count']
            print(f"\nPapers ingested: {papers_count}")
            print(f"  [OK] Expected: 20" if papers_count == 20 else f"  [WARN]  Expected 20, got {papers_count}")

            # Claims generated
            query = "MATCH (:Paper)-[:HAS_CLAIM]->(c:Claim) RETURN count(c) AS count"
            result = session.run(query).single()
            claims_count = result['count']
            print(f"\nClaims generated: {claims_count}")
            print(f"  [OK] Target: >0" if claims_count > 0 else f"  [FAIL] No claims found!")

            # Claims mapped to propositions
            query = "MATCH (c:Claim) WHERE NOT (c)-[:MAPS_TO]->(:Proposition) RETURN count(c) AS count"
            result = session.run(query).single()
            unmapped_claims = result['count']
            print(f"\nUnmapped claims (orphans): {unmapped_claims}")
            print(f"  [OK] Target: ~0" if unmapped_claims < claims_count * 0.05 else f"  [WARN]  High orphan rate: {unmapped_claims}/{claims_count}")

            # Propositions without claims (should be ~0)
            query = "MATCH (pr:Proposition) WHERE NOT (pr)<-[:MAPS_TO]-(:Claim) RETURN count(pr) AS count"
            result = session.run(query).single()
            orphan_props = result['count']
            print(f"\nOrphan propositions (no claims): {orphan_props}")
            print(f"  [OK] Target: ~0" if orphan_props == 0 else f"  [WARN]  Found {orphan_props} orphan propositions")

            # Logic steps
            query = "MATCH (:Paper)-[:HAS_LOGIC_STEP]->(s:LogicStep) RETURN count(s) AS count"
            result = session.run(query).single()
            logic_steps_count = result['count']
            print(f"\nLogic steps: {logic_steps_count}")
            print(f"  [OK] Target: >0" if logic_steps_count > 0 else f"  [FAIL] No logic steps found!")

            # References
            query = "MATCH (:Paper)-[:HAS_REFERENCE]->(r:ReferenceEntry) RETURN count(r) AS count"
            result = session.run(query).single()
            refs_count = result['count']
            print(f"\nReferences: {refs_count}")
            print(f"  [OK] Target: >0" if refs_count > 0 else f"  [WARN]  No references found")

            # ==================================================================
            # B. Gate Statistics
            # ==================================================================
            print("\n" + "="*80)
            print("GATE STATISTICS")
            print("="*80)

            # Phase1 gate pass rate
            query = """
            MATCH (p:Paper) WHERE p.ingested=true
            RETURN count(CASE WHEN p.phase1_gate_passed=true THEN 1 END) AS passed,
                   count(p) AS total,
                   round(100.0 * count(CASE WHEN p.phase1_gate_passed=true THEN 1 END) / count(p), 2) AS pass_rate_pct
            """
            result = session.run(query).single()
            print(f"\nPhase1 gate pass rate:")
            print(f"  Passed: {result['passed']}/{result['total']}")
            print(f"  Rate: {result['pass_rate_pct']}%")
            print(f"  [OK] Target: >80%" if result['pass_rate_pct'] >= 80 else f"  [WARN]  Below 80% target")

            # Quality tier distribution
            query = """
            MATCH (p:Paper) WHERE p.ingested=true
            RETURN coalesce(p.phase1_quality_tier,'unknown') AS tier, count(*) AS papers
            ORDER BY papers DESC
            """
            results = list(session.run(query))
            print(f"\nQuality tier distribution:")
            for row in results:
                print(f"  {row['tier']}: {row['papers']} papers")

            # ==================================================================
            # C. Proposition Grouping Coverage
            # ==================================================================
            print("\n" + "="*80)
            print("PROPOSITION GROUPING")
            print("="*80)

            # Check if clustering ran
            query = "MATCH (pg:PropositionGroup) RETURN count(pg) AS count"
            result = session.run(query).single()
            groups_count = result['count']
            print(f"\nProposition groups: {groups_count}")
            if groups_count == 0:
                print(f"  [FAIL] No groups found - clustering may not have run")
            else:
                print(f"  [OK] Clustering completed")

                # Grouping coverage
                query = """
                MATCH (pr:Proposition)
                OPTIONAL MATCH (pr)-[:IN_GROUP]->(pg:PropositionGroup)
                WITH count(DISTINCT pr.prop_id) AS total_props,
                     count(DISTINCT CASE WHEN pg IS NOT NULL THEN pr.prop_id END) AS grouped_props
                RETURN total_props, grouped_props,
                       round(100.0 * toFloat(grouped_props)/toFloat(CASE WHEN total_props=0 THEN 1 ELSE total_props END),2) AS coverage_pct
                """
                result = session.run(query).single()
                print(f"\nGrouping coverage:")
                print(f"  Grouped propositions: {result['grouped_props']}/{result['total_props']}")
                print(f"  Coverage: {result['coverage_pct']}%")
                print(f"  [OK] Target: >80%" if result['coverage_pct'] >= 80 else f"  [WARN]  Below 80% target")

            # ==================================================================
            # D. Conflict Detection
            # ==================================================================
            print("\n" + "="*80)
            print("CONFLICT DETECTION")
            print("="*80)

            # Check for conflict relationships
            query = "MATCH ()-[r:CONFLICTS_WITH]->() RETURN count(r) AS count"
            result = session.run(query).single()
            conflicts_count = result['count']
            print(f"\nConflict relationships: {conflicts_count}")
            if conflicts_count == 0:
                print(f"  [WARN]  No conflicts found - may indicate detection didn't run or no actual conflicts")
            else:
                print(f"  [OK] Conflict detection completed")

            # ==================================================================
            # SUMMARY
            # ==================================================================
            print("\n" + "="*80)
            print("VERIFICATION SUMMARY")
            print("="*80)

            all_checks_passed = (
                papers_count == 20 and
                claims_count > 0 and
                unmapped_claims < claims_count * 0.05 and
                orphan_props == 0 and
                logic_steps_count > 0
            )

            if all_checks_passed:
                print("\n[OK] All integrity checks PASSED")
                print("\nNext step: Run quality_eval_20papers.py for detailed quality metrics")
            else:
                print("\n[WARN]  Some checks did not pass - review output above")
                print("\nRecommendation: Investigate issues before running quality evaluation")


if __name__ == "__main__":
    run_post_ingestion_verification()
