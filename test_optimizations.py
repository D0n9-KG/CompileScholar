"""
Test script for P0 extraction optimizations.

This script:
1. Clears Neo4j database
2. Selects representative papers with known issues
3. Runs ingestion with optimized settings
4. Compares results with baseline
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.graph.neo4j_client import Neo4jClient
from app.settings import settings


def clear_neo4j():
    """Clear all data from Neo4j database."""
    print("\n=== Clearing Neo4j Database ===")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        # Delete all nodes and relationships
        with client._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("[OK] All Neo4j data cleared")

        # Recreate schema
        client.ensure_schema()
        print("[OK] Schema recreated")


def select_test_papers(source_dir: Path, num_papers: int = 5):
    """Select representative papers for testing."""
    papers_dir = Path(source_dir)
    if not papers_dir.exists():
        print(f"Error: {papers_dir} does not exist")
        return []

    # Get all subdirectories (each is a paper)
    all_papers = [d for d in papers_dir.iterdir() if d.is_dir() and (d / "images").exists()]

    if not all_papers:
        print(f"Error: No paper directories with 'images' folder found in {papers_dir}")
        return []

    print(f"\n=== Found {len(all_papers)} papers in {papers_dir} ===")

    # Select papers: prioritize those that had issues in baseline
    # For now, select first N papers
    selected = all_papers[:num_papers]

    print(f"\n=== Selected {len(selected)} papers for testing: ===")
    for i, p in enumerate(selected, 1):
        md_files = list(p.glob("*.md"))
        print(f"{i}. {p.name} ({len(md_files)} markdown files)")

    return selected


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test P0 extraction optimizations")
    parser.add_argument(
        "--source-dir",
        type=str,
        default=r"C:\Users\D0n9\Desktop\hzy_paper\selected_20_md_with_images",
        help="Source directory containing paper folders"
    )
    parser.add_argument(
        "--num-papers",
        type=int,
        default=5,
        help="Number of papers to test (default: 5)"
    )
    parser.add_argument(
        "--skip-clear",
        action="store_true",
        help="Skip Neo4j database clearing"
    )

    args = parser.parse_args()

    # Step 1: Clear Neo4j (unless skipped)
    if not args.skip_clear:
        clear_neo4j()
    else:
        print("\n=== Skipping Neo4j clear ===")

    # Step 2: Select test papers
    selected_papers = select_test_papers(Path(args.source_dir), args.num_papers)

    if not selected_papers:
        print("\nNo papers selected. Exiting.")
        return

    # Step 3: Show ingestion command
    print("\n=== Next Steps ===")
    print("\n1. Start the project (if not running):")
    print("   ./run.ps1")
    print("\n2. Navigate to the Ingest page in the browser:")
    print("   http://127.0.0.1:<frontend-port>/ingest")
    print("\n3. Select the following paper directories to upload:")
    for p in selected_papers:
        print(f"   - {p}")
    print("\n4. Or use the API directly:")
    print("   You can also use the /ingest/path endpoint via FastAPI docs")
    print(f"   POST to http://127.0.0.1:<backend-port>/ingest/path")
    print(f'   with body: {{"root_path": "{args.source_dir}"}}')
    print("\nNote: The optimizations are now active via schema v7 configuration.")
    print("      Check backend/storage/schemas/research/v7.json for details.")


if __name__ == "__main__":
    main()
