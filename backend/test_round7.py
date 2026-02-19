"""Round 7 test with symmetric formula normalization."""
from pathlib import Path
from app.ingest.pipeline import ingest_markdowns

# Same 10 papers
papers = [
    "01_1478", "03_491", "05_340", "07_1605",
    "09_1007", "11_251", "12_1606", "15_1396",
    "02_1050", "04_1228"
]

base_path = Path("C:/Users/D0n9/Desktop/hzy_paper/selected_20_md_with_images")
md_files = []

for paper_id in papers:
    md_path = base_path / paper_id / "paper.md"
    if md_path.exists():
        md_files.append(str(md_path))
        print(f"[OK] {paper_id}")

print(f"\n=== Round 7 Test ===")
print(f"Papers: {len(md_files)}")
print(f"Optimizations:")
print(f"  1. Token-window matching (Round 4: 17.4% -> 61.2%)")
print(f"  2. Symmetric formula normalization (P0)")
print(f"     - LaTeX commands: \\mathrm{{}}, \\text{{}}, \\mathbf{{}}")
print(f"     - Space removal: 'σ 1' -> 'σ1'")
print(f"     - Greek letters: theta->θ, alpha->α, etc.")
print(f"Target: 02_1050 (36.6% -> 45%+), Overall (61.2% -> 65%+)\n")

result = ingest_markdowns(md_files)

print("\n=== Result ===")
print(result)
