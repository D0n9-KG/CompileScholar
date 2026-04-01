from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.ingest.parse_md import _coerce_windows_extended_path, parse_mineru_markdown


class ParseMarkdownSectionsTests(unittest.TestCase):
    def test_sections_are_bound_per_block_not_global(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "paper.md"
            p.write_text(
                "# Intro\n\n"
                "Intro body sentence.\n\n"
                "# Method\n\n"
                "Method body sentence.\n\n"
                "# REFERENCES\n\n"
                "[1] Ref entry.\n",
                encoding="utf-8",
            )
            doc = parse_mineru_markdown(str(p))

        blocks = [c for c in doc.chunks if c.kind == "block"]
        intro_blocks = [c for c in blocks if "Intro body sentence." in c.text]
        method_blocks = [c for c in blocks if "Method body sentence." in c.text]
        self.assertEqual(len(intro_blocks), 1)
        self.assertEqual(len(method_blocks), 1)
        self.assertEqual(intro_blocks[0].section, "Intro")
        self.assertEqual(method_blocks[0].section, "Method")

    def test_reference_section_doi_is_not_promoted_to_paper_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "paper.md"
            p.write_text(
                "# DEM investigation of particle anti-rotation effects on the micromechanical response of granular materials\n\n"
                "Bo Zhou, Runqiu Huang\n\n"
                "Main body paragraph.\n\n"
                "# References\n\n"
                "[1] Example cited paper. doi:10.1061/(ASCE)GT.1943-5606.0000890\n",
                encoding="utf-8",
            )
            doc = parse_mineru_markdown(str(p))

        self.assertIsNone(doc.paper.doi)

    def test_bilingual_title_alt_prefers_real_title_over_numbered_section_heading(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "paper.md"
            p.write_text(
                "# 改性双基推进剂松弛模量的确定方法\n\n"
                "孟红磊，鞠玉涛，周长省\n\n"
                "# DETERMINATION WAY OF RELAXATION MODULUS OF MODIFIED DB PROPELLANT\n\n"
                "MENG Hong-lei, JU Yu-tao, ZHOU Chang-sheng\n\n"
                "# 1.3 Sorvari法\n\n"
                "Section body.\n",
                encoding="utf-8",
            )
            doc = parse_mineru_markdown(str(p))

        self.assertEqual(doc.paper.title, "DETERMINATION WAY OF RELAXATION MODULUS OF MODIFIED DB PROPELLANT")
        self.assertEqual(doc.paper.title_alt, "改性双基推进剂松弛模量的确定方法")

    def test_primary_title_prefers_clean_prebody_heading_over_later_all_caps_section_heading(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "paper.md"
            p.write_text(
                "# Initiation of Solid Explosives by Mechanical Impact\n\n"
                "V. I. Pepekin, B. L. Korsunskii\n\n"
                "# INTRODUCTION\n\n"
                "Intro body.\n\n"
                "# RELATION BETWEEN THE CRITICAL PRESSURE OF EXPLOSION INITIATION AND THE VOLUMETRIC HEAT OF EXPLOSION\n\n"
                "Section body.\n",
                encoding="utf-8",
            )
            doc = parse_mineru_markdown(str(p))

        self.assertEqual(doc.paper.title, "Initiation of Solid Explosives by Mechanical Impact")
        self.assertIsNone(doc.paper.title_alt)

    def test_front_matter_headings_and_citation_blocks_are_not_promoted_into_metadata_or_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "paper.md"
            p.write_text(
                "# Accepted Manuscript\n\n"
                "Early front matter.\n\n"
                "# Highlights\n\n"
                "- Placeholder highlights.\n\n"
                "# Adversarial Uncertainty Quantification in Physics-Informed Neural Networks\n\n"
                "Citation: Journal of Example 1, 1 (2024); doi: 10.0000/example\n\n"
                "View Table of Contents: https://example.com/toc\n\n"
                "Published by Example Press\n\n"
                "# Abstract\n\n"
                "We present a probabilistic physics-informed neural network for uncertainty quantification.\n",
                encoding="utf-8",
            )
            doc = parse_mineru_markdown(str(p))

        self.assertEqual(doc.paper.title, "Adversarial Uncertainty Quantification in Physics-Informed Neural Networks")
        self.assertIsNone(doc.paper.title_alt)
        chunk_texts = [chunk.text for chunk in doc.chunks]
        self.assertFalse(any("Accepted Manuscript" in text for text in chunk_texts))
        self.assertFalse(any("Citation:" in text for text in chunk_texts))
        self.assertFalse(any("View Table of Contents:" in text for text in chunk_texts))
        self.assertFalse(any("Published by Example Press" in text for text in chunk_texts))
        self.assertTrue(
            any(
                "probabilistic physics-informed neural network" in text
                for text in chunk_texts
            )
        )

    def test_pre_abstract_ad_heading_and_brand_block_are_not_kept_as_title_alt_or_content(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "paper.md"
            p.write_text(
                "# Shear thickening in dense non-Brownian suspensions: Viscous to inertial transition\n\n"
                "Y. Madraki, A. Oakley\n\n"
                "# ARTICLES YOU MAY BE INTERESTED IN\n\n"
                "Recommended paper.\n\n"
                "# True powder rheology\n\n"
                "Find out more\n\n"
                "Anton Paar\n\n"
                "# Shear thickening in dense non-Brownian suspensions: Viscous to inertial transition\n\n"
                "Y. Madraki, A. Oakley\n\n"
                "# Abstract\n\n"
                "We present an experimental study on the viscous to inertial mode of shear thickening.\n",
                encoding="utf-8",
            )
            doc = parse_mineru_markdown(str(p))

        self.assertEqual(doc.paper.title, "Shear thickening in dense non-Brownian suspensions: Viscous to inertial transition")
        self.assertIsNone(doc.paper.title_alt)
        chunk_texts = [chunk.text for chunk in doc.chunks]
        self.assertFalse(any("True powder rheology" in text for text in chunk_texts))
        self.assertFalse(any("Anton Paar" in text for text in chunk_texts))
        self.assertTrue(any("experimental study on the viscous to inertial mode" in text for text in chunk_texts))

    def test_coerce_windows_extended_path_for_long_unc(self) -> None:
        raw = (
            "\\\\192.168.199.138\\Share400T\\pub\\LLM_Data\\data\\hzy\\第一批文献\\文献\\文献中心\\"
            "HZY第一批论文全文\\output\\863_Decision Tree Classification of Land Cover from Remotely Sensed Data\\"
            "863_Decision_Tree_Classification_of_Land_Cover_from_Remotely_Sensed_Data\\"
            "863_Decision_Tree_Classification_of_Land_Cover_from_Remotely_Sensed_Data.md"
        )

        coerced = _coerce_windows_extended_path(raw)

        self.assertTrue(coerced.startswith("\\\\?\\UNC\\192.168.199.138\\Share400T\\"))


    def test_author_line_html_sup_tags_are_stripped_from_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "paper.md"
            p.write_text(
                "# Machine-learning prediction for safety of RDX-CMDB propellants\n\n"
                "郭延芝<sup>1</sup>, 吴艳玲<sup>2</sup>, 刘润青<sup>3</sup>\n\n"
                "Main body paragraph.\n",
                encoding="utf-8",
            )
            doc = parse_mineru_markdown(str(p))

        self.assertEqual(doc.paper.authors, ["郭延芝", "吴艳玲", "刘润青"])


if __name__ == "__main__":
    unittest.main()

