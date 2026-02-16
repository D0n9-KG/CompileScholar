"""
Test retry logic for update_similarity_for_paper (Commit 1: remove lexical fallback)
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from app.similarity.service import update_similarity_for_paper


class TestSimilarityUpdateRetry(unittest.TestCase):
    """Test that update_similarity_for_paper retries embedding failures and never falls back to lexical."""

    def test_embedding_retry_success_on_third_attempt(self):
        """Test that embedding succeeds on 3rd attempt with no degradation."""
        paper_id = "test_paper_001"

        # Mock Neo4j client to return test data
        with patch("app.similarity.service.Neo4jClient") as mock_neo4j_cls:
            mock_client = MagicMock()
            mock_neo4j_cls.return_value.__enter__.return_value = mock_client

            # Setup: claims and logic steps exist
            mock_client.list_propositions_for_paper.return_value = [
                {"prop_id": "claim_001", "text": "Test claim 1"}
            ]
            mock_client.list_logic_steps_for_paper.return_value = [
                {"step_id": "logic_001", "description": "Test logic 1"}
            ]

            # Mock embedding function to fail twice, then succeed
            call_count = {"count": 0}

            def mock_embed(*args, **kwargs):
                call_count["count"] += 1
                if call_count["count"] < 3:
                    raise RuntimeError("Error code: 502")
                # Success on 3rd call
                import numpy as np
                return np.array([[0.1, 0.2, 0.3]])

            with patch("app.similarity.service._embed_items", side_effect=mock_embed):
                # Should succeed after retry
                result = update_similarity_for_paper(paper_id)

                # Verify no degradation
                self.assertTrue(result.get("ok"))
                self.assertEqual(result.get("mode"), "embedding")
                self.assertIsNone(result.get("degraded_kinds"))
                self.assertIsNone(result.get("degradation_events"))

                # Verify retry happened (3 calls total)
                self.assertEqual(call_count["count"], 3)

    def test_embedding_retry_fails_after_three_attempts(self):
        """Test that embedding failure after 3 retries raises clear error."""
        paper_id = "test_paper_002"

        with patch("app.similarity.service.Neo4jClient") as mock_neo4j_cls:
            mock_client = MagicMock()
            mock_neo4j_cls.return_value.__enter__.return_value = mock_client

            mock_client.list_propositions_for_paper.return_value = [
                {"prop_id": "claim_002", "text": "Test claim 2"}
            ]
            mock_client.list_logic_steps_for_paper.return_value = []

            # Mock embedding to always fail
            call_count = {"count": 0}

            def mock_embed_fail(*args, **kwargs):
                call_count["count"] += 1
                raise RuntimeError("Error code: 502")

            with patch("app.similarity.service._embed_items", side_effect=mock_embed_fail):
                # Should raise RuntimeError after 3 attempts
                with self.assertRaises(RuntimeError) as ctx:
                    update_similarity_for_paper(paper_id)

                error_msg = str(ctx.exception)
                # Verify error mentions attempts and kind
                self.assertIn("3 attempts", error_msg)
                self.assertIn("embedding unavailable", error_msg.lower())

                # Verify 3 retries happened
                self.assertEqual(call_count["count"], 3)

    def test_embedding_never_falls_back_to_lexical(self):
        """Test that lexical fallback is completely removed - no lexical mode in results."""
        paper_id = "test_paper_003"

        with patch("app.similarity.service.Neo4jClient") as mock_neo4j_cls:
            mock_client = MagicMock()
            mock_neo4j_cls.return_value.__enter__.return_value = mock_client

            mock_client.list_propositions_for_paper.return_value = [
                {"prop_id": "claim_003", "text": "Test claim 3"}
            ]
            mock_client.list_logic_steps_for_paper.return_value = []

            # Success case
            import numpy as np
            with patch("app.similarity.service._embed_items", return_value=np.array([[0.1, 0.2, 0.3]])):
                result = update_similarity_for_paper(paper_id)

                # Verify mode is embedding, never lexical
                self.assertEqual(result.get("mode"), "embedding")
                self.assertNotEqual(result.get("mode"), "lexical")
                self.assertNotEqual(result.get("mode"), "mixed")

                # Verify no degradation fields
                self.assertNotIn("degraded_kinds", result)
                self.assertNotIn("degradation_events", result)


if __name__ == "__main__":
    unittest.main()
