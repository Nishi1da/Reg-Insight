"""
Policy Router - Routes each regulation to relevant company policies only.

This fixes your #1 problem: the system currently compares every regulation
against every policy, which creates noise (DPDP Act vs AML Policy etc.)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── ROUTING TABLE ──────────────────────────────────────────────────────────
# Maps regulation keywords → which policy documents to search
# Keys are substrings that appear in your chunk metadata 'source' field
# Values list policy source filenames (as stored in your ChromaDB metadata)
#
# IMPORTANT: Check what your actual source filenames are in ChromaDB.
# Run this to see: 
#   from embeddings.chroma_manager import ChromaManager
#   db = ChromaManager()
#   coll = db.get_collection("regulations")
#   data = coll.get(limit=5)
#   print(set(m['source'] for m in data['metadatas']))
#
# Then update the lists below to match your actual filenames.

ROUTING_TABLE = {
    "aml": {
        "primary": ["AML Policy.pdf", "VDA_service_pvt_ltd.pdf"],
        "secondary": ["paytm.pdf", "Razorpay software.pdf"],
        "keywords": [
            "aml", "pmla", "vda", "anti-money", "fatf",
            "suspicious transaction", "str", "kyc",
            "service provider", "virtual digital asset",
            "reporting entity", "beneficial owner",
            "customer due diligence", "cdd", "edd",
            "politically exposed", "pep", "wire transfer"
        ]
    },
    "cert": {
        "primary": ["paytm.pdf", "bajaj finance.pdf", "Finsecure_pvt_ltd.pdf"],
        "secondary": ["Razorpay software.pdf"],
        "keywords": [
            "cert-in", "cert_in", "cyber incident",
            "cybersecurity", "information security",
            "cert", "csirt", "cyber security incident",
            "malware", "ransomware", "data breach notification",
            "vulnerability", "point of contact", "poc"
        ]
    },
    "dpdp": {
        "primary": ["Data Protection Policy.pdf", "datasafe_pvt.pdf"],
        "secondary": ["paytm.pdf", "bajaj finance.pdf"],
        "keywords": [
            "dpdp", "data protection", "personal data",
            "data fiduciary", "data principal",
            "consent manager", "digital personal data",
            "data processor", "significant data fiduciary",
            "data protection board", "data protection impact"
        ]
    },
    "nbfc": {
        "primary": ["bajaj finance.pdf", "Horizon_nbfc.pdf"],
        "secondary": ["paytm.pdf", "Razorpay software.pdf"],
        "keywords": [
            "nbfc", "it framework", "it policy",
            "information technology", "bcp", "disaster recovery",
            "is audit", "it strategy", "it steering",
            "computer assisted audit", "caats",
            "it governance", "it risk", "it infrastructure"
        ]
    },
    "rbi_digital": {
        "primary": ["Razorpay software.pdf", "swiftpay_soln.pdf"],
        "secondary": ["paytm.pdf", "bajaj finance.pdf"],
        "keywords": [
            "digital payment", "payment security",
            "res shall", "re shall", "regulated entity",
            "acquiring bank", "fraud risk management",
            "payment application", "mobile banking",
            "authentication", "tokenisation", "tokenization",
            "card payment", "payment infrastructure",
            "payment system", "payment product",
            "payment architecture", "secure by design",
            "customer protection", "grievance redress",
            "payment fraud", "transaction alert"
        ]
    }
}


class PolicyRouter:
    """
    Routes regulation chunks to relevant policy collections.
    
    Usage:
        router = PolicyRouter()
        policy_sources = router.get_relevant_policies(
            regulation_text="All SPs must implement KYC...",
            regulation_source="aml_guidelines.pdf"
        )
        # Returns: ["nirmal_bang_aml", "razorpay_payment"]
    """

    def __init__(self):
        self.routing_table = ROUTING_TABLE
        self.stats = {
            'total_routed': 0,
            'matched': 0,
            'unmatched': 0,
            'by_category': {}
        }

    def detect_category(
        self,
        regulation_text: str,
        regulation_source: str = ""
    ) -> Optional[str]:
        """
        Detect which regulatory category a chunk belongs to.
        
        Checks source filename first (more reliable),
        then falls back to keyword matching in text.
        """
        combined = (regulation_source + " " + regulation_text).lower()

        # Score each category by keyword hits
        scores = {}
        for category, config in self.routing_table.items():
            hits = sum(1 for kw in config["keywords"] if kw in combined)
            if hits > 0:
                scores[category] = hits

        if not scores:
            return None

        # Return category with most keyword hits
        return max(scores, key=scores.get)

    def get_relevant_policies(
        self,
        regulation_text: str,
        regulation_source: str = "",
        include_secondary: bool = True
    ) -> List[str]:
        """
        Get list of relevant policy source names for a regulation chunk.
        
        Args:
            regulation_text: The regulation chunk text
            regulation_source: The source filename from metadata
            include_secondary: Whether to include secondary policies
        
        Returns:
            List of policy source names to search against.
            Empty list means "skip this chunk entirely" (definitional etc.)
        """
        self.stats['total_routed'] += 1

        category = self.detect_category(regulation_text, regulation_source)

        if not category:
            self.stats['unmatched'] += 1
            logger.debug(f"No category detected for: {regulation_text[:60]}")
            # Return all policies rather than nothing - better than missing
            all_policies = []
            for config in self.routing_table.values():
                all_policies.extend(config["primary"])
            return list(set(all_policies))

        self.stats['matched'] += 1
        self.stats['by_category'][category] = (
            self.stats['by_category'].get(category, 0) + 1
        )

        config = self.routing_table[category]
        policies = list(config["primary"])

        if include_secondary:
            policies.extend(config["secondary"])

        return list(dict.fromkeys(policies))  # deduplicate preserving order

    def get_category_label(self, regulation_source: str, regulation_text: str) -> str:
        """Get human-readable category label"""
        labels = {
            "aml": "AML/CFT/PMLA",
            "cert": "CERT-In 2022",
            "dpdp": "DPDP Act 2023",
            "nbfc": "NBFC IT Framework",
            "rbi_digital": "RBI Digital Payments"
        }
        cat = self.detect_category(regulation_text, regulation_source)
        return labels.get(cat, "Unknown Regulation")

    def get_stats(self) -> Dict:
        return self.stats.copy()
    
    def get_policy_chunks_from_chroma(
            self,
            regulation_text: str,
            regulation_source: str,
            chroma_manager,
            query_embedding: list,
            top_k: int = 5
) -> list:
        """
    Query ChromaDB policies collection filtered to relevant sources only.
    
    This replaces the old approach of searching ALL policy chunks.
    Now only searches policy documents relevant to this regulation type.
    """
        relevant_sources = self.get_relevant_policies(
            regulation_text, regulation_source
            )
        if not relevant_sources:
            return []
        # Build ChromaDB where filter for relevant sources only
        if len(relevant_sources) == 1:
            where_filter = {"source": relevant_sources[0]}
        else:
            where_filter = {"source": {"$in": relevant_sources}}
        try:
            results = chroma_manager.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                collection_name="policies"
                )
            return results
        except Exception as e:
            # Fallback: search without filter if where clause fails
            import logging
            logging.getLogger(__name__).warning(
                f"Filtered query failed: {e}, falling back to unfiltered"
                )
            results = chroma_manager.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                collection_name="policies"
                )
            return results


# ── TEST ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    router = PolicyRouter()

    tests = [
        ("All SPs must implement KYC before onboarding customers.",
         "aml_guidelines.pdf"),
        ("The Data Fiduciary shall notify the Board of personal data breach.",
         "dpdp_act_2023.pdf"),
        ("NBFCs shall put in place a cyber-security policy.",
         "nbfc_it_framework.pdf"),
        ("REs shall implement multi-tier application architecture.",
         "rbi_digital_payments.pdf"),
        ("(r) notification means a notification published in the gazette.",
         "dpdp_act_2023.pdf"),  # definition
    ]

    print("Policy Router Test")
    print("=" * 60)
    for text, source in tests:
        policies = router.get_relevant_policies(text, source)
        category = router.detect_category(text, source)
        print(f"\nText: {text[:60]}...")
        print(f"Category: {category}")
        print(f"Route to: {policies}")

    print(f"\nStats: {router.get_stats()}")