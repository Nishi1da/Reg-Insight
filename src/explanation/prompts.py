"""Prompt Templates for LLM Explanation Generation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass
class PromptVersion:
    """Track prompt versions for reproducibility"""
    version_id: str
    created_at: str
    description: str
    template_hash: str

    def to_dict(self) -> Dict:
        return {
            'version_id': self.version_id,
            'created_at': self.created_at,
            'description': self.description,
            'template_hash': self.template_hash
        }


class PromptTemplateManager:
    """
    Manages prompt templates with versioning

    Features:
    - Version tracking for reproducibility
    - Template hashing
    - Simple string formatting
    """

    def __init__(self):
        self.templates: Dict[str, Dict] = {}
        self.current_versions: Dict[str, str] = {}
        self._register_default_templates()

    def _register_default_templates(self):
        """Register default prompt templates"""

        # System prompt - used in all calls
        self.system_prompt = """You are a senior regulatory compliance analyst.
Your job is to analyze gaps between regulatory requirements and company policies.
Always respond with valid JSON only. No explanations before or after the JSON.
Be specific, factual, and actionable. Never invent information."""

        # Template v1: Basic gap analysis
        gap_analysis_v1 = """Analyze this regulation vs policy gap.

REGULATION:
{regulation_text}

POLICY:
{policy_text}

SCORES:
- Match score: {final_score:.2f}/1.0
- Semantic: {cross_encoder_score:.2f}
- Keyword: {bi_encoder_score:.2f}

Respond ONLY with this JSON:
{{
    "summary": "1-2 sentence gap assessment (max 200 chars)",
    "recommendation": "Specific action to close gap",
    "risk_level": "low|medium|high|critical",
    "key_differences": ["gap 1", "gap 2", "gap 3"],
    "confidence": "high|medium|low"
}}"""

        self.register_template(
            name="gap_analysis",
            version="1.0.0",
            template=gap_analysis_v1,
            description="Basic gap analysis"
        )

        # Template v2: Enhanced with context (recommended)
        gap_analysis_v2 = """Analyze this regulatory compliance gap.

CONTEXT:
- Regulation: {regulation_source}
- Policy: {policy_source}
- Classification: {classification}
- System confidence: {confidence:.2f}

REGULATION REQUIREMENT:
{regulation_text}

COMPANY POLICY:
{policy_text}

SIMILARITY SCORES:
- Final score: {final_score:.2f} (0=no match, 1=perfect)
- Cross-encoder: {cross_encoder_score:.2f}
- Bi-encoder: {bi_encoder_score:.2f}

Analyze and respond ONLY with this JSON structure:
{{
    "summary": "Concise gap assessment (max 200 chars)",
    "recommendation": "Specific actionable compliance fix",
    "risk_level": "low|medium|high|critical",
    "key_differences": [
        "Specific gap or mismatch 1",
        "Specific gap or mismatch 2",
        "Specific gap or mismatch 3"
    ],
    "regulatory_intent": "What this regulation aims to achieve",
    "policy_coverage": "What the policy covers and misses",
    "remediation_priority": "immediate|short_term|long_term",
    "confidence": "high|medium|low"
}}"""

        self.register_template(
            name="gap_analysis",
            version="2.0.0",
            template=gap_analysis_v2,
            description="Enhanced gap analysis with context"
        )

        # Template: Unsupported requirement
        unsupported_v1 = """Analyze this regulation requirement that has NO matching policy.

REGULATION:
{regulation_text}

SOURCE: {regulation_source} | Page: {page_number}
SEVERITY: {severity} | Score: {severity_score:.2f}

This requirement has zero policy coverage. Analyze and respond ONLY with JSON:
{{
    "summary": "Why this gap matters (max 200 chars)",
    "recommendation": "What policy to create",
    "risk_level": "low|medium|high|critical",
    "key_differences": ["Missing: element 1", "Missing: element 2"],
    "policy_type_needed": "e.g., Data Retention Policy",
    "implementation_complexity": "high|medium|low",
    "urgency": "immediate|high|medium|low",
    "estimated_effort": "e.g., 2-4 weeks",
    "confidence": "high|medium|low"
}}"""

        self.register_template(
            name="unsupported_analysis",
            version="1.0.0",
            template=unsupported_v1,
            description="Analysis for unmatched requirements"
        )

    def register_template(
        self,
        name: str,
        version: str,
        template: str,
        description: str
    ) -> PromptVersion:
        """Register a template version"""
        template_hash = hashlib.md5(template.encode()).hexdigest()[:12]

        version_obj = PromptVersion(
            version_id=f"{name}@{version}",
            created_at=datetime.now().isoformat(),
            description=description,
            template_hash=template_hash
        )

        if name not in self.templates:
            self.templates[name] = {}

        self.templates[name][version] = {
            'template': template,
            'metadata': version_obj
        }

        if name not in self.current_versions:
            self.current_versions[name] = version

        return version_obj

    def get_template(
        self,
        name: str,
        version: Optional[str] = None
    ) -> str:
        """Get template by name and version"""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")

        use_version = version or self.current_versions.get(name)
        if use_version not in self.templates[name]:
            raise ValueError(
                f"Version '{use_version}' not found for '{name}'"
            )

        return self.templates[name][use_version]['template']

    def format_prompt(
        self,
        name: str,
        variables: Dict,
        version: Optional[str] = None
    ) -> str:
        """Format template with variables"""
        template = self.get_template(name, version)
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(
                f"Missing variable {e} for template '{name}'"
            )

    def set_current_version(self, name: str, version: str):
        """Set active version"""
        if (name not in self.templates
                or version not in self.templates[name]):
            raise ValueError("Template or version not found")
        self.current_versions[name] = version

    def list_versions(self, name: Optional[str] = None) -> Dict:
        """List available versions"""
        if name:
            return {
                name: {
                    v: data['metadata'].to_dict()
                    for v, data in self.templates.get(name, {}).items()
                }
            }
        return {
            n: {
                v: data['metadata'].to_dict()
                for v, data in versions.items()
            }
            for n, versions in self.templates.items()
        }

    def compare_versions(self, name: str, v1: str, v2: str) -> Dict:
        """Compare two template versions"""
        t1 = self.templates[name][v1]['template']
        t2 = self.templates[name][v2]['template']
        return {
            'version_1': v1,
            'version_2': v2,
            'length_diff': len(t2) - len(t1),
            'hash_1': hashlib.md5(t1.encode()).hexdigest()[:12],
            'hash_2': hashlib.md5(t2.encode()).hexdigest()[:12]
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 29: Prompt Template Manager Test")
    print("=" * 60)

    manager = PromptTemplateManager()

    print("\n1. Available templates:")
    for name, vers in manager.list_versions().items():
        print(f"   - {name}: {list(vers.keys())}")

    print("\n2. Testing v2.0.0 prompt formatting...")
    variables = {
        'regulation_text': 'Organizations must encrypt all data at rest.',
        'policy_text': 'We use industry standard encryption algorithms.',
        'final_score': 0.55,
        'cross_encoder_score': 0.60,
        'bi_encoder_score': 0.48,
        'regulation_source': 'GDPR Article 32',
        'policy_source': 'IT Security Policy v2.1',
        'classification': 'partial',
        'confidence': 0.78
    }

    prompt = manager.format_prompt("gap_analysis", variables, "2.0.0")
    print(f"   Prompt length: {len(prompt)} chars")
    print(f"   Contains regulation text: "
          f"{'PASS' if 'encrypt' in prompt else 'FAIL'}")
    print(f"   Contains scores: "
          f"{'PASS' if '0.55' in prompt else 'FAIL'}")

    print("\n3. System prompt preview:")
    print(f"   {manager.system_prompt[:100]}...")

    print("\n4. Version comparison:")
    diff = manager.compare_versions("gap_analysis", "1.0.0", "2.0.0")
    print(f"   v2 is {diff['length_diff']} chars longer than v1")

    print("\n" + "=" * 60)
    print("=" * 60)