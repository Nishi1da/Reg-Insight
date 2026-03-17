"""Pydantic Schemas for LLM Explanation Output"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RemediationPriority(str, Enum):
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class ImplementationComplexity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GapExplanation(BaseModel):
    """
    Structured explanation for a regulation-policy gap
    Primary output schema for LLM explanations
    """

    schema_version: str = Field(default="1.0.0")
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    summary: str = Field(..., max_length=200)
    recommendation: str = Field(..., min_length=10)
    risk_level: RiskLevel = Field(...)
    key_differences: List[str] = Field(
        ...,
        min_length=1,    # Fixed: was min_items
        max_length=6     # Fixed: was max_items
    )
    confidence: ConfidenceLevel = Field(...)
    regulatory_intent: Optional[str] = Field(None, max_length=300)
    policy_coverage: Optional[str] = Field(None, max_length=300)
    remediation_priority: Optional[RemediationPriority] = Field(None)

    @field_validator('summary')    # Fixed: was @validator
    @classmethod                   # Required in Pydantic V2
    def summary_not_empty(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError('Summary too short')
        return v.strip()

    @field_validator('key_differences')    # Fixed: was @validator
    @classmethod                           # Required in Pydantic V2
    def validate_differences(cls, v):
        if not v:
            raise ValueError('At least one difference required')
        return [d for d in v if len(d) >= 5]

    def to_dict(self) -> dict:
        return self.model_dump()    # Fixed: was .dict()

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)  # Fixed: was .json()

    def get_risk_score(self) -> int:
        return {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }.get(self.risk_level, 0)


class UnsupportedExplanation(BaseModel):
    """Explanation for requirements with no policy match"""

    schema_version: str = Field(default="1.0.0")
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    summary: str = Field(..., max_length=200)
    recommendation: str = Field(..., min_length=10)
    risk_level: RiskLevel = Field(...)
    key_differences: List[str] = Field(
        ...,
        min_length=1,    # Fixed: was min_items
        max_length=6     # Fixed: was max_items
    )
    policy_type_needed: str = Field(...)
    implementation_complexity: ImplementationComplexity = Field(...)
    urgency: str = Field(..., pattern="^(immediate|high|medium|low)$")
    estimated_effort: str = Field(...)
    confidence: ConfidenceLevel = Field(...)

    def to_dict(self) -> dict:
        return self.model_dump()    # Fixed: was .dict()


class ExplanationOutput(BaseModel):
    """Complete explanation output wrapper"""

    explanation_type: Literal["gap", "unsupported"] = Field(...)
    regulation_chunk_id: str = Field(...)
    policy_chunk_id: Optional[str] = Field(None)
    explanation: GapExplanation = Field(...)
    llm_model: str = Field(default="llama3-8b-8192")
    prompt_version: str = Field(...)
    generation_time_ms: float = Field(...)
    parsing_success: bool = Field(default=True)

    def to_dict(self) -> dict:
        return {
            'explanation_type': self.explanation_type,
            'regulation_chunk_id': self.regulation_chunk_id,
            'policy_chunk_id': self.policy_chunk_id,
            'explanation': self.explanation.to_dict(),
            'metadata': {
                'llm_model': self.llm_model,
                'prompt_version': self.prompt_version,
                'generation_time_ms': self.generation_time_ms,
                'parsing_success': self.parsing_success
            }
        }


class SchemaValidator:
    """Validates LLM output with auto-fix recovery"""

    def __init__(self):
        self.stats = {
            'total': 0, 'success': 0,
            'failed': 0, 'fixed': 0
        }

    def validate_gap_explanation(
        self,
        data: dict,
        attempt_fix: bool = True
    ) -> tuple:
        """
        Validate against GapExplanation schema

        Returns: (success, explanation_object, error_message)
        """
        self.stats['total'] += 1

        try:
            explanation = GapExplanation(**data)
            self.stats['success'] += 1
            return True, explanation, None

        except Exception as e:
            self.stats['failed'] += 1

            if attempt_fix:
                fixed = self._attempt_fix(data)
                try:
                    explanation = GapExplanation(**fixed)
                    self.stats['fixed'] += 1
                    self.stats['success'] += 1
                    self.stats['failed'] -= 1
                    return True, explanation, f"Auto-fixed: {str(e)}"
                except Exception as e2:
                    return False, None, f"{str(e)} | Fix failed: {str(e2)}"

            return False, None, str(e)

    def _attempt_fix(self, data: dict) -> dict:
        """Fix common validation errors from LLM output"""
        fixed = data.copy()

        # Fix 1: Truncate long summary
        if 'summary' in fixed and len(str(fixed['summary'])) > 200:
            fixed['summary'] = str(fixed['summary'])[:197] + "..."

        # Fix 2: Ensure summary is long enough
        if 'summary' in fixed and len(str(fixed['summary'])) < 10:
            fixed['summary'] = str(fixed['summary']) + " - review required"

        # Fix 3: Ensure key_differences is a list
        if 'key_differences' in fixed:
            if isinstance(fixed['key_differences'], str):
                fixed['key_differences'] = [fixed['key_differences']]
            elif not isinstance(fixed['key_differences'], list):
                fixed['key_differences'] = [str(fixed['key_differences'])]
        else:
            fixed['key_differences'] = ["No differences identified"]

        # Fix 4: Normalize risk_level
        if 'risk_level' in fixed:
            rl = str(fixed['risk_level']).lower().strip()
            if 'critical' in rl:
                fixed['risk_level'] = 'critical'
            elif 'high' in rl:
                fixed['risk_level'] = 'high'
            elif 'medium' in rl or 'med' in rl:
                fixed['risk_level'] = 'medium'
            else:
                fixed['risk_level'] = 'low'

        # Fix 5: Normalize confidence
        if 'confidence' in fixed:
            conf = str(fixed['confidence']).lower().strip()
            if conf not in ['high', 'medium', 'low']:
                fixed['confidence'] = 'medium'

        # Fix 6: Ensure all required fields exist
        defaults = {
            'summary': 'Gap analysis completed - review required',
            'recommendation': 'Review regulation and update policy accordingly',
            'risk_level': 'medium',
            'confidence': 'low',
            'key_differences': ['Manual review needed']
        }
        for field, default in defaults.items():
            if field not in fixed or not fixed[field]:
                fixed[field] = default

        return fixed

    def get_stats(self) -> dict:
        total = self.stats['total']
        return {
            **self.stats,
            'success_rate': round(
                self.stats['success'] / total * 100
                if total > 0 else 0, 1
            )
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 31: Schema Validation Test")
    print("=" * 60)

    print("\n1. Valid explanation...")
    valid = {
        'summary': 'Policy lacks MFA requirement for remote access.',
        'recommendation': 'Update policy to mandate TOTP or hardware key MFA.',
        'risk_level': 'high',
        'key_differences': [
            'Missing: MFA requirement',
            'Missing: Approved MFA methods list'
        ],
        'confidence': 'high',
        'regulatory_intent': 'Secure remote access',
        'policy_coverage': 'Covers passwords only'
    }

    try:
        exp = GapExplanation(**valid)
        print(f"    Valid | Risk score: {exp.get_risk_score()}")
    except Exception as e:
        print(f"    {e}")

    print("\n2. Auto-fix invalid data...")
    invalid = {
        'summary': 'Short',
        'recommendation': '',
        'risk_level': 'HIGH',
        'key_differences': 'Missing MFA',
        'confidence': 'very high'
    }

    validator = SchemaValidator()
    success, fixed, error = validator.validate_gap_explanation(invalid)
    print(f"   Fixed: {'PASS' if success else 'FAIL'}")
    if success:
        print(f"   Summary: {fixed.summary}")
        print(f"   Risk: {fixed.risk_level.value}")
        print(f"   Differences: {fixed.key_differences}")

    print("\n3. Testing unsupported explanation...")
    unsupported_data = {
        'summary': 'No policy exists for quantum encryption requirements.',
        'recommendation': 'Create Quantum-Safe Cryptography Policy.',
        'risk_level': 'critical',
        'key_differences': [
            'Missing: Quantum-resistant algorithm standards',
            'Missing: Migration timeline'
        ],
        'policy_type_needed': 'Quantum-Safe Cryptography Policy',
        'implementation_complexity': 'high',
        'urgency': 'immediate',
        'estimated_effort': '6-12 months',
        'confidence': 'high'
    }

    try:
        unsupp = UnsupportedExplanation(**unsupported_data)
        print(f"    Valid | Effort: {unsupp.estimated_effort}")
    except Exception as e:
        print(f"    {e}")

    print("\n4. Stats:")
    stats = validator.get_stats()
    print(f"   Total: {stats['total']}")
    print(f"   Success rate: {stats['success_rate']}%")
    print(f"   Fixed: {stats['fixed']}")

    print("\n" + "=" * 60)
    print("=" * 60)