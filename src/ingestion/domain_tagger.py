# src/ingestion/domain_tagger.py

import re
from dataclasses import dataclass
from typing import Optional

DOMAIN_PATTERNS = {
    "aml": [
        r'\b(anti.money.laundering|aml|pmla|money laundering|suspicious transaction'
        r'|reporting entity|financial intelligence|know your customer|kyc'
        r'|beneficial owner|wire transfer|cash transaction|vda|virtual digital asset'
        r'|tipping.off|service provider.*vda|aml.cft|cft|cpf|cross.border'
        r'|designated director|principal officer|fiu|financial intelligence unit)\b'
    ],
    "cybersecurity": [
        r'\b(cyber|incident response|vulnerability|penetration test|soc|siem'
        r'|cert.in|malware|ransomware|threat|intrusion|patch management'
        r'|security operations|cyber incident|information security'
        r'|security incident|sir.*rbi|attack vectors?|ip address.*security'
        r'|forensic|log.*retention|network security|firewall|ddos|apt'
        r'|incident reporting|cyber drill|red team|blue team)\b'
    ],
    "data_protection": [
        r'\b(personal data|data principal|data fiduciary|dpdp|privacy|consent'
        r'|data breach|sensitive personal|data retention|data localisation'
        r'|right to erasure|purpose limitation|data processor|grievance officer'
        r'|data protection board|significant data fiduciary|cross.border.*data)\b'
    ],
    "it_governance": [
        r'\b(it governance|board oversight|it strategy|it policy|it committee'
        r'|technology risk|it audit|it infrastructure|disaster recovery'
        r'|business continuity|bcp|drp|rpo|rto|it operations|mis\b'
        r'|management information system|change management|it framework'
        r'|nbfc.*it|it.*nbfc|software|hardware|system.*availability'
        r'|data centre|cloud|outsourc|vendor.*it|third.party.*it)\b'
    ],
    "payment_security": [
        r'\b(payment|transaction|upi|neft|rtgs|card|pos|atm|tokenis'
        r'|two.factor|authentication|otp|fraud detection|chargeback'
        r'|merchant|acquiring|issuing|pci.dss|reconcili|digital payment'
        r'|fraud risk|payment.*security|secure.*payment|re\b.*payment'
        r'|acquirer|issuer|prepaid|wallet|emi|payment.*architecture)\b'
    ],
}

@dataclass
class DomainResult:
    primary: str
    confidence: float
    all_matches: dict

def tag_domain(text: str) -> DomainResult:
    """
    Assign a regulatory domain to a text chunk.
    Returns primary domain + confidence score.
    """
    lower = text.lower()
    scores = {}

    for domain, patterns in DOMAIN_PATTERNS.items():
        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, lower)
            count += len(matches)
        scores[domain] = count

    total = sum(scores.values())

    if total == 0:
        return DomainResult(
            primary="general",
            confidence=0.0,
            all_matches=scores
        )

    best_domain = max(scores, key=scores.get)
    confidence = scores[best_domain] / total

    return DomainResult(
        primary=best_domain,
        confidence=round(confidence, 3),
        all_matches=scores
    )