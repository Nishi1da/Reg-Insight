# src/ingestion/enforceable_filter.py

import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class FilterResult:
    keep: bool
    reason: str
    actor: Optional[str] = None

class EnforceableFilter:
    DEONTIC = r'\b(shall|must|required to)\b'
    ACTIONS = [
        'maintain', 'implement', 'report', 'ensure', 'verify',
        'monitor', 'establish', 'record', 'assess', 'review',
        'identify', 'authenticate', 'protect', 'retain', 'submit',
        'notify', 'preserve', 'conduct', 'appoint', 'designate'
    ]
    ACTION_PATTERN = r'\b(' + '|'.join(ACTIONS) + r')\b'

    DESCRIPTIVE = [
        r'^\s*(purpose|scope|background|introduction|definitions?)',
        r'\bapplies? to\b',
        r'\bprovides? for\b',
        r'\baims? to\b',
        r'\bfocuses? on\b',
        r'\bintended to\b',
    ]
    GOVERNANCE = [
        r'\bchairperson\b',
        r'\btribunal\b',
        r'\bcourt.*jurisdiction\b',
        r'\bterm of office\b',
        r'\bappointed by the (central|state) government\b',
        r'\bheadquarters.*shall be\b',
        r'\bno civil court\b',
        r'\bappellate\b',
        r'\bmember.*board.*shall\b',
    ]
    def analyze(self, text: str) -> FilterResult:
        text = text.strip()
        lower = text.lower()

    # Always reject governance/administrative clauses
        for p in self.GOVERNANCE:
            if re.search(p, lower):
                return FilterResult(False, "Governance clause")

    # Has deontic keyword — keep it, it's an obligation
        if re.search(self.DEONTIC, lower):
            return FilterResult(True, "Valid obligation")

    # No deontic — check if it's descriptive/scope text
        for p in self.DESCRIPTIVE:
            if re.search(p, lower):
                return FilterResult(False, "Descriptive text")

    # Longer text with no exclusion signal — keep
        if len(text.split()) >= 20:
            return FilterResult(True, "Kept — substantive text")
        
        return FilterResult(False, "Too short, no obligation signal")
    
def analyze_text(text: str) -> FilterResult:
    return _filter.analyze(text)