"""Retrieval Logger - Audit logging and metadata tracking"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict
from datetime import datetime
import json
import logging

from candidate_generation.candidate_generator import CandidatePair

# Module-level logger - use different name to avoid conflict
_module_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RetrievalLogger:
    def __init__(self, log_dir: str = "data/retrieval_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"retrieval_{self.session_id}.jsonl"
        
        self.stats = {
            'total_queries': 0,
            'total_candidates': 0,
            'avg_candidates_per_query': 0.0
        }
    
    def log_retrieval(
        self,
        regulation_chunk_id: str,
        candidates: List[CandidatePair],
        retrieval_time_ms: float,
        parameters: Dict
    ):
        """Log a single retrieval operation"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'regulation_chunk_id': regulation_chunk_id,
            'num_candidates': len(candidates),
            'candidate_ids': [c.policy_chunk_id for c in candidates],
            'scores': [c.bi_encoder_score for c in candidates],
            'retrieval_time_ms': retrieval_time_ms,
            'parameters': parameters
        }
        
        # Append to JSONL
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Update stats
        self.stats['total_queries'] += 1
        self.stats['total_candidates'] += len(candidates)
        self.stats['avg_candidates_per_query'] = (
            self.stats['total_candidates'] / self.stats['total_queries']
        )
    
    def get_statistics(self) -> Dict:
        """Get retrieval statistics"""
        return {
            'session_id': self.session_id,
            'log_file': str(self.log_file),
            **self.stats
        }
    
    def export_audit_trail(self, output_path: str):
        """Export complete audit trail"""
        # Read all entries
        entries = []
        if self.log_file.exists():
            with open(self.log_file) as f:
                for line in f:
                    entries.append(json.loads(line.strip()))
        
        # Export with summary
        audit = {
            'export_timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'summary': self.stats,
            'entries': entries
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(audit, f, indent=2)
        
        # Use module-level logger (renamed to avoid conflict)
        _module_logger.info(f"Exported audit trail to {output_path}")


# Simple dashboard (text-based for now)
def print_dashboard(log_file: Path):
    """Print simple statistics from log file"""
    if not log_file.exists():
        print("No log file found")
        return
    
    entries = []
    with open(log_file) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    
    print("=" * 60)
    print("RETRIEVAL DASHBOARD")
    print("=" * 60)
    print(f"Total queries: {len(entries)}")
    
    if entries:
        avg_candidates = sum(e['num_candidates'] for e in entries) / len(entries)
        avg_time = sum(e['retrieval_time_ms'] for e in entries) / len(entries)
        
        print(f"Avg candidates/query: {avg_candidates:.2f}")
        print(f"Avg retrieval time: {avg_time:.1f}ms")
        
        # Score distribution
        all_scores = [s for e in entries for s in e['scores']]
        if all_scores:
            print(f"\nScore distribution:")
            print(f"  Mean: {sum(all_scores)/len(all_scores):.3f}")
            print(f"  Max: {max(all_scores):.3f}")
            print(f"  Min: {min(all_scores):.3f}")


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 19: Retrieval Logger Test")
    print("=" * 60)
    
    # Use different variable name to avoid shadowing module logger
    retrieval_logger = RetrievalLogger()
    
    # Simulate some retrievals
    print("\n1. Logging sample retrievals...")
    
    for i in range(5):
        candidates = [
            CandidatePair(f"reg_{i}", "text", {}, f"pol_{i}_{j}", "text", {}, 0.8 - j*0.1, j+1)
            for j in range(3)
        ]
        
        retrieval_logger.log_retrieval(
            regulation_chunk_id=f"reg_{i}",
            candidates=candidates,
            retrieval_time_ms=50 + i*10,
            parameters={'top_k': 3, 'min_score': 0.3}
        )
    
    # Check stats
    print("\n2. Statistics:")
    stats = retrieval_logger.get_statistics()
    print(f"   {stats}")
    
    # Export
    print("\n3. Exporting audit trail...")
    retrieval_logger.export_audit_trail("outputs/day19_audit.json")
    
    # Dashboard
    print("\n4. Dashboard:")
    print_dashboard(retrieval_logger.log_file)
    
    print("\n" + "=" * 60)
    print("=" * 60)