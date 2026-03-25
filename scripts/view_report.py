import json
import sys
from pathlib import Path

def view_report():
    report_path = Path('outputs/gap_analysis_report.json')
    
    if not report_path.exists():
        print("Report not found! Run: python src/scoring/gap_analyzer.py -a -l 5")
        return
    
    with open(report_path, 'r') as f:
        data = json.load(f)

    print('=' * 70)
    print('DETAILED GAP ANALYSIS BREAKDOWN')
    print('=' * 70)
    print(f"Report: {data['report_metadata']['report_id']}")
    print(f"Total: {data['summary']['total_regulations']} regulations")
    print()

    # GAP chunks
    print('❌ GAP CHUNKS (candidates found but scores too low)')
    print('-' * 70)
    gap_count = 0
    for item in data['regulation_analysis']:
        if item['classification'] == 'gap':
            gap_count += 1
            print(f"\n{gap_count}. ID: {item['regulation_chunk_id']}")
            print(f"   Text: {item['regulation_text'][:100]}...")
            print(f"   Confidence: {item['confidence']} ({item['confidence_level']})")
            if item['policy_matches']:
                best = item['policy_matches'][0]
                print(f"   Best match: {best.get('policy_chunk_id', 'unknown')}")
                print(f"   Scores: Bi={best.get('bi_encoder_score', 0):.3f}, Cross={best.get('cross_encoder_score', 0):.3f}, Final={best.get('final_score', 0):.3f}")
            print(f"   Why: {item['reasoning'][:80]}...")

    # UNMATCHED chunks
    print('\n\n🔍 UNMATCHED CHUNKS (no candidates found at all)')
    print('-' * 70)
    unmatched_count = 0
    for item in data['regulation_analysis']:
        if item['classification'] == 'unmatched':
            unmatched_count += 1
            print(f"\n{unmatched_count}. ID: {item['regulation_chunk_id']}")
            print(f"   Text: {item['regulation_text'][:100]}...")
            print(f"   Confidence: {item['confidence']}")
            print(f"   Why: {item['reasoning'][:80]}...")

    # Summary
    print('\n' + '=' * 70)
    print(f'SUMMARY: {gap_count} GAPs, {unmatched_count} UNMATCHED')
    print('=' * 70)

if __name__ == '__main__':
    view_report()