import sys
sys.path.insert(0, 'src')

from scoring.gap_analyzer import GapAnalyzer
from explanation.generator import ExplanationGenerator

# Week 4
print('Running Week 4 analysis...')
analyzer = GapAnalyzer()
report = analyzer.analyze_document()  # remove limit entirely
print(f"Week 4: {report['summary']['total_regulations']} regulations")
print(f"Coverage: {report['summary']['coverage_percentage']}%")

# Week 5
print()
print('Checking Week 5 Groq connection...')
gen = ExplanationGenerator()
connected = gen.ensure_connection()

if connected:
    print()
    print('Full pipeline ready!')
    print('Next: gen.generate_batch(your_gap_classifications)')