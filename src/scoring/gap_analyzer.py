"""Gap Analyzer - Main integration module for Week 4"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional, Tuple
import logging
import time
import argparse
from datetime import datetime
import json

from candidate_generation.candidate_generator import CandidateGenerator
from candidate_generation.pipeline import CandidateGenerationPipeline
from scoring.cross_encoder import CrossEncoderScorer
from scoring.precision_pipeline import PrecisionScoringPipeline
from scoring.gap_classifier import GapClassifier, GapClassification, GapClass
from scoring.unsupported_detector import UnsupportedRequirementsDetector
from scoring.gap_report import GapReportGenerator
from scoring.confidence_calibrator import ConfidenceCalibrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GapAnalyzer:
    """
    End-to-end Gap Analysis System
    
    Complete pipeline:
    1. Candidate Generation (Week 3)
    2. Cross-Encoder Scoring (Day 22)
    3. Precision Scoring (Day 23)
    4. Gap Classification (Day 24)
    5. Unsupported Detection (Day 25)
    6. Report Generation (Day 26)
    7. Confidence Calibration (Day 27)
    """
    
    def __init__(
        self,
        collection_name: str = "regulations",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_calibration: bool = False,
        calibration_data_path: Optional[str] = None
    ):
        logger.info("Initializing Gap Analyzer...")
        
        # Initialize components
        self.candidate_pipeline = CandidateGenerationPipeline(
            collection_name=collection_name,
            top_k=5,
            min_score=0.2
        )
        
        self.precision_pipeline = PrecisionScoringPipeline(
            cross_encoder_model=cross_encoder_model,
            bi_encoder_weight=0.3,
            cross_encoder_weight=0.7,
            use_cache=True
        )
        
        self.classifier = GapClassifier()
        self.unsupported_detector = UnsupportedRequirementsDetector()
        self.report_generator = GapReportGenerator(version="1.0.0")
        
        # Optional calibration
        self.calibrator = None
        if enable_calibration and calibration_data_path:
            self.calibrator = ConfidenceCalibrator(method="isotonic")
            self._load_calibration(calibration_data_path)
        
        self.stats = {
            'regulations_processed': 0,
            'total_processing_time_ms': 0,
            'avg_time_per_regulation_ms': 0
        }
        
        logger.info("Gap Analyzer initialized successfully")
    
    def _load_calibration(self, path: str):
        """Load calibration data"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.calibrator.fit(data['confidences'], data['accuracies'])
            logger.info("Calibration loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load calibration: {e}")
    
    def analyze_regulation_chunk(
        self,
        regulation_chunk: Dict
    ) -> Tuple[GapClassification, Optional[Dict]]:
        """
        Analyze single regulation chunk
        
        Args:
            regulation_chunk: Dict with chunk_id, content, metadata
        
        Returns:
            (GapClassification, unsupported_info)
        """
        start_time = time.time()
        
        # Step 1: Generate candidates
        raw_candidates = self.candidate_pipeline.generator.get_candidates(
            regulation_chunk,
            top_k=5,
            min_score=0.2
        )
        
        # Step 2: Rank candidates (from Week 3)
        ranked = self.candidate_pipeline.ranker.rank_candidates(raw_candidates)
        unique = self.candidate_pipeline.ranker.deduplicate_candidates(ranked)
        
        # Step 3: Precision scoring with cross-encoder
        if unique:
            scored = self.precision_pipeline.score_candidates(unique[:3])
        else:
            scored = []
        
        # Step 4: Gap classification
        classification = self.classifier.classify(
            regulation_chunk['chunk_id'],
            regulation_chunk['content'],
            regulation_chunk.get('metadata', {}),
            scored
        )
        
        # Step 5: Calibrate confidence if enabled
        if self.calibrator:
            calibration = self.calibrator.calibrate(classification.confidence)
            # Update classification with calibrated confidence
            classification.confidence = calibration.calibrated_confidence
        
        # Step 6: Detect unsupported
        unsupported = self.unsupported_detector.detect_unsupported(
            regulation_chunk,
            scored,
            classification.classification
        )
        
        elapsed = (time.time() - start_time) * 1000
        
        # Update stats
        self.stats['regulations_processed'] += 1
        self.stats['total_processing_time_ms'] += elapsed
        self.stats['avg_time_per_regulation_ms'] = (
            self.stats['total_processing_time_ms'] / self.stats['regulations_processed']
        )
        
        return classification, unsupported.to_dict() if unsupported else None
    
    def analyze_document(
        self,
        limit: Optional[int] = None,
        progress_callback=None
    ) -> Dict:
        """
        Analyze full document (all regulation chunks)
        
        Args:
            limit: Optional limit for testing
            progress_callback: Optional callback function(progress_pct)
        
        Returns:
            Complete gap report
        """
        logger.info(f"Starting document analysis (limit={limit})...")
        start_time = time.time()
        
        # Get all regulation chunks
        collection = self.candidate_pipeline.generator.chroma.get_collection(
            self.candidate_pipeline.collection_name
        )
        all_data = collection.get(limit=limit)
        
        total = len(all_data['ids'])
        logger.info(f"Found {total} regulation chunks to analyze")
        
        classifications = []
        unsupported_list = []
        
        for i, (chunk_id, text, metadata) in enumerate(zip(
            all_data['ids'], all_data['documents'], all_data['metadatas']
        )):
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{total} ({i/total*100:.1f}%)")
                if progress_callback:
                    progress_callback(i / total * 100)
            
            reg_chunk = {
                'chunk_id': chunk_id,
                'content': text,
                'metadata': metadata
            }
            
            classification, unsupported = self.analyze_regulation_chunk(reg_chunk)
            classifications.append(classification)
            
            if unsupported:
                # Convert dict back to object for report
                from scoring.unsupported_detector import UnsupportedRequirement
                req = UnsupportedRequirement(
                     regulation_chunk_id=unsupported['regulation_chunk_id'],
                    regulation_text=unsupported['regulation_text'],
                    regulation_source=unsupported['source'],
                    regulation_page=unsupported['page'],
                    severity=unsupported['severity'],
                    severity_score=unsupported['severity_score'],
                    detection_method=unsupported['detection_method'],
                    confidence=unsupported['confidence'],
                    section_header=None,
                    requirement_type=unsupported['requirement_type'],
                    recommended_priority=unsupported['recommended_priority'],
                    estimated_effort=unsupported['estimated_effort']
                    )
                unsupported_list.append(req)
        
        # Generate report
        report = self.report_generator.create_batch_report(
            classifications=classifications,
            unsupported=unsupported_list,
            metadata={
                'processing_time_seconds': time.time() - start_time,
                'limit': limit,
                'model': self.precision_pipeline.cross_encoder.model_name
            }
        )
        
        logger.info(f"Analysis complete: {total} chunks in {time.time()-start_time:.1f}s")
        
        return report
    
    def analyze_single(
        self,
        regulation_text: str,
        regulation_id: str = "manual_input"
    ) -> Dict:
        """
        Analyze single regulation text (for CLI/testing)
        
        Args:
            regulation_text: Regulation requirement text
            regulation_id: ID for tracking
        
        Returns:
            Analysis result dict
        """
        reg_chunk = {
            'chunk_id': regulation_id,
            'content': regulation_text,
            'metadata': {'source': 'manual', 'input_time': datetime.now().isoformat()}
        }
        
        classification, unsupported = self.analyze_regulation_chunk(reg_chunk)
        
        return {
            'classification': classification.to_dict(),
            'is_unsupported': unsupported is not None,
            'unsupported_details': unsupported if unsupported else None,
            'processing_stats': {
                'avg_time_ms': self.stats['avg_time_per_regulation_ms']
            }
        }
    
    def benchmark(self, n_chunks: int = 100) -> Dict:
        """
        Performance benchmark
        
        Args:
            n_chunks: Number of chunks to process
        
        Returns:
            Benchmark results
        """
        logger.info(f"Running benchmark with {n_chunks} chunks...")
        
        collection = self.candidate_pipeline.generator.chroma.get_collection(
            self.candidate_pipeline.collection_name
        )
        sample = collection.get(limit=n_chunks)
        
        times = []
        
        for chunk_id, text, metadata in zip(
            sample['ids'], sample['documents'], sample['metadatas']
        ):
            start = time.time()
            
            reg_chunk = {
                'chunk_id': chunk_id,
                'content': text,
                'metadata': metadata
            }
            
            self.analyze_regulation_chunk(reg_chunk)
            times.append((time.time() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        # Target: < 100ms per regulation (Week 3 requirement)
        meets_target = avg_time < 100
        
        return {
            'chunks_processed': len(times),
            'avg_time_ms': round(avg_time, 2),
            'max_time_ms': round(max_time, 2),
            'min_time_ms': round(min_time, 2),
            'target_ms': 100,
            'meets_target': meets_target,
            'throughput_per_second': round(1000 / avg_time, 2) if avg_time > 0 else 0
        }
    
    def get_stats(self) -> Dict:
        """Get analyzer statistics"""
        return {
            **self.stats,
            'classifier_stats': self.classifier.get_stats(),
            'unsupported_stats': self.unsupported_detector.get_stats()
        }


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Regulatory Gap Analyzer - Week 4'
    )
    parser.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Run full document analysis'
    )
    parser.add_argument(
        '--text', '-t',
        type=str,
        help='Analyze single regulation text'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of chunks (for testing)'
    )
    parser.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Run performance benchmark'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/gap_analysis_report.json',
        help='Output path for report'
    )
    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='Print executive summary'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = GapAnalyzer()
    
    if args.text:
        # Single text analysis
        print(f"Analyzing: {args.text[:80]}...")
        result = analyzer.analyze_single(args.text)
        print(json.dumps(result, indent=2))
    
    elif args.benchmark:
        # Benchmark
        print("Running performance benchmark...")
        results = analyzer.benchmark(n_chunks=100)
        print(json.dumps(results, indent=2))
        
        if results['meets_target']:
            print(f"\n Meets target: {results['avg_time_ms']}ms < 100ms")
        else:
            print(f"\n Below target: {results['avg_time_ms']}ms > 100ms")
    
    elif args.analyze:
        # Full analysis
        print("Running full document analysis...")
        report = analyzer.analyze_document(limit=args.limit)
        
        # Export
        success = analyzer.report_generator.export_report(report, args.output)
        if success:
            print(f"\n Report saved: {args.output}")
            
            if args.summary:
                print(analyzer.report_generator.generate_executive_summary(report))
        else:
            print("\n Failed to export report")
    
    else:
        parser.print_help()


# Test
if __name__ == "__main__":
    main()