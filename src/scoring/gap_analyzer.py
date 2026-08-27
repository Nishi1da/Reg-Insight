"""Gap Analyzer - Main integration module"""

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

# Minimum bi-encoder score to proceed to cross-encoder
COSINE_GATE = 0.25


class GapAnalyzer:
    """
    End-to-end Gap Analysis System.

    Only analyzes chunks from the regulations collection (doc_type="regulation").
    Matches each regulation chunk against the policies collection using
    domain-filtered retrieval — same-domain policy chunks are searched first,
    falling back to all policies if too few domain matches exist.
    """

    def __init__(
        self,
        collection_name: str = "regulations",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_calibration: bool = False,
        calibration_data_path: Optional[str] = None
    ):
        logger.info("Initializing Gap Analyzer...")

        self.candidate_pipeline = CandidateGenerationPipeline(
            collection_name=collection_name,
            top_k=5,
            min_score=0.25
        )

        self.precision_pipeline = PrecisionScoringPipeline(
            cross_encoder_model=cross_encoder_model,
            bi_encoder_weight=0.3,
            cross_encoder_weight=0.7,
            use_cache=True
        )

        self.classifier           = GapClassifier()
        self.unsupported_detector = UnsupportedRequirementsDetector()
        self.report_generator     = GapReportGenerator(version="1.0.0")

        self.calibrator = None
        if enable_calibration and calibration_data_path:
            self.calibrator = ConfidenceCalibrator(method="isotonic")
            self._load_calibration(calibration_data_path)

        self.stats = {
            'regulations_processed': 0,
            'total_processing_time_ms': 0,
            'avg_time_per_regulation_ms': 0,
            'domain_filter_hits': 0,
            'domain_filter_fallbacks': 0,
            'cosine_gate_rejections': 0,
        }

        logger.info("Gap Analyzer initialized successfully")


    # ------------------------------------------------------------------ #
    #  CALIBRATION                                                         #
    # ------------------------------------------------------------------ #

    def _load_calibration(self, path: str):
        try:
            with open(path) as f:
                data = json.load(f)
            self.calibrator.fit(data['confidences'], data['accuracies'])
            logger.info("Calibration loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load calibration: {e}")

    # ------------------------------------------------------------------ #
    #  DOMAIN-FILTERED RETRIEVAL                                          #
    # ------------------------------------------------------------------ #

    def _get_candidates_with_domain_filter(
        self,
        regulation_chunk: Dict,
        reg_domain: str
    ) -> list:
        """
        Retrieve policy candidates filtered by regulatory domain.

        Strategy:
          1. Query policy collection filtered to same domain.
          2. If fewer than 2 results, fall back to unfiltered policy query.
          3. Apply cosine gate — drop candidates with bi-encoder score < COSINE_GATE.
        """
        generator = self.candidate_pipeline.generator

        # ── Step 1: try domain-filtered query ────────────────────────────
        raw_candidates = []

        if reg_domain and reg_domain != "general":
            try:
                raw_candidates = generator.get_candidates(
                    regulation_chunk,
                    top_k=5,
                    min_score=0.2,
                    where_filter={
                        "$and": [
                            {"doc_type": {"$eq": "policy"}},
                            {"domain":   {"$eq": reg_domain}}
                        ]
                    }
                )

                if len(raw_candidates) >= 2:
                    self.stats['domain_filter_hits'] += 1
                    logger.debug(
                        f"Domain filter '{reg_domain}': {len(raw_candidates)} candidates"
                    )
                else:
                    raw_candidates = []

            except Exception as e:
                logger.debug(f"Domain filter query failed ({e}), falling back")
                raw_candidates = []

        # ── Step 2: fallback — any policy chunk ──────────────────────────
        if not raw_candidates:
            self.stats['domain_filter_fallbacks'] += 1
            try:
                raw_candidates = generator.get_candidates(
                    regulation_chunk,
                    top_k=5,
                    min_score=0.2,
                    where_filter={"doc_type": {"$eq": "policy"}}
                )
            except Exception:
                raw_candidates = generator.get_candidates(
                    regulation_chunk,
                    top_k=5,
                    min_score=0.2
                )

        # ── Step 3: cosine gate ───────────────────────────────────────────
        before_gate = len(raw_candidates)
        raw_candidates = self._apply_cosine_gate(raw_candidates)
        rejected = before_gate - len(raw_candidates)

        if rejected > 0:
            self.stats['cosine_gate_rejections'] += rejected
            logger.debug(f"Cosine gate rejected {rejected}/{before_gate} candidates")

        return raw_candidates

    def _apply_cosine_gate(self, candidates: list) -> list:
        """Drop candidates whose bi-encoder similarity is below COSINE_GATE."""
        filtered = []
        for c in candidates:
            if hasattr(c, 'bi_encoder_score'):
                score = c.bi_encoder_score
            elif hasattr(c, 'similarity_score'):
                score = c.similarity_score
            elif hasattr(c, 'score'):
                score = c.score
            elif isinstance(c, dict):
                score = c.get('bi_encoder_score',
                              c.get('similarity_score',
                              c.get('score', 0.0)))
            else:
                score = 0.0

            if score >= COSINE_GATE:
                filtered.append(c)

        return filtered

    # ------------------------------------------------------------------ #
    #  SINGLE CHUNK ANALYSIS                                               #
    # ------------------------------------------------------------------ #

    def analyze_regulation_chunk(
        self,
        regulation_chunk: Dict
    ) -> Tuple[GapClassification, Optional[Dict]]:
        """
        Analyze a single regulation chunk against all policy chunks.

        Pipeline:
          1. Domain-filtered candidate retrieval
          2. Rank + deduplicate
          3. Cross-encoder precision scoring
          4. Gap classification
          5. (Optional) confidence calibration
          6. Unsupported requirement detection
        """
        start_time = time.time()

        reg_domain = regulation_chunk.get('metadata', {}).get('domain', 'general')

        # ── Step 1: retrieve candidates ───────────────────────────────────
        raw_candidates = self._get_candidates_with_domain_filter(
            regulation_chunk, reg_domain
        )

        # ── Step 2: rank + deduplicate ────────────────────────────────────
        ranked = self.candidate_pipeline.ranker.rank_candidates(raw_candidates)
        unique = self.candidate_pipeline.ranker.deduplicate_candidates(ranked)

        # ── Step 3: cross-encoder scoring ─────────────────────────────────
        if unique:
            scored = self.precision_pipeline.score_candidates(unique[:3])
        else:
            scored = []

        # ── Step 4: classify ──────────────────────────────────────────────
        classification = self.classifier.classify(
            regulation_chunk['chunk_id'],
            regulation_chunk['content'],
            regulation_chunk.get('metadata', {}),
            scored
        )

        # ── Step 5: optional calibration ──────────────────────────────────
        if self.calibrator:
            calibration = self.calibrator.calibrate(classification.confidence)
            classification.confidence = calibration.calibrated_confidence

        # ── Step 6: unsupported detection ─────────────────────────────────
        unsupported = self.unsupported_detector.detect_unsupported(
            regulation_chunk,
            scored,
            classification.classification
        )

        elapsed = (time.time() - start_time) * 1000
        self.stats['regulations_processed'] += 1
        self.stats['total_processing_time_ms'] += elapsed
        self.stats['avg_time_per_regulation_ms'] = (
            self.stats['total_processing_time_ms'] / self.stats['regulations_processed']
        )

        return classification, unsupported.to_dict() if unsupported else None

    # ------------------------------------------------------------------ #
    #  FULL DOCUMENT ANALYSIS                                              #
    # ------------------------------------------------------------------ #

    def analyze_document(
        self,
        limit: Optional[int] = None,
        progress_callback=None,
        source_filter: Optional[str] = None
    ) -> Dict:
        """
        Analyze all regulation chunks from ChromaDB.

        CRITICAL: filters collection to doc_type="regulation" only so policy
        chunks are never analyzed as if they were regulations.

        source_filter: if provided, further filters to chunks from that source
        filename only — used when analyzing a freshly uploaded PDF.
        """
        logger.info(f"Starting document analysis (limit={limit})...")
        start_time = time.time()

        collection = self.candidate_pipeline.generator.chroma.get_collection(
            self.candidate_pipeline.collection_name
        )

        # ── Fetch regulation chunks only ──────────────────────────────────
        try:
            where_filter = {"doc_type": "regulation"}
            if source_filter:
                where_filter = {
                    "$and": [
                        {"doc_type": "regulation"},
                        {"source": source_filter}
                    ]
                }
            all_data = collection.get(
                limit=limit,
                where=where_filter
            )
            logger.info("Filtered to doc_type='regulation' chunks only")
        except Exception:
            logger.warning(
                "doc_type filter failed — fetching all chunks (re-ingest recommended)"
            )
            all_data = collection.get(limit=limit)

        total = len(all_data['ids'])
        logger.info(f"Found {total} regulation chunks to analyze")

        classifications  = []
        unsupported_list = []

        for i, (chunk_id, text, metadata) in enumerate(zip(
            all_data['ids'],
            all_data['documents'],
            all_data['metadatas']
        )):
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{total} ({i/total*100:.1f}%)")
                if progress_callback:
                    progress_callback(i / total * 100)

            reg_chunk = {
                'chunk_id': chunk_id,
                'content':  text,
                'metadata': metadata or {}
            }

            classification, unsupported = self.analyze_regulation_chunk(reg_chunk)
            classifications.append(classification)

            if unsupported:
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

        report = self.report_generator.create_batch_report(
            classifications=classifications,
            unsupported=unsupported_list,
            metadata={
                'processing_time_seconds': time.time() - start_time,
                'limit': limit,
                'model': self.precision_pipeline.cross_encoder.model_name
            }
        )

        logger.info(
            f"Domain filter stats — hits: {self.stats['domain_filter_hits']}, "
            f"fallbacks: {self.stats['domain_filter_fallbacks']}, "
            f"cosine rejections: {self.stats['cosine_gate_rejections']}"
        )
        logger.info(f"Analysis complete: {total} chunks in {time.time()-start_time:.1f}s")
        return report

    # ------------------------------------------------------------------ #
    #  SINGLE TEXT ANALYSIS                                                #
    # ------------------------------------------------------------------ #

    def analyze_single(
        self,
        regulation_text: str,
        regulation_id: str = "manual_input"
    ) -> Dict:
        """Analyze a single regulation text string."""
        reg_chunk = {
            'chunk_id': regulation_id,
            'content':  regulation_text,
            'metadata': {
                'source':     'manual',
                'doc_type':   'regulation',
                'domain':     'general',
                'input_time': datetime.now().isoformat()
            }
        }
        classification, unsupported = self.analyze_regulation_chunk(reg_chunk)
        return {
            'classification':      classification.to_dict(),
            'is_unsupported':      unsupported is not None,
            'unsupported_details': unsupported,
            'processing_stats':    {
                'avg_time_ms': self.stats['avg_time_per_regulation_ms']
            }
        }

    # ------------------------------------------------------------------ #
    #  BENCHMARK                                                           #
    # ------------------------------------------------------------------ #

    def benchmark(self, n_chunks: int = 100) -> Dict:
        collection = self.candidate_pipeline.generator.chroma.get_collection(
            self.candidate_pipeline.collection_name
        )
        try:
            sample = collection.get(
                limit=n_chunks,
                where={"doc_type": "regulation"}
            )
        except Exception:
            sample = collection.get(limit=n_chunks)

        times = []
        for chunk_id, text, metadata in zip(
            sample['ids'], sample['documents'], sample['metadatas']
        ):
            start = time.time()
            reg_chunk = {
                'chunk_id': chunk_id,
                'content':  text,
                'metadata': metadata or {}
            }
            self.analyze_regulation_chunk(reg_chunk)
            times.append((time.time() - start) * 1000)

        avg_time = sum(times) / len(times) if times else 0
        return {
            'chunks_processed':      len(times),
            'avg_time_ms':           round(avg_time, 2),
            'max_time_ms':           round(max(times), 2) if times else 0,
            'min_time_ms':           round(min(times), 2) if times else 0,
            'target_ms':             100,
            'meets_target':          avg_time < 100,
            'throughput_per_second': round(1000 / avg_time, 2) if avg_time > 0 else 0
        }

    # ------------------------------------------------------------------ #
    #  STATS + VALIDATION                                                  #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'classifier_stats':  self.classifier.get_stats(),
            'unsupported_stats': self.unsupported_detector.get_stats()
        }

    def export_validation_sample(
        self,
        output_path: str = "outputs/validation_sample.json"
    ):
        report   = self.analyze_document(limit=20)
        sample   = []
        analysis = report.get('regulation_analysis', [])

        for cls_label in ['unmatched', 'gap', 'partial', 'aligned']:
            matches = [r for r in analysis if r['classification'] == cls_label]
            if matches:
                sample.append(matches[0])
            if len(sample) >= 5:
                break

        for item in sample:
            item['manual_review'] = {
                'your_classification':         '',
                'confidence_in_your_judgment': '',
                'system_was_wrong':            False,
                'why_wrong':                   '',
                'notes':                       ''
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(sample, f, indent=2)

        logger.info(f"Exported {len(sample)} validation cases to {output_path}")
        return sample


# ---------------------------------------------------------------------- #
#  CLI                                                                     #
# ---------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description='Regulatory Gap Analyzer')
    parser.add_argument('--analyze',   '-a', action='store_true', help='Run full analysis')
    parser.add_argument('--text',      '-t', type=str,            help='Analyze single text')
    parser.add_argument('--limit',     '-l', type=int,            help='Limit chunks (testing)')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Performance benchmark')
    parser.add_argument('--output',    '-o', type=str,
                        default='outputs/gap_analysis_report.json')
    parser.add_argument('--summary',   '-s', action='store_true', help='Print summary')
    parser.add_argument('--validate',        action='store_true', help='Export validation sample')
    args = parser.parse_args()

    analyzer = GapAnalyzer()

    if args.text:
        result = analyzer.analyze_single(args.text)
        print(json.dumps(result, indent=2))

    elif args.benchmark:
        results = analyzer.benchmark(n_chunks=100)
        print(json.dumps(results, indent=2))

    elif args.validate:
        analyzer.export_validation_sample()

    elif args.analyze:
        print("Running full document analysis...")
        report  = analyzer.analyze_document(limit=args.limit)
        success = analyzer.report_generator.export_report(report, args.output)
        if success:
            print(f"\nReport saved: {args.output}")
            if args.summary:
                print(analyzer.report_generator.generate_executive_summary(report))
        else:
            print("\nFailed to export report")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()