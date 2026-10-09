"""Recursive Self-Improvement (RSI) Engine and Rollback Controller for L.O.R.A Outer Loop."""

import re
import statistics
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lora.outer_loop.genome import GenomeRegistry, PromptGenome, TemplateCompiler
from lora.outer_loop.optimizer import PromptOptimizerAgent, TraceAnalyzer
from lora.outer_loop.scorer import OuterLoopScorer, ScoreBreakdown


class IterationRecord(BaseModel):
    """Immutable audit record capturing a single optimization round."""

    model_config = ConfigDict(frozen=True)

    iteration: int
    genome_version: int
    genome_hash: str
    change_summary: str
    score_breakdown: ScoreBreakdown
    decision: str  # baseline, accepted, rolled_back, rejected
    answer: str


class OptimizationTrajectory(BaseModel):
    """Full trajectory history of prompt evolution and final verified answer."""

    model_config = ConfigDict(frozen=True)

    best_genome: PromptGenome
    best_answer: str
    best_score: float
    baseline_score: float
    iterations_run: int
    noise_epsilon: float
    history: tuple[IterationRecord, ...] = Field(default_factory=tuple)
    net_improvement: float = 0.0


class RSIEngine:
    """Orchestrates recursive hill-climbing search, noise variance calculation, and instant rollback."""

    def __init__(
        self,
        registry: GenomeRegistry,
        compiler: TemplateCompiler,
        optimizer: PromptOptimizerAgent,
        scorer: OuterLoopScorer,
        max_iters: int = 5,
        hard_cap: int = 8,
    ) -> None:
        self.registry = registry
        self.compiler = compiler
        self.optimizer = optimizer
        self.scorer = scorer
        self.analyzer = TraceAnalyzer()
        self.max_iters = min(max_iters, hard_cap)

    def _extract_citations(self, text: str) -> list[str]:
        return re.findall(r"\[\d+\]", text)

    def run(
        self,
        initial_genome: PromptGenome,
        user_request: str,
        evidence_cards: list[dict[str, Any]],
        synthesis_fn: Callable[[str], str],
        verification_fn: Callable[[str], dict[str, Any]],
        depth_mode: str = "standard",
    ) -> OptimizationTrajectory:
        """Execute bounded hill-climbing optimization over cached evidence cards."""
        # Save baseline to registry
        self.registry.save_genome(initial_genome, parent_hash=None)

        # -------------------------------------------------------------
        # Iteration 0: Baseline measurement & noise floor estimation
        # -------------------------------------------------------------
        compiled_base = self.compiler.compile(initial_genome, user_request, evidence_cards)

        # Run baseline 3 times to measure variance and noise margin epsilon
        baseline_scores = []
        baseline_answers = []
        baseline_breakdowns = []
        baseline_v_reports = []

        for _ in range(3):
            ans = synthesis_fn(compiled_base)
            v_report = verification_fn(ans)
            cits = self._extract_citations(ans)
            b_down = self.scorer.calculate_score(
                answer=ans,
                user_request=user_request,
                evidence_cards=evidence_cards,
                verification_report=v_report,
                citations_found=cits,
                depth_mode=depth_mode,
            )
            baseline_answers.append(ans)
            baseline_scores.append(b_down.composite_score)
            baseline_breakdowns.append(b_down)
            baseline_v_reports.append(v_report)

        mean_baseline_score = statistics.mean(baseline_scores)
        score_stdev = statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0.0
        epsilon = max(0.02, score_stdev)

        best_idx = baseline_scores.index(max(baseline_scores))
        best_genome = initial_genome
        best_score = mean_baseline_score
        best_answer = baseline_answers[best_idx]
        best_breakdown = baseline_breakdowns[best_idx]
        best_v_report = baseline_v_reports[best_idx]
        best_faithfulness = best_breakdown.faithfulness

        history: list[IterationRecord] = [
            IterationRecord(
                iteration=0,
                genome_version=initial_genome.version,
                genome_hash=initial_genome.content_hash,
                change_summary="Baseline prompt snapshot.",
                score_breakdown=best_breakdown,
                decision="baseline",
                answer=best_answer,
            )
        ]

        # -------------------------------------------------------------
        # Iterations 1..max_iters: Trace-driven hill climbing
        # -------------------------------------------------------------
        current_genome = best_genome

        for i in range(1, self.max_iters + 1):
            # 1. Analyze traces of current best
            diag = self.analyzer.analyze(
                verification_report=best_v_report,
                roundtable_report={"veto_reasons": []},
                score_breakdown=best_breakdown,
            )

            # 2. Propose mutation
            proposal = self.optimizer.propose_mutation(diag)

            try:
                candidate_genome = self.optimizer.apply_mutation(current_genome, proposal)
            except Exception as e:
                # Edit rejected by constraints -> halt loop safely
                history.append(
                    IterationRecord(
                        iteration=i,
                        genome_version=current_genome.version,
                        genome_hash=current_genome.content_hash,
                        change_summary=f"Mutation rejected: {e}",
                        score_breakdown=best_breakdown,
                        decision="rejected",
                        answer=best_answer,
                    )
                )
                break

            # 3. Compile and execute candidate
            compiled_cand = self.compiler.compile(candidate_genome, user_request, evidence_cards)
            cand_ans = synthesis_fn(compiled_cand)
            cand_v_report = verification_fn(cand_ans)
            cand_cits = self._extract_citations(cand_ans)

            cand_breakdown = self.scorer.calculate_score(
                answer=cand_ans,
                user_request=user_request,
                evidence_cards=evidence_cards,
                verification_report=cand_v_report,
                citations_found=cand_cits,
                depth_mode=depth_mode,
            )

            # 4. Confirmation re-run if score looks lower
            cand_score = cand_breakdown.composite_score
            if cand_score < best_score - epsilon:
                # Re-run once to rule out sampling fluke
                confirm_ans = synthesis_fn(compiled_cand)
                confirm_v = verification_fn(confirm_ans)
                confirm_cits = self._extract_citations(confirm_ans)
                confirm_b = self.scorer.calculate_score(
                    answer=confirm_ans,
                    user_request=user_request,
                    evidence_cards=evidence_cards,
                    verification_report=confirm_v,
                    citations_found=confirm_cits,
                    depth_mode=depth_mode,
                )
                cand_score = (cand_score + confirm_b.composite_score) / 2.0
                cand_ans = confirm_ans
                cand_breakdown = confirm_b
                cand_v_report = confirm_v

            # 5. Deadlock-free faithfulness gate check
            faithfulness_passed = self.scorer.check_faithfulness_gate(
                candidate_faithfulness=cand_breakdown.faithfulness,
                baseline_faithfulness=best_faithfulness,
            )

            # 6. Decision: Accept vs Rollback
            if faithfulness_passed and (cand_score > best_score + epsilon):
                # ACCEPT: Persist to registry
                self.registry.save_genome(candidate_genome, parent_hash=best_genome.content_hash)
                best_genome = candidate_genome
                best_score = cand_score
                best_answer = cand_ans
                best_breakdown = cand_breakdown
                best_v_report = cand_v_report
                best_faithfulness = cand_breakdown.faithfulness
                current_genome = candidate_genome

                history.append(
                    IterationRecord(
                        iteration=i,
                        genome_version=candidate_genome.version,
                        genome_hash=candidate_genome.content_hash,
                        change_summary=f"Accepted: {proposal.rule_to_add}",
                        score_breakdown=cand_breakdown,
                        decision="accepted",
                        answer=cand_ans,
                    )
                )
            else:
                # REJECT & ROLLBACK
                self.registry.rollback_to(best_genome.content_hash)
                history.append(
                    IterationRecord(
                        iteration=i,
                        genome_version=candidate_genome.version,
                        genome_hash=candidate_genome.content_hash,
                        change_summary=f"Regressed/stagnant (Score {cand_score:.2f} <= {best_score + epsilon:.2f}). Rolled back.",
                        score_breakdown=cand_breakdown,
                        decision="rolled_back",
                        answer=cand_ans,
                    )
                )
                # Per LORA spec: worse than previous -> rollback and stop loop
                break

        net_gain = round(best_score - mean_baseline_score, 4)
        return OptimizationTrajectory(
            best_genome=best_genome,
            best_answer=best_answer,
            best_score=round(best_score, 4),
            baseline_score=round(mean_baseline_score, 4),
            iterations_run=len(history) - 1,
            noise_epsilon=round(epsilon, 4),
            history=history,
            net_improvement=net_gain,
        )
