"""V8: End-to-end verification pipeline, composite scoring, and repair loop."""

import asyncio
import logging
from typing import Any

from lora.grounding.cards import EvidenceCard
from lora.grounding.copycheck import CopyChecker
from lora.verify.citation import CitationExistenceChecker
from lora.verify.claims import ClaimDecomposer
from lora.verify.corroborate import CorroborationChecker
from lora.verify.llm_check import LlmFactChecker
from lora.verify.nli import NliGroundednessChecker
from lora.verify.numbers import NumberEntityChecker
from lora.verify.report import (
    ClaimVerificationDetail,
    ConfidenceLevel,
    VerificationReport,
    VerificationVerdict,
)

logger = logging.getLogger(__name__)

COMPOSITE_GATE_THRESHOLD = 0.85
PER_CLAIM_NLI_THRESHOLD = 0.80


class VerificationPipeline:
    """Orchestrates V1-V8 verification stages, composite scoring, and truthfulness repair."""

    def __init__(
        self,
        ollama_client: Any = None,
        composite_threshold: float = COMPOSITE_GATE_THRESHOLD,
        nli_threshold: float = PER_CLAIM_NLI_THRESHOLD,
    ) -> None:
        self.composite_threshold = composite_threshold
        self.nli_threshold = nli_threshold
        self.v1_citation = CitationExistenceChecker()
        self.v2_decomposer = ClaimDecomposer()
        self.v3_nli = NliGroundednessChecker(threshold=self.nli_threshold)
        self.v4_llm = LlmFactChecker(ollama_client=ollama_client)
        self.v5_corroborate = CorroborationChecker()
        self.v6_numbers = NumberEntityChecker()
        self.v7_copy = CopyChecker(n_gram=8, max_overlap_ratio=0.35)
        self.last_v4_stats: dict[str, int] = {"attempted": 0, "succeeded": 0, "timed_out": 0}

    async def verify(
        self,
        answer_text: str,
        cards: list[EvidenceCard],
    ) -> VerificationReport:
        """Run full V1-V8 verification pipeline over an answer against source evidence."""
        self.last_v4_stats = {"attempted": 0, "succeeded": 0, "timed_out": 0}
        if not cards or not answer_text or not answer_text.strip():
            return VerificationReport(
                verdict=VerificationVerdict.ABSTAIN,
                composite_faithfulness_score=0.0,
                confidence=ConfidenceLevel.LOW,
                claim_details=[],
                repaired_answer="I could not find reliable evidence to support this answer.",
                unverified_claims=["No source evidence available"],
                requires_abstention=True,
            )

        card_map = {c.card_id: c for c in cards}
        combined_premise = "\n".join(c.exact_quote for c in cards)

        # V1: Citation existence check
        v1_res = self.v1_citation.check_citations(text=answer_text, cards=cards)

        # V2: Claim decomposition
        atomic_claims = self.v2_decomposer.decompose(text=answer_text)
        if not atomic_claims:
            return VerificationReport(
                verdict=VerificationVerdict.ABSTAIN,
                composite_faithfulness_score=0.0,
                confidence=ConfidenceLevel.LOW,
                claim_details=[],
                repaired_answer="Answer contains no verifiable atomic statements.",
                requires_abstention=True,
            )

        claim_details: list[ClaimVerificationDetail] = []
        claim_premises: list[str] = []
        supported_count = 0
        unverified_list: list[str] = []
        llm_probes = 0
        llm_succeeded = 0
        llm_timed_out = 0
        max_llm_probes = 2

        for claim in atomic_claims:
            cited_cards = [card_map[cid] for cid in claim.cited_sources if cid in card_map]
            premise_text = (
                "\n".join(c.exact_quote for c in cited_cards)
                if cited_cards
                else combined_premise
            )
            claim_premises.append(premise_text)

            # V6: Number check (check cited cards first, then full evidence pool if multi-source synthesis)
            num_res = self.v6_numbers.verify_numbers(premise=premise_text, claim=claim.claim_text)
            if not num_res.is_valid and cited_cards and len(cards) > len(cited_cards):
                num_res_all = self.v6_numbers.verify_numbers(premise=combined_premise, claim=claim.claim_text)
                if num_res_all.is_valid:
                    num_res = num_res_all

            # V3: NLI groundedness check
            nli_res = self.v3_nli.check_entailment(
                premise=premise_text, hypothesis=claim.claim_text
            )
            if not nli_res.is_entailed and cited_cards and len(cards) > len(cited_cards):
                nli_all = self.v3_nli.check_entailment(
                    premise=combined_premise, hypothesis=claim.claim_text
                )
                if nli_all.score > nli_res.score:
                    nli_res = nli_all

            is_supported = nli_res.is_entailed and num_res.is_valid
            reason = ""

            if not num_res.is_valid:
                reason = f"Mismatched numbers: {', '.join(num_res.mismatched_numbers)}"
            elif not nli_res.is_entailed:
                # V4: Escalate to LLM fact checker (bounded probes + 20s timeout so verification never hangs)
                if self.v4_llm.ollama_client is not None and llm_probes < max_llm_probes:
                    llm_probes += 1
                    try:
                        v4_res = await asyncio.wait_for(
                            self.v4_llm.verify_claim(premise=premise_text[:2400], claim=claim.claim_text),
                            timeout=20.0,
                        )
                        if "Error during LLM verification" in (v4_res.reason or ""):
                            llm_timed_out += 1
                            v4_res = await LlmFactChecker(ollama_client=None).verify_claim(
                                premise=premise_text, claim=claim.claim_text
                            )
                        else:
                            llm_succeeded += 1
                    except Exception as exc:
                        llm_timed_out += 1
                        logger.warning("V4 LLM fact-check probe timed out or failed: %s", exc)
                        v4_res = await LlmFactChecker(ollama_client=None).verify_claim(
                            premise=premise_text, claim=claim.claim_text
                        )
                else:
                    v4_res = await LlmFactChecker(ollama_client=None).verify_claim(
                        premise=premise_text, claim=claim.claim_text
                    )
                if v4_res.is_supported and num_res.is_valid:
                    is_supported = True
                    reason = "Verified via LLM escalation"
                else:
                    reason = v4_res.reason or "Low NLI entailment score"

            # V5: Corroboration check
            corroboration = self.v5_corroborate.check_corroboration(
                cards=cited_cards if cited_cards else cards
            )

            if is_supported:
                supported_count += 1
            else:
                unverified_list.append(claim.claim_text)

            claim_details.append(
                ClaimVerificationDetail(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    nli_score=nli_res.score,
                    is_supported=is_supported,
                    number_check_passed=num_res.is_valid,
                    corroboration=corroboration,
                    unsupported_reason=reason,
                )
            )

        # When an LLM verifier client is provided and all claims passed heuristic NLI (0 escalations),
        # run 1 real LLM audit probe on the lowest-NLI claim so V4 is genuinely exercised.
        if self.v4_llm.ollama_client is not None and llm_probes == 0 and claim_details:
            weakest_idx = min(range(len(claim_details)), key=lambda idx: claim_details[idx].nli_score)
            weakest_cd = claim_details[weakest_idx]
            weakest_premise = claim_premises[weakest_idx]
            llm_probes += 1
            try:
                v4_spot = await asyncio.wait_for(
                    self.v4_llm.verify_claim(premise=weakest_premise[:2400], claim=weakest_cd.claim_text),
                    timeout=20.0,
                )
                if "Error during LLM verification" in (v4_spot.reason or ""):
                    llm_timed_out += 1
                else:
                    llm_succeeded += 1
                    # If the LLM audit explicitly rejects a borderline heuristic NLI match (< 0.86), flag it
                    if not v4_spot.is_supported and weakest_cd.nli_score < 0.86 and weakest_cd.is_supported:
                        claim_details[weakest_idx] = ClaimVerificationDetail(
                            claim_id=weakest_cd.claim_id,
                            claim_text=weakest_cd.claim_text,
                            nli_score=weakest_cd.nli_score,
                            is_supported=False,
                            number_check_passed=weakest_cd.number_check_passed,
                            corroboration=weakest_cd.corroboration,
                            unsupported_reason=v4_spot.reason or "Flagged by V4 LLM spot-check audit",
                        )
                        supported_count = max(0, supported_count - 1)
                        unverified_list.append(weakest_cd.claim_text)
            except Exception as exc:
                llm_timed_out += 1
                logger.warning("V4 LLM spot-check probe timed out or failed: %s", exc)

        self.last_v4_stats = {
            "attempted": llm_probes,
            "succeeded": llm_succeeded,
            "timed_out": llm_timed_out,
        }

        # V8: Composite Faithfulness Score (blends binary support ratio with continuous NLI entailment)
        support_ratio = supported_count / len(atomic_claims)
        avg_nli = sum(cd.nli_score for cd in claim_details) / len(claim_details)
        composite_score = round(0.65 * support_ratio + 0.35 * avg_nli, 4)
        if not v1_res.is_valid:
            composite_score = min(composite_score, 0.70)
        if llm_timed_out > 0 and llm_succeeded == 0:
            composite_score = round(min(composite_score, 0.82), 4)

        # Confidence level
        if composite_score >= 0.90 and llm_timed_out == 0:
            confidence = ConfidenceLevel.HIGH
        elif composite_score >= 0.70:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW

        # Repair loop
        if composite_score >= self.composite_threshold and len(unverified_list) == 0:
            verdict = VerificationVerdict.VERIFIED
            repaired_answer = answer_text
        elif supported_count > 0:
            verdict = VerificationVerdict.REPAIRED
            supported_texts = [cd.claim_text for cd in claim_details if cd.is_supported]
            repaired_core = " ".join(supported_texts)
            unverified_block = "\n\n*Could not verify: " + "; ".join(unverified_list) + "*"
            repaired_answer = f"{repaired_core}{unverified_block}"
        else:
            verdict = VerificationVerdict.ABSTAIN
            repaired_answer = "Could not verify statements with the provided sources."

        requires_abstain = verdict == VerificationVerdict.ABSTAIN

        logger.info(
            "Verification result: verdict=%s score=%.2f confidence=%s (%d/%d claims supported, v4_probes=%d/%d ok)",
            verdict,
            composite_score,
            confidence,
            supported_count,
            len(atomic_claims),
            llm_succeeded,
            llm_probes,
        )

        all_contradictions = self.v5_corroborate.detect_contradictions(cards)
        return VerificationReport(
            verdict=verdict,
            composite_faithfulness_score=float(composite_score),
            confidence=confidence,
            claim_details=claim_details,
            repaired_answer=repaired_answer,
            unverified_claims=unverified_list,
            contradictions=all_contradictions,
            requires_abstention=requires_abstain,
        )
