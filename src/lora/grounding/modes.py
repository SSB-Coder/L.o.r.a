"""Depth modes and prompt formulation for research synthesis."""

from enum import StrEnum

from lora.grounding.cards import EvidenceCard


class DepthMode(StrEnum):
    """The depth level for synthesizing answers over the same grounded evidence."""

    ELI5 = "eli5"
    SIMPLE = "simple"
    STANDARD = "standard"
    DETAILED = "detailed"
    EXPERT = "expert"


class DepthPromptBuilder:
    """Builds targeted system and user prompts for each depth mode."""

    def build_prompt(
        self,
        query: str,
        cards: list[EvidenceCard],
        depth: DepthMode = DepthMode.STANDARD,
    ) -> str:
        """Constructs an LLM synthesis prompt adhering strictly to evidence cards."""
        card_texts: list[str] = []
        for c in cards:
            ref = f"[{c.card_id} {c.page_or_section}]" if c.page_or_section else f"[{c.card_id}]"
            card_texts.append(f"{ref} {c.title}: \"{c.exact_quote}\"")

        cards_block = "\n".join(card_texts)

        instructions = {
            DepthMode.ELI5: (
                "Explain using everyday analogies and plain 6th-grade vocabulary. "
                "Clearly label any metaphor as an analogy. Keep explanations relatable."
            ),
            DepthMode.SIMPLE: (
                "Explain in plain, clear language with short paragraphs and at least one concrete example. "
                "Avoid obscure technical jargon."
            ),
            DepthMode.STANDARD: (
                "Provide a balanced, structured synthesis highlighting core principles and key terms."
            ),
            DepthMode.DETAILED: (
                "Explain the concrete mechanisms, definitions, empirical measurements, numbers, "
                "and caveats. Include detailed explanations of how and why."
            ),
            DepthMode.EXPERT: (
                "Use precise domain-specific technical vocabulary, analyze methodologies, assumptions, "
                "limitations, and open research questions."
            ),
        }

        depth_instruction = instructions.get(depth, instructions[DepthMode.STANDARD])

        prompt = (
            f"You are LORA, a rigorous research assistant.\n"
            f"Task: Answer the user's research question strictly based on the provided evidence cards.\n\n"
            f"Style & Depth: {depth_instruction}\n\n"
            f"Rules:\n"
            f"1. Write in your own words. Do not copy sentences verbatim from sources.\n"
            f"2. Every factual claim MUST include an inline bracket citation referencing the evidence card ID (e.g. [S1 p. 2]).\n"
            f"3. Do not invent facts not supported by the evidence cards.\n\n"
            f"Evidence Cards:\n{cards_block}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        return prompt
