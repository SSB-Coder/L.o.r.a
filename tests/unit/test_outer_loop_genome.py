"""Unit tests for Outer Loop Genome, Validator, Compiler, and Registry."""

import tempfile
from pathlib import Path

import pytest

from lora.outer_loop.genome import (
    EditValidator,
    GenomeRegistry,
    PromptGenome,
    TemplateCompiler,
)


@pytest.fixture
def temp_db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "genomes.db"


@pytest.fixture
def base_genome():
    return PromptGenome(
        system_prompt_base="You are L.O.R.A, a grounded research assistant.",
        refinement_rules=["Assert only claims directly grounded in evidence."],
        writer_style="concise_academic",
        version=0,
    )


def test_prompt_genome_immutability_and_hash(base_genome):
    assert base_genome.version == 0
    assert len(base_genome.refinement_rules) == 1
    # Hash is non-empty and deterministic
    assert base_genome.content_hash != ""

    # Immutable model cannot be modified in place
    with pytest.raises(Exception):
        base_genome.writer_style = "journalistic"

    # Rules are stored as immutable tuple
    assert isinstance(base_genome.refinement_rules, tuple)
    with pytest.raises(AttributeError):
        base_genome.refinement_rules.append("hack")


def test_prompt_genome_few_shot_hash_uniqueness(base_genome):
    """Genomes differing only by few-shot examples must produce distinct hashes."""
    g_with_examples = base_genome.model_copy(
        update={"few_shot_examples": ({"user": "q", "assistant": "a"},)}
    )
    assert g_with_examples.content_hash != base_genome.content_hash


def test_template_compiler_slotting(base_genome):
    compiler = TemplateCompiler()
    cards = [
        {"source_title": "Paper 1", "snippet": "Room temperature superconductivity."},
    ]
    compiled = compiler.compile(
        genome=base_genome,
        user_request="Explain superconductivity",
        evidence_cards=cards,
    )

    # Core skeleton is present
    assert "You are L.O.R.A, a grounded research assistant." in compiled
    assert "Explain superconductivity" in compiled
    assert "Room temperature superconductivity" in compiled
    assert "Assert only claims directly grounded in evidence." in compiled


def test_edit_validator_allows_small_valid_edit(base_genome):
    validator = EditValidator(max_rules_added=2, max_chars_added=250)
    new_rules = tuple([
        *base_genome.refinement_rules,
        "Explicitly name research institutions for replications.",
    ])
    candidate = base_genome.model_copy(
        update={"refinement_rules": new_rules, "version": 1}
    )

    is_valid, reason = validator.validate_edit(base=base_genome, candidate=candidate)
    assert is_valid is True
    assert reason == "Valid edit"


def test_edit_validator_rejects_excessive_length(base_genome):
    validator = EditValidator(max_rules_added=2, max_chars_added=100)
    bloated_rule = "A" * 150  # Exceeds 100 chars
    candidate = base_genome.model_copy(
        update={
            "refinement_rules": tuple([*base_genome.refinement_rules, bloated_rule]),
            "version": 1,
        }
    )

    is_valid, reason = validator.validate_edit(base=base_genome, candidate=candidate)
    assert is_valid is False
    assert "Character limit exceeded" in reason


def test_edit_validator_rejects_excessive_rules(base_genome):
    validator = EditValidator(max_rules_added=1, max_chars_added=250)
    candidate = base_genome.model_copy(
        update={
            "refinement_rules": tuple([
                *base_genome.refinement_rules,
                "Rule 1",
                "Rule 2",
            ]),
            "version": 1,
        }
    )

    is_valid, reason = validator.validate_edit(base=base_genome, candidate=candidate)
    assert is_valid is False
    assert "Rule limit exceeded" in reason


def test_genome_registry_save_lineage_and_rollback(temp_db_path, base_genome):
    registry = GenomeRegistry(temp_db_path)

    # Save base genome G0
    g0_hash = registry.save_genome(base_genome, parent_hash=None)
    assert g0_hash == base_genome.content_hash
    assert registry.get_latest_genome().version == 0

    # Save mutated genome G1
    g1 = base_genome.model_copy(
        update={
            "refinement_rules": tuple([*base_genome.refinement_rules, "Add attribution."]),
            "version": 1,
        }
    )
    g1_hash = registry.save_genome(g1, parent_hash=g0_hash)
    assert registry.get_latest_genome().version == 1

    # Save mutated genome G2
    g2 = g1.model_copy(
        update={
            "refinement_rules": tuple([*g1.refinement_rules, "Clarify acronyms."]),
            "version": 2,
        }
    )
    registry.save_genome(g2, parent_hash=g1_hash)
    assert registry.get_latest_genome().version == 2

    # Check lineage history
    history = registry.get_lineage()
    assert len(history) == 3

    # Roll back to G1
    rolled_back = registry.rollback_to(g1_hash)
    assert rolled_back.version == 1
    assert registry.get_latest_genome().version == 1
    assert len(rolled_back.refinement_rules) == 2
