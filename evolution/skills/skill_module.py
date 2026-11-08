"""Wraps a SKILL.md file as a DSPy module for optimization.

The key abstraction: a skill file becomes a parameterized DSPy module
where the skill text is the optimizable parameter. GEPA can then
mutate the skill text and evaluate the results.

The optimizable text lives in ``SkillModule.predictor.signature.instructions``.
That is the exact attribute GEPA reads as the seed candidate and writes back
with ``pred.signature = pred.signature.with_instructions(...)`` after every
reflective proposal, so the evolved text can be read back from the compiled
module instead of being lost in a plain Python attribute.
"""

import re
from pathlib import Path
from typing import Optional

import dspy


def load_skill(skill_path: Path) -> dict:
    """Load a skill file and parse its frontmatter + body.

    Returns:
        {
            "path": Path,
            "raw": str (full file content),
            "frontmatter": str (YAML between --- markers),
            "body": str (markdown after frontmatter),
            "name": str,
            "description": str,
        }
    """
    raw = skill_path.read_text()

    # Parse YAML frontmatter
    frontmatter = ""
    body = raw
    if raw.strip().startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()

    # Extract name and description from frontmatter
    name = ""
    description = ""
    for line in frontmatter.split("\n"):
        if line.strip().startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("'\"")
        elif line.strip().startswith("description:"):
            description = line.split(":", 1)[1].strip().strip("'\"")

    return {
        "path": skill_path,
        "raw": raw,
        "frontmatter": frontmatter,
        "body": body,
        "name": name,
        "description": description,
    }


def find_skill(skill_name: str, hermes_agent_path: Path) -> Optional[Path]:
    """Find a skill by name in the hermes-agent skills directory.

    Searches recursively for a SKILL.md in a directory matching the skill name.
    """
    skills_dir = hermes_agent_path / "skills"
    if not skills_dir.exists():
        return None

    # Direct match: skills/<category>/<skill_name>/SKILL.md
    for skill_md in skills_dir.rglob("SKILL.md"):
        if skill_md.parent.name == skill_name:
            return skill_md

    # Fuzzy match: check the name field in frontmatter
    for skill_md in skills_dir.rglob("SKILL.md"):
        try:
            content = skill_md.read_text()[:500]
            if f"name: {skill_name}" in content or f'name: "{skill_name}"' in content:
                return skill_md
        except Exception:
            continue

    return None


class SkillModule(dspy.Module):
    """A DSPy module that wraps a skill file for optimization.

    The skill text is the parameter GEPA optimizes, so it is stored in the
    predictor's signature instructions -- the attribute GEPA mutates -- rather
    than in a plain Python attribute that no optimizer can see. On each forward
    pass, the module:
    1. Uses the skill text as the predictor's instructions
    2. Processes the task input
    3. Returns the agent's response

    ``frontmatter`` and ``raw_skill`` are captured at construction so
    :func:`extract_skill_markdown` can compare the evolved instructions against
    the original body and reassemble the file with the original YAML header.
    """

    class TaskWithSkill(dspy.Signature):
        """Complete a task following the provided skill instructions.

        You are an AI agent following specific skill instructions to complete a task.
        Read the skill instructions carefully and follow the procedure described.
        """
        task_input: str = dspy.InputField(desc="The task to complete")
        output: str = dspy.OutputField(desc="Your response following the skill instructions")

    def __init__(
        self,
        skill_text: str,
        frontmatter: str = "",
        raw_skill: Optional[str] = None,
    ):
        super().__init__()
        self.frontmatter = frontmatter
        self.original_raw = raw_skill
        self.predictor = dspy.Predict(self.TaskWithSkill)
        # The skill body is the predictor's instructions: this is the text GEPA
        # evolves. Do not mirror it into a plain attribute -- a copy would not
        # track the optimizer's updates.
        signature = self.predictor.signature
        assert signature is not None
        self.predictor.signature = signature.with_instructions(skill_text)
        # DSPy normalizes signature instructions (trims edge whitespace), so
        # capture the normalized text as the baseline: the unchanged check in
        # extract_skill_markdown must compare against the form GEPA sees.
        self.original_body = self.skill_text

    @property
    def skill_text(self) -> str:
        """The current skill instructions held by the predictor signature.

        Reads ``predictor.signature.instructions`` so the value reflects GEPA's
        evolved text after ``compile`` instead of the original body.
        """
        signature = self.predictor.signature
        assert signature is not None
        return getattr(signature, "instructions")

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input)
        return dspy.Prediction(output=result.output)


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    """Reassemble a skill file from frontmatter and evolved body.

    Preserves the original YAML frontmatter (name, description, metadata)
    and replaces only the body with the evolved version.
    """
    return f"---\n{frontmatter}\n---\n\n{evolved_body}\n"


def extract_skill_markdown(
    module: SkillModule,
    frontmatter: Optional[str] = None,
    original_raw: Optional[str] = None,
    max_growth_ratio: Optional[float] = None,
    baseline_text: Optional[str] = None,
) -> str:
    """Extract the current SKILL.md text from a (possibly optimized) SkillModule.

    The skill body is read from ``module.predictor.signature.instructions`` --
    the attribute GEPA rewrites. If those instructions are unchanged from the
    module's initial skill text, the original raw file content is returned
    verbatim. If they changed, the evolved instructions are reassembled with the
    original YAML frontmatter, so name/description/metadata survive intact.

    Ockham budget: when ``max_growth_ratio`` is given, a changed body that
    would push the reassembled markdown past that fractional growth over the
    baseline (``baseline_text`` or the module's original raw file) is
    deterministically distilled via
    :func:`evolution.skills.ockham.ockham_extract_markdown` so the extracted
    skill complies with the growth cap *instead of* reverting to the old
    text -- evolved rules survive in compressed form, frontmatter intact.

    Args:
        module: The SkillModule to read (baseline or GEPA-optimized).
        frontmatter: Optional frontmatter override. Defaults to the frontmatter
            captured when the module was constructed.
        original_raw: Optional original raw SKILL.md override. Defaults to the
            raw text captured when the module was constructed.
        max_growth_ratio: Optional Ockham growth cap. When set, the returned
            markdown is guaranteed to be within this fractional growth of the
            baseline.
        baseline_text: Optional baseline markdown the growth cap is measured
            against. Defaults to ``original_raw`` / the module's original raw.

    Returns:
        The full SKILL.md content: original YAML frontmatter + current body.
    """
    evolved_body = module.skill_text
    fm = module.frontmatter if frontmatter is None else frontmatter
    raw = module.original_raw if original_raw is None else original_raw

    # Unchanged instructions: return the original bytes exactly rather than a
    # re-serialization, so an unoptimized run is byte-identical to the source.
    if evolved_body == module.original_body and raw is not None:
        return raw

    markdown = reassemble_skill(fm, evolved_body)

    if max_growth_ratio is not None:
        baseline = baseline_text if baseline_text is not None else raw
        if baseline is None:
            raise ValueError(
                "max_growth_ratio needs a baseline: pass baseline_text "
                "or construct the module with raw_skill=..."
            )
        from evolution.skills.ockham import ockham_extract_markdown

        markdown = ockham_extract_markdown(markdown, fm, baseline, max_growth_ratio)

    return markdown
