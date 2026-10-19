# Prompt Tasks

The three templates preserve the exact frozen bytes. The builder substitutes
`{question}` and `{options}`. SC-3 reuses the same CoT template.

Structured is an augmented task/prompt view, not a format-only intervention:
select an answer AND judge the answer space as VALID, INCOMPLETE or CONTRADICTED,
then return confidence in JSON. It requires exactly one available option letter
even for a non-VALID SPACE judgment and forbids null/None/unknown/phrases.
Empty answers are not authorized abstention. Syntax, recovery and correctness
are separate dimensions. The CoT instruction alone does not establish extended
reasoning; output length and agreement are descriptive measurements.
