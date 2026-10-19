# Frozen Parser

The benchmark uses the byte-preserved `src/evaluation/parser.py`, not the older
`parse.py` compatibility utility.

1. Normalize tokenizer glyphs; remove complete `<think>...</think>` spans.
2. Scan flat non-nested JSON candidates and select the first decodable dictionary.
3. Try truthy `answer`, `FINAL_ANSWER`, then `final_answer`; accept a stripped,
   uppercased single A-E string only.
4. If absent, apply regex families in code order: JSON answer key, FINAL_ANSWER,
   Final answer, Answer:, the answer is, Option. Pattern-family precedence is NOT
   global earliest text position.
5. Finally allow a bare A-E first nonempty line with the limited punctuation in
   code. Plain yes/no/maybe is not mapped.

SPACE accepts three named labels; confidence is parsed/clipped to [0,1] with zero
fallback. Neither affects voting/correctness. Unrecovered outputs are incorrect
in full-N accuracy; recovered-only accuracy has a different denominator.

Raw exactness means JSON decoding to a dict with exactly space_label, answer and
confidence keys. Values/types/ranges are NOT checked by that metric. Fence-stripped
exactness removes one enclosing standard triple-backtick fence with optional
json/JSON tag, then repeats the key-set check. This is syntax normalization, not
semantic repair, and never changes the frozen parser's answer.
