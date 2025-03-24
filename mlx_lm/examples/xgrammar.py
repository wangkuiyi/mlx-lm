#!/usr/bin/env python
# Copyright © 2023-2024 Apple Inc.

"""
Example that demonstrates how to use XGrammar with mlx-lm for constrained JSON generation.

This script shows how to create a logits processor using XGrammar and apply it
to mlx-lm's generation to produce valid JSON outputs.

Usage:
    python -m mlx_lm.examples.xgrammar --model <model_name_or_path> [--prompt <prompt_text>]
"""

import argparse
import json

import mlx.core as mx
from mlx_lm.generate import generate
from mlx_lm.utils import load
from transformers import AutoTokenizer

import xgrammar


class XGrammarLogitsProcessor:
    """
    A logits processor that constrains generation according to a grammar.

    This processor uses XGrammar's GrammarMatcher to create token bitmasks that
    constrain the model to only generate tokens that conform to the specified grammar.
    """

    def __init__(
        self,
        grammar: xgrammar.CompiledGrammar,
        max_rollback_tokens: int = 16,
    ):
        """
        Initialize the XGrammar logits processor.

        Args:
            grammar: A compiled grammar from XGrammar
            max_rollback_tokens: Maximum number of tokens to rollback for grammar matching
        """
        self.matcher = xgrammar.GrammarMatcher(grammar, max_rollback_tokens=max_rollback_tokens)
        self.vocab_size = grammar.tokenizer_info.vocab_size
        self.batch_bitmask_shape = xgrammar.get_bitmask_shape(1, self.vocab_size)
        self.bitmask = xgrammar.allocate_token_bitmask(1, self.vocab_size)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """
        Apply the grammar constraint to the logits.

        Args:
            tokens: The tokens generated so far (mx.array)
            logits: The logits from the model (mx.array) for the next token

        Returns:
            The constrained logits (mx.array)
        """
        # Reset the bitmask
        xgrammar.reset_token_bitmask(self.bitmask)

        # Convert the tokens to a list
        token_ids = tokens.tolist()

        # Check if we need to reset the matcher
        if len(token_ids) == 1:
            self.matcher.reset()

        # Let's make sure each newly generated token conforms to our grammar
        for token_id in token_ids:
            if not self.matcher.accept_token(token_id):
                # If the token is not accepted, we should reset and try again
                # This is a fallback that shouldn't normally happen with the bitmask
                self.matcher.reset()
                self.matcher.accept_token(token_id)

        # Fill the bitmask with valid next tokens according to the grammar
        self.matcher.fill_next_token_bitmask(self.bitmask)

        # Convert the bitmask to logits mask where invalid tokens have -inf logits
        bitmask_mx = mx.array(self.bitmask.view())

        # Apply the mask to the logits
        # We achieve this by setting the logits of invalid tokens to -infinity
        logit_mask = mx.zeros((1, self.vocab_size))

        # For each 32-bit chunk in the bitmask
        for i in range(self.batch_bitmask_shape[1]):
            chunk = bitmask_mx[0, i]
            # Extract each bit from the 32-bit chunk
            for j in range(32):
                idx = i * 32 + j
                if idx < self.vocab_size:
                    # If bit is 0 (invalid token), set corresponding logit to -inf
                    if (chunk & (1 << j)) == 0:
                        logit_mask[0, idx] = float("-inf")

        # Apply the mask
        return logits + logit_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with a grammar constraint")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The path to the local model directory or Hugging Face repo.",
        required=True,
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="Generate a JSON object with name, age and city:",
        help="The prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.6,
        help="The sampling temperature",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The PRNG seed",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    model, _ = load(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)  # XGrammar needs a HuggingFace tokenizer.

    mx.random.seed(args.seed)

    generated_text = generate(
        model,
        tokenizer,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True
        ),
        args.max_tokens,
        # logits_processors=[
        #     XGrammarLogitsProcessor(
        #         grammar=xgrammar.GrammarCompiler(
        #             tokenizer_info=xgrammar.TokenizerInfo.from_huggingface(tokenizer)
        #         ).compile_builtin_json_grammar()
        #     )
        # ],
    )

    print("\nGenerated text:")
    print(generated_text)
    """
    # Try to parse the JSON to validate it
    try:
        # Extract JSON from the generated text
        json_start = generated_text.find('{')
        json_end = generated_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_text = generated_text[json_start:json_end]
            parsed_json = json.loads(json_text)
            print("\nSuccessfully parsed JSON:")
            print(json.dumps(parsed_json, indent=2))
        else:
            print("\nNo valid JSON found in the generated text")
    except json.JSONDecodeError as e:
        print(f"\nError parsing JSON: {e}")
    """


if __name__ == "__main__":
    main()
