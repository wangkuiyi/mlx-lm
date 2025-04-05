import mlx_lm

MAIN_MODELS = [
    "mlx-community/Qwen2.5-Coder-32B-Instruct-3bit",
    "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-32B-Instruct-8bit",
    "mlx-community/Qwen2.5-Coder-32B-Instruct-bf16",
]
for MM in MAIN_MODELS:
    main_model, tokenizer = mlx_lm.load(MM)

    prompt = "Write a Python program to sum up the Fibonacci series"
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    print(f"\nAutoregression: {MM}")
    mlx_lm.generate(
        model=main_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=256,
    )

    DRAFT_MODELS = [
        "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    ]

    for DM in DRAFT_MODELS:
        print(f"\nDraft Model: {DM}")
        draft_model, _ = mlx_lm.load(DM)
        assert draft_model

        for num_draft_tokens in range(1, 30):
            print(f"num_draft_tokens: {num_draft_tokens}")
            mlx_lm.generate(
                model=main_model,
                tokenizer=tokenizer,
                prompt=prompt,
                draft_model=draft_model,
                num_draft_tokens=num_draft_tokens,
                max_tokens=256,
            )
