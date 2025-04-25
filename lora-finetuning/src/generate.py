"""
Text generation with a LoRA-fine-tuned model.
"""

import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


def generate(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> str:
    """Generate text from a LoRA-fine-tuned model.

    Loads the model from the saved directory (including LoRA adapter weights).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--prompt", type=str,
                        default="### Instruction:\nExplain what machine learning is in simple terms.\n\n### Response:\n")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.")
        while True:
            instruction = input("\nInstruction: ")
            if instruction.lower() == "quit":
                break
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            response = generate(
                args.model_path, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(f"\nResponse: {response}")
    else:
        response = generate(args.model_path, args.prompt,
                           max_new_tokens=args.max_tokens,
                           temperature=args.temperature)
        print(f"Prompt: {args.prompt}")
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
