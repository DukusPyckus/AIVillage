import argparse
import json
from .config import create_default_config, Configuration, ModelReference
from .evolutionary_tournament import run_evolutionary_tournament
from .utils import generate_text, evaluate_model
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="EvoMerge: Evolutionary Model Merging System")
    parser.add_argument("--config", type=str, help="Path to a JSON configuration file")
    parser.add_argument("--run", action="store_true", help="Run the evolutionary tournament")
    parser.add_argument("--evaluate", type=str, help="Evaluate a merged model at the given path")
    parser.add_argument("--generate", type=str, help="Generate text using a merged model at the given path")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Prompt for text generation")
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Configuration(**config_dict)
    else:
        config = create_default_config()

    if args.run:
        print("Running evolutionary tournament...")
        best_model_path = run_evolutionary_tournament(config)
        print(f"Best model saved at: {best_model_path}")
        
        print("\nEvaluating best model:")
        evaluation_result = evaluate_model(best_model_path)
        print(f"Evaluation result: {evaluation_result}")

        print("\nGenerating sample text:")
        model = AutoModelForCausalLM.from_pretrained(best_model_path)
        tokenizer = AutoTokenizer.from_pretrained(best_model_path)
        generated_text = generate_text(model, tokenizer, args.prompt)
        print(f"Generated text: {generated_text}")

    elif args.evaluate:
        print(f"Evaluating model at {args.evaluate}")
        evaluation_result = evaluate_model(args.evaluate)
        print(f"Evaluation result: {evaluation_result}")

    elif args.generate:
        print(f"Generating text using model at {args.generate}")
        model = AutoModelForCausalLM.from_pretrained(args.generate)
        tokenizer = AutoTokenizer.from_pretrained(args.generate)
        generated_text = generate_text(model, tokenizer, args.prompt)
        print(f"Generated text: {generated_text}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
