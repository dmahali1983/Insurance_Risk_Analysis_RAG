import argparse
from training.train import train_model
from training.evaluate import evaluate_model
from training.finetune import fine_tune
from frontend.app import main as run_app

def main():
    parser = argparse.ArgumentParser(description="Risk Analysis RAG Model Entry Point")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune GPT model")
    parser.add_argument("--runapp", action="store_true", help="Run the Streamlit app")
    
    args = parser.parse_args()
    
    if args.train:
        print("Starting model training...")
        train_model()
    elif args.evaluate:
        print("Evaluating the model...")
        evaluate_model()
    elif args.finetune:
        print("Fine-tuning the model...")
        fine_tune()
    elif args.runapp:
        print("Launching Streamlit app...")
        run_app()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
