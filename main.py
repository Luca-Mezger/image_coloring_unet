from train import train
from eval import evaluate
from utils import save_checkpoint
from config import config

def main():
    # Train the model
    state = train(config)
    # Evaluate the model
    evaluate(state, config)
    # Save the trained model
    save_checkpoint(state, config['checkpoint_path'])

if __name__ == "__main__":
    main()
