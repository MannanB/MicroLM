import pickle
import matplotlib.pyplot as plt

def plot_losses(losses_file):
    with open(losses_file, "rb") as f:
        all_losses = pickle.load(f)
    print(f"Loaded {len(all_losses)} loss entries from {losses_file}")
    plt.figure(figsize=(10, 5))
    plt.plot(all_losses, label='Training Loss', color='blue')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid()
    plt.savefig('training_loss_plot.png')
    plt.show()


if __name__ == "__main__":
    plot_losses('losses.pkl')
    print("Loss plot saved as 'training_loss_plot.png'")