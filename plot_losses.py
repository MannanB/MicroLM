import pickle
import matplotlib.pyplot as plt

def plot_losses(losses_file):
    with open(losses_file, "rb") as f:
        all_losses = pickle.load(f)
    print(f"Loaded {len(all_losses)} loss entries from {losses_file}")

    if not all_losses:
        print("No losses to plot.")
        return

    # Support legacy scalar loss lists and new dict-based loss records.
    def _is_number(x):
        return isinstance(x, (int, float))

    main_losses = []
    aux_series = {}  # step -> list of values aligned with main_losses length

    if _is_number(all_losses[0]):
        main_losses = all_losses
    elif isinstance(all_losses[0], dict):
        for idx, entry in enumerate(all_losses):
            loss_val = entry.get("loss")
            main_losses.append(loss_val)

            # Collect aux losses in a normalized list of dict(step, loss)
            entry_aux = []
            if "aux_losses" in entry and isinstance(entry["aux_losses"], list):
                entry_aux.extend([aux for aux in entry["aux_losses"] if isinstance(aux, dict)])
            elif "aux_loss" in entry:
                entry_aux.append({"step": entry.get("aux_step"), "loss": entry["aux_loss"]})

            for aux in entry_aux:
                step = aux.get("step")
                if step is None:
                    continue
                if step not in aux_series:
                    aux_series[step] = [None] * idx
                aux_series[step].append(aux.get("loss"))

            # Pad existing series for entries without that aux
            for series in aux_series.values():
                if len(series) < idx + 1:
                    series.append(None)
    else:
        raise ValueError("Unsupported loss entry type. Expected number or dict with 'loss' key.")

    plt.figure(figsize=(10, 5))
    plt.plot(main_losses, label="Main Loss", color="blue")

    for step in sorted(aux_series.keys()):
        series = aux_series[step]
        label = f"Aux Loss (step {step})"
        plt.plot(series, label=label)

    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("training_loss_plot.png")
    plt.show()


if __name__ == "__main__":
    plot_losses('losses.pkl')
    print("Loss plot saved as 'training_loss_plot.png'")
