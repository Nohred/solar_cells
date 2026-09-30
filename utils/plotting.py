import matplotlib.pyplot as plt

# history = {
#         "train_loss": [],
#         "val_loss": [],
#         "learning_rate": []
#     }

def plot_history( history,
                ):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax, lr_ax) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax.plot(
        epochs,
        history['train_loss'],
        label='Training loss',
        color='#2563eb',
        linewidth=2.5,
        marker='o',
        markersize=4,
    )

    if 'val_loss' in history:
        ax.plot(
            epochs,
            history['val_loss'],
            label='Validation loss',
            color='#dc2626',
            linewidth=2.5,
            marker='o',
            markersize=4,
        )

    ax.set_title('Model Loss', fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xticks(list(epochs))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False)
    ax.set_facecolor('#f8fafc')


    lr_ax.plot(
        epochs,
        history['learning_rate'],
        label='Learning rate',
        color='#16a34a',
        linewidth=2.5,
        marker='o',
        markersize=4,
    )
    lr_ax.set_yscale('log')  # Set y-axis to logarithmic scale for better visualization
    lr_ax.set_title('Learning Rate', fontsize=16, fontweight='bold', pad=12)
    lr_ax.set_xlabel('Epoch')
    lr_ax.set_ylabel('Learning rate')
    lr_ax.set_xticks(list(epochs))
    lr_ax.grid(True, linestyle='--', alpha=0.3)
    lr_ax.spines['top'].set_visible(False)
    lr_ax.spines['right'].set_visible(False)
    lr_ax.legend(frameon=False)
    lr_ax.set_facecolor('#f8fafc')

    fig.patch.set_facecolor('white')

    #show the plot
    plt.show()
    return fig, ax

    