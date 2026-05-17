"""t-SNE visualisation of the 256-dimensional embedding space.

Produces both 2D and 3D scatter plots coloured by emotion label.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from audiovec.config import EMOTION_MAPPING
from audiovec.model import extract_embedding_model


def compute_embeddings(
    model: "tf.keras.Model",
    X: np.ndarray,
) -> np.ndarray:
    """Run the full dataset through the embedding model and return 256-d vectors."""
    import tensorflow as tf

    embedding_model = extract_embedding_model(model)
    return embedding_model.predict(X, verbose=0)


def plot_2d_tsne(
    X_embed: np.ndarray,
    y_encoded: np.ndarray,
    save_path: str | None = "embeddings_2d.png",
) -> None:
    """Reduce embeddings to 2D with t-SNE and scatter-plot them."""
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(X_embed)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                          c=y_encoded, cmap="viridis")

    cbar = plt.colorbar(scatter, ticks=sorted(np.unique(y_encoded)))
    cbar.ax.set_yticklabels(
        [EMOTION_MAPPING[i + 1] for i in sorted(np.unique(y_encoded))]
    )

    plt.title("2D t-SNE projection of audio emotion embeddings")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved 2D plot → {save_path}")
    plt.show()


def plot_3d_tsne(
    X_embed: np.ndarray,
    y_encoded: np.ndarray,
    emotion_labels: list[str],
    save_path: str | None = "embeddings_3d.png",
) -> None:
    """Reduce embeddings to 3D with t-SNE and plot with Plotly."""
    import plotly.express as px

    tsne = TSNE(n_components=3, random_state=42)
    X_reduced = tsne.fit_transform(X_embed)

    # Map numeric labels to readable names for the colour legend
    label_names = [emotion_labels[i] for i in y_encoded]

    fig = px.scatter_3d(
        x=X_reduced[:, 0],
        y=X_reduced[:, 1],
        z=X_reduced[:, 2],
        color=label_names,
        title="3D t-SNE projection of audio emotion embeddings",
    )
    fig.update_traces(marker=dict(size=5))
    fig.write_html("embeddings_3d.html")
    print("Saved 3D interactive plot → embeddings_3d.html")

    if save_path:
        try:
            fig.write_image(save_path)
            print(f"Saved 3D static plot → {save_path}")
        except Exception as e:
            print(f"Note: could not save static 3D image ({e}). The interactive HTML is saved.")

    fig.show()


def visualize(
    model: "tf.keras.Model",
    X: np.ndarray,
    y_encoded: np.ndarray,
) -> None:
    """Convenience: compute embeddings and show both 2D + 3D plots."""
    emotion_labels = [EMOTION_MAPPING[i + 1] for i in sorted(np.unique(y_encoded))]

    print("Extracting embeddings…")
    X_embed = compute_embeddings(model, X)
    print(f"Embedding shape: {X_embed.shape}")

    print("Computing 2D t-SNE…")
    plot_2d_tsne(X_embed, y_encoded)

    print("Computing 3D t-SNE…")
    plot_3d_tsne(X_embed, y_encoded, emotion_labels)
