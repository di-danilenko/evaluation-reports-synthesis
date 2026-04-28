import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.cluster import KMeans # type: ignore
import umap.umap_ as umap # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Load embeddings from parquet
df = pd.read_parquet("title_embeddings/all_md_section_titles.parquet")
# if embeddings are stored as lists/arrays in a column
X = np.vstack(df['embedding'].to_numpy())  # shape: (n_samples, dim)

def embeddings_map2d(n_clusters=10, random_state=16, top_k=10, X=X):
    labels = df['cluster'].values
    # Reduce to 2D for visualisation
    umap_2d = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=30, min_dist=0.1) # this takes ages to run -- move to cluster? 
    coords = umap_2d.fit_transform(X)  # shape: (n_samples, 2)

    plot_df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": labels,
    })

    # Keep most prominent clusters
    largest_clusters = (
        plot_df["cluster"].value_counts()
        .head(top_k)
        .index
    )
    plot_df = plot_df[plot_df["cluster"].isin(largest_clusters)]

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="cluster",
        s=10,
        palette="tab10",
    )
    plt.title("Embedding clusters (UMAP)")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("embedding_clusters_umap.png", dpi=300)
