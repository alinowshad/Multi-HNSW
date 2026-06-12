import argparse
import math
import os
import struct
from pathlib import Path

import faiss
import numpy as np

from utils.io import write_fvecs


SOURCE = "/data/ali"
DATASET = "fullopenaidecompose"
K = 32
L = 2
TRAIN_SIZE = 1_000_000
BATCH_SIZE = 100_000
SEED = 123


class FvecsMemmap:
    def __init__(self, filename: str):
        self.filename = filename
        self._raw = np.memmap(filename, dtype=np.int32, mode="r")
        self.dim = int(self._raw[0])
        self.row_width = self.dim + 1
        if self._raw.size % self.row_width != 0:
            raise ValueError(f"Invalid fvecs file shape: {filename}")
        self.rows = self._raw.size // self.row_width
        self._mat = self._raw.reshape(self.rows, self.row_width)

    def get_rows(self, row_ids):
        return self._mat[row_ids, 1:].view(np.float32).copy()

    def get_slice(self, start: int, end: int):
        return self._mat[start:end, 1:].view(np.float32).copy()


def save_float_bin(filename: str, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.float32, order="C")
    with open(filename, "wb") as f:
        f.write(struct.pack("I", matrix.shape[0]))
        f.write(struct.pack("I", matrix.shape[1]))
        matrix.tofile(f)


def stream_write_ivecs(filename: str, rows: np.ndarray) -> None:
    rows = np.asarray(rows, dtype=np.int32, order="C")
    dim = rows.shape[1]
    with open(filename, "wb") as f:
        for row in rows:
            f.write(struct.pack("i", dim))
            row.tofile(f)


def stream_write_fvecs(filename: str, rows: np.ndarray) -> None:
    rows = np.asarray(rows, dtype=np.float32, order="C")
    dim = rows.shape[1]
    with open(filename, "wb") as f:
        for row in rows:
            f.write(struct.pack("i", dim))
            row.view(np.int32).tofile(f)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(x)
    return x


def train_centroids(
    dataset: FvecsMemmap,
    k: int,
    train_size: int,
    metric: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sample_size = min(train_size, dataset.rows)
    sample_ids = np.sort(rng.integers(0, dataset.rows, size=sample_size, endpoint=False))
    train_x = dataset.get_rows(sample_ids).astype(np.float32, copy=False)

    if metric == "ip":
        normalize_rows(train_x)
        kmeans = faiss.Kmeans(
            dataset.dim,
            k,
            niter=25,
            verbose=True,
            spherical=True,
            seed=seed,
            gpu=False,
        )
    else:
        kmeans = faiss.Kmeans(
            dataset.dim,
            k,
            niter=25,
            verbose=True,
            seed=seed,
            gpu=False,
        )

    kmeans.train(train_x)
    centroids = np.asarray(kmeans.centroids, dtype=np.float32)
    if metric == "ip":
        normalize_rows(centroids)
    return centroids


def main():
    parser = argparse.ArgumentParser(description="Build overlapping IVF-style clusters for local HNSW graphs.")
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--k", type=int, default=K, help="Number of clusters")
    parser.add_argument("--overlap", type=int, default=L, help="Assign each vector to top-L clusters")
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--metric", choices=["ip", "l2"], default="ip")
    args = parser.parse_args()

    dataset_dir = Path(args.source) / args.dataset
    data_path = dataset_dir / f"{args.dataset}_base.fvecs"
    output_dir = dataset_dir / f"overlap_ivf_k{args.k}_l{args.overlap}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Clustering - {args.dataset}")
    print(f"\tdata path: {data_path}")
    print(f"\tk={args.k}, overlap={args.overlap}, metric={args.metric}")

    dataset = FvecsMemmap(str(data_path))
    print(f"\trows={dataset.rows}, dim={dataset.dim}")

    centroids = train_centroids(
        dataset=dataset,
        k=args.k,
        train_size=args.train_size,
        metric=args.metric,
        seed=args.seed,
    )

    if args.metric == "ip":
        assign_index = faiss.IndexFlatIP(dataset.dim)
    else:
        assign_index = faiss.IndexFlatL2(dataset.dim)
    assign_index.add(centroids)

    centroids_fvecs_path = output_dir / "cluster_centroids.fvecs"
    centroids_bin_path = output_dir / "cluster_centroids.bin"
    membership_path = output_dir / "cluster_membership.txt"
    cluster_ids_path = output_dir / "topl_cluster_ids.ivecs"
    cluster_scores_path = output_dir / "topl_cluster_scores.fvecs"
    stats_path = output_dir / "cluster_stats.txt"

    write_fvecs(str(centroids_fvecs_path), centroids)
    save_float_bin(str(centroids_bin_path), centroids)

    temp_dir = output_dir / "membership_parts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    member_files = [
        open(temp_dir / f"cluster_{cluster_id:06d}.txt", "w", buffering=1024 * 1024)
        for cluster_id in range(args.k)
    ]

    cluster_sizes = np.zeros(args.k, dtype=np.int64)
    cluster_id_writer = open(cluster_ids_path, "wb")
    cluster_score_writer = open(cluster_scores_path, "wb")

    try:
        for start in range(0, dataset.rows, args.batch_size):
            end = min(start + args.batch_size, dataset.rows)
            batch = dataset.get_slice(start, end).astype(np.float32, copy=False)
            if args.metric == "ip":
                normalize_rows(batch)

            scores, cluster_ids = assign_index.search(batch, args.overlap)
            cluster_ids_i32 = np.asarray(cluster_ids, dtype=np.int32, order="C")
            scores_f32 = np.asarray(scores, dtype=np.float32, order="C")

            for row in cluster_ids_i32:
                cluster_id_writer.write(struct.pack("i", args.overlap))
                row.tofile(cluster_id_writer)
            for row in scores_f32:
                cluster_score_writer.write(struct.pack("i", args.overlap))
                row.view(np.int32).tofile(cluster_score_writer)

            for local_row, assigned_clusters in enumerate(cluster_ids_i32):
                global_id = start + local_row
                for cluster_id in assigned_clusters:
                    member_files[int(cluster_id)].write(f"{global_id} ")
                    cluster_sizes[int(cluster_id)] += 1

            if start == 0 or ((start // args.batch_size) + 1) % 10 == 0:
                print(f"\tassigned {end}/{dataset.rows} vectors")
    finally:
        cluster_id_writer.close()
        cluster_score_writer.close()
        for f in member_files:
            f.close()

    with open(membership_path, "w", buffering=1024 * 1024) as out:
        for cluster_id in range(args.k):
            part_path = temp_dir / f"cluster_{cluster_id:06d}.txt"
            with open(part_path, "r") as part:
                out.write(part.read().strip())
            out.write("\n")

    with open(stats_path, "w") as f:
        f.write(f"dataset={args.dataset}\n")
        f.write(f"rows={dataset.rows}\n")
        f.write(f"dim={dataset.dim}\n")
        f.write(f"k={args.k}\n")
        f.write(f"overlap={args.overlap}\n")
        f.write(f"metric={args.metric}\n")
        f.write(f"train_size={min(args.train_size, dataset.rows)}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"membership_path={membership_path}\n")
        f.write(f"centroids_fvecs={centroids_fvecs_path}\n")
        f.write(f"centroids_bin={centroids_bin_path}\n")
        f.write(f"topl_cluster_ids={cluster_ids_path}\n")
        f.write(f"topl_cluster_scores={cluster_scores_path}\n")
        for cluster_id, cluster_size in enumerate(cluster_sizes):
            f.write(f"cluster_{cluster_id}_size={int(cluster_size)}\n")

    print("Finished overlap clustering")
    print(f"\tmembership file: {membership_path}")
    print(f"\tcentroids bin: {centroids_bin_path}")
    print(f"\ttop-L cluster ids: {cluster_ids_path}")


if __name__ == "__main__":
    main()
