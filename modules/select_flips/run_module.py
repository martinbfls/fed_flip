"""
Chooses the optimal set of label flips for a given budget.
"""

from pathlib import Path
import sys, glob
import numpy as np

from modules.base_utils.util import extract_toml, slurmify_path


def run(experiment_name, module_name, **kwargs):
    slurm_id = kwargs.get("slurm_id", None)

    # === Load configuration ===
    args = extract_toml(experiment_name, module_name)
    budgets = args.get("budgets", [150, 300, 500, 1000, 1500])
    input_label_glob = slurmify_path(args["input_label_glob"], slurm_id)
    true_labels = slurmify_path(args["true_labels"], slurm_id)
    output_dir = slurmify_path(args["output_dir"], slurm_id)

    num_honests = args.get("num_honests", 2)
    num_poisoned = args.get("num_poisoned", 1)
    num_workers = num_honests + num_poisoned

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # === Load true labels ===
    true = np.load(true_labels)
    N = len(true)

    # === Compute margins ===
    print("Calculating margins...")
    distances = []
    label_sets = []

    for f in glob.glob(input_label_glob):
        labels = np.load(f)

        dists = np.zeros(N)
        wrong = labels.argmax(axis=1) != true.argmax(axis=1)

        dists[wrong] = (
            labels[wrong].max(axis=1)
            - labels[wrong][np.arange(wrong.sum()), true[wrong].argmax(axis=1)]
        )

        sorted_logits = np.sort(labels[~wrong])
        dists[~wrong] = sorted_logits[:, -2] - sorted_logits[:, -1]

        distances.append(dists)
        label_sets.append(labels)

    distances = np.stack(distances)
    all_labels = np.stack(label_sets).mean(axis=0)

    # Save global true labels (reference)
    np.save(f"{output_dir}/true.npy", true)

    # === Budget loop ===
    print("Selecting flips...")
    for n in budgets:
        labels_final = true.copy()
        all_labels_local = all_labels.copy()

        if n > 0:
            idx_flipped = np.argsort(distances.min(axis=0))[-n:]
            labels_final[idx_flipped] = (
                all_labels_local[idx_flipped]
                - 50000 * true[idx_flipped]
            )
        else:
            idx_flipped = np.array([], dtype=int)

        idx_flipped = np.unique(idx_flipped)
        idx_clean = np.setdiff1d(np.arange(N), idx_flipped)

        # Shuffle clean indices to avoid bias
        rng = np.random.default_rng(0)
        rng.shuffle(idx_clean)

        # === Compute worker sizes ===
        base = N // num_workers
        remainder = N % num_workers
        sizes = np.array([base + (w < remainder) for w in range(num_workers)], dtype=int)

        # === Split flipped samples among poisoned workers ===
        flipped_split = np.array_split(idx_flipped, num_poisoned)

        worker_datasets = []
        clean_ptr = 0

        # Honest workers (clean only)
        for w in range(num_honests):
            sz = sizes[w]
            sel = idx_clean[clean_ptr:clean_ptr + sz]
            clean_ptr += sz
            worker_datasets.append(labels_final[sel])

        # Poisoned workers (mix clean + flipped)
        for p in range(num_poisoned):
            w = num_honests + p
            sz = sizes[w]

            flipped_p = flipped_split[p]
            remaining = sz - len(flipped_p)
            if remaining < 0:
                raise ValueError("Too many flipped samples for poisoned worker")

            sel_clean = idx_clean[clean_ptr:clean_ptr + remaining]
            clean_ptr += remaining

            sel = np.concatenate([sel_clean, flipped_p])
            worker_datasets.append(labels_final[sel])

        # Save per-worker datasets
        for w, data in enumerate(worker_datasets):
            np.save(f"{output_dir}/{n}_worker{w}.npy", data)


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
