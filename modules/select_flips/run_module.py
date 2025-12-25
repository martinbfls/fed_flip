"""
Chooses the optimal set of label flips for a given budget
and saves both labels and indices for dataset reconstruction.
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

        # Wrong predictions
        dists[wrong] = (
            labels[wrong].max(axis=1)
            - labels[wrong][np.arange(wrong.sum()), true[wrong].argmax(axis=1)]
        )

        # Correct predictions
        sorted_logits = np.sort(labels[~wrong])
        dists[~wrong] = sorted_logits[:, -2] - sorted_logits[:, -1]

        distances.append(dists)
        label_sets.append(labels)

    distances = np.stack(distances)
    all_labels = np.stack(label_sets).mean(axis=0)

    # Save reference labels
    np.save(output_dir / "true.npy", true)

    # === Budget loop ===
    print("Selecting flips...")
    for n in budgets:
        labels_final = true.copy()
        all_labels_local = all_labels.copy()

        # ---- Select flipped indices ----
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

        # Shuffle clean indices (deterministic)
        rng = np.random.default_rng(0)
        rng.shuffle(idx_clean)

        # Save global split (for reconstruction/debug)
        np.save(output_dir / f"{n}_idx_flipped.npy", idx_flipped)
        np.save(output_dir / f"{n}_idx_clean.npy", idx_clean)

        # === Compute worker sizes ===
        base = N // num_workers
        remainder = N % num_workers
        sizes = np.array(
            [base + (w < remainder) for w in range(num_workers)],
            dtype=int
        )

        # === Split flipped samples among poisoned workers ===
        flipped_split = np.array_split(idx_flipped, num_poisoned)

        worker_indices = []
        worker_labels = []

        clean_ptr = 0

        # ---- Honest workers (clean only) ----
        for w in range(num_honests):
            sz = sizes[w]
            sel = idx_clean[clean_ptr:clean_ptr + sz]
            clean_ptr += sz

            worker_indices.append(sel)
            worker_labels.append(labels_final[sel])

        # ---- Poisoned workers (clean + flipped) ----
        for p in range(num_poisoned):
            w = num_honests + p
            sz = sizes[w]

            flipped_p = flipped_split[p]
            remaining = sz - len(flipped_p)
            if remaining < 0:
                raise ValueError(
                    f"Too many flipped samples for poisoned worker {w}"
                )

            sel_clean = idx_clean[clean_ptr:clean_ptr + remaining]
            clean_ptr += remaining

            sel = np.concatenate([sel_clean, flipped_p])

            worker_indices.append(sel)
            worker_labels.append(labels_final[sel])

        # === Save per-worker datasets ===
        for w in range(num_workers):
            np.save(output_dir / f"{n}_worker{w}_labels.npy", worker_labels[w])
            np.save(output_dir / f"{n}_worker{w}_indices.npy", worker_indices[w])


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
