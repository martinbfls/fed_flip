import subprocess
from pathlib import Path

# Liste des budgets
budgets = [150, 300, 500, 1000, 1500, 2000, 2500, 5000, 10000]

# Dossier des configs
config_dir = Path("experiments")

# Dossier des logs
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

for b in budgets:
    config_file = f"train_user_backdoor_mean_{b}"
    log_file = f"train_user_backdoor_mean_{b}.log"
    
    print(f"=== Running budget {b} ===")
    print(f"Config: {config_file}")
    print(f"Log: {log_file}")
    
    # Lance le run et attend qu'il se termine
    with open(log_file, "w") as f:
        subprocess.run(
            ["python", "run_experiment.py", str(config_file)],
            stdout=f,
            stderr=subprocess.STDOUT
        )
    
    print(f"=== Finished budget {b} ===\n")
