from dataclasses import dataclass
import multiprocessing as mp

@dataclass
class ModelConfig:
    blocks: int = 3
    filters: int = 32


@dataclass
class MCTSConfig:
    num_simulations: int = 200
    fast_simulations: int = 40
    pcr_fraction: float = 1.0
    fpu_reduction: float = 0.1
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.5
    exploration_fraction: float = 0.25
    temp_threshold: int = 10

@dataclass
class AsyncTrainConfig:
    buffer_max_size: int = 100_000
    batch_size: int = 256
    warmup_size: int = 10_000
    checkpoint_interval: int = 500   # Sauvegarder un .onnx tous les 500 batchs traités
    worker_sync_interval: int = 30   # Les workers vérifient s'il y a un nouveau modèle toutes les 30s
    learning_rate: float = 0.0005
    l2_reg: float = 3e-4
    num_workers: int = max(1, mp.cpu_count() - 2)
    target_replay_ratio: float = 20.0  # Cible du Replay Ratio
    train_steps_per_cycle: int = 10  # Nombre de pas de gradient avant d'activer le frein

@dataclass
class TrainConfig:
    batch_size: int = 256
    buffer_size: int = 100_000
    train_threshold: int = 5000
    learning_rate: float = 0.0005
    l2_reg: float = 3e-4
    num_workers: int = max(1, mp.cpu_count() - 1)

@dataclass
class ArenaConfig:
    num_games: int = 50
    win_rate_threshold: float = 0.52
    num_workers: int = max(1, mp.cpu_count() - 1)