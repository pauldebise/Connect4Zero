import multiprocessing as mp
import os
import time
import random
import datetime
import pickle
import shutil
from collections import deque
import numpy as np

# 1. On coupe l'accès au GPU pour TOUS les processus (Workers inclus)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from game import Connect4Env
from mcts import MCTS
from config import ModelConfig, MCTSConfig, TrainConfig, ArenaConfig


class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def add(self, game_data):
        self.buffer.extend(game_data)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([x[0] for x in batch])
        probs = np.array([x[1] for x in batch])
        values = np.array([x[2] for x in batch])
        return states, probs, values

    def __len__(self):
        return len(self.buffer)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"\n💾 [ReplayBuffer] Sauvegardé dans {filepath} ({len(self.buffer)} éléments).")

    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)
                self.buffer.extend(loaded_data)
            print(f"📥 [ReplayBuffer] Chargé depuis {filepath} ({len(self.buffer)} éléments).")


def self_play_worker(queue: mp.Queue, onnx_path: str, model_config: ModelConfig,
                     train_config: TrainConfig, mcts_config: MCTSConfig):
    """
    Worker de Self-Play purement ONNX. Aucune dépendance ni chargement TensorFlow.
    Recharge dynamiquement le fichier ONNX si sa signature temporelle change.
    """
    from game import Connect4Env
    from mcts import MCTS
    from onnx_inference import Connect4ONNXInferenceModel

    np.random.seed(int(time.time() * 1000) % (2 ** 32 - 1) + os.getpid())

    current_mtime = 0
    model = None

    while True:
        # Surveillance de la mise à jour du fichier ONNX par le process central
        try:
            if not os.path.exists(onnx_path):
                time.sleep(0.5)
                continue

            mtime = os.path.getmtime(onnx_path)
            if mtime != current_mtime or model is None:
                time.sleep(0.1)  # Laisse le temps de finaliser l'écriture I/O
                model = Connect4ONNXInferenceModel(onnx_path)
                current_mtime = mtime
        except Exception:
            time.sleep(0.5)
            continue

        env = Connect4Env()
        mcts = MCTS(model, mcts_config=mcts_config)
        game_history = []
        move_count = 0

        # Boucle de jeu
        while True:
            # --- NOUVEAU : Logique du Playout Cap Randomization (PCR) ---
            # On tire au sort pour ce tour spécifique
            if np.random.random() < mcts_config.pcr_fraction:
                current_sims = mcts_config.num_simulations
            else:
                current_sims = mcts_config.fast_simulations

            # On injecte la limite directement dans l'instance config du MCTS
            mcts.mcts_config.num_simulations = current_sims
            # ------------------------------------------------------------

            mcts.run(env, add_dirichlet_noise=True)
            tau = 1.0 if move_count < mcts_config.temp_threshold else 0.0
            probs = mcts.get_action_probs(env, temperature=tau)

            # ... (la suite reste identique : extraction de la valeur MCTS, encodage, etc.)

            # --- (Rappel du code précédent avec l'astuce MuZero si tu l'avais intégrée) ---
            mcts_value = mcts.get_root_value()
            current_player_channel = (env.board == env.current_player).astype(np.float32)
            opponent_channel = (env.board == -env.current_player).astype(np.float32)
            encoded_state = np.stack([current_player_channel, opponent_channel], axis=-1)

            game_history.append((encoded_state, probs, env.current_player, mcts_value))

            action = np.random.choice(env.cols, p=probs)
            _, winner = env.step(action)

            move_count += 1
            if winner is not None:
                break

        # Traitement des données finales et mirroring
        data_to_push = []
        # On unpack les 4 éléments maintenant
        for state, probs, player, mcts_val in game_history:

            # 1. Le résultat réel (AlphaZero classique)
            if winner == 0:
                z = 0.0
            elif winner == player:
                z = 1.0
            else:
                z = -1.0

            # 2. LE BLEND (Mélange 50/50 entre la Réalité et le MCTS)
            # On pondère le résultat final et l'évaluation instantanée du MCTS
            target_value = (0.5 * z) + (0.5 * mcts_val)

            data_to_push.append((state, probs, target_value))

            # Data Augmentation (Mirroring)
            mirrored_state = np.flip(state, axis=1)
            mirrored_probs = np.flip(probs)
            data_to_push.append((mirrored_state, mirrored_probs, target_value))

        queue.put({
            'data': data_to_push,
            'winner': winner,
            'moves': len(game_history)
        })


def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    from model import Connect4Model
    from model import export_keras_to_onnx
    from tqdm import tqdm

    # Initialisation des configurations
    model_config = ModelConfig()
    mcts_config = MCTSConfig()
    train_config = TrainConfig()
    arena_config = ArenaConfig()

    best_weights_path = 'best_weights.weights.h5'
    candidate_weights_path = 'candidate_weights.weights.h5'
    temp_weights_path = 'temp_weights.weights.h5'

    best_onnx_path = 'best_model.onnx'
    candidate_onnx_path = 'candidate_model.onnx'

    # --- NOUVEAU : Création du dossier d'historique pour le tournoi Elo ---
    os.makedirs('history', exist_ok=True)

    # --- Initialisation du réseau maître (TensorFlow) ---
    global_model = Connect4Model(model_config=model_config, train_config=train_config)
    global_model.summary()

    if os.path.exists(best_weights_path):
        global_model.model.load_weights(best_weights_path)
        print(f"📥 [Modèle] Champion restauré depuis {best_weights_path}")
    else:
        global_model.model.save_weights(temp_weights_path)
        os.replace(temp_weights_path, best_weights_path)
        print(f"🆕 [Modèle] Initialisation d'un champion vierge.")

    # Génération du fichier ONNX initial du champion s'il n'existe pas
    if not os.path.exists(best_onnx_path):
        from model import export_keras_to_onnx
        export_keras_to_onnx(global_model.model, best_onnx_path)

    # --- NOUVEAU : Archivage du modèle initial (Le repère à 1000 Elo) ---
    if not os.path.exists('history/gen_000.onnx'):
        shutil.copyfile(best_onnx_path, 'history/gen_000.onnx')
        print("📦 [Historique] Modèle initial (Gen 0) archivé. Il servira d'ancre à 1000 Elo.")

    # --- Chargement du Buffer ---
    buffer = ReplayBuffer(max_size=train_config.buffer_size)
    buffer.load('replay_buffer.pkl')

    queue = mp.Queue()
    num_workers = train_config.num_workers
    print(f"Démarrage de {num_workers} processus workers basés sur {best_onnx_path}...\n")

    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=self_play_worker, args=(queue, best_onnx_path, model_config, train_config, mcts_config))
        p.daemon = True
        p.start()
        workers.append(p)

    train_threshold = train_config.train_threshold
    batch_size = train_config.batch_size
    new_samples_count = 0
    generation = 1

    champions_crowned = 0

    total_games_in_gen = 0
    p1_wins, p2_wins, draws = 0, 0, 0
    game_lengths = []

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    try:
        print(f"\n" + "=" * 70)
        print(f"=== [GÉNÉRATION {generation}] Collecte des données ===".center(70))
        print(
            f"=== HISTORIQUE : {champions_crowned}/{generation - 1} Candidats ont été promus Champions ===".center(70))
        print("=" * 70 + "\n")

        pbar = tqdm(total=train_threshold, desc="Génération Self-Play", unit="états")

        while True:
            if not queue.empty():
                packet = queue.get()

                data_len = len(packet['data'])
                buffer.add(packet['data'])
                new_samples_count += data_len

                total_games_in_gen += 1
                game_lengths.append(packet['moves'])
                if packet['winner'] == 1:
                    p1_wins += 1
                elif packet['winner'] == -1:
                    p2_wins += 1
                else:
                    draws += 1

                pbar.update(data_len)

                if new_samples_count >= train_threshold and len(buffer) >= train_threshold:
                    pbar.close()

                    print(f"\n🚀 Optimisation du réseau de neurones (Candidat)... \n")

                    sample_size = min(len(buffer), 32768)
                    states, probs, values = buffer.sample(sample_size)

                    epochs_per_gen = 3
                    target_epoch = generation * epochs_per_gen
                    start_epoch = (generation - 1) * epochs_per_gen

                    global_model.model.fit(
                        x=states, y={'policy_head': probs, 'value_head': values},
                        batch_size=train_config.batch_size,
                        epochs=target_epoch, initial_epoch=start_epoch,
                        verbose=1, callbacks=[tensorboard_callback]
                    )

                    global_model.model.save_weights(temp_weights_path)
                    os.replace(temp_weights_path, candidate_weights_path)

                    from model import export_keras_to_onnx
                    export_keras_to_onnx(global_model.model, candidate_onnx_path)

                    # --- ÉVALUATION EN ARÈNE PARALLÈLE ONNX ---
                    from arena import run_parallel_arena
                    candidate_approved = run_parallel_arena(
                        candidate_onnx_path=candidate_onnx_path,
                        best_onnx_path=best_onnx_path,
                        arena_config=arena_config,
                        mcts_config=mcts_config
                    )

                    if candidate_approved:
                        global_model.model.save_weights(temp_weights_path)
                        os.replace(temp_weights_path, best_weights_path)
                        shutil.copyfile(candidate_onnx_path, best_onnx_path)

                        # --- NOUVEAU : Archivage du modèle de cette génération ---
                        archive_path = f"history/gen_{generation:03d}.onnx"
                        shutil.copyfile(candidate_onnx_path, archive_path)

                        print(f"🏆 [Arène] Nouveau champion validé et déployé sur l'ensemble des processus !")
                        print(f"📦 [Historique] Modèle sauvegardé dans {archive_path}")
                        champions_crowned += 1
                    else:
                        print(f"ℹ️ [Arène] Le champion conserve son titre.")
                        global_model.model.load_weights(best_weights_path)
                        print("🔄 [Rollback] Les poids du candidat ont été réinitialisés sur ceux du Champion.")

                    new_samples_count = 0
                    total_games_in_gen = 0
                    p1_wins, p2_wins, draws = 0, 0, 0
                    game_lengths = []

                    generation += 1

                    print(f"\n" + "=" * 70)
                    print(f"=== [GÉNÉRATION {generation}] Collecte des données ===".center(70))
                    print(
                        f"=== HISTORIQUE : {champions_crowned}/{generation - 1} Candidats ont été promus Champions ===".center(
                            70))
                    print("=" * 70 + "\n")

                    pbar = tqdm(total=train_threshold, desc="Génération Self-Play", unit="états")

            else:
                time.sleep(0.5)

    except KeyboardInterrupt:
        if 'pbar' in locals():
            pbar.close()
        print("\n\n👋 Arrêt manuel. Terminaison des processus enfants...")
        for w in workers:
            w.terminate()
            w.join()
        buffer.save('replay_buffer.pkl')
        print("Exécution terminée proprement.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()