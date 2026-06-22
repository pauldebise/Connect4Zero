import os
import time
import numpy as np

# On coupe l'accès au GPU dès l'import pour garantir que ONNX reste sur le CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from game import Connect4Env
from mcts import MCTS
from onnx_inference import Connect4ONNXInferenceModel


def async_self_play_worker(worker_id, shared_queue, config, mcts_config):
    """
    Processus de Self-Play asynchrone. Tourne en boucle infinie, recharge le modèle
    à chaud s'il détecte une nouvelle version, et pousse les parties terminées dans la file.
    """
    # Graine aléatoire unique par processus pour éviter qu'ils ne jouent tous les mêmes parties
    np.random.seed(int(time.time() * 1000) % (2 ** 32 - 1) + os.getpid())

    model_path = 'latest_model.onnx'
    current_mtime = 0
    model = None
    last_sync_time = 0

    print(f"🚀 [Actor {worker_id}] Démarrage sur CPU.")

    while True:
        current_time = time.time()

        # 1. Vérification périodique d'un nouveau modèle (sans bloquer le CPU à chaque boucle)
        if current_time - last_sync_time > config.worker_sync_interval:
            if os.path.exists(model_path):
                try:
                    mtime = os.path.getmtime(model_path)
                    if mtime != current_mtime:
                        # Petite pause pour laisser le système de fichiers (ext4/ntfs) finaliser l'écriture
                        time.sleep(0.1)
                        model = Connect4ONNXInferenceModel(model_path)
                        current_mtime = mtime
                        if worker_id == 0:  # Seul le worker 0 affiche le log pour ne pas spammer
                            print(f"📥 [Actor] Nouveau réseau chargé en mémoire.")
                except Exception:
                    pass  # Fichier en cours d'écriture ou verrouillé, on réessaiera au prochain tour
            last_sync_time = current_time

        # Si aucun modèle n'est encore disponible (début du script), on patiente
        if model is None:
            time.sleep(1)
            continue

        # 2. Initialisation d'une nouvelle partie
        env = Connect4Env()

        # On injecte la fraction PCR directement pour ce worker
        if np.random.random() < mcts_config.pcr_fraction:
            current_sims = mcts_config.num_simulations
        else:
            current_sims = mcts_config.fast_simulations

        # On clone temporairement la config pour ne pas modifier l'originale
        import copy
        local_mcts_config = copy.deepcopy(mcts_config)
        local_mcts_config.num_simulations = current_sims

        mcts = MCTS(model, mcts_config=local_mcts_config)
        game_history = []
        move_count = 0

        # 3. Boucle de jeu
        while True:
            mcts.run(env, add_dirichlet_noise=True)
            tau = 1.0 if move_count < mcts_config.temp_threshold else 0.0
            probs = mcts.get_action_probs(env, temperature=tau)

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

        # 4. Traitement des données et Mirroring
        data_to_push = []
        for state, probs, player, mcts_val in game_history:
            if winner == 0:
                z = 0.0
            elif winner == player:
                z = 1.0
            else:
                z = -1.0

            # Target value : blend entre MCTS et issue finale
            target_value = (0.5 * z) + (0.5 * mcts_val)

            # État normal
            data_to_push.append((state, probs, target_value))

            # État inversé (Data Augmentation)
            mirrored_state = np.flip(state, axis=1)
            mirrored_probs = np.flip(probs)
            data_to_push.append((mirrored_state, mirrored_probs, target_value))

        # 5. Envoi direct à la file partagée (Fire and Forget)
        shared_queue.put(data_to_push)