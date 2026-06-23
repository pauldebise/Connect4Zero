import os
import glob
import time
import json
import copy
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from config import ArenaConfig

# On bloque l'accès au GPU de manière globale pour forcer l'exécution pure CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from game import Connect4Env
from mcts import MCTS
from onnx_inference import Connect4ONNXInferenceModel
from config import MCTSConfig

# Variables globales pour les workers parallèles
_model_a = None
_model_b = None


def _init_tournament_worker(path_a, path_b):
    global _model_a, _model_b
    _model_a = Connect4ONNXInferenceModel(path_a)
    _model_b = Connect4ONNXInferenceModel(path_b)


def _play_tournament_game(args):
    global _model_a, _model_b
    game_idx, mcts_config = args

    env = Connect4Env()
    mcts_a = MCTS(_model_a, mcts_config)
    mcts_b = MCTS(_model_b, mcts_config)

    # Alternance des couleurs
    if game_idx % 2 == 0:
        p_a, p_b = 1, -1
    else:
        p_a, p_b = -1, 1

    move_count = 0
    while True:
        current_mcts = mcts_a if env.current_player == p_a else mcts_b
        current_mcts.run(env, add_dirichlet_noise=False)

        tau = 1.0 if move_count < 4 else 0.0
        probs = current_mcts.get_action_probs(env, temperature=tau)

        if tau > 0:
            action = np.random.choice(env.cols, p=probs)
        else:
            action = np.argmax(probs)

        _, winner = env.step(action)
        move_count += 1

        if winner is not None:
            if winner == 0:
                return 0.5  # Nul
            elif winner == p_a:
                return 1.0  # Victoire Modèle A
            else:
                return 0.0  # Défaite Modèle A


def run_matchup(path_a, path_b, num_games=20):
    """Orchestre une série de matchs entre deux modèles spécifiques."""
    # MCTS très léger (50 sims) pour évaluer rapidement
    tourney_mcts_config = MCTSConfig(num_simulations=50)

    tasks = [(i, copy.deepcopy(tourney_mcts_config)) for i in range(num_games)]
    score_a = 0.0

    arena_config = ArenaConfig()
    num_workers = arena_config.num_workers

    # On isole l'affichage tqdm pour qu'il soit clair
    opp_name = os.path.basename(path_b)

    with mp.Pool(processes=num_workers,
                 initializer=_init_tournament_worker,
                 initargs=(path_a, path_b)) as pool:
        for result in tqdm(pool.imap_unordered(_play_tournament_game, tasks),
                           total=num_games, desc=f"vs {opp_name:12}", unit="pt"):
            score_a += result

    return score_a


def main():
    # --- SILENCE ABSOLU DE TENSORFLOW ---
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    # ------------------------------------------------------------------

    print("=" * 60)
    print(" 🏆 DÉMARRAGE DU TRAQUEUR ELO (CHECKPOINTS STEPS) ")
    print("=" * 60)

    os.makedirs('history', exist_ok=True)
    elo_file = 'history/elo_ratings.json'

    # Récupération de tous les modèles pour gérer l'ancre dynamique
    models = sorted(glob.glob('history/model_step_*.onnx'))
    if not models:
        print("Aucun modèle trouvé dans le dossier 'history/'. En attente...")

    # Ancre absolue : Le modèle 0 est strictement défini à 0 Elo
    anchor_model = "model_step_000000.onnx"

    if os.path.exists(elo_file):
        with open(elo_file, 'r') as f:
            elo_ratings = json.load(f)
        print(f"📊 Dictionnaire Elo chargé ({len(elo_ratings)} modèles existants).")
    else:
        # Création du dictionnaire avec l'ancre à 0 Elo
        elo_ratings = {anchor_model: 0.0}
        with open(elo_file, 'w') as f:
            json.dump(elo_ratings, f)
        print(f"📊 Nouveau dictionnaire Elo créé (Ancre: {anchor_model} à 0 Elo).")

    log_dir = "logs/elo_tracking"
    os.makedirs(log_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(log_dir)

    with summary_writer.as_default():
        for model_name, elo in elo_ratings.items():
            try:
                # Extraction du step au lieu du gen_num
                step_num = int(model_name.replace('model_step_', '').replace('.onnx', ''))
                tf.summary.scalar('Elo_Rating', elo, step=step_num)
            except ValueError:
                pass
        summary_writer.flush()

    K_PER_GAME = 16  # K-factor par partie

    try:
        while True:
            # Recherche des checkpoints
            models = sorted(glob.glob('history/model_step_*.onnx'))
            if len(models) < 2:
                time.sleep(5)
                continue

            # Trouver le premier modèle non évalué
            model_to_evaluate = None
            for m in models:
                basename = os.path.basename(m)
                if basename not in elo_ratings:
                    model_to_evaluate = basename
                    break

            if model_to_evaluate is None:
                time.sleep(5)
                continue

            step_num = int(model_to_evaluate.replace('model_step_', '').replace('.onnx', ''))
            path_a = f"history/{model_to_evaluate}"

            # --- CRÉATION DU POOL D'ADVERSAIRES (GAUNTLET) ---
            opponents = []

            # 1. Le prédécesseur immédiat (basé sur l'index de la liste triée, pas sur le nom)
            basenames = [os.path.basename(m) for m in models]
            current_idx = basenames.index(model_to_evaluate)
            pred = basenames[current_idx - 1] if current_idx > 0 else None

            if pred and pred in elo_ratings:
                opponents.append(pred)

            # 2. L'ancre absolue (Model 0)
            if anchor_model in elo_ratings and anchor_model not in opponents:
                opponents.append(anchor_model)

            # 3. Jusqu'à 2 modèles historiques aléatoires
            known_models = [m for m in elo_ratings.keys() if m not in opponents and m != model_to_evaluate]
            if len(known_models) > 0:
                num_random = min(2, len(known_models))
                random_opps = np.random.choice(known_models, size=num_random, replace=False)
                opponents.extend(random_opps)

            print(f"\n" + "-" * 60)
            print(f"⚔️  NOUVEAU CHAMPION EN ÉVALUATION : {model_to_evaluate}")
            print(f"🎯 Opposants sélectionnés : {', '.join(opponents)}")
            print("-" * 60)

            # On initialise son Elo de départ à celui de son prédécesseur (ou 0.0)
            current_elo = elo_ratings.get(pred, 0.0) if pred else 0.0
            games_per_opponent = 20

            # Déroulement du Gauntlet
            total_score = 0
            total_games = 0

            for opp in opponents:
                path_b = f"history/{opp}"
                score_a = run_matchup(path_a, path_b, num_games=games_per_opponent)

                # Mise à jour Elo itérative contre cet adversaire
                elo_opp = elo_ratings[opp]
                expected_prob = 1 / (1 + 10 ** ((elo_opp - current_elo) / 400))
                expected_score = games_per_opponent * expected_prob

                # Formule : Ancien_Elo + K * (Score_Réel - Score_Attendu)
                current_elo = current_elo + K_PER_GAME * (score_a - expected_score)

                total_score += score_a
                total_games += games_per_opponent
                print(f"   ↳ Bilan provisoire : {score_a}/{games_per_opponent} pts | Elo ajusté: {current_elo:.1f}")

            # Enregistrement final
            elo_ratings[model_to_evaluate] = current_elo

            print(f"\n📊 Bilan Final Gauntlet : {total_score}/{total_games} points")
            print(f"📈 Elo officiel validé pour {model_to_evaluate} : {current_elo:.1f}")

            with open(elo_file, 'w') as f:
                json.dump(elo_ratings, f, indent=4)

            # Log TensorBoard en utilisant le numéro de step exact sur l'axe X
            with summary_writer.as_default():
                tf.summary.scalar('Elo_Rating', current_elo, step=step_num)
                summary_writer.flush()

    except KeyboardInterrupt:
        print("\n👋 Traqueur Elo arrêté.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()