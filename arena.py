import copy
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from game import Connect4Env
from mcts import MCTS
from onnx_inference import Connect4ONNXInferenceModel  # Importation du wrapper ONNX

# Variables globales confinées à l'espace mémoire de chaque cœur worker
_candidate_model = None
_champion_model = None


def _init_arena_worker(candidate_onnx_path, best_onnx_path):
    """Initialise de façon isolée les moteurs ONNX sur chaque cœur CPU."""
    global _candidate_model, _champion_model
    _candidate_model = Connect4ONNXInferenceModel(candidate_onnx_path)
    _champion_model = Connect4ONNXInferenceModel(best_onnx_path)


def _play_arena_game(args):
    """Joue une seule confrontation d'arène à l'aide des sessions ONNX."""
    global _candidate_model, _champion_model
    game_idx, half_games, arena_mcts_config = args

    env = Connect4Env()

    # Alternance parfaite des couleurs
    if game_idx < half_games:
        candidate_player = 1
    else:
        candidate_player = -1

    mcts_candidate = MCTS(_candidate_model, arena_mcts_config)
    mcts_champion = MCTS(_champion_model, arena_mcts_config)

    move_count = 0
    while True:
        current_mcts = mcts_candidate if env.current_player == candidate_player else mcts_champion
        current_mcts.run(env, add_dirichlet_noise=False)

        # Température à 1.0 pour les 4 premiers demi-coups (2 tours), puis 0.0
        tau = 1.0 if move_count < 4 else 0.0
        probs = current_mcts.get_action_probs(env, temperature=tau)

        # Si tau > 0, on tire au sort selon les probabilités. Sinon, on prend le coup majoritaire.
        if tau > 0:
            action = np.random.choice(env.cols, p=probs)
        else:
            action = np.argmax(probs)

        _, winner = env.step(action)
        move_count += 1

        if winner is not None:
            if winner == 0:
                return (game_idx, candidate_player, 0.5, 0, 0, 1)
            elif winner == candidate_player:
                return (game_idx, candidate_player, 1.0, 1, 0, 0)
            else:
                return (game_idx, candidate_player, 0.0, 0, 1, 0)

def run_parallel_arena(candidate_onnx_path, best_onnx_path, arena_config, mcts_config):
    """Orchestre les matchs d'arène parallélisés sous environnement ONNX Runtime."""
    print("\n" + "=" * 70)
    print(f" ⚔️ ENTRÉE EN ARÈNE (ONNX) : CANDIDAT vs CHAMPION ({arena_config.num_games} parties) ".center(70, "="))
    print("=" * 70)

    arena_mcts_config = copy.deepcopy(mcts_config)
    num_games = arena_config.num_games
    half_games = num_games // 2

    tasks = [(i, half_games, arena_mcts_config) for i in range(num_games)]

    candidate_points = 0.0
    candidate_wins = 0
    champion_wins = 0
    draws = 0

    num_workers = arena_config.num_workers

    with mp.Pool(processes=num_workers,
                 initializer=_init_arena_worker,
                 initargs=(candidate_onnx_path, best_onnx_path)) as pool:

        # --- MODIFICATION ICI : On enveloppe la boucle avec tqdm ---
        for result in tqdm(pool.imap_unordered(_play_arena_game, tasks), total=num_games, desc="Combats d'Arène", unit="partie"):
            game_idx, cand_player, pts, c_win, ch_win, draw = result

            candidate_points += pts
            candidate_wins += c_win
            champion_wins += ch_win
            draws += draw
            # Le print par partie a été supprimé pour laisser tqdm afficher la barre proprement

    win_rate = candidate_points / num_games
    is_accepted = win_rate >= arena_config.win_rate_threshold

    print("\n" + "=" * 70)
    print(f" 📊 RÉSULTAT DE L'ARÈNE ONNX ".center(70, "="))
    print("=" * 70)
    print(f"• Victoires Candidat : {candidate_wins}")
    print(f"• Victoires Champion : {champion_wins}")
    print(f"• Matchs Nuls        : {draws}")
    print(f"• Score Candidat     : {candidate_points}/{num_games} points")
    print(f"• Taux de succès     : {win_rate * 100:.1f}% (Seuil requis : {arena_config.win_rate_threshold * 100:.1f}%)")

    if is_accepted:
        print("🎉 LE CANDIDAT EST APPROUVÉ ET DEVIENT LE NOUVEAU CHAMPION ! 🎉")
    else:
        print("❌ LE CANDIDAT N'EST PAS ASSEZ FORT. LE CHAMPION CONSERVE SON TITRE. ❌")
    print("=" * 70 + "\n")

    return is_accepted