import os
import sys
import time
import numpy as np

from game import Connect4Env
from mcts import MCTS
from config import MCTSConfig
from onnx_inference import Connect4ONNXInferenceModel


def load_trained_model():
    """Initialise le modèle et charge le réseau figé ONNX."""
    print("Initialisation du moteur d'inférence ONNX...")

    onnx_file = 'best_model.onnx'

    if os.path.exists(onnx_file):
        print(f"Chargement des poids depuis {onnx_file}...")
        try:
            model = Connect4ONNXInferenceModel(onnx_file)
            return model
        except Exception as e:
            print(f"Erreur lors du chargement de {onnx_file} : {e}")
            sys.exit(1)
    else:
        print(f"\n/!\\ ERREUR: Le fichier {onnx_file} est introuvable.")
        print("Veuillez d'abord lancer train.py pour générer un champion au format ONNX.\n")
        sys.exit(1)


def get_human_move(env):
    """Gère la saisie et la validation du coup du joueur humain."""
    legal_moves = env.get_legal_moves()
    while True:
        try:
            choice = input(f"\nÀ vous de jouer ! Choisissez une colonne {legal_moves} : ")
            col = int(choice)
            if col in legal_moves:
                return col
            else:
                print("Coup invalide. La colonne est pleine ou n'existe pas.")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre entier.")


def play_game(model):
    """Gère le déroulement d'une partie complète avec affichage des métriques IA."""
    env = Connect4Env()

    mcts_config = MCTSConfig(num_simulations=100)
    mcts = MCTS(model, mcts_config=mcts_config)

    print("\n" + "=" * 50)
    print("      PUISSANCE 4 - HUMAIN VS IA (Mode Debug ONNX)")
    print("=" * 50)

    # Choix du joueur
    human_player = 0
    while human_player not in [1, -1]:
        try:
            choice = input("\nVoulez-vous jouer en premier ? (o/n) : ").lower()
            if choice == 'o':
                human_player = 1
            elif choice == 'n':
                human_player = -1
            else:
                print("Veuillez répondre par 'o' ou 'n'.")
        except ValueError:
            pass

    ai_player = -human_player
    print(f"\nVous êtes le joueur {'X (1)' if human_player == 1 else 'O (-1)'}.")
    print(f"L'IA est le joueur {'X (1)' if ai_player == 1 else 'O (-1)'}.")

    # Boucle de jeu
    while True:
        print("\n" + "-" * 40)
        env.render()

        if env.current_player == human_player:
            # Tour de l'humain
            col = get_human_move(env)
            print(f"\nVous avez joué dans la colonne {col}.")
        else:
            # Tour de l'IA
            print(f"\n🧠 L'IA (MCTS {mcts_config.num_simulations} sims) réfléchit...")

            # On demande au réseau son évaluation brute (Valeur) avant que le MCTS ne fouille
            _, raw_value = model.predict(env.board, env.current_player)

            # Exécution de l'arbre de recherche MCTS
            mcts.run(env, add_dirichlet_noise=False)

            # Récupération des visites brutes pour le debug
            visits = mcts.get_visit_counts(env)
            total_visits = np.sum(visits)

            # Récupération de la décision finale (100% sur le meilleur coup, Température = 0)
            probs = mcts.get_action_probs(env, temperature=0.0)

            # Affichage de la "pensée" de l'IA
            print("\n📊 --- Analyse de l'IA ---")
            win_chance = ((raw_value + 1) / 2) * 100
            print(f"Évaluation de la position : {raw_value:+.3f} ({win_chance:.1f}% de chance de victoire)")

            print("Répartition des visites de l'arbre MCTS :")
            legal_moves = env.get_legal_moves()
            for c in range(env.cols):
                if c in legal_moves:
                    # On affiche la part des visites allouée à chaque branche
                    visit_pct = (visits[c] / total_visits) * 100 if total_visits > 0 else 0
                    marker = "⭐" if probs[c] == 1.0 else "  "
                    print(f"  Col {c}: {visit_pct:>5.1f}% ({int(visits[c])} sims) {marker}")
                else:
                    print(f"  Col {c}: Pleine")
            print("--------------------------\n")
            col = np.argmax(probs)
            print(f"L'IA choisit de jouer dans la colonne {col}.")

        # Appliquer le coup
        _, winner = env.step(col)

        # Vérification de la fin de partie
        if winner is not None:
            print("\n" + "=" * 50)
            print("         FIN DE LA PARTIE")
            print("=" * 50 + "\n")
            env.render()

            if winner == human_player:
                print("\n🎉 FÉLICITATIONS ! Vous avez battu l'IA ! 🎉")
            elif winner == ai_player:
                print("\n💀 L'IA a gagné. Meilleure chance la prochaine fois ! 💀")
            else:
                print("\n🤝 Match nul ! La grille est pleine. 🤝")
            break


def main():
    try:
        # Plus besoin des TrainConfig / ModelConfig !
        model = load_trained_model()

        while True:
            play_game(model)

            choice = input("\nVoulez-vous rejouer ? (o/n) : ").lower()
            if choice != 'o':
                print("Merci d'avoir joué ! À bientôt.")
                break

    except KeyboardInterrupt:
        print("\nPartie interrompue. Au revoir !")
        sys.exit(0)


if __name__ == "__main__":
    main()