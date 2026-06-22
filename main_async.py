import multiprocessing as mp
import sys
import time

# Importation des configurations
from config import AsyncTrainConfig, MCTSConfig, ModelConfig

# Importation des fonctions maîtresses de tes nouveaux modules
from actor import async_self_play_worker
from learner import async_learner_process


def main():
    # 1. Configuration système cruciale sous Linux
    # Force la méthode 'spawn' au lieu de 'fork'.
    # C'est absolument obligatoire quand on mélange TensorFlow, CUDA et Multiprocessing,
    # sinon la VRAM se corrompt lors du clonage des processus.
    mp.set_start_method('spawn', force=True)

    print("=" * 70)
    print(" 🚀 INITIALISATION DU PIPELINE ASYNCHRONE (ALPHA ZERO) ".center(70))
    print("=" * 70)

    # 2. Instanciation des configurations
    config = AsyncTrainConfig()
    mcts_config = MCTSConfig()
    model_config = ModelConfig()

    # 3. Création de l'artère de communication (IPC)
    # Cette file d'attente transfère les données de la RAM des processus enfants
    # vers le thread de lecture du processus principal.
    shared_queue = mp.Queue()

    workers = []

    # 4. Démarrage de la flotte d'Acteurs (CPU)
    print(f"Déploiement de {config.num_workers} processus Acteurs (Self-Play) sur le processeur...")
    for i in range(config.num_workers):
        p = mp.Process(
            target=async_self_play_worker,
            args=(i, shared_queue, config, mcts_config)
        )
        # daemon = True garantit que si tu fermes le script principal,
        # les processus enfants meurent instantanément sans rester en zombie.
        p.daemon = True
        p.start()
        workers.append(p)

    # Petite pause esthétique pour laisser les workers afficher leurs logs de démarrage
    time.sleep(2)

    # 5. Démarrage du Learner (GPU) dans le thread principal
    # On exécute le Learner ici plutôt que dans un processus séparé pour deux raisons :
    # - TensorFlow s'initialise parfaitement dans le process "Main".
    # - C'est beaucoup plus simple d'attraper l'interruption clavier (Ctrl+C).
    try:
        async_learner_process(shared_queue, config, model_config)

    except KeyboardInterrupt:
        print("\n\n👋 Interruption manuelle détectée (Ctrl+C).")
        print("Arrêt de tous les processus en cours...")

        # Nettoyage propre des ressources multiprocessing
        for w in workers:
            w.terminate()
            w.join()

        print("Arrêt complet. À bientôt !")
        sys.exit(0)


if __name__ == "__main__":
    main()