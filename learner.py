import os
import time
import threading
import random
import shutil
import datetime  # 🔴 NOUVEAU : Requis pour horodater les dossiers TensorBoard
from collections import deque
import numpy as np

# 🛑 Désactivation des compilateurs expérimentaux (pour éviter les crashs NVIDIA)
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

# On s'assure de NE PAS masquer les GPU ici
os.environ.pop('CUDA_VISIBLE_DEVICES', None)
# On garde les optimisations TF pour CPU au cas où
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from model import Connect4Model, export_keras_to_onnx


def async_learner_process(shared_queue, config, model_config):
    """
    Processus central d'apprentissage.
    Tourne en boucle infinie sur le GPU, pioche dans le buffer local et exporte
    régulièrement les nouvelles versions du modèle.
    """
    print("\n" + "=" * 50)
    print(" 🧠 DÉMARRAGE DU LEARNER (Processus Principal)")
    print("=" * 50)

    # 1. Configuration matérielle (Crucial sous Linux pour éviter les crashs VRAM)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU activé et configuré avec Memory Growth : {physical_devices}")
        except RuntimeError as e:
            print(f"⚠️ Erreur de configuration GPU : {e}")
    else:
        print("⚠️ Aucun GPU détecté. L'entraînement se fera sur CPU.")

    # 2. Initialisation du Replay Buffer local
    replay_buffer = deque(maxlen=config.buffer_max_size)
    total_states_received = [0]  # Nouveau compteur absolu

    # 3. Le Thread de réception (I/O Asynchrone)
    def queue_reader():
        while True:
            # Attend de recevoir un paquet de l'un des workers
            data_packet = shared_queue.get()
            replay_buffer.extend(data_packet)
            # On incrémente le compteur absolu
            total_states_received[0] += len(data_packet)

    reader_thread = threading.Thread(target=queue_reader, daemon=True)
    reader_thread.start()
    print("📡 Thread de réception du Replay Buffer activé.")

    # 4. Initialisation du Réseau et de l'Optimiseur (avec Cosine Decay)
    estimated_steps_per_cycle = config.buffer_max_size // config.batch_size

    lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.learning_rate,
        first_decay_steps=max(1000, estimated_steps_per_cycle),
        alpha=0.01
    )

    global_model = Connect4Model(model_config=model_config, train_config=config)
    global_model.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

    # 🔴 NOUVEAU (1/3) : Ajout de 'metrics' pour calculer l'Accuracy et la MAE
    global_model.model.compile(
        optimizer=global_model.optimizer,
        loss={'policy_head': 'categorical_crossentropy', 'value_head': 'mean_squared_error'},
        loss_weights={'policy_head': 1.0, 'value_head': 0.5},
        metrics={'policy_head': 'accuracy', 'value_head': 'mae'},
        jit_compile=False
    )

    # 🔴 NOUVEAU (2/3) : Initialisation du Writer TensorBoard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "fit", current_time)
    summary_writer = tf.summary.create_file_writer(log_dir)
    print(f"📈 TensorBoard configuré. Logs prêts dans : {log_dir}")

    # Chargement des poids existants (format .keras)
    weights_path = 'best_weights.keras'
    if os.path.exists(weights_path):
        global_model.model.load_weights(weights_path)
        print(f"📥 Poids restaurés depuis {weights_path}")
    else:
        print("🆕 Initialisation d'un réseau vierge.")
        global_model.model.save(weights_path)

    # Création du premier export ONNX pour débloquer les workers
    onnx_path = 'latest_model.onnx'
    export_keras_to_onnx(global_model.model, onnx_path)

    # Archivage du Gen 0 pour l'historique Elo (le modèle de base à 0 Elo)
    os.makedirs('history', exist_ok=True)
    if not os.path.exists('history/model_step_000000.onnx'):
        shutil.copyfile(onnx_path, 'history/model_step_000000.onnx')

    step_counter = 0

    # 5. Boucle d'entraînement infinie
    while True:
        # Attente de la phase de Warmup
        if len(replay_buffer) < config.warmup_size:
            print(f"⏳ Warmup en cours : {len(replay_buffer)} / {config.warmup_size} états...", end='\r')
            time.sleep(5)
            continue

        # 1. On mémorise le nombre TOTAL d'états reçus jusqu'ici
        current_received = total_states_received[0]

        # 2. Le GPU fait son cycle d'entraînement (ex: 10 pas de gradient)
        for _ in range(config.train_steps_per_cycle):
            batch = random.sample(replay_buffer, config.batch_size)
            states = np.array([x[0] for x in batch])
            probs = np.array([x[1] for x in batch])
            values = np.array([x[2] for x in batch])

            logs = global_model.model.train_on_batch(
                x=states, y={'policy_head': probs, 'value_head': values}, return_dict=True
            )

        # On incrémente le compteur de steps proprement en dehors du For et du While
        step_counter += config.train_steps_per_cycle

        # 3. Le Frein (Throttling) basé sur le compteur absolu
        states_processed = config.train_steps_per_cycle * config.batch_size
        states_to_wait = int(states_processed / config.target_replay_ratio)

        # Le GPU se met en veille tant que le THREAD n'a pas traité le quota
        while total_states_received[0] < current_received + states_to_wait:
            time.sleep(0.5)

        # Affichage régulier dans la console
        if step_counter % 50 < config.train_steps_per_cycle:
            print(
                f"⚡ Step {step_counter:06d} | Loss Globale: {logs['loss']:.4f} | Value Loss: {logs['value_head_loss']:.4f} | Buffer: {len(replay_buffer)}"
            )

        # 6. Checkpoint et Déploiement périodique
        if step_counter % config.checkpoint_interval < config.train_steps_per_cycle:
            print(f"\n💾 [Checkpoint] Sauvegarde des poids et export ONNX au step {step_counter}...")

            # 🔴 NOUVEAU (3/3) : Écriture des métriques dans TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar('Entrainement/Loss_Totale', logs['loss'], step=step_counter)
                tf.summary.scalar('Entrainement/Loss_Policy', logs['policy_head_loss'], step=step_counter)
                tf.summary.scalar('Entrainement/Loss_Value', logs['value_head_loss'], step=step_counter)

                # Ces métriques existent maintenant grâce à l'ajout dans model.compile()
                tf.summary.scalar('Metriques/Policy_Accuracy', logs['policy_head_accuracy'], step=step_counter)
                tf.summary.scalar('Metriques/Value_MAE', logs['value_head_mae'], step=step_counter)

                # Facultatif mais recommandé : on surveille l'évolution du learning rate
                current_lr = lr_scheduler(step_counter).numpy()
                tf.summary.scalar('Hyperparametres/Learning_Rate', current_lr, step=step_counter)

                summary_writer.flush()  # Force l'écriture sur le disque immédiatement

            # Sauvegarde des poids
            global_model.model.save(weights_path)

            # Exportation propre
            temp_onnx = 'temp_model.onnx'
            export_keras_to_onnx(global_model.model, temp_onnx)
            os.replace(temp_onnx, onnx_path)

            # Archivage
            archive_path = f"history/model_step_{step_counter:06d}.onnx"
            shutil.copyfile(onnx_path, archive_path)
            print(f"✅ Modèle déployé. Archive sauvegardée : {archive_path}\n")