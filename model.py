import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
import tf2onnx


# Importation des classes de configuration
from config import ModelConfig, TrainConfig

class Connect4Model:
    """
    Modèle ResNet pour l'évaluation et la prédiction de coups au Puissance 4.
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig, input_shape=(6, 7, 2)):
        self.input_shape = input_shape
        self.model_config = model_config
        self.train_config = train_config
        
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.train_config.learning_rate)
        
        # Compilation du modèle avec les pertes appropriées pour chaque tête
        self.model.compile(
            optimizer=self.optimizer,
            loss={
                'policy_head': 'categorical_crossentropy',
                'value_head': 'mean_squared_error'
            },
            loss_weights={
                'policy_head': 1.0,
                'value_head': 0.5  # Divise l'impact des erreurs d'évaluation par 2
            },
            metrics={
                'policy_head': ['categorical_accuracy'],
                'value_head': ['mae']
            }
        )

    @staticmethod
    def encode_board(board: np.ndarray, current_player: int) -> np.ndarray:
        """
        Encode le plateau pour le réseau de neurones.
        
        Args:
            board: Matrice (6, 7) avec 1, -1 ou 0.
            current_player: Le joueur dont c'est le tour (1 ou -1).
            
        Returns:
            np.ndarray: Tenseur de forme (6, 7, 2).
        """
        # Canal 0: Jetons du joueur actuel
        # Canal 1: Jetons de l'adversaire
        current_player_channel = (board == current_player).astype(np.float32)
        opponent_channel = (board == -current_player).astype(np.float32)
        
        return np.stack([current_player_channel, opponent_channel], axis=-1)

    def _residual_block(self, x, filters: int):
        """Un bloc résiduel standard avec régularisation L2."""
        l2 = regularizers.l2(self.train_config.l2_reg)
        shortcut = x

        x = layers.Conv2D(filters, kernel_size=3, padding='same', kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters, kernel_size=3, padding='same', kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def _build_model(self) -> models.Model:
        """Construit l'architecture ResNet avec contraintes L2."""
        l2 = regularizers.l2(self.train_config.l2_reg)
        inputs = Input(shape=self.input_shape)

        x = layers.Conv2D(self.model_config.filters, kernel_size=3, padding='same', kernel_regularizer=l2)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        for _ in range(self.model_config.blocks):
            x = self._residual_block(x, self.model_config.filters)

        # Tête de Politique
        policy_net = layers.Conv2D(2, kernel_size=1, padding='same', kernel_regularizer=l2)(x)
        policy_net = layers.BatchNormalization()(policy_net)
        policy_net = layers.Activation('relu')(policy_net)
        policy_net = layers.Flatten()(policy_net)
        policy_head = layers.Dense(7, activation='softmax', name='policy_head', kernel_regularizer=l2)(policy_net)

        # ==========================================
        # 2. TÊTE DE VALEUR (Value Head) - Façon KataGo
        # Utilise le résumé spatial pour comprendre les menaces globales
        # ==========================================
        v = layers.Conv2D(filters=32, kernel_size=1, padding='same', use_bias=False)(x)
        v = layers.BatchNormalization()(v)
        v = layers.Activation('relu')(v)

        # --- NOUVEAU : L'extraction de l'essence spatiale ---
        # L'Alarme : Y a-t-il un signal fort quelque part ?
        v_max = layers.GlobalMaxPooling2D()(v)

        # Le Thermomètre : Quelle est la température stratégique globale ?
        v_avg = layers.GlobalAveragePooling2D()(v)

        # On fusionne les deux résumés (32 + 32 = vecteur de 64 neurones)
        v_pool = layers.Concatenate()([v_max, v_avg])
        # ----------------------------------------------------

        # On connecte ce résumé à notre réseau dense final
        # Note : On a besoin de beaucoup moins de neurones ici (ex: 64 suffisent amplement)
        v_dense = layers.Dense(64, activation='relu')(v_pool)

        # La sortie finale entre -1 et 1
        value_head = layers.Dense(1, activation='tanh', name='value_head')(v_dense)

        m = models.Model(inputs=inputs, outputs=[policy_head, value_head], name="Connect4ResNet")

        return m

    def summary(self):
        self.model.summary()

    def predict(self, board: np.ndarray, current_player: int) -> tuple[np.ndarray, float]:
        """
        Réalise une prédiction ultra-rapide sur un état de jeu.
        
        Returns:
            tuple: (probalités_politique (7,), valeur_état (float))
        """
        encoded = self.encode_board(board, current_player)
        # Ajout de la dimension de batch (1, 6, 7, 2)
        input_tensor = np.expand_dims(encoded, axis=0)
        
        # APPEL DIRECT (10x à 50x plus rapide que .predict() dans une boucle MCTS)
        policy, value = self.model(input_tensor, training=False)
        
        # Les sorties sont des tenseurs TensorFlow, on extrait les valeurs avec .numpy()
        return policy[0].numpy(), float(value[0][0].numpy())



def export_keras_to_onnx(keras_model, output_path: str):
    """Convertit l'architecture fonctionnelle Keras en fichier .onnx figé."""
    spec = (tf.TensorSpec((None, 6, 7, 2), tf.float32, name="input"),)

    print(f"📦 [ONNX] Conversion du modèle Keras vers {output_path}...")
    tf2onnx.convert.from_keras(keras_model, input_signature=spec, output_path=output_path)

if __name__ == "__main__":
    # Test de validation local en instantiant les configurations par défaut
    m_config = ModelConfig()
    t_config = TrainConfig()
    
    model = Connect4Model(model_config=m_config, train_config=t_config)
    dummy_board = np.zeros((6, 7))
    p, v = model.predict(dummy_board, 1)
    print(f"Policy: {p}")
    print(f"Value: {v}")
    model.model.summary()