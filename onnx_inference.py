import numpy as np
import onnxruntime as ort


class Connect4ONNXInferenceModel:
    """
    Wrapper ultra-léger pour l'inférence ONNX Runtime sur CPU.
    Remplace avantageusement Connect4Model dans tous les workers parallèles.
    """

    def __init__(self, onnx_path: str):
        opts = ort.SessionOptions()
        # Configuration optimale monocœur : empêche les workers de se disputer les threads CPU
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, board: np.ndarray, current_player: int) -> tuple[np.ndarray, float]:
        # Encodage du plateau identique à Connect4Model.encode_board
        current_player_channel = (board == current_player).astype(np.float32)
        opponent_channel = (board == -current_player).astype(np.float32)
        encoded = np.stack([current_player_channel, opponent_channel], axis=-1)

        # Ajout de la dimension de batch (1, 6, 7, 2)
        input_tensor = np.expand_dims(encoded, axis=0)

        # Exécution de l'inférence via ONNX
        policy, value = self.session.run(None, {self.input_name: input_tensor})

        # Retourne (probabilités de politique (7,), valeur de l'état (float))
        return policy[0], float(value[0][0])
