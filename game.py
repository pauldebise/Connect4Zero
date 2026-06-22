import numpy as np
from numba import njit


@njit(cache=True)
def check_win_fast(bitboard):
    """
    Vérifie la victoire avec des opérations bit-à-bit.
    Exécuté à la vitesse du C grâce à Numba.
    """
    # Vérification horizontale (décalage de 7 bits vers la droite)
    m = bitboard & (bitboard >> np.uint64(7))
    if m & (m >> np.uint64(14)):
        return True

    # Vérification diagonale descendante \ (décalage de 6 bits)
    m = bitboard & (bitboard >> np.uint64(6))
    if m & (m >> np.uint64(12)):
        return True

    # Vérification diagonale ascendante / (décalage de 8 bits)
    m = bitboard & (bitboard >> np.uint64(8))
    if m & (m >> np.uint64(16)):
        return True

    # Vérification verticale (décalage de 1 bit vers le haut)
    m = bitboard & (bitboard >> np.uint64(1))
    if m & (m >> np.uint64(2)):
        return True

    return False


class Connect4Env:
    """
    Environnement pour le jeu de Puissance 4 optimisé (Bitboards + NumPy hybride).
    Joueur 1 : 1
    Joueur 2 : -1
    Case vide : 0
    """

    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.reset()

    def reset(self):
        """Réinitialise l'environnement."""
        # --- État NumPy (pour la compatibilité) ---
        self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.current_player = 1

        # --- État Bitboard (pour la performance) ---
        self.p1_board = np.uint64(0)
        self.p2_board = np.uint64(0)
        self.moves_played = 0

        # Chaque colonne a une hauteur de base (0, 7, 14, 21, 28, 35, 42)
        # La ligne "6" de chaque colonne sert de tampon pour éviter les débordements
        self.heights = np.array([0, 7, 14, 21, 28, 35, 42], dtype=np.uint64)

        return self.board.copy()

    def get_legal_moves(self):
        """Renvoie la liste des colonnes où un jeton peut être placé."""
        # Une colonne 'c' est jouable si sa hauteur actuelle ne touche pas la ligne tampon (c*7 + 5 est le max jouable)
        return [c for c in range(self.cols) if self.heights[c] < np.uint64(c * 7 + 6)]

    def step(self, col):
        """
        Place un jeton et renvoie (nouvel_état, gagnant).
        """
        # (Optionnel en RL pour gagner du temps : tu peux retirer cette vérification
        # si ton agent masque déjà les actions illégales)
        if col not in self.get_legal_moves():
            raise ValueError(f"Action invalide : la colonne {col} est pleine ou hors limites.")

        # 1. Mise à jour des Bitboards
        move = np.uint64(1) << self.heights[col]

        if self.current_player == 1:
            self.p1_board |= move
        else:
            self.p2_board |= move

        # 2. Mise à jour ciblée du tableau NumPy (O(1) au lieu de boucler)
        # On calcule la ligne NumPy (5 = bas, 0 = haut) correspondant à l'index binaire
        row_numpy = 5 - int((self.heights[col] % np.uint64(7)))
        self.board[row_numpy, col] = self.current_player

        # 3. Incrémenter la hauteur et le compteur de coups
        self.heights[col] += np.uint64(1)
        self.moves_played += 1

        # 4. Vérification du gagnant ultra-rapide
        winner = self.check_winner()

        # 5. Changement de joueur
        self.current_player = -self.current_player

        return self.board.copy(), winner

    def check_winner(self):
        """
        Vérifie s'il y a un gagnant ou un match nul via le Bitboard.
        """
        if check_win_fast(self.p1_board):
            return 1
        if check_win_fast(self.p2_board):
            return -1

        if self.moves_played == 42:
            return 0

        return None

    def get_state(self):
        """Renvoie une copie de l'état actuel du plateau."""
        return self.board.copy()

    def render(self):
        """Affiche le plateau dans la console."""
        symbols = {1: "X", -1: "O", 0: "."}
        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))
        print("-" * (self.cols * 2 - 1))
        print(" ".join(map(str, range(self.cols))))


if __name__ == "__main__":
    env = Connect4Env()
    env.step(3)  # Joueur 1 (X)
    env.step(3)  # Joueur -1 (O)

    print("État initialisé (Format NumPy intact) :")
    print(env.get_state())
    print(f"\nCoups légaux restants : {env.get_legal_moves()}")
    print(f"Gagnant actuel : {env.check_winner()}")