import numpy as np

class Connect4Env:
    """
    Environnement pour le jeu de Puissance 4.
    Le plateau est représenté par une matrice NumPy de 6 lignes et 7 colonnes.
    Joueur 1 : 1
    Joueur 2 : -1
    Case vide : 0
    """

    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1

    def get_legal_moves(self):
        """Renvoie la liste des colonnes où un jeton peut être placé."""
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def step(self, col):
        """
        Place un jeton dans la colonne spécifiée pour le joueur actuel.
        Change ensuite de joueur.
        
        Args:
            col (int): Index de la colonne (0-6).
            
        Returns:
            tuple: (nouvel_état, gagnant)
                - gagnant: 1 ou -1 si victoire, 0 si match nul, None sinon.
        """
        if col not in self.get_legal_moves():
            raise ValueError(f"Action invalide : la colonne {col} est pleine ou hors limites.")

        # Trouver la ligne la plus basse disponible
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r, col] = self.current_player
                break

        winner = self.check_winner()
        
        # Changement de joueur
        self.current_player = -self.current_player
        
        return self.board.copy(), winner

    def check_winner(self):
        """
        Vérifie s'il y a un gagnant ou un match nul.
        
        Returns:
            int or None: 1 ou -1 (gagnant), 0 (match nul), None (partie en cours).
        """
        # Vérification horizontale
        for r in range(self.rows):
            for c in range(self.cols - 3):
                window = self.board[r, c:c+4]
                if abs(sum(window)) == 4:
                    return window[0]

        # Vérification verticale
        for r in range(self.rows - 3):
            for c in range(self.cols):
                window = self.board[r:r+4, c]
                if abs(sum(window)) == 4:
                    return window[0]

        # Diagonale descendante (\)
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = [self.board[r+i, c+i] for i in range(4)]
                if abs(sum(window)) == 4:
                    return window[0]

        # Diagonale ascendante (/)
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                window = [self.board[r-i, c+i] for i in range(4)]
                if abs(sum(window)) == 4:
                    return window[0]

        # Match nul (grille pleine)
        if len(self.get_legal_moves()) == 0:
            return 0

        # La partie continue
        return None

    def get_state(self):
        """Renvoie une copie de l'état actuel du plateau."""
        return self.board.copy()

    def reset(self):
        """Réinitialise l'environnement."""
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
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
    env.step(3) # Le joueur 1 joue au centre
    env.step(3) # Le joueur -1 joue au centre, par-dessus
    print(env.get_state())
    print(f"Coups légaux restants : {env.get_legal_moves()}")
    print(f"Gagnant actuel : {env.check_winner()}")