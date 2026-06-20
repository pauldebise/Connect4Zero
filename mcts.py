import math
import copy
import numpy as np

# Importation de la configuration du MCTS
from config import MCTSConfig

class Node:
    """
    Représente un nœud dans l'arbre de recherche de Monte Carlo (MCTS).
    """
    def __init__(self, prior: float, parent=None):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.value_score = 0.0  # Q = W/N
        self.parent = parent
        self.children = {}  # dictionnaire {action: Node}

    def is_expanded(self) -> bool:
        """Renvoie vrai si le nœud possède des enfants (déjà étendu)."""
        return len(self.children) > 0

    # Dans mcts.py, classe Node

    def get_puct(self, c_puct: float, fpu_reduction: float = 0.0) -> float:
        """
        Calcule et renvoie la valeur PUCT avec application du First Play Urgency (FPU).
        """
        if self.parent is None:
            return 0.0

        # max(1, ...) empêche le terme d'exploration d'être nul au début
        u = c_puct * self.prior * math.sqrt(max(1, self.parent.visit_count)) / (1 + self.visit_count)

        # --- NOUVEAU : Logique FPU ---
        if self.visit_count == 0:
            # Le nœud est vierge : on hérite de la valeur du parent (moins le doute)
            # Pas besoin d'inverser le signe, le parent a la même perspective que le joueur qui choisit
            base_value = self.parent.value_score - fpu_reduction
        else:
            # Le nœud a été évalué : on utilise sa vraie valeur
            # (On inverse car value_score est du point de vue de l'adversaire)
            base_value = -self.value_score

        return base_value + u


class MCTS:
    """
    Orchestre la recherche Monte Carlo (Monte Carlo Tree Search) guidée
    par un réseau de neurones (façon AlphaZero).
    """
    def __init__(self, model, mcts_config: MCTSConfig):
        """
        Args:
            model: Instance du réseau de neurones (Connect4Model) pour l'évaluation.
            mcts_config: Instance de MCTSConfig contenant les hyperparamètres de recherche.
        """
        self.model = model
        self.mcts_config = mcts_config
        self.root = None

    def run(self, env, add_dirichlet_noise: bool = False):
        """
        Exécute un nombre donné de simulations MCTS à partir de l'état actuel de l'environnement.
        Les paramètres numériques sont extraits de mcts_config.
        
        Args:
            env: L'environnement de jeu actuel (instance de Connect4Env).
            add_dirichlet_noise: Booléen pour ajouter du bruit de Dirichlet à la racine (Self-Play).
        """
        self.root = Node(prior=0.0)
        
        # --- Évaluation initiale de la racine ---
        policy, _ = self.model.predict(env.board, env.current_player)
        legal_moves = env.get_legal_moves()
        
        # Masquage des coups illégaux
        mask = np.zeros(env.cols)
        mask[legal_moves] = 1
        policy = policy * mask
        
        # Renormalisation
        sum_policy = np.sum(policy)
        if sum_policy > 0:
            policy /= sum_policy
        else:
            policy = mask / np.sum(mask)
            
        # Ajout du bruit de Dirichlet à la racine si demandé (Self-Play)
        if add_dirichlet_noise and len(legal_moves) > 0:
            noise = np.random.dirichlet([self.mcts_config.dirichlet_alpha] * len(legal_moves))
            for i, action in enumerate(legal_moves):
                policy[action] = ((1 - self.mcts_config.exploration_fraction) * policy[action] + 
                                  self.mcts_config.exploration_fraction * noise[i])
                
        # Création des nœuds enfants de la racine
        for action in legal_moves:
            self.root.children[action] = Node(prior=policy[action], parent=self.root)
            
        # --- Boucle des simulations MCTS ---
        for _ in range(self.mcts_config.num_simulations):
            node = self.root
            sim_env = copy.deepcopy(env)

            # 1. Sélection
            while node.is_expanded():
                action, node = max(
                    node.children.items(),
                    key=lambda item: item[1].get_puct(self.mcts_config.c_puct, self.mcts_config.fpu_reduction)
                )
                _, winner = sim_env.step(action)
                
            # 2. Expansion & Évaluation
            if winner is None:
                # État non terminal : évaluation par le réseau
                policy, value = self.model.predict(sim_env.board, sim_env.current_player)
                legal_moves = sim_env.get_legal_moves()
                
                # Masquage et renormalisation
                mask = np.zeros(sim_env.cols)
                mask[legal_moves] = 1
                policy = policy * mask
                
                sum_policy = np.sum(policy)
                if sum_policy > 0:
                    policy /= sum_policy
                else:
                    policy = mask / np.sum(mask)
                    
                # Expansion : ajout des enfants
                for action in legal_moves:
                    node.children[action] = Node(prior=policy[action], parent=node)
            else:
                # État terminal
                if winner == 0:
                    value = 0.0  # Match nul
                else:
                    value = -1.0 # Défaite du point de vue du joueur qui vient de basculer
                    
            # 3. Rétropropagation (Backup)
            current_node = node
            current_value = value
            
            while current_node is not None:
                current_node.visit_count += 1
                current_node.total_value += current_value
                current_node.value_score = current_node.total_value / current_node.visit_count
                
                # À chaque niveau de l'arbre, le point de vue change de joueur
                current_value = -current_value
                current_node = current_node.parent

        # Dans mcts.py, à l'intérieur de la classe MCTS
    def get_root_value(self) -> float:
        """Renvoie l'évaluation moyenne de la position actuelle selon le MCTS."""
        if self.root is None:
            return 0.0
        # On retourne le value_score de la racine (l'agrégation des simulations)
        return self.root.value_score

    def get_visit_counts(self, env) -> np.ndarray:
        """Renvoie le nombre brut de visites pour chaque action (utile pour le debug)."""
        if self.root is None:
            raise RuntimeError("La méthode run() doit être appelée avant get_visit_counts().")
            
        action_visits = np.zeros(env.cols)
        for action, child in self.root.children.items():
            action_visits[action] = child.visit_count
        return action_visits

    def get_action_probs(self, env, temperature: float = 1.0) -> np.ndarray:
        """
        Calcule et renvoie un vecteur de probabilités basé sur les comptes de visites.
        """
        if self.root is None:
            raise RuntimeError("La méthode run() doit être appelée avant get_action_probs().")
            
        action_visits = np.zeros(env.cols)
        for action, child in self.root.children.items():
            action_visits[action] = child.visit_count
            
        if temperature == 0:
            best_action = np.argmax(action_visits)
            probs = np.zeros(env.cols)
            probs[best_action] = 1.0
            return probs
            
        visits_temp = action_visits ** (1.0 / temperature)
        sum_visits = np.sum(visits_temp)
        
        if sum_visits == 0:
            return np.ones(env.cols) / env.cols
            
        probs = visits_temp / sum_visits
        return probs