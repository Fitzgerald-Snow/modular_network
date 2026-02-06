"""
Direct Krotov-Hopfield Dense Associative Memory for continuous patterns.

This implementation bypasses SNMF clustering and directly injects patterns
into the Krotov-Hopfield network for storage and retrieval.

Based on: "Large Associative Memory Problem in Neurobiology and Machine Learning"
by Krotov & Hopfield (2021), arXiv:2008.06996

Key equation (softmax dynamics):
    x_new = Ξ^T softmax(β * Ξ x)
where Ξ is the matrix of stored patterns and β is inverse temperature.
"""

import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp
import numpy as np


class KrotovHopfield:
    """
    Dense Associative Memory (Modern Hopfield Network) from Krotov & Hopfield (2021).

    Uses softmax update rule for pattern retrieval:
        x_new = Ξ^T softmax(β * Ξ x)

    where Ξ is the matrix of stored patterns and β is inverse temperature.
    """
    def __init__(self, n_neurons, beta=1.0, decay=0.99, max_patterns=100):
        """
        Parameters
        ----------
        n_neurons : int
            Dimension of each pattern
        beta : float
            Inverse temperature (higher = sharper softmax, more like winner-take-all)
        decay : float
            Decay rate for online learning when at capacity
        max_patterns : int
            Maximum number of patterns that can be stored
        """
        self.n_neurons = n_neurons
        self.beta = beta
        self.decay = decay
        self.max_patterns = max_patterns

        # Store patterns explicitly
        # Shape: (max_patterns, n_neurons)
        self.patterns = np.zeros((max_patterns, n_neurons))
        self.pattern_count = 0

    def store(self, x):
        """
        Store a new pattern in the network.

        If at capacity, uses online learning to update the most similar pattern.

        Parameters
        ----------
        x : ndarray
            Pattern to store, shape (n_neurons,)
        """
        if self.pattern_count < self.max_patterns:
            # Add new pattern
            self.patterns[self.pattern_count] = x
            self.pattern_count += 1
        else:
            # Online learning: update the most similar existing pattern
            similarities = self.patterns[:self.pattern_count] @ x
            most_similar_idx = np.argmax(similarities)
            self.patterns[most_similar_idx] = (
                self.decay * self.patterns[most_similar_idx] +
                (1 - self.decay) * x
            )

    def retrieve(self, x, n_iterations=10):
        """
        Retrieve a pattern using the softmax update rule (Modern Hopfield dynamics).

        Update rule: x_new = Ξ^T softmax(β * Ξ x)

        Parameters
        ----------
        x : ndarray
            Query pattern (possibly noisy), shape (n_neurons,)
        n_iterations : int
            Number of retrieval iterations

        Returns
        -------
        ndarray
            Retrieved pattern, shape (n_neurons,)
        """
        if self.pattern_count == 0:
            return x  # No patterns stored, return input

        # Get stored patterns
        Xi = self.patterns[:self.pattern_count]  # Shape: (count, n_neurons)

        for _ in range(n_iterations):
            # Compute similarities: Ξ x
            similarities = Xi @ x  # Shape: (count,)

            # Softmax with inverse temperature β
            # Using logsumexp for numerical stability
            log_weights = self.beta * similarities
            log_normalizer = logsumexp(log_weights)
            weights = np.exp(log_weights - log_normalizer)  # Shape: (count,)

            # Update: x_new = Ξ^T softmax(β * Ξ x)
            x = Xi.T @ weights  # Shape: (n_neurons,)

        return x

    def reset(self):
        """Clear all stored patterns."""
        self.patterns = np.zeros((self.max_patterns, self.n_neurons))
        self.pattern_count = 0

    def get_pattern_count(self):
        """Return the number of stored patterns."""
        return self.pattern_count


class DirectKrotovNetwork:
    """
    A direct pattern classification and retrieval network using Krotov-Hopfield.

    Unlike TreeNetworkKrotov, this network:
    - Does NOT use SNMF clustering
    - Directly stores and retrieves patterns from Krotov-Hopfield
    - Can optionally use multiple Krotov-Hopfield modules for different classes

    This is useful when:
    - You want a flat (non-hierarchical) associative memory
    - Class labels are known beforehand
    - You want to bypass the clustering step entirely
    """
    def __init__(self, n_neurons, n_classes=1, beta=1.0, decay=0.99, max_patterns_per_class=100):
        """
        Parameters
        ----------
        n_neurons : int
            Dimension of each pattern
        n_classes : int
            Number of classes (each class gets its own Krotov-Hopfield module)
            Use n_classes=1 for a single associative memory
        beta : float
            Inverse temperature for softmax
        decay : float
            Decay rate for online learning
        max_patterns_per_class : int
            Maximum patterns per class
        """
        self.n_neurons = n_neurons
        self.n_classes = n_classes
        self.beta = beta
        self.decay = decay
        self.max_patterns_per_class = max_patterns_per_class

        # Create one Krotov-Hopfield network per class
        self.hopfields = [
            KrotovHopfield(n_neurons, beta, decay, max_patterns_per_class)
            for _ in range(n_classes)
        ]

    def store(self, x, class_idx=0):
        """
        Store a pattern in the specified class.

        Parameters
        ----------
        x : ndarray
            Pattern to store, shape (n_neurons,)
        class_idx : int
            Class index (0 to n_classes-1)
        """
        if class_idx < 0 or class_idx >= self.n_classes:
            raise ValueError(f"class_idx must be between 0 and {self.n_classes - 1}")
        self.hopfields[class_idx].store(x)

    def retrieve(self, x, class_idx=0, n_iterations=10):
        """
        Retrieve a pattern from the specified class.

        Parameters
        ----------
        x : ndarray
            Query pattern, shape (n_neurons,)
        class_idx : int
            Class index
        n_iterations : int
            Number of retrieval iterations

        Returns
        -------
        ndarray
            Retrieved pattern, shape (n_neurons,)
        """
        if class_idx < 0 or class_idx >= self.n_classes:
            raise ValueError(f"class_idx must be between 0 and {self.n_classes - 1}")
        return self.hopfields[class_idx].retrieve(x, n_iterations)

    def classify_and_retrieve(self, x, n_iterations=10):
        """
        Find the best matching class and retrieve the pattern.

        Classification is based on which class's retrieved pattern
        has the highest similarity (dot product) with the query.

        Parameters
        ----------
        x : ndarray
            Query pattern, shape (n_neurons,)
        n_iterations : int
            Number of retrieval iterations

        Returns
        -------
        tuple
            (class_idx, retrieved_pattern)
        """
        best_class = 0
        best_similarity = -np.inf
        best_retrieved = x

        for class_idx in range(self.n_classes):
            if self.hopfields[class_idx].pattern_count == 0:
                continue
            retrieved = self.hopfields[class_idx].retrieve(x, n_iterations)
            similarity = np.dot(x, retrieved)
            if similarity > best_similarity:
                best_similarity = similarity
                best_class = class_idx
                best_retrieved = retrieved

        return best_class, best_retrieved

    def reset(self, class_idx=None):
        """
        Reset stored patterns.

        Parameters
        ----------
        class_idx : int or None
            If None, reset all classes. Otherwise, reset only the specified class.
        """
        if class_idx is None:
            for h in self.hopfields:
                h.reset()
        else:
            if class_idx < 0 or class_idx >= self.n_classes:
                raise ValueError(f"class_idx must be between 0 and {self.n_classes - 1}")
            self.hopfields[class_idx].reset()

    def get_total_pattern_count(self):
        """Return total number of stored patterns across all classes."""
        return sum(h.pattern_count for h in self.hopfields)

    def __repr__(self):
        total = self.get_total_pattern_count()
        return (f"DirectKrotovNetwork(n_neurons={self.n_neurons}, "
                f"n_classes={self.n_classes}, patterns_stored={total})")


class UltraMetricTreeContinuous:
    """
    Generates hierarchical continuous patterns organized in an ultrametric tree.

    Children are created by adding isotropically chosen random unit vectors
    scaled by a fixed norm. The norm halves at each level: 1, 0.5, 0.25, ...
    """
    def __init__(self, key, n_neurons, tree_struct_list, base_norm=1.0):
        """
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for JAX
        n_neurons : int
            Dimension of patterns
        tree_struct_list : tuple
            Structure of the tree, e.g., (4, 4, 4) means 4 branches at each level
        base_norm : float
            Norm of displacement vectors at the first level
            Subsequent levels use base_norm/2, base_norm/4, ...
        """
        self.key, subkey = jax.random.split(key)
        self.tree_struct_list = tree_struct_list
        self.base_norm = base_norm
        self.n_neurons = n_neurons
        # Root ancestor is the zero vector
        self.root_ancestor = jnp.zeros((n_neurons,))
        self.descendents = [self.root_ancestor]
        self.keys = [subkey]

    def get_direct_descendents(self, ancestor, key, n_patterns_per_cluster, fixed_norm):
        """
        Generate descendants by adding isotropic random vectors of fixed norm.
        """
        n_neurons = len(ancestor)
        random_vectors = jax.random.normal(key, (n_patterns_per_cluster, n_neurons))
        norms = jnp.linalg.norm(random_vectors, axis=1, keepdims=True)
        unit_vectors = random_vectors / norms
        displacement_vectors = unit_vectors * fixed_norm
        return ancestor + displacement_vectors

    def get_next_tree(self, pattern_tree, key_tree, n_patterns_per_cluster, fixed_norm):
        def fun(key, ancestor):
            descendents = self.get_direct_descendents(ancestor, key, n_patterns_per_cluster, fixed_norm)
            return [descendents[i] for i in range(len(descendents))]

        self.descendents = jax.tree.map(fun, self.keys, self.descendents)
        self.keys = jax.tree.map(
            lambda x: list(jax.random.split(x, n_patterns_per_cluster)),
            self.keys
        )

    def construct_tree(self):
        """Construct the full tree of continuous patterns."""
        for i in range(len(self.tree_struct_list)):
            n_patterns_per_cluster = self.tree_struct_list[i]
            fixed_norm = self.base_norm / (2 ** i)
            self.get_next_tree(self.descendents, self.keys, n_patterns_per_cluster, fixed_norm)


# Utility functions

def add_gaussian_noise(pattern, key, noise_std=0.1):
    """Add Gaussian noise to a continuous pattern."""
    noise = jax.random.normal(key, pattern.shape) * noise_std
    return pattern + noise


def compute_correlation(pattern1, pattern2):
    """Compute normalized correlation between two continuous patterns."""
    norm1 = jnp.linalg.norm(pattern1)
    norm2 = jnp.linalg.norm(pattern2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return jnp.dot(pattern1, pattern2) / (norm1 * norm2)


def compute_mse(pattern1, pattern2):
    """Compute mean squared error between two patterns."""
    return jnp.mean((pattern1 - pattern2) ** 2)


# Convenience function for flat pattern list from tree
def flatten_tree(tree):
    """Flatten a nested list/tree structure into a flat list of patterns."""
    patterns = []
    def recurse(node):
        if isinstance(node, list):
            for item in node:
                recurse(item)
        else:
            patterns.append(node)
    recurse(tree)
    return patterns
