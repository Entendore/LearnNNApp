import sys
import math
import random
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QStackedWidget, QProgressBar,
    QFrame, QSizePolicy, QRadioButton, QButtonGroup, QDialog,
    QGridLayout, QSpacerItem, QGraphicsDropShadowEffect, QCheckBox
)
from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QSize, QRectF, QPointF, Signal
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QLinearGradient,
    QRadialGradient, QPainterPath, QIcon, QPalette, QPixmap, QPolygonF
)

# ============================================================
# STYLESHEET
# ============================================================
STYLESHEET = """
QMainWindow { background-color: #0d0d1a; }
QWidget { color: #e0e0f0; font-family: 'Segoe UI', 'Helvetica Neue', Arial; }
QScrollBar:vertical {
    background: #16162a; width: 8px; border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #3a3a5c; border-radius: 4px; min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::horizontal {
    background: #16162a; height: 8px; border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #3a3a5c; border-radius: 4px; min-width: 30px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QProgressBar {
    background-color: #1e1e38; border-radius: 6px; height: 8px; text-align: center;
    color: transparent; border: none;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #6c63ff, stop:1 #00d4aa);
    border-radius: 6px;
}
QToolTip {
    background-color: #1a1a32;
    color: #e0e0f0;
    border: 1px solid #3a3a5c;
    border-radius: 6px;
    padding: 6px;
    font-size: 12px;
}
"""

# ============================================================
# NEURAL NETWORK DATA
# ============================================================
NN_TYPES = [
    {
        "id": "perceptron",
        "name": "Perceptron",
        "color": "#FF6B6B",
        "icon": "◉",
        "tagline": "The building block of neural networks",
        "summary": "The Perceptron, invented by Frank Rosenblatt in 1958, is the simplest form of a neural network. It's a single-layer binary classifier that maps input features to an output decision using a linear combination of weights and a step activation function.",
        "architecture": "Input values → Weighted Sum → Activation Function → Output\n\nThe perceptron computes: y = f(Σ(wᵢxᵢ) + b)\nwhere f is a step function (Heaviside), wᵢ are weights, xᵢ are inputs, and b is bias.",
        "layers": [3, 1],
        "special": "perceptron",
        "use_cases": [
            "Simple binary classification tasks",
            "Logic gate implementations (AND, OR)",
            "Foundation for understanding complex networks",
            "Linear decision boundary problems"
        ],
        "pros": [
            "Simple and easy to understand",
            "Guaranteed to converge for linearly separable data",
            "Fast training and inference",
            "Foundation of deep learning theory"
        ],
        "cons": [
            "Cannot solve XOR problem (non-linearly separable)",
            "Only handles binary classification",
            "Limited to linear decision boundaries",
            "Single layer — no hidden representations"
        ],
        "key_equation": "y = step(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)",
        "quiz": [
            {"q": "Who invented the Perceptron?", "options": ["Frank Rosenblatt", "Geoffrey Hinton", "Yann LeCun", "Andrew Ng"], "answer": 0},
            {"q": "What type of decision boundary can a Perceptron learn?", "options": ["Non-linear", "Linear", "Circular", "Arbitrary"], "answer": 1},
            {"q": "Which problem cannot be solved by a single Perceptron?", "options": ["AND gate", "OR gate", "XOR gate", "NOT gate"], "answer": 2},
        ]
    },
    {
        "id": "fnn",
        "name": "Feedforward Neural Network",
        "color": "#FFA94D",
        "icon": "⬡",
        "tagline": "The classic multi-layer architecture",
        "summary": "A Feedforward Neural Network (FNN), also called a Multi-Layer Perceptron (MLP), is the foundational deep network architecture. Information flows in one direction — from input, through hidden layers, to output. With hidden layers and non-linear activation functions, FNNs can approximate any continuous function.",
        "architecture": "Input Layer → Hidden Layer(s) → Output Layer\n\nEach neuron computes: a = σ(Wx + b)\nwhere σ is a non-linear activation (ReLU, sigmoid, tanh).\nBackpropagation + gradient descent are used for training.",
        "layers": [3, 5, 4, 2],
        "special": None,
        "use_cases": [
            "Tabular data classification & regression",
            "Function approximation (Universal Approximation Theorem)",
            "Feature extraction pipelines",
            "Recommendation system components"
        ],
        "pros": [
            "Can approximate any continuous function",
            "Flexible architecture — adjustable depth/width",
            "Well-understood training via backpropagation",
            "Works well with structured/tabular data"
        ],
        "cons": [
            "No spatial awareness — not ideal for images",
            "No temporal awareness — not ideal for sequences",
            "Prone to overfitting on small datasets",
            "Requires careful hyperparameter tuning"
        ],
        "key_equation": "hₗ = σ(Wₗ · hₗ₋₁ + bₗ)",
        "quiz": [
            {"q": "What is another name for a Feedforward Neural Network?", "options": ["Recurrent Network", "Multi-Layer Perceptron", "Convolutional Network", "Autoencoder"], "answer": 1},
            {"q": "What theorem states that an FNN can approximate any continuous function?", "options": ["Bayes Theorem", "Central Limit Theorem", "Universal Approximation Theorem", "No Free Lunch Theorem"], "answer": 2},
            {"q": "Which training algorithm is commonly used for FNNs?", "options": ["K-Means", "Backpropagation", "Q-Learning", "Genetic Algorithm"], "answer": 1},
        ]
    },
    {
        "id": "cnn",
        "name": "Convolutional Neural Network",
        "color": "#FFD93D",
        "icon": "▦",
        "tagline": "The vision specialist",
        "summary": "CNNs revolutionized computer vision by exploiting spatial structure in data. Using convolutional filters that slide across input, they automatically learn hierarchical features — from edges and textures to complex objects. They parameter-share across spatial positions, making them far more efficient than fully-connected networks for image data.",
        "architecture": "Input → [Conv → ReLU → Pool] × N → Flatten → FC → Output\n\nConvolution: extract local features\nPooling: reduce spatial dimensions\nFully Connected: final classification",
        "layers": [4, 6, 8, 4, 2],
        "special": "cnn",
        "use_cases": [
            "Image classification & object detection",
            "Medical image analysis (X-rays, MRIs)",
            "Autonomous vehicle perception",
            "Face recognition & biometrics"
        ],
        "pros": [
            "Translation invariance through pooling",
            "Parameter sharing — much fewer parameters than FC",
            "Hierarchical feature learning",
            "State-of-the-art on vision tasks"
        ],
        "cons": [
            "Loses spatial precision due to pooling",
            "Requires large labeled datasets",
            "Computationally expensive training",
            "Fixed receptive field size"
        ],
        "key_equation": "(f * g)[i,j] = ΣΣ f[m,n] · g[i-m, j-n]",
        "quiz": [
            {"q": "What operation is the core of a CNN?", "options": ["Matrix Multiplication", "Convolution", "Recurrence", "Attention"], "answer": 1},
            {"q": "What is the purpose of pooling layers in CNNs?", "options": ["Add more parameters", "Reduce spatial dimensions", "Increase resolution", "Add non-linearity only"], "answer": 1},
            {"q": "Which CNN architecture won the ImageNet 2012 competition?", "options": ["VGGNet", "AlexNet", "ResNet", "GoogLeNet"], "answer": 1},
        ]
    },
    {
        "id": "rnn",
        "name": "Recurrent Neural Network",
        "color": "#6BCB77",
        "icon": "↻",
        "tagline": "Memory for sequential data",
        "summary": "RNNs process sequential data by maintaining a hidden state that acts as memory, carrying information from previous time steps. At each step, the RNN combines the current input with the previous hidden state, enabling it to model temporal dependencies in data like text, speech, and time series.",
        "architecture": "hₜ = tanh(Wₕh · hₜ₋₁ + Wₓh · xₜ + b)\nyₜ = Wₕy · hₜ\n\nThe hidden state hₜ is the 'memory' that gets updated at each timestep.",
        "layers": [3, 4, 2],
        "special": "rnn",
        "use_cases": [
            "Language modeling & text generation",
            "Speech recognition",
            "Time series forecasting",
            "Music generation"
        ],
        "pros": [
            "Handles variable-length sequences",
            "Shares parameters across time steps",
            "Maintains temporal context",
            "Can process data of any length"
        ],
        "cons": [
            "Vanishing gradient problem for long sequences",
            "Exploding gradient problem",
            "Sequential processing — cannot parallelize",
            "Difficulty with long-term dependencies"
        ],
        "key_equation": "hₜ = tanh(Wₕh · hₜ₋₁ + Wₓh · xₜ + b)",
        "quiz": [
            {"q": "What major problem do basic RNNs suffer from?", "options": ["Overfitting", "Vanishing gradients", "Underfitting", "No activation function"], "answer": 1},
            {"q": "What does the hidden state in an RNN represent?", "options": ["The output", "The weights", "Memory of past inputs", "The loss function"], "answer": 2},
            {"q": "Can RNNs be easily parallelized during training?", "options": ["Yes, fully", "No, due to sequential dependencies", "Only on GPUs", "Only with batch norm"], "answer": 1},
        ]
    },
    {
        "id": "lstm",
        "name": "LSTM",
        "color": "#4D96FF",
        "icon": "⊞",
        "tagline": "Long-term memory solved",
        "summary": "Long Short-Term Memory networks, introduced by Hochreiter & Schmidhuber in 1997, solve the vanishing gradient problem of standard RNNs through a gating mechanism. Three gates — forget, input, and output — control information flow, allowing the network to learn when to remember and when to forget, enabling long-term dependency modeling.",
        "architecture": "Forget Gate: fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)\nInput Gate: iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)\nCandidate: C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)\nCell State: Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ\nOutput Gate: oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)\nHidden: hₜ = oₜ⊙tanh(Cₜ)",
        "layers": [3, 5, 2],
        "special": "lstm",
        "use_cases": [
            "Machine translation",
            "Sentiment analysis on long texts",
            "Video activity recognition",
            "Drug molecule design"
        ],
        "pros": [
            "Solves vanishing gradient problem",
            "Captures long-term dependencies",
            "Learned gating — when to remember/forget",
            "Proven track record on sequence tasks"
        ],
        "cons": [
            "More complex and slower than simple RNN",
            "Still sequential — cannot fully parallelize",
            "Many parameters — risk of overfitting",
            "Memory intensive for very long sequences"
        ],
        "key_equation": "Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ",
        "quiz": [
            {"q": "How many gates does an LSTM have?", "options": ["1", "2", "3", "4"], "answer": 2},
            {"q": "What does the forget gate control?", "options": ["New input", "What to discard from cell state", "The output", "The learning rate"], "answer": 1},
            {"q": "Who introduced the LSTM?", "options": ["Hinton & LeCun", "Hochreiter & Schmidhuber", "Goodfellow & Bengio", "Vaswani et al."], "answer": 1},
        ]
    },
    {
        "id": "gan",
        "name": "GAN",
        "color": "#9B59B6",
        "icon": "⚔",
        "tagline": "Adversarial creativity",
        "summary": "Generative Adversarial Networks, proposed by Ian Goodfellow in 2014, consist of two neural networks locked in a game: a Generator that creates fake data, and a Discriminator that tries to distinguish real from fake. Through this adversarial process, the Generator learns to produce increasingly realistic data — from images to music to text.",
        "architecture": "Generator: G(z) → fake data (from noise z)\nDiscriminator: D(x) → real/fake probability\n\nMin-Max Game:\nmin_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]",
        "layers": [3, 4, 3],
        "special": "gan",
        "use_cases": [
            "Photorealistic image generation",
            "Style transfer & art creation",
            "Data augmentation",
            "Super-resolution imaging"
        ],
        "pros": [
            "Generates highly realistic samples",
            "No explicit density estimation needed",
            "Learns complex data distributions",
            "Has spawned many variants (DCGAN, StyleGAN, CycleGAN)"
        ],
        "cons": [
            "Training instability — mode collapse",
            "Difficult to evaluate quantitatively",
            "No explicit likelihood computation",
            "Hyperparameter sensitive"
        ],
        "key_equation": "min_G max_D E[log D(x)] + E[log(1-D(G(z)))]",
        "quiz": [
            {"q": "What are the two components of a GAN?", "options": ["Encoder & Decoder", "Generator & Discriminator", "Teacher & Student", "Actor & Critic"], "answer": 1},
            {"q": "What is mode collapse in GANs?", "options": ["The model converges perfectly", "Generator produces limited variety", "Discriminator gets too good", "Training is too fast"], "answer": 1},
            {"q": "Who proposed GANs?", "options": ["Geoffrey Hinton", "Ian Goodfellow", "Yann LeCun", "Andrew Ng"], "answer": 1},
        ]
    },
    {
        "id": "transformer",
        "name": "Transformer",
        "color": "#E056A0",
        "icon": "⟐",
        "tagline": "Attention is all you need",
        "summary": "The Transformer, introduced in the 2017 paper 'Attention Is All You Need', replaced recurrence with self-attention, enabling parallel processing of entire sequences. By computing attention scores between all token pairs, Transformers capture relationships regardless of distance. They are the foundation of GPT, BERT, and virtually all modern LLMs.",
        "architecture": "Multi-Head Attention:\nAttention(Q,K,V) = softmax(QKᵀ/√dₖ)V\n\nPosition Encoding → [Self-Attention → FFN] × N → Output\nEncoder-Decoder or Decoder-only variants",
        "layers": [4, 6, 6, 4],
        "special": "transformer",
        "use_cases": [
            "Large Language Models (GPT, BERT, LLaMA)",
            "Machine translation",
            "Text summarization & generation",
            "Vision Transformers (ViT) for image tasks"
        ],
        "pros": [
            "Parallelizable — much faster training than RNNs",
            "Captures long-range dependencies via attention",
            "Scalable to billions of parameters",
            "Foundation of modern AI (GPT, BERT, etc.)"
        ],
        "cons": [
            "Quadratic memory in sequence length (O(n²))",
            "Requires massive data and compute for training",
            "Positional encoding is a workaround, not inherent",
            "Attention maps can be hard to interpret"
        ],
        "key_equation": "Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V",
        "quiz": [
            {"q": "What mechanism replaced recurrence in Transformers?", "options": ["Convolution", "Self-Attention", "Pooling", "Dropout"], "answer": 1},
            {"q": "What is the memory complexity of self-attention?", "options": ["O(n)", "O(n log n)", "O(n²)", "O(1)"], "answer": 2},
            # FIX: was ""ImageNet"" causing syntax error — changed to single-quoted strings
            {"q": "Which famous paper introduced the Transformer?", "options": ['"ImageNet"', '"Attention Is All You Need"', 'GANs', 'BERT'], "answer": 1},
        ]
    },
    {
        "id": "autoencoder",
        "name": "Autoencoder",
        "color": "#1ABC9C",
        "icon": "⧖",
        "tagline": "Learn to compress, learn to reconstruct",
        "summary": "Autoencoders are unsupervised networks that learn efficient data encodings by compressing input into a lower-dimensional bottleneck and then reconstructing it. The encoder maps input to a latent code, and the decoder reconstructs the original. They're used for dimensionality reduction, denoising, anomaly detection, and as building blocks for generative models (VAEs).",
        "architecture": "Input → Encoder → Latent Space (bottleneck) → Decoder → Output\n\nLoss: L = ||x - x̂||² (reconstruction error)\nVariants: Denoising AE, Sparse AE, Variational AE (VAE)",
        "layers": [4, 3, 2, 3, 4],
        "special": "autoencoder",
        "use_cases": [
            "Dimensionality reduction (better than PCA)",
            "Anomaly & fraud detection",
            "Image denoising & inpainting",
            "Feature disentanglement (VAE)"
        ],
        "pros": [
            "Learns meaningful compressed representations",
            "No labeled data needed (unsupervised)",
            "Flexible architecture for different modalities",
            "Foundation for VAEs and diffusion models"
        ],
        "cons": [
            "Reconstruction loss ≠ perceptual quality",
            "Can memorize input (poor generalization)",
            "Latent space may not be smooth/continuous",
            "Bottleneck size is a critical hyperparameter"
        ],
        "key_equation": "L = ||x - D(E(x))||²",
        "quiz": [
            {"q": "What type of learning do Autoencoders use?", "options": ["Supervised", "Unsupervised", "Reinforcement", "Semi-supervised"], "answer": 1},
            {"q": "What does the bottleneck layer in an Autoencoder represent?", "options": ["The output class", "Compressed latent representation", "The loss function", "The attention weights"], "answer": 1},
            {"q": "Which variant of Autoencoder enables generative sampling?", "options": ["Denoising AE", "Sparse AE", "Variational AE (VAE)", "Contractive AE"], "answer": 2},
        ]
    },
    {
        "id": "rbfn",
        "name": "Radial Basis Function Network",
        "color": "#3498DB",
        "icon": "◎",
        "tagline": "Distance-based pattern recognition",
        "summary": "RBF Networks use radial basis functions (typically Gaussians) as activation functions in the hidden layer. Each hidden neuron represents a center in the input space and computes the distance from the input to its center. The output is a linear combination of these radial responses. They excel at interpolation and can be trained quickly using two-stage methods.",
        "architecture": "Input → RBF Hidden Layer → Linear Output\n\nRBF: φ(x) = exp(-||x - c||² / 2σ²)\nOutput: y = Σ wᵢφᵢ(x) + b\n\nCenters (c) can be set via k-means; weights via linear regression.",
        "layers": [2, 6, 2],
        "special": "rbf",
        "use_cases": [
            "Function interpolation & approximation",
            "Time series prediction",
            "Classification with localized decision boundaries",
            "Robot control systems"
        ],
        "pros": [
            "Fast training (two-stage: centers + linear weights)",
            "Universal approximator with one hidden layer",
            "Good at interpolation in low dimensions",
            "Simple, well-understood mathematically"
        ],
        "cons": [
            "Poor scalability to high-dimensional data",
            "Number of RBF units grows with data complexity",
            "Not suitable for spatial/temporal data",
            "Sensitive to center placement and width σ"
        ],
        "key_equation": "φ(x) = exp(-||x - c||² / 2σ²)",
        "quiz": [
            {"q": "What activation function does the hidden layer of an RBFN use?", "options": ["Sigmoid", "ReLU", "Gaussian (RBF)", "Tanh"], "answer": 2},
            {"q": "How can RBFN centers be determined?", "options": ["Backpropagation only", "K-means clustering", "Random initialization only", "Gradient descent only"], "answer": 1},
            {"q": "What does the RBF function measure?", "options": ["Weight magnitude", "Distance from center", "Gradient direction", "Activation slope"], "answer": 1},
        ]
    },
    {
        "id": "som",
        "name": "Self-Organizing Map",
        "color": "#E67E22",
        "icon": "▦",
        "tagline": "Topological data visualization",
        "summary": "Self-Organizing Maps (Kohonen Maps) are unsupervised networks that produce a low-dimensional (typically 2D) representation of high-dimensional data while preserving topological structure. Neurons compete to respond to input (competitive learning), and the winning neuron and its neighbors update together, creating a smooth topology-preserving map.",
        "architecture": "Input Layer → 2D Grid of Neurons\n\nEach neuron has a weight vector of same dimension as input.\nBMU: Best Matching Unit = closest neuron to input.\nUpdate: wᵢ += η·h(d)·(x - wᵢ)\nwhere h(d) is a neighborhood function.",
        "layers": [3, 5],
        "special": "som",
        "use_cases": [
            "High-dimensional data visualization",
            "Customer segmentation & clustering",
            "Color quantization & image compression",
            "Feature map visualization in neuroscience"
        ],
        "pros": [
            "Preserves topological relationships",
            "Intuitive 2D visualization of complex data",
            "No labeled data required",
            "Robust to noise"
        ],
        "cons": [
            "No guarantee of optimal mapping",
            "Slow convergence",
            "Grid size must be chosen a priori",
            "Not probabilistic — no density estimation"
        ],
        "key_equation": "BMU = argminᵢ ||x - wᵢ||",
        "quiz": [
            {"q": "What type of learning do SOMs use?", "options": ["Supervised", "Competitive/Unsupervised", "Reinforcement", "Self-supervised"], "answer": 1},
            {"q": "What does BMU stand for in SOMs?", "options": ["Batch Mean Update", "Best Matching Unit", "Base Model Unit", "Balanced Memory Utilization"], "answer": 1},
            {"q": "What do SOMs preserve in their mapping?", "options": ["Exact distances", "Topological relationships", "Probability densities", "Classification labels"], "answer": 1},
        ]
    },
]

# Build a lookup dict for quick access
NN_LOOKUP = {nn["id"]: nn for nn in NN_TYPES}


# ============================================================
# HELPER CLASSES
# ============================================================

class Particle:
    """Animated particle flowing along a connection in the network diagram."""
    def __init__(self, start, end, color):
        self.start = start
        self.end = end
        self.progress = 0.0
        self.speed = random.uniform(0.008, 0.025)
        self.color = color
        self.size = random.uniform(2.5, 4.5)
        self.alive = True

    def update(self):
        self.progress += self.speed
        if self.progress >= 1.0:
            self.alive = False

    def pos(self):
        x = self.start[0] + (self.end[0] - self.start[0]) * self.progress
        y = self.start[1] + (self.end[1] - self.start[1]) * self.progress
        return (x, y)


class NetworkDiagram(QWidget):
    """Custom widget that draws an animated neural network diagram."""

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.nn_data = data
        self.layers = data["layers"]
        self.color = QColor(data["color"])
        self.special = data.get("special")
        self.setMinimumHeight(340)
        self.particles = []
        self.node_positions = []  # [[(x,y), ...], ...]
        self.anim_tick = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def _tick(self):
        self.anim_tick += 1
        # Spawn particles
        if self.anim_tick % 4 == 0:
            self._spawn_particle()
        # Update particles
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.alive]
        self.update()

    def _spawn_particle(self):
        if len(self.node_positions) < 2:
            return
        if self.special == "gan":
            self._spawn_gan_particle()
            return
        src_layer = random.randint(0, len(self.node_positions) - 2)
        if not self.node_positions[src_layer] or not self.node_positions[src_layer + 1]:
            return
        s = random.choice(self.node_positions[src_layer])
        e = random.choice(self.node_positions[src_layer + 1])
        self.particles.append(Particle(s, e, self.color))

    def _spawn_gan_particle(self):
        half = len(self.node_positions) // 2
        if half < 1:
            return
        side = random.randint(0, 1)
        if side == 0:
            start_idx = 0
            end_idx = half - 1
        else:
            start_idx = half
            end_idx = len(self.node_positions) - 1
        if end_idx <= start_idx:
            return
        src_layer = random.randint(start_idx, end_idx - 1)
        if src_layer >= len(self.node_positions) or src_layer + 1 >= len(self.node_positions):
            return
        if not self.node_positions[src_layer] or not self.node_positions[src_layer + 1]:
            return
        s = random.choice(self.node_positions[src_layer])
        e = random.choice(self.node_positions[src_layer + 1])
        self.particles.append(Particle(s, e, self.color))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._calc_positions()

    def showEvent(self, event):
        super().showEvent(event)
        self._calc_positions()

    def _calc_positions(self):
        w = self.width()
        h = self.height()
        margin_x = 70
        margin_y = 40
        usable_w = w - 2 * margin_x
        usable_h = h - 2 * margin_y

        if self.special == "som":
            self._calc_som_positions(w, h, margin_x, margin_y)
            return

        if self.special == "gan":
            self._calc_gan_positions(w, h, margin_x, margin_y)
            return

        self.node_positions = []
        n_layers = len(self.layers)
        for i, count in enumerate(self.layers):
            layer_nodes = []
            if n_layers > 1:
                x = margin_x + usable_w * i / (n_layers - 1)
            else:
                x = w / 2.0
            for j in range(count):
                y = margin_y + usable_h * (j + 1) / (count + 1)
                layer_nodes.append((x, y))
            self.node_positions.append(layer_nodes)

    def _calc_som_positions(self, w, h, mx, my):
        self.node_positions = []
        uw = w - 2 * mx
        uh = h - 2 * my
        # Input layer
        input_nodes = []
        n_in = self.layers[0]
        ix = mx + uw * 0.2
        for j in range(n_in):
            iy = my + uh * (j + 1) / (n_in + 1)
            input_nodes.append((ix, iy))
        self.node_positions.append(input_nodes)
        # Grid
        grid_nodes = []
        n_out = self.layers[1]
        cols = min(n_out, 5)
        rows = math.ceil(n_out / cols)
        gx_start = mx + uw * 0.55
        gx_end = mx + uw * 0.95
        gy_start = my + uh * 0.15
        gy_end = my + uh * 0.85
        for idx in range(n_out):
            r = idx // cols
            c = idx % cols
            gx = gx_start + (gx_end - gx_start) * c / max(cols - 1, 1)
            gy = gy_start + (gy_end - gy_start) * r / max(rows - 1, 1)
            grid_nodes.append((gx, gy))
        self.node_positions.append(grid_nodes)

    def _calc_gan_positions(self, w, h, mx, my):
        self.node_positions = []
        uw = w - 2 * mx
        uh = h - 2 * my
        half = len(self.layers) // 2
        # Generator: left half
        for i in range(half):
            layer_nodes = []
            x = mx + uw * (0.05 + 0.42 * i / max(half - 1, 1))
            for j in range(self.layers[i]):
                y = my + uh * (0.15 + 0.3 * (j + 1) / (self.layers[i] + 1))
                layer_nodes.append((x, y))
            self.node_positions.append(layer_nodes)
        # Discriminator: right half
        for i in range(half, len(self.layers)):
            layer_nodes = []
            x = mx + uw * (0.53 + 0.42 * (i - half) / max(len(self.layers) - half - 1, 1))
            for j in range(self.layers[i]):
                y = my + uh * (0.55 + 0.3 * (j + 1) / (self.layers[i] + 1))
                layer_nodes.append((x, y))
            self.node_positions.append(layer_nodes)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        bg = QLinearGradient(0, 0, 0, self.height())
        bg.setColorAt(0, QColor("#12122a"))
        bg.setColorAt(1, QColor("#0a0a1e"))
        painter.fillRect(self.rect(), bg)

        if not self.node_positions:
            painter.end()
            return

        # Draw connections
        self._draw_connections(painter)
        # Draw special features
        self._draw_special(painter)
        # Draw particles
        self._draw_particles(painter)
        # Draw nodes
        self._draw_nodes(painter)
        # Draw labels
        self._draw_labels(painter)

        painter.end()

    def _draw_connections(self, painter):
        pen = QPen()
        pen.setWidth(1)

        if self.special == "som":
            # Connect input to grid
            pen.setColor(QColor(self.color.red(), self.color.green(), self.color.blue(), 30))
            painter.setPen(pen)
            if len(self.node_positions) >= 2:
                for src in self.node_positions[0]:
                    for dst in self.node_positions[1]:
                        painter.drawLine(QPointF(*src), QPointF(*dst))
            # Connect grid neighbors
            pen.setColor(QColor(self.color.red(), self.color.green(), self.color.blue(), 50))
            painter.setPen(pen)
            n = len(self.node_positions[1]) if len(self.node_positions) > 1 else 0
            cols = min(n, 5)
            for idx in range(n):
                r, c = idx // cols, idx % cols
                if c + 1 < cols and idx + 1 < n:
                    painter.drawLine(QPointF(*self.node_positions[1][idx]),
                                     QPointF(*self.node_positions[1][idx + 1]))
                if r + 1 < math.ceil(n / cols) and idx + cols < n:
                    painter.drawLine(QPointF(*self.node_positions[1][idx]),
                                     QPointF(*self.node_positions[1][idx + cols]))
            return

        if self.special == "transformer":
            for i in range(len(self.node_positions) - 1):
                alpha = 25
                if i == len(self.node_positions) // 2 - 1:
                    alpha = 60
                pen.setColor(QColor(self.color.red(), self.color.green(), self.color.blue(), alpha))
                painter.setPen(pen)
                for src in self.node_positions[i]:
                    for dst in self.node_positions[i + 1]:
                        painter.drawLine(QPointF(*src), QPointF(*dst))
            # Self-attention: cross connections within layers
            for layer_idx in [1, 2]:
                if layer_idx < len(self.node_positions):
                    pen.setColor(QColor(self.color.red(), self.color.green(), self.color.blue(), 40))
                    pen.setStyle(Qt.DashLine)
                    painter.setPen(pen)
                    nodes = self.node_positions[layer_idx]
                    for a in range(len(nodes)):
                        for b in range(a + 1, len(nodes)):
                            painter.drawLine(QPointF(*nodes[a]), QPointF(*nodes[b]))
                    pen.setStyle(Qt.SolidLine)
            return

        for i in range(len(self.node_positions) - 1):
            alpha = 25
            if self.special == "autoencoder":
                total = len(self.node_positions)
                mid = total // 2
                dist = abs(i - mid)
                alpha = max(15, 60 - dist * 12)
            pen.setColor(QColor(self.color.red(), self.color.green(), self.color.blue(), alpha))
            painter.setPen(pen)
            for src in self.node_positions[i]:
                for dst in self.node_positions[i + 1]:
                    painter.drawLine(QPointF(*src), QPointF(*dst))

    def _draw_special(self, painter):
        if self.special == "rnn":
            if len(self.node_positions) >= 2:
                pen = QPen(QColor(self.color.red(), self.color.green(), self.color.blue(), 120))
                pen.setWidth(2)
                painter.setPen(pen)
                for node in self.node_positions[1]:
                    x, y = node
                    path = QPainterPath()
                    path.moveTo(x + 10, y - 5)
                    path.cubicTo(x + 35, y - 25, x + 35, y + 25, x + 10, y + 5)
                    painter.drawPath(path)
                    # FIX: use QPolygonF instead of .toPoint()
                    painter.setBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), 120))
                    triangle = QPolygonF([
                        QPointF(x + 10, y + 5),
                        QPointF(x + 14, y),
                        QPointF(x + 6, y),
                    ])
                    painter.drawPolygon(triangle)
                    painter.setBrush(Qt.NoBrush)

        elif self.special == "lstm":
            if len(self.node_positions) >= 2:
                font = QFont("Segoe UI", 7, QFont.Bold)
                painter.setFont(font)
                for node in self.node_positions[1]:
                    x, y = node
                    gate_w, gate_h = 22, 14
                    gates = [(-12, -18, "f"), (12, -18, "i"), (-12, 18, "C̃"), (12, 18, "o")]
                    for dx, dy, label in gates:
                        rect = QRectF(x + dx - gate_w / 2, y + dy - gate_h / 2, gate_w, gate_h)
                        painter.setPen(QPen(QColor(self.color.red(), self.color.green(), self.color.blue(), 100), 1))
                        painter.drawRect(rect)
                        painter.setPen(QColor(self.color.red(), self.color.green(), self.color.blue(), 200))
                        painter.drawText(rect, Qt.AlignCenter, label)

        elif self.special == "gan":
            font = QFont("Segoe UI", 11, QFont.Bold)
            painter.setFont(font)
            w = self.width()
            painter.setPen(QColor("#9B59B6"))
            painter.drawText(QRectF(0, 5, w * 0.45, 30), Qt.AlignCenter, "Generator")
            painter.setPen(QColor("#E74C3C"))
            painter.drawText(QRectF(w * 0.55, 5, w * 0.45, 30), Qt.AlignCenter, "Discriminator")
            font.setPointSize(9)
            painter.setFont(font)
            painter.setPen(QColor("#888"))
            painter.drawText(QRectF(w * 0.45, self.height() // 2 - 15, w * 0.1, 30), Qt.AlignCenter, "⚔ vs")

        elif self.special == "cnn":
            if len(self.node_positions) >= 2:
                pen = QPen(QColor(self.color.red(), self.color.green(), self.color.blue(), 80), 1.5)
                painter.setPen(pen)
                for node in self.node_positions[1]:
                    x, y = node
                    size = 12
                    painter.drawRect(QRectF(x - size / 2, y - size / 2, size, size))

        elif self.special == "rbf":
            if len(self.node_positions) >= 2:
                for node in self.node_positions[1]:
                    x, y = node
                    for r in [18, 28, 38]:
                        grad = QRadialGradient(x, y, r)
                        grad.setColorAt(0, QColor(self.color.red(), self.color.green(), self.color.blue(), 0))
                        grad.setColorAt(0.7, QColor(self.color.red(), self.color.green(), self.color.blue(), 15))
                        grad.setColorAt(1, QColor(self.color.red(), self.color.green(), self.color.blue(), 0))
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(grad)
                        painter.drawEllipse(QPointF(x, y), r, r)

    def _draw_particles(self, painter):
        for p in self.particles:
            x, y = p.pos()
            grad = QRadialGradient(x, y, p.size * 2)
            c = p.color
            grad.setColorAt(0, QColor(c.red(), c.green(), c.blue(), 220))
            grad.setColorAt(0.5, QColor(c.red(), c.green(), c.blue(), 80))
            grad.setColorAt(1, QColor(c.red(), c.green(), c.blue(), 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(grad)
            painter.drawEllipse(QPointF(x, y), p.size * 2, p.size * 2)

    def _draw_nodes(self, painter):
        node_r = 9
        if self.special == "som" and len(self.node_positions) > 1:
            node_r = 7
        pulse = math.sin(self.anim_tick * 0.08) * 2

        for layer_idx, layer in enumerate(self.node_positions):
            for node_idx, (x, y) in enumerate(layer):
                r = node_r + pulse * 0.3
                # Glow
                grad = QRadialGradient(x, y, r * 2.5)
                grad.setColorAt(0, QColor(self.color.red(), self.color.green(), self.color.blue(), 60))
                grad.setColorAt(1, QColor(self.color.red(), self.color.green(), self.color.blue(), 0))
                painter.setPen(Qt.NoPen)
                painter.setBrush(grad)
                painter.drawEllipse(QPointF(x, y), r * 2.5, r * 2.5)

                # Node fill
                grad2 = QRadialGradient(x - r * 0.3, y - r * 0.3, r * 1.2)
                grad2.setColorAt(0, self.color.lighter(160))
                grad2.setColorAt(1, self.color.darker(120))
                painter.setBrush(grad2)
                painter.setPen(QPen(self.color.lighter(180), 1.5))
                painter.drawEllipse(QPointF(x, y), r, r)

    def _draw_labels(self, painter):
        font = QFont("Segoe UI", 8)
        painter.setFont(font)
        painter.setPen(QColor("#8888aa"))

        if self.special == "som":
            if self.node_positions and self.node_positions[0]:
                x = self.node_positions[0][0][0]
                painter.drawText(QRectF(x - 30, self.height() - 25, 60, 20), Qt.AlignCenter, "Input")
            if len(self.node_positions) > 1 and self.node_positions[1]:
                x = self.node_positions[1][0][0]
                painter.drawText(QRectF(x - 40, self.height() - 25, 80, 20), Qt.AlignCenter, "2D Map")
            return

        if self.special == "gan":
            return

        layer_names = []
        if self.special == "autoencoder":
            total = len(self.layers)
            mid = total // 2
            for i in range(total):
                if i == 0:
                    layer_names.append("Input")
                elif i == mid:
                    layer_names.append("Latent")
                elif i == total - 1:
                    layer_names.append("Output")
                elif i < mid:
                    layer_names.append("Enc")
                else:
                    layer_names.append("Dec")
        else:
            for i in range(len(self.layers)):
                if i == 0:
                    layer_names.append("Input")
                elif i == len(self.layers) - 1:
                    layer_names.append("Output")
                else:
                    layer_names.append(f"Hidden {i}")

        for i, name in enumerate(layer_names):
            if i < len(self.node_positions) and self.node_positions[i]:
                x = self.node_positions[i][0][0]
                painter.drawText(QRectF(x - 35, self.height() - 22, 70, 18), Qt.AlignCenter, name)


# ============================================================
# UI COMPONENTS
# ============================================================

class SidebarButton(QPushButton):
    def __init__(self, text, color, icon_str="", parent=None):
        super().__init__(parent)
        self.color = color
        self.icon_str = icon_str
        self.selected = False
        self.hovered = False
        self.setText(f"  {icon_str}  {text}")
        self.setFixedHeight(42)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.selected:
            grad = QLinearGradient(0, 0, self.width(), 0)
            grad.setColorAt(0, QColor(self.color.red(), self.color.green(), self.color.blue(), 40))
            grad.setColorAt(1, QColor(self.color.red(), self.color.green(), self.color.blue(), 5))
            painter.fillRect(self.rect(), grad)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(self.color))
            painter.drawRoundedRect(0, 4, 4, self.height() - 8, 2, 2)
        elif self.hovered:
            painter.fillRect(self.rect(), QColor(255, 255, 255, 8))

        painter.setPen(QColor(self.color) if self.selected else QColor("#b0b0cc"))
        font = QFont("Segoe UI", 10, QFont.Bold if self.selected else QFont.Normal)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(12, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, self.text())
        painter.end()

    def enterEvent(self, event):
        self.hovered = True
        self.update()

    def leaveEvent(self, event):
        self.hovered = False
        self.update()


class ContentCard(QFrame):
    def __init__(self, title, color, parent=None):
        super().__init__(parent)
        self.color = color
        self.setObjectName("contentCard")
        self.setStyleSheet(f"""
            #contentCard {{
                background-color: #1a1a32;
                border: 1px solid #2a2a4a;
                border-radius: 12px;
                padding: 16px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        # Title with color bar
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        color_bar = QFrame()
        color_bar.setFixedWidth(4)
        color_bar.setFixedHeight(22)
        color_bar.setStyleSheet(f"background-color: {color}; border-radius: 2px;")
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #e8e8f0;")
        title_layout.addWidget(color_bar)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addWidget(title_widget)

        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(6)
        layout.addLayout(self.content_layout)

    def add_text(self, text, style="font-size: 12px; color: #b0b0cc; line-height: 1.5;"):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(style)
        self.content_layout.addWidget(label)
        return label

    def add_list(self, items, bullet_color="#6c63ff"):
        for item in items:
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(4, 2, 4, 2)
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {bullet_color}; font-size: 7px;")
            dot.setFixedWidth(14)
            txt = QLabel(item)
            txt.setWordWrap(True)
            txt.setStyleSheet("font-size: 12px; color: #b0b0cc;")
            h.addWidget(dot)
            h.addWidget(txt)
            h.addStretch()
            self.content_layout.addWidget(w)


class QuizDialog(QDialog):
    def __init__(self, nn_data, parent=None):
        super().__init__(parent)
        self.nn_data = nn_data
        self.quiz = nn_data["quiz"]
        self.current_q = 0
        self.score = 0
        self.answered = False

        self.setWindowTitle(f"Quiz: {nn_data['name']}")
        self.setMinimumSize(520, 400)
        self.setStyleSheet("""
            QDialog { background-color: #12122a; }
            QLabel { color: #e0e0f0; }
            QRadioButton { color: #c0c0d8; font-size: 13px; spacing: 8px; }
            QRadioButton::indicator { width: 18px; height: 18px; }
            QPushButton { background-color: #6c63ff; color: white; border: none;
                          border-radius: 8px; padding: 10px 24px; font-size: 13px;
                          font-weight: bold; }
            QPushButton:hover { background-color: #7c73ff; }
            QPushButton:disabled { background-color: #3a3a5c; color: #666; }
        """)

        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(16)
        self.layout.setContentsMargins(32, 28, 32, 28)

        # Title
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.layout.addWidget(self.title_label)

        # Progress
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("font-size: 11px; color: #8888aa;")
        self.layout.addWidget(self.progress_label)

        # Question
        self.question_label = QLabel()
        self.question_label.setWordWrap(True)
        self.question_label.setStyleSheet("font-size: 15px; color: #e8e8f0; margin-top: 8px;")
        self.layout.addWidget(self.question_label)

        # Options
        self.radio_group = QButtonGroup(self)
        self.options = []
        for i in range(4):
            rb = QRadioButton()
            rb.setCursor(Qt.PointingHandCursor)
            self.radio_group.addButton(rb, i)
            self.options.append(rb)
            self.layout.addWidget(rb)

        self.radio_group.idClicked.connect(self._on_option_selected)

        # Feedback
        self.feedback_label = QLabel()
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setStyleSheet("font-size: 13px; font-weight: bold; margin-top: 8px;")
        self.layout.addWidget(self.feedback_label)

        self.layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        self.next_btn = QPushButton("Next Question")
        self.next_btn.clicked.connect(self._next_question)
        self.next_btn.setEnabled(False)
        btn_layout.addStretch()
        btn_layout.addWidget(self.next_btn)
        self.layout.addLayout(btn_layout)

        self._show_question()

    def _show_question(self):
        if self.current_q >= len(self.quiz):
            self._show_results()
            return
        q = self.quiz[self.current_q]
        self.title_label.setText(f"🎯  {self.nn_data['name']} Quiz")
        self.progress_label.setText(f"Question {self.current_q + 1} of {len(self.quiz)}")
        self.question_label.setText(q["q"])
        for i, opt in enumerate(q["options"]):
            self.options[i].setText(opt)
            self.options[i].setChecked(False)
            self.options[i].setStyleSheet("color: #c0c0d8; font-size: 13px;")
            self.options[i].setEnabled(True)
            self.options[i].show()
        self.feedback_label.setText("")
        self.answered = False
        self.next_btn.setEnabled(False)
        self.next_btn.setText("Next Question" if self.current_q < len(self.quiz) - 1 else "See Results")

    def _on_option_selected(self, idx):
        if self.answered:
            return
        self.answered = True
        correct = self.quiz[self.current_q]["answer"]
        if idx == correct:
            self.score += 1
            self.feedback_label.setStyleSheet("color: #6BCB77; font-size: 13px; font-weight: bold; margin-top: 8px;")
            self.feedback_label.setText("✅ Correct!")
            self.options[idx].setStyleSheet("color: #6BCB77; font-size: 13px; font-weight: bold;")
        else:
            self.feedback_label.setStyleSheet("color: #FF6B6B; font-size: 13px; font-weight: bold; margin-top: 8px;")
            self.feedback_label.setText(f"❌ Incorrect. The correct answer is: {self.quiz[self.current_q]['options'][correct]}")
            self.options[idx].setStyleSheet("color: #FF6B6B; font-size: 13px;")
            self.options[correct].setStyleSheet("color: #6BCB77; font-size: 13px; font-weight: bold;")

        for rb in self.options:
            rb.setEnabled(False)
        self.next_btn.setEnabled(True)

    def _next_question(self):
        self.current_q += 1
        self._show_question()

    def _show_results(self):
        total = len(self.quiz)
        pct = int(self.score / total * 100)
        self.title_label.setText("🏆  Quiz Complete!")
        self.progress_label.setText("")
        self.question_label.setText(f"Your Score: {self.score}/{total} ({pct}%)")

        if pct == 100:
            msg = "🌟 Perfect! You've mastered this topic!"
        elif pct >= 67:
            msg = "👍 Good job! Review the ones you missed."
        else:
            msg = "📚 Keep studying! Review the material and try again."

        self.feedback_label.setStyleSheet("color: #e0e0f0; font-size: 14px; margin-top: 8px;")
        self.feedback_label.setText(msg)

        for rb in self.options:
            rb.hide()
        self.next_btn.setText("Close")
        try:
            self.next_btn.clicked.disconnect()
        except RuntimeError:
            pass
        self.next_btn.clicked.connect(self.accept)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)


# ============================================================
# HOME PAGE CARD
# ============================================================

class HomeCard(QFrame):
    """Clickable card on the home page for each network type."""
    clicked = Signal(str)  # emits nn_id

    def __init__(self, nn_data, parent=None):
        super().__init__(parent)
        self.nn_data = nn_data
        self.color = QColor(nn_data["color"])
        self.hovered = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(180)
        self.setObjectName("homeCard")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Card background
        if self.hovered:
            bg_color = QColor("#1e1e3a")
            border_color = QColor(self.color.red(), self.color.green(), self.color.blue(), 120)
        else:
            bg_color = QColor("#16162e")
            border_color = QColor("#2a2a4a")

        painter.setPen(QPen(border_color, 1.5))
        painter.setBrush(bg_color)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 12, 12)

        # Top color accent line
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.color)
        painter.drawRoundedRect(QRectF(12, 8, 40, 3), 1.5, 1.5)

        # Icon
        painter.setPen(self.color)
        font = QFont("Segoe UI", 28)
        painter.setFont(font)
        painter.drawText(QRectF(16, 22, 50, 50), Qt.AlignCenter, self.nn_data["icon"])

        # Name
        painter.setPen(QColor("#e8e8f0"))
        font = QFont("Segoe UI", 13, QFont.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(16, 70, self.width() - 32, 24), Qt.AlignLeft | Qt.AlignVCenter,
                         self.nn_data["name"])

        # Tagline
        painter.setPen(QColor("#8888aa"))
        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        painter.drawText(QRectF(16, 96, self.width() - 32, 40), Qt.AlignLeft | Qt.TextWordWrap,
                         self.nn_data["tagline"])

        # Bottom "Explore →"
        painter.setPen(self.color)
        font = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(16, self.height() - 32, self.width() - 32, 20),
                         Qt.AlignLeft, "Explore →")

        painter.end()

    def enterEvent(self, event):
        self.hovered = True
        self.update()

    def leaveEvent(self, event):
        self.hovered = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.nn_data["id"])
        super().mousePressEvent(event)


# ============================================================
# MAIN WINDOW
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Neural Network Explorer — Interactive Learning App")
        self.setMinimumSize(1200, 800)
        self.resize(1350, 880)
        self.setStyleSheet(STYLESHEET)

        self.visited = set()
        self.quiz_scores = {}
        self.current_id = "home"
        self.sidebar_buttons = {}

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        sidebar = self._build_sidebar()
        main_layout.addWidget(sidebar)

        # Content area
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.stack, 1)

        # Build pages
        self._build_home_page()
        for nn in NN_TYPES:
            self._build_detail_page(nn)

    # ----------------------------------------------------------
    # Sidebar
    # ----------------------------------------------------------
    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #10102a;
                border-right: 1px solid #1e1e3a;
            }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo area
        logo_widget = QWidget()
        logo_layout = QVBoxLayout(logo_widget)
        logo_layout.setContentsMargins(20, 20, 20, 8)
        logo_label = QLabel("🧠 NN Explorer")
        logo_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #e0e0f0;")
        logo_layout.addWidget(logo_label)

        # Progress
        self.progress_label = QLabel("0 / 10 explored")
        self.progress_label.setStyleSheet("font-size: 10px; color: #6a6a8a; margin-left: 2px;")
        logo_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(6)
        logo_layout.addWidget(self.progress_bar)

        layout.addWidget(logo_widget)

        # Separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #1e1e3a; margin: 8px 16px;")
        layout.addWidget(sep)

        # Home button
        home_btn = SidebarButton("Home", QColor("#6c63ff"), "🏠")
        home_btn.selected = True
        home_btn.clicked.connect(lambda: self._navigate("home"))
        layout.addWidget(home_btn)
        self.sidebar_buttons["home"] = home_btn

        # Separator
        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background-color: #1e1e3a; margin: 8px 16px;")
        layout.addWidget(sep2)

        # Section label
        section = QLabel("  NETWORKS")
        section.setStyleSheet("font-size: 9px; font-weight: bold; color: #5a5a7a; letter-spacing: 1.5px; margin-left: 4px;")
        layout.addWidget(section)
        layout.addSpacing(4)

        # NN buttons
        for nn in NN_TYPES:
            btn = SidebarButton(nn["name"], QColor(nn["color"]), nn["icon"])
            btn.clicked.connect(lambda checked, nid=nn["id"]: self._navigate(nid))
            layout.addWidget(btn)
            self.sidebar_buttons[nn["id"]] = btn

        layout.addStretch()

        # Footer
        footer = QLabel("  v1.0  •  PySide6")
        footer.setStyleSheet("font-size: 9px; color: #3a3a5a; margin-bottom: 10px;")
        layout.addWidget(footer)

        return sidebar

    # ----------------------------------------------------------
    # Home page
    # ----------------------------------------------------------
    def _build_home_page(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #0d0d1a; }")

        container = QWidget()
        container.setStyleSheet("background-color: #0d0d1a;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(40, 36, 40, 36)
        layout.setSpacing(24)

        # Hero
        hero = QWidget()
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(0, 0, 0, 0)
        hero_layout.setSpacing(8)

        title = QLabel("🧠 Neural Network Explorer")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #e8e8f0;")
        hero_layout.addWidget(title)

        subtitle = QLabel(
            "Explore the most important neural network architectures interactively.\n"
            "Click any card to dive deep into its architecture, use cases, and take a quiz."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 13px; color: #8888aa; line-height: 1.6;")
        hero_layout.addWidget(subtitle)
        layout.addWidget(hero)

        # Grid of cards
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(16)
        cols = 3
        for idx, nn in enumerate(NN_TYPES):
            card = HomeCard(nn)
            card.clicked.connect(self._navigate)
            r = idx // cols
            c = idx % cols
            grid_layout.addWidget(card, r, c)
        layout.addWidget(grid_widget)
        layout.addStretch()

        scroll.setWidget(container)
        self.stack.addWidget(scroll)

    # ----------------------------------------------------------
    # Detail page for a single network type
    # ----------------------------------------------------------
    def _build_detail_page(self, nn_data):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #0d0d1a; }")

        container = QWidget()
        container.setStyleSheet("background-color: #0d0d1a;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(40, 30, 40, 40)
        layout.setSpacing(20)

        nn_id = nn_data["id"]
        color = nn_data["color"]
        qcolor = QColor(color)

        # Back button + title row
        top_row = QHBoxLayout()
        back_btn = QPushButton("← Back")
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent; color: #8888aa;
                border: 1px solid #2a2a4a; border-radius: 6px;
                padding: 6px 14px; font-size: 12px;
            }}
            QPushButton:hover {{ background-color: #1a1a32; color: #c0c0d8; }}
        """)
        back_btn.clicked.connect(lambda: self._navigate("home"))
        top_row.addWidget(back_btn)
        top_row.addStretch()
        layout.addLayout(top_row)

        # Header
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        name_label = QLabel(f'{nn_data["icon"]}  {nn_data["name"]}')
        name_label.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {color};")
        header_layout.addWidget(name_label)

        tagline_label = QLabel(nn_data["tagline"])
        tagline_label.setStyleSheet("font-size: 13px; color: #8888aa; font-style: italic;")
        header_layout.addWidget(tagline_label)
        layout.addWidget(header)

        # Network Diagram
        diagram_card = ContentCard("Network Architecture", color)
        diagram = NetworkDiagram(nn_data)
        diagram.setMinimumHeight(300)
        diagram_card.content_layout.addWidget(diagram)
        layout.addWidget(diagram_card)

        # Summary
        summary_card = ContentCard("Overview", color)
        summary_card.add_text(nn_data["summary"])
        layout.addWidget(summary_card)

        # Architecture
        arch_card = ContentCard("Architecture", color)
        arch_card.add_text(nn_data["architecture"], "font-size: 12px; color: #c0c0d8; font-family: 'Consolas', 'Courier New', monospace; line-height: 1.6;")
        layout.addWidget(arch_card)

        # Key Equation
        eq_card = ContentCard("Key Equation", color)
        eq_card.add_text(nn_data["key_equation"], f"font-size: 16px; color: {color}; font-weight: bold; font-family: 'Consolas', 'Courier New', monospace; padding: 8px;")
        layout.addWidget(eq_card)

        # Use Cases, Pros, Cons — 3-column layout
        col_widget = QWidget()
        col_layout = QHBoxLayout(col_widget)
        col_layout.setContentsMargins(0, 0, 0, 0)
        col_layout.setSpacing(16)

        uc_card = ContentCard("Use Cases", color)
        uc_card.add_list(nn_data["use_cases"], color)
        col_layout.addWidget(uc_card)

        pros_card = ContentCard("Pros", "#6BCB77")
        pros_card.add_list(nn_data["pros"], "#6BCB77")
        col_layout.addWidget(pros_card)

        cons_card = ContentCard("Cons", "#FF6B6B")
        cons_card.add_list(nn_data["cons"], "#FF6B6B")
        col_layout.addWidget(cons_card)

        layout.addWidget(col_widget)

        # Quiz button
        quiz_row = QHBoxLayout()
        quiz_row.addStretch()
        quiz_btn = QPushButton(f"🎯  Take the {nn_data['name']} Quiz")
        quiz_btn.setCursor(Qt.PointingHandCursor)
        quiz_btn.setFixedSize(300, 48)
        quiz_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color}; color: white; border: none;
                border-radius: 10px; font-size: 14px; font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {qcolor.lighter(120).name()};
            }}
        """)
        quiz_btn.clicked.connect(lambda checked, d=nn_data: self._open_quiz(d))
        quiz_row.addWidget(quiz_btn)
        quiz_row.addStretch()
        layout.addLayout(quiz_row)

        # Score display (if already taken quiz)
        score_label = QLabel()
        score_label.setAlignment(Qt.AlignCenter)
        score_label.setStyleSheet("font-size: 11px; color: #6a6a8a;")
        score_label.setObjectName(f"score_{nn_id}")
        layout.addWidget(score_label)

        layout.addStretch()

        scroll.setWidget(container)
        self.stack.addWidget(scroll)

    # ----------------------------------------------------------
    # Navigation
    # ----------------------------------------------------------
    def _navigate(self, nn_id):
        # Update sidebar selection
        for key, btn in self.sidebar_buttons.items():
            btn.selected = (key == nn_id)
            btn.update()

        self.current_id = nn_id

        if nn_id == "home":
            self.stack.setCurrentIndex(0)
        else:
            # Find the index for this nn_id
            idx = next((i + 1 for i, nn in enumerate(NN_TYPES) if nn["id"] == nn_id), 0)
            self.stack.setCurrentIndex(idx)
            # Mark as visited
            self.visited.add(nn_id)
            self._update_progress()

    def _update_progress(self):
        total = len(NN_TYPES)
        visited_count = len(self.visited)
        quiz_count = len(self.quiz_scores)
        pct = int((visited_count + quiz_count) / (total * 2) * 100)
        self.progress_bar.setValue(pct)
        self.progress_label.setText(f"{visited_count}/{total} explored  •  {quiz_count}/{total} quizzes")

    # ----------------------------------------------------------
    # Quiz
    # ----------------------------------------------------------
    def _open_quiz(self, nn_data):
        dialog = QuizDialog(nn_data, self)
        result = dialog.exec()
        if result == QDialog.Accepted:
            self.quiz_scores[nn_data["id"]] = dialog.score
            self._update_progress()
            # Update score label on the detail page
            nn_id = nn_data["id"]
            idx = next((i + 1 for i, nn in enumerate(NN_TYPES) if nn["id"] == nn_id), -1)
            if idx >= 0:
                page = self.stack.widget(idx)
                score_label = page.findChild(QLabel, f"score_{nn_id}")
                if score_label:
                    total_q = len(nn_data["quiz"])
                    pct = int(dialog.score / total_q * 100)
                    score_label.setText(f"🏆 Last quiz score: {dialog.score}/{total_q} ({pct}%)")

    # ----------------------------------------------------------
    # Keyboard shortcut
    # ----------------------------------------------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self._navigate("home")
        else:
            super().keyPressEvent(event)


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    app = QApplication(sys.argv)

    # Set application-wide palette for consistent dark theme
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#0d0d1a"))
    palette.setColor(QPalette.WindowText, QColor("#e0e0f0"))
    palette.setColor(QPalette.Base, QColor("#12122a"))
    palette.setColor(QPalette.AlternateBase, QColor("#16162e"))
    palette.setColor(QPalette.Text, QColor("#e0e0f0"))
    palette.setColor(QPalette.Button, QColor("#1a1a32"))
    palette.setColor(QPalette.ButtonText, QColor("#e0e0f0"))
    palette.setColor(QPalette.Highlight, QColor("#6c63ff"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()