"""
Mathematical Story:
- XOR is not linearly separable (proven via convex hull argument)
- A single-layer perceptron CANNOT solve it (Minsky & Papert, 1969)
- A hidden layer learns a non-linear feature map that makes XOR linearly separable
- Backpropagation = chain rule of calculus applied systematically
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)  # 4 × 2
Y = np.array([[0],[1],[1],[0]], dtype=float)           # 4 × 1  (XOR labels)

# ─────────────────────────────────────────────
# 2. ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)          # d/dz σ(z) = σ(z)(1−σ(z))

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

# ─────────────────────────────────────────────
# 3. NEURAL NETWORK CLASS
# ─────────────────────────────────────────────
class NeuralNetwork:
    """
    Architecture: input(2) → hidden(n_hidden) → output(1)
    
    Forward pass:
        z1 = X W1 + b1,   a1 = σ(z1)
        z2 = a1 W2 + b2,  a2 = σ(z2)   ← prediction ŷ
    
    Loss (Binary Cross-Entropy):
        L = -1/m Σ [y log(ŷ) + (1−y) log(1−ŷ)]
    
    Backpropagation (chain rule):
        δ2 = (a2 − y) ⊙ σ'(z2)          ← output error
        δ1 = (δ2 W2ᵀ) ⊙ σ'(z1)          ← hidden error
        
        ∂L/∂W2 = a1ᵀ δ2 / m
        ∂L/∂W1 = Xᵀ δ1 / m
    """
    def __init__(self, n_hidden=4, lr=0.1, activation='sigmoid', seed=42):
        np.random.seed(seed)
        self.lr = lr
        self.act = sigmoid if activation == 'sigmoid' else relu
        self.act_d = sigmoid_deriv if activation == 'sigmoid' else relu_deriv
        
        # Xavier initialization — important for convergence!
        # Var(W) = 1/fan_in prevents vanishing/exploding gradients
        self.W1 = np.random.randn(2, n_hidden) * np.sqrt(1/2)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, 1) * np.sqrt(1/n_hidden)
        self.b2 = np.zeros((1, 1))
        
        self.loss_history = []

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.act(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)          # output always sigmoid for binary
        return self.a2

    def loss(self, y, y_hat):
        eps = 1e-9
        return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

    def backward(self, X, y):
        m = X.shape[0]
        
        # Output layer delta
        delta2 = (self.a2 - y) * sigmoid_deriv(self.z2)   # shape: m×1
        
        # Hidden layer delta (backprop through W2)
        delta1 = (delta2 @ self.W2.T) * self.act_d(self.z1)  # shape: m×n_hidden
        
        # Gradients
        dW2 = self.a1.T @ delta2 / m
        db2 = np.mean(delta2, axis=0, keepdims=True)
        dW1 = X.T @ delta1 / m
        db1 = np.mean(delta1, axis=0, keepdims=True)
        
        # Gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            y_hat = self.forward(X)
            l = self.loss(y, y_hat)
            self.loss_history.append(l)
            self.backward(X, y)
        return self

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y) * 100

# ─────────────────────────────────────────────
# 4. EXPERIMENT: Can 1 layer solve XOR?
# ─────────────────────────────────────────────
print("=" * 55)
print("  Neural Network from Scratch — XOR Problem")
print("=" * 55)

# Single layer perceptron (n_hidden=1 squashes to a linear boundary)
single = NeuralNetwork(n_hidden=1, lr=0.5, seed=0)
single.train(X, Y, epochs=10000)

# Deep enough network
nn = NeuralNetwork(n_hidden=4, lr=0.5, seed=42)
nn.train(X, Y, epochs=10000)

print(f"\n{'INPUT':<12} {'TRUE':>6} {'1-Layer':>10} {'2-Layer':>10}")
print("-" * 42)
for i, (x, y) in enumerate(zip(X, Y)):
    p1 = single.forward(x.reshape(1,-1))[0,0]
    p2 = nn.forward(x.reshape(1,-1))[0,0]
    label = int(y[0])
    print(f"[{int(x[0])}, {int(x[1])}]      {label:>6}    {p1:>8.4f}    {p2:>8.4f}")

print(f"\n1-Layer Accuracy : {single.accuracy(X,Y):.1f}%")
print(f"2-Layer Accuracy : {nn.accuracy(X,Y):.1f}%")

# ─────────────────────────────────────────────
# 5. EFFECT OF HIDDEN UNITS
# ─────────────────────────────────────────────
hidden_sizes = [1, 2, 3, 4, 8, 16]
results = {}
for h in hidden_sizes:
    net = NeuralNetwork(n_hidden=h, lr=0.5, seed=42)
    net.train(X, Y, epochs=10000)
    results[h] = {
        'acc': net.accuracy(X, Y),
        'loss': net.loss_history
    }
    print(f"Hidden units={h:2d} → Accuracy: {results[h]['acc']:.1f}%  Final loss: {net.loss_history[-1]:.4f}")

# ─────────────────────────────────────────────
# 6. EFFECT OF LEARNING RATE
# ─────────────────────────────────────────────
learning_rates = [0.01, 0.1, 0.5, 1.0, 3.0]
lr_results = {}
for lr in learning_rates:
    net = NeuralNetwork(n_hidden=4, lr=lr, seed=42)
    net.train(X, Y, epochs=10000)
    lr_results[lr] = net.loss_history

# ─────────────────────────────────────────────
# 7. DECISION BOUNDARY HELPER
# ─────────────────────────────────────────────
def get_decision_boundary(model, resolution=300):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, resolution),
                          np.linspace(-0.5, 1.5, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid).reshape(xx.shape)
    return xx, yy, Z

# ─────────────────────────────────────────────
# 8. PLOTTING
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0d0d1a')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

colors_xor = ['#ff4d6d', '#4dffb4']
point_colors = [colors_xor[int(y[0])] for y in Y]

# --- Plot 1: 1-Layer decision boundary ---
ax1 = fig.add_subplot(gs[0, 0])
xx, yy, Z = get_decision_boundary(single)
ax1.contourf(xx, yy, Z, levels=50, cmap='RdYlGn', alpha=0.8)
ax1.contour(xx, yy, Z, levels=[0.5], colors='white', linewidths=2)
for i, (x, c) in enumerate(zip(X, point_colors)):
    ax1.scatter(*x, color=c, s=200, zorder=5, edgecolors='white', linewidths=2)
ax1.set_title('1-Layer Perceptron\n(Cannot solve XOR)', color='white', fontsize=11, fontweight='bold')
ax1.set_facecolor('#1a1a2e')
ax1.tick_params(colors='white')
for spine in ax1.spines.values(): spine.set_color('#444')

# --- Plot 2: 2-Layer decision boundary ---
ax2 = fig.add_subplot(gs[0, 1])
xx, yy, Z = get_decision_boundary(nn)
ax2.contourf(xx, yy, Z, levels=50, cmap='RdYlGn', alpha=0.8)
ax2.contour(xx, yy, Z, levels=[0.5], colors='white', linewidths=2)
for i, (x, c) in enumerate(zip(X, point_colors)):
    ax2.scatter(*x, color=c, s=200, zorder=5, edgecolors='white', linewidths=2)
ax2.set_title('2-Layer Network (4 hidden)\nSolves XOR ✓', color='white', fontsize=11, fontweight='bold')
ax2.set_facecolor('#1a1a2e')
ax2.tick_params(colors='white')
for spine in ax2.spines.values(): spine.set_color('#444')

# --- Plot 3: XOR in hidden space ---
ax3 = fig.add_subplot(gs[0, 2])
# Project XOR points into the hidden representation of the trained 2-layer net
hidden_rep = nn.forward(X)  # run forward to populate a1
h = nn.a1  # 4×4 hidden activations
# Use first 2 hidden units for visualization
ax3.scatter(h[:,0], h[:,1], c=point_colors, s=300, zorder=5, edgecolors='white', linewidths=2)
for i, (xi, yi) in enumerate(zip(X, Y)):
    ax3.annotate(f"({int(xi[0])},{int(xi[1])})", (h[i,0]+0.02, h[i,1]+0.02), color='white', fontsize=9)
ax3.set_title('Hidden Layer Representation\n(XOR becomes linearly separable)', color='white', fontsize=11, fontweight='bold')
ax3.set_xlabel('Hidden unit 1', color='#aaa', fontsize=9)
ax3.set_ylabel('Hidden unit 2', color='#aaa', fontsize=9)
ax3.set_facecolor('#1a1a2e')
ax3.tick_params(colors='white')
for spine in ax3.spines.values(): spine.set_color('#444')

# --- Plot 4: Loss convergence comparison ---
ax4 = fig.add_subplot(gs[1, 0:2])
palette = ['#ff4d6d','#ff9f43','#ffd700','#4dffb4','#00d2ff','#a29bfe']
for i, (h, res) in enumerate(results.items()):
    ax4.plot(res['loss'], color=palette[i], linewidth=1.8, label=f'hidden={h}', alpha=0.9)
ax4.set_title('Loss Convergence vs Hidden Layer Size', color='white', fontsize=12, fontweight='bold')
ax4.set_xlabel('Epoch', color='#aaa')
ax4.set_ylabel('Binary Cross-Entropy Loss', color='#aaa')
ax4.legend(framealpha=0.2, labelcolor='white', fontsize=9)
ax4.set_facecolor('#1a1a2e')
ax4.tick_params(colors='white')
ax4.set_yscale('log')
for spine in ax4.spines.values(): spine.set_color('#444')
ax4.grid(alpha=0.15)

# --- Plot 5: Learning rate effect ---
ax5 = fig.add_subplot(gs[1, 2])
lr_palette = ['#ff4d6d','#ff9f43','#4dffb4','#00d2ff','#a29bfe']
for i, (lr, hist) in enumerate(lr_results.items()):
    ax5.plot(hist[:3000], color=lr_palette[i], linewidth=1.8, label=f'lr={lr}', alpha=0.9)
ax5.set_title('Learning Rate Sensitivity\n(first 3000 epochs)', color='white', fontsize=11, fontweight='bold')
ax5.set_xlabel('Epoch', color='#aaa')
ax5.set_ylabel('Loss', color='#aaa')
ax5.legend(framealpha=0.2, labelcolor='white', fontsize=9)
ax5.set_facecolor('#1a1a2e')
ax5.tick_params(colors='white')
ax5.set_yscale('log')
for spine in ax5.spines.values(): spine.set_color('#444')
ax5.grid(alpha=0.15)

# --- Plot 6: Backprop gradient flow diagram ---
ax6 = fig.add_subplot(gs[2, :])
ax6.set_facecolor('#1a1a2e')
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 4)
ax6.axis('off')

# Draw network diagram
layer_x = [1, 3, 5.5, 7.5, 9.5]
node_cols = ['#4dffb4','#00d2ff','#00d2ff','#ff4d6d','#ffd700']
node_labels = [['x₁','x₂'], ['h₁','h₂','h₃','h₄'], ['ŷ']]
layer_xs = [1.5, 4.5, 8.5]
layer_ys = [[1.5, 2.5], [0.8, 1.6, 2.4, 3.2], [2.0]]

# Draw connections
for i, (x1, y1) in enumerate(zip([1.5]*2, [1.5, 2.5])):
    for j, (x2, y2) in enumerate(zip([4.5]*4, [0.8, 1.6, 2.4, 3.2])):
        ax6.plot([x1, x2], [y1, y2], color='#333', lw=1.2, alpha=0.7)
for i, (x1, y1) in enumerate(zip([4.5]*4, [0.8, 1.6, 2.4, 3.2])):
    ax6.plot([x1, 8.5], [y1, 2.0], color='#333', lw=1.2, alpha=0.7)

# Draw nodes
for (xs, ys, col, labels) in zip(layer_xs, layer_ys, ['#4dffb4','#00d2ff','#ff9f43'],
                                   [['x₁','x₂'],['h₁','h₂','h₃','h₄'],['ŷ']]):
    for y, lbl in zip(ys, labels):
        circle = plt.Circle((xs, y), 0.28, color=col, zorder=5)
        ax6.add_patch(circle)
        ax6.text(xs, y, lbl, ha='center', va='center', fontsize=9, color='#0d0d1a', fontweight='bold', zorder=6)

# Annotations
ax6.text(1.5, 3.8, 'INPUT LAYER', ha='center', color='#4dffb4', fontsize=10, fontweight='bold')
ax6.text(4.5, 3.8, 'HIDDEN LAYER\nz¹=XW¹+b¹, a¹=σ(z¹)', ha='center', color='#00d2ff', fontsize=9, fontweight='bold')
ax6.text(8.5, 3.8, 'OUTPUT\nŷ=σ(a¹W²+b²)', ha='center', color='#ff9f43', fontsize=9, fontweight='bold')

# Backprop arrows
ax6.annotate('', xy=(4.5, 0.2), xytext=(8.5, 0.2),
             arrowprops=dict(arrowstyle='->', color='#ff4d6d', lw=2))
ax6.annotate('', xy=(1.5, 0.2), xytext=(4.0, 0.2),
             arrowprops=dict(arrowstyle='->', color='#ff4d6d', lw=2))
ax6.text(6.5, 0.0, '← Backpropagation (chain rule)', ha='center', color='#ff4d6d', fontsize=10, fontweight='bold')
ax6.text(5.0, 0.38, 'δ¹=(δ²W²ᵀ)⊙σ\'(z¹)', ha='center', color='#ff4d6d', fontsize=8)
ax6.text(8.2, 0.38, 'δ²=(ŷ−y)⊙σ\'(z²)', ha='center', color='#ff4d6d', fontsize=8)

ax6.set_title('Network Architecture & Backpropagation Flow', color='white', fontsize=12, fontweight='bold', pad=10)

plt.suptitle('Neural Network from Scratch — XOR Problem\n(NumPy only · No ML libraries)',
             color='white', fontsize=15, fontweight='bold', y=1.01)

plt.savefig('/mnt/user-data/outputs/xor_neural_net.png', dpi=150, bbox_inches='tight',
            facecolor='#0d0d1a')
plt.close()
print("\n✓ Plot saved.")
