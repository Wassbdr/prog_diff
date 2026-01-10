"""
Moteur d'Autodifférentiation - Micro-PyTorch
Implémentation complète du Reverse Mode Automatic Differentiation
"""

import numpy as np

# =============================================================================
# CLASSE TENSOR - Coeur du moteur d'autodifférentiation
# =============================================================================

class Tensor:
    """
    Classe Tensor avec support pour l'autodifférentiation en mode inverse.
    Gère les données, les gradients, et construit un graphe de calcul dynamique (DAG).
    """
    
    def __init__(self, data, _children=(), _op='', requires_grad=True):
        # Conversion en array numpy si nécessaire
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float64)
        elif isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float64)
        
        self.data = data
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.requires_grad = requires_grad
        
        # Variables internes pour la construction du graphe
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def T(self):
        """Transposition du tenseur"""
        out = Tensor(self.data.T, (self,), 'T')
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    # -------------------------------------------------------------------------
    # Opérations arithmétiques de base
    # -------------------------------------------------------------------------
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Gestion du broadcasting
            self_grad = out.grad
            other_grad = out.grad
            
            # Réduction si broadcasting a eu lieu
            if self.data.shape != out.data.shape:
                self_grad = np.sum(out.grad, axis=tuple(range(out.grad.ndim - self.data.ndim)))
                if self.data.shape != self_grad.shape:
                    self_grad = np.sum(self_grad, axis=0, keepdims=True) if len(self.data.shape) > 0 else np.sum(self_grad)
            if other.data.shape != out.data.shape:
                other_grad = np.sum(out.grad, axis=tuple(range(out.grad.ndim - other.data.ndim)))
                if other.data.shape != other_grad.shape:
                    other_grad = np.sum(other_grad, axis=0, keepdims=True) if len(other.data.shape) > 0 else np.sum(other_grad)
            
            self.grad = self.grad + self_grad
            other.grad = other.grad + other_grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        out = Tensor(-self.data, (self,), 'neg')
        def _backward():
            self.grad = self.grad - out.grad
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data - other.data, (self, other), '-')
        
        def _backward():
            self_grad = out.grad
            other_grad = -out.grad
            
            if self.data.shape != out.data.shape:
                self_grad = np.sum(out.grad, axis=tuple(range(out.grad.ndim - self.data.ndim)))
            if other.data.shape != out.data.shape:
                other_grad = np.sum(-out.grad, axis=tuple(range(out.grad.ndim - other.data.ndim)))
            
            self.grad = self.grad + self_grad
            other.grad = other.grad + other_grad
        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        return Tensor(other, requires_grad=False) - self
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            self_grad = other.data * out.grad
            other_grad = self.data * out.grad
            
            # Gestion du broadcasting
            while self_grad.ndim > self.data.ndim:
                self_grad = self_grad.sum(axis=0)
            while other_grad.ndim > other.data.ndim:
                other_grad = other_grad.sum(axis=0)
            
            self.grad = self.grad + self_grad
            other.grad = other.grad + other_grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data / other.data, (self, other), '/')
        
        def _backward():
            self_grad = out.grad / other.data
            other_grad = -self.data * out.grad / (other.data ** 2)
            
            while self_grad.ndim > self.data.ndim:
                self_grad = self_grad.sum(axis=0)
            while other_grad.ndim > other.data.ndim:
                other_grad = other_grad.sum(axis=0)
            
            self.grad = self.grad + self_grad
            other.grad = other.grad + other_grad
        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        return Tensor(other, requires_grad=False) / self
    
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Seules les puissances scalaires sont supportées"
        out = Tensor(self.data ** power, (self,), f'**{power}')
        
        def _backward():
            self.grad = self.grad + power * (self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        """Produit matriciel - crucial pour le MLP"""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, (self, other), '@')
        
        def _backward():
            # Pour les matrices 2D: dL/dA = dL/dC @ B.T, dL/dB = A.T @ dL/dC
            if self.data.ndim == 2 and other.data.ndim == 2:
                self.grad = self.grad + out.grad @ other.data.T
                other.grad = other.grad + self.data.T @ out.grad
            elif self.data.ndim == 1 and other.data.ndim == 2:
                # Vecteur @ Matrice
                self.grad = self.grad + out.grad @ other.data.T
                other.grad = other.grad + np.outer(self.data, out.grad)
            elif self.data.ndim == 2 and other.data.ndim == 1:
                # Matrice @ Vecteur
                self.grad = self.grad + np.outer(out.grad, other.data)
                other.grad = other.grad + self.data.T @ out.grad
            else:
                # Cas batch avec ndim > 2
                self.grad = self.grad + out.grad @ np.swapaxes(other.data, -1, -2)
                other.grad = other.grad + np.swapaxes(self.data, -1, -2) @ out.grad
        out._backward = _backward
        return out
    
    # -------------------------------------------------------------------------
    # Fonctions d'activation et mathématiques
    # -------------------------------------------------------------------------
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')
        def _backward():
            self.grad = self.grad + (self.data > 0).astype(float) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad = self.grad + out.data * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data + 1e-8), (self,), 'log')
        def _backward():
            self.grad = self.grad + out.grad / (self.data + 1e-8)
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad = self.grad + np.ones_like(self.data) * grad
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), 'mean')
        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad = self.grad + np.ones_like(self.data) * grad / n
        out._backward = _backward
        return out
    
    def softmax(self, axis=-1):
        """Softmax numériquement stable"""
        # Soustraction du max pour la stabilité numérique
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp_vals = np.exp(shifted)
        softmax_out = exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)
        out = Tensor(softmax_out, (self,), 'softmax')
        
        def _backward():
            # Jacobien de softmax: diag(s) - s @ s.T
            # Pour chaque échantillon du batch
            s = out.data
            if s.ndim == 1:
                jacobian = np.diag(s) - np.outer(s, s)
                self.grad = self.grad + jacobian @ out.grad
            else:
                # Batch mode
                for i in range(s.shape[0]):
                    si = s[i]
                    jacobian = np.diag(si) - np.outer(si, si)
                    self.grad[i] = self.grad[i] + jacobian @ out.grad[i]
        out._backward = _backward
        return out
    
    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad = self.grad + (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(sig, (self,), 'sigmoid')
        def _backward():
            self.grad = self.grad + out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out
    
    def sin(self):
        out = Tensor(np.sin(self.data), (self,), 'sin')
        def _backward():
            self.grad = self.grad + np.cos(self.data) * out.grad
        out._backward = _backward
        return out
    
    def cos(self):
        out = Tensor(np.cos(self.data), (self,), 'cos')
        def _backward():
            self.grad = self.grad - np.sin(self.data) * out.grad
        out._backward = _backward
        return out
    
    def sqrt(self):
        out = Tensor(np.sqrt(self.data), (self,), 'sqrt')
        def _backward():
            self.grad = self.grad + 0.5 / np.sqrt(self.data + 1e-8) * out.grad
        out._backward = _backward
        return out
    
    def abs(self):
        out = Tensor(np.abs(self.data), (self,), 'abs')
        def _backward():
            self.grad = self.grad + np.sign(self.data) * out.grad
        out._backward = _backward
        return out
    
    def reshape(self, *shape):
        """Reshape le tenseur"""
        original_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), (self,), 'reshape')
        def _backward():
            self.grad = self.grad + out.grad.reshape(original_shape)
        out._backward = _backward
        return out
    
    def flatten(self):
        """Aplatit le tenseur en 1D (ou 2D si batch)"""
        if self.data.ndim == 1:
            return self
        original_shape = self.data.shape
        out = Tensor(self.data.reshape(self.data.shape[0], -1), (self,), 'flatten')
        def _backward():
            self.grad = self.grad + out.grad.reshape(original_shape)
        out._backward = _backward
        return out
    
    def max(self, axis=None, keepdims=False):
        """Retourne le maximum"""
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), (self,), 'max')
        def _backward():
            # Gradient seulement pour les éléments max
            mask = (self.data == np.max(self.data, axis=axis, keepdims=True))
            self.grad = self.grad + mask * out.grad
        out._backward = _backward
        return out
    
    # -------------------------------------------------------------------------
    # Backward pass avec tri topologique
    # -------------------------------------------------------------------------
    
    def _build_topo(self):
        """Construit l'ordre topologique du graphe de calcul"""
        topo = []
        visited = set()
        
        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)
        
        build(self)
        return topo
    
    def backward(self):
        """Rétropropagation des gradients via tri topologique"""
        topo = self._build_topo()
        
        # Initialiser le gradient de sortie à 1
        self.grad = np.ones_like(self.data, dtype=np.float64)
        
        # Parcourir le graphe en ordre inverse
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        """Remet le gradient à zéro"""
        self.grad = np.zeros_like(self.data, dtype=np.float64)


# =============================================================================
# MODULES ET COUCHES
# =============================================================================

class Module:
    """Classe de base pour les modules (similaire à torch.nn.Module)"""
    
    def __init__(self):
        self._parameters = []
        self._modules = []
    
    def parameters(self):
        """Retourne tous les paramètres du module et sous-modules"""
        params = list(self._parameters)
        for module in self._modules:
            params.extend(module.parameters())
        return params
    
    def zero_grad(self):
        """Remet tous les gradients à zéro"""
        for param in self.parameters():
            param.zero_grad()
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    """Couche linéaire (Dense): y = x @ W + b"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Initialisation Xavier/Glorot
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(np.random.randn(in_features, out_features) * scale)
        self._parameters.append(self.weight)
        
        if bias:
            self.bias = Tensor(np.zeros(out_features))
            self._parameters.append(self.bias)
        else:
            self.bias = None
    
    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Sequential(Module):
    """Conteneur séquentiel de modules"""
    
    def __init__(self, *modules):
        super().__init__()
        self._modules = list(modules)
    
    def forward(self, x):
        for module in self._modules:
            x = module(x)
        return x


class ReLU(Module):
    """Activation ReLU comme module"""
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    """Activation Tanh comme module"""
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    """Activation Sigmoid comme module"""
    def forward(self, x):
        return x.sigmoid()


class Flatten(Module):
    """Aplatit l'entrée (utile entre Conv et Linear)"""
    def forward(self, x):
        return x.flatten()


# =============================================================================
# COUCHES CONVOLUTIONNELLES
# =============================================================================

class Conv2d(Module):
    """
    Couche de convolution 2D.
    Input: (batch, in_channels, height, width)
    Output: (batch, out_channels, out_height, out_width)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialisation Xavier
        k = 1.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1])
        self.weight = Tensor(
            np.random.uniform(-np.sqrt(k), np.sqrt(k), 
                            (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        )
        self.bias = Tensor(np.zeros(out_channels))
        self._parameters = [self.weight, self.bias]
    
    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.data.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        
        # Padding
        if ph > 0 or pw > 0:
            x_padded = np.pad(x.data, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')
        else:
            x_padded = x.data
        
        # Output dimensions
        out_height = (in_height + 2 * ph - kh) // sh + 1
        out_width = (in_width + 2 * pw - kw) // sw + 1
        
        # im2col transformation pour efficacité
        col = np.zeros((batch_size, in_channels, kh, kw, out_height, out_width))
        for i in range(kh):
            i_max = i + sh * out_height
            for j in range(kw):
                j_max = j + sw * out_width
                col[:, :, i, j, :, :] = x_padded[:, :, i:i_max:sh, j:j_max:sw]
        
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_height * out_width, -1)
        weight_col = self.weight.data.reshape(self.out_channels, -1).T
        
        out_data = col @ weight_col + self.bias.data
        out_data = out_data.reshape(batch_size, out_height, out_width, self.out_channels)
        out_data = out_data.transpose(0, 3, 1, 2)
        
        out = Tensor(out_data, (x, self.weight, self.bias), 'conv2d')
        
        # Sauvegarder pour backward
        saved_col = col
        saved_x_padded_shape = x_padded.shape
        saved_x_shape = x.data.shape
        
        def _backward():
            # Gradient par rapport au bias
            self.bias.grad = self.bias.grad + np.sum(out.grad, axis=(0, 2, 3))
            
            # Gradient par rapport aux poids
            dout_reshaped = out.grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
            dW = saved_col.T @ dout_reshaped
            self.weight.grad = self.weight.grad + dW.T.reshape(self.weight.data.shape)
            
            # Gradient par rapport à l'entrée
            dcol = dout_reshaped @ self.weight.data.reshape(self.out_channels, -1)
            dcol = dcol.reshape(batch_size, out_height, out_width, in_channels, kh, kw)
            dcol = dcol.transpose(0, 3, 4, 5, 1, 2)
            
            dx_padded = np.zeros(saved_x_padded_shape)
            for i in range(kh):
                i_max = i + sh * out_height
                for j in range(kw):
                    j_max = j + sw * out_width
                    dx_padded[:, :, i:i_max:sh, j:j_max:sw] += dcol[:, :, i, j, :, :]
            
            if ph > 0 or pw > 0:
                x.grad = x.grad + dx_padded[:, :, ph:-ph, pw:-pw]
            else:
                x.grad = x.grad + dx_padded
        
        out._backward = _backward
        return out


class MaxPool2d(Module):
    """
    Max Pooling 2D.
    Réduit les dimensions spatiales en prenant le maximum sur des fenêtres.
    """
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        if not isinstance(self.stride, tuple):
            self.stride = (self.stride, self.stride)
    
    def forward(self, x):
        batch_size, channels, in_height, in_width = x.data.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        
        out_height = (in_height - kh) // sh + 1
        out_width = (in_width - kw) // sw + 1
        
        # Reshape pour le pooling
        out_data = np.zeros((batch_size, channels, out_height, out_width))
        max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * sh
                w_start = j * sw
                window = x.data[:, :, h_start:h_start+kh, w_start:w_start+kw]
                
                # Trouver le max et son indice
                window_reshaped = window.reshape(batch_size, channels, -1)
                max_idx = np.argmax(window_reshaped, axis=2)
                out_data[:, :, i, j] = np.max(window_reshaped, axis=2)
                
                # Sauvegarder les indices pour backward
                max_indices[:, :, i, j, 0] = h_start + max_idx // kw
                max_indices[:, :, i, j, 1] = w_start + max_idx % kw
        
        out = Tensor(out_data, (x,), 'maxpool2d')
        saved_indices = max_indices
        saved_input_shape = x.data.shape
        
        def _backward():
            dx = np.zeros(saved_input_shape)
            for i in range(out_height):
                for j in range(out_width):
                    for b in range(batch_size):
                        for c in range(channels):
                            h_idx = saved_indices[b, c, i, j, 0]
                            w_idx = saved_indices[b, c, i, j, 1]
                            dx[b, c, h_idx, w_idx] += out.grad[b, c, i, j]
            x.grad = x.grad + dx
        
        out._backward = _backward
        return out


class AvgPool2d(Module):
    """
    Average Pooling 2D.
    Réduit les dimensions spatiales en prenant la moyenne sur des fenêtres.
    """
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        if not isinstance(self.stride, tuple):
            self.stride = (self.stride, self.stride)
    
    def forward(self, x):
        batch_size, channels, in_height, in_width = x.data.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        
        out_height = (in_height - kh) // sh + 1
        out_width = (in_width - kw) // sw + 1
        
        out_data = np.zeros((batch_size, channels, out_height, out_width))
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * sh
                w_start = j * sw
                window = x.data[:, :, h_start:h_start+kh, w_start:w_start+kw]
                out_data[:, :, i, j] = np.mean(window, axis=(2, 3))
        
        out = Tensor(out_data, (x,), 'avgpool2d')
        pool_size = kh * kw
        
        def _backward():
            dx = np.zeros(x.data.shape)
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * sh
                    w_start = j * sw
                    dx[:, :, h_start:h_start+kh, w_start:w_start+kw] += \
                        out.grad[:, :, i:i+1, j:j+1] / pool_size
            x.grad = x.grad + dx
        
        out._backward = _backward
        return out


class Dropout(Module):
    """
    Dropout pour la régularisation.
    Désactive aléatoirement des neurones pendant l'entraînement.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.training = True
        self.mask = None
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        # Créer le masque de dropout
        self.mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float64)
        scale = 1.0 / (1.0 - self.p)  # Inverted dropout
        
        out = Tensor(x.data * self.mask * scale, (x,), 'dropout')
        mask = self.mask
        
        def _backward():
            x.grad = x.grad + out.grad * mask * scale
        out._backward = _backward
        
        return out
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class BatchNorm1d(Module):
    """
    Batch Normalization 1D pour accélérer l'entraînement.
    Normalise les activations sur le batch.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = True
        
        # Paramètres apprenables
        self.gamma = Tensor(np.ones(num_features))
        self.beta = Tensor(np.zeros(num_features))
        self._parameters = [self.gamma, self.beta]
        
        # Running statistics (non apprenables)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x):
        if self.training:
            mean = np.mean(x.data, axis=0)
            var = np.var(x.data, axis=0)
            
            # Mettre à jour les running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalisation
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = self.gamma.data * x_norm + self.beta.data
        
        out = Tensor(out_data, (x, self.gamma, self.beta), 'batchnorm')
        
        # Sauvegarder pour backward
        saved_x_norm = x_norm
        saved_var = var
        saved_mean = mean
        batch_size = x.data.shape[0]
        
        def _backward():
            # Gradient par rapport à gamma et beta
            self.gamma.grad = self.gamma.grad + np.sum(out.grad * saved_x_norm, axis=0)
            self.beta.grad = self.beta.grad + np.sum(out.grad, axis=0)
            
            # Gradient par rapport à x
            dx_norm = out.grad * self.gamma.data
            dvar = np.sum(dx_norm * (x.data - saved_mean) * -0.5 * (saved_var + self.eps) ** -1.5, axis=0)
            dmean = np.sum(dx_norm * -1 / np.sqrt(saved_var + self.eps), axis=0) + dvar * np.mean(-2 * (x.data - saved_mean), axis=0)
            x.grad = x.grad + dx_norm / np.sqrt(saved_var + self.eps) + dvar * 2 * (x.data - saved_mean) / batch_size + dmean / batch_size
        
        out._backward = _backward
        return out
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


# =============================================================================
# FONCTIONS DE PERTE
# =============================================================================

class CrossEntropyLoss:
    """
    Cross-Entropy Loss combinée avec Softmax pour la stabilité numérique.
    Formule: -sum(y_true * log(softmax(logits)))
    """
    
    def __call__(self, logits, targets):
        return self.forward(logits, targets)
    
    def forward(self, logits, targets):
        """
        logits: Tensor de forme (batch_size, num_classes) - scores bruts
        targets: array numpy de forme (batch_size,) - indices des classes
        """
        batch_size = logits.data.shape[0]
        
        # Softmax stable (LogSumExp trick)
        shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp
        
        # Sélectionner les log-probabilités des classes correctes
        if isinstance(targets, Tensor):
            targets = targets.data.astype(int)
        
        correct_log_probs = log_probs[np.arange(batch_size), targets]
        loss_value = -np.mean(correct_log_probs)
        
        # Créer le tenseur de sortie
        out = Tensor(loss_value, (logits,), 'cross_entropy')
        
        # Calculer le gradient
        probs = np.exp(log_probs)
        grad = probs.copy()
        grad[np.arange(batch_size), targets] -= 1
        grad /= batch_size
        
        def _backward():
            logits.grad = logits.grad + grad * out.grad
        out._backward = _backward
        
        return out


class MSELoss:
    """Mean Squared Error Loss"""
    
    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)
    
    def forward(self, predictions, targets):
        if not isinstance(targets, Tensor):
            targets = Tensor(targets, requires_grad=False)
        diff = predictions - targets
        return (diff * diff).mean()


# =============================================================================
# OPTIMISEURS
# =============================================================================

class Optimizer:
    """Classe de base pour les optimiseurs"""
    
    def __init__(self, params):
        self.params = params
    
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
    
    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent avec momentum optionnel"""
    
    def __init__(self, params, learning_rate=0.01, momentum=0.0):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        for i, param in enumerate(self.params):
            self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * param.grad
            param.data = param.data + self.velocities[i]


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) - Optimiseur standard industriel
    Combine les avantages de AdaGrad et RMSProp
    """
    
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Moments de premier et second ordre
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.params):
            # Mise à jour des moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
            
            # Correction du biais
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Mise à jour des paramètres
            param.data = param.data - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """Adam avec Weight Decay découplé"""
    
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.params):
            # Weight decay découplé
            param.data = param.data - self.learning_rate * self.weight_decay * param.data
            
            # Adam classique
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param.data = param.data - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSProp(Optimizer):
    """RMSProp optimizer"""
    
    def __init__(self, params, learning_rate=0.01, decay=0.9, eps=1e-8):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.decay = decay
        self.eps = eps
        self.cache = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        for i, param in enumerate(self.params):
            self.cache[i] = self.decay * self.cache[i] + (1 - self.decay) * (param.grad ** 2)
            param.data = param.data - self.learning_rate * param.grad / (np.sqrt(self.cache[i]) + self.eps)


class Adagrad(Optimizer):
    """Adagrad optimizer - adapte le learning rate par paramètre"""
    
    def __init__(self, params, learning_rate=0.01, eps=1e-8):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.eps = eps
        self.cache = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        for i, param in enumerate(self.params):
            self.cache[i] = self.cache[i] + param.grad ** 2
            param.data = param.data - self.learning_rate * param.grad / (np.sqrt(self.cache[i]) + self.eps)


class Momentum(Optimizer):
    """SGD avec Momentum (comme dans le cours a.ipynb)"""
    
    def __init__(self, params, learning_rate=0.01, momentum=0.9):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        for i, param in enumerate(self.params):
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * param.grad
            param.data = param.data + self.velocity[i]


# =============================================================================
# LEARNING RATE SCHEDULERS
# =============================================================================

class LRScheduler:
    """Classe de base pour les schedulers de learning rate"""
    
    def __init__(self, optimizer, initial_lr):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.lr = initial_lr
        self.iteration = 0
    
    def step(self, metric=None):
        self.update_lr(metric)
        self.optimizer.learning_rate = self.lr
    
    def update_lr(self, metric=None):
        raise NotImplementedError


class LRSchedulerOnPlateau(LRScheduler):
    """Réduit le learning rate quand une métrique stagne"""
    
    def __init__(self, optimizer, initial_lr, patience=10, factor=0.1, min_lr=1e-6, mode='min'):
        super().__init__(optimizer, initial_lr)
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
    
    def update_lr(self, metric):
        improved = (self.mode == 'min' and metric < self.best_metric) or \
                   (self.mode == 'max' and metric > self.best_metric)
        
        if improved:
            self.best_metric = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            self.lr = max(self.lr * self.factor, self.min_lr)
            self.num_bad_epochs = 0
            print(f"Reducing learning rate to {self.lr}")


class StepLR(LRScheduler):
    """Réduit le learning rate par un facteur tous les step_size epochs"""
    
    def __init__(self, optimizer, initial_lr, step_size=10, gamma=0.1):
        super().__init__(optimizer, initial_lr)
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0
    
    def update_lr(self, metric=None):
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.lr = self.lr * self.gamma
            print(f"StepLR: Reducing learning rate to {self.lr}")


class ExponentialLR(LRScheduler):
    """Décroissance exponentielle du learning rate"""
    
    def __init__(self, optimizer, initial_lr, gamma=0.95):
        super().__init__(optimizer, initial_lr)
        self.gamma = gamma
    
    def update_lr(self, metric=None):
        self.lr = self.lr * self.gamma


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing du learning rate"""
    
    def __init__(self, optimizer, initial_lr, T_max, eta_min=0):
        super().__init__(optimizer, initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min
        self.epoch = 0
    
    def update_lr(self, metric=None):
        self.epoch += 1
        self.lr = self.eta_min + (self.initial_lr - self.eta_min) * (1 + np.cos(np.pi * self.epoch / self.T_max)) / 2


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def clip_grad_norm(params, max_norm):
    """
    Gradient clipping pour éviter l'explosion des gradients.
    Très utile pour les RNN.
    """
    total_norm = 0.0
    for param in params:
        total_norm += np.sum(param.grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for param in params:
            param.grad = param.grad * scale
    
    return total_norm


def save_model(module, filepath):
    """Sauvegarde les paramètres du modèle"""
    params = {i: p.data for i, p in enumerate(module.parameters())}
    np.savez(filepath, **{str(k): v for k, v in params.items()})
    print(f"Model saved to {filepath}")


def load_model(module, filepath):
    """Charge les paramètres du modèle"""
    data = np.load(filepath)
    for i, param in enumerate(module.parameters()):
        param.data = data[str(i)]
    print(f"Model loaded from {filepath}")


# =============================================================================
# FONCTIONS DU COURS (a.ipynb) - Fonctions d'activation externes
# =============================================================================

def sin_d(dual_number: Tensor):
    """Sinus avec autodiff inverse (comme dans le cours a.ipynb)"""
    out = Tensor(np.sin(dual_number.data), (dual_number,), 'sin')
    def _backward():
        dual_number.grad += np.cos(dual_number.data) * out.grad
    out._backward = _backward
    return out


def cos_d(dual_number: Tensor):
    """Cosinus avec autodiff inverse (comme dans le cours a.ipynb)"""
    out = Tensor(np.cos(dual_number.data), (dual_number,), 'cos')
    def _backward():
        dual_number.grad += -np.sin(dual_number.data) * out.grad
    out._backward = _backward
    return out


def tan_d(dual_number: Tensor):
    """Tangente avec autodiff inverse (comme dans le cours a.ipynb)"""
    out = Tensor(np.tan(dual_number.data), (dual_number,), 'tan')
    def _backward():
        dual_number.grad += (1 / np.cos(dual_number.data)**2) * out.grad
    out._backward = _backward
    return out


def sigmoid_d(dual_number: Tensor):
    """Sigmoid avec autodiff inverse (comme dans le cours a.ipynb)"""
    out = Tensor(1 / (1 + np.exp(-dual_number.data)), (dual_number,), 'sigmoid')
    def _backward():
        dual_number.grad += out.data * (1 - out.data) * out.grad
    out._backward = _backward
    return out


def tanh_d(dual_number: Tensor):
    """Tanh avec autodiff inverse (comme dans le cours a.ipynb)"""
    out = Tensor(np.tanh(dual_number.data), (dual_number,), 'tanh')
    def _backward():
        dual_number.grad += (1 - out.data**2) * out.grad
    out._backward = _backward
    return out


def relu_d(dual_number: Tensor):
    """ReLU avec autodiff inverse"""
    out = Tensor(np.maximum(0, dual_number.data), (dual_number,), 'relu')
    def _backward():
        dual_number.grad += (dual_number.data > 0).astype(float) * out.grad
    out._backward = _backward
    return out


def sqrt_d(dual_number: Tensor):
    """Racine carrée avec autodiff inverse (comme dans le cours a.ipynb)"""
    out = Tensor(np.sqrt(dual_number.data), (dual_number,), 'sqrt')
    def _backward():
        dual_number.grad += (0.5 / np.sqrt(dual_number.data)) * out.grad
    out._backward = _backward
    return out


def pow_d(dual_number: Tensor, power: int):
    """Puissance avec autodiff inverse (comme dans le cours a.ipynb)"""
    out = Tensor(dual_number.data ** power, (dual_number,), f'**{power}')
    def _backward():
        dual_number.grad += (power * dual_number.data ** (power - 1)) * out.grad
    out._backward = _backward
    return out


def exp_d(dual_number: Tensor):
    """Exponentielle avec autodiff inverse"""
    out = Tensor(np.exp(dual_number.data), (dual_number,), 'exp')
    def _backward():
        dual_number.grad += out.data * out.grad
    out._backward = _backward
    return out


def log_d(dual_number: Tensor):
    """Logarithme avec autodiff inverse"""
    out = Tensor(np.log(dual_number.data + 1e-8), (dual_number,), 'log')
    def _backward():
        dual_number.grad += out.grad / (dual_number.data + 1e-8)
    out._backward = _backward
    return out


def softmax_d(dual_number: Tensor):
    """Softmax avec autodiff inverse (comme dans le cours a.ipynb)"""
    exp_vals = np.exp(dual_number.data - np.max(dual_number.data))  # Stabilité numérique
    out = Tensor(exp_vals / np.sum(exp_vals), (dual_number,), 'softmax')
    def _backward():
        dual_number.grad += out.grad
    out._backward = _backward
    return out


# =============================================================================
# FONCTIONS RÉSEAU DU COURS (a.ipynb)
# =============================================================================

def func_nn(x, W1, b1, W2, b2):
    """
    MLP avec 1 couche cachée (comme dans le cours a.ipynb)
    Architecture: x -> tanh(W1*x + b1) -> W2*h + b2 -> y
    """
    h1 = tanh_d(W1 * x + b1)
    y = W2 * h1 + b2
    return y


def mse(y, y_hat):
    """
    Mean Squared Error (comme dans le cours a.ipynb)
    loss = (y - y_hat)^2
    """
    return (y - y_hat) ** 2


def func_rnn(x, h, Wx, Wh, b):
    """
    Cellule RNN simple (comme dans le cours a.ipynb)
    h_new = tanh(Wx*x + Wh*h + b)
    """
    h_new = tanh_d(Wx * x + Wh * h + b)
    return h_new


def func_nn_output(h, Wy, by):
    """
    Couche de sortie pour RNN (comme dans le cours a.ipynb)
    y = Wy*h + by
    """
    y = Wy * h + by
    return y
