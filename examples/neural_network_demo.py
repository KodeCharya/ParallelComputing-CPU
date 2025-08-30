"""
Comprehensive neural network demonstration
"""

import numpy as np
import time
from parallelcore import VirtualCoreManager, Tensor, Variable, AutogradEngine
from parallelcore.nn import Linear, ReLU, Sigmoid, Dropout, BatchNorm1d
from parallelcore.nn.loss import MSELoss, CrossEntropyLoss
from parallelcore.nn.optimizer import Adam, SGD


class MLP:
    """Multi-layer perceptron for classification"""
    
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int,
                 dropout_p: float = 0.1):
        self.layers = []
        self.activations = []
        self.dropouts = []
        self.batch_norms = []
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(Linear(prev_size, hidden_size))
            self.batch_norms.append(BatchNorm1d(hidden_size))
            self.activations.append(ReLU())
            self.dropouts.append(Dropout(dropout_p))
            prev_size = hidden_size
        
        # Output layer
        self.layers.append(Linear(prev_size, output_size))
        self.output_activation = Sigmoid() if output_size == 1 else None
    
    def forward(self, x: Variable) -> Variable:
        """Forward pass through the network"""
        for i, (layer, bn, activation, dropout) in enumerate(
            zip(self.layers[:-1], self.batch_norms, self.activations, self.dropouts)
        ):
            x = layer(x)
            x = bn(x)
            x = activation(x)
            x = dropout(x)
        
        # Output layer
        x = self.layers[-1](x)
        if self.output_activation:
            x = self.output_activation(x)
        
        return x
    
    def parameters(self) -> list:
        """Get all parameters"""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        for bn in self.batch_norms:
            params.extend(bn.parameters())
        return params
    
    def train(self):
        """Set training mode"""
        for layer in self.layers:
            layer.train()
        for bn in self.batch_norms:
            bn.train()
        for dropout in self.dropouts:
            dropout.train()
    
    def eval(self):
        """Set evaluation mode"""
        for layer in self.layers:
            layer.eval()
        for bn in self.batch_norms:
            bn.eval()
        for dropout in self.dropouts:
            dropout.eval()


def generate_classification_data(n_samples: int = 1000, n_features: int = 20, 
                               n_classes: int = 3) -> tuple:
    """Generate synthetic classification dataset"""
    np.random.seed(42)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create class-dependent patterns
    class_centers = np.random.randn(n_classes, n_features) * 2
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Assign class based on distance to centers
        distances = [np.linalg.norm(X[i] - center) for center in class_centers]
        y[i] = np.argmin(distances)
        
        # Add some class-specific noise
        X[i] += 0.5 * class_centers[y[i]] + 0.1 * np.random.randn(n_features)
    
    return X, y


def train_classification_model():
    """Train a classification model with parallel processing"""
    print("=== Neural Network Classification Demo ===")
    
    # Generate dataset
    X_train, y_train = generate_classification_data(2000, 50, 5)
    X_test, y_test = generate_classification_data(500, 50, 5)
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Create autograd engine
    engine = AutogradEngine()
    
    # Convert to variables
    X_train_var = engine.create_variable(X_train.astype(np.float32))
    y_train_var = engine.create_variable(y_train, requires_grad=False)
    X_test_var = engine.create_variable(X_test.astype(np.float32))
    y_test_var = engine.create_variable(y_test, requires_grad=False)
    
    # Create model
    model = MLP(
        input_size=50,
        hidden_sizes=[128, 64, 32],
        output_size=5,
        dropout_p=0.2
    )
    
    # Loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    batch_size = 32
    epochs = 20
    n_batches = len(X_train) // batch_size
    
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}")
    print(f"Batches per epoch: {n_batches}")
    
    # Training loop
    model.train()
    training_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        
        for batch_idx in range(n_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = Variable(Tensor(X_train[batch_indices]), engine=engine)
            y_batch = Variable(Tensor(y_train[batch_indices]), requires_grad=False, engine=engine)
            
            # Forward pass
            output = model.forward(X_batch)
            loss = criterion(output, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches
        training_losses.append(avg_loss)
        
        # Evaluate on test set every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            test_output = model.forward(X_test_var)
            test_loss = criterion(test_output, y_test_var)
            
            # Calculate accuracy
            predictions = np.argmax(test_output.data, axis=1)
            accuracy = np.mean(predictions == y_test) * 100
            
            print(f"Epoch {epoch:2d}: Loss={avg_loss:.6f}, Test Loss={test_loss.item():.6f}, "
                  f"Accuracy={accuracy:.2f}%, Time={epoch_time:.2f}s")
            
            model.train()
        else:
            print(f"Epoch {epoch:2d}: Loss={avg_loss:.6f}, Time={epoch_time:.2f}s")
    
    print("\nTraining completed!")
    
    # Final evaluation
    model.eval()
    final_output = model.forward(X_test_var)
    final_predictions = np.argmax(final_output.data, axis=1)
    final_accuracy = np.mean(final_predictions == y_test) * 100
    
    print(f"Final test accuracy: {final_accuracy:.2f}%")
    
    return model, training_losses


def example_tensor_performance():
    """Demonstrate tensor performance with SIMD"""
    print("\n=== Tensor Performance Comparison ===")
    
    sizes = [100, 1000, 5000, 10000]
    
    for size in sizes:
        print(f"\nTensor size: {size}x{size}")
        
        # Create tensors
        a = Tensor.randn(size, size)
        b = Tensor.randn(size, size)
        
        # NumPy baseline
        a_np = a.numpy()
        b_np = b.numpy()
        
        start_time = time.time()
        c_np = a_np + b_np
        d_np = a_np * b_np
        e_np = np.matmul(a_np, b_np)
        numpy_time = time.time() - start_time
        
        # Tensor operations
        start_time = time.time()
        c_tensor = a + b
        d_tensor = a * b
        e_tensor = a.matmul(b)
        tensor_time = time.time() - start_time
        
        print(f"  NumPy time: {numpy_time:.4f}s")
        print(f"  Tensor time: {tensor_time:.4f}s")
        print(f"  Speedup: {numpy_time / tensor_time:.2f}x")


def example_parallel_reduction():
    """Demonstrate parallel reduction operations"""
    print("\n=== Parallel Reduction Example ===")
    
    # Large dataset
    data = list(range(1000000))
    
    with VirtualCoreManager() as manager:
        # Parallel sum
        start_time = time.time()
        parallel_sum = manager.reduce_parallel(lambda x, y: x + y, data)
        parallel_time = time.time() - start_time
        
        # Sequential sum for comparison
        start_time = time.time()
        sequential_sum = sum(data)
        sequential_time = time.time() - start_time
        
        print(f"Dataset size: {len(data):,}")
        print(f"Sequential sum: {sequential_sum:,} ({sequential_time:.4f}s)")
        print(f"Parallel sum: {parallel_sum:,} ({parallel_time:.4f}s)")
        print(f"Speedup: {sequential_time / parallel_time:.2f}x")
        print(f"Results match: {parallel_sum == sequential_sum}")


if __name__ == "__main__":
    print("ParallelCore Library - Comprehensive Demo")
    print("=" * 50)
    
    # Run all examples
    example_parallel_processing()
    example_tensor_operations()
    train_classification_model()
    example_tensor_performance()
    example_parallel_reduction()
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")