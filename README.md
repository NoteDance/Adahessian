# Adahessian

**Overview**:

The **Adahessian Optimizer** is an advanced second-order optimization algorithm that leverages the Hessian trace (approximated using Hutchinson's method) to adaptively scale learning rates for each parameter. Adahessian extends first-order optimization techniques like Adam by incorporating curvature information from the loss surface, which enables better adaptation to the optimization landscape, especially for highly non-convex problems.

**Parameters**:

- **`learning_rate`** *(float)*: Initial learning rate (default: `0.1`).
- **`beta1`** *(float)*: Exponential decay rate for the first moment estimates (default: `0.9`).
- **`beta2`** *(float)*: Exponential decay rate for the Hessian diagonal squared estimates (default: `0.999`).
- **`epsilon`** *(float)*: Small value to prevent division by zero (default: `1e-8`).
- **`weight_decay`** *(float)*: L2 regularization factor for weights (default: `0.0`).
- **`hessian_power`** *(float)*: Scaling factor for the Hessian diagonal (default: `1.0`).
- **`update_each`** *(int)*: Frequency (in steps) for Hessian trace updates (default: `1`).
- **`n_samples`** *(int)*: Number of samples for Hutchinson’s approximation (default: `1`).
- **`avg_conv_kernel`** *(bool)*: Whether to average Hessian diagonals over convolutional kernel dimensions (default: `False`).
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="adahessian")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf

# Define model and loss
model = tf.keras.Sequential([...])
loss_fn = tf.keras.losses.MeanSquaredError()

# Initialize optimizer
optimizer = Adahessian(
    learning_rate=0.01, 
    beta1=0.9, 
    beta2=0.999, 
    weight_decay=0.01
)

# Training step
@tf.function
def train_step(x, y, model, optimizer):
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables), tape)

# Training loop
for epoch in range(epochs):
    for x_batch, y_batch in dataset:
        train_step(x_batch, y_batch, model, optimizer)
```
