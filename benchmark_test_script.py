import torch
import torch.nn as nn
from typing import Callable
import time
from statistics import mean
from torch.utils.cpp_extension import load_inline
from torch.utils.cpp_extension import load_inline


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """Simple MLP: linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""

    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.gelu(x)
        return x


def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    # Define a model (with random weights)
    model = MLP(dim, num_layers).to(get_device())
    # Define an input (random)
    x = torch.randn(batch_size, dim, device=get_device())

    def run():
        # Run the model `num_steps` times (note: no optimizer updates)
        for step in range(num_steps):
            # Forward
            y = model(x).mean()
            # Backward
            y.backward()
    return run


def run_operation1(dim: int, operation: Callable) -> Callable:
    # Setup: create one random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda: operation(x)


def run_operation2(dim: int, operation: Callable) -> Callable:
    # Setup: create two random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda: operation(x, y)


def benchmarking():
    benchmark("sleep", lambda: time.sleep(50 / 1000))
    if torch.cuda.is_available():
        dims = (1024, 2048, 4096, 8192, 16384)  # @inspect dims
    else:
        dims = (1024, 2048)  # @inspect dims

    matmul_results = []
    for dim in dims:
        # @ inspect dim
        result = benchmark(f"matmul(dim={dim})", run_operation2(
            dim=dim, operation=lambda a, b: a @ b))
        matmul_results.append((dim, result))  # @inspect matmul_results

    dim = 256  # @inspect dim
    num_layers = 4  # @inspect num_layers
    batch_size = 256  # @inspect batch_size
    num_steps = 2  # @inspect num_steps
    mlp_base = benchmark("run_mlp", run_mlp(dim=dim, num_layers=num_layers,
                         batch_size=batch_size, num_steps=num_steps))  # @inspect mlp_base

    step_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x num_steps)",
                           run_mlp(dim=dim, num_layers=num_layers,
                                   # @inspect result, @inspect scale, @inspect num_steps
                                   batch_size=batch_size, num_steps=scale * num_steps))
        step_results.append((scale, result))  # @inspect step_results

    layer_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x num_layers)",
                           run_mlp(dim=dim, num_layers=scale * num_layers,
                                   # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
                                   batch_size=batch_size, num_steps=num_steps))
        layer_results.append((scale, result))  # @inspect layer_results

    batch_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x batch_size)",
                           run_mlp(dim=dim, num_layers=num_layers,
                                   # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
                                   batch_size=scale * batch_size, num_steps=num_steps))
        batch_results.append((scale, result))  # @inspect batch_results

    dim_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x dim)",
                           run_mlp(dim=scale * dim, num_layers=num_layers,
                                   # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
                                   batch_size=batch_size, num_steps=num_steps))
        dim_results.append((scale, result))  # @inspect dim_results


def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Time it for real now!
    times: list[float] = []  # @inspect times, @inspect description
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = time.time()
        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # @inspect times
    mean_time = mean(times)  # @inspect mean_time
    return mean_time


if __name__ == "__main__":
    benchmarking()
