from collections.abc import Callable

import torch


class Solver:
    def __init__(
        self,
        A: torch.Tensor | Callable,
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        tol: float = 1e-6,
        max_iter: int = 1000,
        device: str = 'cuda',
    ):  
        self.input_dtype = self.check_dtype(A, b, x0)
        self.A = A.clone().to(device) if isinstance(A, torch.Tensor) else None
        self.A_func = A if isinstance(A, Callable) else lambda x: torch.mm(self.A, x)
        self.N = self.check_A()
        if not b.shape == torch.Size([self.N, 1]):
            raise ValueError(f"Expected b to have shape torch.size([{self.N}, 1]), got {b.shape}.")
        self.b = b.clone().to(device)
        if x0 is not None and not x0.shape == torch.Size([self.N, 1]):
            raise ValueError(f"Expected x0 to have shape torch.size([{self.N}, 1]), got {x0.shape}.")
        self.x = torch.zeros(self.N, 1, dtype=self.input_dtype, device=device) if x0 is None else x0.clone().to(device)
        self.r = self.b - self.A_func(self.x) if x0 is not None else b
        self.tol = tol
        self.max_iter = max_iter
        self.iter = 0
        self.residual = 0
        self.residuals = []
        self.device = device
        self.solver_name = 'Solver'

    def check_dtype(self, A, b, x0):
        if not A.dtype == b.dtype:
            raise ValueError(f"Expected A and b to have the same dtype, got {A.dtype} and {b.dtype}.")
        if x0 is not None and not A.dtype == x0.dtype:
            raise ValueError(f"Expected A and x0 to have the same dtype, got {A.dtype} and {x0.dtype}.")
        return A.dtype

    def check_A(self):
        if self.A is not None:
            if self.A.ndim != 2:
                raise ValueError(f"Expected A to be 2D, got {self.A.ndim}D.")
            try: 
                torch.linalg.cholesky(self.A)
            except RuntimeError:
                raise ValueError("Matrix A is not positive definite.")
        return self.A.shape[0]

    def solve(self):
        raise NotImplementedError
    
    def log_residual(self, residual):
        self.residuals.append(residual.item())

    def __call__(self):
        return self.solve()
    
    def __repr__(self):
        return f"{self.solver_name}(A={self.A.shape}, b={self.b.shape}, x0={self.x.shape}, tol={self.tol}, max_iter={self.max_iter})"
        


class CG(Solver):
    def __init__(
        self, 
        A: torch.Tensor | Callable,
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        tol: float = 1e-6,
        max_iter: int = 1000,
        device: str = 'cuda',
    ):
        super().__init__(A, b, x0, tol, max_iter, device)
        self.p = self.r.clone()
        self.solver_name = 'CG'
        self.check_A()

    def solve(self):
        for self.iter in range(self.max_iter):
            Ap = self.A_func(self.p)
            rTr = torch.mm(self.r.mT, self.r)
            alpha = rTr / torch.mm(self.p.mT, Ap)
            self.x += alpha * self.p
            self.r -= alpha * Ap
            self.log_residual(
                (residual := torch.linalg.matrix_norm(self.r))
            )
            self.residual = residual
            if residual < self.tol:
                break
            beta = torch.mm(self.r.mT, self.r) / rTr
            self.p = self.r + beta * self.p
        return {
            'x': self.x,
            'iter': self.iter,
            'residual': self.residual
        }


class GD(Solver):
    def __init__(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        tol: float = 1e-6,
        max_iter: int = 10000,
        device: str = 'cuda',
    ):
        super().__init__(A, b, x0, tol, max_iter, device)
        self.solver_name = 'GD'
        self.check_A()
    
    def solve(self):
        for self.iter in range(self.max_iter):
            Ar = self.A_func(self.r)
            gamma = torch.mm(self.r.mT, self.r) / torch.mm(self.r.mT, Ar)
            self.x += gamma * self.r
            self.log_residual(
                (residual := torch.linalg.matrix_norm(self.r))
            )
            self.residual = residual
            if residual < self.tol:
                break
            self.r -= gamma * Ar
        return {
            'x': self.x,
            'iter': self.iter,
            'residual': self.residual
        }
