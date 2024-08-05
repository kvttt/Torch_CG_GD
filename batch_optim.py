from collections.abc import Callable

import torch


class BatchSolver:
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
        self.A_func = A if self.A is None else lambda x: torch.bmm(self.A, x)
        self.B, self.N = self.check_A()
        if not b.shape == torch.Size([self.B, self.N, 1]):
            raise ValueError(f"Expected b to have shape torch.size([{self.B}, {self.N}, 1]), got {b.shape}.")       
        self.b = b.clone().to(device)
        if x0 is not None and not x0.shape == torch.Size([self.B, self.N, 1]):
            raise ValueError(f"Expected x0 to have shape torch.size([{self.B}, {self.N}, 1]), got {x0.shape}.")
        self.x = torch.zeros(self.B, self.N, 1, dtype=self.input_dtype, device=device) if x0 is None else x0.clone().to(device)
        self.r = self.b - self.A_func(self.x) if x0 is not None else b # B, N, 1
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
            if self.A.ndim != 3:
                raise ValueError(f"Expected A to be 3D, got {self.A.ndim}D.")
            try: 
                torch.linalg.cholesky(self.A)
            except RuntimeError:
                raise ValueError("Matrix A is not positive definite.")
        return self.A.shape[0], self.A.shape[1] # B, N
    
    def solve(self):
        raise NotImplementedError
    
    def log_residual(self, residual):
        self.residuals.append(residual.item())

    def __call__(self):
        return self.solve()
    
    def __repr__(self):
        return f"{self.solver_name}(A={self.A.shape}, b={self.b.shape}, x0={self.x.shape}, tol={self.tol}, max_iter={self.max_iter})"
        


class CG_Batch(BatchSolver):
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
        self.p = self.r.clone() # B, N, 1
        self.solver_name = 'CG'

    def solve(self):
        for self.iter in range(self.max_iter):
            Ap = self.A_func(self.p) # B, N, 1
            rTr = torch.bmm(self.r.mT, self.r) # B, 1, 1
            alpha = rTr / (torch.bmm(self.p.mT, Ap)) # B, 1, 1
            self.x += alpha * self.p # B, N, 1
            self.r -= alpha * Ap # B, N, 1
            self.log_residual(
                (residual := torch.linalg.matrix_norm(self.r)) # B
            )
            self.residual = residual # B
            if residual.max() < self.tol:
                break
            beta = torch.bmm(self.r.mT, self.r) / rTr # B, 1, 1
            self.p = self.r + beta * self.p # B, N, 1
        return {
            'x': self.x,
            'iter': self.iter,
            'residual': self.residual
        }


class GD_Batch(BatchSolver):
    def __init__(
        self,
        A: torch.Tensor | Callable,
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        tol: float = 1e-6,
        max_iter: int = 10000,
        device: str = 'cuda',
    ):
        super().__init__(A, b, x0, tol, max_iter, device)
        self.solver_name = 'GD'
    
    def solve(self):
        for self.iter in range(self.max_iter):
            Ar = self.A_func(self.r) # B, N, 1
            gamma = torch.bmm(self.r.mT, self.r) / torch.bmm(self.r.mT, Ar) # B, 1, 1
            self.x += gamma * self.r # B, N, 1
            self.log_residual(
                (residual := torch.linalg.matrix_norm(self.r)) # B
            )
            self.residual = residual # B
            if residual.max() < self.tol:
                break
            self.r -= gamma * Ar # B, N, 1
        return {
            'x': self.x,
            'iter': self.iter,
            'residual': self.residual
        }
        


if __name__ == "__main__":
    import time
    import scipy
    import numpy as np
    from optim import CG, GD
    import matplotlib.pyplot as plt

    # CUDA
    A = torch.randn(1, 20, 10).cuda().double()
    A = torch.bmm(A, A.mT) + 1 * torch.eye(20).cuda().double().unsqueeze(0)
    b = torch.randn(1, 20, 1).cuda().double()
    x0 = torch.zeros(1, 20, 1).cuda().double()
    
    cgb_cuda_lst = []
    for _ in range(100):
        t0 = time.time()
        cgb = CG_Batch(A, b, x0, tol=1e-6)
        res = cgb()
        cgb_cuda_lst.append((time.time() - t0) * 1000)
    cgb_cuda_mean = np.mean(cgb_cuda_lst)
    cgb_cuda_std = np.std(cgb_cuda_lst)
    print('=' * 30)
    print(f"CGB CUDA time: {cgb_cuda_mean:.3f}ms ± {cgb_cuda_std:.3f}ms")
    print(f"Iter: {res['iter']}")
    print(f"Residual: {res['residual'].max().item():.3e}")

    cg_cuda_lst = []
    for _ in range(100):
        t0 = time.time()
        cg = CG(A.squeeze(0), b.squeeze(0), x0.squeeze(0), tol=1e-6)
        res = cg()
        cg_cuda_lst.append((time.time() - t0) * 1000)
    cg_cuda_mean = np.mean(cg_cuda_lst)
    cg_cuda_std = np.std(cg_cuda_lst)
    print('=' * 30)
    print(f"CG CUDA time: {cg_cuda_mean:.3f}ms ± {cg_cuda_std:.3f}ms")
    print(f"Iter: {res['iter']}")
    print(f"Residual: {res['residual'].item():.3e}")

    assert torch.allclose(cgb.x, cg.x)

    gdb_cuda_lst = []
    for _ in range(100):
        t0 = time.time()
        gdb = GD_Batch(A, b, x0, tol=1e-6)
        res = gdb()
        gdb_cuda_lst.append((time.time() - t0) * 1000)
    gdb_cuda_mean = np.mean(gdb_cuda_lst)
    gdb_cuda_std = np.std(gdb_cuda_lst)
    print('=' * 30)
    print(f"GDB CUDA time: {gdb_cuda_mean:.3f}ms ± {gdb_cuda_std:.3f}ms")
    print(f"Iter: {res['iter']}")
    print(f"Residual: {res['residual'].max().item():.3e}")

    gd_cuda_lst = []
    for _ in range(100):
        t0 = time.time()
        gd = GD(A.squeeze(0), b.squeeze(0), x0.squeeze(0), tol=1e-6)
        res = gd()
        gd_cuda_lst.append((time.time() - t0) * 1000)
    gd_cuda_mean = np.mean(gd_cuda_lst)
    gd_cuda_std = np.std(gd_cuda_lst)
    print('=' * 30)
    print(f"GD CUDA time: {gd_cuda_mean:.3f}ms ± {gd_cuda_std:.3f}ms")
    print(f"Iter: {res['iter']}")
    print(f"Residual: {res['residual'].max().item():.3e}")

    assert torch.allclose(gdb.x, gd.x)

    assert torch.allclose(cgb.x, gdb.x)

    # CPU
    cgb_cpu_lst = []
    for _ in range(100):
        t0 = time.time()
        cgb = CG_Batch(A, b, x0, tol=1e-6, device='cpu')
        res = cgb()
        cgb_cpu_lst.append((time.time() - t0) * 1000)
    cgb_cpu_mean = np.mean(cgb_cpu_lst)
    cgb_cpu_std = np.std(cgb_cpu_lst)
    print('=' * 30)
    print(f"CGB CPU time: {cgb_cpu_mean:.3f}ms ± {cgb_cpu_std:.3f}ms")
    print(f"Iter: {res['iter']}")
    print(f"Residual: {res['residual'].max().item():.3e}")

    cg_cpu_lst = []
    for _ in range(100):
        t0 = time.time()
        cg = CG(A.squeeze(0), b.squeeze(0), x0.squeeze(0), tol=1e-6, device='cpu')
        res = cg()
        cg_cpu_lst.append((time.time() - t0) * 1000)
    cg_cpu_mean = np.mean(cg_cpu_lst)
    cg_cpu_std = np.std(cg_cpu_lst)
    print('=' * 30)
    print(f"CG CPU time: {cg_cpu_mean:.3f}ms ± {cg_cpu_std:.3f}ms")
    print(f"Iter: {res['iter']}")
    print(f"Residual: {res['residual'].item():.3e}")

    assert torch.allclose(cgb.x, cg.x)

    gdb_cpu_lst = []
    for _ in range(100):
        t0 = time.time()
        gdb = GD_Batch(A, b, x0, tol=1e-6, device='cpu')
        res = gdb()
        gdb_cpu_lst.append((time.time() - t0) * 1000)
    gdb_cpu_mean = np.mean(gdb_cpu_lst)
    gdb_cpu_std = np.std(gdb_cpu_lst)
    print('=' * 30)
    print(f"GDB CPU time: {gdb_cpu_mean:.3f}ms ± {gdb_cpu_std:.3f}ms")
    print(f"Iter: {res['iter']}")
    print(f"Residual: {res['residual'].max().item():.3e}")

    gd_cpu_lst = []
    for _ in range(100):
        t0 = time.time()
        gd = GD(A.squeeze(0), b.squeeze(0), x0.squeeze(0), tol=1e-6, device='cpu')
        res = gd()
        gd_cpu_lst.append((time.time() - t0) * 1000)
    gd_cpu_mean = np.mean(gd_cpu_lst)
    gd_cpu_std = np.std(gd_cpu_lst)
    print('=' * 30)
    print(f"GD CPU time: {gd_cpu_mean:.3f}ms ± {gd_cpu_std:.3f}ms")
    print(f"Iter: {res['iter']}")
    print(f"Residual: {res['residual'].max().item():.3e}")

    assert torch.allclose(gdb.x, gd.x)

    assert torch.allclose(cgb.x, gdb.x)

    A = A.squeeze(0).cpu().numpy()
    b = b.squeeze(0).cpu().numpy()
    x0 = x0.squeeze(0).cpu().numpy()

    scipy_lst = []
    for _ in range(100):
        t0 = time.time()
        x, _ = scipy.sparse.linalg.cg(A, b, x0=x0, atol=1e-6, rtol=0)
        scipy_lst.append((time.time() - t0) * 1000)
    scipy_mean = np.mean(scipy_lst)
    scipy_std = np.std(scipy_lst)
    print('=' * 30)
    print(f"SciPy CPU time: {scipy_mean:.3f}ms ± {scipy_std:.3f}ms")

    f, ax = plt.subplots(1, 1)
    ax.plot(cgb.residuals, label='CG_Batch')
    ax.plot(gdb.residuals, label='GD_Batch')
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    f.tight_layout()
    f.savefig('./resid_iter.png', dpi=600)
    
    f, ax = plt.subplots(1, 1)
    bp = ax.boxplot([cgb_cuda_lst, cg_cuda_lst, cgb_cpu_lst, cg_cpu_lst, gdb_cuda_lst, gd_cuda_lst, gdb_cpu_lst, gd_cpu_lst, scipy_lst], 0, '')
    ax.set_xticklabels(['CG_Batch CUDA', 'CG CUDA', 'CG_Batch CPU', 'CG CPU', 'GD_Batch CUDA', 'GD CUDA', 'GD_Batch CPU', 'GD CPU', 'SciPy'], rotation=45, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Solver")
    f.tight_layout()
    f.savefig('./time_solver.png', dpi=600)



