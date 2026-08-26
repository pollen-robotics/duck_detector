"""Make CUDA usable on a machine whose wheel disagrees with itself.

`torch 2.13+cu130` here ships cuDNN sublibraries that fail each other's version check — the first
convolution raises `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`, in this venv, from the wheel's own
bundled libraries. It is **not** the loader path: a scrubbed `LD_LIBRARY_PATH` fails identically.

Convolutions run fine on CUDA with cuDNN switched off, a little slower, so that is the fallback
rather than dropping to the CPU. The real fix is a torch build whose cuDNN is consistent; until
somebody picks one, this keeps every tool here working instead of making each of them discover the
same stack trace.
"""

from __future__ import annotations


# Said in one place, because both the pre-labeller and the trainer can hit it, and the useful part
# is the second sentence: extras are not additive. `uv sync --extra train` uninstalls the `label`
# extra, and a bare `uv sync` uninstalls both — which is how a working checkout loses transformers
# between two commands.
def missing(package: str) -> str:
    """What to say when a heavy dependency is not installed.

    Names the package rather than the extra, because the extras overlap — `train` brings torch in
    through ultralytics but not transformers, so "this needs torch" is wrong exactly when somebody
    has synced the other extra and is confused already.
    """
    return (
        f"this needs {package}, which is not installed:\n"
        "    uv sync --all-extras\n"
        "(a sync for one extra uninstalls the others; --all-extras is the one that keeps working)"
    )


def prepare_cuda(verbose: bool = True) -> str:
    """The device to use, having proved it can convolve. `"cuda"` or `"cpu"`."""
    try:
        import torch
    except ImportError as error:
        raise SystemExit(missing("torch")) from error

    if not torch.cuda.is_available():
        return "cpu"

    def probe() -> None:
        torch.nn.functional.conv2d(
            torch.randn(1, 3, 32, 32, device="cuda"), torch.randn(4, 3, 3, 3, device="cuda")
        )

    try:
        probe()
        return "cuda"
    except RuntimeError as error:
        if "CUDNN" not in str(error).upper():
            raise
        torch.backends.cudnn.enabled = False
        try:
            probe()
        except RuntimeError:
            if verbose:
                print("cuda cannot convolve at all; falling back to the cpu")
            return "cpu"
        if verbose:
            print("cuda with cudnn disabled (this wheel's cudnn disagrees with itself)")
        return "cuda"
