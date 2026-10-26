"""Experimental CUDA views of pinned host activations on ATS-capable GPUs.

This is selective saved-tensor storage, not a replacement CUDA allocator. Weights,
outputs and workspaces still use HBM. The owner must retain the pinned allocation
until all GPU reads complete; CUDA array interface ownership alone does not record
those reads with PyTorch's pinned-memory allocator.
"""
import ctypes
from functools import lru_cache

import torch


def storage_span(tensor):
    """Elements from a view's first to last address, including padding/holes."""
    return (1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
            if tensor.numel() else 0)


@lru_cache(maxsize=None)
def require_coherent_host_access(device_index):
    """Fail explicitly on PCIe/HMM devices rather than silently staging copies."""
    driver = ctypes.CDLL('libcuda.so.1')
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    if driver.cuInit(0):
        raise RuntimeError('Cannot initialize CUDA for direct host activation access')
    # CU_DEVICE_ATTRIBUTE_* from cuda.h. Host page tables distinguish ATS from
    # software-managed HMM on discrete GPUs in mixed-coherency machines.
    for name, attribute in [('PAGEABLE_MEMORY_ACCESS', 88),
                            ('CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM', 91),
                            ('PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES', 100)]:
        value = ctypes.c_int()
        error = driver.cuDeviceGetAttribute(ctypes.byref(value), attribute, device_index)
        if error or not value.value:
            raise RuntimeError(f'Direct activation storage requires coherent ATS host access ({name}); '
                               'select auto/cpu for staged copies or ranking-backward replay')


class _PinnedCUDAView:
    def __init__(self, host):
        self.host = host
        # Byte representation supports BF16 without depending on a NumPy BF16
        # dtype. Reinterpretation below restores dtype and shape without copying.
        self.__cuda_array_interface__ = dict(
            shape=(storage_span(host) * host.element_size(),), typestr='|u1',
            data=(host.data_ptr(), False), strides=None, version=3)


def pinned_cuda_view(host, device):
    """Borrow pinned storage, preserving strides; caller handles stream/lifetime."""
    device = torch.device(device)
    if device.type != 'cuda' or device.index is None:
        raise ValueError('Direct host views require an explicit CUDA device index')
    require_coherent_host_access(device.index)
    if host.device.type != 'cpu' or not host.is_pinned():
        raise ValueError('Direct host views require pinned CPU storage')
    view = torch.as_tensor(_PinnedCUDAView(host), device=device).view(host.dtype).as_strided(host.shape, host.stride())
    if view.data_ptr() != host.data_ptr():
        raise RuntimeError('CUDA host view unexpectedly copied its allocation')
    return view
