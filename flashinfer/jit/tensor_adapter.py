"""Recipe-driven eager tensor adapter for the two-stage ACLNN package ABI.

This module owns tensor metadata/lifetimes, not package discovery or loading.
Tiling is produced by the package's host callback and retained by the ACLNN
executor; its device buffer is an implicit final kernel argument.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from threading import RLock

from flashinfer.runtime import SchemaError


def _reject_output_overlap(parameters, tensors):
    for output_index, (parameter, output) in enumerate(zip(parameters, tensors)):
        if parameter.role != "output":
            continue
        out_lo = output.data_ptr()
        out_hi = out_lo + output.numel() * output.element_size()
        for tensor_index, tensor in enumerate(tensors):
            if tensor_index == output_index or tensor is None:
                continue
            lo = tensor.data_ptr()
            hi = lo + tensor.numel() * tensor.element_size()
            if max(lo, out_lo) < min(hi, out_hi):
                raise ValueError("output tensors must not overlap other arguments")


def package_tensor_runner(module, recipe, *, spec):
    """Validate a previously loaded package and return its generic adapter."""
    from .loading import JitLoadedModule
    from .op_package import OpPackageModule

    if (spec.kernel_id != recipe.kernel_id or spec.source != recipe.source
            or spec.target_soc != recipe.target_soc
            or spec.entry_points != (recipe.host_entry,)
            or dict(spec.input_fingerprints).get("build_recipe") != recipe.fingerprint):
        raise SchemaError("tensor adapter spec does not bind the recipe/entry")
    if isinstance(module, JitLoadedModule):
        if (module.receipt.spec_fingerprint != spec.fingerprint
                or tuple(symbol.name for symbol in module.receipt.symbols)
                != (recipe.host_entry,)):
            raise SchemaError("tensor adapter loaded receipt does not bind its spec")
        module = module.opaque_module
    if not isinstance(module, OpPackageModule) or module.recipe != recipe:
        raise SchemaError("tensor adapter requires the verified recipe's OpPackageModule")
    return TorchOpPackageRunner(module, recipe)


@dataclass(frozen=True)
class TorchOpPackageRunner:
    module: object
    recipe: object
    _native_runner: object = field(default=None, init=False, repr=False, compare=False)
    _native_lock: object = field(default_factory=RLock, init=False, repr=False, compare=False)
    _validate_metadata: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        recipe = self.recipe
        names = recipe.tensor_parameter_names

        @lru_cache(maxsize=128)
        def validate(metadata):
            recipe.validate_tensor_metadata(
                shapes={n: None if v is None else v[0] for n, v in zip(names, metadata)},
                dtypes={n: None if v is None else str(v[1]).removeprefix("torch.")
                        for n, v in zip(names, metadata)},
            )

        object.__setattr__(self, "_validate_metadata", validate)

    def _validate_storage(self, tensors):
        import torch
        import torch_npu

        present = tuple(t for t in tensors if t is not None)
        device = present[0].device
        if device.type != torch._C._get_privateuse1_backend_name():
            raise ValueError("Ascend C tensor adapter requires NPU tensors")
        if any(t.device != device for t in present):
            raise ValueError("Ascend C tensors must share one device")
        for parameter, tensor in zip(self.recipe.tensor_parameters, tensors):
            if tensor is None:
                continue
            layout = {2: "ND", 29: "NZ"}.get(torch_npu.get_npu_format(tensor))
            if not tensor.is_contiguous() or layout not in parameter.physical_layouts:
                raise ValueError("Ascend C tensor adapter requires contiguous %s tensors for %s"
                                 % ("/".join(parameter.physical_layouts), parameter.name))
        _reject_output_overlap(self.recipe.tensor_parameters, tensors)

    def __call__(self, *tensors, **attributes):
        names = self.recipe.tensor_parameter_names
        if len(tensors) != len(names):
            raise SchemaError("tensor adapter expected parameters %s" % (names,))
        self._validate_metadata(tuple(None if t is None else (tuple(t.shape), t.dtype) for t in tensors))
        scalar_values = self.recipe.validate_attributes(attributes)

        if self._native_runner is None:
            self._validate_storage(tensors)
            from flashinfer.aclnn import _native

            with self._native_lock:
                if self._native_runner is None:
                    object.__setattr__(self, "_native_runner", _native.prepare(self.module, self.recipe))
        # Both native entry points check the current device, format, contiguity
        # and storage ranges before calling ACLNN, including on cache hits.
        try:
            if self.recipe.extended_aclnn:
                return self._native_runner(tensors, scalar_values)
            return self._native_runner(tensors)
        except RuntimeError:
            # Preserve the eager adapter's diagnostic for invalid storage.
            self._validate_storage(tensors)
            raise


__all__ = ["TorchOpPackageRunner", "package_tensor_runner"]
