"""Declarative Ascend C package for sparse_indexer.quant_lightning_indexer."""

from flashinfer.ascendc.definition import (
    AscendCImplementation,
    BuildConfig,
    NativeBinding,
)
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import (
    CATLASS_REVISION,
    BuildDependency,
    ShapeConstraint,
    TensorParameter,
)
from flashinfer.operators.author_api import Support
from flashinfer.operators.sparse_indexer.quant_lightning_indexer.operator import (
    HEAD_DIM,
    MAX_BATCH,
    MAX_HEADS,
    MAX_TOPK,
    MAX_TOTAL,
    OPERATOR,
)
from flashinfer.runtime import CapabilityCase

from .runner import admission, make_runner

KERNEL_ID = "ascendc/sparse_indexer/quant_lightning_indexer/catlass/op_package/Ascend950"


def _tensor(name, role, dtype, shape, *, alternative=(), optional=False, allow_strided=False):
    return TensorParameter(
        name=name,
        role=role,
        dtype=dtype,
        shape=shape,
        shape_alternatives=alternative,
        optional=optional,
        allow_strided=allow_strided,
    )


_TENSORS = (
    _tensor("query", "input", "float8_e4m3fn", ("TQ", "N1", "D")),
    _tensor("key", "input", "float8_e4m3fn", ("PAGES", "BLOCK", "D"),
            alternative=(("PAGES", "BLOCK", "NK", "D"),), allow_strided=True),
    _tensor("weights", "input", "float32", ("TQ", "N1")),
    _tensor("query_dequant_scale", "input", "float32", ("TQ", "N1")),
    _tensor("key_dequant_scale", "input", "float32", ("PAGES", "BLOCK"),
            alternative=(("PAGES", "BLOCK", "NK"),), allow_strided=True),
    _tensor("block_table", "input", "int32", ("B", "MB"), optional=True),
    _tensor("cu_seqlens_q", "input", "int32", ("BC",), optional=True),
    _tensor("cu_seqlens_k", "input", "int32", ("BC",), optional=True),
    _tensor("seqused_q", "input", "int32", ("B",), optional=True),
    _tensor("seqused_k", "input", "int32", ("B",), optional=True),
    _tensor("cmp_residual_k", "input", "int32", ("B",), optional=True),
    _tensor("indices", "output", "int32", ("TQ", "NK", "K")),
    _tensor("values", "output", "bfloat16", ("TQ", "NK", "K")),
)

_DTYPE_SIGNATURES = (
    (
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float32",
        "float32",
        "float32",
        "int32",
        "int32",
        "int32",
        "int32",
        "int32",
        "int32",
        "int32",
        "bfloat16",
    ),
)

IMPLEMENTATION = AscendCImplementation(
    operator=OPERATOR,
    kernel_id=KERNEL_ID,
    support=Support(
        soc_models=("Ascend950",),
        cases=(
            CapabilityCase(
                name="ascend950_quant_lightning_indexer",
                compute_dtypes=("float8_e4m3fn",),
                accumulator_dtypes=("float32",),
                attention_variants=("quant_lightning_indexer",),
                kv_cache_types=("paged",),
                causal_values=(True,),
            ),
        ),
    ),
    build=BuildConfig(
        device="op_kernel/quant_lightning_indexer.cpp",
        host=(
            "op_host/quant_lightning_indexer_def.cpp",
            "op_host/quant_lightning_indexer_infershape.cpp",
            "op_host/arch35/quant_lightning_indexer_tiling.cpp",
        ),
        op_type="FiQuantLightningIndexer",
        tiling="FiQuantLightningIndexerTiling",
        includes=("op_kernel", "op_host", "include"),
        shared_native_roots=("csrc/kernels/sparse_indexer/common",),
        shared_includes=(
            "csrc/kernels/sparse_indexer/common/include",
            "csrc/kernels/sparse_indexer/common/op_kernel",
        ),
        implementation="catlass",
        dependencies=(BuildDependency(name="catlass", revision=CATLASS_REVISION),),
        definitions=("CATLASS_ARCH=3510",),
        compile_options=("-O2",),
        link_options=("-lascendcl",),
        workspace_bytes=0,
        device_entry="fi_quant_lightning_indexer",
        host_entry="aclnnFiQuantLightningIndexer",
        launch_abi="aclnn.quant_lightning_indexer.v1",
        package_vendor="flashinfer_quant_lightning_indexer",
        recipe_id="quant_lightning_indexer",
    ),
    source_dir="csrc/kernels/sparse_indexer/quant_lightning_indexer",
    binding=NativeBinding(
        tensors=_TENSORS,
        dimensions=(
            ShapeConstraint("TQ", 1, MAX_TOTAL),
            ShapeConstraint("N1", 1, MAX_HEADS),
            ShapeConstraint("D", HEAD_DIM, HEAD_DIM, HEAD_DIM),
            ShapeConstraint("PAGES", 1, 1 << 20),
            ShapeConstraint("BLOCK", 1, 4096),
            ShapeConstraint("NK", 1, 1),
            ShapeConstraint("B", 1, MAX_BATCH),
            ShapeConstraint("MB", 1, 1 << 16),
            ShapeConstraint("BC", 2, MAX_BATCH + 1),
            ShapeConstraint("K", 1, MAX_TOPK),
        ),
        dtype_signatures=_DTYPE_SIGNATURES,
        attributes=(
            AttributeParameter("topk", "int64", 1, 1, MAX_TOPK),
            AttributeParameter("quant_mode", "int64", 1, 1, 1),
            AttributeParameter("mask_mode", "int64", 3, 3, 3),
            AttributeParameter("cmp_ratio", "int64", 4, 4, 4),
            AttributeParameter("max_seq_len_k", "int64", 1, 1, MAX_TOTAL),
            AttributeParameter("max_seqlen_q", "int64", -1, -1, MAX_TOTAL),
        ),
    ),
    runner_factory=make_runner,
    admission=admission,
    workspace_bound=(1 << 30) + (128 << 20),
    allow_jit_fallback=False,
)

NATIVE = IMPLEMENTATION.native
descriptor = IMPLEMENTATION.bind
