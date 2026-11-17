"""Native admission and runner stub for quant_lightning_indexer.

P0 only validates the public→native shape contract. Launch is intentionally
unimplemented until the AOT package exists.
"""

from dataclasses import dataclass, replace

from flashinfer.jit.tensor_adapter import package_tensor_runner


def admission(recipe, signature):
    from flashinfer.ascendc._runtime import native_admission

    names = recipe.tensor_parameter_names
    positions = {name: index for index, name in enumerate(names)}
    try:
        specializations = dict(signature.specializations)
        shapes = list(signature.shapes)
        query = shapes[positions["query"]]
        key = shapes[positions["key"]]
        if len(query) != 3:
            return ("query must be flat TND [TQ,N1,D]",)
        if len(key) == 4 and key[2] == 1:
            shapes[positions["key"]] = (key[0], key[1], key[3])
            key = shapes[positions["key"]]
        if len(key) != 3:
            return ("paged key must reduce to [PAGES,BLOCK,D]",)
        k_scale = shapes[positions["key_dequant_scale"]]
        if len(k_scale) == 3 and k_scale[2] == 1:
            shapes[positions["key_dequant_scale"]] = (k_scale[0], k_scale[1])
        tq = shapes[positions["query"]][0]
        topk = shapes[positions["indices"]][-1]
        nk = shapes[positions["indices"]][1] if len(shapes[positions["indices"]]) == 3 else 1
        native_shapes = list(shapes)
        # Public may expose empty values when return_value=0; native ABI keeps a
        # full [TQ,NK,K] buffer like general_indexer.
        native_shapes[positions["indices"]] = (tq, nk, topk)
        native_shapes[positions["values"]] = (tq, nk, topk)
        signature = replace(signature, shapes=tuple(native_shapes))
    except (IndexError, KeyError, TypeError, ValueError) as error:
        return ("quant_lightning_indexer requires complete native Tensor shapes: %s" % error,)

    reasons = list(native_admission(recipe, signature))
    max_seq = int(specializations.get("attribute.max_seq_len_k", "0"))
    if max_seq <= 0:
        reasons.append("paged execution requires max_seq_len_k")
    return tuple(reasons)


@dataclass(frozen=True)
class QuantLightningIndexerRunner:
    native: object

    def __call__(self, *tensors, **attributes):
        raise NotImplementedError(
            "quant_lightning_indexer AOT package is not built yet (P0 stub)"
        )


def make_runner(module, spec, recipe):
    return QuantLightningIndexerRunner(package_tensor_runner(module, recipe, spec=spec))
