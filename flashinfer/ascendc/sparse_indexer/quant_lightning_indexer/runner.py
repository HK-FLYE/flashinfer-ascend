"""Native admission and runner for quant_lightning_indexer first grid."""

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


def _squeeze_nk(tensor):
    if tensor is None:
        return None
    if tensor.dim() == 4 and tensor.size(2) == 1:
        return tensor.squeeze(2)
    if tensor.dim() == 3 and tensor.size(-1) == 1:
        return tensor.squeeze(-1)
    return tensor


def _nd_page_view(tensor, torch_npu):
    """Flatten singleton Nk while preserving the page-axis storage stride."""
    import torch

    flat = _squeeze_nk(tensor)
    if flat.device.type == torch._C._get_privateuse1_backend_name():
        # Format tagging is metadata-only for the strided ND view; it must not
        # materialize a contiguous copy or the device-side stride is lost.
        flat = torch_npu.npu_format_cast(flat, 2)
    return flat


@dataclass(frozen=True)
class QuantLightningIndexerRunner:
    native: object

    def __call__(
        self,
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        block_table,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        indices,
        values,
        **attributes,
    ):
        import torch

        import torch_npu

        def _nd_contig(tensor):
            out = tensor.contiguous()
            # Strided PA views can leave a non-ND acl format after contiguous().
            if out.device.type == torch._C._get_privateuse1_backend_name():
                out = torch_npu.npu_format_cast(out, 2)
            return out

        # PA key/scales keep their page-axis stride; native tiling forwards it.
        key_flat = _nd_page_view(key, torch_npu)
        k_scale = _nd_page_view(key_dequant_scale, torch_npu)
        query_flat = _nd_contig(query)
        weights_flat = _nd_contig(weights)
        q_scale = _nd_contig(query_dequant_scale)

        # Native values buffer is always full-shaped even when public return_value=0.
        if values.numel() == 0:
            values_work = torch.empty(
                indices.shape, dtype=torch.bfloat16, device=query.device
            )
        else:
            values_work = values

        if query.device.type == torch._C._get_privateuse1_backend_name():
            torch.npu.current_stream().synchronize()

        self.native(
            query_flat,
            key_flat,
            weights_flat,
            q_scale,
            k_scale,
            block_table,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            indices,
            values_work,
            topk=int(indices.shape[-1]),
            quant_mode=int(attributes["quant_mode"]),
            mask_mode=int(attributes["mask_mode"]),
            cmp_ratio=int(attributes["cmp_ratio"]),
            max_seq_len_k=int(attributes["max_seq_len_k"]),
            max_seqlen_q=int(attributes.get("max_seqlen_q", -1)),
            return_value=int(attributes.get("return_value", 0)),
        )

        if query.device.type == torch._C._get_privateuse1_backend_name():
            torch.npu.current_stream().synchronize()


def make_runner(module, spec, recipe):
    return QuantLightningIndexerRunner(package_tensor_runner(module, recipe, spec=spec))
