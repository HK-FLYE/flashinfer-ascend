// Included only by the per-package typed call site generated at build time.
#pragma once
#include "call.h"
#include <deque>
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#define FI_ACLNN_WORKSPACE_FN(api) invocation.workspace_fn
#define FI_ACLNN_LAUNCH_FN(api) invocation.launch_fn
#define FI_ACLNN_OBSERVE_WORKSPACE(n) invocation.workspace_bytes.store(static_cast<int64_t>(n))
#define FI_ACLNN_RETAIN_ARGUMENTS() operands.keepalive
#include "op_api_common.h"

namespace flashinfer_aclnn {
struct Owners {
    std::vector<at::Tensor> tensors;
    c10::List<c10::IValue> attributes;
    std::deque<std::vector<at::Tensor>> wrapped;
    explicit Owners(const Call& c) : attributes(c.attributes) {
        tensors.insert(tensors.end(), c.inputs.begin(), c.inputs.end());
        tensors.insert(tensors.end(), c.outputs.begin(), c.outputs.end());
    }
};

class Operands {
    at::TensorList inputs_, outputs_;
    at::IntArrayRef widths_;
    bool lists_;
    const c10_npu::NPUStream stream_ = c10_npu::getCurrentNPUStream();
public:
    const std::shared_ptr<Owners> keepalive;
    explicit Operands(const Call& c) : inputs_(c.inputs), outputs_(c.outputs), widths_(c.widths),
            lists_(!c.widths.empty()), keepalive(std::make_shared<Owners>(c)) {
        for (size_t i = 0; i < c.outputs.size(); ++i) {
            const auto& out = c.outputs[i];
            auto overlaps = [&](const at::Tensor& t) {
                if (t.device() != out.device()) return false;
                auto range = [](const at::Tensor& v) {
                    bool nz = at_npu::native::get_npu_format(v) == ACL_FORMAT_FRACTAL_NZ;
                    auto lo = reinterpret_cast<uintptr_t>(nz ? v.storage().data() : v.data_ptr());
                    return std::make_pair(lo, lo + (nz ? v.storage().nbytes() : v.nbytes()));
                };
                auto a = range(out), b = range(t);
                return std::max(a.first,b.first) < std::min(a.second,b.second);
            };
            for (const auto& t : c.inputs) TORCH_CHECK(!overlaps(t), "ACLNN output tensors must not overlap other arguments");
            for (size_t j = 0; j < i; ++j) TORCH_CHECK(!overlaps(c.outputs[j]), "ACLNN outputs must not overlap");
        }
    }
    // -1 is an absent optional argument. Zero is a present empty TensorList.
    c10::optional<at::TensorList> take(bool output, bool list, bool optional, int layouts, bool cpu) {
        auto& source = output ? outputs_ : inputs_;
        int64_t width = 1;
        if (lists_) {
            TORCH_CHECK(!widths_.empty(), "Missing ACLNN operand width");
            width = widths_.front();
            widths_ = widths_.slice(1);
        }
        if (width == -1 && optional) return c10::nullopt;
        TORCH_CHECK(width >= 0 && static_cast<size_t>(width) <= source.size() && (list || width == 1),
                    "Invalid ACLNN Tensor/List argument count");
        auto tensors = source.slice(0, width);
        source = source.slice(width);
        for (const auto& tensor : tensors) {
            if (cpu) {
                TORCH_CHECK(tensor.device().is_cpu(), "Expected a declared CPU tensor");
                continue;
            }
            TORCH_CHECK(tensor.device() == stream_.device() && tensor.is_contiguous(),
                        "ACLNN tensor violates its declared device/format");
            const auto format = at_npu::native::get_npu_format(tensor);
            const int mask = format == ACL_FORMAT_ND ? 1 : format == ACL_FORMAT_FRACTAL_NZ ? 2 : 0;
            TORCH_CHECK(layouts & mask, "ACLNN tensor violates its declared device/format");
            c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream_);
        }
        // Wrapped CPU numbers use device storage. Retain the copies through queued submission.
        if (cpu && std::any_of(tensors.begin(), tensors.end(), [](const auto& t) {
                return t.unsafeGetTensorImpl()->is_wrapped_number();
            })) {
            auto& copied = keepalive->wrapped.emplace_back(tensors.begin(), tensors.end());
            for (auto& t : copied) {
                if (t.unsafeGetTensorImpl()->is_wrapped_number()) {
                    t = CopyTensorHostToDevice(t);
                    c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), stream_);
                }
            }
            return at::TensorList(copied);
        }
        return tensors;
    }
    void finish() const {
        TORCH_CHECK(inputs_.empty() && outputs_.empty() && widths_.empty(), "Extra ACLNN operands");
    }
};

// The two pre-existing NativeBinding packed representations are views of storage.
// They pass through the same upstream conversion/release and submission template.
struct StorageView { at::Tensor tensor; bool packed_int4; };
inline aclTensor* ConvertType(const StorageView& v) {
    const auto& t = v.tensor;
    if (!t.defined()) return nullptr;
    const bool int4 = v.packed_int4 && t.scalar_type() == at::kInt;
    const bool uint64 = !v.packed_int4 && t.scalar_type() == at::kLong;
    if (!int4 && !uint64) return ::ConvertType(t);
    auto sizes = t.sizes().vec(), strides = t.strides().vec();
    auto offset = t.storage_offset();
    auto format = static_cast<aclFormat>(at_npu::native::get_npu_format(t));
    auto storage = format == ACL_FORMAT_FRACTAL_NZ ? at_npu::native::get_npu_storage_sizes(t)
        : std::vector<int64_t>{static_cast<int64_t>(t.storage().nbytes() / t.element_size())};
    if (int4) {
        TORCH_CHECK(!sizes.empty() && !storage.empty(), "Packed int4 requires dimensional metadata");
        sizes.back() *= 8;
        for (size_t i = 1; i < strides.size(); ++i) strides[i-1] *= 8;
        storage.back() *= 8;
        offset *= 8;
        if (format == ACL_FORMAT_FRACTAL_NZ) {
            TORCH_CHECK(sizes.size() >= 2 && sizes.back() % 64 == 0 && sizes[sizes.size()-2] % 16 == 0,
                        "Native INT4 NZ requires N aligned to 64 and K aligned to 16");
            storage.assign(sizes.begin(), sizes.end()-2);
            storage.insert(storage.end(), {sizes.back()/64, sizes[sizes.size()-2]/16, 16, 64});
        }
    }
    static const auto create = GET_OP_API_FUNC(aclCreateTensor);
    TORCH_CHECK(create, "aclCreateTensor unavailable");
    auto* descriptor = create(sizes.data(), sizes.size(), int4 ? ACL_INT4 : ACL_UINT64,
                              strides.data(), offset, format, storage.data(), storage.size(),
                              const_cast<void*>(t.storage().data()));
    TORCH_CHECK(descriptor, "aclCreateTensor failed for packed storage");
    return descriptor;
}

template <typename T>
std::vector<T> array_values(const c10::IValue& value) {
    // A typed recipe determines the type of an empty sequence as well.
    const auto& elements = value.toTupleRef().elements();
    std::vector<T> data;
    data.reserve(elements.size());
    for (const auto& element : elements) data.push_back(element.to<T>());
    return data;
}
class BoolValues {
    std::unique_ptr<bool[]> ptr_;
    size_t size_;
public:
    explicit BoolValues(const c10::IValue& value) : size_(value.toTupleRef().elements().size()) {
        ptr_ = std::make_unique<bool[]>(size_);
        for (size_t i = 0; i < size_; ++i) ptr_[i] = value.toTupleRef().elements()[i].toBool();
    }
    at::ArrayRef<bool> view() const { return at::ArrayRef<bool>(ptr_.get(), size_); }
};
} // namespace flashinfer_aclnn
