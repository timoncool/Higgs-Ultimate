#include "engine/framework/modules/optimizations/fast_kv_modules.h"

#include "../tensor_layout_utils.h"

#include <stdexcept>

namespace engine::modules {

namespace {

const core::ModulePortSpec kSetRowsInputs[] = {
    {"cache", core::PortKind::Activation, false},
    {"row", core::PortKind::Activation, false},
    {"row_index", core::PortKind::Activation, false},
};

const core::ModulePortSpec kSingleOutput[] = {
    {"output", core::PortKind::Activation, false},
};

const core::ModuleSchema kFastKVSetRowsSchema = {
    "FastKVSetRows",
    "tensor.optimization",
    kSetRowsInputs,
    3,
    kSingleOutput,
    1,
    "Appends one flattened KV row into a cache tensor using ggml_set_rows.",
};

}  // namespace

const core::ModuleSchema & FastKVSetRowsModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue FastKVSetRowsModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    const core::TensorValue & row,
    const core::TensorValue & row_index) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(cache, 4, 4, "cache");
    core::validate_shape(
        row,
        core::TensorShape::from_dims({cache.shape.dims[0], 1, cache.shape.dims[2], cache.shape.dims[3]}),
        "row");
    core::validate_shape(row_index, core::TensorShape::from_dims({1}), "row_index");
    if (cache.type != GGML_TYPE_F32 && cache.type != GGML_TYPE_F16) {
        throw std::runtime_error("FastKVSetRowsModule requires an f32 or f16 cache tensor");
    }
    if (row.type != GGML_TYPE_F32 && row.type != GGML_TYPE_F16) {
        throw std::runtime_error("FastKVSetRowsModule requires an f32 or f16 row tensor");
    }
    if (row_index.type != GGML_TYPE_I32 && row_index.type != GGML_TYPE_I64) {
        throw std::runtime_error("FastKVSetRowsModule requires i32 or i64 row_index tensor");
    }
    if (!core::has_backend_addressable_layout(cache.tensor)) {
        throw std::runtime_error("FastKVSetRowsModule requires a contiguous cache tensor");
    }

    const int64_t steps = cache.shape.dims[1];
    const int64_t row_elems = cache.shape.dims[2] * cache.shape.dims[3];
    auto flat_cache = core::reshape_tensor(ctx, cache, core::TensorShape::from_dims({steps, row_elems}));
    auto contiguous_row = tensor_layout::ensure_contiguous_layout_if_needed(ctx, row);
    if (contiguous_row.type != GGML_TYPE_F32) {
        contiguous_row = core::wrap_tensor(
            ggml_cast(ctx.ggml, contiguous_row.tensor, GGML_TYPE_F32),
            contiguous_row.shape,
            GGML_TYPE_F32);
        contiguous_row = tensor_layout::ensure_contiguous_layout_if_needed(ctx, contiguous_row);
    }
    auto flat_row = core::reshape_tensor(ctx, contiguous_row, core::TensorShape::from_dims({1, row_elems}));
    ggml_tensor * updated = ggml_set_rows(ctx.ggml, flat_cache.tensor, flat_row.tensor, row_index.tensor);
    auto flat_updated = core::wrap_tensor(updated, flat_cache.shape, cache.type);
    return core::reshape_tensor(ctx, flat_updated, cache.shape);
}

const core::ModuleSchema & FastKVSetRowsModule::static_schema() noexcept {
    return kFastKVSetRowsSchema;
}

}  // namespace engine::modules
