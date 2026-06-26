// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>
#include "common/types.h"

namespace Shader {
struct FragmentRuntimeInfo;
}

namespace Shader::Backend::SPIRV {

enum class AuxShaderType : u32 {
    RectListTCS,
    QuadListTCS,
    PassthroughTES,
};

// `vs_output_mask` is a bitmask of attribute Locations the upstream vertex shader actually
// declares as Output. The aux TCS must only read locations the VS provides, otherwise it
// declares Inputs with no matching VS Output (VUID-RuntimeSpirv-OpEntryPoint-08743) and reads
// undefined attributes. Locations not provided by the VS are zero-filled so the downstream
// TES/fragment interface stays complete.
[[nodiscard]] std::vector<u32> EmitAuxilaryTessShader(AuxShaderType type,
                                                      const FragmentRuntimeInfo& fs_info,
                                                      u32 vs_output_mask);

} // namespace Shader::Backend::SPIRV
