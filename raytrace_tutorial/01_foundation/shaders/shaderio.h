
/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef SHADERIO_H
#define SHADERIO_H

#include "common/io_gltf.h"
NAMESPACE_SHADERIO_BEGIN()
// Binding Points
enum BindingPoints
{
  eTextures = 0,  // Binding point for textures
  eOutImage,      // Binding point for output image
  eTlas,          // Top-level acceleration structure
  ePhotonBuffer,
  ePhotonCounter
};


struct TutoPushConstant
{
  float3x3       normalMatrix;
  int            instanceIndex;              // Instance index for the current draw call
  GltfSceneInfo* sceneInfoAddress;           // Address of the scene information buffer
  float2         metallicRoughnessOverride;  // Metallic and roughness override values
  int depthMax = 3; // maximum reflection depth


  int usePhotonMapping;   // toggle photon mapping
  int photonsPerLight;    //how many photons to trace
  int photonGatherCount;  // how many photons to gather (K-nearest)
};

struct Photon
{
  float3 position;     // 12 bytes - world position where photon hit
  uint   powerPacked;  // 4 bytes  - RGB9E5 packed photon power
  float3 direction;    // 12 bytes - incoming light direction
  float  _padding;     // 4 bytes  - alignment
};

NAMESPACE_SHADERIO_END()
#endif  // SHADERIO_H
