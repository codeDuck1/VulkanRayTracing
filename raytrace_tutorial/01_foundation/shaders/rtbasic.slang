#include "shaderio.h"
#include "common/shaders/pbr.h.slang"
#include "nvshaders/constants.h.slang"
#include "nvshaders/sky_functions.h.slang"

#define MISS_DEPTH 1000
#define PHOTON_RAY_MARKER 999999

#ifndef PI
#define PI 3.14159265358979323846
#endif

//#define REFLECTION_RECURSIVE  // Hardware recursion approach
#define REFLECTION_ITERATIVE    // Loop-based approach

// clang-format off
// Push constants containing scene information and camera data
[[vk::push_constant]]                           ConstantBuffer<TutoPushConstant> pushConst;
// Texture array for material textures
[[vk::binding(BindingPoints::eTextures, 0)]]    Sampler2D textures[];
// Top-level acceleration structure containing the scene geometry
[[vk::binding(BindingPoints::eTlas, 1)]]        RaytracingAccelerationStructure topLevelAS;
// Output image where the final rendered result will be stored
[[vk::binding(BindingPoints::eOutImage, 1)]]    RWTexture2D<float4> outImage;

[[vk::binding(BindingPoints::ePhotonBuffer, 1)]]  RWStructuredBuffer<Photon> photonBuffer;
[[vk::binding(BindingPoints::ePhotonCounter, 1)]] RWStructuredBuffer<uint> photonCounter;

// clang-format on

// Ray payload structure - carries data through the ray tracing pipeline
struct HitPayload
{
    float3 color;   // Accumulated color along the ray path
    float  weight;  // Reflection weight (decreases with each bounce)
    int    depth;   // Current reflection depth (for limiting bounces). BOUNCE COUNT!

    // for iterative reflection
    float3 rayOrigin;
    float3 rayDirection;
};

// Minimal payload for shadow rays - only what we need for shadow testing
struct ShadowPayload
{
  int depth;  // Only need to know if we hit something or not
};


__generic<T : IFloat> T getAttribute(uint8_t* dataBufferAddress, BufferView bufferView, uint attributeIndex)
{
  if(bufferView.count > 0)
  {
    T* ptr = (T*)(dataBufferAddress + bufferView.offset + attributeIndex * bufferView.byteStride);
    return ptr[0];
  }

  return T(1);  // Error case
}

int3 getTriangleIndices(uint8_t* dataBufferAddress, const TriangleMesh mesh, int primitiveID)
{
  if(mesh.indices.byteStride == sizeof(int16_t))
  {
    int16_t3* indices = (int16_t3*)(dataBufferAddress + mesh.indices.offset);
    return indices[primitiveID];
  }
  else if(mesh.indices.byteStride == sizeof(int32_t))
  {
    int3* indices = (int3*)(dataBufferAddress + mesh.indices.offset);
    return indices[primitiveID];
  }

  return int3(-1);  // Error case
}

__generic<T : IFloat> T getTriangleAttribute(uint8_t* dataBufferAddress, BufferView bufferView, uint3 attributeIndex, float3 barycentrics)
{
  T attr0 = getAttribute<T>(dataBufferAddress, bufferView, attributeIndex.x);
  T attr1 = getAttribute<T>(dataBufferAddress, bufferView, attributeIndex.y);
  T attr2 = getAttribute<T>(dataBufferAddress, bufferView, attributeIndex.z);
  return T(barycentrics.x) * attr0 + T(barycentrics.y) * attr1 + T(barycentrics.z) * attr2;
}

//-----------------------------------------------------------------------
// SHADOW TESTING
//-----------------------------------------------------------------------
float testShadow(float3 worldPos, float3 worldNormal, float3 lightDirection, GltfPunctual light)
{
  RayDesc shadowRay;
  shadowRay.Origin    = worldPos + worldNormal * 0.001;  // Offset to avoid self-intersection
  shadowRay.Direction = normalize(lightDirection); // ray was shot from camera, now go in light direction and see if hit anything.
  shadowRay.TMin      = 0.001;
  // end point to test for intersections. for direct lights, shadow ray length limited to dist to the light source
  shadowRay.TMax      = light.type == GltfLightType::eDirectional ? INFINITE : length(lightDirection); 

  // Create a simple payload for shadow ray
  ShadowPayload shadowPayload;
  shadowPayload.depth = 0;

  // Trace the shadow ray using the dedicated shadow miss shader (group 1)
  TraceRay(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xff, 0, 0, 1,
           shadowRay, shadowPayload);

  // If the shadow ray hit something, the light is occluded
  // we know hit something is depth of ray is not miss depth
  // which is set for rays that miss
  return shadowPayload.depth != MISS_DEPTH ? 0.0 : 1.0;
}

//-----------------------------------------------------------------------
// Handles sky override, point light distance attenuation, and spot light cone attenuation
//-----------------------------------------------------------------------
GltfPunctual processLight(GltfSceneInfo sceneInfo, float3 worldPos)
{
  GltfPunctual light = sceneInfo.punctualLights[0];  // Using the first light for simplicity

  // Sky override: replace punctual light with sun parameters
  if(sceneInfo.useSky == 1)
  {
    light.direction = sceneInfo.skySimpleParam.sunDirection;
    light.color     = sceneInfo.skySimpleParam.sunColor;
    light.intensity = sceneInfo.skySimpleParam.sunIntensity;
    light.type      = GltfLightType::eDirectional;
  }

  // Point light: calculate direction and apply distance attenuation
  if(light.type == GltfLightType::ePoint)
  {
    light.direction = light.position - worldPos;
    float d         = length(light.direction);
    light.intensity /= (d * d);  // Inverse square law
  }
  // Spot light: calculate direction, apply distance and cone attenuation
  else if(light.type == GltfLightType::eSpot)
  {
    float3 lightDir = light.position - worldPos;
    float  d        = length(lightDir);
    light.intensity /= (d * d);  // Distance attenuation
    
    // Cone attenuation
    float theta         = dot(normalize(lightDir), normalize(light.direction));
    float spotIntensity = clamp((theta - cos(light.coneAngle)) / (1.0 - cos(light.coneAngle)), 0.0, 1.0);
    light.intensity *= spotIntensity;
    light.direction = lightDir;
  }

  return light;
}

//-----------------------------------------------------------------------
// PHOTON MAPPING HELPERS
//-----------------------------------------------------------------------

// Simple RNG for photon tracing
struct RandomState
{
  uint state;
};

void initRandom(inout RandomState rng, uint seed)
{
  rng.state = seed;
}

float randomFloat(inout RandomState rng)
{
  rng.state = rng.state * 747796405u + 2891336453u;
  uint result = ((rng.state >> ((rng.state >> 28u) + 4u)) ^ rng.state) * 277803737u;
  result = (result >> 22u) ^ result;
  return float(result) / 4294967295.0;
}

// Pack RGB into RGB9E5 format (shared exponent)
uint packRGB9E5(float3 rgb)
{
  // Clamp to positive values
  rgb = max(rgb, float3(0, 0, 0));
  
  // Find max component
  float maxComponent = max(max(rgb.r, rgb.g), rgb.b);
  
  if(maxComponent < 1e-6)
    return 0;
  
  // Calculate shared exponent
  int exponent = int(floor(log2(maxComponent))) + 1;
  exponent = clamp(exponent + 15, 0, 31);
  
  float scale = pow(2.0, float(exponent - 24));
  
  // Quantize mantissas
  uint r = uint(clamp(rgb.r / scale + 0.5, 0.0, 511.0));
  uint g = uint(clamp(rgb.g / scale + 0.5, 0.0, 511.0));
  uint b = uint(clamp(rgb.b / scale + 0.5, 0.0, 511.0));
  
  return (exponent << 27) | (b << 18) | (g << 9) | r;
}

// Unpack RGB9E5 format
float3 unpackRGB9E5(uint packed)
{
  int exponent = int((packed >> 27) & 0x1F);
  uint r = packed & 0x1FF;
  uint g = (packed >> 9) & 0x1FF;
  uint b = (packed >> 18) & 0x1FF;
  
  float scale = pow(2.0, float(exponent - 24));
  
  return float3(float(r), float(g), float(b)) * scale;
}

// Generate random direction in hemisphere around normal
float3 randomHemisphere(float3 normal, inout RandomState rng)
{
  float z = randomFloat(rng);
  float r = sqrt(1.0 - z * z);
  float phi = 2.0 * PI * randomFloat(rng);
  
  float3 dir = float3(r * cos(phi), r * sin(phi), z);
  
  // Create coordinate system from normal
  float3 up = abs(normal.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
  float3 tangent = normalize(cross(up, normal));
  float3 bitangent = cross(normal, tangent);
  
  return normalize(tangent * dir.x + bitangent * dir.y + normal * dir.z);
}

// Gather photons at a point (brute force for now)
float3 gatherPhotons(float3 worldPos, float3 worldNormal)
{
  if(pushConst.usePhotonMapping == 0)
    return float3(0, 0, 0);
  
  float3 irradiance = float3(0, 0, 0);
  float gatherRadius = 0.3;
  
  uint totalPhotons = photonCounter[0];
  uint photonsFound = 0;
  
  // Stop early once we find enough photons
  const uint MAX_SEARCH = min(totalPhotons, 50000u); // Don't search more than 50k
  
  for(uint i = 0; i < MAX_SEARCH; i++)  // CHANGED: was totalPhotons
  {
    Photon p = photonBuffer[i];
    
    float dist = length(worldPos - p.position);
    
    if(dist < gatherRadius)
    {
      float3 power = unpackRGB9E5(p.powerPacked);
      
      float normalizedDist = dist / gatherRadius;
      float weight = exp(-3.0 * normalizedDist * normalizedDist);
      
      irradiance += power * weight;
      photonsFound++;
      
      // Exit early when we have enough
      if(photonsFound >= pushConst.photonGatherCount)
        break;
    }
  }
  
  if(photonsFound > 0)
  {
    irradiance /= (PI * gatherRadius * gatherRadius);
  }
  
  return irradiance;
}

// PHOTON TRACING RAY GENERATION
[shader("raygeneration")]
void photonRayGen()
{
  uint2 launchID = DispatchRaysIndex().xy;
  uint2 launchSize = DispatchRaysDimensions().xy;
  uint photonID = launchID.y * launchSize.x + launchID.x;
  
  if(photonID >= pushConst.photonsPerLight)
    return;
  
  GltfSceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
  GltfPunctual light = sceneInfo.punctualLights[0];
  
  if(sceneInfo.useSky == 1)
  {
    light.direction = sceneInfo.skySimpleParam.sunDirection;
    light.color = sceneInfo.skySimpleParam.sunColor;
    light.intensity = sceneInfo.skySimpleParam.sunIntensity;
    light.type = GltfLightType::eDirectional;
  }
  
  RandomState rng;
  initRandom(rng, photonID * 747796405u + 1u);
  
  RayDesc ray;
  
  // Divide total light power by number of photons
  float3 totalLightPower = light.color * light.intensity;
  float3 photonPower = totalLightPower / float(pushConst.photonsPerLight); 
  
  if(light.type == GltfLightType::eDirectional)
  {
    float r = sqrt(randomFloat(rng)) * 5.0;
    float theta = randomFloat(rng) * 2.0 * PI;
    
    float3 diskOffset = float3(r * cos(theta), 0, r * sin(theta));
    ray.Origin = float3(0, 0, 0) + diskOffset - light.direction * 10.0;
    ray.Direction = light.direction;
  }
  else if(light.type == GltfLightType::ePoint)
  {
    ray.Origin = light.position;
    float3 randomDir = randomHemisphere(float3(0, 1, 0), rng);
    if(randomFloat(rng) > 0.5)
      randomDir = -randomDir;
    ray.Direction = randomDir;
    
    // Account for sphere solid angle
    photonPower *= 4.0 * PI;
  }
  else if(light.type == GltfLightType::eSpot)
  {
    // ADD spot light support
    ray.Origin = light.position;
    
    // Sample within cone
    float cosAngle = cos(light.coneAngle);
    float z = randomFloat(rng) * (1.0 - cosAngle) + cosAngle;
    float phi = randomFloat(rng) * 2.0 * PI;
    float r = sqrt(1.0 - z * z);
    
    // Build direction in cone space
    float3 localDir = float3(r * cos(phi), r * sin(phi), z);
    
    // Transform to world space (align with light direction)
    float3 up = abs(light.direction.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, light.direction));
    float3 bitangent = cross(light.direction, tangent);
    
    ray.Direction = normalize(
      tangent * localDir.x +
      bitangent * localDir.y +
      light.direction * localDir.z
    );
    
    // Account for cone solid angle
    float coneSolidAngle = 2.0 * PI * (1.0 - cosAngle);
    photonPower *= coneSolidAngle;
  }
  
  ray.TMin = 0.001;
  ray.TMax = INFINITE;
  
  HitPayload payload;
  payload.color = photonPower;  
  payload.weight = 1.0;
  payload.depth = PHOTON_RAY_MARKER;
  
  TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, payload);
}

// Ray generation shader - entry point for each pixel
[shader("raygeneration")]
void rgenMain()
{
    // Get the current pixel coordinates and image dimensions
    float2 launchID = (float2)DispatchRaysIndex().xy;     // Current pixel (x,y)
    float2 launchSize = (float2)DispatchRaysDimensions().xy; // Image size (width,height)

    // Retrieve scene information from push constants
    GltfSceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    // Set ray tracing flags (0 = no special flags)
    const uint rayFlags = 0;
    
    // Convert pixel coordinates to normalized device coordinates (NDC)
    // Range: [-1,1] for both x and y
    const float2 clipCoords = launchID / launchSize * 2.0 - 1.0;
    
    // Transform from NDC to view space using inverse projection matrix
    const float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);

    // Create the primary ray
    RayDesc ray;
    // Transform camera origin (0,0,0) from view space to world space
    ray.Origin = mul(float4(0.0, 0.0, 0.0, 1.0), sceneInfo.viewInvMatrix).xyz;
    // Transform ray direction from view space to world space
    ray.Direction = mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz;
    ray.TMin = 0.001;    // Minimum distance to avoid self-intersection
    ray.TMax = INFINITE; // Maximum distance (infinite for primary rays)

    // Initialize ray payload with default values
    HitPayload payload;
    payload.color = float3(0, 0, 0);  // Start with black
    payload.weight = 1;               // Full weight for primary rays
    payload.depth = 0;                // Start at depth 0

#ifdef REFLECTION_ITERATIVE
  float3 accumulatedColor = float3(0, 0, 0);

  // Iterative reflection loop
  while(payload.depth < pushConst.depthMax && payload.weight > 0.01)
  {
    float prevWeight = payload.weight;
    TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, payload);

    accumulatedColor += payload.color * prevWeight;

    // Update ray for next iteration if continuing
    ray.Origin    = payload.rayOrigin;
    ray.Direction = payload.rayDirection;
  }

  float3 color = accumulatedColor;
#else
  // Recursive reflection (handled in closest hit shader)
  TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, payload);
  float3 color = payload.color;
#endif
    
    // Write the result to the output image
    outImage[int2(launchID)] = float4(color, 1.0);
}

// Closest hit shader - called when ray hits geometry
[shader("closesthit")]
void rchitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{

   // Check if this is a photon tracing ray (hacky but simple)
  bool isPhotonRay = (payload.depth == PHOTON_RAY_MARKER); // Photon IDs are large
  if(isPhotonRay && pushConst.usePhotonMapping == 1)
  {
    // PHOTON STORAGE PATH
    
    // Get hit information
    float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, 
                                 attr.barycentrics.x, attr.barycentrics.y);
    
    uint instanceID = InstanceIndex();
    uint triID = PrimitiveIndex();
    
    GltfSceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    GltfInstance instance = sceneInfo.instances[instanceID];
    GltfMesh mesh = sceneInfo.meshes[instance.meshIndex];
    
    // Get world position
    int3 indices = getTriangleIndices(mesh.gltfBuffer, mesh.triMesh, triID);
    float3 pos = getTriangleAttribute<float3>(mesh.gltfBuffer, mesh.triMesh.positions, 
                                              indices, barycentrics);
    float3 worldPos = float3(mul(float4(pos, 1.0), ObjectToWorld4x3()));
    
    // Store photon atomically
    uint photonIndex;
    InterlockedAdd(photonCounter[0], 1, photonIndex);
    
    if(photonIndex < pushConst.photonsPerLight)
    {
      Photon p;
      p.position = worldPos;
      p.direction = -WorldRayDirection();
      p.powerPacked = packRGB9E5(payload.color);
      p._padding = 0;
      
      photonBuffer[photonIndex] = p;
    }
    
    // Don't continue bouncing for now
    return;
  }

    // Get barycentric coordinates (pts position in triangle using weights based on tris 3 vertices)
    float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
    
    // Get instance and primitive information
    // raw functions automatically available in rt shaders!
    uint instanceID = InstanceIndex();
    uint meshID = InstanceID(); // corresponds to mesh index for each instance
    uint triID = PrimitiveIndex();

    GltfSceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    GltfInstance instance = sceneInfo.instances[instanceID];
    GltfMesh mesh = sceneInfo.meshes[instance.meshIndex];
    GltfMetallicRoughness material = sceneInfo.materials[instance.materialIndex];

    // Get world position and normal
    int3 indices = getTriangleIndices(mesh.gltfBuffer, mesh.triMesh, triID);
    float3 pos = getTriangleAttribute<float3>(mesh.gltfBuffer, mesh.triMesh.positions, indices, barycentrics);
    float3 nrm = getTriangleAttribute<float3>(mesh.gltfBuffer, mesh.triMesh.normals, indices, barycentrics);
    float3 worldPos = float3(mul(float4(pos, 1.0), ObjectToWorld4x3()));
    float3 worldNormal = normalize(mul(WorldToObject4x3(), nrm).xyz);

    // Simple PBR shading
    float3 albedo = material.baseColorFactor.xyz;
    float metallic = material.metallicFactor;
    float roughness = material.roughnessFactor;
    
    // Process light with sky override and distance attenuation
    GltfPunctual light = processLight(sceneInfo, worldPos);

    float3 N = normalize(worldNormal);
    float3 V = -WorldRayDirection();
    float3 L = normalize(light.direction);

    // Test for shadows
    float shadowFactor = testShadow(worldPos, N, light.direction, light);

    // Calculate PBR lighting with shadow factor
    float3 color = pbrMetallicRoughness(albedo, metallic, roughness, N, V, L);
    color *= light.color * light.intensity * shadowFactor;

    if(pushConst.usePhotonMapping == 1)
    {
      float3 indirectLight = gatherPhotons(worldPos, N);
      color += indirectLight * albedo / PI; // Simple diffuse reflection
    }

  // Handle reflection based on material properties
  float3 reflectionColor = float3(0, 0, 0);

#ifdef REFLECTION_RECURSIVE
  // Recursive reflection
  if(payload.depth < pushConst.depthMax && metallic > 0.01)
  {
    // Calculate reflection direction
    float3 reflectionDir = reflect(-V, N);

    // Create reflection ray
    RayDesc reflectionRay;
    reflectionRay.Origin    = worldPos + N * 0.001;  // Offset to avoid self-intersection
    reflectionRay.Direction = reflectionDir;
    reflectionRay.TMin      = 0.001;
    reflectionRay.TMax      = INFINITE;

    // Create payload for reflection ray
    HitPayload reflectionPayload;
    reflectionPayload.color  = float3(0, 0, 0);
    reflectionPayload.weight = payload.weight * metallic;
    reflectionPayload.depth  = payload.depth + 1;

    // Trace reflection ray
    TraceRay(topLevelAS, 0, 0xff, 0, 0, 0, reflectionRay, reflectionPayload);

    // Add reflection contribution
    reflectionColor = reflectionPayload.color * metallic;
  }

  payload.color = color + reflectionColor;

#else
  // Iterative reflection - set up next iteration
  payload.rayOrigin    = worldPos;        // + N * 0.001;
  payload.rayDirection = reflect(-V, N);  // Calculate reflection direction
  payload.color        = color;
  payload.weight *= metallic;
  payload.depth++;
#endif

}

// Miss shader - called when ray doesn't hit any geometry
[shader("miss")]
void rmissMain(inout HitPayload payload)
{
    // Set background color when ray misses all geometry
    // In a full implementation, this could sample from:
    // - Skybox textures
    // - Procedural sky gradients
    // - Environment maps
    GltfSceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    
    if(sceneInfo.useSky == 1)
    {
        float3 skyColor = evalSimpleSky(sceneInfo.skySimpleParam, WorldRayDirection());
        payload.color += skyColor * payload.weight;
    }
    else
    {
        payload.color += sceneInfo.backgroundColor * payload.weight;
    }
    
    payload.depth = MISS_DEPTH; // if didnt hit anything, set payload depth for this ray as our defined miss depth
}

[shader("miss")]
void rmissShadowMain(inout ShadowPayload payload)
{
  // Simple shadow miss shader - no complex lighting calculations
  payload.depth = MISS_DEPTH;
}