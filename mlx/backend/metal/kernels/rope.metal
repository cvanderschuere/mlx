// Copyright © 2023-2024 Apple Inc.

#include <metal_math>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/utils.h"
template <typename T, bool traditional, bool forward>
void rope_single_impl(
    const device T* in,          // Shape: [seq_length, num_heads * head_dim]
    device T* out,
    constant const int& offset,
    const float inv_freq,
    constant const float& scale,
    constant const size_t& stride,  // Stride between sequence elements
    uint2 pos,                   // (x: head_dim position, y: seq position)
    uint2 grid) {                // grid.x = head_dim/2 (for pair processing)
  float L = scale * static_cast<float>(offset);

  // Compute costheta, sintheta for this sequence position
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);

  // Compute the input and output indices for adjacent pairs
  uint index_1, index_2;
  if (traditional) {
    index_1 = 2 * pos.x + pos.y * stride;
    index_2 = index_1 + 1;
  } else {
    // Update to handle adjacent pairs in head_dim
    index_1 = pos.x * 2 + pos.y * stride;     // Even elements
    index_2 = index_1 + 1;                    // Odd elements (adjacent pair)
  }

  // Read adjacent pairs of elements
  float x1 = static_cast<float>(in[index_1]);  // Element at position x
  float x2 = static_cast<float>(in[index_2]);  // Element at position x+1
  
  float rx1, rx2;
  if (forward) {
    // For each adjacent pair (x1,x2):
    rx1 = x1 * costheta - x2 * sintheta;  // (a*cos(θ) - b*sin(θ))
    rx2 = x1 * sintheta + x2 * costheta;  // (a*sin(θ) + b*cos(θ))
  } else {
    // Inverse rotation
    rx1 = x2 * sintheta + x1 * costheta;
    rx2 = x2 * costheta - x1 * sintheta;
  }
  
  // Write rotated pairs back to adjacent positions
  out[index_1] = static_cast<T>(rx1);
  out[index_2] = static_cast<T>(rx2);
}

template <typename T, bool traditional, bool forward>
[[kernel]] void rope_single(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset,
    constant const float& scale,
    constant const size_t& stride,
    constant const float& base [[buffer(10)]],
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  float inv_freq = metal::exp2(-d * base);
  rope_single_impl<T, traditional, forward>(
      in, out, offset, inv_freq, scale, stride, pos, grid);
}

template <typename T, bool traditional, bool forward>
[[kernel]] void rope_single_freqs(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset,
    constant const float& scale,
    constant const size_t& stride,
    const device float* freqs [[buffer(10)]],
    constant const size_t& freq_stride [[buffer(11)]],
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  float inv_freq = 1.0 / (freqs[freq_stride * pos.x]);
  rope_single_impl<T, traditional, forward>(
      in, out, offset, inv_freq, scale, stride, pos, grid);
}

template <typename T, bool traditional, bool forward, int N = 4>
void rope_impl(
    const device T* in,          // Shape: [batch_size, seq_length, num_heads * head_dim]
    device T* out,
    constant const int& offset,
    const float inv_freq,
    constant const float& scale,
    constant const size_t strides[3],        // [batch_stride, seq_stride, head_stride]
    constant const size_t out_strides[3],    // Output strides matching input dimensions
    constant const size_t& n_batch,
    uint3 pos,                   // (x: head_dim position, y: seq position, z: batch chunk)
    uint3 grid) {                // grid.x = head_dim/2 (for pair processing)
  float L = scale * static_cast<float>(pos.y + offset);

  // Compute costheta, sintheta for this sequence position
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);

  // Compute the input and output indices for adjacent pairs
  size_t in_index_1, in_index_2;
  size_t out_index_1, out_index_2;
  if (traditional) {
    // Keep traditional implementation
    out_index_1 = 2 * pos.x * out_strides[2] + pos.y * out_strides[1] +
        N * pos.z * out_strides[0];
    out_index_2 = out_index_1 + 1;
    in_index_1 =
        2 * pos.x * strides[2] + pos.y * strides[1] + N * pos.z * strides[0];
    in_index_2 = in_index_1 + strides[2];
  } else {
    // Update to handle adjacent pairs in head_dim
    // For head_dim value at position x, we want to pair it with x+1
    out_index_1 = pos.x * 2 * out_strides[2] + pos.y * out_strides[1] +
        N * pos.z * out_strides[0];  // Position for even elements
    out_index_2 = out_index_1 + out_strides[2];  // Next element (odd)
    
    // Input indices similarly handle adjacent pairs
    in_index_1 = pos.x * 2 * strides[2] + pos.y * strides[1] + 
        N * pos.z * strides[0];  // Even elements
    in_index_2 = in_index_1 + strides[2];  // Odd elements (x+1)
  }

  // Process N items in the batch dimension at once
  for (int i = 0; i < N && pos.z * N + i < n_batch; ++i) {
    // Read adjacent pairs of elements
    float x1 = static_cast<float>(in[in_index_1]);  // Element at position x
    float x2 = static_cast<float>(in[in_index_2]);  // Element at position x+1
    
    float rx1, rx2;
    if (forward) {
      // For each adjacent pair (x1,x2):
      rx1 = x1 * costheta - x2 * sintheta;  // (a*cos(θ) - b*sin(θ))
      rx2 = x1 * sintheta + x2 * costheta;  // (a*sin(θ) + b*cos(θ))
    } else {
      // Inverse rotation
      rx1 = x2 * sintheta + x1 * costheta;
      rx2 = x2 * costheta - x1 * sintheta;
    }
    
    // Write rotated pairs back to adjacent positions
    out[out_index_1] = static_cast<T>(rx1);
    out[out_index_2] = static_cast<T>(rx2);
    
    // Move to next batch item
    in_index_1 += strides[0];
    in_index_2 += strides[0];
    out_index_1 += out_strides[0];
    out_index_2 += out_strides[0];
  }
}

template <typename T, bool traditional, bool forward, int N = 4>
[[kernel]] void rope(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset,
    constant const float& scale,
    constant const size_t strides[3],
    constant const size_t out_strides[3],
    constant const size_t& n_batch,
    constant const float& base [[buffer(10)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  float inv_freq = metal::exp2(-d * base);
  rope_impl<T, traditional, forward, N>(
      in,
      out,
      offset,
      inv_freq,
      scale,
      strides,
      out_strides,
      n_batch,
      pos,
      grid);
}

template <typename T, bool traditional, bool forward, int N = 4>
[[kernel]] void rope_freqs(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset,
    constant const float& scale,
    constant const size_t strides[3],
    constant const size_t out_strides[3],
    constant const size_t& n_batch,
    const device float* freqs [[buffer(10)]],
    constant const size_t& freq_stride [[buffer(11)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  float inv_freq = 1.0 / (freqs[freq_stride * pos.x]);
  rope_impl<T, traditional, forward, N>(
      in,
      out,
      offset,
      inv_freq,
      scale,
      strides,
      out_strides,
      n_batch,
      pos,
      grid);
}

// clang-format off
#define instantiate_rope_g(name, type, traditional, forward) \
  template [[host_name("rope_" #name)]] [[kernel]] void      \
  rope<type, traditional, forward>(                          \
      const device type* in [[buffer(0)]],                   \
      device type* out [[buffer(1)]],                        \
      constant const int& offset,                            \
      constant const float& scale,                           \
      constant const size_t strides[3],                      \
      constant const size_t out_strides[3],                  \
      constant const size_t& n_batch,                        \
      constant const float& base [[buffer(10)]],             \
      uint3 pos [[thread_position_in_grid]],                 \
      uint3 grid [[threads_per_grid]]);                      \
  template [[host_name("rope_freqs_" #name)]]                \
  [[kernel]] void rope_freqs<type, traditional, forward>(    \
      const device type* in [[buffer(0)]],                   \
      device type* out [[buffer(1)]],                        \
      constant const int& offset,                            \
      constant const float& scale,                           \
      constant const size_t strides[3],                      \
      constant const size_t out_strides[3],                  \
      constant const size_t& n_batch,                        \
      const device float* freqs [[buffer(10)]],              \
      constant const size_t& freq_stride [[buffer(11)]],     \
      uint3 pos [[thread_position_in_grid]],                 \
      uint3 grid [[threads_per_grid]]);

#define instantiate_rope_s(name, type, traditional, forward)     \
  template [[host_name("rope_single_" #name)]] [[kernel]] void   \
  rope_single<type, traditional, forward>(                       \
      const device type* in [[buffer(0)]],                       \
      device type* out [[buffer(1)]],                            \
      constant const int& offset,                                \
      constant const float& scale,                               \
      constant const size_t& stride,                             \
      constant const float& base [[buffer(10)]],                 \
      uint2 pos [[thread_position_in_grid]],                     \
      uint2 grid [[threads_per_grid]]);                          \
  template [[host_name("rope_single_freqs_" #name)]]             \
  [[kernel]] void rope_single_freqs<type, traditional, forward>( \
      const device type* in [[buffer(0)]],                       \
      device type* out [[buffer(1)]],                            \
      constant const int& offset,                                \
      constant const float& scale,                               \
      constant const size_t& stride,                             \
      const device float* freqs [[buffer(10)]],                  \
      constant const size_t& freq_stride [[buffer(11)]],         \
      uint2 pos [[thread_position_in_grid]],                     \
      uint2 grid [[threads_per_grid]]);

#define instantiate_rope(name, type, traditional, forward) \
  instantiate_rope_s(name, type, traditional, forward)     \
  instantiate_rope_g(name, type, traditional, forward)

instantiate_rope(traditional_float16, half, true, true)
instantiate_rope(traditional_bfloat16, bfloat16_t, true, true)
instantiate_rope(traditional_float32, float, true, true)
instantiate_rope(float16, half, false, true)
instantiate_rope(bfloat16, bfloat16_t, false, true)
instantiate_rope(float32, float, false, true)
instantiate_rope(vjp_traditional_float16, half, true, false)
instantiate_rope(vjp_traditional_bfloat16, bfloat16_t, true, false)
instantiate_rope(vjp_traditional_float32, float, true, false)
instantiate_rope(vjp_float16, half, false, false)
instantiate_rope(vjp_bfloat16, bfloat16_t, false, false)
instantiate_rope(vjp_float32, float, false, false) // clang-format on
