# CUDA-parallel-cumulative-sum-scan
CUDA-based Parallel Cumulative Sum (Prefix Scan) implementation using Python Numba. Demonstrates GPU parallel computing concepts, memory transfer, kernel execution, and scan operations on MNIST-style datasets.
# CUDA Parallel Cumulative Sum (Scan)

## Overview
This project implements Parallel Cumulative Sum (Prefix Scan) using CUDA Python (Numba). The GPU computes cumulative sums efficiently by executing multiple threads in parallel.

## Objective
Implement Parallel Cumulative Sum (Scan) using CUDA and compare GPU-based computation with traditional sequential processing.

## Dataset
### DS4 – MNIST Handwritten Digits Dataset

- Grayscale handwritten digit images (0–9)
- Image Size: 28 × 28 pixels
- Used to demonstrate parallel scan operations on large datasets

## Theory

A cumulative sum (scan) computes the running sum of elements in an array.

### Example

Input:
```text
[1, 2, 3, 4, 5]
```

Output:
```text
[1, 3, 6, 10, 15]
```

Formula:

```text
Output[i] = Input[0] + Input[1] + ... + Input[i]
```

CUDA performs this computation using parallel GPU threads.

---

## Algorithm

1. Load input data.
2. Allocate GPU memory.
3. Copy input data from CPU to GPU.
4. Launch CUDA kernel.
5. Each thread computes cumulative sum up to its index.
6. Store results in output array.
7. Copy results back to CPU.
8. Display output.

---

## Result

The Parallel Cumulative Sum (CUDA Scan) was successfully implemented using CUDA Python (Numba). The GPU computed the cumulative sum correctly and demonstrated parallel processing concepts.

---

## Technologies Used

- Python
- CUDA
- Numba
- NumPy
- Google Colab
- NVIDIA GPU

---

## Author

Jay Prakash
M.Tech Artificial Intelligence
Dr. B.R. Ambedkar National Institute of Technology Jalandhar
