#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import argparse
from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col4, NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D

dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}

def ceildiv(a, b):
    return (a + b - 1) // b

def linear_layer(
    dev,
    M,         # Batch size
    K,         # Input features
    N,         # Output features
    m,         # Tile size for batch
    k,         # Tile size for input features
    n,         # Tile size for output features
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    trace_size,
    generate_taps=False,
):
    # Validation
    assert M % m == 0
    assert K % k == 0
    assert N % n == 0

    # Set vector sizes based on device and data type
    if dev == "npu":
        if dtype_in_str == "bf16":
            r = 4
            s = 8
            t = 4
        elif dtype_in_str == "i8":
            r = 4
            s = 8
            t = 8
        elif dtype_in_str == "i16":
            r = 4
            s = 4
            t = 4
    else:
        if dtype_in_str == "bf16":
            r = 8
            s = 8
            t = 8
        elif dtype_in_str == "i8":
            r = 8
            s = 8
            t = 8
        elif dtype_in_str == "i16":
            r = 4
            s = 4
            t = 8

    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

    # Compute tile counts
    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    # TAPs for visualization
    W_taps = []
    X_taps = []
    B_taps = []
    Y_taps = []

    # Define tensor types
    # W = weight matrix (K x N)
    # X = input matrix (M x K)
    # B = bias vector (N)
    # Y = output matrix (M x N)
    W_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    X_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(N,), np.dtype[dtype_in]]
    Y_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    
    # Tile types
    w_ty = np.ndarray[(k, n), np.dtype[dtype_in]]  # Weight tile
    x_ty = np.ndarray[(m, k), np.dtype[dtype_in]]  # Input tile
    b_ty = np.ndarray[(n,), np.dtype[dtype_in]]    # Bias tile
    y_ty = np.ndarray[(m, n), np.dtype[dtype_out]] # Output tile

    # AIE Core Function declarations
    func_type = "" if vectorized else "scalar_"
    zero_kernel = Kernel(
        f"zero_{func_type}{dtype_out_str}", f"mm_{m}x{k}x{n}.o", [y_ty]
    )
    matmul_vectorized_func_name = (
        f"matmul_{dtype_in_str}_{dtype_out_str}"
        if not b_col_maj
        else f"matmul_{dtype_in_str}_{dtype_out_str}_b_col_maj"
    )
    matmul_kernel = Kernel(
        matmul_vectorized_func_name,
        f"mm_{m}x{k}x{n}.o",
        [x_ty, w_ty, y_ty],
    )
    eltwise_add_kernel = Kernel(
        f"eltwise_add_{dtype_in_str}_vector", "add.o", [y_ty, b_ty, y_ty]
    )

    # AIE-array data movement with object fifos
    # Input X (input feature matrix)
    inX = ObjectFifo(x_ty, name="inX")
    x_dims = None
    if vectorized:
        x_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memX = inX.cons().forward(name="memX", dims_to_stream=x_dims)

    # Input W (weight matrix)
    inW = ObjectFifo(w_ty, name="inW")
    w_dims = None
    if vectorized:
        if b_col_maj:
            w_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            w_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    memW = inW.cons().forward(name="memW", dims_to_stream=w_dims)

    # Input B (bias vector)
    inB = ObjectFifo(b_ty, name="inB")
    memB = inB.cons().forward(name="memB")

    # Output Y
    memY = ObjectFifo(y_ty, name="memY")
    y_dims = None
    if vectorized:
        y_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outY = memY.cons().forward(name="outY", dims_to_stream=y_dims)

    # Task for matrix multiplication worker
    def matmul_fn(of_x, of_w, of_intermediate, zero, matmul):
        for _ in range_(tiles) if tiles > 1 else range(1):
            elem_out = of_intermediate.acquire(1)
            zero(elem_out)

            # Matrix multiplication: Y = X * W
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                elem_in_x = of_x.acquire(1)
                elem_in_w = of_w.acquire(1)
                matmul(elem_in_x, elem_in_w, elem_out)
                of_x.release(1)
                of_w.release(1)
            
            of_intermediate.release(1)

    # Task for element-wise addition worker
    def eltwise_add_fn(of_intermediate, of_b, of_y, eltwise_add):
        for _ in range_(tiles) if tiles > 1 else range(1):
            elem_in = of_intermediate.acquire(1)
            elem_in_b = of_b.acquire(1)
            elem_out = of_y.acquire(1)
            
            # Element-wise addition: Y = intermediate + B
            eltwise_add(elem_in, elem_in_b, elem_out)
            
            of_intermediate.release(1)
            of_b.release(1)
            of_y.release(1)

    # Create an intermediate ObjectFIFO to pass data between the two workers
    memIntermediate = ObjectFifo(y_ty, name="memIntermediate")
    outIntermediate = memIntermediate.cons().forward(
        name="outIntermediate", 
        dims_to_stream=y_dims,
        placement=Tile(0, 1)  # Place on a different tile than the workers
    )

    # Create separate workers with explicit placement on different tiles
    matmul_worker = Worker(
        matmul_fn, 
        [memX.cons(), memW.cons(), memIntermediate.prod(), zero_kernel, matmul_kernel],
        placement=Tile(0, 2)  # Place on tile (0,2)
    )

    eltwise_worker = Worker(
        eltwise_add_fn,
        [outIntermediate.cons(), memB.cons(), memY.prod(), eltwise_add_kernel],
        placement=Tile(0, 3)  # Place on tile (0,3)
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(W_ty, X_ty, B_ty, Y_ty) as (W, X, B, Y):
        rt.start(matmul_worker, eltwise_worker)  # Start both workers

        # only do 4 tile rows at a time before synchronizing, so we can reuse BDs
        rows_per_block = 4

        # Define tensor access patterns for inputs/outputs
        X_tiles = TensorTiler2D.group_tiler(
            (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n
        )

        # Weight matrix tiling
        if b_col_maj:
            w_tap = TensorTiler2D.group_tiler((K, N), (k, n), (K_div_k, N_div_n))[0]
        else:
            w_tap = TensorTiler2D.group_tiler(
                (K, N), (k, n), (K_div_k, N_div_n), tile_group_col_major=True
            )[0]

        # Bias vector tiling
        b_tap = TensorTiler2D.group_tiler((1, N), (1, n), (1, N_div_n))[0]

        Y_tiles = TensorTiler2D.group_tiler((M, N), (m, n), (rows_per_block // 2, N_div_n))
        y_index = 0

        tgs = []
        for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
            for pingpong in [0, 1]:
                row_base = (
                    tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                )
                num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                if num_tile_rows <= 0:
                    break
                tgs.append(rt.task_group())
                for tile_row in range(num_tile_rows):
                    # -- X --
                    tile_offset = (row_base + tile_row) % len(X_tiles)
                    rt.fill(inX.prod(), X, tap=X_tiles[tile_offset], task_group=tgs[-1])
                    X_taps.append(X_tiles[tile_offset])

                    # -- W --
                    rt.fill(inW.prod(), W, tap=w_tap, task_group=tgs[-1])
                    W_taps.append(w_tap)
                    
                    # -- B --
                    rt.fill(inB.prod(), B, tap=b_tap, task_group=tgs[-1])
                    B_taps.append(b_tap)

                # -- Y --
                rt.drain(
                    outY.cons(), Y, tap=Y_tiles[y_index], task_group=tgs[-1], wait=True
                )
                Y_taps.append(Y_tiles[y_index])
                y_index += 1

                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]

        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    if generate_taps:
        return (
            TensorAccessSequence.from_taps(W_taps),
            TensorAccessSequence.from_taps(X_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(Y_taps),
        )

    # Create the program from the device type and runtime
    if dev == "npu":
        dev_ty = NPU1Col4()
    else:
        dev_ty = NPU2()
    my_program = Program(dev_ty, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())
    return module

def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Linear Layer MLIR Design",
        description="Emits MLIR code for a linear layer (W*x+b) design of the given input size",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu2")
    argparser.add_argument("-M", type=int, default=128)  # Batch size
    argparser.add_argument("-K", type=int, default=256)  # Input features
    argparser.add_argument("-N", type=int, default=128)  # Output features
    argparser.add_argument("-m", type=int, default=32)   # Tile size for batch
    argparser.add_argument("-k", type=int, default=64)   # Tile size for input features
    argparser.add_argument("-n", type=int, default=32)   # Tile size for output features
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="bf16"
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="bf16",
    )
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument(
        "--generate-taps",
        action="store_true",
        help="Generate TensorAccessPatterns for visualization",
    )
    args = argparser.parse_args()
    maybe_module = linear_layer(
        args.dev,
        args.M,
        args.K,
        args.N,
        args.m,
        args.k,
        args.n,
        args.dtype_in,
        args.dtype_out,
        args.b_col_maj,
        args.trace_size,
        args.generate_taps,
    )
    if args.generate_taps:
        return maybe_module
    else:
        print(maybe_module)

if __name__ == "__main__":
    main()