#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
from ml_dtypes import bfloat16
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.util import np_ndarray_type_get_shape


def my_eltwise_mul_add(dev, trace_size):
    N = 65536

    # Tile sizes
    n = 1024
    N_div_n = N // n

    # Number of cores to use
    n_cores = 1
    tiles = N_div_n // n_cores

    tensor_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    tile_ty = np.ndarray[(n,), np.dtype[bfloat16]]

    # Type used in the tile memory
    A_ty = np.ndarray[(n,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(n,), np.dtype[bfloat16]]
    C_ty = np.ndarray[(n,), np.dtype[bfloat16]]
    D_ty = np.ndarray[(n,), np.dtype[bfloat16]]

    # Type used in the memory tile which aggregates across the 2 cores
    A_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]
    B_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]
    C_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]
    D_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]

    # AIE Core Function declarations
    eltwise_mul_bf16_vector = Kernel(
        "eltwise_mul_bf16_vector", "mul.o", [tile_ty, tile_ty, tile_ty]
    )
    eltwise_add_bf16_vector = Kernel(
        "eltwise_add_bf16_vector", "add.o", [tile_ty, tile_ty, tile_ty]
    )

    # AIE-array data movement with object fifos
    # Input A
    inA = ObjectFifo(A_memTile_ty, name="inA")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(A_memTile_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    inA_fifos = inA.cons().split(
        of_offsets,
        obj_types=[A_ty] * n_cores,
        names=[f"memA{i}" for i in range(n_cores)],
    )

    # Input B
    inB = ObjectFifo(B_memTile_ty, name="inB")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(B_memTile_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    inB_fifos = inB.cons().split(
        of_offsets,
        obj_types=[B_ty] * n_cores,
        names=[f"memB{i}" for i in range(n_cores)],
    )

    # Input C
    inC = ObjectFifo(C_memTile_ty, name="inC")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(C_memTile_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    inC_fifos = inC.cons().split(
        of_offsets,
        obj_types=[C_ty] * n_cores,
        names=[f"memC{i}" for i in range(n_cores)],
    )

    # Intermediate buffer for multiplication result
    intBuf = ObjectFifo(D_memTile_ty, name="intBuf")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(D_memTile_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    intBuf_prod_fifos = intBuf.prod().join(
        of_offsets,
        obj_types=[D_ty] * n_cores,
        names=[f"intBuf_prod{i}" for i in range(n_cores)],
    )
    intBuf_cons_fifos = intBuf.cons().split(
        of_offsets,
        obj_types=[D_ty] * n_cores,
        names=[f"intBuf_cons{i}" for i in range(n_cores)],
    )

    # Output D (final result)
    outD = ObjectFifo(D_memTile_ty, name="outD")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(D_memTile_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    outD_fifos = outD.prod().join(
        of_offsets,
        obj_types=[D_ty] * n_cores,
        names=[f"memD{i}" for i in range(n_cores)],
    )

    # Task for the cores to perform multiplication
    def mul_core_fn(of_a, of_b, of_out, eltwise_mul):
        for _ in range_(tiles):
            elem_out = of_out.acquire(1)
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            eltwise_mul(elem_in_a, elem_in_b, elem_out)
            of_a.release(1)
            of_b.release(1)
            of_out.release(1)

    # Task for the cores to perform addition
    def add_core_fn(of_a, of_b, of_out, eltwise_add):
        for _ in range_(tiles):
            elem_out = of_out.acquire(1)
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            eltwise_add(elem_in_a, elem_in_b, elem_out)
            of_a.release(1)
            of_b.release(1)
            of_out.release(1)

    # Create workers to perform the multiplication
    workers = []
    for i in range(n_cores):
        workers.append(
            Worker(
                mul_core_fn,
                fn_args=[
                    inA_fifos[i].cons(),
                    inB_fifos[i].cons(),
                    intBuf_prod_fifos[i].prod(),
                    eltwise_mul_bf16_vector,
                ],
            )
        )

    # Create workers to perform the addition
    for i in range(n_cores):
        workers.append(
            Worker(
                add_core_fn,
                fn_args=[
                    intBuf_cons_fifos[i].cons(),
                    inC_fifos[i].cons(),
                    outD_fifos[i].prod(),
                    eltwise_add_bf16_vector,
                ],
            )
        )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty, tensor_ty) as (A, B, C, D):
        rt.start(*workers)
        rt.fill(inA.prod(), A)
        rt.fill(inB.prod(), B)
        rt.fill(inC.prod(), C)
        rt.drain(outD.cons(), D, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        dev = NPU2()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))
    trace_size = 0 if (len(sys.argv) != 3) else int(sys.argv[2])
except ValueError:
    print("Argument is not an integer")
module = my_eltwise_mul_add(dev, trace_size)
print(module)