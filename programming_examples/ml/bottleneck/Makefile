#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

mlirFileName = aie

aie_py_src=bottleneck.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=bottleneck_placed.py
endif

VPATH := ${srcdir}/../../../aie_kernels/aie2

all: build/conv2dk1.o build/conv2dk3.o build/conv2dk1_skip.o build/final.xclbin

build/${mlirFileName}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< > $@

build/conv2dk1.o: conv2dk1.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2_FLAGS} -DINT8_ACT -c $< -o ${@F}

build/conv2dk3.o: conv2dk3.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2_FLAGS} -DUINT8_ACT -c $< -o ${@F}

build/conv2dk1_skip.o: conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2_FLAGS} -DINT8_ACT -c $< -o$ ${@F} 

build/final.xclbin: build/${mlirFileName}.mlir build/conv2dk1.o build/conv2dk3.o build/conv2dk1_skip.o  
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--alloc-scheme=basic-sequential \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%) 

clean:
	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log .xclbin sim \
		chess* *.o insts.bin \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

run_py: build/final.xclbin
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE
