#!/usr/bin/env dub
/+ dub.json:
{
    "name": "RNN_example",
    "targetType":"executable",
    "lflags": ["-L/usr/local/cuda/lib64"],
    "dependencies": {
    		"d-nv": "0.0.1",
        "derelict-cudnn": {"path": "."}
    },
    "libs": [
        "nvrtc",
        "cudart",
        "cublas",
        "cudnn"
    ]
}
+/

// RNN example
import std.stdio;
import std.conv : to;
import std.string : fromStringz;

import derelict.cuda;
import derelict.cudnn;
import dnv;


void cudaErrCheck(string file = __FILE__, int line = __LINE__)(int result) {
    if (result != CUDNN_STATUS_SUCCESS) {
        DerelictCUDARuntime.load();
        throw new Exception("cudaError_t: " ~ cudaGetErrorString(result).fromStringz.to!string, file, line);
    }
}


void cudnnErrCheck(string file = __FILE__, int line = __LINE__)(int result) {
    if (result != CUDNN_STATUS_SUCCESS) {
        throw new Exception("cudnnStatus_t: " ~ cudnnGetErrorString(result).fromStringz.to!string, file, line);
    }
}

enum code = Code("fill", q{float *data, int numElements, float value}, q{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < numElements) {
            data[tid] = value;
        }
    });

struct CustomLauncher {
    uint[3] grids = [1024, 1, 1];
    uint[3] blocks;

    void setup(float *data, int numElements, float value) {
        uint bx = to!uint((grids[0] + numElements - 1) / grids[0]);
        blocks = [bx, 1, 1];
    }
}


void main(string[] argv) {
    import std.getopt;
    int seqLength = 20;
    int numLayers = 2;
    int hiddenSize = 128; // 256 error!
    int miniBatch = 64;
    float dropout = 0.0;
    bool bidirectional = false;
    int mode = 0;
    int persistent = 0;

    auto parsed = getopt(
                         argv,
                         "seqLength", "sequence length", &seqLength,
                         "numLayers", "number of layers", &numLayers,
                         "hiddenSize", "size of hidden units", &hiddenSize,
                         // "inputSize", "input size", &inputSize,
                         "miniBatch", "mini batch", &miniBatch,
                         "dropout", "dropout rate", &dropout,
                         "bidirectional", "bidirectional RNN", &bidirectional,
                         "mode", "Modes: 0 = RNN_RELU, 1 = RNN_TANH, 2 = LSTM, 3 = GRU", &mode,
                         "persistent", "Persistent RNN", &persistent
                         );
    if (parsed.helpWanted) {
        defaultGetoptPrinter("RNN example", parsed.options);
    }
    int inputSize = hiddenSize;

    DerelictCUDADriver.load();
    DerelictCuDNN.load();
    writeln("RNN example!");
    // cudnnCheck(CUDNN_STATUS_SUCCESS);
    // cudnnCheck(CUDNN_STATUS_SUCCESS + 1);

    import std.stdio : File;
    auto fp = File("result.txt", "w");

    // -------------------------
    // Create cudnn context
    // -------------------------
    cudnnHandle_t cudnnHandle;
    scope(exit) cudnnDestroy(cudnnHandle);
    cudnnErrCheck(cudnnCreate(&cudnnHandle));

    // Memory allocation. hx, cx, dhx, dcx, hy, cy, dhy and dcy can be NULL.
    auto x = new Array!float(seqLength * inputSize * miniBatch);

    auto hx = new Array!float(numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1));
    auto cx = new Array!float(numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1));

    auto dx = new Array!float(seqLength * inputSize * miniBatch);
    auto dhx = new Array!float(numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1));
    auto dcx = new Array!float(numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1));

    auto y = new Array!float(seqLength * hiddenSize * miniBatch * (bidirectional ? 2 : 1));
    auto hy = new Array!float(numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1));
    auto cy = new Array!float(numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1));

    auto dy = new Array!float(seqLength * hiddenSize * miniBatch * (bidirectional ? 2 : 1));
    auto dhy = new Array!float(numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1));
    auto dcy = new Array!float(numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1));

    // Set up tensor descriptors. x/y/dx/dy are arrays, one per time step.
    cudnnTensorDescriptor_t[] xDesc, yDesc, dxDesc, dyDesc;
    cudnnTensorDescriptor_t hxDesc, cxDesc;
    cudnnTensorDescriptor_t hyDesc, cyDesc;
    cudnnTensorDescriptor_t dhxDesc, dcxDesc;
    cudnnTensorDescriptor_t dhyDesc, dcyDesc;

    xDesc.length = seqLength;
    yDesc.length = seqLength;
    dxDesc.length = seqLength;
    dyDesc.length = seqLength;

    int[3] dimA, strideA;

    foreach (i; 0 .. seqLength) {
        cudnnErrCheck(cudnnCreateTensorDescriptor(&xDesc[i]));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&yDesc[i]));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&dxDesc[i]));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&dyDesc[i]));

        dimA[0] = miniBatch;
        dimA[1] = inputSize;
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        import std.conv : to;
        cudnnErrCheck(cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3,
                                                 // cast(const(int))
                                                 dimA.ptr,
                                                 // cast(const(int))
                                                 strideA.ptr));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(dxDesc[i], CUDNN_DATA_FLOAT, 3,
                                                 // cast(const(int))
                                                 dimA.ptr,
                                                 // cast(const(int))
                                                 strideA.ptr));

        dimA[0] = miniBatch;
        dimA[1] = bidirectional ? hiddenSize * 2 : hiddenSize;
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        cudnnErrCheck(cudnnSetTensorNdDescriptor(yDesc[i], CUDNN_DATA_FLOAT, 3,
                                                 // cast(const(int))
                                                 dimA.ptr,
                                                 // cast(const(int))
                                                 strideA.ptr));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(dyDesc[i], CUDNN_DATA_FLOAT, 3,
                                                 // cast(const(int))
                                                 dimA.ptr,
                                                 // cast(const(int))
                                                 strideA.ptr));
    }

    dimA[0] = numLayers * (bidirectional ? 2 : 1);
    dimA[1] = miniBatch;
    dimA[2] = hiddenSize;

    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;

    cudnnErrCheck(cudnnCreateTensorDescriptor(&hxDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&cxDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&hyDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&cyDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&dhxDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&dcxDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&dhyDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&dcyDesc));

    cudnnErrCheck(cudnnSetTensorNdDescriptor(hxDesc, CUDNN_DATA_FLOAT, 3, dimA.ptr, strideA.ptr));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(cxDesc, CUDNN_DATA_FLOAT, 3, dimA.ptr, strideA.ptr));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(hyDesc, CUDNN_DATA_FLOAT, 3, dimA.ptr, strideA.ptr));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(cyDesc, CUDNN_DATA_FLOAT, 3, dimA.ptr, strideA.ptr));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(dhxDesc, CUDNN_DATA_FLOAT, 3, dimA.ptr, strideA.ptr));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(dcxDesc, CUDNN_DATA_FLOAT, 3, dimA.ptr, strideA.ptr));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(dhyDesc, CUDNN_DATA_FLOAT, 3, dimA.ptr, strideA.ptr));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(dcyDesc, CUDNN_DATA_FLOAT, 3, dimA.ptr, strideA.ptr));


    // -------------------------
    // Set up the dropout descriptor (needed for the RNN descriptor)
    // -------------------------
    ulong seed = 1337UL; // Pick a seed.

    cudnnDropoutDescriptor_t dropoutDesc;
    cudnnErrCheck(cudnnCreateDropoutDescriptor(&dropoutDesc));

    // How much memory does dropout need for states?
    // These states are used to generate random numbers internally
    // and should not be freed until the RNN descriptor is no longer used
    size_t stateSize;
    cudnnErrCheck(cudnnDropoutGetStatesSize(cudnnHandle, &stateSize));
    auto states = new Array!void(stateSize);
    cudnnErrCheck(cudnnSetDropoutDescriptor(dropoutDesc,
                                            cudnnHandle,
                                            dropout,
                                            states.data,
                                            stateSize,
                                            seed));

    // -------------------------
    // Set up the RNN descriptor
    // -------------------------
    cudnnRNNDescriptor_t rnnDesc;
    cudnnRNNMode_t RNNMode;
    cudnnRNNAlgo_t RNNAlgo;

    cudnnErrCheck(cudnnCreateRNNDescriptor(&rnnDesc));

    if      (mode == 0) RNNMode = CUDNN_RNN_RELU;
    else if (mode == 1) RNNMode = CUDNN_RNN_TANH;
    else if (mode == 2) RNNMode = CUDNN_LSTM;
    else if (mode == 3) RNNMode = CUDNN_GRU;

    // Persistent RNNs are only supported on Pascal+ GPUs.
    if      (persistent == 0) RNNAlgo = CUDNN_RNN_ALGO_STANDARD;
    else { assert(false, "not supported yet"); }
    // TODO:
    // else if (persistent == 1) RNNAlgo = CUDNN_RNN_ALGO_PERSIST_STATIC;
    // else if (persistent == 2) RNNAlgo = CUDNN_RNN_ALGO_PERSIST_DYNAMIC;

    cudnnErrCheck(cudnnSetRNNDescriptor_v6(cudnnHandle,
                                           rnnDesc,
                                           hiddenSize,
                                           numLayers,
                                           dropoutDesc,
                                           CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
                                           bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
                                           RNNMode,
                                           RNNAlgo, // Can be changed to use persistent RNNs on Pascal+ GPUs.
                                           CUDNN_DATA_FLOAT));

    // -------------------------
    // Set up parameters
    // -------------------------
    // This needs to be done after the rnn descriptor is set as otherwise
    // we don't know how many parameters we have to allocate
    cudnnFilterDescriptor_t wDesc, dwDesc;

    cudnnErrCheck(cudnnCreateFilterDescriptor(&wDesc));
    cudnnErrCheck(cudnnCreateFilterDescriptor(&dwDesc));

    size_t weightsSize;
    cudnnErrCheck(cudnnGetRNNParamsSize(cudnnHandle, rnnDesc, xDesc[0], &weightsSize, CUDNN_DATA_FLOAT));

    int[3] dimW;
    dimW[0] = cast(int) (weightsSize / float.sizeof);
    dimW[1] = 1;
    dimW[2] = 1;

    cudnnErrCheck(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW.ptr));
    cudnnErrCheck(cudnnSetFilterNdDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW.ptr));

    auto w = new Array!void(weightsSize);
    auto dw = new Array!void(weightsSize);

    // -------------------------
    // Set up work space and reserved memory
    // -------------------------
    auto initGPUData = new TypedKernel!(code, CustomLauncher);

    size_t workSize;
    size_t reserveSize;
    // Need for every pass
    cudnnErrCheck(cudnnGetRNNWorkspaceSize(cudnnHandle, rnnDesc, seqLength, xDesc.ptr, &workSize));
    // Only needed in training, shouldn't be touched between passes.
    cudnnErrCheck(cudnnGetRNNTrainingReserveSize(cudnnHandle, rnnDesc, seqLength, xDesc.ptr, &reserveSize));
    auto workspace = new Array!void(workSize);
    auto reserveSpace = new Array!void(reserveSize);

    // We initialise to something simple.
    // Matrices are initialised to 1 / matrixSize, biases to 1, data is 1.
    initGPUData(x.data, seqLength * inputSize * miniBatch, 1.0f);
    if (hx.data != null) initGPUData(hx.data, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.0f);
    if (cx.data != null) initGPUData(cx.data, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.0f);

    initGPUData(dy.data, seqLength * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.0f);
    if (dhy.data != null) initGPUData(dhy.data, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.0f);
    if (dcy.data != null) initGPUData(dcy.data, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.0f);

    // Weights
    int numLinearLayers = 0;
    if (RNNMode == CUDNN_RNN_RELU || RNNMode == CUDNN_RNN_TANH) {
        numLinearLayers = 2;
    }
    else if (RNNMode == CUDNN_LSTM) {
        numLinearLayers = 8;
    }
    else if (RNNMode == CUDNN_GRU) {
        numLinearLayers = 6;
    }

    for (int layer = 0; layer < numLayers * (bidirectional ? 2 : 1); layer++) {
        for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
            cudnnFilterDescriptor_t linLayerMatDesc;
            cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerMatDesc));
            float *linLayerMat;
            cudnnErrCheck(cudnnGetRNNLinLayerMatrixParams( cudnnHandle,
                                                           rnnDesc,
                                                           layer,
                                                           xDesc[0],
                                                           wDesc,
                                                           w.data,
                                                           linLayerID,
                                                           linLayerMatDesc,
                                                           cast(void**) &linLayerMat));

            cudnnDataType_t dataType;
            cudnnTensorFormat_t format;
            int nbDims;
            int[3] filterDimA;
            cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerMatDesc,
                                                     3,
                                                     &dataType,
                                                     &format,
                                                     &nbDims,
                                                     filterDimA.ptr));

            initGPUData(linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2],
                        1.0f / cast(float) (filterDimA[0] * filterDimA[1] * filterDimA[2]));

            cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerMatDesc));

            cudnnFilterDescriptor_t linLayerBiasDesc;
            cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
            float *linLayerBias;

            cudnnErrCheck(cudnnGetRNNLinLayerBiasParams( cudnnHandle,
                                                         rnnDesc,
                                                         layer,
                                                         xDesc[0],
                                                         wDesc,
                                                         w.data,
                                                         linLayerID,
                                                         linLayerBiasDesc,
                                                         cast(void**) &linLayerBias));

            cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerBiasDesc,
                                                     3,
                                                     &dataType,
                                                     &format,
                                                     &nbDims,
                                                     filterDimA.ptr));

            initGPUData(linLayerBias, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.0f);

            cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
        }
    }

    // TODO:
    // *********************************************************************************************************
    // Dynamic persistent RNN plan (if using this algo)
    // *********************************************************************************************************
    // cudnnPersistentRNNPlan_t rnnPlan;
    // if (RNNAlgo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
    //     // Note: This step is expensive. Once completed the plan can be reused so long as the descriptor
    //     //       minibatch or datatype don't change.
    //     cudnnErrCheck(cudnnCreatePersistentRNNPlan(rnnDesc, miniBatch, CUDNN_DATA_FLOAT, &rnnPlan));
    //     // Tell calls using this descriptor which plan to use.
    //     cudnnErrCheck(cudnnSetPersistentRNNPlan(rnnDesc, rnnPlan));
    // }


    // *********************************************************************************************************
    // At this point all of the setup is done. We now need to pass through the RNN.
    // *********************************************************************************************************

    // Alternatively:
    DerelictCUDARuntime.load();
    // Now CUDA Runtime API functions can be called. Driver and Runtime API are exclusive.
    cudaErrCheck(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    float timeForward, timeBackward1, timeBackward2;
    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&stop));
    cudaErrCheck(cudaEventRecord(start));

    cudnnErrCheck(cudnnRNNForwardTraining(cudnnHandle,
                                          rnnDesc,
                                          seqLength,
                                          xDesc.ptr,
                                          cast(const void*) x.data,
                                          hxDesc,
                                          cast(const void*) hx.data,
                                          cxDesc,
                                          cast(const void*) cx.data,
                                          wDesc,
                                          w.data,
                                          yDesc.ptr,
                                          cast(void*) y.data,
                                          hyDesc,
                                          cast(void*) hy.data,
                                          cyDesc,
                                          cast(void*) cy.data,
                                          workspace.data,
                                          workSize,
                                          reserveSpace.data,
                                          reserveSize));

    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&timeForward, start, stop));
    cudaErrCheck(cudaEventRecord(start));

    cudnnErrCheck(cudnnRNNBackwardData(cudnnHandle,
                                       rnnDesc,
                                       seqLength,
                                       yDesc.ptr,
                                       cast(void*) y.data,
                                       dyDesc.ptr,
                                       cast(void*) dy.data,
                                       dhyDesc,
                                       cast(void*) dhy.data,
                                       dcyDesc,
                                       cast(void*) dcy.data,
                                       wDesc,
                                       w.data,
                                       hxDesc,
                                       cast(void*) hx.data,
                                       cxDesc,
                                       cast(void*) cx.data,
                                       dxDesc.ptr,
                                       cast(void*) dx.data,
                                       dhxDesc,
                                       cast(void*)  dhx.data,
                                       dcxDesc,
                                       cast(void*) dcx.data,
                                       workspace.data,
                                       workSize,
                                       reserveSpace.data,
                                       reserveSize ));

    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&timeBackward1, start, stop));
    cudaErrCheck(cudaEventRecord(start));
    // cudnnRNNBackwardWeights adds to the data in dw.
    // cudaErrCheck(cudaMemset(dw.data, 0, weightsSize));
    initGPUData(cast(float*) dw.data, cast(int) (workSize / float.sizeof), 0.0f);

    cudnnErrCheck(cudnnRNNBackwardWeights(cudnnHandle,
                                          rnnDesc,
                                          seqLength,
                                          xDesc.ptr,
                                          cast(const void*) x.data,
                                          hxDesc,
                                          cast(const void*) hx.data,
                                          yDesc.ptr,
                                          cast(const void*) y.data,
                                          workspace.data,
                                          workSize,
                                          dwDesc,
                                          cast(void*) dw.data,
                                          reserveSpace.data,
                                          reserveSize ));

    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&timeBackward2, start, stop));

    int numMats = 0;

    if (RNNMode == CUDNN_RNN_RELU || RNNMode == CUDNN_RNN_TANH) {
        numMats = 2;
    }
    else if (RNNMode == CUDNN_LSTM) {
        numMats = 8;
    }
    else if (RNNMode == CUDNN_GRU) {
        numMats = 6;
    }

    // Calculate FLOPS
    // writeln(timeForward, ", ", timeBackward1 + timeBackward2);
    writef("Forward: %3.0f GFLOPS\n", cast(float) numMats * 2UL * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeForward));
    writef("Backward: %3.0f GFLOPS, ", cast(float) numMats * 4UL * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * (timeBackward1 + timeBackward2)));
    writef("(%3.0f GFLOPS), ", cast(float) numMats * 2UL * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeBackward1));
    writef("(%3.0f GFLOPS)\n", cast(float) numMats * 2UL * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeBackward2));

    // Make double-sure everything is finished before we copy for result checking.
    cudaDeviceSynchronize();


    // *********************************************************************************************************
    // Print checksums.
    // *********************************************************************************************************
    if (true) {
        int biDirScale = (bidirectional ? 2 : 1);
        auto testOutputi = y.to_cpu();
        auto testOutputh = hy.to_cpu();
        auto testOutputc = cy.to_cpu();

        double checksumi = 0.0f;
        double checksumh = 0.0f;
        double checksumc = 0.0f;

        for (int m = 0; m < miniBatch; m++) {
            double localSumi = 0;
            double localSumh = 0;
            double localSumc = 0;

            for (int j = 0; j < seqLength; j++) {
                for (int i = 0; i < hiddenSize * biDirScale; i++) {
                    localSumi += testOutputi[j * miniBatch * hiddenSize * biDirScale + m * hiddenSize * biDirScale + i];
                }
            }
            for (int j = 0; j < numLayers * biDirScale; j++) {
                for (int i = 0; i < hiddenSize; i++) {
                    if (hy.data != null) localSumh += testOutputh[j * hiddenSize * miniBatch + m * hiddenSize + i];
                    if (cy.data != null) if (RNNMode == CUDNN_LSTM) localSumc += testOutputc[j * hiddenSize * miniBatch + m * hiddenSize + i];
                }
            }

            checksumi += localSumi;
            checksumh += localSumh;
            checksumc += localSumc;
        }

        writef("i checksum %E     ", checksumi);
        if (RNNMode == CUDNN_LSTM) { writef("c checksum %E     ", checksumc); }
        writef("h checksum %E\n", checksumh);
    }

    if (true) {
        auto testOutputdi = dx.to_cpu();
        auto testOutputdh = dhx.to_cpu();
        auto testOutputdc = dcx.to_cpu();
        int biDirScale = (bidirectional ? 2 : 1);

        float checksumdi = 0.0f;
        float checksumdh = 0.0f;
        float checksumdc = 0.0f;

        for (int m = 0; m < miniBatch; m++) {
            double localSumdi = 0;
            double localSumdh = 0;
            double localSumdc = 0;

            for (int j = 0; j < seqLength; j++) {
                for (int i = 0; i < inputSize; i++) {
                    localSumdi += testOutputdi[j * miniBatch * inputSize + m * inputSize + i];
                }
            }

            for (int j = 0; j < numLayers * biDirScale; j++) {
                for (int i = 0; i < hiddenSize; i++) {
                    localSumdh += testOutputdh[j * hiddenSize * miniBatch + m * hiddenSize + i];
                    if (RNNMode == CUDNN_LSTM) localSumdc += testOutputdc[j * hiddenSize * miniBatch + m * hiddenSize + i];
                }
            }

            checksumdi += localSumdi;
            checksumdh += localSumdh;
            checksumdc += localSumdc;

        }

        writef("di checksum %E    ", checksumdi);
        if (RNNMode == CUDNN_LSTM) { writef("dc checksum %E    ", checksumdc); }
        writef("dh checksum %E\n", checksumdh);
    }

    if (true) {
        auto testOutputdw = cast(float[]) dw.to_cpu();
        double checksumdw = 0.;
        for (int i = 0; i < weightsSize / float.sizeof; i++) {
            checksumdw += testOutputdw[i];
        }
        writef("dw checksum %E\n", checksumdw);
    }

    // TODO:
    // if (RNNAlgo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
    //     cudnnDestroyPersistentRNNPlan(rnnPlan);
    // }
}
