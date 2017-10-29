DerelictCuDNN
=============

2017.10.29 fork to support CuDNN v7 RNN

``` console
$ dub ./RNN_example.d --seqLength 20 --numLayers 2 --hiddenSize 128 --miniBatch 64 --mode 1
RNN example!
>>> Using CUDA Device [0]: Quadro M1000M
Forward:   0 GFLOPS
Backward: 324 GFLOPS, (227 GFLOPS), (562 GFLOPS)
i checksum 1.581185E+05     h checksum 1.581199E+04
di checksum 4.441525E+00    dh checksum 4.394408E+00
dw checksum 3.088443E+06

$ nvcc <CUDNN_ROOT>/RNN_example.cu -lcudnn

$ ./RNN_example.out 20 2 128 64 1
Forward: 170 GFLOPS
Backward: 331 GFLOPS, (224 GFLOPS), (633 GFLOPS)
i checksum 1.581185E+05     h checksum 1.581199E+04
di checksum 4.441525E+00    dh checksum 4.394408E+00
dw checksum 3.088443E+06

```


----

Dynamic bindings to cuDNN for the D programming language.

Usage
-----

Similar to other Derelict-based dynamic bindings, simply call ```DerelictCuDNN.load()``` and begin using cuDNN!

```
    import std.conv : to;
    import std.stdio : writeln;
    import derelict.cudnn;

    void main(string[] args)
    {
        DerelictCuDNN.load();

        writeln(cudnnGetErrorString(CUDNN_STATUS_SUCCESS).to!string);
    }
```
