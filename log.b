# nThread 1 nGpus 8 minBytes 4 maxBytes 2147483648 step: 2(factor) warmup iters: 50 iters: 100 agg iters: 1 validation: 0 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 422771 on nf5688m7-1 device  0 [0x0f] NVIDIA H800
#  Rank  1 Group  0 Pid 422771 on nf5688m7-1 device  1 [0x34] NVIDIA H800
#  Rank  2 Group  0 Pid 422771 on nf5688m7-1 device  2 [0x48] NVIDIA H800
#  Rank  3 Group  0 Pid 422771 on nf5688m7-1 device  3 [0x5a] NVIDIA H800
#  Rank  4 Group  0 Pid 422771 on nf5688m7-1 device  4 [0x87] NVIDIA H800
#  Rank  5 Group  0 Pid 422771 on nf5688m7-1 device  5 [0xae] NVIDIA H800
#  Rank  6 Group  0 Pid 422771 on nf5688m7-1 device  6 [0xc2] NVIDIA H800
#  Rank  7 Group  0 Pid 422771 on nf5688m7-1 device  7 [0xd7] NVIDIA H800
nf5688m7-1:422771:422771 [0] NCCL INFO Bootstrap : Using enx785ee8a015a1:10.59.6.30<0>
nf5688m7-1:422771:422771 [0] NCCL INFO NET/Plugin : Plugin load (libnccl-net.so) returned 2 : libnccl-net.so: cannot open shared object file: No such file or directory
nf5688m7-1:422771:422771 [0] NCCL INFO NET/Plugin : No plugin found, using internal implementation
nf5688m7-1:422771:422771 [7] NCCL INFO cudaDriverVersion 12020
NCCL version 2.18.3+cuda12.2
nf5688m7-1:422771:422791 [5] NCCL INFO NET/IB : No device found.
nf5688m7-1:422771:422791 [5] NCCL INFO NET/Socket : Using [0]enx785ee8a015a1:10.59.6.30<0> [1]veth6f0b7dd:fe80::90a4:35ff:fe22:2c98%veth6f0b7dd<0>
nf5688m7-1:422771:422791 [5] NCCL INFO Using network Socket
nf5688m7-1:422771:422792 [6] NCCL INFO Using network Socket
nf5688m7-1:422771:422787 [1] NCCL INFO Using network Socket
nf5688m7-1:422771:422793 [7] NCCL INFO Using network Socket
nf5688m7-1:422771:422789 [3] NCCL INFO Using network Socket
nf5688m7-1:422771:422786 [0] NCCL INFO Using network Socket
nf5688m7-1:422771:422788 [2] NCCL INFO Using network Socket
nf5688m7-1:422771:422790 [4] NCCL INFO Using network Socket
nf5688m7-1:422771:422788 [2] NCCL INFO comm 0x5598478aaae0 rank 2 nranks 8 cudaDev 2 nvmlDev 2 busId 48000 commId 0xefdd95e4ad4062d9 - Init START
nf5688m7-1:422771:422790 [4] NCCL INFO comm 0x5598478b64a0 rank 4 nranks 8 cudaDev 4 nvmlDev 4 busId 87000 commId 0xefdd95e4ad4062d9 - Init START
nf5688m7-1:422771:422787 [1] NCCL INFO comm 0x5598478a4d90 rank 1 nranks 8 cudaDev 1 nvmlDev 1 busId 34000 commId 0xefdd95e4ad4062d9 - Init START
nf5688m7-1:422771:422786 [0] NCCL INFO comm 0x55984789bae0 rank 0 nranks 8 cudaDev 0 nvmlDev 0 busId f000 commId 0xefdd95e4ad4062d9 - Init START
nf5688m7-1:422771:422793 [7] NCCL INFO comm 0x5598478c7b40 rank 7 nranks 8 cudaDev 7 nvmlDev 7 busId d7000 commId 0xefdd95e4ad4062d9 - Init START
nf5688m7-1:422771:422791 [5] NCCL INFO comm 0x5598478bc180 rank 5 nranks 8 cudaDev 5 nvmlDev 5 busId ae000 commId 0xefdd95e4ad4062d9 - Init START
nf5688m7-1:422771:422792 [6] NCCL INFO comm 0x5598478c1e60 rank 6 nranks 8 cudaDev 6 nvmlDev 6 busId c2000 commId 0xefdd95e4ad4062d9 - Init START
nf5688m7-1:422771:422789 [3] NCCL INFO comm 0x5598478b07c0 rank 3 nranks 8 cudaDev 3 nvmlDev 3 busId 5a000 commId 0xefdd95e4ad4062d9 - Init START
nf5688m7-1:422771:422790 [4] NCCL INFO Setting affinity for GPU 4 to 0f,ffffff00,00000000,00000000,000fffff,ff000000,00000000
nf5688m7-1:422771:422790 [4] NCCL INFO NVLS multicast support is available on dev 4
nf5688m7-1:422771:422787 [1] NCCL INFO Setting affinity for GPU 1 to ff,fffff000,00000000,00000000,00ffffff,f0000000
nf5688m7-1:422771:422787 [1] NCCL INFO NVLS multicast support is available on dev 1
nf5688m7-1:422771:422786 [0] NCCL INFO Setting affinity for GPU 0 to 0fff,ffff0000,00000000,00000000,0fffffff
nf5688m7-1:422771:422786 [0] NCCL INFO NVLS multicast support is available on dev 0
nf5688m7-1:422771:422791 [5] NCCL INFO Setting affinity for GPU 5 to fffffff0,00000000,00000000,0000ffff,fff00000,00000000,00000000
nf5688m7-1:422771:422791 [5] NCCL INFO NVLS multicast support is available on dev 5
nf5688m7-1:422771:422793 [7] NCCL INFO Setting affinity for GPU 7 to 0f,ffffff00,00000000,00000000,000fffff,ff000000,00000000
nf5688m7-1:422771:422793 [7] NCCL INFO NVLS multicast support is available on dev 7
nf5688m7-1:422771:422789 [3] NCCL INFO Setting affinity for GPU 3 to 0fff,ffff0000,00000000,00000000,0fffffff
nf5688m7-1:422771:422789 [3] NCCL INFO NVLS multicast support is available on dev 3
nf5688m7-1:422771:422792 [6] NCCL INFO Setting affinity for GPU 6 to fffffff0,00000000,00000000,0000ffff,fff00000,00000000,00000000
nf5688m7-1:422771:422792 [6] NCCL INFO NVLS multicast support is available on dev 6
nf5688m7-1:422771:422788 [2] NCCL INFO Setting affinity for GPU 2 to ff,fffff000,00000000,00000000,00ffffff,f0000000
nf5688m7-1:422771:422788 [2] NCCL INFO NVLS multicast support is available on dev 2
nf5688m7-1:422771:422788 [2] NCCL INFO Trees [0] 4/-1/-1->2->1 [1] 4/-1/-1->2->1 [2] 4/-1/-1->2->1 [3] 4/-1/-1->2->1 [4] 4/-1/-1->2->1 [5] 4/-1/-1->2->1 [6] 4/-1/-1->2->1 [7] 4/-1/-1->2->1 [8] 4/-1/-1->2->1 [9] 4/-1/-1->2->1 [10] 4/-1/-1->2->1 [11] 4/-1/-1->2->1 [12] 4/-1/-1->2->1 [13] 4/-1/-1->2->1 [14] 4/-1/-1->2->1 [15] 4/-1/-1->2->1
nf5688m7-1:422771:422788 [2] NCCL INFO P2P Chunksize set to 524288
nf5688m7-1:422771:422789 [3] NCCL INFO Trees [0] 1/-1/-1->3->0 [1] 1/-1/-1->3->0 [2] 1/-1/-1->3->0 [3] 1/-1/-1->3->0 [4] 1/-1/-1->3->0 [5] 1/-1/-1->3->0 [6] 1/-1/-1->3->0 [7] 1/-1/-1->3->0 [8] 1/-1/-1->3->0 [9] 1/-1/-1->3->0 [10] 1/-1/-1->3->0 [11] 1/-1/-1->3->0 [12] 1/-1/-1->3->0 [13] 1/-1/-1->3->0 [14] 1/-1/-1->3->0 [15] 1/-1/-1->3->0
nf5688m7-1:422771:422789 [3] NCCL INFO P2P Chunksize set to 524288
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 00/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 01/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 02/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 03/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 04/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 05/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 06/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 07/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 08/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 09/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 10/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 11/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 12/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 13/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 14/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 15/16 :    0   3   1   2   4   7   5   6
nf5688m7-1:422771:422786 [0] NCCL INFO Trees [0] 3/-1/-1->0->-1 [1] 3/-1/-1->0->-1 [2] 3/-1/-1->0->-1 [3] 3/-1/-1->0->-1 [4] 3/-1/-1->0->-1 [5] 3/-1/-1->0->-1 [6] 3/-1/-1->0->-1 [7] 3/-1/-1->0->-1 [8] 3/-1/-1->0->-1 [9] 3/-1/-1->0->-1 [10] 3/-1/-1->0->-1 [11] 3/-1/-1->0->-1 [12] 3/-1/-1->0->-1 [13] 3/-1/-1->0->-1 [14] 3/-1/-1->0->-1 [15] 3/-1/-1->0->-1
nf5688m7-1:422771:422786 [0] NCCL INFO P2P Chunksize set to 524288
nf5688m7-1:422771:422792 [6] NCCL INFO Trees [0] -1/-1/-1->6->5 [1] -1/-1/-1->6->5 [2] -1/-1/-1->6->5 [3] -1/-1/-1->6->5 [4] -1/-1/-1->6->5 [5] -1/-1/-1->6->5 [6] -1/-1/-1->6->5 [7] -1/-1/-1->6->5 [8] -1/-1/-1->6->5 [9] -1/-1/-1->6->5 [10] -1/-1/-1->6->5 [11] -1/-1/-1->6->5 [12] -1/-1/-1->6->5 [13] -1/-1/-1->6->5 [14] -1/-1/-1->6->5 [15] -1/-1/-1->6->5
nf5688m7-1:422771:422790 [4] NCCL INFO Trees [0] 7/-1/-1->4->2 [1] 7/-1/-1->4->2 [2] 7/-1/-1->4->2 [3] 7/-1/-1->4->2 [4] 7/-1/-1->4->2 [5] 7/-1/-1->4->2 [6] 7/-1/-1->4->2 [7] 7/-1/-1->4->2 [8] 7/-1/-1->4->2 [9] 7/-1/-1->4->2 [10] 7/-1/-1->4->2 [11] 7/-1/-1->4->2 [12] 7/-1/-1->4->2 [13] 7/-1/-1->4->2 [14] 7/-1/-1->4->2 [15] 7/-1/-1->4->2
nf5688m7-1:422771:422792 [6] NCCL INFO P2P Chunksize set to 524288
nf5688m7-1:422771:422793 [7] NCCL INFO Trees [0] 5/-1/-1->7->4 [1] 5/-1/-1->7->4 [2] 5/-1/-1->7->4 [3] 5/-1/-1->7->4 [4] 5/-1/-1->7->4 [5] 5/-1/-1->7->4 [6] 5/-1/-1->7->4 [7] 5/-1/-1->7->4 [8] 5/-1/-1->7->4 [9] 5/-1/-1->7->4 [10] 5/-1/-1->7->4 [11] 5/-1/-1->7->4 [12] 5/-1/-1->7->4 [13] 5/-1/-1->7->4 [14] 5/-1/-1->7->4 [15] 5/-1/-1->7->4
nf5688m7-1:422771:422791 [5] NCCL INFO Trees [0] 6/-1/-1->5->7 [1] 6/-1/-1->5->7 [2] 6/-1/-1->5->7 [3] 6/-1/-1->5->7 [4] 6/-1/-1->5->7 [5] 6/-1/-1->5->7 [6] 6/-1/-1->5->7 [7] 6/-1/-1->5->7 [8] 6/-1/-1->5->7 [9] 6/-1/-1->5->7 [10] 6/-1/-1->5->7 [11] 6/-1/-1->5->7 [12] 6/-1/-1->5->7 [13] 6/-1/-1->5->7 [14] 6/-1/-1->5->7 [15] 6/-1/-1->5->7
nf5688m7-1:422771:422790 [4] NCCL INFO P2P Chunksize set to 524288
nf5688m7-1:422771:422791 [5] NCCL INFO P2P Chunksize set to 524288
nf5688m7-1:422771:422793 [7] NCCL INFO P2P Chunksize set to 524288
nf5688m7-1:422771:422787 [1] NCCL INFO Trees [0] 2/-1/-1->1->3 [1] 2/-1/-1->1->3 [2] 2/-1/-1->1->3 [3] 2/-1/-1->1->3 [4] 2/-1/-1->1->3 [5] 2/-1/-1->1->3 [6] 2/-1/-1->1->3 [7] 2/-1/-1->1->3 [8] 2/-1/-1->1->3 [9] 2/-1/-1->1->3 [10] 2/-1/-1->1->3 [11] 2/-1/-1->1->3 [12] 2/-1/-1->1->3 [13] 2/-1/-1->1->3 [14] 2/-1/-1->1->3 [15] 2/-1/-1->1->3
nf5688m7-1:422771:422787 [1] NCCL INFO P2P Chunksize set to 524288
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 00/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 00/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 01/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 01/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 02/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 02/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 03/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 03/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 04/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 04/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 05/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 05/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 06/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 06/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 07/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 07/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 08/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 08/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 09/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 09/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 10/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 10/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 11/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 11/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 12/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 12/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 13/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 13/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 14/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 14/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 15/0 : 1[1] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 15/0 : 5[5] -> 6[6] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 00/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 01/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 02/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 03/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 04/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 05/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 06/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 07/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 08/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 09/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 10/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 11/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 12/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 13/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 14/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 15/0 : 6[6] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 00/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 00/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 01/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 01/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Connected all rings
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 00/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 02/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 02/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 01/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 03/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 03/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 02/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 04/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 04/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 03/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 05/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 05/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 04/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 06/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 06/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 05/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 07/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 07/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 06/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 08/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 08/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 07/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 09/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 09/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 08/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 09/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 10/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 10/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 10/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 11/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 11/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 11/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 12/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 12/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 12/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 13/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 13/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 13/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 14/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 14/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 14/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Channel 15/0 : 0[0] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 15/0 : 2[2] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422792 [6] NCCL INFO Channel 15/0 : 6[6] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 00/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 00/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 01/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 01/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 02/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 02/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 03/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 03/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Connected all rings
nf5688m7-1:422771:422788 [2] NCCL INFO Connected all rings
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 04/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 04/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 05/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 05/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 06/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 06/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 07/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 07/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 08/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 08/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 09/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 09/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 10/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 10/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 11/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 11/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 12/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 12/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 13/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 13/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 14/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 14/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 15/0 : 3[3] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 15/0 : 4[4] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 00/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 01/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 02/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 03/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 04/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Connected all rings
nf5688m7-1:422771:422787 [1] NCCL INFO Connected all rings
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 00/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Connected all rings
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 05/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 01/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 06/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 02/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 07/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 03/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 08/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 04/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 09/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 05/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 10/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 06/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 11/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 07/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 12/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 08/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 13/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 09/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 14/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 10/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 15/0 : 7[7] -> 5[5] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 11/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 12/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 13/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 14/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422787 [1] NCCL INFO Channel 15/0 : 1[1] -> 3[3] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 00/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Connected all rings
nf5688m7-1:422771:422791 [5] NCCL INFO Connected all rings
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 00/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 01/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 01/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 02/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 02/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 03/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 03/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 04/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 04/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 05/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 05/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 06/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 06/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 07/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 07/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 08/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 08/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 09/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 09/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 10/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 10/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 11/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 11/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 12/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 12/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 13/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 13/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 14/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 14/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Channel 15/0 : 3[3] -> 0[0] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Channel 15/0 : 5[5] -> 7[7] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 00/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422786 [0] NCCL INFO Connected all trees
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 01/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO Connected all trees
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 02/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 03/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 04/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 05/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422789 [3] NCCL INFO NVLS comm 0x5598478b07c0 headRank 1 nHeads 8 buffSize 4194304 memSize 2097152 nvlsPerRankSize 201326592 nvlsTotalSize 1610612736
nf5688m7-1:422771:422786 [0] NCCL INFO NVLS comm 0x55984789bae0 headRank 0 nHeads 8 buffSize 4194304 memSize 2097152 nvlsPerRankSize 201326592 nvlsTotalSize 1610612736
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 06/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 07/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 08/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 09/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 10/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 11/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 12/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 13/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 14/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Channel 15/0 : 7[7] -> 4[4] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 00/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 01/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO Connected all trees
nf5688m7-1:422771:422792 [6] NCCL INFO Connected all trees
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 02/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 03/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422791 [5] NCCL INFO Connected all trees
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 04/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 05/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 06/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422793 [7] NCCL INFO NVLS comm 0x5598478c7b40 headRank 5 nHeads 8 buffSize 4194304 memSize 2097152 nvlsPerRankSize 201326592 nvlsTotalSize 1610612736
nf5688m7-1:422771:422792 [6] NCCL INFO NVLS comm 0x5598478c1e60 headRank 7 nHeads 8 buffSize 4194304 memSize 2097152 nvlsPerRankSize 201326592 nvlsTotalSize 1610612736
nf5688m7-1:422771:422791 [5] NCCL INFO NVLS comm 0x5598478bc180 headRank 6 nHeads 8 buffSize 4194304 memSize 2097152 nvlsPerRankSize 201326592 nvlsTotalSize 1610612736
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 07/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 08/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 09/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 10/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 11/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 12/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 13/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 14/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Channel 15/0 : 4[4] -> 2[2] via P2P/direct pointer
nf5688m7-1:422771:422790 [4] NCCL INFO Connected all trees
nf5688m7-1:422771:422790 [4] NCCL INFO NVLS comm 0x5598478b64a0 headRank 4 nHeads 8 buffSize 4194304 memSize 2097152 nvlsPerRankSize 201326592 nvlsTotalSize 1610612736
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 00/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 01/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 02/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 03/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 04/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 05/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 06/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 07/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 08/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 09/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 10/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 11/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 12/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 13/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 14/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Channel 15/0 : 2[2] -> 1[1] via P2P/direct pointer
nf5688m7-1:422771:422788 [2] NCCL INFO Connected all trees
nf5688m7-1:422771:422787 [1] NCCL INFO Connected all trees
nf5688m7-1:422771:422788 [2] NCCL INFO NVLS comm 0x5598478aaae0 headRank 3 nHeads 8 buffSize 4194304 memSize 2097152 nvlsPerRankSize 201326592 nvlsTotalSize 1610612736
nf5688m7-1:422771:422787 [1] NCCL INFO NVLS comm 0x5598478a4d90 headRank 2 nHeads 8 buffSize 4194304 memSize 2097152 nvlsPerRankSize 201326592 nvlsTotalSize 1610612736
nf5688m7-1:422771:422792 [6] NCCL INFO Connected NVLS tree
nf5688m7-1:422771:422792 [6] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 512 | 512
nf5688m7-1:422771:422792 [6] NCCL INFO 16 coll channels, 16 nvls channels, 16 p2p channels, 16 p2p channels per peer
nf5688m7-1:422771:422789 [3] NCCL INFO Connected NVLS tree
nf5688m7-1:422771:422789 [3] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 512 | 512
nf5688m7-1:422771:422789 [3] NCCL INFO 16 coll channels, 16 nvls channels, 16 p2p channels, 16 p2p channels per peer
nf5688m7-1:422771:422786 [0] NCCL INFO Connected NVLS tree
nf5688m7-1:422771:422786 [0] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 512 | 512
nf5688m7-1:422771:422786 [0] NCCL INFO 16 coll channels, 16 nvls channels, 16 p2p channels, 16 p2p channels per peer
nf5688m7-1:422771:422790 [4] NCCL INFO Connected NVLS tree
nf5688m7-1:422771:422790 [4] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 512 | 512
nf5688m7-1:422771:422790 [4] NCCL INFO 16 coll channels, 16 nvls channels, 16 p2p channels, 16 p2p channels per peer
nf5688m7-1:422771:422788 [2] NCCL INFO Connected NVLS tree
nf5688m7-1:422771:422788 [2] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 512 | 512
nf5688m7-1:422771:422788 [2] NCCL INFO 16 coll channels, 16 nvls channels, 16 p2p channels, 16 p2p channels per peer
nf5688m7-1:422771:422787 [1] NCCL INFO Connected NVLS tree
nf5688m7-1:422771:422787 [1] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 512 | 512
nf5688m7-1:422771:422787 [1] NCCL INFO 16 coll channels, 16 nvls channels, 16 p2p channels, 16 p2p channels per peer
nf5688m7-1:422771:422793 [7] NCCL INFO Connected NVLS tree
nf5688m7-1:422771:422793 [7] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 512 | 512
nf5688m7-1:422771:422793 [7] NCCL INFO 16 coll channels, 16 nvls channels, 16 p2p channels, 16 p2p channels per peer
nf5688m7-1:422771:422791 [5] NCCL INFO Connected NVLS tree
nf5688m7-1:422771:422791 [5] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 512 | 512
nf5688m7-1:422771:422791 [5] NCCL INFO 16 coll channels, 16 nvls channels, 16 p2p channels, 16 p2p channels per peer
nf5688m7-1:422771:422789 [3] NCCL INFO comm 0x5598478b07c0 rank 3 nranks 8 cudaDev 3 nvmlDev 3 busId 5a000 commId 0xefdd95e4ad4062d9 - Init COMPLETE
nf5688m7-1:422771:422793 [7] NCCL INFO comm 0x5598478c7b40 rank 7 nranks 8 cudaDev 7 nvmlDev 7 busId d7000 commId 0xefdd95e4ad4062d9 - Init COMPLETE
nf5688m7-1:422771:422786 [0] NCCL INFO comm 0x55984789bae0 rank 0 nranks 8 cudaDev 0 nvmlDev 0 busId f000 commId 0xefdd95e4ad4062d9 - Init COMPLETE
nf5688m7-1:422771:422790 [4] NCCL INFO comm 0x5598478b64a0 rank 4 nranks 8 cudaDev 4 nvmlDev 4 busId 87000 commId 0xefdd95e4ad4062d9 - Init COMPLETE
nf5688m7-1:422771:422788 [2] NCCL INFO comm 0x5598478aaae0 rank 2 nranks 8 cudaDev 2 nvmlDev 2 busId 48000 commId 0xefdd95e4ad4062d9 - Init COMPLETE
nf5688m7-1:422771:422792 [6] NCCL INFO comm 0x5598478c1e60 rank 6 nranks 8 cudaDev 6 nvmlDev 6 busId c2000 commId 0xefdd95e4ad4062d9 - Init COMPLETE
nf5688m7-1:422771:422787 [1] NCCL INFO comm 0x5598478a4d90 rank 1 nranks 8 cudaDev 1 nvmlDev 1 busId 34000 commId 0xefdd95e4ad4062d9 - Init COMPLETE
nf5688m7-1:422771:422791 [5] NCCL INFO comm 0x5598478bc180 rank 5 nranks 8 cudaDev 5 nvmlDev 5 busId ae000 commId 0xefdd95e4ad4062d9 - Init COMPLETE
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           4             1     float    none       0    27.06    0.00    0.00    N/A    27.03    0.00    0.00    N/A
           8             2     float    none       0    27.27    0.00    0.00    N/A    26.97    0.00    0.00    N/A
          16             4     float    none       0    27.14    0.00    0.00    N/A    27.11    0.00    0.00    N/A
          32             8     float    none       0    27.03    0.00    0.00    N/A    27.08    0.00    0.00    N/A
          64            16     float    none       0    27.00    0.00    0.00    N/A    25.95    0.00    0.00    N/A
         128            32     float    none       0    25.63    0.00    0.00    N/A    25.64    0.00    0.00    N/A
         256            64     float    none       0    25.57    0.01    0.01    N/A    25.72    0.01    0.01    N/A
         512           128     float    none       0    25.51    0.02    0.02    N/A    25.68    0.02    0.02    N/A
        1024           256     float    none       0    25.57    0.04    0.04    N/A    25.50    0.04    0.04    N/A
        2048           512     float    none       0    25.61    0.08    0.08    N/A    25.65    0.08    0.08    N/A
        4096          1024     float    none       0    25.72    0.16    0.16    N/A    25.62    0.16    0.16    N/A
        8192          2048     float    none       0    25.74    0.32    0.32    N/A    25.71    0.32    0.32    N/A
       16384          4096     float    none       0    25.62    0.64    0.64    N/A    25.66    0.64    0.64    N/A
       32768          8192     float    none       0    25.45    1.29    1.29    N/A    25.34    1.29    1.29    N/A
       65536         16384     float    none       0    26.72    2.45    2.45    N/A    26.93    2.43    2.43    N/A
      131072         32768     float    none       0    29.56    4.43    4.43    N/A    29.52    4.44    4.44    N/A
      262144         65536     float    none       0    33.37    7.86    7.86    N/A    33.30    7.87    7.87    N/A
      524288        131072     float    none       0    43.25   12.12   12.12    N/A    42.88   12.23   12.23    N/A
     1048576        262144     float    none       0    43.40   24.16   24.16    N/A    43.60   24.05   24.05    N/A
     2097152        524288     float    none       0    43.71   47.98   47.98    N/A    44.38   47.25   47.25    N/A
     4194304       1048576     float    none       0    45.13   92.94   92.94    N/A    45.15   92.89   92.89    N/A
     8388608       2097152     float    none       0    69.51  120.67  120.67    N/A    69.45  120.79  120.79    N/A
    16777216       4194304     float    none       0    123.0  136.42  136.42    N/A    122.8  136.58  136.58    N/A
    33554432       8388608     float    none       0    226.2  148.36  148.36    N/A    226.0  148.46  148.46    N/A
    67108864      16777216     float    none       0    430.6  155.87  155.87    N/A    430.4  155.92  155.92    N/A
   134217728      33554432     float    none       0    837.8  160.21  160.21    N/A    837.9  160.19  160.19    N/A
   268435456      67108864     float    none       0   1649.9  162.70  162.70    N/A   1650.4  162.65  162.65    N/A
   536870912     134217728     float    none       0   3274.5  163.96  163.96    N/A   3274.6  163.95  163.95    N/A
  1073741824     268435456     float    none       0   6520.1  164.68  164.68    N/A   6520.5  164.67  164.67    N/A
  2147483648     536870912     float    none       0    13018  164.97  164.97    N/A    13014  165.02  165.02    N/A
nf5688m7-1:422771:422771 [0] NCCL INFO comm 0x55984789bae0 rank 0 nranks 8 cudaDev 0 busId f000 - Destroy COMPLETE
nf5688m7-1:422771:422771 [0] NCCL INFO comm 0x5598478a4d90 rank 1 nranks 8 cudaDev 1 busId 34000 - Destroy COMPLETE
nf5688m7-1:422771:422771 [0] NCCL INFO comm 0x5598478aaae0 rank 2 nranks 8 cudaDev 2 busId 48000 - Destroy COMPLETE
nf5688m7-1:422771:422771 [0] NCCL INFO comm 0x5598478b07c0 rank 3 nranks 8 cudaDev 3 busId 5a000 - Destroy COMPLETE
nf5688m7-1:422771:422771 [0] NCCL INFO comm 0x5598478b64a0 rank 4 nranks 8 cudaDev 4 busId 87000 - Destroy COMPLETE
nf5688m7-1:422771:422771 [0] NCCL INFO comm 0x5598478bc180 rank 5 nranks 8 cudaDev 5 busId ae000 - Destroy COMPLETE
nf5688m7-1:422771:422771 [0] NCCL INFO comm 0x5598478c1e60 rank 6 nranks 8 cudaDev 6 busId c2000 - Destroy COMPLETE
nf5688m7-1:422771:422771 [0] NCCL INFO comm 0x5598478c7b40 rank 7 nranks 8 cudaDev 7 busId d7000 - Destroy COMPLETE
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 52.4049 
#

