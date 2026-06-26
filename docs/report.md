# report of various comparison

Here is the comparison on energy consumption.

CPU: 34 W
FPGA: 4 W

CPU is Ryzen 9950@5.7 GHz. FPGA is Stratix 5 with our design programmed and working at 320 MHz. The CPU power setting is set to high performance both on Windows and UEFI of the mother board. The program to process the problem is prepared in pure C language code and optimized using the latest compiler, gcc version 14.2. The power consumption is estimated as the difference from idle state and the state during the computation. The speed comparison is shown in speed_comp.svg (Fig 1). From Fig. 1, therefore we can say at the condition number of qbits is 16, CPU is almost twice faster than FPGA. If we let the power efficiency of CPU 1, then, we have the following comparison on power efficiency between our FPGA design and CPU.

CPU: 1
FPGA: 34/4*0.5 =  4.25

So our FPGA is 4.25 time power efficient than the CPU.

However, the date of our FPGA is 2012, where the CPU is in 2025. So the level of technology inside these 2 processors are quit different and this is not fair comparison. The literature [1] provides the history of power efficiency of the most 2 major CPUs from Intel and AMD. According to [1], the power efficiency of the AMD's cpu improved 14 times from 2011 to 2026. So, 

CPU: 1
FPGA: 34/4*0.5 =  59.5

Because FPGA is much simpler than CPU when we talk about its circuit structure, it can be assumed the improvement in modern CPU can be applied directly into FPGAs as the minimum expected improvement. So we believe this comparison is not unrealistic assumption.

Here is the comparison on theoretical performance.

Ryzen 9950@5.7 theoretical speed: 8 * 2 * 5.7 = 91.2 GFlops.
FPGA 320 MHz speed: 0.32* 6 = 1.92 GFlops

Because Ryzen 9950@5.7 have AVX 512 instruction set and it can operate 1 FMA instruction per one clock, 8 multiplication and 8 addition on 64 bits floating point can be processed in one clock cycle. So 8 + 8 = 16 floating points can be processed in one clock cycle. Multiplying this value to the clock rate of 5.7 GHz, we obtain 91.2 GFlops as mentioned above. For FPGA, our design provide 4 multiplication and 2 addtion in mixer operation unit, 6 floating point can be processed in one clocl cycle. So we have 0.32 * 6 = 1.92 GFlops as mentioned. Inspite of the big difference in theoretical speed among CPU and FPGA, we have almost the same speed from Fig. 1. It can be said, our archtecture is far more efficient than the  program run on CPU.

Here are the flag ship FPGAs in 2026 and 2011. Agilex is the flagship in 2026, and Stratix 5 is our FPGA that was flagship in 2011. 

Agilex 9@2025: 12,300 DSPs
Stratix 5@2011: 340 DSPs

Even only seeing the number of DSPs, we have 34 times more DSPs in 2026 than in 2011. And expected FMax on the flagship chip in 2026 will be 1000 MHz (need citation), we can expect 3 times faster clock rate than the clock of 320 MHz in our current design. So, roughly we can expect 34 * 3 = 92 times imporvement in terms of the speed.

Here is the comparison on theoretical speed between flagship CPU and GPU in 2026. CPUs speed is estimated using all cores. Because 9950x have 16 cores, we have 91.2*16 = 2560 GFlops.

THperf 9950x: 2.56 TFlops
RTX 5090: 104 TFlops

GPU is 50 times faster than CPU. If we talk about 1 CPU core vs GPU, then we have 50*16 = 900 times faster speed in GPU than in 1 CPU core. It is much faster than 92 times CPU speed FPGAs suppose to have as mentioned. But we cannot expect this theoretical speed in GPUs because the parallel efficiency in GPU is much less than theoretical performance.


[1] Years of SPEC Power: An Analysis of x86 Energy Efficiency Trends, https://arxiv.org/pdf/2411.07062v116 