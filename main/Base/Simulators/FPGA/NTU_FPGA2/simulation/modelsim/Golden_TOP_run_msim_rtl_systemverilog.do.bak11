transcript on
if ![file isdirectory Golden_TOP_iputf_libs] {
	file mkdir Golden_TOP_iputf_libs
}

if {[file exists rtl_work]} {
	vdel -lib rtl_work -all
}
vlib rtl_work
vmap work rtl_work

###### Libraries for IPUTF cores 
###### End libraries for IPUTF cores 
###### MIF file copy and HDL compilation commands for IPUTF cores 


vlog "C:/home/verilog_sources/NTU_FPGA2/pll2_sim/pll2.vo"        
vlog "C:/home/verilog_sources/NTU_FPGA2/mulFPF64_sim/mulFPF64.vo"
vlog "C:/home/verilog_sources/NTU_FPGA2/addFPF64_sim/addFPF64.vo"

vlog -sv -work work +incdir+C:/home/verilog_sources/NTU_FPGA2 {C:/home/verilog_sources/NTU_FPGA2/transmitter.v}
vlog -sv -work work +incdir+C:/home/verilog_sources/NTU_FPGA2 {C:/home/verilog_sources/NTU_FPGA2/receiver.v}
vlog -sv -work work +incdir+C:/home/verilog_sources/NTU_FPGA2 {C:/home/verilog_sources/NTU_FPGA2/fifo1.v}
vlog -sv -work work +incdir+C:/home/verilog_sources/NTU_FPGA2 {C:/home/verilog_sources/NTU_FPGA2/addfix8.v}
vlog -sv -work work +incdir+C:/home/verilog_sources/NTU_FPGA2 {C:/home/verilog_sources/NTU_FPGA2/mulfix8.v}
vlog -sv -work work +incdir+C:/home/verilog_sources/NTU_FPGA2 {C:/home/verilog_sources/NTU_FPGA2/ram.v}
vlog -sv -work work +incdir+C:/home/verilog_sources/NTU_FPGA2/db {C:/home/verilog_sources/NTU_FPGA2/db/mult_c0p.v}
vlog -sv -work work +incdir+C:/home/verilog_sources/NTU_FPGA2 {C:/home/verilog_sources/NTU_FPGA2/qaoa_system.sv}
vlog -sv -work work +incdir+C:/home/verilog_sources/NTU_FPGA2 {C:/home/verilog_sources/NTU_FPGA2/ntu_smachine.sv}

vlog -sv -work work +incdir+C:/home/verilog_sources/NTU_FPGA2 {C:/home/verilog_sources/NTU_FPGA2/qaoa_system_tb.sv}

vsim -t 1ps -L altera_ver -L lpm_ver -L sgate_ver -L altera_mf_ver -L altera_lnsim_ver -L stratixv_ver -L stratixv_hssi_ver -L stratixv_pcie_hip_ver -L rtl_work -L work -voptargs="+acc"  qaoa_system_tb

add wave *
view structure
view signals
run -all
