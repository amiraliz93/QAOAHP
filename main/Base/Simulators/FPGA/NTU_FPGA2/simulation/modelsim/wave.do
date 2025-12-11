onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /ntu_smachine_tb/RST
add wave -noupdate /ntu_smachine_tb/CLK
add wave -noupdate /ntu_smachine_tb/rx_clk
add wave -noupdate /ntu_smachine_tb/rx_dv
add wave -noupdate /ntu_smachine_tb/rx_serial
add wave -noupdate -radix hexadecimal /ntu_smachine_tb/rx_data_out
add wave -noupdate /ntu_smachine_tb/tx_data_in
add wave -noupdate /ntu_smachine_tb/tx_dv
add wave -noupdate /ntu_smachine_tb/tx_active
add wave -noupdate /ntu_smachine_tb/tx_serial
add wave -noupdate /ntu_smachine_tb/o_Status
add wave -noupdate /ntu_smachine_tb/fp64
add wave -noupdate -radix float64 /ntu_smachine_tb/fp64rx
add wave -noupdate /ntu_smachine_tb/recState
add wave -noupdate /ntu_smachine_tb/recCount
add wave -noupdate /ntu_smachine_tb/i
add wave -noupdate /ntu_smachine_tb/ntuS/UART_CLKS_PER_BIT
add wave -noupdate /ntu_smachine_tb/ntuS/s_IDLE
add wave -noupdate /ntu_smachine_tb/ntuS/s_TXData
add wave -noupdate /ntu_smachine_tb/ntuS/s_Fetch
add wave -noupdate /ntu_smachine_tb/ntuS/s_FetchData
add wave -noupdate /ntu_smachine_tb/ntuS/s_WAIT
add wave -noupdate /ntu_smachine_tb/ntuS/s_Operation
add wave -noupdate /ntu_smachine_tb/ntuS/s_Operand
add wave -noupdate /ntu_smachine_tb/ntuS/OP_MOV_rA
add wave -noupdate /ntu_smachine_tb/ntuS/OP_MOV_rB
add wave -noupdate /ntu_smachine_tb/ntuS/OP_ADD_rB2rA
add wave -noupdate /ntu_smachine_tb/ntuS/OP_MUL_rB2rA
add wave -noupdate /ntu_smachine_tb/ntuS/OP_INC_rA
add wave -noupdate /ntu_smachine_tb/ntuS/OP_INC_rB
add wave -noupdate /ntu_smachine_tb/ntuS/OP_READ_rA
add wave -noupdate /ntu_smachine_tb/ntuS/OP_READ_rB
add wave -noupdate /ntu_smachine_tb/ntuS/OP_MOV_rA64
add wave -noupdate /ntu_smachine_tb/ntuS/OP_MOV_rA64rB
add wave -noupdate /ntu_smachine_tb/ntuS/OP_READ_rA64
add wave -noupdate /ntu_smachine_tb/ntuS/OP_ADD_rBrA64FP
add wave -noupdate /ntu_smachine_tb/ntuS/OP_MUL_rBrA64FP
add wave -noupdate /ntu_smachine_tb/ntuS/CLK
add wave -noupdate /ntu_smachine_tb/ntuS/RST
add wave -noupdate /ntu_smachine_tb/ntuS/i_Rx_Serial
add wave -noupdate /ntu_smachine_tb/ntuS/o_Tx_Serial
add wave -noupdate /ntu_smachine_tb/ntuS/o_Status
add wave -noupdate /ntu_smachine_tb/ntuS/tx_data_in
add wave -noupdate /ntu_smachine_tb/ntuS/tx_active
add wave -noupdate /ntu_smachine_tb/ntuS/rx_dv
add wave -noupdate /ntu_smachine_tb/ntuS/CP
add wave -noupdate -radix unsigned /ntu_smachine_tb/ntuS/state
add wave -noupdate -radix float32 /ntu_smachine_tb/ntuS/rA
add wave -noupdate /ntu_smachine_tb/ntuS/rB
add wave -noupdate -radix float64 /ntu_smachine_tb/ntuS/rA64
add wave -noupdate -radix float64 /ntu_smachine_tb/ntuS/rB64
add wave -noupdate /ntu_smachine_tb/ntuS/rT
add wave -noupdate /ntu_smachine_tb/ntuS/ope_state
add wave -noupdate /ntu_smachine_tb/ntuS/rfifo_data
add wave -noupdate /ntu_smachine_tb/ntuS/o_Rx_Byte
add wave -noupdate /ntu_smachine_tb/ntuS/rdreq
add wave -noupdate /ntu_smachine_tb/ntuS/empty
add wave -noupdate /ntu_smachine_tb/ntuS/full
add wave -noupdate /ntu_smachine_tb/ntuS/tx_empty
add wave -noupdate /ntu_smachine_tb/ntuS/tx_full
add wave -noupdate /ntu_smachine_tb/ntuS/tx_fifo_write
add wave -noupdate /ntu_smachine_tb/ntuS/tx_dv
add wave -noupdate /ntu_smachine_tb/ntuS/tx_fifo_data
add wave -noupdate /ntu_smachine_tb/ntuS/c_wait
add wave -noupdate /ntu_smachine_tb/ntuS/r8Pos
add wave -noupdate /ntu_smachine_tb/ntuS/res_addFP64
add wave -noupdate /ntu_smachine_tb/ntuS/res_mulFP64
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {2315877022 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 304
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 0
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ns
update
WaveRestoreZoom {2078092063 ps} {5467104875 ps}
