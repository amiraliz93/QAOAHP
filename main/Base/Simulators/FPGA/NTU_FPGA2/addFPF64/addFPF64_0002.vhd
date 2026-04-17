-- ------------------------------------------------------------------------- 
-- High Level Design Compiler for Intel(R) FPGAs Version 25.1std (Release Build #1129)
-- Quartus Prime development tool and MATLAB/Simulink Interface
-- 
-- Legal Notice: Copyright 2025 Intel Corporation.  All rights reserved.
-- Your use of  Intel Corporation's design tools,  logic functions and other
-- software and  tools, and its AMPP partner logic functions, and any output
-- files any  of the foregoing (including  device programming  or simulation
-- files), and  any associated  documentation  or information  are expressly
-- subject  to the terms and  conditions of the  Intel FPGA Software License
-- Agreement, Intel MegaCore Function License Agreement, or other applicable
-- license agreement,  including,  without limitation,  that your use is for
-- the  sole  purpose of  programming  logic devices  manufactured by  Intel
-- and  sold by Intel  or its authorized  distributors. Please refer  to the
-- applicable agreement for further details.
-- ---------------------------------------------------------------------------

-- VHDL created from addFPF64_0002
-- VHDL created on Fri Apr 17 23:45:55 2026


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.NUMERIC_STD.all;
use IEEE.MATH_REAL.all;
use std.TextIO.all;
use work.dspba_library_package.all;

LIBRARY altera_mf;
USE altera_mf.altera_mf_components.all;
LIBRARY altera_lnsim;
USE altera_lnsim.altera_lnsim_components.altera_syncram;
LIBRARY lpm;
USE lpm.lpm_components.all;

entity addFPF64_0002 is
    port (
        a : in std_logic_vector(63 downto 0);  -- float64_m52
        b : in std_logic_vector(63 downto 0);  -- float64_m52
        q : out std_logic_vector(63 downto 0);  -- float64_m52
        clk : in std_logic;
        areset : in std_logic
    );
end addFPF64_0002;

architecture normal of addFPF64_0002 is

    attribute altera_attribute : string;
    attribute altera_attribute of normal : architecture is "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name PHYSICAL_SYNTHESIS_REGISTER_DUPLICATION ON; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007";
    
    signal GND_q : STD_LOGIC_VECTOR (0 downto 0);
    signal VCC_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracX_uid6_fpAddTest_b : STD_LOGIC_VECTOR (62 downto 0);
    signal expFracY_uid7_fpAddTest_b : STD_LOGIC_VECTOR (62 downto 0);
    signal sigY_uid9_fpAddTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal fracY_uid10_fpAddTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal expY_uid11_fpAddTest_b : STD_LOGIC_VECTOR (10 downto 0);
    signal ypn_uid12_fpAddTest_q : STD_LOGIC_VECTOR (63 downto 0);
    signal bSig_uid17_fpAddTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal bSig_uid17_fpAddTest_q : STD_LOGIC_VECTOR (63 downto 0);
    signal cstAllOWE_uid18_fpAddTest_q : STD_LOGIC_VECTOR (10 downto 0);
    signal cstZeroWF_uid19_fpAddTest_q : STD_LOGIC_VECTOR (51 downto 0);
    signal cstAllZWE_uid20_fpAddTest_q : STD_LOGIC_VECTOR (10 downto 0);
    signal exp_aSig_uid21_fpAddTest_in : STD_LOGIC_VECTOR (62 downto 0);
    signal exp_aSig_uid21_fpAddTest_b : STD_LOGIC_VECTOR (10 downto 0);
    signal frac_aSig_uid22_fpAddTest_in : STD_LOGIC_VECTOR (51 downto 0);
    signal frac_aSig_uid22_fpAddTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal excZ_aSig_uid16_uid23_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid24_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsNotZero_uid26_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_aSig_uid27_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_aSig_uid27_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_aSig_uid28_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_aSig_uid28_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invExpXIsMax_uid29_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal InvExpXIsZero_uid30_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excR_aSig_uid31_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excR_aSig_uid31_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal exp_bSig_uid35_fpAddTest_in : STD_LOGIC_VECTOR (62 downto 0);
    signal exp_bSig_uid35_fpAddTest_b : STD_LOGIC_VECTOR (10 downto 0);
    signal frac_bSig_uid36_fpAddTest_in : STD_LOGIC_VECTOR (51 downto 0);
    signal frac_bSig_uid36_fpAddTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal excZ_bSig_uid17_uid37_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excZ_bSig_uid17_uid37_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid38_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid38_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsNotZero_uid40_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_bSig_uid41_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_bSig_uid41_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_bSig_uid42_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_bSig_uid42_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invExpXIsMax_uid43_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal InvExpXIsZero_uid44_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excR_bSig_uid45_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excR_bSig_uid45_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal sigA_uid50_fpAddTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal sigB_uid51_fpAddTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal effSub_uid52_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracBz_uid56_fpAddTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracBz_uid56_fpAddTest_q : STD_LOGIC_VECTOR (51 downto 0);
    signal oFracB_uid59_fpAddTest_q : STD_LOGIC_VECTOR (52 downto 0);
    signal expAmExpB_uid60_fpAddTest_a : STD_LOGIC_VECTOR (11 downto 0);
    signal expAmExpB_uid60_fpAddTest_b : STD_LOGIC_VECTOR (11 downto 0);
    signal expAmExpB_uid60_fpAddTest_o : STD_LOGIC_VECTOR (11 downto 0);
    signal expAmExpB_uid60_fpAddTest_q : STD_LOGIC_VECTOR (11 downto 0);
    signal oFracBR_uid67_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal oFracBREX_uid68_fpAddTest_b : STD_LOGIC_VECTOR (55 downto 0);
    signal oFracBREX_uid68_fpAddTest_qi : STD_LOGIC_VECTOR (55 downto 0);
    signal oFracBREX_uid68_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal oFracBREXC2_uid70_fpAddTest_in : STD_LOGIC_VECTOR (55 downto 0);
    signal oFracBREXC2_uid70_fpAddTest_b : STD_LOGIC_VECTOR (55 downto 0);
    signal fracAddResultNoSignExt_uid73_fpAddTest_in : STD_LOGIC_VECTOR (55 downto 0);
    signal fracAddResultNoSignExt_uid73_fpAddTest_b : STD_LOGIC_VECTOR (55 downto 0);
    signal cAmA_uid76_fpAddTest_q : STD_LOGIC_VECTOR (5 downto 0);
    signal aMinusA_uid77_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expInc_uid78_fpAddTest_a : STD_LOGIC_VECTOR (11 downto 0);
    signal expInc_uid78_fpAddTest_b : STD_LOGIC_VECTOR (11 downto 0);
    signal expInc_uid78_fpAddTest_o : STD_LOGIC_VECTOR (11 downto 0);
    signal expInc_uid78_fpAddTest_q : STD_LOGIC_VECTOR (11 downto 0);
    signal expPostNorm_uid79_fpAddTest_a : STD_LOGIC_VECTOR (12 downto 0);
    signal expPostNorm_uid79_fpAddTest_b : STD_LOGIC_VECTOR (12 downto 0);
    signal expPostNorm_uid79_fpAddTest_o : STD_LOGIC_VECTOR (12 downto 0);
    signal expPostNorm_uid79_fpAddTest_q : STD_LOGIC_VECTOR (12 downto 0);
    signal fracPostNormRndRange_uid80_fpAddTest_in : STD_LOGIC_VECTOR (54 downto 0);
    signal fracPostNormRndRange_uid80_fpAddTest_b : STD_LOGIC_VECTOR (52 downto 0);
    signal expFracR_uid81_fpAddTest_q : STD_LOGIC_VECTOR (65 downto 0);
    signal wEP2AllOwE_uid82_fpAddTest_q : STD_LOGIC_VECTOR (12 downto 0);
    signal rndExp_uid83_fpAddTest_b : STD_LOGIC_VECTOR (12 downto 0);
    signal rOvf_uid84_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal rUdf_uid85_fpAddTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPreExc_uid86_fpAddTest_in : STD_LOGIC_VECTOR (52 downto 0);
    signal fracRPreExc_uid86_fpAddTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal expRPreExc_uid87_fpAddTest_in : STD_LOGIC_VECTOR (63 downto 0);
    signal expRPreExc_uid87_fpAddTest_b : STD_LOGIC_VECTOR (10 downto 0);
    signal regInputs_uid88_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal regInputs_uid88_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRZeroVInC_uid89_fpAddTest_q : STD_LOGIC_VECTOR (4 downto 0);
    signal excRZero_uid90_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal rInfOvf_uid91_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal rInfOvf_uid91_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRInfVInC_uid92_fpAddTest_q : STD_LOGIC_VECTOR (5 downto 0);
    signal excRInf_uid93_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRNaN2_uid94_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excAIBISub_uid95_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRNaN_uid96_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excRNaN_uid96_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal concExc_uid97_fpAddTest_q : STD_LOGIC_VECTOR (2 downto 0);
    signal excREnc_uid98_fpAddTest_q : STD_LOGIC_VECTOR (1 downto 0);
    signal invAMinusA_uid99_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal signRReg_uid100_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal sigBBInf_uid101_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal sigAAInf_uid102_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal signRInf_uid103_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excAZBZSigASigB_uid104_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excBZARSigA_uid105_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal signRZero_uid106_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal signRInfRZRReg_uid107_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal signRInfRZRReg_uid107_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invExcRNaN_uid108_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal signRPostExc_uid109_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal signRPostExc_uid109_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal oneFracRPostExc2_uid110_fpAddTest_q : STD_LOGIC_VECTOR (51 downto 0);
    signal fracRPostExc_uid113_fpAddTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal fracRPostExc_uid113_fpAddTest_q : STD_LOGIC_VECTOR (51 downto 0);
    signal expRPostExc_uid117_fpAddTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal expRPostExc_uid117_fpAddTest_q : STD_LOGIC_VECTOR (10 downto 0);
    signal R_uid118_fpAddTest_q : STD_LOGIC_VECTOR (63 downto 0);
    signal zs_uid120_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (31 downto 0);
    signal rVStage_uid121_lzCountVal_uid74_fpAddTest_b : STD_LOGIC_VECTOR (31 downto 0);
    signal vCount_uid122_lzCountVal_uid74_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid122_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal mO_uid123_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (7 downto 0);
    signal vStage_uid124_lzCountVal_uid74_fpAddTest_in : STD_LOGIC_VECTOR (23 downto 0);
    signal vStage_uid124_lzCountVal_uid74_fpAddTest_b : STD_LOGIC_VECTOR (23 downto 0);
    signal cStage_uid125_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (31 downto 0);
    signal vStagei_uid127_lzCountVal_uid74_fpAddTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid127_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (31 downto 0);
    signal zs_uid128_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (15 downto 0);
    signal vCount_uid130_lzCountVal_uid74_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid130_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid133_lzCountVal_uid74_fpAddTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid133_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (15 downto 0);
    signal zs_uid134_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (7 downto 0);
    signal vCount_uid136_lzCountVal_uid74_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid136_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid139_lzCountVal_uid74_fpAddTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid139_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (7 downto 0);
    signal zs_uid140_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (3 downto 0);
    signal vCount_uid142_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid145_lzCountVal_uid74_fpAddTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid145_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (3 downto 0);
    signal zs_uid146_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vCount_uid148_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid151_lzCountVal_uid74_fpAddTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid151_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (1 downto 0);
    signal rVStage_uid153_lzCountVal_uid74_fpAddTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid154_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid155_lzCountVal_uid74_fpAddTest_q : STD_LOGIC_VECTOR (5 downto 0);
    signal eq0_uid159_fracXIsZero_uid25_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq1_uid162_fracXIsZero_uid25_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq2_uid165_fracXIsZero_uid25_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq3_uid168_fracXIsZero_uid25_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq4_uid171_fracXIsZero_uid25_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq5_uid174_fracXIsZero_uid25_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq6_uid177_fracXIsZero_uid25_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq7_uid180_fracXIsZero_uid25_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq8_uid183_fracXIsZero_uid25_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid184_fracXIsZero_uid25_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid184_fracXIsZero_uid25_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid185_fracXIsZero_uid25_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid185_fracXIsZero_uid25_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev1_uid186_fracXIsZero_uid25_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev1_uid186_fracXIsZero_uid25_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq0_uid189_fracXIsZero_uid39_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq1_uid192_fracXIsZero_uid39_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq2_uid195_fracXIsZero_uid39_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq3_uid198_fracXIsZero_uid39_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq4_uid201_fracXIsZero_uid39_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq5_uid204_fracXIsZero_uid39_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq6_uid207_fracXIsZero_uid39_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq7_uid210_fracXIsZero_uid39_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq8_uid213_fracXIsZero_uid39_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid214_fracXIsZero_uid39_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid214_fracXIsZero_uid39_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid215_fracXIsZero_uid39_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid215_fracXIsZero_uid39_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev1_uid216_fracXIsZero_uid39_fpAddTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev1_uid216_fracXIsZero_uid39_fpAddTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal xMSB_uid217_alignmentShifter_uid71_fpAddTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_a : STD_LOGIC_VECTOR (13 downto 0);
    signal shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_b : STD_LOGIC_VECTOR (13 downto 0);
    signal shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_o : STD_LOGIC_VECTOR (13 downto 0);
    signal shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n : STD_LOGIC_VECTOR (0 downto 0);
    signal seMsb_to16_uid221_in : STD_LOGIC_VECTOR (15 downto 0);
    signal seMsb_to16_uid221_b : STD_LOGIC_VECTOR (15 downto 0);
    signal rightShiftStage0Idx1Rng16_uid222_alignmentShifter_uid71_fpAddTest_b : STD_LOGIC_VECTOR (39 downto 0);
    signal rightShiftStage0Idx1_uid223_alignmentShifter_uid71_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal seMsb_to32_uid224_in : STD_LOGIC_VECTOR (31 downto 0);
    signal seMsb_to32_uid224_b : STD_LOGIC_VECTOR (31 downto 0);
    signal rightShiftStage0Idx2Rng32_uid225_alignmentShifter_uid71_fpAddTest_b : STD_LOGIC_VECTOR (23 downto 0);
    signal rightShiftStage0Idx2_uid226_alignmentShifter_uid71_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal seMsb_to48_uid227_in : STD_LOGIC_VECTOR (47 downto 0);
    signal seMsb_to48_uid227_b : STD_LOGIC_VECTOR (47 downto 0);
    signal rightShiftStage0Idx3Rng48_uid228_alignmentShifter_uid71_fpAddTest_b : STD_LOGIC_VECTOR (7 downto 0);
    signal rightShiftStage0Idx3_uid229_alignmentShifter_uid71_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal seMsb_to4_uid232_in : STD_LOGIC_VECTOR (3 downto 0);
    signal seMsb_to4_uid232_b : STD_LOGIC_VECTOR (3 downto 0);
    signal rightShiftStage1Idx1Rng4_uid233_alignmentShifter_uid71_fpAddTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal rightShiftStage1Idx1_uid234_alignmentShifter_uid71_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal seMsb_to8_uid235_in : STD_LOGIC_VECTOR (7 downto 0);
    signal seMsb_to8_uid235_b : STD_LOGIC_VECTOR (7 downto 0);
    signal rightShiftStage1Idx2Rng8_uid236_alignmentShifter_uid71_fpAddTest_b : STD_LOGIC_VECTOR (47 downto 0);
    signal rightShiftStage1Idx2_uid237_alignmentShifter_uid71_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal seMsb_to12_uid238_in : STD_LOGIC_VECTOR (11 downto 0);
    signal seMsb_to12_uid238_b : STD_LOGIC_VECTOR (11 downto 0);
    signal rightShiftStage1Idx3Rng12_uid239_alignmentShifter_uid71_fpAddTest_b : STD_LOGIC_VECTOR (43 downto 0);
    signal rightShiftStage1Idx3_uid240_alignmentShifter_uid71_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal rightShiftStage2Idx1Rng1_uid243_alignmentShifter_uid71_fpAddTest_b : STD_LOGIC_VECTOR (54 downto 0);
    signal rightShiftStage2Idx1_uid244_alignmentShifter_uid71_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal seMsb_to2_uid245_in : STD_LOGIC_VECTOR (1 downto 0);
    signal seMsb_to2_uid245_b : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStage2Idx2Rng2_uid246_alignmentShifter_uid71_fpAddTest_b : STD_LOGIC_VECTOR (53 downto 0);
    signal rightShiftStage2Idx2_uid247_alignmentShifter_uid71_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal seMsb_to3_uid248_in : STD_LOGIC_VECTOR (2 downto 0);
    signal seMsb_to3_uid248_b : STD_LOGIC_VECTOR (2 downto 0);
    signal rightShiftStage2Idx3Rng3_uid249_alignmentShifter_uid71_fpAddTest_b : STD_LOGIC_VECTOR (52 downto 0);
    signal rightShiftStage2Idx3_uid250_alignmentShifter_uid71_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage0Idx1Rng16_uid260_fracPostNorm_uid75_fpAddTest_in : STD_LOGIC_VECTOR (39 downto 0);
    signal leftShiftStage0Idx1Rng16_uid260_fracPostNorm_uid75_fpAddTest_b : STD_LOGIC_VECTOR (39 downto 0);
    signal leftShiftStage0Idx1_uid261_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage0Idx2_uid264_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage0Idx3Pad48_uid265_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (47 downto 0);
    signal leftShiftStage0Idx3Rng48_uid266_fracPostNorm_uid75_fpAddTest_in : STD_LOGIC_VECTOR (7 downto 0);
    signal leftShiftStage0Idx3Rng48_uid266_fracPostNorm_uid75_fpAddTest_b : STD_LOGIC_VECTOR (7 downto 0);
    signal leftShiftStage0Idx3_uid267_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage1Idx1Rng4_uid271_fracPostNorm_uid75_fpAddTest_in : STD_LOGIC_VECTOR (51 downto 0);
    signal leftShiftStage1Idx1Rng4_uid271_fracPostNorm_uid75_fpAddTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal leftShiftStage1Idx1_uid272_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage1Idx2Rng8_uid274_fracPostNorm_uid75_fpAddTest_in : STD_LOGIC_VECTOR (47 downto 0);
    signal leftShiftStage1Idx2Rng8_uid274_fracPostNorm_uid75_fpAddTest_b : STD_LOGIC_VECTOR (47 downto 0);
    signal leftShiftStage1Idx2_uid275_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage1Idx3Pad12_uid276_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (11 downto 0);
    signal leftShiftStage1Idx3Rng12_uid277_fracPostNorm_uid75_fpAddTest_in : STD_LOGIC_VECTOR (43 downto 0);
    signal leftShiftStage1Idx3Rng12_uid277_fracPostNorm_uid75_fpAddTest_b : STD_LOGIC_VECTOR (43 downto 0);
    signal leftShiftStage1Idx3_uid278_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage2Idx1Rng1_uid282_fracPostNorm_uid75_fpAddTest_in : STD_LOGIC_VECTOR (54 downto 0);
    signal leftShiftStage2Idx1Rng1_uid282_fracPostNorm_uid75_fpAddTest_b : STD_LOGIC_VECTOR (54 downto 0);
    signal leftShiftStage2Idx1_uid283_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage2Idx2Rng2_uid285_fracPostNorm_uid75_fpAddTest_in : STD_LOGIC_VECTOR (53 downto 0);
    signal leftShiftStage2Idx2Rng2_uid285_fracPostNorm_uid75_fpAddTest_b : STD_LOGIC_VECTOR (53 downto 0);
    signal leftShiftStage2Idx2_uid286_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage2Idx3Pad3_uid287_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (2 downto 0);
    signal leftShiftStage2Idx3Rng3_uid288_fracPostNorm_uid75_fpAddTest_in : STD_LOGIC_VECTOR (52 downto 0);
    signal leftShiftStage2Idx3Rng3_uid288_fracPostNorm_uid75_fpAddTest_b : STD_LOGIC_VECTOR (52 downto 0);
    signal leftShiftStage2Idx3_uid289_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage2_uid291_fracPostNorm_uid75_fpAddTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStage2_uid291_fracPostNorm_uid75_fpAddTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal xGTEy_uid8_fpAddTest_BitExpansion_for_a_q : STD_LOGIC_VECTOR (63 downto 0);
    signal xGTEy_uid8_fpAddTest_BitExpansion_for_b_q : STD_LOGIC_VECTOR (63 downto 0);
    signal xGTEy_uid8_fpAddTest_BitSelect_for_a_b : STD_LOGIC_VECTOR (38 downto 0);
    signal xGTEy_uid8_fpAddTest_BitSelect_for_a_c : STD_LOGIC_VECTOR (24 downto 0);
    signal xGTEy_uid8_fpAddTest_BitSelect_for_b_b : STD_LOGIC_VECTOR (38 downto 0);
    signal xGTEy_uid8_fpAddTest_BitSelect_for_b_c : STD_LOGIC_VECTOR (24 downto 0);
    signal xGTEy_uid8_fpAddTest_p1_of_2_a : STD_LOGIC_VECTOR (39 downto 0);
    signal xGTEy_uid8_fpAddTest_p1_of_2_b : STD_LOGIC_VECTOR (39 downto 0);
    signal xGTEy_uid8_fpAddTest_p1_of_2_o : STD_LOGIC_VECTOR (39 downto 0);
    signal xGTEy_uid8_fpAddTest_p1_of_2_c : STD_LOGIC_VECTOR (0 downto 0);
    signal xGTEy_uid8_fpAddTest_p2_of_2_a : STD_LOGIC_VECTOR (26 downto 0);
    signal xGTEy_uid8_fpAddTest_p2_of_2_b : STD_LOGIC_VECTOR (26 downto 0);
    signal xGTEy_uid8_fpAddTest_p2_of_2_o : STD_LOGIC_VECTOR (26 downto 0);
    signal xGTEy_uid8_fpAddTest_p2_of_2_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal xGTEy_uid8_fpAddTest_p2_of_2_c : STD_LOGIC_VECTOR (0 downto 0);
    signal xGTEy_uid8_fpAddTest_cout_n_q : STD_LOGIC_VECTOR (0 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_BitExpansion_for_a_q : STD_LOGIC_VECTOR (56 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_BitExpansion_for_b_q : STD_LOGIC_VECTOR (56 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_UpperBits_for_b_q : STD_LOGIC_VECTOR (55 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_BitSelect_for_a_b : STD_LOGIC_VECTOR (38 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_BitSelect_for_a_c : STD_LOGIC_VECTOR (17 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_BitSelect_for_b_b : STD_LOGIC_VECTOR (38 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_p1_of_2_a : STD_LOGIC_VECTOR (39 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_p1_of_2_b : STD_LOGIC_VECTOR (39 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_p1_of_2_o : STD_LOGIC_VECTOR (39 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_p1_of_2_c : STD_LOGIC_VECTOR (0 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_p1_of_2_q : STD_LOGIC_VECTOR (38 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_p2_of_2_a : STD_LOGIC_VECTOR (19 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_p2_of_2_b : STD_LOGIC_VECTOR (19 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_p2_of_2_o : STD_LOGIC_VECTOR (19 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_p2_of_2_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_p2_of_2_q : STD_LOGIC_VECTOR (17 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (56 downto 0);
    signal fracAddResult_uid72_fpAddTest_p1_of_2_a : STD_LOGIC_VECTOR (39 downto 0);
    signal fracAddResult_uid72_fpAddTest_p1_of_2_b : STD_LOGIC_VECTOR (39 downto 0);
    signal fracAddResult_uid72_fpAddTest_p1_of_2_o : STD_LOGIC_VECTOR (39 downto 0);
    signal fracAddResult_uid72_fpAddTest_p1_of_2_c : STD_LOGIC_VECTOR (0 downto 0);
    signal fracAddResult_uid72_fpAddTest_p1_of_2_q : STD_LOGIC_VECTOR (38 downto 0);
    signal fracAddResult_uid72_fpAddTest_p2_of_2_a : STD_LOGIC_VECTOR (19 downto 0);
    signal fracAddResult_uid72_fpAddTest_p2_of_2_b : STD_LOGIC_VECTOR (19 downto 0);
    signal fracAddResult_uid72_fpAddTest_p2_of_2_o : STD_LOGIC_VECTOR (19 downto 0);
    signal fracAddResult_uid72_fpAddTest_p2_of_2_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal fracAddResult_uid72_fpAddTest_p2_of_2_q : STD_LOGIC_VECTOR (17 downto 0);
    signal fracAddResult_uid72_fpAddTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (56 downto 0);
    signal aSig_uid16_fpAddTest_BitSelect_for_b_b : STD_LOGIC_VECTOR (51 downto 0);
    signal aSig_uid16_fpAddTest_BitSelect_for_b_c : STD_LOGIC_VECTOR (10 downto 0);
    signal aSig_uid16_fpAddTest_BitSelect_for_b_d : STD_LOGIC_VECTOR (0 downto 0);
    signal aSig_uid16_fpAddTest_BitSelect_for_a_b : STD_LOGIC_VECTOR (51 downto 0);
    signal aSig_uid16_fpAddTest_BitSelect_for_a_c : STD_LOGIC_VECTOR (10 downto 0);
    signal aSig_uid16_fpAddTest_BitSelect_for_a_d : STD_LOGIC_VECTOR (0 downto 0);
    signal aSig_uid16_fpAddTest_p0_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aSig_uid16_fpAddTest_p0_q : STD_LOGIC_VECTOR (51 downto 0);
    signal aSig_uid16_fpAddTest_p1_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aSig_uid16_fpAddTest_p1_q : STD_LOGIC_VECTOR (10 downto 0);
    signal aSig_uid16_fpAddTest_p2_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aSig_uid16_fpAddTest_p2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal aSig_uid16_fpAddTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (63 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_b : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_c : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_d : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_e : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_f : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_g : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_h : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_i : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_j : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_k : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_l : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_m : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_n : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_o : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_p : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_r : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_t : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_u : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_v : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_w : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_x : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_y : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_z : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_aa : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_bb : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_cc : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_dd : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ee : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ff : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_gg : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_hh : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ii : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_jj : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_kk : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ll : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_mm : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_nn : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_oo : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_pp : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_qq : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_rr : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ss : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_tt : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_uu : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_vv : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ww : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_xx : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_yy : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_zz : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_1 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_2 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_3 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_4 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_5 : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p0_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p0_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p1_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p2_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p3_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p4_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p5_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p6_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p6_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p7_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p7_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p8_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p8_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p9_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p9_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p10_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p10_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p11_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p11_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p12_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p12_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p13_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p13_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p14_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p14_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p15_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p15_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p16_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p16_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p17_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p17_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p18_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p18_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p19_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p19_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p20_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p20_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p21_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p21_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p22_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p22_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p23_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p23_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p24_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p24_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p25_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p25_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p26_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p26_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p27_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p27_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p28_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p28_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p29_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p29_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p30_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p30_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p31_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p31_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p32_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p32_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p33_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p33_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p34_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p34_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p35_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p35_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p36_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p36_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p37_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p37_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p38_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p38_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p39_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p39_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p40_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p40_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p41_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p41_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p42_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p42_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p43_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p43_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p44_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p44_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p45_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p45_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p46_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p46_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p47_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p47_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p48_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p48_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p49_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p49_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p50_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p50_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p51_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p51_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p52_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p52_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p53_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p53_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p54_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p54_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p55_s : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_p55_q : STD_LOGIC_VECTOR (0 downto 0);
    signal oFracBREXC2_uid69_fpAddTest_BitSelect_for_b_tessel1_0_b : STD_LOGIC_VECTOR (17 downto 0);
    signal fracAddResult_uid72_fpAddTest_BitSelect_for_a_BitJoin_for_b_q : STD_LOGIC_VECTOR (38 downto 0);
    signal fracAddResult_uid72_fpAddTest_BitSelect_for_a_BitJoin_for_c_q : STD_LOGIC_VECTOR (17 downto 0);
    signal fracAddResult_uid72_fpAddTest_BitSelect_for_b_BitJoin_for_b_q : STD_LOGIC_VECTOR (38 downto 0);
    signal fracAddResult_uid72_fpAddTest_BitSelect_for_b_BitJoin_for_c_q : STD_LOGIC_VECTOR (17 downto 0);
    signal r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_b : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_c : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_d : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_e : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_f : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_g : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_h : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_i : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_j : STD_LOGIC_VECTOR (3 downto 0);
    signal z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_b : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_c : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_d : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_e : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_f : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_g : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_h : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_i : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_j : STD_LOGIC_VECTOR (3 downto 0);
    signal z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_b : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_c : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_d : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_e : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_f : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_g : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_h : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_i : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_j : STD_LOGIC_VECTOR (3 downto 0);
    signal rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_in : STD_LOGIC_VECTOR (5 downto 0);
    signal rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_b : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_c : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_d : STD_LOGIC_VECTOR (1 downto 0);
    signal rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_b : STD_LOGIC_VECTOR (15 downto 0);
    signal rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_c : STD_LOGIC_VECTOR (15 downto 0);
    signal rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_b : STD_LOGIC_VECTOR (7 downto 0);
    signal rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_c : STD_LOGIC_VECTOR (7 downto 0);
    signal rVStage_uid141_lzCountVal_uid74_fpAddTest_merged_bit_select_b : STD_LOGIC_VECTOR (3 downto 0);
    signal rVStage_uid141_lzCountVal_uid74_fpAddTest_merged_bit_select_c : STD_LOGIC_VECTOR (3 downto 0);
    signal rVStage_uid147_lzCountVal_uid74_fpAddTest_merged_bit_select_b : STD_LOGIC_VECTOR (1 downto 0);
    signal rVStage_uid147_lzCountVal_uid74_fpAddTest_merged_bit_select_c : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_b : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_c : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_d : STD_LOGIC_VECTOR (1 downto 0);
    signal fracAddResult_uid72_fpAddTest_BitSelect_for_a_tessel0_2_merged_bit_select_b : STD_LOGIC_VECTOR (36 downto 0);
    signal fracAddResult_uid72_fpAddTest_BitSelect_for_a_tessel0_2_merged_bit_select_c : STD_LOGIC_VECTOR (14 downto 0);
    signal redist0_fracAddResult_uid72_fpAddTest_BitSelect_for_a_tessel0_2_merged_bit_select_c_1_q : STD_LOGIC_VECTOR (14 downto 0);
    signal redist1_leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_c_1_q : STD_LOGIC_VECTOR (1 downto 0);
    signal redist2_leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_d_2_q : STD_LOGIC_VECTOR (1 downto 0);
    signal redist3_rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_b_1_q : STD_LOGIC_VECTOR (7 downto 0);
    signal redist4_rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_c_1_q : STD_LOGIC_VECTOR (7 downto 0);
    signal redist5_rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_b_1_q : STD_LOGIC_VECTOR (15 downto 0);
    signal redist6_rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_c_1_q : STD_LOGIC_VECTOR (15 downto 0);
    signal redist7_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_b_3_q : STD_LOGIC_VECTOR (1 downto 0);
    signal redist8_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_c_4_q : STD_LOGIC_VECTOR (1 downto 0);
    signal redist9_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_d_5_q : STD_LOGIC_VECTOR (1 downto 0);
    signal redist10_r_uid255_alignmentShifter_uid71_fpAddTest_p55_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist11_r_uid255_alignmentShifter_uid71_fpAddTest_p54_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist12_r_uid255_alignmentShifter_uid71_fpAddTest_p53_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist13_r_uid255_alignmentShifter_uid71_fpAddTest_p52_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist14_r_uid255_alignmentShifter_uid71_fpAddTest_p51_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist15_r_uid255_alignmentShifter_uid71_fpAddTest_p50_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist16_r_uid255_alignmentShifter_uid71_fpAddTest_p49_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist17_r_uid255_alignmentShifter_uid71_fpAddTest_p48_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist18_r_uid255_alignmentShifter_uid71_fpAddTest_p47_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist19_r_uid255_alignmentShifter_uid71_fpAddTest_p46_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist20_r_uid255_alignmentShifter_uid71_fpAddTest_p45_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist21_r_uid255_alignmentShifter_uid71_fpAddTest_p44_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist22_r_uid255_alignmentShifter_uid71_fpAddTest_p43_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist23_r_uid255_alignmentShifter_uid71_fpAddTest_p42_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist24_r_uid255_alignmentShifter_uid71_fpAddTest_p41_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist25_r_uid255_alignmentShifter_uid71_fpAddTest_p40_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist26_r_uid255_alignmentShifter_uid71_fpAddTest_p39_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist27_aSig_uid16_fpAddTest_p0_q_9_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist28_fracAddResult_uid72_fpAddTest_p1_of_2_q_1_q : STD_LOGIC_VECTOR (38 downto 0);
    signal redist29_oFracBREXC2_uid69_fpAddTest_p2_of_2_q_3_q : STD_LOGIC_VECTOR (17 downto 0);
    signal redist30_oFracBREXC2_uid69_fpAddTest_p1_of_2_q_1_q : STD_LOGIC_VECTOR (38 downto 0);
    signal redist31_oFracBREXC2_uid69_fpAddTest_BitSelect_for_a_c_1_q : STD_LOGIC_VECTOR (17 downto 0);
    signal redist32_xGTEy_uid8_fpAddTest_BitSelect_for_b_c_1_q : STD_LOGIC_VECTOR (24 downto 0);
    signal redist33_xGTEy_uid8_fpAddTest_BitSelect_for_a_c_1_q : STD_LOGIC_VECTOR (24 downto 0);
    signal redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist35_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist36_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist37_and_lev1_uid216_fracXIsZero_uid39_fpAddTest_q_15_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist38_and_lev1_uid186_fracXIsZero_uid25_fpAddTest_q_16_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist39_r_uid155_lzCountVal_uid74_fpAddTest_q_1_q : STD_LOGIC_VECTOR (5 downto 0);
    signal redist40_vCount_uid142_lzCountVal_uid74_fpAddTest_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist41_vCount_uid136_lzCountVal_uid74_fpAddTest_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist42_vCount_uid130_lzCountVal_uid74_fpAddTest_q_5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist43_vStage_uid124_lzCountVal_uid74_fpAddTest_b_7_q : STD_LOGIC_VECTOR (23 downto 0);
    signal redist44_vCount_uid122_lzCountVal_uid74_fpAddTest_q_7_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist45_rVStage_uid121_lzCountVal_uid74_fpAddTest_b_1_q : STD_LOGIC_VECTOR (31 downto 0);
    signal redist46_signRInfRZRReg_uid107_fpAddTest_q_5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist47_excRZero_uid90_fpAddTest_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist48_regInputs_uid88_fpAddTest_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist49_expRPreExc_uid87_fpAddTest_b_3_q : STD_LOGIC_VECTOR (10 downto 0);
    signal redist50_fracRPreExc_uid86_fpAddTest_b_3_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist51_fracPostNormRndRange_uid80_fpAddTest_b_1_q : STD_LOGIC_VECTOR (52 downto 0);
    signal redist52_expPostNorm_uid79_fpAddTest_q_3_q : STD_LOGIC_VECTOR (12 downto 0);
    signal redist53_aMinusA_uid77_fpAddTest_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist54_fracAddResultNoSignExt_uid73_fpAddTest_b_1_q : STD_LOGIC_VECTOR (55 downto 0);
    signal redist55_fracAddResultNoSignExt_uid73_fpAddTest_b_8_q : STD_LOGIC_VECTOR (55 downto 0);
    signal redist56_effSub_uid52_fpAddTest_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist57_effSub_uid52_fpAddTest_q_21_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist58_sigB_uid51_fpAddTest_b_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist59_sigB_uid51_fpAddTest_b_19_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist60_sigA_uid50_fpAddTest_b_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist61_sigA_uid50_fpAddTest_b_18_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist62_InvExpXIsZero_uid44_fpAddTest_q_16_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist63_excN_bSig_uid42_fpAddTest_q_5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist64_excI_bSig_uid41_fpAddTest_q_5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist65_expXIsMax_uid38_fpAddTest_q_17_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist66_excZ_bSig_uid17_uid37_fpAddTest_q_18_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist67_excZ_bSig_uid17_uid37_fpAddTest_q_21_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist68_frac_bSig_uid36_fpAddTest_b_2_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist69_exp_bSig_uid35_fpAddTest_b_1_q : STD_LOGIC_VECTOR (10 downto 0);
    signal redist70_excN_aSig_uid28_fpAddTest_q_5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist71_excI_aSig_uid27_fpAddTest_q_5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist72_excZ_aSig_uid16_uid23_fpAddTest_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist73_excZ_aSig_uid16_uid23_fpAddTest_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist74_exp_aSig_uid21_fpAddTest_b_17_q : STD_LOGIC_VECTOR (10 downto 0);
    signal redist75_expY_uid11_fpAddTest_b_2_q : STD_LOGIC_VECTOR (10 downto 0);
    signal redist76_fracY_uid10_fpAddTest_b_2_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist77_sigY_uid9_fpAddTest_b_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist78_xIn_a_2_q : STD_LOGIC_VECTOR (63 downto 0);

begin


    -- cAmA_uid76_fpAddTest(CONSTANT,75)
    cAmA_uid76_fpAddTest_q <= "111000";

    -- zs_uid120_lzCountVal_uid74_fpAddTest(CONSTANT,119)
    zs_uid120_lzCountVal_uid74_fpAddTest_q <= "00000000000000000000000000000000";

    -- GND(CONSTANT,0)
    GND_q <= "0";

    -- oFracBREXC2_uid69_fpAddTest_UpperBits_for_b(CONSTANT,305)
    oFracBREXC2_uid69_fpAddTest_UpperBits_for_b_q <= "00000000000000000000000000000000000000000000000000000000";

    -- sigY_uid9_fpAddTest(BITSELECT,8)@0
    sigY_uid9_fpAddTest_b <= STD_LOGIC_VECTOR(b(63 downto 63));

    -- redist77_sigY_uid9_fpAddTest_b_2(DELAY,656)
    redist77_sigY_uid9_fpAddTest_b_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => sigY_uid9_fpAddTest_b, xout => redist77_sigY_uid9_fpAddTest_b_2_q, clk => clk, aclr => areset );

    -- expY_uid11_fpAddTest(BITSELECT,10)@0
    expY_uid11_fpAddTest_b <= b(62 downto 52);

    -- redist75_expY_uid11_fpAddTest_b_2(DELAY,654)
    redist75_expY_uid11_fpAddTest_b_2 : dspba_delay
    GENERIC MAP ( width => 11, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => expY_uid11_fpAddTest_b, xout => redist75_expY_uid11_fpAddTest_b_2_q, clk => clk, aclr => areset );

    -- fracY_uid10_fpAddTest(BITSELECT,9)@0
    fracY_uid10_fpAddTest_b <= b(51 downto 0);

    -- redist76_fracY_uid10_fpAddTest_b_2(DELAY,655)
    redist76_fracY_uid10_fpAddTest_b_2 : dspba_delay
    GENERIC MAP ( width => 52, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => fracY_uid10_fpAddTest_b, xout => redist76_fracY_uid10_fpAddTest_b_2_q, clk => clk, aclr => areset );

    -- ypn_uid12_fpAddTest(BITJOIN,11)@2
    ypn_uid12_fpAddTest_q <= redist77_sigY_uid9_fpAddTest_b_2_q & redist75_expY_uid11_fpAddTest_b_2_q & redist76_fracY_uid10_fpAddTest_b_2_q;

    -- redist78_xIn_a_2(DELAY,657)
    redist78_xIn_a_2 : dspba_delay
    GENERIC MAP ( width => 64, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => a, xout => redist78_xIn_a_2_q, clk => clk, aclr => areset );

    -- expFracY_uid7_fpAddTest(BITSELECT,6)@0
    expFracY_uid7_fpAddTest_b <= b(62 downto 0);

    -- xGTEy_uid8_fpAddTest_BitExpansion_for_b(BITJOIN,294)@0
    xGTEy_uid8_fpAddTest_BitExpansion_for_b_q <= GND_q & expFracY_uid7_fpAddTest_b;

    -- xGTEy_uid8_fpAddTest_BitSelect_for_b(BITSELECT,297)@0
    xGTEy_uid8_fpAddTest_BitSelect_for_b_b <= xGTEy_uid8_fpAddTest_BitExpansion_for_b_q(38 downto 0);
    xGTEy_uid8_fpAddTest_BitSelect_for_b_c <= xGTEy_uid8_fpAddTest_BitExpansion_for_b_q(63 downto 39);

    -- expFracX_uid6_fpAddTest(BITSELECT,5)@0
    expFracX_uid6_fpAddTest_b <= a(62 downto 0);

    -- xGTEy_uid8_fpAddTest_BitExpansion_for_a(BITJOIN,292)@0
    xGTEy_uid8_fpAddTest_BitExpansion_for_a_q <= GND_q & expFracX_uid6_fpAddTest_b;

    -- xGTEy_uid8_fpAddTest_BitSelect_for_a(BITSELECT,296)@0
    xGTEy_uid8_fpAddTest_BitSelect_for_a_b <= xGTEy_uid8_fpAddTest_BitExpansion_for_a_q(38 downto 0);
    xGTEy_uid8_fpAddTest_BitSelect_for_a_c <= xGTEy_uid8_fpAddTest_BitExpansion_for_a_q(63 downto 39);

    -- xGTEy_uid8_fpAddTest_p1_of_2(COMPARE,298)@0 + 1
    xGTEy_uid8_fpAddTest_p1_of_2_a <= STD_LOGIC_VECTOR("0" & xGTEy_uid8_fpAddTest_BitSelect_for_a_b);
    xGTEy_uid8_fpAddTest_p1_of_2_b <= STD_LOGIC_VECTOR("0" & xGTEy_uid8_fpAddTest_BitSelect_for_b_b);
    xGTEy_uid8_fpAddTest_p1_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            xGTEy_uid8_fpAddTest_p1_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            xGTEy_uid8_fpAddTest_p1_of_2_o <= STD_LOGIC_VECTOR(UNSIGNED(xGTEy_uid8_fpAddTest_p1_of_2_a) - UNSIGNED(xGTEy_uid8_fpAddTest_p1_of_2_b));
        END IF;
    END PROCESS;
    xGTEy_uid8_fpAddTest_p1_of_2_c(0) <= xGTEy_uid8_fpAddTest_p1_of_2_o(39);

    -- redist32_xGTEy_uid8_fpAddTest_BitSelect_for_b_c_1(DELAY,611)
    redist32_xGTEy_uid8_fpAddTest_BitSelect_for_b_c_1 : dspba_delay
    GENERIC MAP ( width => 25, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xGTEy_uid8_fpAddTest_BitSelect_for_b_c, xout => redist32_xGTEy_uid8_fpAddTest_BitSelect_for_b_c_1_q, clk => clk, aclr => areset );

    -- redist33_xGTEy_uid8_fpAddTest_BitSelect_for_a_c_1(DELAY,612)
    redist33_xGTEy_uid8_fpAddTest_BitSelect_for_a_c_1 : dspba_delay
    GENERIC MAP ( width => 25, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xGTEy_uid8_fpAddTest_BitSelect_for_a_c, xout => redist33_xGTEy_uid8_fpAddTest_BitSelect_for_a_c_1_q, clk => clk, aclr => areset );

    -- xGTEy_uid8_fpAddTest_p2_of_2(COMPARE,299)@1 + 1
    xGTEy_uid8_fpAddTest_p2_of_2_cin <= xGTEy_uid8_fpAddTest_p1_of_2_c;
    xGTEy_uid8_fpAddTest_p2_of_2_a <= STD_LOGIC_VECTOR("0" & redist33_xGTEy_uid8_fpAddTest_BitSelect_for_a_c_1_q) & '0';
    xGTEy_uid8_fpAddTest_p2_of_2_b <= STD_LOGIC_VECTOR("0" & redist32_xGTEy_uid8_fpAddTest_BitSelect_for_b_c_1_q) & xGTEy_uid8_fpAddTest_p2_of_2_cin(0);
    xGTEy_uid8_fpAddTest_p2_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            xGTEy_uid8_fpAddTest_p2_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            xGTEy_uid8_fpAddTest_p2_of_2_o <= STD_LOGIC_VECTOR(UNSIGNED(xGTEy_uid8_fpAddTest_p2_of_2_a) - UNSIGNED(xGTEy_uid8_fpAddTest_p2_of_2_b));
        END IF;
    END PROCESS;
    xGTEy_uid8_fpAddTest_p2_of_2_c(0) <= xGTEy_uid8_fpAddTest_p2_of_2_o(26);

    -- xGTEy_uid8_fpAddTest_cout_n(LOGICAL,301)@2
    xGTEy_uid8_fpAddTest_cout_n_q <= STD_LOGIC_VECTOR(not (xGTEy_uid8_fpAddTest_p2_of_2_c));

    -- bSig_uid17_fpAddTest(MUX,16)@2
    bSig_uid17_fpAddTest_s <= xGTEy_uid8_fpAddTest_cout_n_q;
    bSig_uid17_fpAddTest_combproc: PROCESS (bSig_uid17_fpAddTest_s, redist78_xIn_a_2_q, ypn_uid12_fpAddTest_q)
    BEGIN
        CASE (bSig_uid17_fpAddTest_s) IS
            WHEN "0" => bSig_uid17_fpAddTest_q <= redist78_xIn_a_2_q;
            WHEN "1" => bSig_uid17_fpAddTest_q <= ypn_uid12_fpAddTest_q;
            WHEN OTHERS => bSig_uid17_fpAddTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- sigB_uid51_fpAddTest(BITSELECT,50)@2
    sigB_uid51_fpAddTest_b <= STD_LOGIC_VECTOR(bSig_uid17_fpAddTest_q(63 downto 63));

    -- redist58_sigB_uid51_fpAddTest_b_2(DELAY,637)
    redist58_sigB_uid51_fpAddTest_b_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => sigB_uid51_fpAddTest_b, xout => redist58_sigB_uid51_fpAddTest_b_2_q, clk => clk, aclr => areset );

    -- aSig_uid16_fpAddTest_BitSelect_for_b(BITSELECT,322)@2
    aSig_uid16_fpAddTest_BitSelect_for_b_b <= STD_LOGIC_VECTOR(redist78_xIn_a_2_q(51 downto 0));
    aSig_uid16_fpAddTest_BitSelect_for_b_c <= STD_LOGIC_VECTOR(redist78_xIn_a_2_q(62 downto 52));
    aSig_uid16_fpAddTest_BitSelect_for_b_d <= STD_LOGIC_VECTOR(redist78_xIn_a_2_q(63 downto 63));

    -- aSig_uid16_fpAddTest_BitSelect_for_a(BITSELECT,323)@2
    aSig_uid16_fpAddTest_BitSelect_for_a_b <= STD_LOGIC_VECTOR(ypn_uid12_fpAddTest_q(51 downto 0));
    aSig_uid16_fpAddTest_BitSelect_for_a_c <= STD_LOGIC_VECTOR(ypn_uid12_fpAddTest_q(62 downto 52));
    aSig_uid16_fpAddTest_BitSelect_for_a_d <= STD_LOGIC_VECTOR(ypn_uid12_fpAddTest_q(63 downto 63));

    -- aSig_uid16_fpAddTest_p2(MUX,326)@2 + 1
    aSig_uid16_fpAddTest_p2_s <= xGTEy_uid8_fpAddTest_cout_n_q;
    aSig_uid16_fpAddTest_p2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            aSig_uid16_fpAddTest_p2_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (aSig_uid16_fpAddTest_p2_s) IS
                WHEN "0" => aSig_uid16_fpAddTest_p2_q <= aSig_uid16_fpAddTest_BitSelect_for_a_d;
                WHEN "1" => aSig_uid16_fpAddTest_p2_q <= aSig_uid16_fpAddTest_BitSelect_for_b_d;
                WHEN OTHERS => aSig_uid16_fpAddTest_p2_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- aSig_uid16_fpAddTest_p1(MUX,325)@2 + 1
    aSig_uid16_fpAddTest_p1_s <= xGTEy_uid8_fpAddTest_cout_n_q;
    aSig_uid16_fpAddTest_p1_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            aSig_uid16_fpAddTest_p1_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (aSig_uid16_fpAddTest_p1_s) IS
                WHEN "0" => aSig_uid16_fpAddTest_p1_q <= aSig_uid16_fpAddTest_BitSelect_for_a_c;
                WHEN "1" => aSig_uid16_fpAddTest_p1_q <= aSig_uid16_fpAddTest_BitSelect_for_b_c;
                WHEN OTHERS => aSig_uid16_fpAddTest_p1_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- aSig_uid16_fpAddTest_p0(MUX,324)@2 + 1
    aSig_uid16_fpAddTest_p0_s <= xGTEy_uid8_fpAddTest_cout_n_q;
    aSig_uid16_fpAddTest_p0_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            aSig_uid16_fpAddTest_p0_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (aSig_uid16_fpAddTest_p0_s) IS
                WHEN "0" => aSig_uid16_fpAddTest_p0_q <= aSig_uid16_fpAddTest_BitSelect_for_a_b;
                WHEN "1" => aSig_uid16_fpAddTest_p0_q <= aSig_uid16_fpAddTest_BitSelect_for_b_b;
                WHEN OTHERS => aSig_uid16_fpAddTest_p0_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- aSig_uid16_fpAddTest_BitJoin_for_q(BITJOIN,327)@3
    aSig_uid16_fpAddTest_BitJoin_for_q_q <= aSig_uid16_fpAddTest_p2_q & aSig_uid16_fpAddTest_p1_q & aSig_uid16_fpAddTest_p0_q;

    -- sigA_uid50_fpAddTest(BITSELECT,49)@3
    sigA_uid50_fpAddTest_b <= STD_LOGIC_VECTOR(aSig_uid16_fpAddTest_BitJoin_for_q_q(63 downto 63));

    -- redist60_sigA_uid50_fpAddTest_b_1(DELAY,639)
    redist60_sigA_uid50_fpAddTest_b_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sigA_uid50_fpAddTest_b, xout => redist60_sigA_uid50_fpAddTest_b_1_q, clk => clk, aclr => areset );

    -- effSub_uid52_fpAddTest(LOGICAL,51)@4
    effSub_uid52_fpAddTest_q <= redist60_sigA_uid50_fpAddTest_b_1_q xor redist58_sigB_uid51_fpAddTest_b_2_q;

    -- redist56_effSub_uid52_fpAddTest_q_1(DELAY,635)
    redist56_effSub_uid52_fpAddTest_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => effSub_uid52_fpAddTest_q, xout => redist56_effSub_uid52_fpAddTest_q_1_q, clk => clk, aclr => areset );

    -- oFracBREXC2_uid69_fpAddTest_BitExpansion_for_b(BITJOIN,304)@5
    oFracBREXC2_uid69_fpAddTest_BitExpansion_for_b_q <= oFracBREXC2_uid69_fpAddTest_UpperBits_for_b_q & redist56_effSub_uid52_fpAddTest_q_1_q;

    -- oFracBREXC2_uid69_fpAddTest_BitSelect_for_b(BITSELECT,307)@5
    oFracBREXC2_uid69_fpAddTest_BitSelect_for_b_b <= oFracBREXC2_uid69_fpAddTest_BitExpansion_for_b_q(38 downto 0);

    -- cstAllZWE_uid20_fpAddTest(CONSTANT,19)
    cstAllZWE_uid20_fpAddTest_q <= "00000000000";

    -- exp_bSig_uid35_fpAddTest(BITSELECT,34)@2
    exp_bSig_uid35_fpAddTest_in <= bSig_uid17_fpAddTest_q(62 downto 0);
    exp_bSig_uid35_fpAddTest_b <= exp_bSig_uid35_fpAddTest_in(62 downto 52);

    -- redist69_exp_bSig_uid35_fpAddTest_b_1(DELAY,648)
    redist69_exp_bSig_uid35_fpAddTest_b_1 : dspba_delay
    GENERIC MAP ( width => 11, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => exp_bSig_uid35_fpAddTest_b, xout => redist69_exp_bSig_uid35_fpAddTest_b_1_q, clk => clk, aclr => areset );

    -- excZ_bSig_uid17_uid37_fpAddTest(LOGICAL,36)@3 + 1
    excZ_bSig_uid17_uid37_fpAddTest_qi <= "1" WHEN redist69_exp_bSig_uid35_fpAddTest_b_1_q = cstAllZWE_uid20_fpAddTest_q ELSE "0";
    excZ_bSig_uid17_uid37_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excZ_bSig_uid17_uid37_fpAddTest_qi, xout => excZ_bSig_uid17_uid37_fpAddTest_q, clk => clk, aclr => areset );

    -- InvExpXIsZero_uid44_fpAddTest(LOGICAL,43)@4
    InvExpXIsZero_uid44_fpAddTest_q <= not (excZ_bSig_uid17_uid37_fpAddTest_q);

    -- cstZeroWF_uid19_fpAddTest(CONSTANT,18)
    cstZeroWF_uid19_fpAddTest_q <= "0000000000000000000000000000000000000000000000000000";

    -- frac_bSig_uid36_fpAddTest(BITSELECT,35)@2
    frac_bSig_uid36_fpAddTest_in <= bSig_uid17_fpAddTest_q(51 downto 0);
    frac_bSig_uid36_fpAddTest_b <= frac_bSig_uid36_fpAddTest_in(51 downto 0);

    -- redist68_frac_bSig_uid36_fpAddTest_b_2(DELAY,647)
    redist68_frac_bSig_uid36_fpAddTest_b_2 : dspba_delay
    GENERIC MAP ( width => 52, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => frac_bSig_uid36_fpAddTest_b, xout => redist68_frac_bSig_uid36_fpAddTest_b_2_q, clk => clk, aclr => areset );

    -- fracBz_uid56_fpAddTest(MUX,55)@4
    fracBz_uid56_fpAddTest_s <= excZ_bSig_uid17_uid37_fpAddTest_q;
    fracBz_uid56_fpAddTest_combproc: PROCESS (fracBz_uid56_fpAddTest_s, redist68_frac_bSig_uid36_fpAddTest_b_2_q, cstZeroWF_uid19_fpAddTest_q)
    BEGIN
        CASE (fracBz_uid56_fpAddTest_s) IS
            WHEN "0" => fracBz_uid56_fpAddTest_q <= redist68_frac_bSig_uid36_fpAddTest_b_2_q;
            WHEN "1" => fracBz_uid56_fpAddTest_q <= cstZeroWF_uid19_fpAddTest_q;
            WHEN OTHERS => fracBz_uid56_fpAddTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- oFracB_uid59_fpAddTest(BITJOIN,58)@4
    oFracB_uid59_fpAddTest_q <= InvExpXIsZero_uid44_fpAddTest_q & fracBz_uid56_fpAddTest_q;

    -- oFracBR_uid67_fpAddTest(BITJOIN,66)@4
    oFracBR_uid67_fpAddTest_q <= GND_q & oFracB_uid59_fpAddTest_q & GND_q & GND_q;

    -- oFracBREX_uid68_fpAddTest(LOGICAL,67)@4 + 1
    oFracBREX_uid68_fpAddTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((55 downto 1 => effSub_uid52_fpAddTest_q(0)) & effSub_uid52_fpAddTest_q));
    oFracBREX_uid68_fpAddTest_qi <= oFracBR_uid67_fpAddTest_q xor oFracBREX_uid68_fpAddTest_b;
    oFracBREX_uid68_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 56, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => oFracBREX_uid68_fpAddTest_qi, xout => oFracBREX_uid68_fpAddTest_q, clk => clk, aclr => areset );

    -- oFracBREXC2_uid69_fpAddTest_BitExpansion_for_a(BITJOIN,302)@5
    oFracBREXC2_uid69_fpAddTest_BitExpansion_for_a_q <= GND_q & oFracBREX_uid68_fpAddTest_q;

    -- oFracBREXC2_uid69_fpAddTest_BitSelect_for_a(BITSELECT,306)@5
    oFracBREXC2_uid69_fpAddTest_BitSelect_for_a_b <= oFracBREXC2_uid69_fpAddTest_BitExpansion_for_a_q(38 downto 0);
    oFracBREXC2_uid69_fpAddTest_BitSelect_for_a_c <= oFracBREXC2_uid69_fpAddTest_BitExpansion_for_a_q(56 downto 39);

    -- oFracBREXC2_uid69_fpAddTest_p1_of_2(ADD,308)@5 + 1
    oFracBREXC2_uid69_fpAddTest_p1_of_2_a <= STD_LOGIC_VECTOR("0" & oFracBREXC2_uid69_fpAddTest_BitSelect_for_a_b);
    oFracBREXC2_uid69_fpAddTest_p1_of_2_b <= STD_LOGIC_VECTOR("0" & oFracBREXC2_uid69_fpAddTest_BitSelect_for_b_b);
    oFracBREXC2_uid69_fpAddTest_p1_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            oFracBREXC2_uid69_fpAddTest_p1_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            oFracBREXC2_uid69_fpAddTest_p1_of_2_o <= STD_LOGIC_VECTOR(UNSIGNED(oFracBREXC2_uid69_fpAddTest_p1_of_2_a) + UNSIGNED(oFracBREXC2_uid69_fpAddTest_p1_of_2_b));
        END IF;
    END PROCESS;
    oFracBREXC2_uid69_fpAddTest_p1_of_2_c(0) <= oFracBREXC2_uid69_fpAddTest_p1_of_2_o(39);
    oFracBREXC2_uid69_fpAddTest_p1_of_2_q <= oFracBREXC2_uid69_fpAddTest_p1_of_2_o(38 downto 0);

    -- oFracBREXC2_uid69_fpAddTest_BitSelect_for_b_tessel1_0(BITSELECT,387)
    oFracBREXC2_uid69_fpAddTest_BitSelect_for_b_tessel1_0_b <= STD_LOGIC_VECTOR(oFracBREXC2_uid69_fpAddTest_UpperBits_for_b_q(55 downto 38));

    -- redist31_oFracBREXC2_uid69_fpAddTest_BitSelect_for_a_c_1(DELAY,610)
    redist31_oFracBREXC2_uid69_fpAddTest_BitSelect_for_a_c_1 : dspba_delay
    GENERIC MAP ( width => 18, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => oFracBREXC2_uid69_fpAddTest_BitSelect_for_a_c, xout => redist31_oFracBREXC2_uid69_fpAddTest_BitSelect_for_a_c_1_q, clk => clk, aclr => areset );

    -- oFracBREXC2_uid69_fpAddTest_p2_of_2(ADD,309)@6 + 1
    oFracBREXC2_uid69_fpAddTest_p2_of_2_cin <= oFracBREXC2_uid69_fpAddTest_p1_of_2_c;
    oFracBREXC2_uid69_fpAddTest_p2_of_2_a <= STD_LOGIC_VECTOR("0" & redist31_oFracBREXC2_uid69_fpAddTest_BitSelect_for_a_c_1_q) & '1';
    oFracBREXC2_uid69_fpAddTest_p2_of_2_b <= STD_LOGIC_VECTOR("0" & oFracBREXC2_uid69_fpAddTest_BitSelect_for_b_tessel1_0_b) & oFracBREXC2_uid69_fpAddTest_p2_of_2_cin(0);
    oFracBREXC2_uid69_fpAddTest_p2_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            oFracBREXC2_uid69_fpAddTest_p2_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            oFracBREXC2_uid69_fpAddTest_p2_of_2_o <= STD_LOGIC_VECTOR(UNSIGNED(oFracBREXC2_uid69_fpAddTest_p2_of_2_a) + UNSIGNED(oFracBREXC2_uid69_fpAddTest_p2_of_2_b));
        END IF;
    END PROCESS;
    oFracBREXC2_uid69_fpAddTest_p2_of_2_q <= oFracBREXC2_uid69_fpAddTest_p2_of_2_o(18 downto 1);

    -- redist29_oFracBREXC2_uid69_fpAddTest_p2_of_2_q_3(DELAY,608)
    redist29_oFracBREXC2_uid69_fpAddTest_p2_of_2_q_3 : dspba_delay
    GENERIC MAP ( width => 18, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => oFracBREXC2_uid69_fpAddTest_p2_of_2_q, xout => redist29_oFracBREXC2_uid69_fpAddTest_p2_of_2_q_3_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0(BITSELECT,457)@10
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b <= STD_LOGIC_VECTOR(redist29_oFracBREXC2_uid69_fpAddTest_p2_of_2_q_3_q(16 downto 16));

    -- redist30_oFracBREXC2_uid69_fpAddTest_p1_of_2_q_1(DELAY,609)
    redist30_oFracBREXC2_uid69_fpAddTest_p1_of_2_q_1 : dspba_delay
    GENERIC MAP ( width => 39, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => oFracBREXC2_uid69_fpAddTest_p1_of_2_q, xout => redist30_oFracBREXC2_uid69_fpAddTest_p1_of_2_q_1_q, clk => clk, aclr => areset );

    -- oFracBREXC2_uid69_fpAddTest_BitJoin_for_q(BITJOIN,310)@7
    oFracBREXC2_uid69_fpAddTest_BitJoin_for_q_q <= oFracBREXC2_uid69_fpAddTest_p2_of_2_q & redist30_oFracBREXC2_uid69_fpAddTest_p1_of_2_q_1_q;

    -- oFracBREXC2_uid70_fpAddTest(BITSELECT,69)@7
    oFracBREXC2_uid70_fpAddTest_in <= STD_LOGIC_VECTOR(oFracBREXC2_uid69_fpAddTest_BitJoin_for_q_q(55 downto 0));
    oFracBREXC2_uid70_fpAddTest_b <= STD_LOGIC_VECTOR(oFracBREXC2_uid70_fpAddTest_in(55 downto 0));

    -- xMSB_uid217_alignmentShifter_uid71_fpAddTest(BITSELECT,216)@7
    xMSB_uid217_alignmentShifter_uid71_fpAddTest_b <= STD_LOGIC_VECTOR(oFracBREXC2_uid70_fpAddTest_b(55 downto 55));

    -- redist35_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_1(DELAY,614)
    redist35_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid217_alignmentShifter_uid71_fpAddTest_b, xout => redist35_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_1_q, clk => clk, aclr => areset );

    -- redist36_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_2(DELAY,615)
    redist36_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist35_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_1_q, xout => redist36_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_2_q, clk => clk, aclr => areset );

    -- seMsb_to3_uid248(BITSELECT,247)@9
    seMsb_to3_uid248_in <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((2 downto 1 => redist36_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_2_q(0)) & redist36_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_2_q));
    seMsb_to3_uid248_b <= STD_LOGIC_VECTOR(seMsb_to3_uid248_in(2 downto 0));

    -- rightShiftStage2Idx3Rng3_uid249_alignmentShifter_uid71_fpAddTest(BITSELECT,248)@9
    rightShiftStage2Idx3Rng3_uid249_alignmentShifter_uid71_fpAddTest_b <= rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_q(55 downto 3);

    -- rightShiftStage2Idx3_uid250_alignmentShifter_uid71_fpAddTest(BITJOIN,249)@9
    rightShiftStage2Idx3_uid250_alignmentShifter_uid71_fpAddTest_q <= seMsb_to3_uid248_b & rightShiftStage2Idx3Rng3_uid249_alignmentShifter_uid71_fpAddTest_b;

    -- seMsb_to2_uid245(BITSELECT,244)@9
    seMsb_to2_uid245_in <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((1 downto 1 => redist36_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_2_q(0)) & redist36_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_2_q));
    seMsb_to2_uid245_b <= STD_LOGIC_VECTOR(seMsb_to2_uid245_in(1 downto 0));

    -- rightShiftStage2Idx2Rng2_uid246_alignmentShifter_uid71_fpAddTest(BITSELECT,245)@9
    rightShiftStage2Idx2Rng2_uid246_alignmentShifter_uid71_fpAddTest_b <= rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_q(55 downto 2);

    -- rightShiftStage2Idx2_uid247_alignmentShifter_uid71_fpAddTest(BITJOIN,246)@9
    rightShiftStage2Idx2_uid247_alignmentShifter_uid71_fpAddTest_q <= seMsb_to2_uid245_b & rightShiftStage2Idx2Rng2_uid246_alignmentShifter_uid71_fpAddTest_b;

    -- rightShiftStage2Idx1Rng1_uid243_alignmentShifter_uid71_fpAddTest(BITSELECT,242)@9
    rightShiftStage2Idx1Rng1_uid243_alignmentShifter_uid71_fpAddTest_b <= rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_q(55 downto 1);

    -- rightShiftStage2Idx1_uid244_alignmentShifter_uid71_fpAddTest(BITJOIN,243)@9
    rightShiftStage2Idx1_uid244_alignmentShifter_uid71_fpAddTest_q <= redist36_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_2_q & rightShiftStage2Idx1Rng1_uid243_alignmentShifter_uid71_fpAddTest_b;

    -- seMsb_to12_uid238(BITSELECT,237)@8
    seMsb_to12_uid238_in <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((11 downto 1 => redist35_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_1_q(0)) & redist35_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_1_q));
    seMsb_to12_uid238_b <= STD_LOGIC_VECTOR(seMsb_to12_uid238_in(11 downto 0));

    -- rightShiftStage1Idx3Rng12_uid239_alignmentShifter_uid71_fpAddTest(BITSELECT,238)@8
    rightShiftStage1Idx3Rng12_uid239_alignmentShifter_uid71_fpAddTest_b <= rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_q(55 downto 12);

    -- rightShiftStage1Idx3_uid240_alignmentShifter_uid71_fpAddTest(BITJOIN,239)@8
    rightShiftStage1Idx3_uid240_alignmentShifter_uid71_fpAddTest_q <= seMsb_to12_uid238_b & rightShiftStage1Idx3Rng12_uid239_alignmentShifter_uid71_fpAddTest_b;

    -- seMsb_to8_uid235(BITSELECT,234)@8
    seMsb_to8_uid235_in <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((7 downto 1 => redist35_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_1_q(0)) & redist35_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_1_q));
    seMsb_to8_uid235_b <= STD_LOGIC_VECTOR(seMsb_to8_uid235_in(7 downto 0));

    -- rightShiftStage1Idx2Rng8_uid236_alignmentShifter_uid71_fpAddTest(BITSELECT,235)@8
    rightShiftStage1Idx2Rng8_uid236_alignmentShifter_uid71_fpAddTest_b <= rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_q(55 downto 8);

    -- rightShiftStage1Idx2_uid237_alignmentShifter_uid71_fpAddTest(BITJOIN,236)@8
    rightShiftStage1Idx2_uid237_alignmentShifter_uid71_fpAddTest_q <= seMsb_to8_uid235_b & rightShiftStage1Idx2Rng8_uid236_alignmentShifter_uid71_fpAddTest_b;

    -- seMsb_to4_uid232(BITSELECT,231)@8
    seMsb_to4_uid232_in <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((3 downto 1 => redist35_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_1_q(0)) & redist35_xMSB_uid217_alignmentShifter_uid71_fpAddTest_b_1_q));
    seMsb_to4_uid232_b <= STD_LOGIC_VECTOR(seMsb_to4_uid232_in(3 downto 0));

    -- rightShiftStage1Idx1Rng4_uid233_alignmentShifter_uid71_fpAddTest(BITSELECT,232)@8
    rightShiftStage1Idx1Rng4_uid233_alignmentShifter_uid71_fpAddTest_b <= rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_q(55 downto 4);

    -- rightShiftStage1Idx1_uid234_alignmentShifter_uid71_fpAddTest(BITJOIN,233)@8
    rightShiftStage1Idx1_uid234_alignmentShifter_uid71_fpAddTest_q <= seMsb_to4_uid232_b & rightShiftStage1Idx1Rng4_uid233_alignmentShifter_uid71_fpAddTest_b;

    -- seMsb_to48_uid227(BITSELECT,226)@7
    seMsb_to48_uid227_in <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((47 downto 1 => xMSB_uid217_alignmentShifter_uid71_fpAddTest_b(0)) & xMSB_uid217_alignmentShifter_uid71_fpAddTest_b));
    seMsb_to48_uid227_b <= STD_LOGIC_VECTOR(seMsb_to48_uid227_in(47 downto 0));

    -- rightShiftStage0Idx3Rng48_uid228_alignmentShifter_uid71_fpAddTest(BITSELECT,227)@7
    rightShiftStage0Idx3Rng48_uid228_alignmentShifter_uid71_fpAddTest_b <= oFracBREXC2_uid70_fpAddTest_b(55 downto 48);

    -- rightShiftStage0Idx3_uid229_alignmentShifter_uid71_fpAddTest(BITJOIN,228)@7
    rightShiftStage0Idx3_uid229_alignmentShifter_uid71_fpAddTest_q <= seMsb_to48_uid227_b & rightShiftStage0Idx3Rng48_uid228_alignmentShifter_uid71_fpAddTest_b;

    -- seMsb_to32_uid224(BITSELECT,223)@7
    seMsb_to32_uid224_in <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((31 downto 1 => xMSB_uid217_alignmentShifter_uid71_fpAddTest_b(0)) & xMSB_uid217_alignmentShifter_uid71_fpAddTest_b));
    seMsb_to32_uid224_b <= STD_LOGIC_VECTOR(seMsb_to32_uid224_in(31 downto 0));

    -- rightShiftStage0Idx2Rng32_uid225_alignmentShifter_uid71_fpAddTest(BITSELECT,224)@7
    rightShiftStage0Idx2Rng32_uid225_alignmentShifter_uid71_fpAddTest_b <= oFracBREXC2_uid70_fpAddTest_b(55 downto 32);

    -- rightShiftStage0Idx2_uid226_alignmentShifter_uid71_fpAddTest(BITJOIN,225)@7
    rightShiftStage0Idx2_uid226_alignmentShifter_uid71_fpAddTest_q <= seMsb_to32_uid224_b & rightShiftStage0Idx2Rng32_uid225_alignmentShifter_uid71_fpAddTest_b;

    -- seMsb_to16_uid221(BITSELECT,220)@7
    seMsb_to16_uid221_in <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((15 downto 1 => xMSB_uid217_alignmentShifter_uid71_fpAddTest_b(0)) & xMSB_uid217_alignmentShifter_uid71_fpAddTest_b));
    seMsb_to16_uid221_b <= STD_LOGIC_VECTOR(seMsb_to16_uid221_in(15 downto 0));

    -- rightShiftStage0Idx1Rng16_uid222_alignmentShifter_uid71_fpAddTest(BITSELECT,221)@7
    rightShiftStage0Idx1Rng16_uid222_alignmentShifter_uid71_fpAddTest_b <= oFracBREXC2_uid70_fpAddTest_b(55 downto 16);

    -- rightShiftStage0Idx1_uid223_alignmentShifter_uid71_fpAddTest(BITJOIN,222)@7
    rightShiftStage0Idx1_uid223_alignmentShifter_uid71_fpAddTest_q <= seMsb_to16_uid221_b & rightShiftStage0Idx1Rng16_uid222_alignmentShifter_uid71_fpAddTest_b;

    -- exp_aSig_uid21_fpAddTest(BITSELECT,20)@3
    exp_aSig_uid21_fpAddTest_in <= aSig_uid16_fpAddTest_BitJoin_for_q_q(62 downto 0);
    exp_aSig_uid21_fpAddTest_b <= exp_aSig_uid21_fpAddTest_in(62 downto 52);

    -- expAmExpB_uid60_fpAddTest(SUB,59)@3 + 1
    expAmExpB_uid60_fpAddTest_a <= STD_LOGIC_VECTOR("0" & exp_aSig_uid21_fpAddTest_b);
    expAmExpB_uid60_fpAddTest_b <= STD_LOGIC_VECTOR("0" & redist69_exp_bSig_uid35_fpAddTest_b_1_q);
    expAmExpB_uid60_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expAmExpB_uid60_fpAddTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expAmExpB_uid60_fpAddTest_o <= STD_LOGIC_VECTOR(UNSIGNED(expAmExpB_uid60_fpAddTest_a) - UNSIGNED(expAmExpB_uid60_fpAddTest_b));
        END IF;
    END PROCESS;
    expAmExpB_uid60_fpAddTest_q <= expAmExpB_uid60_fpAddTest_o(11 downto 0);

    -- rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select(BITSELECT,572)@4
    rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_in <= expAmExpB_uid60_fpAddTest_q(5 downto 0);
    rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_b <= rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_in(5 downto 4);
    rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_c <= rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_in(3 downto 2);
    rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_d <= rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_in(1 downto 0);

    -- redist7_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_b_3(DELAY,586)
    redist7_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_b_3 : dspba_delay
    GENERIC MAP ( width => 2, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_b, xout => redist7_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_b_3_q, clk => clk, aclr => areset );

    -- rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest(MUX,230)@7 + 1
    rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_s <= redist7_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_b_3_q;
    rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_s) IS
                WHEN "00" => rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_q <= oFracBREXC2_uid70_fpAddTest_b;
                WHEN "01" => rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_q <= rightShiftStage0Idx1_uid223_alignmentShifter_uid71_fpAddTest_q;
                WHEN "10" => rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_q <= rightShiftStage0Idx2_uid226_alignmentShifter_uid71_fpAddTest_q;
                WHEN "11" => rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_q <= rightShiftStage0Idx3_uid229_alignmentShifter_uid71_fpAddTest_q;
                WHEN OTHERS => rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist8_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_c_4(DELAY,587)
    redist8_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_c_4 : dspba_delay
    GENERIC MAP ( width => 2, depth => 4, reset_kind => "ASYNC" )
    PORT MAP ( xin => rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_c, xout => redist8_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_c_4_q, clk => clk, aclr => areset );

    -- rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest(MUX,241)@8 + 1
    rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_s <= redist8_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_c_4_q;
    rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_s) IS
                WHEN "00" => rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_q <= rightShiftStage0_uid231_alignmentShifter_uid71_fpAddTest_q;
                WHEN "01" => rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_q <= rightShiftStage1Idx1_uid234_alignmentShifter_uid71_fpAddTest_q;
                WHEN "10" => rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_q <= rightShiftStage1Idx2_uid237_alignmentShifter_uid71_fpAddTest_q;
                WHEN "11" => rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_q <= rightShiftStage1Idx3_uid240_alignmentShifter_uid71_fpAddTest_q;
                WHEN OTHERS => rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist9_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_d_5(DELAY,588)
    redist9_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_d_5 : dspba_delay
    GENERIC MAP ( width => 2, depth => 5, reset_kind => "ASYNC" )
    PORT MAP ( xin => rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_d, xout => redist9_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_d_5_q, clk => clk, aclr => areset );

    -- rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest(MUX,251)@9 + 1
    rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_s <= redist9_rightShiftStageSel5Dto4_uid230_alignmentShifter_uid71_fpAddTest_merged_bit_select_d_5_q;
    rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_s) IS
                WHEN "00" => rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q <= rightShiftStage1_uid242_alignmentShifter_uid71_fpAddTest_q;
                WHEN "01" => rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q <= rightShiftStage2Idx1_uid244_alignmentShifter_uid71_fpAddTest_q;
                WHEN "10" => rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q <= rightShiftStage2Idx2_uid247_alignmentShifter_uid71_fpAddTest_q;
                WHEN "11" => rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q <= rightShiftStage2Idx3_uid250_alignmentShifter_uid71_fpAddTest_q;
                WHEN OTHERS => rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a(BITSELECT,329)@10
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_b <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(0 downto 0);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_c <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(1 downto 1);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_d <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(2 downto 2);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_e <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(3 downto 3);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_f <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(4 downto 4);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_g <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(5 downto 5);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_h <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(6 downto 6);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_i <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(7 downto 7);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_j <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(8 downto 8);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_k <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(9 downto 9);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_l <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(10 downto 10);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_m <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(11 downto 11);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_n <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(12 downto 12);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_o <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(13 downto 13);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_p <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(14 downto 14);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_q <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(15 downto 15);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_r <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(16 downto 16);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_s <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(17 downto 17);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_t <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(18 downto 18);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_u <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(19 downto 19);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_v <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(20 downto 20);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_w <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(21 downto 21);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_x <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(22 downto 22);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_y <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(23 downto 23);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_z <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(24 downto 24);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_aa <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(25 downto 25);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_bb <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(26 downto 26);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_cc <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(27 downto 27);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_dd <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(28 downto 28);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ee <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(29 downto 29);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ff <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(30 downto 30);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_gg <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(31 downto 31);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_hh <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(32 downto 32);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ii <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(33 downto 33);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_jj <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(34 downto 34);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_kk <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(35 downto 35);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ll <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(36 downto 36);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_mm <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(37 downto 37);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_nn <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(38 downto 38);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_oo <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(39 downto 39);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_pp <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(40 downto 40);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_qq <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(41 downto 41);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_rr <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(42 downto 42);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ss <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(43 downto 43);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_tt <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(44 downto 44);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_uu <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(45 downto 45);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_vv <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(46 downto 46);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ww <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(47 downto 47);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_xx <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(48 downto 48);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_yy <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(49 downto 49);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_zz <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(50 downto 50);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_1 <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(51 downto 51);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_2 <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(52 downto 52);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_3 <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(53 downto 53);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_4 <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(54 downto 54);
    r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_5 <= rightShiftStage2_uid252_alignmentShifter_uid71_fpAddTest_q(55 downto 55);

    -- shiftedOut_uid220_alignmentShifter_uid71_fpAddTest(COMPARE,219)@4 + 1
    shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_a <= STD_LOGIC_VECTOR("00" & expAmExpB_uid60_fpAddTest_q);
    shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_b <= STD_LOGIC_VECTOR("00000000" & cAmA_uid76_fpAddTest_q);
    shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_o <= STD_LOGIC_VECTOR(UNSIGNED(shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_a) - UNSIGNED(shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_b));
        END IF;
    END PROCESS;
    shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n(0) <= not (shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_o(13));

    -- redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6(DELAY,613)
    redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6 : dspba_delay
    GENERIC MAP ( width => 1, depth => 5, reset_kind => "ASYNC" )
    PORT MAP ( xin => shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n, xout => redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p38(MUX,368)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p38_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p38_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p38_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p38_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p38_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_nn;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p38_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p38_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p37(MUX,367)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p37_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p37_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p37_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p37_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p37_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_mm;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p37_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p37_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p36(MUX,366)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p36_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p36_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p36_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p36_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p36_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ll;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p36_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p36_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p35(MUX,365)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p35_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p35_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p35_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p35_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p35_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_kk;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p35_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p35_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p34(MUX,364)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p34_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p34_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p34_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p34_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p34_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_jj;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p34_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p34_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p33(MUX,363)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p33_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p33_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p33_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p33_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p33_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ii;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p33_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p33_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p32(MUX,362)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p32_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p32_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p32_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p32_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p32_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_hh;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p32_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p32_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p31(MUX,361)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p31_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p31_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p31_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p31_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p31_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_gg;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p31_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p31_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p30(MUX,360)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p30_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p30_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p30_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p30_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p30_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ff;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p30_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p30_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p29(MUX,359)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p29_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p29_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p29_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p29_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p29_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ee;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p29_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p29_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p28(MUX,358)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p28_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p28_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p28_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p28_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p28_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_dd;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p28_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p28_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p27(MUX,357)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p27_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p27_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p27_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p27_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p27_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_cc;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p27_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p27_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p26(MUX,356)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p26_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p26_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p26_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p26_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p26_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_bb;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p26_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p26_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p25(MUX,355)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p25_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p25_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p25_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p25_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p25_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_aa;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p25_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p25_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p24(MUX,354)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p24_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p24_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p24_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p24_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p24_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_z;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p24_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p24_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p23(MUX,353)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p23_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p23_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p23_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p23_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p23_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_y;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p23_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p23_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p22(MUX,352)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p22_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p22_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p22_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p22_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p22_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_x;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p22_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p22_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p21(MUX,351)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p21_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p21_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p21_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p21_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p21_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_w;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p21_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p21_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p20(MUX,350)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p20_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p20_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p20_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p20_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p20_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_v;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p20_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p20_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p19(MUX,349)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p19_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p19_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p19_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p19_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p19_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_u;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p19_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p19_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p18(MUX,348)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p18_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p18_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p18_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p18_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p18_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_t;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p18_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p18_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p17(MUX,347)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p17_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p17_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p17_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p17_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p17_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_s;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p17_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p17_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p16(MUX,346)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p16_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p16_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p16_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p16_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p16_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_r;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p16_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p16_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p15(MUX,345)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p15_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p15_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p15_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p15_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p15_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_q;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p15_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p15_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p14(MUX,344)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p14_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p14_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p14_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p14_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p14_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_p;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p14_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p14_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p13(MUX,343)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p13_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p13_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p13_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p13_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p13_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_o;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p13_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p13_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p12(MUX,342)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p12_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p12_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p12_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p12_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p12_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_n;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p12_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p12_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p11(MUX,341)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p11_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p11_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p11_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p11_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p11_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_m;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p11_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p11_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p10(MUX,340)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p10_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p10_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p10_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p10_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p10_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_l;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p10_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p10_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p9(MUX,339)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p9_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p9_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p9_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p9_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p9_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_k;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p9_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p9_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p8(MUX,338)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p8_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p8_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p8_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p8_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p8_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_j;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p8_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p8_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p7(MUX,337)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p7_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p7_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p7_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p7_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p7_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_i;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p7_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p7_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p6(MUX,336)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p6_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p6_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p6_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p6_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p6_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_h;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p6_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p6_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p5(MUX,335)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p5_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p5_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p5_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p5_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p5_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_g;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p5_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p5_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p4(MUX,334)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p4_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p4_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p4_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p4_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_f;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p4_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p4_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p3(MUX,333)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p3_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p3_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p3_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p3_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p3_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_e;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p3_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p3_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p2(MUX,332)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p2_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p2_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p2_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p2_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_d;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p2_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p2_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p1(MUX,331)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p1_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p1_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p1_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p1_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p1_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_c;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p1_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p1_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p0(MUX,330)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p0_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p0_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p0_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p0_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p0_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_b;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p0_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p0_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- fracAddResult_uid72_fpAddTest_BitSelect_for_b_BitJoin_for_b(BITJOIN,437)@11
    fracAddResult_uid72_fpAddTest_BitSelect_for_b_BitJoin_for_b_q <= r_uid255_alignmentShifter_uid71_fpAddTest_p38_q & r_uid255_alignmentShifter_uid71_fpAddTest_p37_q & r_uid255_alignmentShifter_uid71_fpAddTest_p36_q & r_uid255_alignmentShifter_uid71_fpAddTest_p35_q & r_uid255_alignmentShifter_uid71_fpAddTest_p34_q & r_uid255_alignmentShifter_uid71_fpAddTest_p33_q & r_uid255_alignmentShifter_uid71_fpAddTest_p32_q & r_uid255_alignmentShifter_uid71_fpAddTest_p31_q & r_uid255_alignmentShifter_uid71_fpAddTest_p30_q & r_uid255_alignmentShifter_uid71_fpAddTest_p29_q & r_uid255_alignmentShifter_uid71_fpAddTest_p28_q & r_uid255_alignmentShifter_uid71_fpAddTest_p27_q & r_uid255_alignmentShifter_uid71_fpAddTest_p26_q & r_uid255_alignmentShifter_uid71_fpAddTest_p25_q & r_uid255_alignmentShifter_uid71_fpAddTest_p24_q & r_uid255_alignmentShifter_uid71_fpAddTest_p23_q & r_uid255_alignmentShifter_uid71_fpAddTest_p22_q & r_uid255_alignmentShifter_uid71_fpAddTest_p21_q & r_uid255_alignmentShifter_uid71_fpAddTest_p20_q & r_uid255_alignmentShifter_uid71_fpAddTest_p19_q & r_uid255_alignmentShifter_uid71_fpAddTest_p18_q & r_uid255_alignmentShifter_uid71_fpAddTest_p17_q & r_uid255_alignmentShifter_uid71_fpAddTest_p16_q & r_uid255_alignmentShifter_uid71_fpAddTest_p15_q & r_uid255_alignmentShifter_uid71_fpAddTest_p14_q & r_uid255_alignmentShifter_uid71_fpAddTest_p13_q & r_uid255_alignmentShifter_uid71_fpAddTest_p12_q & r_uid255_alignmentShifter_uid71_fpAddTest_p11_q & r_uid255_alignmentShifter_uid71_fpAddTest_p10_q & r_uid255_alignmentShifter_uid71_fpAddTest_p9_q & r_uid255_alignmentShifter_uid71_fpAddTest_p8_q & r_uid255_alignmentShifter_uid71_fpAddTest_p7_q & r_uid255_alignmentShifter_uid71_fpAddTest_p6_q & r_uid255_alignmentShifter_uid71_fpAddTest_p5_q & r_uid255_alignmentShifter_uid71_fpAddTest_p4_q & r_uid255_alignmentShifter_uid71_fpAddTest_p3_q & r_uid255_alignmentShifter_uid71_fpAddTest_p2_q & r_uid255_alignmentShifter_uid71_fpAddTest_p1_q & r_uid255_alignmentShifter_uid71_fpAddTest_p0_q;

    -- redist27_aSig_uid16_fpAddTest_p0_q_9(DELAY,606)
    redist27_aSig_uid16_fpAddTest_p0_q_9 : dspba_delay
    GENERIC MAP ( width => 52, depth => 8, reset_kind => "ASYNC" )
    PORT MAP ( xin => aSig_uid16_fpAddTest_p0_q, xout => redist27_aSig_uid16_fpAddTest_p0_q_9_q, clk => clk, aclr => areset );

    -- fracAddResult_uid72_fpAddTest_BitSelect_for_a_tessel0_2_merged_bit_select(BITSELECT,578)@11
    fracAddResult_uid72_fpAddTest_BitSelect_for_a_tessel0_2_merged_bit_select_b <= STD_LOGIC_VECTOR(redist27_aSig_uid16_fpAddTest_p0_q_9_q(36 downto 0));
    fracAddResult_uid72_fpAddTest_BitSelect_for_a_tessel0_2_merged_bit_select_c <= STD_LOGIC_VECTOR(redist27_aSig_uid16_fpAddTest_p0_q_9_q(51 downto 37));

    -- fracAddResult_uid72_fpAddTest_BitSelect_for_a_BitJoin_for_b(BITJOIN,392)@11
    fracAddResult_uid72_fpAddTest_BitSelect_for_a_BitJoin_for_b_q <= fracAddResult_uid72_fpAddTest_BitSelect_for_a_tessel0_2_merged_bit_select_b & GND_q & GND_q;

    -- fracAddResult_uid72_fpAddTest_p1_of_2(ADD,319)@11 + 1
    fracAddResult_uid72_fpAddTest_p1_of_2_a <= STD_LOGIC_VECTOR("0" & fracAddResult_uid72_fpAddTest_BitSelect_for_a_BitJoin_for_b_q);
    fracAddResult_uid72_fpAddTest_p1_of_2_b <= STD_LOGIC_VECTOR("0" & fracAddResult_uid72_fpAddTest_BitSelect_for_b_BitJoin_for_b_q);
    fracAddResult_uid72_fpAddTest_p1_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            fracAddResult_uid72_fpAddTest_p1_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            fracAddResult_uid72_fpAddTest_p1_of_2_o <= STD_LOGIC_VECTOR(UNSIGNED(fracAddResult_uid72_fpAddTest_p1_of_2_a) + UNSIGNED(fracAddResult_uid72_fpAddTest_p1_of_2_b));
        END IF;
    END PROCESS;
    fracAddResult_uid72_fpAddTest_p1_of_2_c(0) <= fracAddResult_uid72_fpAddTest_p1_of_2_o(39);
    fracAddResult_uid72_fpAddTest_p1_of_2_q <= fracAddResult_uid72_fpAddTest_p1_of_2_o(38 downto 0);

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p55(MUX,385)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p55_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p55_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p55_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p55_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p55_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_5;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p55_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p55_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist10_r_uid255_alignmentShifter_uid71_fpAddTest_p55_q_2(DELAY,589)
    redist10_r_uid255_alignmentShifter_uid71_fpAddTest_p55_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p55_q, xout => redist10_r_uid255_alignmentShifter_uid71_fpAddTest_p55_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p54(MUX,384)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p54_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p54_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p54_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p54_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p54_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_4;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p54_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p54_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist11_r_uid255_alignmentShifter_uid71_fpAddTest_p54_q_2(DELAY,590)
    redist11_r_uid255_alignmentShifter_uid71_fpAddTest_p54_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p54_q, xout => redist11_r_uid255_alignmentShifter_uid71_fpAddTest_p54_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p53(MUX,383)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p53_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p53_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p53_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p53_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p53_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_3;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p53_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p53_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist12_r_uid255_alignmentShifter_uid71_fpAddTest_p53_q_2(DELAY,591)
    redist12_r_uid255_alignmentShifter_uid71_fpAddTest_p53_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p53_q, xout => redist12_r_uid255_alignmentShifter_uid71_fpAddTest_p53_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p52(MUX,382)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p52_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p52_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p52_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p52_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p52_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_2;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p52_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p52_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist13_r_uid255_alignmentShifter_uid71_fpAddTest_p52_q_2(DELAY,592)
    redist13_r_uid255_alignmentShifter_uid71_fpAddTest_p52_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p52_q, xout => redist13_r_uid255_alignmentShifter_uid71_fpAddTest_p52_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p51(MUX,381)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p51_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p51_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p51_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p51_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p51_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_1;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p51_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p51_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist14_r_uid255_alignmentShifter_uid71_fpAddTest_p51_q_2(DELAY,593)
    redist14_r_uid255_alignmentShifter_uid71_fpAddTest_p51_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p51_q, xout => redist14_r_uid255_alignmentShifter_uid71_fpAddTest_p51_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p50(MUX,380)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p50_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p50_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p50_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p50_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p50_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_zz;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p50_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p50_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist15_r_uid255_alignmentShifter_uid71_fpAddTest_p50_q_2(DELAY,594)
    redist15_r_uid255_alignmentShifter_uid71_fpAddTest_p50_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p50_q, xout => redist15_r_uid255_alignmentShifter_uid71_fpAddTest_p50_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p49(MUX,379)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p49_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p49_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p49_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p49_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p49_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_yy;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p49_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p49_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist16_r_uid255_alignmentShifter_uid71_fpAddTest_p49_q_2(DELAY,595)
    redist16_r_uid255_alignmentShifter_uid71_fpAddTest_p49_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p49_q, xout => redist16_r_uid255_alignmentShifter_uid71_fpAddTest_p49_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p48(MUX,378)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p48_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p48_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p48_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p48_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p48_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_xx;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p48_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p48_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist17_r_uid255_alignmentShifter_uid71_fpAddTest_p48_q_2(DELAY,596)
    redist17_r_uid255_alignmentShifter_uid71_fpAddTest_p48_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p48_q, xout => redist17_r_uid255_alignmentShifter_uid71_fpAddTest_p48_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p47(MUX,377)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p47_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p47_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p47_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p47_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p47_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ww;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p47_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p47_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist18_r_uid255_alignmentShifter_uid71_fpAddTest_p47_q_2(DELAY,597)
    redist18_r_uid255_alignmentShifter_uid71_fpAddTest_p47_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p47_q, xout => redist18_r_uid255_alignmentShifter_uid71_fpAddTest_p47_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p46(MUX,376)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p46_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p46_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p46_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p46_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p46_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_vv;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p46_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p46_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist19_r_uid255_alignmentShifter_uid71_fpAddTest_p46_q_2(DELAY,598)
    redist19_r_uid255_alignmentShifter_uid71_fpAddTest_p46_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p46_q, xout => redist19_r_uid255_alignmentShifter_uid71_fpAddTest_p46_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p45(MUX,375)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p45_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p45_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p45_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p45_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p45_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_uu;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p45_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p45_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist20_r_uid255_alignmentShifter_uid71_fpAddTest_p45_q_2(DELAY,599)
    redist20_r_uid255_alignmentShifter_uid71_fpAddTest_p45_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p45_q, xout => redist20_r_uid255_alignmentShifter_uid71_fpAddTest_p45_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p44(MUX,374)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p44_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p44_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p44_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p44_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p44_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_tt;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p44_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p44_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist21_r_uid255_alignmentShifter_uid71_fpAddTest_p44_q_2(DELAY,600)
    redist21_r_uid255_alignmentShifter_uid71_fpAddTest_p44_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p44_q, xout => redist21_r_uid255_alignmentShifter_uid71_fpAddTest_p44_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p43(MUX,373)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p43_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p43_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p43_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p43_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p43_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_ss;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p43_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p43_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist22_r_uid255_alignmentShifter_uid71_fpAddTest_p43_q_2(DELAY,601)
    redist22_r_uid255_alignmentShifter_uid71_fpAddTest_p43_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p43_q, xout => redist22_r_uid255_alignmentShifter_uid71_fpAddTest_p43_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p42(MUX,372)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p42_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p42_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p42_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p42_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p42_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_rr;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p42_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p42_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist23_r_uid255_alignmentShifter_uid71_fpAddTest_p42_q_2(DELAY,602)
    redist23_r_uid255_alignmentShifter_uid71_fpAddTest_p42_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p42_q, xout => redist23_r_uid255_alignmentShifter_uid71_fpAddTest_p42_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p41(MUX,371)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p41_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p41_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p41_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p41_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p41_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_qq;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p41_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p41_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist24_r_uid255_alignmentShifter_uid71_fpAddTest_p41_q_2(DELAY,603)
    redist24_r_uid255_alignmentShifter_uid71_fpAddTest_p41_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p41_q, xout => redist24_r_uid255_alignmentShifter_uid71_fpAddTest_p41_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p40(MUX,370)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p40_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p40_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p40_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p40_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p40_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_pp;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p40_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p40_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist25_r_uid255_alignmentShifter_uid71_fpAddTest_p40_q_2(DELAY,604)
    redist25_r_uid255_alignmentShifter_uid71_fpAddTest_p40_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p40_q, xout => redist25_r_uid255_alignmentShifter_uid71_fpAddTest_p40_q_2_q, clk => clk, aclr => areset );

    -- r_uid255_alignmentShifter_uid71_fpAddTest_p39(MUX,369)@10 + 1
    r_uid255_alignmentShifter_uid71_fpAddTest_p39_s <= redist34_shiftedOut_uid220_alignmentShifter_uid71_fpAddTest_n_6_q;
    r_uid255_alignmentShifter_uid71_fpAddTest_p39_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            r_uid255_alignmentShifter_uid71_fpAddTest_p39_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (r_uid255_alignmentShifter_uid71_fpAddTest_p39_s) IS
                WHEN "0" => r_uid255_alignmentShifter_uid71_fpAddTest_p39_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_a_oo;
                WHEN "1" => r_uid255_alignmentShifter_uid71_fpAddTest_p39_q <= r_uid255_alignmentShifter_uid71_fpAddTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => r_uid255_alignmentShifter_uid71_fpAddTest_p39_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist26_r_uid255_alignmentShifter_uid71_fpAddTest_p39_q_2(DELAY,605)
    redist26_r_uid255_alignmentShifter_uid71_fpAddTest_p39_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid255_alignmentShifter_uid71_fpAddTest_p39_q, xout => redist26_r_uid255_alignmentShifter_uid71_fpAddTest_p39_q_2_q, clk => clk, aclr => areset );

    -- fracAddResult_uid72_fpAddTest_BitSelect_for_b_BitJoin_for_c(BITJOIN,456)@12
    fracAddResult_uid72_fpAddTest_BitSelect_for_b_BitJoin_for_c_q <= redist10_r_uid255_alignmentShifter_uid71_fpAddTest_p55_q_2_q & redist10_r_uid255_alignmentShifter_uid71_fpAddTest_p55_q_2_q & redist11_r_uid255_alignmentShifter_uid71_fpAddTest_p54_q_2_q & redist12_r_uid255_alignmentShifter_uid71_fpAddTest_p53_q_2_q & redist13_r_uid255_alignmentShifter_uid71_fpAddTest_p52_q_2_q & redist14_r_uid255_alignmentShifter_uid71_fpAddTest_p51_q_2_q & redist15_r_uid255_alignmentShifter_uid71_fpAddTest_p50_q_2_q & redist16_r_uid255_alignmentShifter_uid71_fpAddTest_p49_q_2_q & redist17_r_uid255_alignmentShifter_uid71_fpAddTest_p48_q_2_q & redist18_r_uid255_alignmentShifter_uid71_fpAddTest_p47_q_2_q & redist19_r_uid255_alignmentShifter_uid71_fpAddTest_p46_q_2_q & redist20_r_uid255_alignmentShifter_uid71_fpAddTest_p45_q_2_q & redist21_r_uid255_alignmentShifter_uid71_fpAddTest_p44_q_2_q & redist22_r_uid255_alignmentShifter_uid71_fpAddTest_p43_q_2_q & redist23_r_uid255_alignmentShifter_uid71_fpAddTest_p42_q_2_q & redist24_r_uid255_alignmentShifter_uid71_fpAddTest_p41_q_2_q & redist25_r_uid255_alignmentShifter_uid71_fpAddTest_p40_q_2_q & redist26_r_uid255_alignmentShifter_uid71_fpAddTest_p39_q_2_q;

    -- redist0_fracAddResult_uid72_fpAddTest_BitSelect_for_a_tessel0_2_merged_bit_select_c_1(DELAY,579)
    redist0_fracAddResult_uid72_fpAddTest_BitSelect_for_a_tessel0_2_merged_bit_select_c_1 : dspba_delay
    GENERIC MAP ( width => 15, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => fracAddResult_uid72_fpAddTest_BitSelect_for_a_tessel0_2_merged_bit_select_c, xout => redist0_fracAddResult_uid72_fpAddTest_BitSelect_for_a_tessel0_2_merged_bit_select_c_1_q, clk => clk, aclr => areset );

    -- fracAddResult_uid72_fpAddTest_BitSelect_for_a_BitJoin_for_c(BITJOIN,397)@12
    fracAddResult_uid72_fpAddTest_BitSelect_for_a_BitJoin_for_c_q <= GND_q & GND_q & VCC_q & redist0_fracAddResult_uid72_fpAddTest_BitSelect_for_a_tessel0_2_merged_bit_select_c_1_q;

    -- fracAddResult_uid72_fpAddTest_p2_of_2(ADD,320)@12 + 1
    fracAddResult_uid72_fpAddTest_p2_of_2_cin <= fracAddResult_uid72_fpAddTest_p1_of_2_c;
    fracAddResult_uid72_fpAddTest_p2_of_2_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((18 downto 18 => fracAddResult_uid72_fpAddTest_BitSelect_for_a_BitJoin_for_c_q(17)) & fracAddResult_uid72_fpAddTest_BitSelect_for_a_BitJoin_for_c_q) & '1');
    fracAddResult_uid72_fpAddTest_p2_of_2_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((18 downto 18 => fracAddResult_uid72_fpAddTest_BitSelect_for_b_BitJoin_for_c_q(17)) & fracAddResult_uid72_fpAddTest_BitSelect_for_b_BitJoin_for_c_q) & fracAddResult_uid72_fpAddTest_p2_of_2_cin(0));
    fracAddResult_uid72_fpAddTest_p2_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            fracAddResult_uid72_fpAddTest_p2_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            fracAddResult_uid72_fpAddTest_p2_of_2_o <= STD_LOGIC_VECTOR(SIGNED(fracAddResult_uid72_fpAddTest_p2_of_2_a) + SIGNED(fracAddResult_uid72_fpAddTest_p2_of_2_b));
        END IF;
    END PROCESS;
    fracAddResult_uid72_fpAddTest_p2_of_2_q <= fracAddResult_uid72_fpAddTest_p2_of_2_o(18 downto 1);

    -- redist28_fracAddResult_uid72_fpAddTest_p1_of_2_q_1(DELAY,607)
    redist28_fracAddResult_uid72_fpAddTest_p1_of_2_q_1 : dspba_delay
    GENERIC MAP ( width => 39, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => fracAddResult_uid72_fpAddTest_p1_of_2_q, xout => redist28_fracAddResult_uid72_fpAddTest_p1_of_2_q_1_q, clk => clk, aclr => areset );

    -- fracAddResult_uid72_fpAddTest_BitJoin_for_q(BITJOIN,321)@13
    fracAddResult_uid72_fpAddTest_BitJoin_for_q_q <= fracAddResult_uid72_fpAddTest_p2_of_2_q & redist28_fracAddResult_uid72_fpAddTest_p1_of_2_q_1_q;

    -- fracAddResultNoSignExt_uid73_fpAddTest(BITSELECT,72)@13
    fracAddResultNoSignExt_uid73_fpAddTest_in <= fracAddResult_uid72_fpAddTest_BitJoin_for_q_q(55 downto 0);
    fracAddResultNoSignExt_uid73_fpAddTest_b <= fracAddResultNoSignExt_uid73_fpAddTest_in(55 downto 0);

    -- rVStage_uid121_lzCountVal_uid74_fpAddTest(BITSELECT,120)@13
    rVStage_uid121_lzCountVal_uid74_fpAddTest_b <= fracAddResultNoSignExt_uid73_fpAddTest_b(55 downto 24);

    -- vCount_uid122_lzCountVal_uid74_fpAddTest(LOGICAL,121)@13 + 1
    vCount_uid122_lzCountVal_uid74_fpAddTest_qi <= "1" WHEN rVStage_uid121_lzCountVal_uid74_fpAddTest_b = zs_uid120_lzCountVal_uid74_fpAddTest_q ELSE "0";
    vCount_uid122_lzCountVal_uid74_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid122_lzCountVal_uid74_fpAddTest_qi, xout => vCount_uid122_lzCountVal_uid74_fpAddTest_q, clk => clk, aclr => areset );

    -- redist44_vCount_uid122_lzCountVal_uid74_fpAddTest_q_7(DELAY,623)
    redist44_vCount_uid122_lzCountVal_uid74_fpAddTest_q_7 : dspba_delay
    GENERIC MAP ( width => 1, depth => 6, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid122_lzCountVal_uid74_fpAddTest_q, xout => redist44_vCount_uid122_lzCountVal_uid74_fpAddTest_q_7_q, clk => clk, aclr => areset );

    -- zs_uid128_lzCountVal_uid74_fpAddTest(CONSTANT,127)
    zs_uid128_lzCountVal_uid74_fpAddTest_q <= "0000000000000000";

    -- redist54_fracAddResultNoSignExt_uid73_fpAddTest_b_1(DELAY,633)
    redist54_fracAddResultNoSignExt_uid73_fpAddTest_b_1 : dspba_delay
    GENERIC MAP ( width => 56, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => fracAddResultNoSignExt_uid73_fpAddTest_b, xout => redist54_fracAddResultNoSignExt_uid73_fpAddTest_b_1_q, clk => clk, aclr => areset );

    -- vStage_uid124_lzCountVal_uid74_fpAddTest(BITSELECT,123)@14
    vStage_uid124_lzCountVal_uid74_fpAddTest_in <= redist54_fracAddResultNoSignExt_uid73_fpAddTest_b_1_q(23 downto 0);
    vStage_uid124_lzCountVal_uid74_fpAddTest_b <= vStage_uid124_lzCountVal_uid74_fpAddTest_in(23 downto 0);

    -- mO_uid123_lzCountVal_uid74_fpAddTest(CONSTANT,122)
    mO_uid123_lzCountVal_uid74_fpAddTest_q <= "11111111";

    -- cStage_uid125_lzCountVal_uid74_fpAddTest(BITJOIN,124)@14
    cStage_uid125_lzCountVal_uid74_fpAddTest_q <= vStage_uid124_lzCountVal_uid74_fpAddTest_b & mO_uid123_lzCountVal_uid74_fpAddTest_q;

    -- redist45_rVStage_uid121_lzCountVal_uid74_fpAddTest_b_1(DELAY,624)
    redist45_rVStage_uid121_lzCountVal_uid74_fpAddTest_b_1 : dspba_delay
    GENERIC MAP ( width => 32, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => rVStage_uid121_lzCountVal_uid74_fpAddTest_b, xout => redist45_rVStage_uid121_lzCountVal_uid74_fpAddTest_b_1_q, clk => clk, aclr => areset );

    -- vStagei_uid127_lzCountVal_uid74_fpAddTest(MUX,126)@14 + 1
    vStagei_uid127_lzCountVal_uid74_fpAddTest_s <= vCount_uid122_lzCountVal_uid74_fpAddTest_q;
    vStagei_uid127_lzCountVal_uid74_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid127_lzCountVal_uid74_fpAddTest_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid127_lzCountVal_uid74_fpAddTest_s) IS
                WHEN "0" => vStagei_uid127_lzCountVal_uid74_fpAddTest_q <= redist45_rVStage_uid121_lzCountVal_uid74_fpAddTest_b_1_q;
                WHEN "1" => vStagei_uid127_lzCountVal_uid74_fpAddTest_q <= cStage_uid125_lzCountVal_uid74_fpAddTest_q;
                WHEN OTHERS => vStagei_uid127_lzCountVal_uid74_fpAddTest_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select(BITSELECT,573)@15
    rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_b <= vStagei_uid127_lzCountVal_uid74_fpAddTest_q(31 downto 16);
    rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_c <= vStagei_uid127_lzCountVal_uid74_fpAddTest_q(15 downto 0);

    -- vCount_uid130_lzCountVal_uid74_fpAddTest(LOGICAL,129)@15 + 1
    vCount_uid130_lzCountVal_uid74_fpAddTest_qi <= "1" WHEN rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_b = zs_uid128_lzCountVal_uid74_fpAddTest_q ELSE "0";
    vCount_uid130_lzCountVal_uid74_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid130_lzCountVal_uid74_fpAddTest_qi, xout => vCount_uid130_lzCountVal_uid74_fpAddTest_q, clk => clk, aclr => areset );

    -- redist42_vCount_uid130_lzCountVal_uid74_fpAddTest_q_5(DELAY,621)
    redist42_vCount_uid130_lzCountVal_uid74_fpAddTest_q_5 : dspba_delay
    GENERIC MAP ( width => 1, depth => 4, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid130_lzCountVal_uid74_fpAddTest_q, xout => redist42_vCount_uid130_lzCountVal_uid74_fpAddTest_q_5_q, clk => clk, aclr => areset );

    -- zs_uid134_lzCountVal_uid74_fpAddTest(CONSTANT,133)
    zs_uid134_lzCountVal_uid74_fpAddTest_q <= "00000000";

    -- redist6_rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_c_1(DELAY,585)
    redist6_rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_c_1 : dspba_delay
    GENERIC MAP ( width => 16, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_c, xout => redist6_rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_c_1_q, clk => clk, aclr => areset );

    -- redist5_rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_b_1(DELAY,584)
    redist5_rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_b_1 : dspba_delay
    GENERIC MAP ( width => 16, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_b, xout => redist5_rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_b_1_q, clk => clk, aclr => areset );

    -- vStagei_uid133_lzCountVal_uid74_fpAddTest(MUX,132)@16 + 1
    vStagei_uid133_lzCountVal_uid74_fpAddTest_s <= vCount_uid130_lzCountVal_uid74_fpAddTest_q;
    vStagei_uid133_lzCountVal_uid74_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid133_lzCountVal_uid74_fpAddTest_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid133_lzCountVal_uid74_fpAddTest_s) IS
                WHEN "0" => vStagei_uid133_lzCountVal_uid74_fpAddTest_q <= redist5_rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_b_1_q;
                WHEN "1" => vStagei_uid133_lzCountVal_uid74_fpAddTest_q <= redist6_rVStage_uid129_lzCountVal_uid74_fpAddTest_merged_bit_select_c_1_q;
                WHEN OTHERS => vStagei_uid133_lzCountVal_uid74_fpAddTest_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select(BITSELECT,574)@17
    rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_b <= vStagei_uid133_lzCountVal_uid74_fpAddTest_q(15 downto 8);
    rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_c <= vStagei_uid133_lzCountVal_uid74_fpAddTest_q(7 downto 0);

    -- vCount_uid136_lzCountVal_uid74_fpAddTest(LOGICAL,135)@17 + 1
    vCount_uid136_lzCountVal_uid74_fpAddTest_qi <= "1" WHEN rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_b = zs_uid134_lzCountVal_uid74_fpAddTest_q ELSE "0";
    vCount_uid136_lzCountVal_uid74_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid136_lzCountVal_uid74_fpAddTest_qi, xout => vCount_uid136_lzCountVal_uid74_fpAddTest_q, clk => clk, aclr => areset );

    -- redist41_vCount_uid136_lzCountVal_uid74_fpAddTest_q_3(DELAY,620)
    redist41_vCount_uid136_lzCountVal_uid74_fpAddTest_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid136_lzCountVal_uid74_fpAddTest_q, xout => redist41_vCount_uid136_lzCountVal_uid74_fpAddTest_q_3_q, clk => clk, aclr => areset );

    -- zs_uid140_lzCountVal_uid74_fpAddTest(CONSTANT,139)
    zs_uid140_lzCountVal_uid74_fpAddTest_q <= "0000";

    -- redist4_rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_c_1(DELAY,583)
    redist4_rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_c_1 : dspba_delay
    GENERIC MAP ( width => 8, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_c, xout => redist4_rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_c_1_q, clk => clk, aclr => areset );

    -- redist3_rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_b_1(DELAY,582)
    redist3_rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_b_1 : dspba_delay
    GENERIC MAP ( width => 8, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_b, xout => redist3_rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_b_1_q, clk => clk, aclr => areset );

    -- vStagei_uid139_lzCountVal_uid74_fpAddTest(MUX,138)@18 + 1
    vStagei_uid139_lzCountVal_uid74_fpAddTest_s <= vCount_uid136_lzCountVal_uid74_fpAddTest_q;
    vStagei_uid139_lzCountVal_uid74_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid139_lzCountVal_uid74_fpAddTest_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid139_lzCountVal_uid74_fpAddTest_s) IS
                WHEN "0" => vStagei_uid139_lzCountVal_uid74_fpAddTest_q <= redist3_rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_b_1_q;
                WHEN "1" => vStagei_uid139_lzCountVal_uid74_fpAddTest_q <= redist4_rVStage_uid135_lzCountVal_uid74_fpAddTest_merged_bit_select_c_1_q;
                WHEN OTHERS => vStagei_uid139_lzCountVal_uid74_fpAddTest_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- rVStage_uid141_lzCountVal_uid74_fpAddTest_merged_bit_select(BITSELECT,575)@19
    rVStage_uid141_lzCountVal_uid74_fpAddTest_merged_bit_select_b <= vStagei_uid139_lzCountVal_uid74_fpAddTest_q(7 downto 4);
    rVStage_uid141_lzCountVal_uid74_fpAddTest_merged_bit_select_c <= vStagei_uid139_lzCountVal_uid74_fpAddTest_q(3 downto 0);

    -- vCount_uid142_lzCountVal_uid74_fpAddTest(LOGICAL,141)@19
    vCount_uid142_lzCountVal_uid74_fpAddTest_q <= "1" WHEN rVStage_uid141_lzCountVal_uid74_fpAddTest_merged_bit_select_b = zs_uid140_lzCountVal_uid74_fpAddTest_q ELSE "0";

    -- redist40_vCount_uid142_lzCountVal_uid74_fpAddTest_q_1(DELAY,619)
    redist40_vCount_uid142_lzCountVal_uid74_fpAddTest_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid142_lzCountVal_uid74_fpAddTest_q, xout => redist40_vCount_uid142_lzCountVal_uid74_fpAddTest_q_1_q, clk => clk, aclr => areset );

    -- zs_uid146_lzCountVal_uid74_fpAddTest(CONSTANT,145)
    zs_uid146_lzCountVal_uid74_fpAddTest_q <= "00";

    -- vStagei_uid145_lzCountVal_uid74_fpAddTest(MUX,144)@19 + 1
    vStagei_uid145_lzCountVal_uid74_fpAddTest_s <= vCount_uid142_lzCountVal_uid74_fpAddTest_q;
    vStagei_uid145_lzCountVal_uid74_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid145_lzCountVal_uid74_fpAddTest_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid145_lzCountVal_uid74_fpAddTest_s) IS
                WHEN "0" => vStagei_uid145_lzCountVal_uid74_fpAddTest_q <= rVStage_uid141_lzCountVal_uid74_fpAddTest_merged_bit_select_b;
                WHEN "1" => vStagei_uid145_lzCountVal_uid74_fpAddTest_q <= rVStage_uid141_lzCountVal_uid74_fpAddTest_merged_bit_select_c;
                WHEN OTHERS => vStagei_uid145_lzCountVal_uid74_fpAddTest_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- rVStage_uid147_lzCountVal_uid74_fpAddTest_merged_bit_select(BITSELECT,576)@20
    rVStage_uid147_lzCountVal_uid74_fpAddTest_merged_bit_select_b <= vStagei_uid145_lzCountVal_uid74_fpAddTest_q(3 downto 2);
    rVStage_uid147_lzCountVal_uid74_fpAddTest_merged_bit_select_c <= vStagei_uid145_lzCountVal_uid74_fpAddTest_q(1 downto 0);

    -- vCount_uid148_lzCountVal_uid74_fpAddTest(LOGICAL,147)@20
    vCount_uid148_lzCountVal_uid74_fpAddTest_q <= "1" WHEN rVStage_uid147_lzCountVal_uid74_fpAddTest_merged_bit_select_b = zs_uid146_lzCountVal_uid74_fpAddTest_q ELSE "0";

    -- vStagei_uid151_lzCountVal_uid74_fpAddTest(MUX,150)@20
    vStagei_uid151_lzCountVal_uid74_fpAddTest_s <= vCount_uid148_lzCountVal_uid74_fpAddTest_q;
    vStagei_uid151_lzCountVal_uid74_fpAddTest_combproc: PROCESS (vStagei_uid151_lzCountVal_uid74_fpAddTest_s, rVStage_uid147_lzCountVal_uid74_fpAddTest_merged_bit_select_b, rVStage_uid147_lzCountVal_uid74_fpAddTest_merged_bit_select_c)
    BEGIN
        CASE (vStagei_uid151_lzCountVal_uid74_fpAddTest_s) IS
            WHEN "0" => vStagei_uid151_lzCountVal_uid74_fpAddTest_q <= rVStage_uid147_lzCountVal_uid74_fpAddTest_merged_bit_select_b;
            WHEN "1" => vStagei_uid151_lzCountVal_uid74_fpAddTest_q <= rVStage_uid147_lzCountVal_uid74_fpAddTest_merged_bit_select_c;
            WHEN OTHERS => vStagei_uid151_lzCountVal_uid74_fpAddTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- rVStage_uid153_lzCountVal_uid74_fpAddTest(BITSELECT,152)@20
    rVStage_uid153_lzCountVal_uid74_fpAddTest_b <= vStagei_uid151_lzCountVal_uid74_fpAddTest_q(1 downto 1);

    -- vCount_uid154_lzCountVal_uid74_fpAddTest(LOGICAL,153)@20
    vCount_uid154_lzCountVal_uid74_fpAddTest_q <= "1" WHEN rVStage_uid153_lzCountVal_uid74_fpAddTest_b = GND_q ELSE "0";

    -- r_uid155_lzCountVal_uid74_fpAddTest(BITJOIN,154)@20
    r_uid155_lzCountVal_uid74_fpAddTest_q <= redist44_vCount_uid122_lzCountVal_uid74_fpAddTest_q_7_q & redist42_vCount_uid130_lzCountVal_uid74_fpAddTest_q_5_q & redist41_vCount_uid136_lzCountVal_uid74_fpAddTest_q_3_q & redist40_vCount_uid142_lzCountVal_uid74_fpAddTest_q_1_q & vCount_uid148_lzCountVal_uid74_fpAddTest_q & vCount_uid154_lzCountVal_uid74_fpAddTest_q;

    -- redist39_r_uid155_lzCountVal_uid74_fpAddTest_q_1(DELAY,618)
    redist39_r_uid155_lzCountVal_uid74_fpAddTest_q_1 : dspba_delay
    GENERIC MAP ( width => 6, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => r_uid155_lzCountVal_uid74_fpAddTest_q, xout => redist39_r_uid155_lzCountVal_uid74_fpAddTest_q_1_q, clk => clk, aclr => areset );

    -- aMinusA_uid77_fpAddTest(LOGICAL,76)@21
    aMinusA_uid77_fpAddTest_q <= "1" WHEN redist39_r_uid155_lzCountVal_uid74_fpAddTest_q_1_q = cAmA_uid76_fpAddTest_q ELSE "0";

    -- invAMinusA_uid99_fpAddTest(LOGICAL,98)@21
    invAMinusA_uid99_fpAddTest_q <= not (aMinusA_uid77_fpAddTest_q);

    -- redist61_sigA_uid50_fpAddTest_b_18(DELAY,640)
    redist61_sigA_uid50_fpAddTest_b_18 : dspba_delay
    GENERIC MAP ( width => 1, depth => 17, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist60_sigA_uid50_fpAddTest_b_1_q, xout => redist61_sigA_uid50_fpAddTest_b_18_q, clk => clk, aclr => areset );

    -- cstAllOWE_uid18_fpAddTest(CONSTANT,17)
    cstAllOWE_uid18_fpAddTest_q <= "11111111111";

    -- expXIsMax_uid38_fpAddTest(LOGICAL,37)@3 + 1
    expXIsMax_uid38_fpAddTest_qi <= "1" WHEN redist69_exp_bSig_uid35_fpAddTest_b_1_q = cstAllOWE_uid18_fpAddTest_q ELSE "0";
    expXIsMax_uid38_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => expXIsMax_uid38_fpAddTest_qi, xout => expXIsMax_uid38_fpAddTest_q, clk => clk, aclr => areset );

    -- redist65_expXIsMax_uid38_fpAddTest_q_17(DELAY,644)
    redist65_expXIsMax_uid38_fpAddTest_q_17 : dspba_delay
    GENERIC MAP ( width => 1, depth => 16, reset_kind => "ASYNC" )
    PORT MAP ( xin => expXIsMax_uid38_fpAddTest_q, xout => redist65_expXIsMax_uid38_fpAddTest_q_17_q, clk => clk, aclr => areset );

    -- invExpXIsMax_uid43_fpAddTest(LOGICAL,42)@20
    invExpXIsMax_uid43_fpAddTest_q <= not (redist65_expXIsMax_uid38_fpAddTest_q_17_q);

    -- redist62_InvExpXIsZero_uid44_fpAddTest_q_16(DELAY,641)
    redist62_InvExpXIsZero_uid44_fpAddTest_q_16 : dspba_delay
    GENERIC MAP ( width => 1, depth => 16, reset_kind => "ASYNC" )
    PORT MAP ( xin => InvExpXIsZero_uid44_fpAddTest_q, xout => redist62_InvExpXIsZero_uid44_fpAddTest_q_16_q, clk => clk, aclr => areset );

    -- excR_bSig_uid45_fpAddTest(LOGICAL,44)@20 + 1
    excR_bSig_uid45_fpAddTest_qi <= redist62_InvExpXIsZero_uid44_fpAddTest_q_16_q and invExpXIsMax_uid43_fpAddTest_q;
    excR_bSig_uid45_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excR_bSig_uid45_fpAddTest_qi, xout => excR_bSig_uid45_fpAddTest_q, clk => clk, aclr => areset );

    -- redist74_exp_aSig_uid21_fpAddTest_b_17(DELAY,653)
    redist74_exp_aSig_uid21_fpAddTest_b_17 : dspba_delay
    GENERIC MAP ( width => 11, depth => 17, reset_kind => "ASYNC" )
    PORT MAP ( xin => exp_aSig_uid21_fpAddTest_b, xout => redist74_exp_aSig_uid21_fpAddTest_b_17_q, clk => clk, aclr => areset );

    -- expXIsMax_uid24_fpAddTest(LOGICAL,23)@20
    expXIsMax_uid24_fpAddTest_q <= "1" WHEN redist74_exp_aSig_uid21_fpAddTest_b_17_q = cstAllOWE_uid18_fpAddTest_q ELSE "0";

    -- invExpXIsMax_uid29_fpAddTest(LOGICAL,28)@20
    invExpXIsMax_uid29_fpAddTest_q <= not (expXIsMax_uid24_fpAddTest_q);

    -- excZ_aSig_uid16_uid23_fpAddTest(LOGICAL,22)@20
    excZ_aSig_uid16_uid23_fpAddTest_q <= "1" WHEN redist74_exp_aSig_uid21_fpAddTest_b_17_q = cstAllZWE_uid20_fpAddTest_q ELSE "0";

    -- InvExpXIsZero_uid30_fpAddTest(LOGICAL,29)@20
    InvExpXIsZero_uid30_fpAddTest_q <= not (excZ_aSig_uid16_uid23_fpAddTest_q);

    -- excR_aSig_uid31_fpAddTest(LOGICAL,30)@20 + 1
    excR_aSig_uid31_fpAddTest_qi <= InvExpXIsZero_uid30_fpAddTest_q and invExpXIsMax_uid29_fpAddTest_q;
    excR_aSig_uid31_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excR_aSig_uid31_fpAddTest_qi, xout => excR_aSig_uid31_fpAddTest_q, clk => clk, aclr => areset );

    -- signRReg_uid100_fpAddTest(LOGICAL,99)@21
    signRReg_uid100_fpAddTest_q <= excR_aSig_uid31_fpAddTest_q and excR_bSig_uid45_fpAddTest_q and redist61_sigA_uid50_fpAddTest_b_18_q and invAMinusA_uid99_fpAddTest_q;

    -- redist59_sigB_uid51_fpAddTest_b_19(DELAY,638)
    redist59_sigB_uid51_fpAddTest_b_19 : dspba_delay
    GENERIC MAP ( width => 1, depth => 17, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist58_sigB_uid51_fpAddTest_b_2_q, xout => redist59_sigB_uid51_fpAddTest_b_19_q, clk => clk, aclr => areset );

    -- redist66_excZ_bSig_uid17_uid37_fpAddTest_q_18(DELAY,645)
    redist66_excZ_bSig_uid17_uid37_fpAddTest_q_18 : dspba_delay
    GENERIC MAP ( width => 1, depth => 17, reset_kind => "ASYNC" )
    PORT MAP ( xin => excZ_bSig_uid17_uid37_fpAddTest_q, xout => redist66_excZ_bSig_uid17_uid37_fpAddTest_q_18_q, clk => clk, aclr => areset );

    -- redist72_excZ_aSig_uid16_uid23_fpAddTest_q_1(DELAY,651)
    redist72_excZ_aSig_uid16_uid23_fpAddTest_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excZ_aSig_uid16_uid23_fpAddTest_q, xout => redist72_excZ_aSig_uid16_uid23_fpAddTest_q_1_q, clk => clk, aclr => areset );

    -- excAZBZSigASigB_uid104_fpAddTest(LOGICAL,103)@21
    excAZBZSigASigB_uid104_fpAddTest_q <= redist72_excZ_aSig_uid16_uid23_fpAddTest_q_1_q and redist66_excZ_bSig_uid17_uid37_fpAddTest_q_18_q and redist61_sigA_uid50_fpAddTest_b_18_q and redist59_sigB_uid51_fpAddTest_b_19_q;

    -- excBZARSigA_uid105_fpAddTest(LOGICAL,104)@21
    excBZARSigA_uid105_fpAddTest_q <= redist66_excZ_bSig_uid17_uid37_fpAddTest_q_18_q and excR_aSig_uid31_fpAddTest_q and redist61_sigA_uid50_fpAddTest_b_18_q;

    -- signRZero_uid106_fpAddTest(LOGICAL,105)@21
    signRZero_uid106_fpAddTest_q <= excBZARSigA_uid105_fpAddTest_q or excAZBZSigASigB_uid104_fpAddTest_q;

    -- c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select(BITSELECT,569)
    c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_b <= cstZeroWF_uid19_fpAddTest_q(5 downto 0);
    c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_c <= cstZeroWF_uid19_fpAddTest_q(11 downto 6);
    c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_d <= cstZeroWF_uid19_fpAddTest_q(17 downto 12);
    c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_e <= cstZeroWF_uid19_fpAddTest_q(23 downto 18);
    c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_f <= cstZeroWF_uid19_fpAddTest_q(29 downto 24);
    c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_g <= cstZeroWF_uid19_fpAddTest_q(35 downto 30);
    c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_h <= cstZeroWF_uid19_fpAddTest_q(41 downto 36);
    c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_i <= cstZeroWF_uid19_fpAddTest_q(47 downto 42);
    c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_j <= cstZeroWF_uid19_fpAddTest_q(51 downto 48);

    -- z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select(BITSELECT,571)@4
    z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_b <= redist68_frac_bSig_uid36_fpAddTest_b_2_q(5 downto 0);
    z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_c <= redist68_frac_bSig_uid36_fpAddTest_b_2_q(11 downto 6);
    z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_d <= redist68_frac_bSig_uid36_fpAddTest_b_2_q(17 downto 12);
    z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_e <= redist68_frac_bSig_uid36_fpAddTest_b_2_q(23 downto 18);
    z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_f <= redist68_frac_bSig_uid36_fpAddTest_b_2_q(29 downto 24);
    z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_g <= redist68_frac_bSig_uid36_fpAddTest_b_2_q(35 downto 30);
    z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_h <= redist68_frac_bSig_uid36_fpAddTest_b_2_q(41 downto 36);
    z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_i <= redist68_frac_bSig_uid36_fpAddTest_b_2_q(47 downto 42);
    z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_j <= redist68_frac_bSig_uid36_fpAddTest_b_2_q(51 downto 48);

    -- eq8_uid213_fracXIsZero_uid39_fpAddTest(LOGICAL,212)@4
    eq8_uid213_fracXIsZero_uid39_fpAddTest_q <= "1" WHEN z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_j = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_j ELSE "0";

    -- eq7_uid210_fracXIsZero_uid39_fpAddTest(LOGICAL,209)@4
    eq7_uid210_fracXIsZero_uid39_fpAddTest_q <= "1" WHEN z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_i = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_i ELSE "0";

    -- eq6_uid207_fracXIsZero_uid39_fpAddTest(LOGICAL,206)@4
    eq6_uid207_fracXIsZero_uid39_fpAddTest_q <= "1" WHEN z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_h = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_h ELSE "0";

    -- and_lev0_uid215_fracXIsZero_uid39_fpAddTest(LOGICAL,214)@4 + 1
    and_lev0_uid215_fracXIsZero_uid39_fpAddTest_qi <= eq6_uid207_fracXIsZero_uid39_fpAddTest_q and eq7_uid210_fracXIsZero_uid39_fpAddTest_q and eq8_uid213_fracXIsZero_uid39_fpAddTest_q;
    and_lev0_uid215_fracXIsZero_uid39_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev0_uid215_fracXIsZero_uid39_fpAddTest_qi, xout => and_lev0_uid215_fracXIsZero_uid39_fpAddTest_q, clk => clk, aclr => areset );

    -- eq5_uid204_fracXIsZero_uid39_fpAddTest(LOGICAL,203)@4
    eq5_uid204_fracXIsZero_uid39_fpAddTest_q <= "1" WHEN z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_g = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_g ELSE "0";

    -- eq4_uid201_fracXIsZero_uid39_fpAddTest(LOGICAL,200)@4
    eq4_uid201_fracXIsZero_uid39_fpAddTest_q <= "1" WHEN z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_f = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_f ELSE "0";

    -- eq3_uid198_fracXIsZero_uid39_fpAddTest(LOGICAL,197)@4
    eq3_uid198_fracXIsZero_uid39_fpAddTest_q <= "1" WHEN z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_e = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_e ELSE "0";

    -- eq2_uid195_fracXIsZero_uid39_fpAddTest(LOGICAL,194)@4
    eq2_uid195_fracXIsZero_uid39_fpAddTest_q <= "1" WHEN z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_d = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_d ELSE "0";

    -- eq1_uid192_fracXIsZero_uid39_fpAddTest(LOGICAL,191)@4
    eq1_uid192_fracXIsZero_uid39_fpAddTest_q <= "1" WHEN z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_c = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_c ELSE "0";

    -- eq0_uid189_fracXIsZero_uid39_fpAddTest(LOGICAL,188)@4
    eq0_uid189_fracXIsZero_uid39_fpAddTest_q <= "1" WHEN z0_uid187_fracXIsZero_uid39_fpAddTest_merged_bit_select_b = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_b ELSE "0";

    -- and_lev0_uid214_fracXIsZero_uid39_fpAddTest(LOGICAL,213)@4 + 1
    and_lev0_uid214_fracXIsZero_uid39_fpAddTest_qi <= eq0_uid189_fracXIsZero_uid39_fpAddTest_q and eq1_uid192_fracXIsZero_uid39_fpAddTest_q and eq2_uid195_fracXIsZero_uid39_fpAddTest_q and eq3_uid198_fracXIsZero_uid39_fpAddTest_q and eq4_uid201_fracXIsZero_uid39_fpAddTest_q and eq5_uid204_fracXIsZero_uid39_fpAddTest_q;
    and_lev0_uid214_fracXIsZero_uid39_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev0_uid214_fracXIsZero_uid39_fpAddTest_qi, xout => and_lev0_uid214_fracXIsZero_uid39_fpAddTest_q, clk => clk, aclr => areset );

    -- and_lev1_uid216_fracXIsZero_uid39_fpAddTest(LOGICAL,215)@5 + 1
    and_lev1_uid216_fracXIsZero_uid39_fpAddTest_qi <= and_lev0_uid214_fracXIsZero_uid39_fpAddTest_q and and_lev0_uid215_fracXIsZero_uid39_fpAddTest_q;
    and_lev1_uid216_fracXIsZero_uid39_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev1_uid216_fracXIsZero_uid39_fpAddTest_qi, xout => and_lev1_uid216_fracXIsZero_uid39_fpAddTest_q, clk => clk, aclr => areset );

    -- redist37_and_lev1_uid216_fracXIsZero_uid39_fpAddTest_q_15(DELAY,616)
    redist37_and_lev1_uid216_fracXIsZero_uid39_fpAddTest_q_15 : dspba_delay
    GENERIC MAP ( width => 1, depth => 14, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev1_uid216_fracXIsZero_uid39_fpAddTest_q, xout => redist37_and_lev1_uid216_fracXIsZero_uid39_fpAddTest_q_15_q, clk => clk, aclr => areset );

    -- excI_bSig_uid41_fpAddTest(LOGICAL,40)@20 + 1
    excI_bSig_uid41_fpAddTest_qi <= redist65_expXIsMax_uid38_fpAddTest_q_17_q and redist37_and_lev1_uid216_fracXIsZero_uid39_fpAddTest_q_15_q;
    excI_bSig_uid41_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excI_bSig_uid41_fpAddTest_qi, xout => excI_bSig_uid41_fpAddTest_q, clk => clk, aclr => areset );

    -- sigBBInf_uid101_fpAddTest(LOGICAL,100)@21
    sigBBInf_uid101_fpAddTest_q <= redist59_sigB_uid51_fpAddTest_b_19_q and excI_bSig_uid41_fpAddTest_q;

    -- frac_aSig_uid22_fpAddTest(BITSELECT,21)@3
    frac_aSig_uid22_fpAddTest_in <= aSig_uid16_fpAddTest_BitJoin_for_q_q(51 downto 0);
    frac_aSig_uid22_fpAddTest_b <= frac_aSig_uid22_fpAddTest_in(51 downto 0);

    -- z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select(BITSELECT,570)@3
    z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_b <= frac_aSig_uid22_fpAddTest_b(5 downto 0);
    z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_c <= frac_aSig_uid22_fpAddTest_b(11 downto 6);
    z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_d <= frac_aSig_uid22_fpAddTest_b(17 downto 12);
    z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_e <= frac_aSig_uid22_fpAddTest_b(23 downto 18);
    z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_f <= frac_aSig_uid22_fpAddTest_b(29 downto 24);
    z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_g <= frac_aSig_uid22_fpAddTest_b(35 downto 30);
    z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_h <= frac_aSig_uid22_fpAddTest_b(41 downto 36);
    z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_i <= frac_aSig_uid22_fpAddTest_b(47 downto 42);
    z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_j <= frac_aSig_uid22_fpAddTest_b(51 downto 48);

    -- eq8_uid183_fracXIsZero_uid25_fpAddTest(LOGICAL,182)@3
    eq8_uid183_fracXIsZero_uid25_fpAddTest_q <= "1" WHEN z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_j = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_j ELSE "0";

    -- eq7_uid180_fracXIsZero_uid25_fpAddTest(LOGICAL,179)@3
    eq7_uid180_fracXIsZero_uid25_fpAddTest_q <= "1" WHEN z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_i = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_i ELSE "0";

    -- eq6_uid177_fracXIsZero_uid25_fpAddTest(LOGICAL,176)@3
    eq6_uid177_fracXIsZero_uid25_fpAddTest_q <= "1" WHEN z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_h = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_h ELSE "0";

    -- and_lev0_uid185_fracXIsZero_uid25_fpAddTest(LOGICAL,184)@3 + 1
    and_lev0_uid185_fracXIsZero_uid25_fpAddTest_qi <= eq6_uid177_fracXIsZero_uid25_fpAddTest_q and eq7_uid180_fracXIsZero_uid25_fpAddTest_q and eq8_uid183_fracXIsZero_uid25_fpAddTest_q;
    and_lev0_uid185_fracXIsZero_uid25_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev0_uid185_fracXIsZero_uid25_fpAddTest_qi, xout => and_lev0_uid185_fracXIsZero_uid25_fpAddTest_q, clk => clk, aclr => areset );

    -- eq5_uid174_fracXIsZero_uid25_fpAddTest(LOGICAL,173)@3
    eq5_uid174_fracXIsZero_uid25_fpAddTest_q <= "1" WHEN z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_g = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_g ELSE "0";

    -- eq4_uid171_fracXIsZero_uid25_fpAddTest(LOGICAL,170)@3
    eq4_uid171_fracXIsZero_uid25_fpAddTest_q <= "1" WHEN z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_f = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_f ELSE "0";

    -- eq3_uid168_fracXIsZero_uid25_fpAddTest(LOGICAL,167)@3
    eq3_uid168_fracXIsZero_uid25_fpAddTest_q <= "1" WHEN z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_e = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_e ELSE "0";

    -- eq2_uid165_fracXIsZero_uid25_fpAddTest(LOGICAL,164)@3
    eq2_uid165_fracXIsZero_uid25_fpAddTest_q <= "1" WHEN z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_d = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_d ELSE "0";

    -- eq1_uid162_fracXIsZero_uid25_fpAddTest(LOGICAL,161)@3
    eq1_uid162_fracXIsZero_uid25_fpAddTest_q <= "1" WHEN z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_c = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_c ELSE "0";

    -- eq0_uid159_fracXIsZero_uid25_fpAddTest(LOGICAL,158)@3
    eq0_uid159_fracXIsZero_uid25_fpAddTest_q <= "1" WHEN z0_uid157_fracXIsZero_uid25_fpAddTest_merged_bit_select_b = c0_uid158_fracXIsZero_uid25_fpAddTest_merged_bit_select_b ELSE "0";

    -- and_lev0_uid184_fracXIsZero_uid25_fpAddTest(LOGICAL,183)@3 + 1
    and_lev0_uid184_fracXIsZero_uid25_fpAddTest_qi <= eq0_uid159_fracXIsZero_uid25_fpAddTest_q and eq1_uid162_fracXIsZero_uid25_fpAddTest_q and eq2_uid165_fracXIsZero_uid25_fpAddTest_q and eq3_uid168_fracXIsZero_uid25_fpAddTest_q and eq4_uid171_fracXIsZero_uid25_fpAddTest_q and eq5_uid174_fracXIsZero_uid25_fpAddTest_q;
    and_lev0_uid184_fracXIsZero_uid25_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev0_uid184_fracXIsZero_uid25_fpAddTest_qi, xout => and_lev0_uid184_fracXIsZero_uid25_fpAddTest_q, clk => clk, aclr => areset );

    -- and_lev1_uid186_fracXIsZero_uid25_fpAddTest(LOGICAL,185)@4 + 1
    and_lev1_uid186_fracXIsZero_uid25_fpAddTest_qi <= and_lev0_uid184_fracXIsZero_uid25_fpAddTest_q and and_lev0_uid185_fracXIsZero_uid25_fpAddTest_q;
    and_lev1_uid186_fracXIsZero_uid25_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev1_uid186_fracXIsZero_uid25_fpAddTest_qi, xout => and_lev1_uid186_fracXIsZero_uid25_fpAddTest_q, clk => clk, aclr => areset );

    -- redist38_and_lev1_uid186_fracXIsZero_uid25_fpAddTest_q_16(DELAY,617)
    redist38_and_lev1_uid186_fracXIsZero_uid25_fpAddTest_q_16 : dspba_delay
    GENERIC MAP ( width => 1, depth => 15, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev1_uid186_fracXIsZero_uid25_fpAddTest_q, xout => redist38_and_lev1_uid186_fracXIsZero_uid25_fpAddTest_q_16_q, clk => clk, aclr => areset );

    -- excI_aSig_uid27_fpAddTest(LOGICAL,26)@20 + 1
    excI_aSig_uid27_fpAddTest_qi <= expXIsMax_uid24_fpAddTest_q and redist38_and_lev1_uid186_fracXIsZero_uid25_fpAddTest_q_16_q;
    excI_aSig_uid27_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excI_aSig_uid27_fpAddTest_qi, xout => excI_aSig_uid27_fpAddTest_q, clk => clk, aclr => areset );

    -- sigAAInf_uid102_fpAddTest(LOGICAL,101)@21
    sigAAInf_uid102_fpAddTest_q <= redist61_sigA_uid50_fpAddTest_b_18_q and excI_aSig_uid27_fpAddTest_q;

    -- signRInf_uid103_fpAddTest(LOGICAL,102)@21
    signRInf_uid103_fpAddTest_q <= sigAAInf_uid102_fpAddTest_q or sigBBInf_uid101_fpAddTest_q;

    -- signRInfRZRReg_uid107_fpAddTest(LOGICAL,106)@21 + 1
    signRInfRZRReg_uid107_fpAddTest_qi <= signRInf_uid103_fpAddTest_q or signRZero_uid106_fpAddTest_q or signRReg_uid100_fpAddTest_q;
    signRInfRZRReg_uid107_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => signRInfRZRReg_uid107_fpAddTest_qi, xout => signRInfRZRReg_uid107_fpAddTest_q, clk => clk, aclr => areset );

    -- redist46_signRInfRZRReg_uid107_fpAddTest_q_5(DELAY,625)
    redist46_signRInfRZRReg_uid107_fpAddTest_q_5 : dspba_delay
    GENERIC MAP ( width => 1, depth => 4, reset_kind => "ASYNC" )
    PORT MAP ( xin => signRInfRZRReg_uid107_fpAddTest_q, xout => redist46_signRInfRZRReg_uid107_fpAddTest_q_5_q, clk => clk, aclr => areset );

    -- fracXIsNotZero_uid40_fpAddTest(LOGICAL,39)@20
    fracXIsNotZero_uid40_fpAddTest_q <= not (redist37_and_lev1_uid216_fracXIsZero_uid39_fpAddTest_q_15_q);

    -- excN_bSig_uid42_fpAddTest(LOGICAL,41)@20 + 1
    excN_bSig_uid42_fpAddTest_qi <= redist65_expXIsMax_uid38_fpAddTest_q_17_q and fracXIsNotZero_uid40_fpAddTest_q;
    excN_bSig_uid42_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excN_bSig_uid42_fpAddTest_qi, xout => excN_bSig_uid42_fpAddTest_q, clk => clk, aclr => areset );

    -- redist63_excN_bSig_uid42_fpAddTest_q_5(DELAY,642)
    redist63_excN_bSig_uid42_fpAddTest_q_5 : dspba_delay
    GENERIC MAP ( width => 1, depth => 4, reset_kind => "ASYNC" )
    PORT MAP ( xin => excN_bSig_uid42_fpAddTest_q, xout => redist63_excN_bSig_uid42_fpAddTest_q_5_q, clk => clk, aclr => areset );

    -- fracXIsNotZero_uid26_fpAddTest(LOGICAL,25)@20
    fracXIsNotZero_uid26_fpAddTest_q <= not (redist38_and_lev1_uid186_fracXIsZero_uid25_fpAddTest_q_16_q);

    -- excN_aSig_uid28_fpAddTest(LOGICAL,27)@20 + 1
    excN_aSig_uid28_fpAddTest_qi <= expXIsMax_uid24_fpAddTest_q and fracXIsNotZero_uid26_fpAddTest_q;
    excN_aSig_uid28_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excN_aSig_uid28_fpAddTest_qi, xout => excN_aSig_uid28_fpAddTest_q, clk => clk, aclr => areset );

    -- redist70_excN_aSig_uid28_fpAddTest_q_5(DELAY,649)
    redist70_excN_aSig_uid28_fpAddTest_q_5 : dspba_delay
    GENERIC MAP ( width => 1, depth => 4, reset_kind => "ASYNC" )
    PORT MAP ( xin => excN_aSig_uid28_fpAddTest_q, xout => redist70_excN_aSig_uid28_fpAddTest_q_5_q, clk => clk, aclr => areset );

    -- excRNaN2_uid94_fpAddTest(LOGICAL,93)@25
    excRNaN2_uid94_fpAddTest_q <= redist70_excN_aSig_uid28_fpAddTest_q_5_q or redist63_excN_bSig_uid42_fpAddTest_q_5_q;

    -- redist57_effSub_uid52_fpAddTest_q_21(DELAY,636)
    redist57_effSub_uid52_fpAddTest_q_21 : dspba_delay
    GENERIC MAP ( width => 1, depth => 20, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist56_effSub_uid52_fpAddTest_q_1_q, xout => redist57_effSub_uid52_fpAddTest_q_21_q, clk => clk, aclr => areset );

    -- redist64_excI_bSig_uid41_fpAddTest_q_5(DELAY,643)
    redist64_excI_bSig_uid41_fpAddTest_q_5 : dspba_delay
    GENERIC MAP ( width => 1, depth => 4, reset_kind => "ASYNC" )
    PORT MAP ( xin => excI_bSig_uid41_fpAddTest_q, xout => redist64_excI_bSig_uid41_fpAddTest_q_5_q, clk => clk, aclr => areset );

    -- redist71_excI_aSig_uid27_fpAddTest_q_5(DELAY,650)
    redist71_excI_aSig_uid27_fpAddTest_q_5 : dspba_delay
    GENERIC MAP ( width => 1, depth => 4, reset_kind => "ASYNC" )
    PORT MAP ( xin => excI_aSig_uid27_fpAddTest_q, xout => redist71_excI_aSig_uid27_fpAddTest_q_5_q, clk => clk, aclr => areset );

    -- excAIBISub_uid95_fpAddTest(LOGICAL,94)@25
    excAIBISub_uid95_fpAddTest_q <= redist71_excI_aSig_uid27_fpAddTest_q_5_q and redist64_excI_bSig_uid41_fpAddTest_q_5_q and redist57_effSub_uid52_fpAddTest_q_21_q;

    -- excRNaN_uid96_fpAddTest(LOGICAL,95)@25 + 1
    excRNaN_uid96_fpAddTest_qi <= excAIBISub_uid95_fpAddTest_q or excRNaN2_uid94_fpAddTest_q;
    excRNaN_uid96_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excRNaN_uid96_fpAddTest_qi, xout => excRNaN_uid96_fpAddTest_q, clk => clk, aclr => areset );

    -- invExcRNaN_uid108_fpAddTest(LOGICAL,107)@26
    invExcRNaN_uid108_fpAddTest_q <= not (excRNaN_uid96_fpAddTest_q);

    -- VCC(CONSTANT,1)
    VCC_q <= "1";

    -- signRPostExc_uid109_fpAddTest(LOGICAL,108)@26 + 1
    signRPostExc_uid109_fpAddTest_qi <= invExcRNaN_uid108_fpAddTest_q and redist46_signRInfRZRReg_uid107_fpAddTest_q_5_q;
    signRPostExc_uid109_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => signRPostExc_uid109_fpAddTest_qi, xout => signRPostExc_uid109_fpAddTest_q, clk => clk, aclr => areset );

    -- expInc_uid78_fpAddTest(ADD,77)@20 + 1
    expInc_uid78_fpAddTest_a <= STD_LOGIC_VECTOR("0" & redist74_exp_aSig_uid21_fpAddTest_b_17_q);
    expInc_uid78_fpAddTest_b <= STD_LOGIC_VECTOR("00000000000" & VCC_q);
    expInc_uid78_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expInc_uid78_fpAddTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expInc_uid78_fpAddTest_o <= STD_LOGIC_VECTOR(UNSIGNED(expInc_uid78_fpAddTest_a) + UNSIGNED(expInc_uid78_fpAddTest_b));
        END IF;
    END PROCESS;
    expInc_uid78_fpAddTest_q <= expInc_uid78_fpAddTest_o(11 downto 0);

    -- expPostNorm_uid79_fpAddTest(SUB,78)@21 + 1
    expPostNorm_uid79_fpAddTest_a <= STD_LOGIC_VECTOR("0" & expInc_uid78_fpAddTest_q);
    expPostNorm_uid79_fpAddTest_b <= STD_LOGIC_VECTOR("0000000" & redist39_r_uid155_lzCountVal_uid74_fpAddTest_q_1_q);
    expPostNorm_uid79_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expPostNorm_uid79_fpAddTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expPostNorm_uid79_fpAddTest_o <= STD_LOGIC_VECTOR(UNSIGNED(expPostNorm_uid79_fpAddTest_a) - UNSIGNED(expPostNorm_uid79_fpAddTest_b));
        END IF;
    END PROCESS;
    expPostNorm_uid79_fpAddTest_q <= expPostNorm_uid79_fpAddTest_o(12 downto 0);

    -- redist52_expPostNorm_uid79_fpAddTest_q_3(DELAY,631)
    redist52_expPostNorm_uid79_fpAddTest_q_3 : dspba_delay
    GENERIC MAP ( width => 13, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => expPostNorm_uid79_fpAddTest_q, xout => redist52_expPostNorm_uid79_fpAddTest_q_3_q, clk => clk, aclr => areset );

    -- leftShiftStage2Idx3Rng3_uid288_fracPostNorm_uid75_fpAddTest(BITSELECT,287)@23
    leftShiftStage2Idx3Rng3_uid288_fracPostNorm_uid75_fpAddTest_in <= leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_q(52 downto 0);
    leftShiftStage2Idx3Rng3_uid288_fracPostNorm_uid75_fpAddTest_b <= leftShiftStage2Idx3Rng3_uid288_fracPostNorm_uid75_fpAddTest_in(52 downto 0);

    -- leftShiftStage2Idx3Pad3_uid287_fracPostNorm_uid75_fpAddTest(CONSTANT,286)
    leftShiftStage2Idx3Pad3_uid287_fracPostNorm_uid75_fpAddTest_q <= "000";

    -- leftShiftStage2Idx3_uid289_fracPostNorm_uid75_fpAddTest(BITJOIN,288)@23
    leftShiftStage2Idx3_uid289_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage2Idx3Rng3_uid288_fracPostNorm_uid75_fpAddTest_b & leftShiftStage2Idx3Pad3_uid287_fracPostNorm_uid75_fpAddTest_q;

    -- leftShiftStage2Idx2Rng2_uid285_fracPostNorm_uid75_fpAddTest(BITSELECT,284)@23
    leftShiftStage2Idx2Rng2_uid285_fracPostNorm_uid75_fpAddTest_in <= leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_q(53 downto 0);
    leftShiftStage2Idx2Rng2_uid285_fracPostNorm_uid75_fpAddTest_b <= leftShiftStage2Idx2Rng2_uid285_fracPostNorm_uid75_fpAddTest_in(53 downto 0);

    -- leftShiftStage2Idx2_uid286_fracPostNorm_uid75_fpAddTest(BITJOIN,285)@23
    leftShiftStage2Idx2_uid286_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage2Idx2Rng2_uid285_fracPostNorm_uid75_fpAddTest_b & zs_uid146_lzCountVal_uid74_fpAddTest_q;

    -- leftShiftStage2Idx1Rng1_uid282_fracPostNorm_uid75_fpAddTest(BITSELECT,281)@23
    leftShiftStage2Idx1Rng1_uid282_fracPostNorm_uid75_fpAddTest_in <= leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_q(54 downto 0);
    leftShiftStage2Idx1Rng1_uid282_fracPostNorm_uid75_fpAddTest_b <= leftShiftStage2Idx1Rng1_uid282_fracPostNorm_uid75_fpAddTest_in(54 downto 0);

    -- leftShiftStage2Idx1_uid283_fracPostNorm_uid75_fpAddTest(BITJOIN,282)@23
    leftShiftStage2Idx1_uid283_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage2Idx1Rng1_uid282_fracPostNorm_uid75_fpAddTest_b & GND_q;

    -- leftShiftStage1Idx3Rng12_uid277_fracPostNorm_uid75_fpAddTest(BITSELECT,276)@22
    leftShiftStage1Idx3Rng12_uid277_fracPostNorm_uid75_fpAddTest_in <= leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_q(43 downto 0);
    leftShiftStage1Idx3Rng12_uid277_fracPostNorm_uid75_fpAddTest_b <= leftShiftStage1Idx3Rng12_uid277_fracPostNorm_uid75_fpAddTest_in(43 downto 0);

    -- leftShiftStage1Idx3Pad12_uid276_fracPostNorm_uid75_fpAddTest(CONSTANT,275)
    leftShiftStage1Idx3Pad12_uid276_fracPostNorm_uid75_fpAddTest_q <= "000000000000";

    -- leftShiftStage1Idx3_uid278_fracPostNorm_uid75_fpAddTest(BITJOIN,277)@22
    leftShiftStage1Idx3_uid278_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage1Idx3Rng12_uid277_fracPostNorm_uid75_fpAddTest_b & leftShiftStage1Idx3Pad12_uid276_fracPostNorm_uid75_fpAddTest_q;

    -- leftShiftStage1Idx2Rng8_uid274_fracPostNorm_uid75_fpAddTest(BITSELECT,273)@22
    leftShiftStage1Idx2Rng8_uid274_fracPostNorm_uid75_fpAddTest_in <= leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_q(47 downto 0);
    leftShiftStage1Idx2Rng8_uid274_fracPostNorm_uid75_fpAddTest_b <= leftShiftStage1Idx2Rng8_uid274_fracPostNorm_uid75_fpAddTest_in(47 downto 0);

    -- leftShiftStage1Idx2_uid275_fracPostNorm_uid75_fpAddTest(BITJOIN,274)@22
    leftShiftStage1Idx2_uid275_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage1Idx2Rng8_uid274_fracPostNorm_uid75_fpAddTest_b & zs_uid134_lzCountVal_uid74_fpAddTest_q;

    -- leftShiftStage1Idx1Rng4_uid271_fracPostNorm_uid75_fpAddTest(BITSELECT,270)@22
    leftShiftStage1Idx1Rng4_uid271_fracPostNorm_uid75_fpAddTest_in <= leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_q(51 downto 0);
    leftShiftStage1Idx1Rng4_uid271_fracPostNorm_uid75_fpAddTest_b <= leftShiftStage1Idx1Rng4_uid271_fracPostNorm_uid75_fpAddTest_in(51 downto 0);

    -- leftShiftStage1Idx1_uid272_fracPostNorm_uid75_fpAddTest(BITJOIN,271)@22
    leftShiftStage1Idx1_uid272_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage1Idx1Rng4_uid271_fracPostNorm_uid75_fpAddTest_b & zs_uid140_lzCountVal_uid74_fpAddTest_q;

    -- leftShiftStage0Idx3Rng48_uid266_fracPostNorm_uid75_fpAddTest(BITSELECT,265)@21
    leftShiftStage0Idx3Rng48_uid266_fracPostNorm_uid75_fpAddTest_in <= redist55_fracAddResultNoSignExt_uid73_fpAddTest_b_8_q(7 downto 0);
    leftShiftStage0Idx3Rng48_uid266_fracPostNorm_uid75_fpAddTest_b <= leftShiftStage0Idx3Rng48_uid266_fracPostNorm_uid75_fpAddTest_in(7 downto 0);

    -- leftShiftStage0Idx3Pad48_uid265_fracPostNorm_uid75_fpAddTest(CONSTANT,264)
    leftShiftStage0Idx3Pad48_uid265_fracPostNorm_uid75_fpAddTest_q <= "000000000000000000000000000000000000000000000000";

    -- leftShiftStage0Idx3_uid267_fracPostNorm_uid75_fpAddTest(BITJOIN,266)@21
    leftShiftStage0Idx3_uid267_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage0Idx3Rng48_uid266_fracPostNorm_uid75_fpAddTest_b & leftShiftStage0Idx3Pad48_uid265_fracPostNorm_uid75_fpAddTest_q;

    -- redist43_vStage_uid124_lzCountVal_uid74_fpAddTest_b_7(DELAY,622)
    redist43_vStage_uid124_lzCountVal_uid74_fpAddTest_b_7 : dspba_delay
    GENERIC MAP ( width => 24, depth => 7, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStage_uid124_lzCountVal_uid74_fpAddTest_b, xout => redist43_vStage_uid124_lzCountVal_uid74_fpAddTest_b_7_q, clk => clk, aclr => areset );

    -- leftShiftStage0Idx2_uid264_fracPostNorm_uid75_fpAddTest(BITJOIN,263)@21
    leftShiftStage0Idx2_uid264_fracPostNorm_uid75_fpAddTest_q <= redist43_vStage_uid124_lzCountVal_uid74_fpAddTest_b_7_q & zs_uid120_lzCountVal_uid74_fpAddTest_q;

    -- leftShiftStage0Idx1Rng16_uid260_fracPostNorm_uid75_fpAddTest(BITSELECT,259)@21
    leftShiftStage0Idx1Rng16_uid260_fracPostNorm_uid75_fpAddTest_in <= redist55_fracAddResultNoSignExt_uid73_fpAddTest_b_8_q(39 downto 0);
    leftShiftStage0Idx1Rng16_uid260_fracPostNorm_uid75_fpAddTest_b <= leftShiftStage0Idx1Rng16_uid260_fracPostNorm_uid75_fpAddTest_in(39 downto 0);

    -- leftShiftStage0Idx1_uid261_fracPostNorm_uid75_fpAddTest(BITJOIN,260)@21
    leftShiftStage0Idx1_uid261_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage0Idx1Rng16_uid260_fracPostNorm_uid75_fpAddTest_b & zs_uid128_lzCountVal_uid74_fpAddTest_q;

    -- redist55_fracAddResultNoSignExt_uid73_fpAddTest_b_8(DELAY,634)
    redist55_fracAddResultNoSignExt_uid73_fpAddTest_b_8 : dspba_delay
    GENERIC MAP ( width => 56, depth => 7, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist54_fracAddResultNoSignExt_uid73_fpAddTest_b_1_q, xout => redist55_fracAddResultNoSignExt_uid73_fpAddTest_b_8_q, clk => clk, aclr => areset );

    -- leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select(BITSELECT,577)@21
    leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_b <= redist39_r_uid155_lzCountVal_uid74_fpAddTest_q_1_q(5 downto 4);
    leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_c <= redist39_r_uid155_lzCountVal_uid74_fpAddTest_q_1_q(3 downto 2);
    leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_d <= redist39_r_uid155_lzCountVal_uid74_fpAddTest_q_1_q(1 downto 0);

    -- leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest(MUX,268)@21 + 1
    leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_s <= leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_b;
    leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_s) IS
                WHEN "00" => leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_q <= redist55_fracAddResultNoSignExt_uid73_fpAddTest_b_8_q;
                WHEN "01" => leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage0Idx1_uid261_fracPostNorm_uid75_fpAddTest_q;
                WHEN "10" => leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage0Idx2_uid264_fracPostNorm_uid75_fpAddTest_q;
                WHEN "11" => leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage0Idx3_uid267_fracPostNorm_uid75_fpAddTest_q;
                WHEN OTHERS => leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist1_leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_c_1(DELAY,580)
    redist1_leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_c_1 : dspba_delay
    GENERIC MAP ( width => 2, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_c, xout => redist1_leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_c_1_q, clk => clk, aclr => areset );

    -- leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest(MUX,279)@22 + 1
    leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_s <= redist1_leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_c_1_q;
    leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_s) IS
                WHEN "00" => leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage0_uid269_fracPostNorm_uid75_fpAddTest_q;
                WHEN "01" => leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage1Idx1_uid272_fracPostNorm_uid75_fpAddTest_q;
                WHEN "10" => leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage1Idx2_uid275_fracPostNorm_uid75_fpAddTest_q;
                WHEN "11" => leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage1Idx3_uid278_fracPostNorm_uid75_fpAddTest_q;
                WHEN OTHERS => leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist2_leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_d_2(DELAY,581)
    redist2_leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_d_2 : dspba_delay
    GENERIC MAP ( width => 2, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_d, xout => redist2_leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_d_2_q, clk => clk, aclr => areset );

    -- leftShiftStage2_uid291_fracPostNorm_uid75_fpAddTest(MUX,290)@23
    leftShiftStage2_uid291_fracPostNorm_uid75_fpAddTest_s <= redist2_leftShiftStageSel5Dto4_uid268_fracPostNorm_uid75_fpAddTest_merged_bit_select_d_2_q;
    leftShiftStage2_uid291_fracPostNorm_uid75_fpAddTest_combproc: PROCESS (leftShiftStage2_uid291_fracPostNorm_uid75_fpAddTest_s, leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_q, leftShiftStage2Idx1_uid283_fracPostNorm_uid75_fpAddTest_q, leftShiftStage2Idx2_uid286_fracPostNorm_uid75_fpAddTest_q, leftShiftStage2Idx3_uid289_fracPostNorm_uid75_fpAddTest_q)
    BEGIN
        CASE (leftShiftStage2_uid291_fracPostNorm_uid75_fpAddTest_s) IS
            WHEN "00" => leftShiftStage2_uid291_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage1_uid280_fracPostNorm_uid75_fpAddTest_q;
            WHEN "01" => leftShiftStage2_uid291_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage2Idx1_uid283_fracPostNorm_uid75_fpAddTest_q;
            WHEN "10" => leftShiftStage2_uid291_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage2Idx2_uid286_fracPostNorm_uid75_fpAddTest_q;
            WHEN "11" => leftShiftStage2_uid291_fracPostNorm_uid75_fpAddTest_q <= leftShiftStage2Idx3_uid289_fracPostNorm_uid75_fpAddTest_q;
            WHEN OTHERS => leftShiftStage2_uid291_fracPostNorm_uid75_fpAddTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- fracPostNormRndRange_uid80_fpAddTest(BITSELECT,79)@23
    fracPostNormRndRange_uid80_fpAddTest_in <= leftShiftStage2_uid291_fracPostNorm_uid75_fpAddTest_q(54 downto 0);
    fracPostNormRndRange_uid80_fpAddTest_b <= fracPostNormRndRange_uid80_fpAddTest_in(54 downto 2);

    -- redist51_fracPostNormRndRange_uid80_fpAddTest_b_1(DELAY,630)
    redist51_fracPostNormRndRange_uid80_fpAddTest_b_1 : dspba_delay
    GENERIC MAP ( width => 53, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => fracPostNormRndRange_uid80_fpAddTest_b, xout => redist51_fracPostNormRndRange_uid80_fpAddTest_b_1_q, clk => clk, aclr => areset );

    -- expFracR_uid81_fpAddTest(BITJOIN,80)@24
    expFracR_uid81_fpAddTest_q <= redist52_expPostNorm_uid79_fpAddTest_q_3_q & redist51_fracPostNormRndRange_uid80_fpAddTest_b_1_q;

    -- expRPreExc_uid87_fpAddTest(BITSELECT,86)@24
    expRPreExc_uid87_fpAddTest_in <= expFracR_uid81_fpAddTest_q(63 downto 0);
    expRPreExc_uid87_fpAddTest_b <= expRPreExc_uid87_fpAddTest_in(63 downto 53);

    -- redist49_expRPreExc_uid87_fpAddTest_b_3(DELAY,628)
    redist49_expRPreExc_uid87_fpAddTest_b_3 : dspba_delay
    GENERIC MAP ( width => 11, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => expRPreExc_uid87_fpAddTest_b, xout => redist49_expRPreExc_uid87_fpAddTest_b_3_q, clk => clk, aclr => areset );

    -- wEP2AllOwE_uid82_fpAddTest(CONSTANT,81)
    wEP2AllOwE_uid82_fpAddTest_q <= "0011111111111";

    -- rndExp_uid83_fpAddTest(BITSELECT,82)@24
    rndExp_uid83_fpAddTest_b <= expFracR_uid81_fpAddTest_q(65 downto 53);

    -- rOvf_uid84_fpAddTest(LOGICAL,83)@24
    rOvf_uid84_fpAddTest_q <= "1" WHEN rndExp_uid83_fpAddTest_b = wEP2AllOwE_uid82_fpAddTest_q ELSE "0";

    -- regInputs_uid88_fpAddTest(LOGICAL,87)@21 + 1
    regInputs_uid88_fpAddTest_qi <= excR_aSig_uid31_fpAddTest_q and excR_bSig_uid45_fpAddTest_q;
    regInputs_uid88_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => regInputs_uid88_fpAddTest_qi, xout => regInputs_uid88_fpAddTest_q, clk => clk, aclr => areset );

    -- redist48_regInputs_uid88_fpAddTest_q_3(DELAY,627)
    redist48_regInputs_uid88_fpAddTest_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => regInputs_uid88_fpAddTest_q, xout => redist48_regInputs_uid88_fpAddTest_q_3_q, clk => clk, aclr => areset );

    -- rInfOvf_uid91_fpAddTest(LOGICAL,90)@24 + 1
    rInfOvf_uid91_fpAddTest_qi <= redist48_regInputs_uid88_fpAddTest_q_3_q and rOvf_uid84_fpAddTest_q;
    rInfOvf_uid91_fpAddTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => rInfOvf_uid91_fpAddTest_qi, xout => rInfOvf_uid91_fpAddTest_q, clk => clk, aclr => areset );

    -- excRInfVInC_uid92_fpAddTest(BITJOIN,91)@25
    excRInfVInC_uid92_fpAddTest_q <= rInfOvf_uid91_fpAddTest_q & redist63_excN_bSig_uid42_fpAddTest_q_5_q & redist70_excN_aSig_uid28_fpAddTest_q_5_q & redist64_excI_bSig_uid41_fpAddTest_q_5_q & redist71_excI_aSig_uid27_fpAddTest_q_5_q & redist57_effSub_uid52_fpAddTest_q_21_q;

    -- excRInf_uid93_fpAddTest(LOOKUP,92)@25 + 1
    excRInf_uid93_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            excRInf_uid93_fpAddTest_q <= "0";
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (excRInfVInC_uid92_fpAddTest_q) IS
                WHEN "000000" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "000001" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "000010" => excRInf_uid93_fpAddTest_q <= "1";
                WHEN "000011" => excRInf_uid93_fpAddTest_q <= "1";
                WHEN "000100" => excRInf_uid93_fpAddTest_q <= "1";
                WHEN "000101" => excRInf_uid93_fpAddTest_q <= "1";
                WHEN "000110" => excRInf_uid93_fpAddTest_q <= "1";
                WHEN "000111" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "001000" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "001001" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "001010" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "001011" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "001100" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "001101" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "001110" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "001111" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "010000" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "010001" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "010010" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "010011" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "010100" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "010101" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "010110" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "010111" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "011000" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "011001" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "011010" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "011011" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "011100" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "011101" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "011110" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "011111" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "100000" => excRInf_uid93_fpAddTest_q <= "1";
                WHEN "100001" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "100010" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "100011" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "100100" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "100101" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "100110" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "100111" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "101000" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "101001" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "101010" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "101011" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "101100" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "101101" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "101110" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "101111" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "110000" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "110001" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "110010" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "110011" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "110100" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "110101" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "110110" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "110111" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "111000" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "111001" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "111010" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "111011" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "111100" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "111101" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "111110" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN "111111" => excRInf_uid93_fpAddTest_q <= "0";
                WHEN OTHERS => -- unreachable
                               excRInf_uid93_fpAddTest_q <= (others => '-');
            END CASE;
        END IF;
    END PROCESS;

    -- redist53_aMinusA_uid77_fpAddTest_q_3(DELAY,632)
    redist53_aMinusA_uid77_fpAddTest_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => aMinusA_uid77_fpAddTest_q, xout => redist53_aMinusA_uid77_fpAddTest_q_3_q, clk => clk, aclr => areset );

    -- rUdf_uid85_fpAddTest(BITSELECT,84)@24
    rUdf_uid85_fpAddTest_b <= STD_LOGIC_VECTOR(expFracR_uid81_fpAddTest_q(65 downto 65));

    -- redist67_excZ_bSig_uid17_uid37_fpAddTest_q_21(DELAY,646)
    redist67_excZ_bSig_uid17_uid37_fpAddTest_q_21 : dspba_delay
    GENERIC MAP ( width => 1, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist66_excZ_bSig_uid17_uid37_fpAddTest_q_18_q, xout => redist67_excZ_bSig_uid17_uid37_fpAddTest_q_21_q, clk => clk, aclr => areset );

    -- redist73_excZ_aSig_uid16_uid23_fpAddTest_q_4(DELAY,652)
    redist73_excZ_aSig_uid16_uid23_fpAddTest_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist72_excZ_aSig_uid16_uid23_fpAddTest_q_1_q, xout => redist73_excZ_aSig_uid16_uid23_fpAddTest_q_4_q, clk => clk, aclr => areset );

    -- excRZeroVInC_uid89_fpAddTest(BITJOIN,88)@24
    excRZeroVInC_uid89_fpAddTest_q <= redist53_aMinusA_uid77_fpAddTest_q_3_q & rUdf_uid85_fpAddTest_b & redist48_regInputs_uid88_fpAddTest_q_3_q & redist67_excZ_bSig_uid17_uid37_fpAddTest_q_21_q & redist73_excZ_aSig_uid16_uid23_fpAddTest_q_4_q;

    -- excRZero_uid90_fpAddTest(LOOKUP,89)@24 + 1
    excRZero_uid90_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            excRZero_uid90_fpAddTest_q <= "0";
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (excRZeroVInC_uid89_fpAddTest_q) IS
                WHEN "00000" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "00001" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "00010" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "00011" => excRZero_uid90_fpAddTest_q <= "1";
                WHEN "00100" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "00101" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "00110" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "00111" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "01000" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "01001" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "01010" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "01011" => excRZero_uid90_fpAddTest_q <= "1";
                WHEN "01100" => excRZero_uid90_fpAddTest_q <= "1";
                WHEN "01101" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "01110" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "01111" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "10000" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "10001" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "10010" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "10011" => excRZero_uid90_fpAddTest_q <= "1";
                WHEN "10100" => excRZero_uid90_fpAddTest_q <= "1";
                WHEN "10101" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "10110" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "10111" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "11000" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "11001" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "11010" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "11011" => excRZero_uid90_fpAddTest_q <= "1";
                WHEN "11100" => excRZero_uid90_fpAddTest_q <= "1";
                WHEN "11101" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "11110" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN "11111" => excRZero_uid90_fpAddTest_q <= "0";
                WHEN OTHERS => -- unreachable
                               excRZero_uid90_fpAddTest_q <= (others => '-');
            END CASE;
        END IF;
    END PROCESS;

    -- redist47_excRZero_uid90_fpAddTest_q_2(DELAY,626)
    redist47_excRZero_uid90_fpAddTest_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excRZero_uid90_fpAddTest_q, xout => redist47_excRZero_uid90_fpAddTest_q_2_q, clk => clk, aclr => areset );

    -- concExc_uid97_fpAddTest(BITJOIN,96)@26
    concExc_uid97_fpAddTest_q <= excRNaN_uid96_fpAddTest_q & excRInf_uid93_fpAddTest_q & redist47_excRZero_uid90_fpAddTest_q_2_q;

    -- excREnc_uid98_fpAddTest(LOOKUP,97)@26 + 1
    excREnc_uid98_fpAddTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            excREnc_uid98_fpAddTest_q <= "01";
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (concExc_uid97_fpAddTest_q) IS
                WHEN "000" => excREnc_uid98_fpAddTest_q <= "01";
                WHEN "001" => excREnc_uid98_fpAddTest_q <= "00";
                WHEN "010" => excREnc_uid98_fpAddTest_q <= "10";
                WHEN "011" => excREnc_uid98_fpAddTest_q <= "10";
                WHEN "100" => excREnc_uid98_fpAddTest_q <= "11";
                WHEN "101" => excREnc_uid98_fpAddTest_q <= "11";
                WHEN "110" => excREnc_uid98_fpAddTest_q <= "11";
                WHEN "111" => excREnc_uid98_fpAddTest_q <= "11";
                WHEN OTHERS => -- unreachable
                               excREnc_uid98_fpAddTest_q <= (others => '-');
            END CASE;
        END IF;
    END PROCESS;

    -- expRPostExc_uid117_fpAddTest(MUX,116)@27
    expRPostExc_uid117_fpAddTest_s <= excREnc_uid98_fpAddTest_q;
    expRPostExc_uid117_fpAddTest_combproc: PROCESS (expRPostExc_uid117_fpAddTest_s, cstAllZWE_uid20_fpAddTest_q, redist49_expRPreExc_uid87_fpAddTest_b_3_q, cstAllOWE_uid18_fpAddTest_q)
    BEGIN
        CASE (expRPostExc_uid117_fpAddTest_s) IS
            WHEN "00" => expRPostExc_uid117_fpAddTest_q <= cstAllZWE_uid20_fpAddTest_q;
            WHEN "01" => expRPostExc_uid117_fpAddTest_q <= redist49_expRPreExc_uid87_fpAddTest_b_3_q;
            WHEN "10" => expRPostExc_uid117_fpAddTest_q <= cstAllOWE_uid18_fpAddTest_q;
            WHEN "11" => expRPostExc_uid117_fpAddTest_q <= cstAllOWE_uid18_fpAddTest_q;
            WHEN OTHERS => expRPostExc_uid117_fpAddTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- oneFracRPostExc2_uid110_fpAddTest(CONSTANT,109)
    oneFracRPostExc2_uid110_fpAddTest_q <= "0000000000000000000000000000000000000000000000000001";

    -- fracRPreExc_uid86_fpAddTest(BITSELECT,85)@24
    fracRPreExc_uid86_fpAddTest_in <= expFracR_uid81_fpAddTest_q(52 downto 0);
    fracRPreExc_uid86_fpAddTest_b <= fracRPreExc_uid86_fpAddTest_in(52 downto 1);

    -- redist50_fracRPreExc_uid86_fpAddTest_b_3(DELAY,629)
    redist50_fracRPreExc_uid86_fpAddTest_b_3 : dspba_delay
    GENERIC MAP ( width => 52, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => fracRPreExc_uid86_fpAddTest_b, xout => redist50_fracRPreExc_uid86_fpAddTest_b_3_q, clk => clk, aclr => areset );

    -- fracRPostExc_uid113_fpAddTest(MUX,112)@27
    fracRPostExc_uid113_fpAddTest_s <= excREnc_uid98_fpAddTest_q;
    fracRPostExc_uid113_fpAddTest_combproc: PROCESS (fracRPostExc_uid113_fpAddTest_s, cstZeroWF_uid19_fpAddTest_q, redist50_fracRPreExc_uid86_fpAddTest_b_3_q, oneFracRPostExc2_uid110_fpAddTest_q)
    BEGIN
        CASE (fracRPostExc_uid113_fpAddTest_s) IS
            WHEN "00" => fracRPostExc_uid113_fpAddTest_q <= cstZeroWF_uid19_fpAddTest_q;
            WHEN "01" => fracRPostExc_uid113_fpAddTest_q <= redist50_fracRPreExc_uid86_fpAddTest_b_3_q;
            WHEN "10" => fracRPostExc_uid113_fpAddTest_q <= cstZeroWF_uid19_fpAddTest_q;
            WHEN "11" => fracRPostExc_uid113_fpAddTest_q <= oneFracRPostExc2_uid110_fpAddTest_q;
            WHEN OTHERS => fracRPostExc_uid113_fpAddTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- R_uid118_fpAddTest(BITJOIN,117)@27
    R_uid118_fpAddTest_q <= signRPostExc_uid109_fpAddTest_q & expRPostExc_uid117_fpAddTest_q & fracRPostExc_uid113_fpAddTest_q;

    -- xOut(GPOUT,4)@27
    q <= R_uid118_fpAddTest_q;

END normal;
