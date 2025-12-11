-- ------------------------------------------------------------------------- 
-- High Level Design Compiler for Intel(R) FPGAs Version 17.1 (Release Build #590)
-- Quartus Prime development tool and MATLAB/Simulink Interface
-- 
-- Legal Notice: Copyright 2017 Intel Corporation.  All rights reserved.
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

-- VHDL created from mulFPF64_0002
-- VHDL created on Fri Sep 05 21:36:49 2025


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

entity mulFPF64_0002 is
    port (
        a : in std_logic_vector(63 downto 0);  -- float64_m52
        b : in std_logic_vector(63 downto 0);  -- float64_m52
        q : out std_logic_vector(63 downto 0);  -- float64_m52
        clk : in std_logic;
        areset : in std_logic
    );
end mulFPF64_0002;

architecture normal of mulFPF64_0002 is

    attribute altera_attribute : string;
    attribute altera_attribute of normal : architecture is "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name PHYSICAL_SYNTHESIS_REGISTER_DUPLICATION ON; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007";
    
    signal GND_q : STD_LOGIC_VECTOR (0 downto 0);
    signal VCC_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expX_uid6_fpMulTest_b : STD_LOGIC_VECTOR (10 downto 0);
    signal expY_uid7_fpMulTest_b : STD_LOGIC_VECTOR (10 downto 0);
    signal signX_uid8_fpMulTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signY_uid9_fpMulTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal cstAllOWE_uid10_fpMulTest_q : STD_LOGIC_VECTOR (10 downto 0);
    signal cstZeroWF_uid11_fpMulTest_q : STD_LOGIC_VECTOR (51 downto 0);
    signal cstAllZWE_uid12_fpMulTest_q : STD_LOGIC_VECTOR (10 downto 0);
    signal frac_x_uid14_fpMulTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal excZ_x_uid15_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excZ_x_uid15_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid16_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid16_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsNotZero_uid18_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_x_uid19_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_x_uid20_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_x_uid20_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invExpXIsMax_uid21_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal InvExpXIsZero_uid22_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excR_x_uid23_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal frac_y_uid28_fpMulTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal excZ_y_uid29_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excZ_y_uid29_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid30_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid30_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsNotZero_uid32_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_y_uid33_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_y_uid34_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_y_uid34_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invExpXIsMax_uid35_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal InvExpXIsZero_uid36_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excR_y_uid37_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal ofracX_uid40_fpMulTest_q : STD_LOGIC_VECTOR (52 downto 0);
    signal ofracY_uid43_fpMulTest_q : STD_LOGIC_VECTOR (52 downto 0);
    signal expSum_uid44_fpMulTest_a : STD_LOGIC_VECTOR (11 downto 0);
    signal expSum_uid44_fpMulTest_b : STD_LOGIC_VECTOR (11 downto 0);
    signal expSum_uid44_fpMulTest_o : STD_LOGIC_VECTOR (11 downto 0);
    signal expSum_uid44_fpMulTest_q : STD_LOGIC_VECTOR (11 downto 0);
    signal biasInc_uid45_fpMulTest_q : STD_LOGIC_VECTOR (12 downto 0);
    signal expSumMBias_uid46_fpMulTest_a : STD_LOGIC_VECTOR (14 downto 0);
    signal expSumMBias_uid46_fpMulTest_b : STD_LOGIC_VECTOR (14 downto 0);
    signal expSumMBias_uid46_fpMulTest_o : STD_LOGIC_VECTOR (14 downto 0);
    signal expSumMBias_uid46_fpMulTest_q : STD_LOGIC_VECTOR (13 downto 0);
    signal signR_uid48_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal signR_uid48_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracPreRound_uid55_fpMulTest_q : STD_LOGIC_VECTOR (66 downto 0);
    signal fracRPreExc_uid59_fpMulTest_in : STD_LOGIC_VECTOR (52 downto 0);
    signal fracRPreExc_uid59_fpMulTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal expRPreExcExt_uid60_fpMulTest_b : STD_LOGIC_VECTOR (14 downto 0);
    signal expRPreExc_uid61_fpMulTest_in : STD_LOGIC_VECTOR (10 downto 0);
    signal expRPreExc_uid61_fpMulTest_b : STD_LOGIC_VECTOR (10 downto 0);
    signal expUdf_uid62_fpMulTest_a : STD_LOGIC_VECTOR (16 downto 0);
    signal expUdf_uid62_fpMulTest_b : STD_LOGIC_VECTOR (16 downto 0);
    signal expUdf_uid62_fpMulTest_o : STD_LOGIC_VECTOR (16 downto 0);
    signal expUdf_uid62_fpMulTest_n : STD_LOGIC_VECTOR (0 downto 0);
    signal expOvf_uid64_fpMulTest_a : STD_LOGIC_VECTOR (16 downto 0);
    signal expOvf_uid64_fpMulTest_b : STD_LOGIC_VECTOR (16 downto 0);
    signal expOvf_uid64_fpMulTest_o : STD_LOGIC_VECTOR (16 downto 0);
    signal expOvf_uid64_fpMulTest_n : STD_LOGIC_VECTOR (0 downto 0);
    signal excXZAndExcYZ_uid65_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excXZAndExcYR_uid66_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excYZAndExcXR_uid67_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excZC3_uid68_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRZero_uid69_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excRZero_uid69_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excXIAndExcYI_uid70_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excXRAndExcYI_uid71_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excYRAndExcXI_uid72_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal ExcROvfAndInReg_uid73_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRInf_uid74_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excRInf_uid74_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excYZAndExcXI_uid75_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excXZAndExcYI_uid76_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal ZeroTimesInf_uid77_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal ZeroTimesInf_uid77_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRNaN_uid78_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal concExc_uid79_fpMulTest_q : STD_LOGIC_VECTOR (2 downto 0);
    signal excREnc_uid80_fpMulTest_q : STD_LOGIC_VECTOR (1 downto 0);
    signal oneFracRPostExc2_uid81_fpMulTest_q : STD_LOGIC_VECTOR (51 downto 0);
    signal fracRPostExc_uid84_fpMulTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal fracRPostExc_uid84_fpMulTest_q : STD_LOGIC_VECTOR (51 downto 0);
    signal expRPostExc_uid89_fpMulTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal expRPostExc_uid89_fpMulTest_q : STD_LOGIC_VECTOR (10 downto 0);
    signal invExcRNaN_uid90_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal signRPostExc_uid91_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal signRPostExc_uid91_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal R_uid92_fpMulTest_q : STD_LOGIC_VECTOR (63 downto 0);
    signal topRangeX_uid104_prod_uid47_fpMulTest_b : STD_LOGIC_VECTOR (26 downto 0);
    signal topRangeY_uid105_prod_uid47_fpMulTest_b : STD_LOGIC_VECTOR (26 downto 0);
    signal aboveLeftY_bottomRange_uid108_prod_uid47_fpMulTest_in : STD_LOGIC_VECTOR (25 downto 0);
    signal aboveLeftY_bottomRange_uid108_prod_uid47_fpMulTest_b : STD_LOGIC_VECTOR (25 downto 0);
    signal aboveLeftY_mergedSignalTM_uid109_prod_uid47_fpMulTest_q : STD_LOGIC_VECTOR (26 downto 0);
    signal rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_in : STD_LOGIC_VECTOR (25 downto 0);
    signal rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_b : STD_LOGIC_VECTOR (25 downto 0);
    signal rightBottomX_mergedSignalTM_uid113_prod_uid47_fpMulTest_q : STD_LOGIC_VECTOR (26 downto 0);
    signal aboveLeftX_uid116_prod_uid47_fpMulTest_in : STD_LOGIC_VECTOR (25 downto 0);
    signal aboveLeftX_uid116_prod_uid47_fpMulTest_b : STD_LOGIC_VECTOR (4 downto 0);
    signal aboveLeftY_uid117_prod_uid47_fpMulTest_in : STD_LOGIC_VECTOR (25 downto 0);
    signal aboveLeftY_uid117_prod_uid47_fpMulTest_b : STD_LOGIC_VECTOR (4 downto 0);
    signal sm0_uid121_prod_uid47_fpMulTest_a0 : STD_LOGIC_VECTOR (4 downto 0);
    signal sm0_uid121_prod_uid47_fpMulTest_b0 : STD_LOGIC_VECTOR (4 downto 0);
    signal sm0_uid121_prod_uid47_fpMulTest_s1 : STD_LOGIC_VECTOR (9 downto 0);
    signal sm0_uid121_prod_uid47_fpMulTest_pr : UNSIGNED (9 downto 0);
    attribute multstyle : string;
    attribute multstyle of sm0_uid121_prod_uid47_fpMulTest_pr : signal is "logic";
    signal sm0_uid121_prod_uid47_fpMulTest_q : STD_LOGIC_VECTOR (9 downto 0);
    signal sumAb_uid122_prod_uid47_fpMulTest_q : STD_LOGIC_VECTOR (63 downto 0);
    signal highBBits_uid124_prod_uid47_fpMulTest_b : STD_LOGIC_VECTOR (37 downto 0);
    signal c0_uid130_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid130_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq0_uid131_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c1_uid133_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (11 downto 0);
    signal c1_uid133_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq1_uid134_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c2_uid136_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (17 downto 0);
    signal c2_uid136_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq2_uid137_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c3_uid139_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (23 downto 0);
    signal c3_uid139_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq3_uid140_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c4_uid142_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (29 downto 0);
    signal c4_uid142_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq4_uid143_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c5_uid145_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (35 downto 0);
    signal c5_uid145_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq5_uid146_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c6_uid148_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (41 downto 0);
    signal c6_uid148_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq6_uid149_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c7_uid151_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (47 downto 0);
    signal c7_uid151_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq7_uid152_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c8_uid154_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (3 downto 0);
    signal eq8_uid155_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid156_fracXIsZero_uid17_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid156_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid157_fracXIsZero_uid17_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid157_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev1_uid158_fracXIsZero_uid17_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq0_uid161_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq1_uid164_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq2_uid167_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq3_uid170_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq4_uid173_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq5_uid176_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq6_uid179_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq7_uid182_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq8_uid185_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid186_fracXIsZero_uid31_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid186_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid187_fracXIsZero_uid31_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid187_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev1_uid188_fracXIsZero_uid31_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitExpansion_for_a_q : STD_LOGIC_VECTOR (67 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_SignBit_for_a_b : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_UpperBits_for_b_q : STD_LOGIC_VECTOR (12 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_c : STD_LOGIC_VECTOR (28 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p1_of_2_a : STD_LOGIC_VECTOR (39 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p1_of_2_b : STD_LOGIC_VECTOR (39 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p1_of_2_o : STD_LOGIC_VECTOR (39 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p1_of_2_c : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p1_of_2_q : STD_LOGIC_VECTOR (38 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p2_of_2_a : STD_LOGIC_VECTOR (30 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p2_of_2_b : STD_LOGIC_VECTOR (30 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p2_of_2_o : STD_LOGIC_VECTOR (30 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p2_of_2_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p2_of_2_q : STD_LOGIC_VECTOR (28 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (67 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_a_q : STD_LOGIC_VECTOR (64 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_b_q : STD_LOGIC_VECTOR (64 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_UpperBits_for_b_q : STD_LOGIC_VECTOR (26 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_b : STD_LOGIC_VECTOR (38 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_c : STD_LOGIC_VECTOR (25 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_b : STD_LOGIC_VECTOR (38 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_a : STD_LOGIC_VECTOR (39 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_b : STD_LOGIC_VECTOR (39 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_o : STD_LOGIC_VECTOR (39 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_c : STD_LOGIC_VECTOR (0 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_q : STD_LOGIC_VECTOR (38 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_a : STD_LOGIC_VECTOR (27 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_b : STD_LOGIC_VECTOR (27 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_o : STD_LOGIC_VECTOR (27 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_q : STD_LOGIC_VECTOR (25 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p0_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p0_q : STD_LOGIC_VECTOR (28 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p1_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p2_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p2_q : STD_LOGIC_VECTOR (22 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (52 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel0_2_b : STD_LOGIC_VECTOR (8 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_b_q : STD_LOGIC_VECTOR (38 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel0_1_b : STD_LOGIC_VECTOR (37 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_b_q : STD_LOGIC_VECTOR (38 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_0_b : STD_LOGIC_VECTOR (13 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_1_b : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_c_q : STD_LOGIC_VECTOR (28 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_tessel1_0_b : STD_LOGIC_VECTOR (25 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel0_0_b : STD_LOGIC_VECTOR (28 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel1_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel2_0_b : STD_LOGIC_VECTOR (22 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel0_0_b : STD_LOGIC_VECTOR (28 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel1_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel2_0_b : STD_LOGIC_VECTOR (22 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_cma_reset : std_logic;
    type sm0_uid118_prod_uid47_fpMulTest_cma_a0type is array(NATURAL range <>) of UNSIGNED(26 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_cma_a0 : sm0_uid118_prod_uid47_fpMulTest_cma_a0type(0 to 0);
    attribute preserve : boolean;
    attribute preserve of sm0_uid118_prod_uid47_fpMulTest_cma_a0 : signal is true;
    signal sm0_uid118_prod_uid47_fpMulTest_cma_c0 : sm0_uid118_prod_uid47_fpMulTest_cma_a0type(0 to 0);
    attribute preserve of sm0_uid118_prod_uid47_fpMulTest_cma_c0 : signal is true;
    type sm0_uid118_prod_uid47_fpMulTest_cma_ptype is array(NATURAL range <>) of UNSIGNED(53 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_cma_p : sm0_uid118_prod_uid47_fpMulTest_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_cma_u : sm0_uid118_prod_uid47_fpMulTest_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_cma_w : sm0_uid118_prod_uid47_fpMulTest_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_cma_x : sm0_uid118_prod_uid47_fpMulTest_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_cma_y : sm0_uid118_prod_uid47_fpMulTest_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_cma_s : sm0_uid118_prod_uid47_fpMulTest_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_cma_qq : STD_LOGIC_VECTOR (53 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_cma_q : STD_LOGIC_VECTOR (53 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_cma_ena0 : std_logic;
    signal sm0_uid118_prod_uid47_fpMulTest_cma_ena1 : std_logic;
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_reset : std_logic;
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_a0 : sm0_uid118_prod_uid47_fpMulTest_cma_a0type(0 to 1);
    attribute preserve of multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_a0 : signal is true;
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_c0 : sm0_uid118_prod_uid47_fpMulTest_cma_a0type(0 to 1);
    attribute preserve of multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_c0 : signal is true;
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_p : sm0_uid118_prod_uid47_fpMulTest_cma_ptype(0 to 1);
    type multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_utype is array(NATURAL range <>) of UNSIGNED(54 downto 0);
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_u : multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_utype(0 to 1);
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_w : multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_utype(0 to 1);
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_x : multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_utype(0 to 1);
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_y : multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_utype(0 to 1);
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_s : multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_utype(0 to 1);
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_qq : STD_LOGIC_VECTOR (54 downto 0);
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_q : STD_LOGIC_VECTOR (54 downto 0);
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_ena0 : std_logic;
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_ena1 : std_logic;
    signal z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_b : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_c : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_d : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_e : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_f : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_g : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_h : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_i : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_j : STD_LOGIC_VECTOR (3 downto 0);
    signal z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_b : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_c : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_d : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_e : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_f : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_g : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_h : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_i : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_j : STD_LOGIC_VECTOR (3 downto 0);
    signal redist0_sm0_uid118_prod_uid47_fpMulTest_cma_q_1_q : STD_LOGIC_VECTOR (53 downto 0);
    signal redist1_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_1_b_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist2_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_q_1_q : STD_LOGIC_VECTOR (38 downto 0);
    signal redist3_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_c_1_q : STD_LOGIC_VECTOR (25 downto 0);
    signal redist4_expFracRPostRounding_uid58_fpMulTest_p1_of_2_q_1_q : STD_LOGIC_VECTOR (38 downto 0);
    signal redist5_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_c_1_q : STD_LOGIC_VECTOR (28 downto 0);
    signal redist6_and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q_9_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist7_and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q_9_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist8_highBBits_uid124_prod_uid47_fpMulTest_b_1_q : STD_LOGIC_VECTOR (37 downto 0);
    signal redist9_sm0_uid121_prod_uid47_fpMulTest_q_1_q : STD_LOGIC_VECTOR (9 downto 0);
    signal redist10_aboveLeftY_uid117_prod_uid47_fpMulTest_b_1_q : STD_LOGIC_VECTOR (4 downto 0);
    signal redist11_aboveLeftX_uid116_prod_uid47_fpMulTest_b_1_q : STD_LOGIC_VECTOR (4 downto 0);
    signal redist12_rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_b_1_q : STD_LOGIC_VECTOR (25 downto 0);
    signal redist13_topRangeY_uid105_prod_uid47_fpMulTest_b_1_q : STD_LOGIC_VECTOR (26 downto 0);
    signal redist14_topRangeX_uid104_prod_uid47_fpMulTest_b_1_q : STD_LOGIC_VECTOR (26 downto 0);
    signal redist15_expRPreExc_uid61_fpMulTest_b_3_q : STD_LOGIC_VECTOR (10 downto 0);
    signal redist16_fracRPreExc_uid59_fpMulTest_b_3_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist17_signR_uid48_fpMulTest_q_11_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist18_expSum_uid44_fpMulTest_q_7_q : STD_LOGIC_VECTOR (11 downto 0);
    signal redist19_expXIsMax_uid30_fpMulTest_q_10_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist20_excZ_y_uid29_fpMulTest_q_10_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist21_expXIsMax_uid16_fpMulTest_q_10_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist22_excZ_x_uid15_fpMulTest_q_10_q : STD_LOGIC_VECTOR (0 downto 0);

begin


    -- cstZeroWF_uid11_fpMulTest(CONSTANT,10)
    cstZeroWF_uid11_fpMulTest_q <= "0000000000000000000000000000000000000000000000000000";

    -- c8_uid154_fracXIsZero_uid17_fpMulTest(BITSELECT,153)
    c8_uid154_fracXIsZero_uid17_fpMulTest_b <= cstZeroWF_uid11_fpMulTest_q(51 downto 48);

    -- frac_x_uid14_fpMulTest(BITSELECT,13)@0
    frac_x_uid14_fpMulTest_b <= a(51 downto 0);

    -- z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select(BITSELECT,243)@0
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_b <= frac_x_uid14_fpMulTest_b(5 downto 0);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_c <= frac_x_uid14_fpMulTest_b(11 downto 6);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_d <= frac_x_uid14_fpMulTest_b(17 downto 12);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_e <= frac_x_uid14_fpMulTest_b(23 downto 18);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_f <= frac_x_uid14_fpMulTest_b(29 downto 24);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_g <= frac_x_uid14_fpMulTest_b(35 downto 30);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_h <= frac_x_uid14_fpMulTest_b(41 downto 36);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_i <= frac_x_uid14_fpMulTest_b(47 downto 42);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_j <= frac_x_uid14_fpMulTest_b(51 downto 48);

    -- eq8_uid155_fracXIsZero_uid17_fpMulTest(LOGICAL,154)@0
    eq8_uid155_fracXIsZero_uid17_fpMulTest_q <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_j = c8_uid154_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- c7_uid151_fracXIsZero_uid17_fpMulTest(BITSELECT,150)
    c7_uid151_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(47 downto 0);
    c7_uid151_fracXIsZero_uid17_fpMulTest_b <= c7_uid151_fracXIsZero_uid17_fpMulTest_in(47 downto 42);

    -- eq7_uid152_fracXIsZero_uid17_fpMulTest(LOGICAL,151)@0
    eq7_uid152_fracXIsZero_uid17_fpMulTest_q <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_i = c7_uid151_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- c6_uid148_fracXIsZero_uid17_fpMulTest(BITSELECT,147)
    c6_uid148_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(41 downto 0);
    c6_uid148_fracXIsZero_uid17_fpMulTest_b <= c6_uid148_fracXIsZero_uid17_fpMulTest_in(41 downto 36);

    -- eq6_uid149_fracXIsZero_uid17_fpMulTest(LOGICAL,148)@0
    eq6_uid149_fracXIsZero_uid17_fpMulTest_q <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_h = c6_uid148_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- and_lev0_uid157_fracXIsZero_uid17_fpMulTest(LOGICAL,156)@0 + 1
    and_lev0_uid157_fracXIsZero_uid17_fpMulTest_qi <= eq6_uid149_fracXIsZero_uid17_fpMulTest_q and eq7_uid152_fracXIsZero_uid17_fpMulTest_q and eq8_uid155_fracXIsZero_uid17_fpMulTest_q;
    and_lev0_uid157_fracXIsZero_uid17_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev0_uid157_fracXIsZero_uid17_fpMulTest_qi, xout => and_lev0_uid157_fracXIsZero_uid17_fpMulTest_q, clk => clk, aclr => areset );

    -- c5_uid145_fracXIsZero_uid17_fpMulTest(BITSELECT,144)
    c5_uid145_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(35 downto 0);
    c5_uid145_fracXIsZero_uid17_fpMulTest_b <= c5_uid145_fracXIsZero_uid17_fpMulTest_in(35 downto 30);

    -- eq5_uid146_fracXIsZero_uid17_fpMulTest(LOGICAL,145)@0
    eq5_uid146_fracXIsZero_uid17_fpMulTest_q <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_g = c5_uid145_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- c4_uid142_fracXIsZero_uid17_fpMulTest(BITSELECT,141)
    c4_uid142_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(29 downto 0);
    c4_uid142_fracXIsZero_uid17_fpMulTest_b <= c4_uid142_fracXIsZero_uid17_fpMulTest_in(29 downto 24);

    -- eq4_uid143_fracXIsZero_uid17_fpMulTest(LOGICAL,142)@0
    eq4_uid143_fracXIsZero_uid17_fpMulTest_q <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_f = c4_uid142_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- c3_uid139_fracXIsZero_uid17_fpMulTest(BITSELECT,138)
    c3_uid139_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(23 downto 0);
    c3_uid139_fracXIsZero_uid17_fpMulTest_b <= c3_uid139_fracXIsZero_uid17_fpMulTest_in(23 downto 18);

    -- eq3_uid140_fracXIsZero_uid17_fpMulTest(LOGICAL,139)@0
    eq3_uid140_fracXIsZero_uid17_fpMulTest_q <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_e = c3_uid139_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- c2_uid136_fracXIsZero_uid17_fpMulTest(BITSELECT,135)
    c2_uid136_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(17 downto 0);
    c2_uid136_fracXIsZero_uid17_fpMulTest_b <= c2_uid136_fracXIsZero_uid17_fpMulTest_in(17 downto 12);

    -- eq2_uid137_fracXIsZero_uid17_fpMulTest(LOGICAL,136)@0
    eq2_uid137_fracXIsZero_uid17_fpMulTest_q <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_d = c2_uid136_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- c1_uid133_fracXIsZero_uid17_fpMulTest(BITSELECT,132)
    c1_uid133_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(11 downto 0);
    c1_uid133_fracXIsZero_uid17_fpMulTest_b <= c1_uid133_fracXIsZero_uid17_fpMulTest_in(11 downto 6);

    -- eq1_uid134_fracXIsZero_uid17_fpMulTest(LOGICAL,133)@0
    eq1_uid134_fracXIsZero_uid17_fpMulTest_q <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_c = c1_uid133_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- c0_uid130_fracXIsZero_uid17_fpMulTest(BITSELECT,129)
    c0_uid130_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(5 downto 0);
    c0_uid130_fracXIsZero_uid17_fpMulTest_b <= c0_uid130_fracXIsZero_uid17_fpMulTest_in(5 downto 0);

    -- eq0_uid131_fracXIsZero_uid17_fpMulTest(LOGICAL,130)@0
    eq0_uid131_fracXIsZero_uid17_fpMulTest_q <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_b = c0_uid130_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- and_lev0_uid156_fracXIsZero_uid17_fpMulTest(LOGICAL,155)@0 + 1
    and_lev0_uid156_fracXIsZero_uid17_fpMulTest_qi <= eq0_uid131_fracXIsZero_uid17_fpMulTest_q and eq1_uid134_fracXIsZero_uid17_fpMulTest_q and eq2_uid137_fracXIsZero_uid17_fpMulTest_q and eq3_uid140_fracXIsZero_uid17_fpMulTest_q and eq4_uid143_fracXIsZero_uid17_fpMulTest_q and eq5_uid146_fracXIsZero_uid17_fpMulTest_q;
    and_lev0_uid156_fracXIsZero_uid17_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev0_uid156_fracXIsZero_uid17_fpMulTest_qi, xout => and_lev0_uid156_fracXIsZero_uid17_fpMulTest_q, clk => clk, aclr => areset );

    -- and_lev1_uid158_fracXIsZero_uid17_fpMulTest(LOGICAL,157)@1 + 1
    and_lev1_uid158_fracXIsZero_uid17_fpMulTest_qi <= and_lev0_uid156_fracXIsZero_uid17_fpMulTest_q and and_lev0_uid157_fracXIsZero_uid17_fpMulTest_q;
    and_lev1_uid158_fracXIsZero_uid17_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev1_uid158_fracXIsZero_uid17_fpMulTest_qi, xout => and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q, clk => clk, aclr => areset );

    -- redist7_and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q_9(DELAY,252)
    redist7_and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q_9 : dspba_delay
    GENERIC MAP ( width => 1, depth => 8, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q, xout => redist7_and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q_9_q, clk => clk, aclr => areset );

    -- cstAllOWE_uid10_fpMulTest(CONSTANT,9)
    cstAllOWE_uid10_fpMulTest_q <= "11111111111";

    -- expX_uid6_fpMulTest(BITSELECT,5)@0
    expX_uid6_fpMulTest_b <= a(62 downto 52);

    -- expXIsMax_uid16_fpMulTest(LOGICAL,15)@0 + 1
    expXIsMax_uid16_fpMulTest_qi <= "1" WHEN expX_uid6_fpMulTest_b = cstAllOWE_uid10_fpMulTest_q ELSE "0";
    expXIsMax_uid16_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => expXIsMax_uid16_fpMulTest_qi, xout => expXIsMax_uid16_fpMulTest_q, clk => clk, aclr => areset );

    -- redist21_expXIsMax_uid16_fpMulTest_q_10(DELAY,266)
    redist21_expXIsMax_uid16_fpMulTest_q_10 : dspba_delay
    GENERIC MAP ( width => 1, depth => 9, reset_kind => "ASYNC" )
    PORT MAP ( xin => expXIsMax_uid16_fpMulTest_q, xout => redist21_expXIsMax_uid16_fpMulTest_q_10_q, clk => clk, aclr => areset );

    -- excI_x_uid19_fpMulTest(LOGICAL,18)@10
    excI_x_uid19_fpMulTest_q <= redist21_expXIsMax_uid16_fpMulTest_q_10_q and redist7_and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q_9_q;

    -- cstAllZWE_uid12_fpMulTest(CONSTANT,11)
    cstAllZWE_uid12_fpMulTest_q <= "00000000000";

    -- expY_uid7_fpMulTest(BITSELECT,6)@0
    expY_uid7_fpMulTest_b <= b(62 downto 52);

    -- excZ_y_uid29_fpMulTest(LOGICAL,28)@0 + 1
    excZ_y_uid29_fpMulTest_qi <= "1" WHEN expY_uid7_fpMulTest_b = cstAllZWE_uid12_fpMulTest_q ELSE "0";
    excZ_y_uid29_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excZ_y_uid29_fpMulTest_qi, xout => excZ_y_uid29_fpMulTest_q, clk => clk, aclr => areset );

    -- redist20_excZ_y_uid29_fpMulTest_q_10(DELAY,265)
    redist20_excZ_y_uid29_fpMulTest_q_10 : dspba_delay
    GENERIC MAP ( width => 1, depth => 9, reset_kind => "ASYNC" )
    PORT MAP ( xin => excZ_y_uid29_fpMulTest_q, xout => redist20_excZ_y_uid29_fpMulTest_q_10_q, clk => clk, aclr => areset );

    -- excYZAndExcXI_uid75_fpMulTest(LOGICAL,74)@10
    excYZAndExcXI_uid75_fpMulTest_q <= redist20_excZ_y_uid29_fpMulTest_q_10_q and excI_x_uid19_fpMulTest_q;

    -- frac_y_uid28_fpMulTest(BITSELECT,27)@0
    frac_y_uid28_fpMulTest_b <= b(51 downto 0);

    -- z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select(BITSELECT,244)@0
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_b <= frac_y_uid28_fpMulTest_b(5 downto 0);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_c <= frac_y_uid28_fpMulTest_b(11 downto 6);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_d <= frac_y_uid28_fpMulTest_b(17 downto 12);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_e <= frac_y_uid28_fpMulTest_b(23 downto 18);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_f <= frac_y_uid28_fpMulTest_b(29 downto 24);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_g <= frac_y_uid28_fpMulTest_b(35 downto 30);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_h <= frac_y_uid28_fpMulTest_b(41 downto 36);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_i <= frac_y_uid28_fpMulTest_b(47 downto 42);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_j <= frac_y_uid28_fpMulTest_b(51 downto 48);

    -- eq8_uid185_fracXIsZero_uid31_fpMulTest(LOGICAL,184)@0
    eq8_uid185_fracXIsZero_uid31_fpMulTest_q <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_j = c8_uid154_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- eq7_uid182_fracXIsZero_uid31_fpMulTest(LOGICAL,181)@0
    eq7_uid182_fracXIsZero_uid31_fpMulTest_q <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_i = c7_uid151_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- eq6_uid179_fracXIsZero_uid31_fpMulTest(LOGICAL,178)@0
    eq6_uid179_fracXIsZero_uid31_fpMulTest_q <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_h = c6_uid148_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- and_lev0_uid187_fracXIsZero_uid31_fpMulTest(LOGICAL,186)@0 + 1
    and_lev0_uid187_fracXIsZero_uid31_fpMulTest_qi <= eq6_uid179_fracXIsZero_uid31_fpMulTest_q and eq7_uid182_fracXIsZero_uid31_fpMulTest_q and eq8_uid185_fracXIsZero_uid31_fpMulTest_q;
    and_lev0_uid187_fracXIsZero_uid31_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev0_uid187_fracXIsZero_uid31_fpMulTest_qi, xout => and_lev0_uid187_fracXIsZero_uid31_fpMulTest_q, clk => clk, aclr => areset );

    -- eq5_uid176_fracXIsZero_uid31_fpMulTest(LOGICAL,175)@0
    eq5_uid176_fracXIsZero_uid31_fpMulTest_q <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_g = c5_uid145_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- eq4_uid173_fracXIsZero_uid31_fpMulTest(LOGICAL,172)@0
    eq4_uid173_fracXIsZero_uid31_fpMulTest_q <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_f = c4_uid142_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- eq3_uid170_fracXIsZero_uid31_fpMulTest(LOGICAL,169)@0
    eq3_uid170_fracXIsZero_uid31_fpMulTest_q <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_e = c3_uid139_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- eq2_uid167_fracXIsZero_uid31_fpMulTest(LOGICAL,166)@0
    eq2_uid167_fracXIsZero_uid31_fpMulTest_q <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_d = c2_uid136_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- eq1_uid164_fracXIsZero_uid31_fpMulTest(LOGICAL,163)@0
    eq1_uid164_fracXIsZero_uid31_fpMulTest_q <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_c = c1_uid133_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- eq0_uid161_fracXIsZero_uid31_fpMulTest(LOGICAL,160)@0
    eq0_uid161_fracXIsZero_uid31_fpMulTest_q <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_b = c0_uid130_fracXIsZero_uid17_fpMulTest_b ELSE "0";

    -- and_lev0_uid186_fracXIsZero_uid31_fpMulTest(LOGICAL,185)@0 + 1
    and_lev0_uid186_fracXIsZero_uid31_fpMulTest_qi <= eq0_uid161_fracXIsZero_uid31_fpMulTest_q and eq1_uid164_fracXIsZero_uid31_fpMulTest_q and eq2_uid167_fracXIsZero_uid31_fpMulTest_q and eq3_uid170_fracXIsZero_uid31_fpMulTest_q and eq4_uid173_fracXIsZero_uid31_fpMulTest_q and eq5_uid176_fracXIsZero_uid31_fpMulTest_q;
    and_lev0_uid186_fracXIsZero_uid31_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev0_uid186_fracXIsZero_uid31_fpMulTest_qi, xout => and_lev0_uid186_fracXIsZero_uid31_fpMulTest_q, clk => clk, aclr => areset );

    -- and_lev1_uid188_fracXIsZero_uid31_fpMulTest(LOGICAL,187)@1 + 1
    and_lev1_uid188_fracXIsZero_uid31_fpMulTest_qi <= and_lev0_uid186_fracXIsZero_uid31_fpMulTest_q and and_lev0_uid187_fracXIsZero_uid31_fpMulTest_q;
    and_lev1_uid188_fracXIsZero_uid31_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev1_uid188_fracXIsZero_uid31_fpMulTest_qi, xout => and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q, clk => clk, aclr => areset );

    -- redist6_and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q_9(DELAY,251)
    redist6_and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q_9 : dspba_delay
    GENERIC MAP ( width => 1, depth => 8, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q, xout => redist6_and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q_9_q, clk => clk, aclr => areset );

    -- expXIsMax_uid30_fpMulTest(LOGICAL,29)@0 + 1
    expXIsMax_uid30_fpMulTest_qi <= "1" WHEN expY_uid7_fpMulTest_b = cstAllOWE_uid10_fpMulTest_q ELSE "0";
    expXIsMax_uid30_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => expXIsMax_uid30_fpMulTest_qi, xout => expXIsMax_uid30_fpMulTest_q, clk => clk, aclr => areset );

    -- redist19_expXIsMax_uid30_fpMulTest_q_10(DELAY,264)
    redist19_expXIsMax_uid30_fpMulTest_q_10 : dspba_delay
    GENERIC MAP ( width => 1, depth => 9, reset_kind => "ASYNC" )
    PORT MAP ( xin => expXIsMax_uid30_fpMulTest_q, xout => redist19_expXIsMax_uid30_fpMulTest_q_10_q, clk => clk, aclr => areset );

    -- excI_y_uid33_fpMulTest(LOGICAL,32)@10
    excI_y_uid33_fpMulTest_q <= redist19_expXIsMax_uid30_fpMulTest_q_10_q and redist6_and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q_9_q;

    -- excZ_x_uid15_fpMulTest(LOGICAL,14)@0 + 1
    excZ_x_uid15_fpMulTest_qi <= "1" WHEN expX_uid6_fpMulTest_b = cstAllZWE_uid12_fpMulTest_q ELSE "0";
    excZ_x_uid15_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excZ_x_uid15_fpMulTest_qi, xout => excZ_x_uid15_fpMulTest_q, clk => clk, aclr => areset );

    -- redist22_excZ_x_uid15_fpMulTest_q_10(DELAY,267)
    redist22_excZ_x_uid15_fpMulTest_q_10 : dspba_delay
    GENERIC MAP ( width => 1, depth => 9, reset_kind => "ASYNC" )
    PORT MAP ( xin => excZ_x_uid15_fpMulTest_q, xout => redist22_excZ_x_uid15_fpMulTest_q_10_q, clk => clk, aclr => areset );

    -- excXZAndExcYI_uid76_fpMulTest(LOGICAL,75)@10
    excXZAndExcYI_uid76_fpMulTest_q <= redist22_excZ_x_uid15_fpMulTest_q_10_q and excI_y_uid33_fpMulTest_q;

    -- ZeroTimesInf_uid77_fpMulTest(LOGICAL,76)@10 + 1
    ZeroTimesInf_uid77_fpMulTest_qi <= excXZAndExcYI_uid76_fpMulTest_q or excYZAndExcXI_uid75_fpMulTest_q;
    ZeroTimesInf_uid77_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => ZeroTimesInf_uid77_fpMulTest_qi, xout => ZeroTimesInf_uid77_fpMulTest_q, clk => clk, aclr => areset );

    -- fracXIsNotZero_uid32_fpMulTest(LOGICAL,31)@10
    fracXIsNotZero_uid32_fpMulTest_q <= not (redist6_and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q_9_q);

    -- excN_y_uid34_fpMulTest(LOGICAL,33)@10 + 1
    excN_y_uid34_fpMulTest_qi <= redist19_expXIsMax_uid30_fpMulTest_q_10_q and fracXIsNotZero_uid32_fpMulTest_q;
    excN_y_uid34_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excN_y_uid34_fpMulTest_qi, xout => excN_y_uid34_fpMulTest_q, clk => clk, aclr => areset );

    -- fracXIsNotZero_uid18_fpMulTest(LOGICAL,17)@10
    fracXIsNotZero_uid18_fpMulTest_q <= not (redist7_and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q_9_q);

    -- excN_x_uid20_fpMulTest(LOGICAL,19)@10 + 1
    excN_x_uid20_fpMulTest_qi <= redist21_expXIsMax_uid16_fpMulTest_q_10_q and fracXIsNotZero_uid18_fpMulTest_q;
    excN_x_uid20_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excN_x_uid20_fpMulTest_qi, xout => excN_x_uid20_fpMulTest_q, clk => clk, aclr => areset );

    -- excRNaN_uid78_fpMulTest(LOGICAL,77)@11
    excRNaN_uid78_fpMulTest_q <= excN_x_uid20_fpMulTest_q or excN_y_uid34_fpMulTest_q or ZeroTimesInf_uid77_fpMulTest_q;

    -- invExcRNaN_uid90_fpMulTest(LOGICAL,89)@11
    invExcRNaN_uid90_fpMulTest_q <= not (excRNaN_uid78_fpMulTest_q);

    -- signY_uid9_fpMulTest(BITSELECT,8)@0
    signY_uid9_fpMulTest_b <= STD_LOGIC_VECTOR(b(63 downto 63));

    -- signX_uid8_fpMulTest(BITSELECT,7)@0
    signX_uid8_fpMulTest_b <= STD_LOGIC_VECTOR(a(63 downto 63));

    -- signR_uid48_fpMulTest(LOGICAL,47)@0 + 1
    signR_uid48_fpMulTest_qi <= signX_uid8_fpMulTest_b xor signY_uid9_fpMulTest_b;
    signR_uid48_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => signR_uid48_fpMulTest_qi, xout => signR_uid48_fpMulTest_q, clk => clk, aclr => areset );

    -- redist17_signR_uid48_fpMulTest_q_11(DELAY,262)
    redist17_signR_uid48_fpMulTest_q_11 : dspba_delay
    GENERIC MAP ( width => 1, depth => 10, reset_kind => "ASYNC" )
    PORT MAP ( xin => signR_uid48_fpMulTest_q, xout => redist17_signR_uid48_fpMulTest_q_11_q, clk => clk, aclr => areset );

    -- VCC(CONSTANT,1)
    VCC_q <= "1";

    -- signRPostExc_uid91_fpMulTest(LOGICAL,90)@11 + 1
    signRPostExc_uid91_fpMulTest_qi <= redist17_signR_uid48_fpMulTest_q_11_q and invExcRNaN_uid90_fpMulTest_q;
    signRPostExc_uid91_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => signRPostExc_uid91_fpMulTest_qi, xout => signRPostExc_uid91_fpMulTest_q, clk => clk, aclr => areset );

    -- GND(CONSTANT,0)
    GND_q <= "0";

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel0_1(BITSELECT,218)
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel0_1_b <= STD_LOGIC_VECTOR(cstZeroWF_uid11_fpMulTest_q(37 downto 0));

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_b(BITJOIN,219)@7
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_b_q <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel0_1_b & VCC_q;

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_UpperBits_for_b(CONSTANT,201)
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_UpperBits_for_b_q <= "000000000000000000000000000";

    -- ofracX_uid40_fpMulTest(BITJOIN,39)@0
    ofracX_uid40_fpMulTest_q <= VCC_q & frac_x_uid14_fpMulTest_b;

    -- topRangeX_uid104_prod_uid47_fpMulTest(BITSELECT,103)@0
    topRangeX_uid104_prod_uid47_fpMulTest_b <= ofracX_uid40_fpMulTest_q(52 downto 26);

    -- ofracY_uid43_fpMulTest(BITJOIN,42)@0
    ofracY_uid43_fpMulTest_q <= VCC_q & frac_y_uid28_fpMulTest_b;

    -- aboveLeftY_bottomRange_uid108_prod_uid47_fpMulTest(BITSELECT,107)@0
    aboveLeftY_bottomRange_uid108_prod_uid47_fpMulTest_in <= ofracY_uid43_fpMulTest_q(25 downto 0);
    aboveLeftY_bottomRange_uid108_prod_uid47_fpMulTest_b <= aboveLeftY_bottomRange_uid108_prod_uid47_fpMulTest_in(25 downto 0);

    -- aboveLeftY_mergedSignalTM_uid109_prod_uid47_fpMulTest(BITJOIN,108)@0
    aboveLeftY_mergedSignalTM_uid109_prod_uid47_fpMulTest_q <= aboveLeftY_bottomRange_uid108_prod_uid47_fpMulTest_b & GND_q;

    -- rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest(BITSELECT,111)@0
    rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_in <= ofracX_uid40_fpMulTest_q(25 downto 0);
    rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_b <= rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_in(25 downto 0);

    -- redist12_rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_b_1(DELAY,257)
    redist12_rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_b_1 : dspba_delay
    GENERIC MAP ( width => 26, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_b, xout => redist12_rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_b_1_q, clk => clk, aclr => areset );

    -- rightBottomX_mergedSignalTM_uid113_prod_uid47_fpMulTest(BITJOIN,112)@1
    rightBottomX_mergedSignalTM_uid113_prod_uid47_fpMulTest_q <= redist12_rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_b_1_q & GND_q;

    -- topRangeY_uid105_prod_uid47_fpMulTest(BITSELECT,104)@0
    topRangeY_uid105_prod_uid47_fpMulTest_b <= ofracY_uid43_fpMulTest_q(52 downto 26);

    -- redist13_topRangeY_uid105_prod_uid47_fpMulTest_b_1(DELAY,258)
    redist13_topRangeY_uid105_prod_uid47_fpMulTest_b_1 : dspba_delay
    GENERIC MAP ( width => 27, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => topRangeY_uid105_prod_uid47_fpMulTest_b, xout => redist13_topRangeY_uid105_prod_uid47_fpMulTest_b_1_q, clk => clk, aclr => areset );

    -- multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma(CHAINMULTADD,242)@0 + 2
    -- in e@1
    -- in g@1
    -- out q@3
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_reset <= areset;
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_ena0 <= '1';
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_ena1 <= multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_ena0;
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_p(0) <= multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_a0(0) * multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_c0(0);
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_p(1) <= multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_a0(1) * multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_c0(1);
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_u(0) <= RESIZE(multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_p(0),55);
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_u(1) <= RESIZE(multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_p(1),55);
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_w(0) <= multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_u(0);
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_w(1) <= multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_u(1);
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_x(0) <= multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_w(0);
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_x(1) <= multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_w(1);
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_y(0) <= multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_s(1) + multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_x(0);
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_y(1) <= multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_x(1);
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_chainmultadd_input: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_a0 <= (others => (others => '0'));
            multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_c0 <= (others => (others => '0'));
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_ena0 = '1') THEN
                multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_a0(0) <= RESIZE(UNSIGNED(redist13_topRangeY_uid105_prod_uid47_fpMulTest_b_1_q),27);
                multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_a0(1) <= RESIZE(UNSIGNED(aboveLeftY_mergedSignalTM_uid109_prod_uid47_fpMulTest_q),27);
                multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_c0(0) <= RESIZE(UNSIGNED(rightBottomX_mergedSignalTM_uid113_prod_uid47_fpMulTest_q),27);
                multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_c0(1) <= RESIZE(UNSIGNED(topRangeX_uid104_prod_uid47_fpMulTest_b),27);
            END IF;
        END IF;
    END PROCESS;
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_chainmultadd_output: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_s <= (others => (others => '0'));
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_ena1 = '1') THEN
                multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_s(0) <= multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_y(0);
                multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_s(1) <= multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_y(1);
            END IF;
        END IF;
    END PROCESS;
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_delay : dspba_delay
    GENERIC MAP ( width => 55, depth => 0, reset_kind => "ASYNC" )
    PORT MAP ( xin => STD_LOGIC_VECTOR(multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_s(0)(54 downto 0)), xout => multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_qq, clk => clk, aclr => areset );
    multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_q <= STD_LOGIC_VECTOR(multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_qq(54 downto 0));

    -- highBBits_uid124_prod_uid47_fpMulTest(BITSELECT,123)@3
    highBBits_uid124_prod_uid47_fpMulTest_b <= multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_q(54 downto 17);

    -- redist8_highBBits_uid124_prod_uid47_fpMulTest_b_1(DELAY,253)
    redist8_highBBits_uid124_prod_uid47_fpMulTest_b_1 : dspba_delay
    GENERIC MAP ( width => 38, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => highBBits_uid124_prod_uid47_fpMulTest_b, xout => redist8_highBBits_uid124_prod_uid47_fpMulTest_b_1_q, clk => clk, aclr => areset );

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_b(BITJOIN,200)@4
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_b_q <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_UpperBits_for_b_q & redist8_highBBits_uid124_prod_uid47_fpMulTest_b_1_q;

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b(BITSELECT,203)@4
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_b <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_b_q(38 downto 0);

    -- redist14_topRangeX_uid104_prod_uid47_fpMulTest_b_1(DELAY,259)
    redist14_topRangeX_uid104_prod_uid47_fpMulTest_b_1 : dspba_delay
    GENERIC MAP ( width => 27, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => topRangeX_uid104_prod_uid47_fpMulTest_b, xout => redist14_topRangeX_uid104_prod_uid47_fpMulTest_b_1_q, clk => clk, aclr => areset );

    -- sm0_uid118_prod_uid47_fpMulTest_cma(CHAINMULTADD,241)@1 + 2
    sm0_uid118_prod_uid47_fpMulTest_cma_reset <= areset;
    sm0_uid118_prod_uid47_fpMulTest_cma_ena0 <= '1';
    sm0_uid118_prod_uid47_fpMulTest_cma_ena1 <= sm0_uid118_prod_uid47_fpMulTest_cma_ena0;
    sm0_uid118_prod_uid47_fpMulTest_cma_p(0) <= sm0_uid118_prod_uid47_fpMulTest_cma_a0(0) * sm0_uid118_prod_uid47_fpMulTest_cma_c0(0);
    sm0_uid118_prod_uid47_fpMulTest_cma_u(0) <= RESIZE(sm0_uid118_prod_uid47_fpMulTest_cma_p(0),54);
    sm0_uid118_prod_uid47_fpMulTest_cma_w(0) <= sm0_uid118_prod_uid47_fpMulTest_cma_u(0);
    sm0_uid118_prod_uid47_fpMulTest_cma_x(0) <= sm0_uid118_prod_uid47_fpMulTest_cma_w(0);
    sm0_uid118_prod_uid47_fpMulTest_cma_y(0) <= sm0_uid118_prod_uid47_fpMulTest_cma_x(0);
    sm0_uid118_prod_uid47_fpMulTest_cma_chainmultadd_input: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_cma_a0 <= (others => (others => '0'));
            sm0_uid118_prod_uid47_fpMulTest_cma_c0 <= (others => (others => '0'));
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (sm0_uid118_prod_uid47_fpMulTest_cma_ena0 = '1') THEN
                sm0_uid118_prod_uid47_fpMulTest_cma_a0(0) <= RESIZE(UNSIGNED(redist14_topRangeX_uid104_prod_uid47_fpMulTest_b_1_q),27);
                sm0_uid118_prod_uid47_fpMulTest_cma_c0(0) <= RESIZE(UNSIGNED(redist13_topRangeY_uid105_prod_uid47_fpMulTest_b_1_q),27);
            END IF;
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_cma_chainmultadd_output: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_cma_s <= (others => (others => '0'));
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (sm0_uid118_prod_uid47_fpMulTest_cma_ena1 = '1') THEN
                sm0_uid118_prod_uid47_fpMulTest_cma_s(0) <= sm0_uid118_prod_uid47_fpMulTest_cma_y(0);
            END IF;
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_cma_delay : dspba_delay
    GENERIC MAP ( width => 54, depth => 0, reset_kind => "ASYNC" )
    PORT MAP ( xin => STD_LOGIC_VECTOR(sm0_uid118_prod_uid47_fpMulTest_cma_s(0)(53 downto 0)), xout => sm0_uid118_prod_uid47_fpMulTest_cma_qq, clk => clk, aclr => areset );
    sm0_uid118_prod_uid47_fpMulTest_cma_q <= STD_LOGIC_VECTOR(sm0_uid118_prod_uid47_fpMulTest_cma_qq(53 downto 0));

    -- redist0_sm0_uid118_prod_uid47_fpMulTest_cma_q_1(DELAY,245)
    redist0_sm0_uid118_prod_uid47_fpMulTest_cma_q_1 : dspba_delay
    GENERIC MAP ( width => 54, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_cma_q, xout => redist0_sm0_uid118_prod_uid47_fpMulTest_cma_q_1_q, clk => clk, aclr => areset );

    -- aboveLeftY_uid117_prod_uid47_fpMulTest(BITSELECT,116)@0
    aboveLeftY_uid117_prod_uid47_fpMulTest_in <= ofracY_uid43_fpMulTest_q(25 downto 0);
    aboveLeftY_uid117_prod_uid47_fpMulTest_b <= aboveLeftY_uid117_prod_uid47_fpMulTest_in(25 downto 21);

    -- redist10_aboveLeftY_uid117_prod_uid47_fpMulTest_b_1(DELAY,255)
    redist10_aboveLeftY_uid117_prod_uid47_fpMulTest_b_1 : dspba_delay
    GENERIC MAP ( width => 5, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aboveLeftY_uid117_prod_uid47_fpMulTest_b, xout => redist10_aboveLeftY_uid117_prod_uid47_fpMulTest_b_1_q, clk => clk, aclr => areset );

    -- aboveLeftX_uid116_prod_uid47_fpMulTest(BITSELECT,115)@0
    aboveLeftX_uid116_prod_uid47_fpMulTest_in <= ofracX_uid40_fpMulTest_q(25 downto 0);
    aboveLeftX_uid116_prod_uid47_fpMulTest_b <= aboveLeftX_uid116_prod_uid47_fpMulTest_in(25 downto 21);

    -- redist11_aboveLeftX_uid116_prod_uid47_fpMulTest_b_1(DELAY,256)
    redist11_aboveLeftX_uid116_prod_uid47_fpMulTest_b_1 : dspba_delay
    GENERIC MAP ( width => 5, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aboveLeftX_uid116_prod_uid47_fpMulTest_b, xout => redist11_aboveLeftX_uid116_prod_uid47_fpMulTest_b_1_q, clk => clk, aclr => areset );

    -- sm0_uid121_prod_uid47_fpMulTest(MULT,120)@1 + 2
    sm0_uid121_prod_uid47_fpMulTest_pr <= UNSIGNED(sm0_uid121_prod_uid47_fpMulTest_a0) * UNSIGNED(sm0_uid121_prod_uid47_fpMulTest_b0);
    sm0_uid121_prod_uid47_fpMulTest_component: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid121_prod_uid47_fpMulTest_a0 <= (others => '0');
            sm0_uid121_prod_uid47_fpMulTest_b0 <= (others => '0');
            sm0_uid121_prod_uid47_fpMulTest_s1 <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            sm0_uid121_prod_uid47_fpMulTest_a0 <= redist11_aboveLeftX_uid116_prod_uid47_fpMulTest_b_1_q;
            sm0_uid121_prod_uid47_fpMulTest_b0 <= redist10_aboveLeftY_uid117_prod_uid47_fpMulTest_b_1_q;
            sm0_uid121_prod_uid47_fpMulTest_s1 <= STD_LOGIC_VECTOR(sm0_uid121_prod_uid47_fpMulTest_pr);
        END IF;
    END PROCESS;
    sm0_uid121_prod_uid47_fpMulTest_q <= sm0_uid121_prod_uid47_fpMulTest_s1;

    -- redist9_sm0_uid121_prod_uid47_fpMulTest_q_1(DELAY,254)
    redist9_sm0_uid121_prod_uid47_fpMulTest_q_1 : dspba_delay
    GENERIC MAP ( width => 10, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid121_prod_uid47_fpMulTest_q, xout => redist9_sm0_uid121_prod_uid47_fpMulTest_q_1_q, clk => clk, aclr => areset );

    -- sumAb_uid122_prod_uid47_fpMulTest(BITJOIN,121)@4
    sumAb_uid122_prod_uid47_fpMulTest_q <= redist0_sm0_uid118_prod_uid47_fpMulTest_cma_q_1_q & redist9_sm0_uid121_prod_uid47_fpMulTest_q_1_q;

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_a(BITJOIN,198)@4
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_a_q <= GND_q & sumAb_uid122_prod_uid47_fpMulTest_q;

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a(BITSELECT,202)@4
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_b <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_a_q(38 downto 0);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_c <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_a_q(64 downto 39);

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2(ADD,204)@4 + 1
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_a <= STD_LOGIC_VECTOR("0" & lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_b);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_b <= STD_LOGIC_VECTOR("0" & lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_b);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_o <= STD_LOGIC_VECTOR(UNSIGNED(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_a) + UNSIGNED(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_b));
        END IF;
    END PROCESS;
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_c(0) <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_o(39);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_q <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_o(38 downto 0);

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_tessel1_0(BITSELECT,225)
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_tessel1_0_b <= STD_LOGIC_VECTOR(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_UpperBits_for_b_q(26 downto 1));

    -- redist3_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_c_1(DELAY,248)
    redist3_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_c_1 : dspba_delay
    GENERIC MAP ( width => 26, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_c, xout => redist3_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_c_1_q, clk => clk, aclr => areset );

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2(ADD,205)@5 + 1
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_cin <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_c;
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_a <= STD_LOGIC_VECTOR("0" & redist3_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_c_1_q) & '1';
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_b <= STD_LOGIC_VECTOR("0" & lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_tessel1_0_b) & lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_cin(0);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_o <= STD_LOGIC_VECTOR(UNSIGNED(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_a) + UNSIGNED(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_b));
        END IF;
    END PROCESS;
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_q <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_o(26 downto 1);

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel2_0(BITSELECT,239)@6
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel2_0_b <= STD_LOGIC_VECTOR(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_q(23 downto 1));

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel2_0(BITSELECT,233)@6
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel2_0_b <= STD_LOGIC_VECTOR(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_q(22 downto 0));

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_1(BITSELECT,221)@6
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_1_b <= STD_LOGIC_VECTOR(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_q(24 downto 24));

    -- fracRPostNorm_uid53_fpMulTest_p2(MUX,211)@6 + 1
    fracRPostNorm_uid53_fpMulTest_p2_s <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_1_b;
    fracRPostNorm_uid53_fpMulTest_p2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            fracRPostNorm_uid53_fpMulTest_p2_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (fracRPostNorm_uid53_fpMulTest_p2_s) IS
                WHEN "0" => fracRPostNorm_uid53_fpMulTest_p2_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel2_0_b;
                WHEN "1" => fracRPostNorm_uid53_fpMulTest_p2_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel2_0_b;
                WHEN OTHERS => fracRPostNorm_uid53_fpMulTest_p2_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel0_2(BITSELECT,215)@7
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel0_2_b <= STD_LOGIC_VECTOR(fracRPostNorm_uid53_fpMulTest_p2_q(8 downto 0));

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel1_0(BITSELECT,237)@6
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel1_0_b <= STD_LOGIC_VECTOR(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_2_q(0 downto 0));

    -- redist2_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_q_1(DELAY,247)
    redist2_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_q_1 : dspba_delay
    GENERIC MAP ( width => 39, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_q, xout => redist2_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_q_1_q, clk => clk, aclr => areset );

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel1_0(BITSELECT,231)@6
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(redist2_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_q_1_q(38 downto 38));

    -- fracRPostNorm_uid53_fpMulTest_p1(MUX,210)@6 + 1
    fracRPostNorm_uid53_fpMulTest_p1_s <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_1_b;
    fracRPostNorm_uid53_fpMulTest_p1_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            fracRPostNorm_uid53_fpMulTest_p1_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (fracRPostNorm_uid53_fpMulTest_p1_s) IS
                WHEN "0" => fracRPostNorm_uid53_fpMulTest_p1_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel1_0_b;
                WHEN "1" => fracRPostNorm_uid53_fpMulTest_p1_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel1_0_b;
                WHEN OTHERS => fracRPostNorm_uid53_fpMulTest_p1_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel0_0(BITSELECT,235)@6
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel0_0_b <= STD_LOGIC_VECTOR(redist2_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_q_1_q(38 downto 10));

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel0_0(BITSELECT,229)@6
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel0_0_b <= STD_LOGIC_VECTOR(redist2_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_2_q_1_q(37 downto 9));

    -- fracRPostNorm_uid53_fpMulTest_p0(MUX,209)@6 + 1
    fracRPostNorm_uid53_fpMulTest_p0_s <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_1_b;
    fracRPostNorm_uid53_fpMulTest_p0_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            fracRPostNorm_uid53_fpMulTest_p0_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (fracRPostNorm_uid53_fpMulTest_p0_s) IS
                WHEN "0" => fracRPostNorm_uid53_fpMulTest_p0_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel0_0_b;
                WHEN "1" => fracRPostNorm_uid53_fpMulTest_p0_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel0_0_b;
                WHEN OTHERS => fracRPostNorm_uid53_fpMulTest_p0_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_b(BITJOIN,216)@7
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_b_q <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel0_2_b & fracRPostNorm_uid53_fpMulTest_p1_q & fracRPostNorm_uid53_fpMulTest_p0_q;

    -- expFracRPostRounding_uid58_fpMulTest_p1_of_2(ADD,195)@7 + 1
    expFracRPostRounding_uid58_fpMulTest_p1_of_2_a <= STD_LOGIC_VECTOR("0" & expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_b_q);
    expFracRPostRounding_uid58_fpMulTest_p1_of_2_b <= STD_LOGIC_VECTOR("0" & expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_b_q);
    expFracRPostRounding_uid58_fpMulTest_p1_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expFracRPostRounding_uid58_fpMulTest_p1_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expFracRPostRounding_uid58_fpMulTest_p1_of_2_o <= STD_LOGIC_VECTOR(UNSIGNED(expFracRPostRounding_uid58_fpMulTest_p1_of_2_a) + UNSIGNED(expFracRPostRounding_uid58_fpMulTest_p1_of_2_b));
        END IF;
    END PROCESS;
    expFracRPostRounding_uid58_fpMulTest_p1_of_2_c(0) <= expFracRPostRounding_uid58_fpMulTest_p1_of_2_o(39);
    expFracRPostRounding_uid58_fpMulTest_p1_of_2_q <= expFracRPostRounding_uid58_fpMulTest_p1_of_2_o(38 downto 0);

    -- expFracRPostRounding_uid58_fpMulTest_UpperBits_for_b(CONSTANT,192)
    expFracRPostRounding_uid58_fpMulTest_UpperBits_for_b_q <= "0000000000000";

    -- redist1_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_1_b_2(DELAY,246)
    redist1_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_1_b_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_1_b, xout => redist1_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_1_b_2_q, clk => clk, aclr => areset );

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_0(BITSELECT,220)
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_0_b <= STD_LOGIC_VECTOR(cstZeroWF_uid11_fpMulTest_q(51 downto 38));

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_c(BITJOIN,224)@8
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_c_q <= expFracRPostRounding_uid58_fpMulTest_UpperBits_for_b_q & GND_q & redist1_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_1_b_2_q & expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_0_b;

    -- expFracRPostRounding_uid58_fpMulTest_SignBit_for_a(BITSELECT,189)@7
    expFracRPostRounding_uid58_fpMulTest_SignBit_for_a_b <= STD_LOGIC_VECTOR(expFracPreRound_uid55_fpMulTest_q(66 downto 66));

    -- biasInc_uid45_fpMulTest(CONSTANT,44)
    biasInc_uid45_fpMulTest_q <= "0001111111111";

    -- expSum_uid44_fpMulTest(ADD,43)@0 + 1
    expSum_uid44_fpMulTest_a <= STD_LOGIC_VECTOR("0" & expX_uid6_fpMulTest_b);
    expSum_uid44_fpMulTest_b <= STD_LOGIC_VECTOR("0" & expY_uid7_fpMulTest_b);
    expSum_uid44_fpMulTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expSum_uid44_fpMulTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expSum_uid44_fpMulTest_o <= STD_LOGIC_VECTOR(UNSIGNED(expSum_uid44_fpMulTest_a) + UNSIGNED(expSum_uid44_fpMulTest_b));
        END IF;
    END PROCESS;
    expSum_uid44_fpMulTest_q <= expSum_uid44_fpMulTest_o(11 downto 0);

    -- redist18_expSum_uid44_fpMulTest_q_7(DELAY,263)
    redist18_expSum_uid44_fpMulTest_q_7 : dspba_delay
    GENERIC MAP ( width => 12, depth => 6, reset_kind => "ASYNC" )
    PORT MAP ( xin => expSum_uid44_fpMulTest_q, xout => redist18_expSum_uid44_fpMulTest_q_7_q, clk => clk, aclr => areset );

    -- expSumMBias_uid46_fpMulTest(SUB,45)@7
    expSumMBias_uid46_fpMulTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("000" & redist18_expSum_uid44_fpMulTest_q_7_q));
    expSumMBias_uid46_fpMulTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((14 downto 13 => biasInc_uid45_fpMulTest_q(12)) & biasInc_uid45_fpMulTest_q));
    expSumMBias_uid46_fpMulTest_o <= STD_LOGIC_VECTOR(SIGNED(expSumMBias_uid46_fpMulTest_a) - SIGNED(expSumMBias_uid46_fpMulTest_b));
    expSumMBias_uid46_fpMulTest_q <= expSumMBias_uid46_fpMulTest_o(13 downto 0);

    -- fracRPostNorm_uid53_fpMulTest_BitJoin_for_q(BITJOIN,212)@7
    fracRPostNorm_uid53_fpMulTest_BitJoin_for_q_q <= fracRPostNorm_uid53_fpMulTest_p2_q & fracRPostNorm_uid53_fpMulTest_p1_q & fracRPostNorm_uid53_fpMulTest_p0_q;

    -- expFracPreRound_uid55_fpMulTest(BITJOIN,54)@7
    expFracPreRound_uid55_fpMulTest_q <= expSumMBias_uid46_fpMulTest_q & fracRPostNorm_uid53_fpMulTest_BitJoin_for_q_q;

    -- expFracRPostRounding_uid58_fpMulTest_BitExpansion_for_a(BITJOIN,188)@7
    expFracRPostRounding_uid58_fpMulTest_BitExpansion_for_a_q <= expFracRPostRounding_uid58_fpMulTest_SignBit_for_a_b & expFracPreRound_uid55_fpMulTest_q;

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a(BITSELECT,193)@7
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_c <= STD_LOGIC_VECTOR(expFracRPostRounding_uid58_fpMulTest_BitExpansion_for_a_q(67 downto 39));

    -- redist5_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_c_1(DELAY,250)
    redist5_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_c_1 : dspba_delay
    GENERIC MAP ( width => 29, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_c, xout => redist5_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_c_1_q, clk => clk, aclr => areset );

    -- expFracRPostRounding_uid58_fpMulTest_p2_of_2(ADD,196)@8 + 1
    expFracRPostRounding_uid58_fpMulTest_p2_of_2_cin <= expFracRPostRounding_uid58_fpMulTest_p1_of_2_c;
    expFracRPostRounding_uid58_fpMulTest_p2_of_2_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((29 downto 29 => redist5_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_c_1_q(28)) & redist5_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_c_1_q) & '1');
    expFracRPostRounding_uid58_fpMulTest_p2_of_2_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0" & expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_c_q) & expFracRPostRounding_uid58_fpMulTest_p2_of_2_cin(0));
    expFracRPostRounding_uid58_fpMulTest_p2_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expFracRPostRounding_uid58_fpMulTest_p2_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expFracRPostRounding_uid58_fpMulTest_p2_of_2_o <= STD_LOGIC_VECTOR(SIGNED(expFracRPostRounding_uid58_fpMulTest_p2_of_2_a) + SIGNED(expFracRPostRounding_uid58_fpMulTest_p2_of_2_b));
        END IF;
    END PROCESS;
    expFracRPostRounding_uid58_fpMulTest_p2_of_2_q <= expFracRPostRounding_uid58_fpMulTest_p2_of_2_o(29 downto 1);

    -- redist4_expFracRPostRounding_uid58_fpMulTest_p1_of_2_q_1(DELAY,249)
    redist4_expFracRPostRounding_uid58_fpMulTest_p1_of_2_q_1 : dspba_delay
    GENERIC MAP ( width => 39, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => expFracRPostRounding_uid58_fpMulTest_p1_of_2_q, xout => redist4_expFracRPostRounding_uid58_fpMulTest_p1_of_2_q_1_q, clk => clk, aclr => areset );

    -- expFracRPostRounding_uid58_fpMulTest_BitJoin_for_q(BITJOIN,197)@9
    expFracRPostRounding_uid58_fpMulTest_BitJoin_for_q_q <= expFracRPostRounding_uid58_fpMulTest_p2_of_2_q & redist4_expFracRPostRounding_uid58_fpMulTest_p1_of_2_q_1_q;

    -- expRPreExcExt_uid60_fpMulTest(BITSELECT,59)@9
    expRPreExcExt_uid60_fpMulTest_b <= STD_LOGIC_VECTOR(expFracRPostRounding_uid58_fpMulTest_BitJoin_for_q_q(67 downto 53));

    -- expRPreExc_uid61_fpMulTest(BITSELECT,60)@9
    expRPreExc_uid61_fpMulTest_in <= expRPreExcExt_uid60_fpMulTest_b(10 downto 0);
    expRPreExc_uid61_fpMulTest_b <= expRPreExc_uid61_fpMulTest_in(10 downto 0);

    -- redist15_expRPreExc_uid61_fpMulTest_b_3(DELAY,260)
    redist15_expRPreExc_uid61_fpMulTest_b_3 : dspba_delay
    GENERIC MAP ( width => 11, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => expRPreExc_uid61_fpMulTest_b, xout => redist15_expRPreExc_uid61_fpMulTest_b_3_q, clk => clk, aclr => areset );

    -- expOvf_uid64_fpMulTest(COMPARE,63)@9 + 1
    expOvf_uid64_fpMulTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((16 downto 15 => expRPreExcExt_uid60_fpMulTest_b(14)) & expRPreExcExt_uid60_fpMulTest_b));
    expOvf_uid64_fpMulTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("000000" & cstAllOWE_uid10_fpMulTest_q));
    expOvf_uid64_fpMulTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expOvf_uid64_fpMulTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expOvf_uid64_fpMulTest_o <= STD_LOGIC_VECTOR(SIGNED(expOvf_uid64_fpMulTest_a) - SIGNED(expOvf_uid64_fpMulTest_b));
        END IF;
    END PROCESS;
    expOvf_uid64_fpMulTest_n(0) <= not (expOvf_uid64_fpMulTest_o(16));

    -- invExpXIsMax_uid35_fpMulTest(LOGICAL,34)@10
    invExpXIsMax_uid35_fpMulTest_q <= not (redist19_expXIsMax_uid30_fpMulTest_q_10_q);

    -- InvExpXIsZero_uid36_fpMulTest(LOGICAL,35)@10
    InvExpXIsZero_uid36_fpMulTest_q <= not (redist20_excZ_y_uid29_fpMulTest_q_10_q);

    -- excR_y_uid37_fpMulTest(LOGICAL,36)@10
    excR_y_uid37_fpMulTest_q <= InvExpXIsZero_uid36_fpMulTest_q and invExpXIsMax_uid35_fpMulTest_q;

    -- invExpXIsMax_uid21_fpMulTest(LOGICAL,20)@10
    invExpXIsMax_uid21_fpMulTest_q <= not (redist21_expXIsMax_uid16_fpMulTest_q_10_q);

    -- InvExpXIsZero_uid22_fpMulTest(LOGICAL,21)@10
    InvExpXIsZero_uid22_fpMulTest_q <= not (redist22_excZ_x_uid15_fpMulTest_q_10_q);

    -- excR_x_uid23_fpMulTest(LOGICAL,22)@10
    excR_x_uid23_fpMulTest_q <= InvExpXIsZero_uid22_fpMulTest_q and invExpXIsMax_uid21_fpMulTest_q;

    -- ExcROvfAndInReg_uid73_fpMulTest(LOGICAL,72)@10
    ExcROvfAndInReg_uid73_fpMulTest_q <= excR_x_uid23_fpMulTest_q and excR_y_uid37_fpMulTest_q and expOvf_uid64_fpMulTest_n;

    -- excYRAndExcXI_uid72_fpMulTest(LOGICAL,71)@10
    excYRAndExcXI_uid72_fpMulTest_q <= excR_y_uid37_fpMulTest_q and excI_x_uid19_fpMulTest_q;

    -- excXRAndExcYI_uid71_fpMulTest(LOGICAL,70)@10
    excXRAndExcYI_uid71_fpMulTest_q <= excR_x_uid23_fpMulTest_q and excI_y_uid33_fpMulTest_q;

    -- excXIAndExcYI_uid70_fpMulTest(LOGICAL,69)@10
    excXIAndExcYI_uid70_fpMulTest_q <= excI_x_uid19_fpMulTest_q and excI_y_uid33_fpMulTest_q;

    -- excRInf_uid74_fpMulTest(LOGICAL,73)@10 + 1
    excRInf_uid74_fpMulTest_qi <= excXIAndExcYI_uid70_fpMulTest_q or excXRAndExcYI_uid71_fpMulTest_q or excYRAndExcXI_uid72_fpMulTest_q or ExcROvfAndInReg_uid73_fpMulTest_q;
    excRInf_uid74_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excRInf_uid74_fpMulTest_qi, xout => excRInf_uid74_fpMulTest_q, clk => clk, aclr => areset );

    -- expUdf_uid62_fpMulTest(COMPARE,61)@9 + 1
    expUdf_uid62_fpMulTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0000000000000000" & GND_q));
    expUdf_uid62_fpMulTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((16 downto 15 => expRPreExcExt_uid60_fpMulTest_b(14)) & expRPreExcExt_uid60_fpMulTest_b));
    expUdf_uid62_fpMulTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expUdf_uid62_fpMulTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expUdf_uid62_fpMulTest_o <= STD_LOGIC_VECTOR(SIGNED(expUdf_uid62_fpMulTest_a) - SIGNED(expUdf_uid62_fpMulTest_b));
        END IF;
    END PROCESS;
    expUdf_uid62_fpMulTest_n(0) <= not (expUdf_uid62_fpMulTest_o(16));

    -- excZC3_uid68_fpMulTest(LOGICAL,67)@10
    excZC3_uid68_fpMulTest_q <= excR_x_uid23_fpMulTest_q and excR_y_uid37_fpMulTest_q and expUdf_uid62_fpMulTest_n;

    -- excYZAndExcXR_uid67_fpMulTest(LOGICAL,66)@10
    excYZAndExcXR_uid67_fpMulTest_q <= redist20_excZ_y_uid29_fpMulTest_q_10_q and excR_x_uid23_fpMulTest_q;

    -- excXZAndExcYR_uid66_fpMulTest(LOGICAL,65)@10
    excXZAndExcYR_uid66_fpMulTest_q <= redist22_excZ_x_uid15_fpMulTest_q_10_q and excR_y_uid37_fpMulTest_q;

    -- excXZAndExcYZ_uid65_fpMulTest(LOGICAL,64)@10
    excXZAndExcYZ_uid65_fpMulTest_q <= redist22_excZ_x_uid15_fpMulTest_q_10_q and redist20_excZ_y_uid29_fpMulTest_q_10_q;

    -- excRZero_uid69_fpMulTest(LOGICAL,68)@10 + 1
    excRZero_uid69_fpMulTest_qi <= excXZAndExcYZ_uid65_fpMulTest_q or excXZAndExcYR_uid66_fpMulTest_q or excYZAndExcXR_uid67_fpMulTest_q or excZC3_uid68_fpMulTest_q;
    excRZero_uid69_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excRZero_uid69_fpMulTest_qi, xout => excRZero_uid69_fpMulTest_q, clk => clk, aclr => areset );

    -- concExc_uid79_fpMulTest(BITJOIN,78)@11
    concExc_uid79_fpMulTest_q <= excRNaN_uid78_fpMulTest_q & excRInf_uid74_fpMulTest_q & excRZero_uid69_fpMulTest_q;

    -- excREnc_uid80_fpMulTest(LOOKUP,79)@11 + 1
    excREnc_uid80_fpMulTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            excREnc_uid80_fpMulTest_q <= "01";
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (concExc_uid79_fpMulTest_q) IS
                WHEN "000" => excREnc_uid80_fpMulTest_q <= "01";
                WHEN "001" => excREnc_uid80_fpMulTest_q <= "00";
                WHEN "010" => excREnc_uid80_fpMulTest_q <= "10";
                WHEN "011" => excREnc_uid80_fpMulTest_q <= "00";
                WHEN "100" => excREnc_uid80_fpMulTest_q <= "11";
                WHEN "101" => excREnc_uid80_fpMulTest_q <= "00";
                WHEN "110" => excREnc_uid80_fpMulTest_q <= "00";
                WHEN "111" => excREnc_uid80_fpMulTest_q <= "00";
                WHEN OTHERS => -- unreachable
                               excREnc_uid80_fpMulTest_q <= (others => '-');
            END CASE;
        END IF;
    END PROCESS;

    -- expRPostExc_uid89_fpMulTest(MUX,88)@12
    expRPostExc_uid89_fpMulTest_s <= excREnc_uid80_fpMulTest_q;
    expRPostExc_uid89_fpMulTest_combproc: PROCESS (expRPostExc_uid89_fpMulTest_s, cstAllZWE_uid12_fpMulTest_q, redist15_expRPreExc_uid61_fpMulTest_b_3_q, cstAllOWE_uid10_fpMulTest_q)
    BEGIN
        CASE (expRPostExc_uid89_fpMulTest_s) IS
            WHEN "00" => expRPostExc_uid89_fpMulTest_q <= cstAllZWE_uid12_fpMulTest_q;
            WHEN "01" => expRPostExc_uid89_fpMulTest_q <= redist15_expRPreExc_uid61_fpMulTest_b_3_q;
            WHEN "10" => expRPostExc_uid89_fpMulTest_q <= cstAllOWE_uid10_fpMulTest_q;
            WHEN "11" => expRPostExc_uid89_fpMulTest_q <= cstAllOWE_uid10_fpMulTest_q;
            WHEN OTHERS => expRPostExc_uid89_fpMulTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- oneFracRPostExc2_uid81_fpMulTest(CONSTANT,80)
    oneFracRPostExc2_uid81_fpMulTest_q <= "0000000000000000000000000000000000000000000000000001";

    -- fracRPreExc_uid59_fpMulTest(BITSELECT,58)@9
    fracRPreExc_uid59_fpMulTest_in <= expFracRPostRounding_uid58_fpMulTest_BitJoin_for_q_q(52 downto 0);
    fracRPreExc_uid59_fpMulTest_b <= fracRPreExc_uid59_fpMulTest_in(52 downto 1);

    -- redist16_fracRPreExc_uid59_fpMulTest_b_3(DELAY,261)
    redist16_fracRPreExc_uid59_fpMulTest_b_3 : dspba_delay
    GENERIC MAP ( width => 52, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => fracRPreExc_uid59_fpMulTest_b, xout => redist16_fracRPreExc_uid59_fpMulTest_b_3_q, clk => clk, aclr => areset );

    -- fracRPostExc_uid84_fpMulTest(MUX,83)@12
    fracRPostExc_uid84_fpMulTest_s <= excREnc_uid80_fpMulTest_q;
    fracRPostExc_uid84_fpMulTest_combproc: PROCESS (fracRPostExc_uid84_fpMulTest_s, cstZeroWF_uid11_fpMulTest_q, redist16_fracRPreExc_uid59_fpMulTest_b_3_q, oneFracRPostExc2_uid81_fpMulTest_q)
    BEGIN
        CASE (fracRPostExc_uid84_fpMulTest_s) IS
            WHEN "00" => fracRPostExc_uid84_fpMulTest_q <= cstZeroWF_uid11_fpMulTest_q;
            WHEN "01" => fracRPostExc_uid84_fpMulTest_q <= redist16_fracRPreExc_uid59_fpMulTest_b_3_q;
            WHEN "10" => fracRPostExc_uid84_fpMulTest_q <= cstZeroWF_uid11_fpMulTest_q;
            WHEN "11" => fracRPostExc_uid84_fpMulTest_q <= oneFracRPostExc2_uid81_fpMulTest_q;
            WHEN OTHERS => fracRPostExc_uid84_fpMulTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- R_uid92_fpMulTest(BITJOIN,91)@12
    R_uid92_fpMulTest_q <= signRPostExc_uid91_fpMulTest_q & expRPostExc_uid89_fpMulTest_q & fracRPostExc_uid84_fpMulTest_q;

    -- xOut(GPOUT,4)@12
    q <= R_uid92_fpMulTest_q;

END normal;
