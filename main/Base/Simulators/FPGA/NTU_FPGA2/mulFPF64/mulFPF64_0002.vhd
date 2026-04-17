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

-- VHDL created from mulFPF64_0002
-- VHDL created on Sat Apr 18 00:08:02 2026


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
    signal excI_x_uid19_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_x_uid19_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_x_uid20_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_x_uid20_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invExpXIsMax_uid21_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal InvExpXIsZero_uid22_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excR_x_uid23_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excR_x_uid23_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal frac_y_uid28_fpMulTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal excZ_y_uid29_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excZ_y_uid29_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid30_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid30_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsNotZero_uid32_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_y_uid33_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_y_uid33_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_y_uid34_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_y_uid34_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invExpXIsMax_uid35_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal InvExpXIsZero_uid36_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excR_y_uid37_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
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
    signal eq0_uid131_fracXIsZero_uid17_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq0_uid131_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c1_uid133_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (11 downto 0);
    signal c1_uid133_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq1_uid134_fracXIsZero_uid17_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq1_uid134_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c2_uid136_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (17 downto 0);
    signal c2_uid136_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq2_uid137_fracXIsZero_uid17_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq2_uid137_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c3_uid139_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (23 downto 0);
    signal c3_uid139_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq3_uid140_fracXIsZero_uid17_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq3_uid140_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c4_uid142_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (29 downto 0);
    signal c4_uid142_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq4_uid143_fracXIsZero_uid17_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq4_uid143_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c5_uid145_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (35 downto 0);
    signal c5_uid145_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq5_uid146_fracXIsZero_uid17_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq5_uid146_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c6_uid148_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (41 downto 0);
    signal c6_uid148_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq6_uid149_fracXIsZero_uid17_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq6_uid149_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c7_uid151_fracXIsZero_uid17_fpMulTest_in : STD_LOGIC_VECTOR (47 downto 0);
    signal c7_uid151_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal eq7_uid152_fracXIsZero_uid17_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq7_uid152_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal c8_uid154_fracXIsZero_uid17_fpMulTest_b : STD_LOGIC_VECTOR (3 downto 0);
    signal eq8_uid155_fracXIsZero_uid17_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq8_uid155_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid156_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid157_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev1_uid158_fracXIsZero_uid17_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq0_uid161_fracXIsZero_uid31_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq0_uid161_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq1_uid164_fracXIsZero_uid31_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq1_uid164_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq2_uid167_fracXIsZero_uid31_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq2_uid167_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq3_uid170_fracXIsZero_uid31_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq3_uid170_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq4_uid173_fracXIsZero_uid31_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq4_uid173_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq5_uid176_fracXIsZero_uid31_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq5_uid176_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq6_uid179_fracXIsZero_uid31_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq6_uid179_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq7_uid182_fracXIsZero_uid31_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq7_uid182_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq8_uid185_fracXIsZero_uid31_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal eq8_uid185_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid186_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid187_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev1_uid188_fracXIsZero_uid31_fpMulTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_join_12_q : STD_LOGIC_VECTOR (53 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_align_13_q : STD_LOGIC_VECTOR (42 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_align_13_qint : STD_LOGIC_VECTOR (42 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_align_15_q : STD_LOGIC_VECTOR (42 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_align_15_qint : STD_LOGIC_VECTOR (42 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitExpansion_for_a_q : STD_LOGIC_VECTOR (67 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_SignBit_for_a_b : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_UpperBits_for_b_q : STD_LOGIC_VECTOR (12 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_e : STD_LOGIC_VECTOR (16 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p1_of_4_a : STD_LOGIC_VECTOR (17 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p1_of_4_b : STD_LOGIC_VECTOR (17 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p1_of_4_o : STD_LOGIC_VECTOR (17 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p1_of_4_c : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p1_of_4_q : STD_LOGIC_VECTOR (16 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p2_of_4_a : STD_LOGIC_VECTOR (18 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p2_of_4_b : STD_LOGIC_VECTOR (18 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p2_of_4_o : STD_LOGIC_VECTOR (18 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p2_of_4_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p2_of_4_c : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p2_of_4_q : STD_LOGIC_VECTOR (16 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p3_of_4_a : STD_LOGIC_VECTOR (18 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p3_of_4_b : STD_LOGIC_VECTOR (18 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p3_of_4_o : STD_LOGIC_VECTOR (18 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p3_of_4_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p3_of_4_c : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p3_of_4_q : STD_LOGIC_VECTOR (16 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p4_of_4_a : STD_LOGIC_VECTOR (18 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p4_of_4_b : STD_LOGIC_VECTOR (18 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p4_of_4_o : STD_LOGIC_VECTOR (18 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p4_of_4_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_p4_of_4_q : STD_LOGIC_VECTOR (16 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (67 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_a_q : STD_LOGIC_VECTOR (64 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_b_q : STD_LOGIC_VECTOR (64 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_UpperBits_for_b_q : STD_LOGIC_VECTOR (26 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_b : STD_LOGIC_VECTOR (16 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_b : STD_LOGIC_VECTOR (16 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_c : STD_LOGIC_VECTOR (16 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_d : STD_LOGIC_VECTOR (16 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_a : STD_LOGIC_VECTOR (17 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_b : STD_LOGIC_VECTOR (17 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_o : STD_LOGIC_VECTOR (17 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_c : STD_LOGIC_VECTOR (0 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_q : STD_LOGIC_VECTOR (16 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_a : STD_LOGIC_VECTOR (18 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_b : STD_LOGIC_VECTOR (18 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_o : STD_LOGIC_VECTOR (18 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_c : STD_LOGIC_VECTOR (0 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_q : STD_LOGIC_VECTOR (16 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_a : STD_LOGIC_VECTOR (18 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_b : STD_LOGIC_VECTOR (18 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_o : STD_LOGIC_VECTOR (18 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_c : STD_LOGIC_VECTOR (0 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_q : STD_LOGIC_VECTOR (16 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_a : STD_LOGIC_VECTOR (15 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_b : STD_LOGIC_VECTOR (15 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_o : STD_LOGIC_VECTOR (15 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_q : STD_LOGIC_VECTOR (13 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitExpansion_for_a_q : STD_LOGIC_VECTOR (54 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitExpansion_for_b_q : STD_LOGIC_VECTOR (54 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_UpperBits_for_b_q : STD_LOGIC_VECTOR (11 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_b : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_c : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_d : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_e : STD_LOGIC_VECTOR (3 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_b : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_c : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_d : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_a : STD_LOGIC_VECTOR (17 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_b : STD_LOGIC_VECTOR (17 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_o : STD_LOGIC_VECTOR (17 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_c : STD_LOGIC_VECTOR (0 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_q : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_a : STD_LOGIC_VECTOR (18 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_b : STD_LOGIC_VECTOR (18 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_o : STD_LOGIC_VECTOR (18 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_c : STD_LOGIC_VECTOR (0 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_q : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_a : STD_LOGIC_VECTOR (18 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_b : STD_LOGIC_VECTOR (18 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_o : STD_LOGIC_VECTOR (18 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_c : STD_LOGIC_VECTOR (0 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_q : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_a : STD_LOGIC_VECTOR (5 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_b : STD_LOGIC_VECTOR (5 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_o : STD_LOGIC_VECTOR (5 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_q : STD_LOGIC_VECTOR (3 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitExpansion_for_b_q : STD_LOGIC_VECTOR (55 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_b : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_c : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_d : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_a : STD_LOGIC_VECTOR (17 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_b : STD_LOGIC_VECTOR (17 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_o : STD_LOGIC_VECTOR (17 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_c : STD_LOGIC_VECTOR (0 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_q : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_a : STD_LOGIC_VECTOR (18 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_b : STD_LOGIC_VECTOR (18 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_o : STD_LOGIC_VECTOR (18 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_c : STD_LOGIC_VECTOR (0 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_q : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_a : STD_LOGIC_VECTOR (18 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_b : STD_LOGIC_VECTOR (18 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_o : STD_LOGIC_VECTOR (18 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_c : STD_LOGIC_VECTOR (0 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q : STD_LOGIC_VECTOR (16 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_a : STD_LOGIC_VECTOR (6 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_b : STD_LOGIC_VECTOR (6 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_o : STD_LOGIC_VECTOR (6 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_q : STD_LOGIC_VECTOR (4 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitJoin_for_q_q : STD_LOGIC_VECTOR (55 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p0_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p0_q : STD_LOGIC_VECTOR (6 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p1_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p2_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p2_q : STD_LOGIC_VECTOR (15 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p3_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p4_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p4_q : STD_LOGIC_VECTOR (15 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p5_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p6_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_p6_q : STD_LOGIC_VECTOR (10 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (52 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_b_q : STD_LOGIC_VECTOR (16 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_c_q : STD_LOGIC_VECTOR (16 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel2_2_b : STD_LOGIC_VECTOR (8 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_d_q : STD_LOGIC_VECTOR (16 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel0_1_b : STD_LOGIC_VECTOR (15 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_b_q : STD_LOGIC_VECTOR (16 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_0_b : STD_LOGIC_VECTOR (16 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel2_0_b : STD_LOGIC_VECTOR (16 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_e_q : STD_LOGIC_VECTOR (16 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_0_b : STD_LOGIC_VECTOR (9 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_BitJoin_for_c_q : STD_LOGIC_VECTOR (16 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_BitJoin_for_d_q : STD_LOGIC_VECTOR (16 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel3_1_b : STD_LOGIC_VECTOR (2 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_BitJoin_for_e_q : STD_LOGIC_VECTOR (13 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_tessel3_0_b : STD_LOGIC_VECTOR (13 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_tessel3_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_a_BitJoin_for_e_q : STD_LOGIC_VECTOR (4 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_tessel3_0_b : STD_LOGIC_VECTOR (4 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel0_0_b : STD_LOGIC_VECTOR (6 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel1_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel2_0_b : STD_LOGIC_VECTOR (15 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel3_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel4_0_b : STD_LOGIC_VECTOR (15 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel5_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel6_0_b : STD_LOGIC_VECTOR (10 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel0_0_b : STD_LOGIC_VECTOR (6 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel1_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel2_0_b : STD_LOGIC_VECTOR (15 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel3_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel4_0_b : STD_LOGIC_VECTOR (15 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel5_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel6_0_b : STD_LOGIC_VECTOR (10 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im0_cma_reset : std_logic;
    type sm0_uid118_prod_uid47_fpMulTest_im0_cma_a0type is array(NATURAL range <>) of UNSIGNED(10 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im0_cma_a0 : sm0_uid118_prod_uid47_fpMulTest_im0_cma_a0type(0 to 0);
    attribute preserve : boolean;
    attribute preserve of sm0_uid118_prod_uid47_fpMulTest_im0_cma_a0 : signal is true;
    signal sm0_uid118_prod_uid47_fpMulTest_im0_cma_c0 : sm0_uid118_prod_uid47_fpMulTest_im0_cma_a0type(0 to 0);
    attribute preserve of sm0_uid118_prod_uid47_fpMulTest_im0_cma_c0 : signal is true;
    type sm0_uid118_prod_uid47_fpMulTest_im0_cma_ptype is array(NATURAL range <>) of UNSIGNED(21 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im0_cma_p : sm0_uid118_prod_uid47_fpMulTest_im0_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im0_cma_u : sm0_uid118_prod_uid47_fpMulTest_im0_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im0_cma_w : sm0_uid118_prod_uid47_fpMulTest_im0_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im0_cma_x : sm0_uid118_prod_uid47_fpMulTest_im0_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im0_cma_y : sm0_uid118_prod_uid47_fpMulTest_im0_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im0_cma_s : sm0_uid118_prod_uid47_fpMulTest_im0_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im0_cma_qq : STD_LOGIC_VECTOR (21 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im0_cma_q : STD_LOGIC_VECTOR (21 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im0_cma_ena0 : std_logic;
    signal sm0_uid118_prod_uid47_fpMulTest_im0_cma_ena1 : std_logic;
    signal sm0_uid118_prod_uid47_fpMulTest_im3_cma_reset : std_logic;
    type sm0_uid118_prod_uid47_fpMulTest_im3_cma_a0type is array(NATURAL range <>) of UNSIGNED(15 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im3_cma_a0 : sm0_uid118_prod_uid47_fpMulTest_im3_cma_a0type(0 to 0);
    attribute preserve of sm0_uid118_prod_uid47_fpMulTest_im3_cma_a0 : signal is true;
    signal sm0_uid118_prod_uid47_fpMulTest_im3_cma_c0 : sm0_uid118_prod_uid47_fpMulTest_im0_cma_a0type(0 to 0);
    attribute preserve of sm0_uid118_prod_uid47_fpMulTest_im3_cma_c0 : signal is true;
    type sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype is array(NATURAL range <>) of UNSIGNED(26 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im3_cma_p : sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im3_cma_u : sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im3_cma_w : sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im3_cma_x : sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im3_cma_y : sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im3_cma_s : sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im3_cma_qq : STD_LOGIC_VECTOR (26 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im3_cma_q : STD_LOGIC_VECTOR (26 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im3_cma_ena0 : std_logic;
    signal sm0_uid118_prod_uid47_fpMulTest_im3_cma_ena1 : std_logic;
    signal sm0_uid118_prod_uid47_fpMulTest_im6_cma_reset : std_logic;
    signal sm0_uid118_prod_uid47_fpMulTest_im6_cma_a0 : sm0_uid118_prod_uid47_fpMulTest_im3_cma_a0type(0 to 0);
    attribute preserve of sm0_uid118_prod_uid47_fpMulTest_im6_cma_a0 : signal is true;
    signal sm0_uid118_prod_uid47_fpMulTest_im6_cma_c0 : sm0_uid118_prod_uid47_fpMulTest_im0_cma_a0type(0 to 0);
    attribute preserve of sm0_uid118_prod_uid47_fpMulTest_im6_cma_c0 : signal is true;
    signal sm0_uid118_prod_uid47_fpMulTest_im6_cma_p : sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im6_cma_u : sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im6_cma_w : sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im6_cma_x : sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im6_cma_y : sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im6_cma_s : sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im6_cma_qq : STD_LOGIC_VECTOR (26 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im6_cma_q : STD_LOGIC_VECTOR (26 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im6_cma_ena0 : std_logic;
    signal sm0_uid118_prod_uid47_fpMulTest_im6_cma_ena1 : std_logic;
    signal sm0_uid118_prod_uid47_fpMulTest_im9_cma_reset : std_logic;
    signal sm0_uid118_prod_uid47_fpMulTest_im9_cma_a0 : sm0_uid118_prod_uid47_fpMulTest_im3_cma_a0type(0 to 0);
    attribute preserve of sm0_uid118_prod_uid47_fpMulTest_im9_cma_a0 : signal is true;
    signal sm0_uid118_prod_uid47_fpMulTest_im9_cma_c0 : sm0_uid118_prod_uid47_fpMulTest_im3_cma_a0type(0 to 0);
    attribute preserve of sm0_uid118_prod_uid47_fpMulTest_im9_cma_c0 : signal is true;
    type sm0_uid118_prod_uid47_fpMulTest_im9_cma_ptype is array(NATURAL range <>) of UNSIGNED(31 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im9_cma_p : sm0_uid118_prod_uid47_fpMulTest_im9_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im9_cma_u : sm0_uid118_prod_uid47_fpMulTest_im9_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im9_cma_w : sm0_uid118_prod_uid47_fpMulTest_im9_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im9_cma_x : sm0_uid118_prod_uid47_fpMulTest_im9_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im9_cma_y : sm0_uid118_prod_uid47_fpMulTest_im9_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im9_cma_s : sm0_uid118_prod_uid47_fpMulTest_im9_cma_ptype(0 to 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im9_cma_qq : STD_LOGIC_VECTOR (31 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im9_cma_q : STD_LOGIC_VECTOR (31 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_im9_cma_ena0 : std_logic;
    signal sm0_uid118_prod_uid47_fpMulTest_im9_cma_ena1 : std_logic;
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_reset : std_logic;
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_a0 : sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype(0 to 1);
    attribute preserve of multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_a0 : signal is true;
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_c0 : sm0_uid118_prod_uid47_fpMulTest_im3_cma_ptype(0 to 1);
    attribute preserve of multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_c0 : signal is true;
    type multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_ptype is array(NATURAL range <>) of UNSIGNED(53 downto 0);
    signal multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_p : multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_ptype(0 to 1);
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
    signal sm0_uid118_prod_uid47_fpMulTest_bs1_merged_bit_select_b : STD_LOGIC_VECTOR (10 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_bs1_merged_bit_select_c : STD_LOGIC_VECTOR (15 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_bs2_merged_bit_select_b : STD_LOGIC_VECTOR (10 downto 0);
    signal sm0_uid118_prod_uid47_fpMulTest_bs2_merged_bit_select_c : STD_LOGIC_VECTOR (15 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_b : STD_LOGIC_VECTOR (6 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_c : STD_LOGIC_VECTOR (9 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel2_1_merged_bit_select_b : STD_LOGIC_VECTOR (6 downto 0);
    signal lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel2_1_merged_bit_select_c : STD_LOGIC_VECTOR (9 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel0_2_merged_bit_select_b : STD_LOGIC_VECTOR (8 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel0_2_merged_bit_select_c : STD_LOGIC_VECTOR (6 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel1_2_merged_bit_select_b : STD_LOGIC_VECTOR (8 downto 0);
    signal expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel1_2_merged_bit_select_c : STD_LOGIC_VECTOR (6 downto 0);
    signal redist0_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel2_1_merged_bit_select_c_1_q : STD_LOGIC_VECTOR (9 downto 0);
    signal redist1_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_b_1_q : STD_LOGIC_VECTOR (6 downto 0);
    signal redist2_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_c_2_q : STD_LOGIC_VECTOR (9 downto 0);
    signal redist3_sm0_uid118_prod_uid47_fpMulTest_bs2_merged_bit_select_c_1_q : STD_LOGIC_VECTOR (15 downto 0);
    signal redist4_sm0_uid118_prod_uid47_fpMulTest_bs1_merged_bit_select_b_1_q : STD_LOGIC_VECTOR (10 downto 0);
    signal redist5_sm0_uid118_prod_uid47_fpMulTest_im9_cma_q_1_q : STD_LOGIC_VECTOR (31 downto 0);
    signal redist6_sm0_uid118_prod_uid47_fpMulTest_im6_cma_q_1_q : STD_LOGIC_VECTOR (26 downto 0);
    signal redist7_sm0_uid118_prod_uid47_fpMulTest_im3_cma_q_1_q : STD_LOGIC_VECTOR (26 downto 0);
    signal redist8_sm0_uid118_prod_uid47_fpMulTest_im0_cma_q_1_q : STD_LOGIC_VECTOR (21 downto 0);
    signal redist9_fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel0_0_b_3_q : STD_LOGIC_VECTOR (6 downto 0);
    signal redist10_fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel1_0_b_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist11_fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel0_0_b_3_q : STD_LOGIC_VECTOR (6 downto 0);
    signal redist12_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel3_1_b_3_q : STD_LOGIC_VECTOR (2 downto 0);
    signal redist13_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_0_b_1_q : STD_LOGIC_VECTOR (9 downto 0);
    signal redist14_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist15_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_d_q_2_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist16_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_c_q_1_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist17_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q_1_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist18_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q_3_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist19_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_q_2_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist20_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_q_3_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist21_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_c_1_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist22_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_d_2_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist23_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_c_1_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist24_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_d_2_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist25_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_c_1_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist26_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_d_2_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist27_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_e_3_q : STD_LOGIC_VECTOR (3 downto 0);
    signal redist28_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_q_1_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist29_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_q_2_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist30_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_c_1_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist31_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_d_2_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist32_expFracRPostRounding_uid58_fpMulTest_p3_of_4_q_1_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist33_expFracRPostRounding_uid58_fpMulTest_p2_of_4_q_2_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist34_expFracRPostRounding_uid58_fpMulTest_p1_of_4_q_3_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist35_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_e_3_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist36_and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q_16_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist37_and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q_16_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist38_highBBits_uid124_prod_uid47_fpMulTest_b_5_q : STD_LOGIC_VECTOR (37 downto 0);
    signal redist39_sm0_uid121_prod_uid47_fpMulTest_q_1_q : STD_LOGIC_VECTOR (9 downto 0);
    signal redist40_aboveLeftY_uid117_prod_uid47_fpMulTest_b_5_q : STD_LOGIC_VECTOR (4 downto 0);
    signal redist41_aboveLeftX_uid116_prod_uid47_fpMulTest_b_5_q : STD_LOGIC_VECTOR (4 downto 0);
    signal redist42_rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_b_1_q : STD_LOGIC_VECTOR (25 downto 0);
    signal redist43_topRangeY_uid105_prod_uid47_fpMulTest_b_1_q : STD_LOGIC_VECTOR (26 downto 0);
    signal redist44_expRPreExc_uid61_fpMulTest_b_3_q : STD_LOGIC_VECTOR (10 downto 0);
    signal redist45_fracRPreExc_uid59_fpMulTest_b_3_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist46_signR_uid48_fpMulTest_q_19_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist47_expSum_uid44_fpMulTest_q_13_q : STD_LOGIC_VECTOR (11 downto 0);
    signal redist48_excN_y_uid34_fpMulTest_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist49_expXIsMax_uid30_fpMulTest_q_17_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist50_excZ_y_uid29_fpMulTest_q_17_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist51_excZ_y_uid29_fpMulTest_q_18_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist52_excN_x_uid20_fpMulTest_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist53_expXIsMax_uid16_fpMulTest_q_17_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist54_excZ_x_uid15_fpMulTest_q_17_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist55_excZ_x_uid15_fpMulTest_q_18_q : STD_LOGIC_VECTOR (0 downto 0);

begin


    -- cstZeroWF_uid11_fpMulTest(CONSTANT,10)
    cstZeroWF_uid11_fpMulTest_q <= "0000000000000000000000000000000000000000000000000000";

    -- c8_uid154_fracXIsZero_uid17_fpMulTest(BITSELECT,153)
    c8_uid154_fracXIsZero_uid17_fpMulTest_b <= cstZeroWF_uid11_fpMulTest_q(51 downto 48);

    -- frac_x_uid14_fpMulTest(BITSELECT,13)@0
    frac_x_uid14_fpMulTest_b <= a(51 downto 0);

    -- z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select(BITSELECT,346)@0
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_b <= frac_x_uid14_fpMulTest_b(5 downto 0);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_c <= frac_x_uid14_fpMulTest_b(11 downto 6);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_d <= frac_x_uid14_fpMulTest_b(17 downto 12);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_e <= frac_x_uid14_fpMulTest_b(23 downto 18);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_f <= frac_x_uid14_fpMulTest_b(29 downto 24);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_g <= frac_x_uid14_fpMulTest_b(35 downto 30);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_h <= frac_x_uid14_fpMulTest_b(41 downto 36);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_i <= frac_x_uid14_fpMulTest_b(47 downto 42);
    z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_j <= frac_x_uid14_fpMulTest_b(51 downto 48);

    -- eq8_uid155_fracXIsZero_uid17_fpMulTest(LOGICAL,154)@0 + 1
    eq8_uid155_fracXIsZero_uid17_fpMulTest_qi <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_j = c8_uid154_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq8_uid155_fracXIsZero_uid17_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq8_uid155_fracXIsZero_uid17_fpMulTest_qi, xout => eq8_uid155_fracXIsZero_uid17_fpMulTest_q, clk => clk, aclr => areset );

    -- c7_uid151_fracXIsZero_uid17_fpMulTest(BITSELECT,150)
    c7_uid151_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(47 downto 0);
    c7_uid151_fracXIsZero_uid17_fpMulTest_b <= c7_uid151_fracXIsZero_uid17_fpMulTest_in(47 downto 42);

    -- eq7_uid152_fracXIsZero_uid17_fpMulTest(LOGICAL,151)@0 + 1
    eq7_uid152_fracXIsZero_uid17_fpMulTest_qi <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_i = c7_uid151_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq7_uid152_fracXIsZero_uid17_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq7_uid152_fracXIsZero_uid17_fpMulTest_qi, xout => eq7_uid152_fracXIsZero_uid17_fpMulTest_q, clk => clk, aclr => areset );

    -- c6_uid148_fracXIsZero_uid17_fpMulTest(BITSELECT,147)
    c6_uid148_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(41 downto 0);
    c6_uid148_fracXIsZero_uid17_fpMulTest_b <= c6_uid148_fracXIsZero_uid17_fpMulTest_in(41 downto 36);

    -- eq6_uid149_fracXIsZero_uid17_fpMulTest(LOGICAL,148)@0 + 1
    eq6_uid149_fracXIsZero_uid17_fpMulTest_qi <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_h = c6_uid148_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq6_uid149_fracXIsZero_uid17_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq6_uid149_fracXIsZero_uid17_fpMulTest_qi, xout => eq6_uid149_fracXIsZero_uid17_fpMulTest_q, clk => clk, aclr => areset );

    -- and_lev0_uid157_fracXIsZero_uid17_fpMulTest(LOGICAL,156)@1
    and_lev0_uid157_fracXIsZero_uid17_fpMulTest_q <= eq6_uid149_fracXIsZero_uid17_fpMulTest_q and eq7_uid152_fracXIsZero_uid17_fpMulTest_q and eq8_uid155_fracXIsZero_uid17_fpMulTest_q;

    -- c5_uid145_fracXIsZero_uid17_fpMulTest(BITSELECT,144)
    c5_uid145_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(35 downto 0);
    c5_uid145_fracXIsZero_uid17_fpMulTest_b <= c5_uid145_fracXIsZero_uid17_fpMulTest_in(35 downto 30);

    -- eq5_uid146_fracXIsZero_uid17_fpMulTest(LOGICAL,145)@0 + 1
    eq5_uid146_fracXIsZero_uid17_fpMulTest_qi <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_g = c5_uid145_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq5_uid146_fracXIsZero_uid17_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq5_uid146_fracXIsZero_uid17_fpMulTest_qi, xout => eq5_uid146_fracXIsZero_uid17_fpMulTest_q, clk => clk, aclr => areset );

    -- c4_uid142_fracXIsZero_uid17_fpMulTest(BITSELECT,141)
    c4_uid142_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(29 downto 0);
    c4_uid142_fracXIsZero_uid17_fpMulTest_b <= c4_uid142_fracXIsZero_uid17_fpMulTest_in(29 downto 24);

    -- eq4_uid143_fracXIsZero_uid17_fpMulTest(LOGICAL,142)@0 + 1
    eq4_uid143_fracXIsZero_uid17_fpMulTest_qi <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_f = c4_uid142_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq4_uid143_fracXIsZero_uid17_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq4_uid143_fracXIsZero_uid17_fpMulTest_qi, xout => eq4_uid143_fracXIsZero_uid17_fpMulTest_q, clk => clk, aclr => areset );

    -- c3_uid139_fracXIsZero_uid17_fpMulTest(BITSELECT,138)
    c3_uid139_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(23 downto 0);
    c3_uid139_fracXIsZero_uid17_fpMulTest_b <= c3_uid139_fracXIsZero_uid17_fpMulTest_in(23 downto 18);

    -- eq3_uid140_fracXIsZero_uid17_fpMulTest(LOGICAL,139)@0 + 1
    eq3_uid140_fracXIsZero_uid17_fpMulTest_qi <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_e = c3_uid139_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq3_uid140_fracXIsZero_uid17_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq3_uid140_fracXIsZero_uid17_fpMulTest_qi, xout => eq3_uid140_fracXIsZero_uid17_fpMulTest_q, clk => clk, aclr => areset );

    -- c2_uid136_fracXIsZero_uid17_fpMulTest(BITSELECT,135)
    c2_uid136_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(17 downto 0);
    c2_uid136_fracXIsZero_uid17_fpMulTest_b <= c2_uid136_fracXIsZero_uid17_fpMulTest_in(17 downto 12);

    -- eq2_uid137_fracXIsZero_uid17_fpMulTest(LOGICAL,136)@0 + 1
    eq2_uid137_fracXIsZero_uid17_fpMulTest_qi <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_d = c2_uid136_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq2_uid137_fracXIsZero_uid17_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq2_uid137_fracXIsZero_uid17_fpMulTest_qi, xout => eq2_uid137_fracXIsZero_uid17_fpMulTest_q, clk => clk, aclr => areset );

    -- c1_uid133_fracXIsZero_uid17_fpMulTest(BITSELECT,132)
    c1_uid133_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(11 downto 0);
    c1_uid133_fracXIsZero_uid17_fpMulTest_b <= c1_uid133_fracXIsZero_uid17_fpMulTest_in(11 downto 6);

    -- eq1_uid134_fracXIsZero_uid17_fpMulTest(LOGICAL,133)@0 + 1
    eq1_uid134_fracXIsZero_uid17_fpMulTest_qi <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_c = c1_uid133_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq1_uid134_fracXIsZero_uid17_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq1_uid134_fracXIsZero_uid17_fpMulTest_qi, xout => eq1_uid134_fracXIsZero_uid17_fpMulTest_q, clk => clk, aclr => areset );

    -- c0_uid130_fracXIsZero_uid17_fpMulTest(BITSELECT,129)
    c0_uid130_fracXIsZero_uid17_fpMulTest_in <= cstZeroWF_uid11_fpMulTest_q(5 downto 0);
    c0_uid130_fracXIsZero_uid17_fpMulTest_b <= c0_uid130_fracXIsZero_uid17_fpMulTest_in(5 downto 0);

    -- eq0_uid131_fracXIsZero_uid17_fpMulTest(LOGICAL,130)@0 + 1
    eq0_uid131_fracXIsZero_uid17_fpMulTest_qi <= "1" WHEN z0_uid129_fracXIsZero_uid17_fpMulTest_merged_bit_select_b = c0_uid130_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq0_uid131_fracXIsZero_uid17_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq0_uid131_fracXIsZero_uid17_fpMulTest_qi, xout => eq0_uid131_fracXIsZero_uid17_fpMulTest_q, clk => clk, aclr => areset );

    -- and_lev0_uid156_fracXIsZero_uid17_fpMulTest(LOGICAL,155)@1
    and_lev0_uid156_fracXIsZero_uid17_fpMulTest_q <= eq0_uid131_fracXIsZero_uid17_fpMulTest_q and eq1_uid134_fracXIsZero_uid17_fpMulTest_q and eq2_uid137_fracXIsZero_uid17_fpMulTest_q and eq3_uid140_fracXIsZero_uid17_fpMulTest_q and eq4_uid143_fracXIsZero_uid17_fpMulTest_q and eq5_uid146_fracXIsZero_uid17_fpMulTest_q;

    -- and_lev1_uid158_fracXIsZero_uid17_fpMulTest(LOGICAL,157)@1 + 1
    and_lev1_uid158_fracXIsZero_uid17_fpMulTest_qi <= and_lev0_uid156_fracXIsZero_uid17_fpMulTest_q and and_lev0_uid157_fracXIsZero_uid17_fpMulTest_q;
    and_lev1_uid158_fracXIsZero_uid17_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev1_uid158_fracXIsZero_uid17_fpMulTest_qi, xout => and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q, clk => clk, aclr => areset );

    -- redist37_and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q_16(DELAY,391)
    redist37_and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q_16 : dspba_delay
    GENERIC MAP ( width => 1, depth => 15, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q, xout => redist37_and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q_16_q, clk => clk, aclr => areset );

    -- cstAllOWE_uid10_fpMulTest(CONSTANT,9)
    cstAllOWE_uid10_fpMulTest_q <= "11111111111";

    -- expX_uid6_fpMulTest(BITSELECT,5)@0
    expX_uid6_fpMulTest_b <= a(62 downto 52);

    -- expXIsMax_uid16_fpMulTest(LOGICAL,15)@0 + 1
    expXIsMax_uid16_fpMulTest_qi <= "1" WHEN expX_uid6_fpMulTest_b = cstAllOWE_uid10_fpMulTest_q ELSE "0";
    expXIsMax_uid16_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => expXIsMax_uid16_fpMulTest_qi, xout => expXIsMax_uid16_fpMulTest_q, clk => clk, aclr => areset );

    -- redist53_expXIsMax_uid16_fpMulTest_q_17(DELAY,407)
    redist53_expXIsMax_uid16_fpMulTest_q_17 : dspba_delay
    GENERIC MAP ( width => 1, depth => 16, reset_kind => "ASYNC" )
    PORT MAP ( xin => expXIsMax_uid16_fpMulTest_q, xout => redist53_expXIsMax_uid16_fpMulTest_q_17_q, clk => clk, aclr => areset );

    -- excI_x_uid19_fpMulTest(LOGICAL,18)@17 + 1
    excI_x_uid19_fpMulTest_qi <= redist53_expXIsMax_uid16_fpMulTest_q_17_q and redist37_and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q_16_q;
    excI_x_uid19_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excI_x_uid19_fpMulTest_qi, xout => excI_x_uid19_fpMulTest_q, clk => clk, aclr => areset );

    -- cstAllZWE_uid12_fpMulTest(CONSTANT,11)
    cstAllZWE_uid12_fpMulTest_q <= "00000000000";

    -- expY_uid7_fpMulTest(BITSELECT,6)@0
    expY_uid7_fpMulTest_b <= b(62 downto 52);

    -- excZ_y_uid29_fpMulTest(LOGICAL,28)@0 + 1
    excZ_y_uid29_fpMulTest_qi <= "1" WHEN expY_uid7_fpMulTest_b = cstAllZWE_uid12_fpMulTest_q ELSE "0";
    excZ_y_uid29_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excZ_y_uid29_fpMulTest_qi, xout => excZ_y_uid29_fpMulTest_q, clk => clk, aclr => areset );

    -- redist50_excZ_y_uid29_fpMulTest_q_17(DELAY,404)
    redist50_excZ_y_uid29_fpMulTest_q_17 : dspba_delay
    GENERIC MAP ( width => 1, depth => 16, reset_kind => "ASYNC" )
    PORT MAP ( xin => excZ_y_uid29_fpMulTest_q, xout => redist50_excZ_y_uid29_fpMulTest_q_17_q, clk => clk, aclr => areset );

    -- redist51_excZ_y_uid29_fpMulTest_q_18(DELAY,405)
    redist51_excZ_y_uid29_fpMulTest_q_18 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist50_excZ_y_uid29_fpMulTest_q_17_q, xout => redist51_excZ_y_uid29_fpMulTest_q_18_q, clk => clk, aclr => areset );

    -- excYZAndExcXI_uid75_fpMulTest(LOGICAL,74)@18
    excYZAndExcXI_uid75_fpMulTest_q <= redist51_excZ_y_uid29_fpMulTest_q_18_q and excI_x_uid19_fpMulTest_q;

    -- frac_y_uid28_fpMulTest(BITSELECT,27)@0
    frac_y_uid28_fpMulTest_b <= b(51 downto 0);

    -- z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select(BITSELECT,347)@0
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_b <= frac_y_uid28_fpMulTest_b(5 downto 0);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_c <= frac_y_uid28_fpMulTest_b(11 downto 6);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_d <= frac_y_uid28_fpMulTest_b(17 downto 12);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_e <= frac_y_uid28_fpMulTest_b(23 downto 18);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_f <= frac_y_uid28_fpMulTest_b(29 downto 24);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_g <= frac_y_uid28_fpMulTest_b(35 downto 30);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_h <= frac_y_uid28_fpMulTest_b(41 downto 36);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_i <= frac_y_uid28_fpMulTest_b(47 downto 42);
    z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_j <= frac_y_uid28_fpMulTest_b(51 downto 48);

    -- eq8_uid185_fracXIsZero_uid31_fpMulTest(LOGICAL,184)@0 + 1
    eq8_uid185_fracXIsZero_uid31_fpMulTest_qi <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_j = c8_uid154_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq8_uid185_fracXIsZero_uid31_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq8_uid185_fracXIsZero_uid31_fpMulTest_qi, xout => eq8_uid185_fracXIsZero_uid31_fpMulTest_q, clk => clk, aclr => areset );

    -- eq7_uid182_fracXIsZero_uid31_fpMulTest(LOGICAL,181)@0 + 1
    eq7_uid182_fracXIsZero_uid31_fpMulTest_qi <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_i = c7_uid151_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq7_uid182_fracXIsZero_uid31_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq7_uid182_fracXIsZero_uid31_fpMulTest_qi, xout => eq7_uid182_fracXIsZero_uid31_fpMulTest_q, clk => clk, aclr => areset );

    -- eq6_uid179_fracXIsZero_uid31_fpMulTest(LOGICAL,178)@0 + 1
    eq6_uid179_fracXIsZero_uid31_fpMulTest_qi <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_h = c6_uid148_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq6_uid179_fracXIsZero_uid31_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq6_uid179_fracXIsZero_uid31_fpMulTest_qi, xout => eq6_uid179_fracXIsZero_uid31_fpMulTest_q, clk => clk, aclr => areset );

    -- and_lev0_uid187_fracXIsZero_uid31_fpMulTest(LOGICAL,186)@1
    and_lev0_uid187_fracXIsZero_uid31_fpMulTest_q <= eq6_uid179_fracXIsZero_uid31_fpMulTest_q and eq7_uid182_fracXIsZero_uid31_fpMulTest_q and eq8_uid185_fracXIsZero_uid31_fpMulTest_q;

    -- eq5_uid176_fracXIsZero_uid31_fpMulTest(LOGICAL,175)@0 + 1
    eq5_uid176_fracXIsZero_uid31_fpMulTest_qi <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_g = c5_uid145_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq5_uid176_fracXIsZero_uid31_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq5_uid176_fracXIsZero_uid31_fpMulTest_qi, xout => eq5_uid176_fracXIsZero_uid31_fpMulTest_q, clk => clk, aclr => areset );

    -- eq4_uid173_fracXIsZero_uid31_fpMulTest(LOGICAL,172)@0 + 1
    eq4_uid173_fracXIsZero_uid31_fpMulTest_qi <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_f = c4_uid142_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq4_uid173_fracXIsZero_uid31_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq4_uid173_fracXIsZero_uid31_fpMulTest_qi, xout => eq4_uid173_fracXIsZero_uid31_fpMulTest_q, clk => clk, aclr => areset );

    -- eq3_uid170_fracXIsZero_uid31_fpMulTest(LOGICAL,169)@0 + 1
    eq3_uid170_fracXIsZero_uid31_fpMulTest_qi <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_e = c3_uid139_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq3_uid170_fracXIsZero_uid31_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq3_uid170_fracXIsZero_uid31_fpMulTest_qi, xout => eq3_uid170_fracXIsZero_uid31_fpMulTest_q, clk => clk, aclr => areset );

    -- eq2_uid167_fracXIsZero_uid31_fpMulTest(LOGICAL,166)@0 + 1
    eq2_uid167_fracXIsZero_uid31_fpMulTest_qi <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_d = c2_uid136_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq2_uid167_fracXIsZero_uid31_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq2_uid167_fracXIsZero_uid31_fpMulTest_qi, xout => eq2_uid167_fracXIsZero_uid31_fpMulTest_q, clk => clk, aclr => areset );

    -- eq1_uid164_fracXIsZero_uid31_fpMulTest(LOGICAL,163)@0 + 1
    eq1_uid164_fracXIsZero_uid31_fpMulTest_qi <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_c = c1_uid133_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq1_uid164_fracXIsZero_uid31_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq1_uid164_fracXIsZero_uid31_fpMulTest_qi, xout => eq1_uid164_fracXIsZero_uid31_fpMulTest_q, clk => clk, aclr => areset );

    -- eq0_uid161_fracXIsZero_uid31_fpMulTest(LOGICAL,160)@0 + 1
    eq0_uid161_fracXIsZero_uid31_fpMulTest_qi <= "1" WHEN z0_uid159_fracXIsZero_uid31_fpMulTest_merged_bit_select_b = c0_uid130_fracXIsZero_uid17_fpMulTest_b ELSE "0";
    eq0_uid161_fracXIsZero_uid31_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => eq0_uid161_fracXIsZero_uid31_fpMulTest_qi, xout => eq0_uid161_fracXIsZero_uid31_fpMulTest_q, clk => clk, aclr => areset );

    -- and_lev0_uid186_fracXIsZero_uid31_fpMulTest(LOGICAL,185)@1
    and_lev0_uid186_fracXIsZero_uid31_fpMulTest_q <= eq0_uid161_fracXIsZero_uid31_fpMulTest_q and eq1_uid164_fracXIsZero_uid31_fpMulTest_q and eq2_uid167_fracXIsZero_uid31_fpMulTest_q and eq3_uid170_fracXIsZero_uid31_fpMulTest_q and eq4_uid173_fracXIsZero_uid31_fpMulTest_q and eq5_uid176_fracXIsZero_uid31_fpMulTest_q;

    -- and_lev1_uid188_fracXIsZero_uid31_fpMulTest(LOGICAL,187)@1 + 1
    and_lev1_uid188_fracXIsZero_uid31_fpMulTest_qi <= and_lev0_uid186_fracXIsZero_uid31_fpMulTest_q and and_lev0_uid187_fracXIsZero_uid31_fpMulTest_q;
    and_lev1_uid188_fracXIsZero_uid31_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev1_uid188_fracXIsZero_uid31_fpMulTest_qi, xout => and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q, clk => clk, aclr => areset );

    -- redist36_and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q_16(DELAY,390)
    redist36_and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q_16 : dspba_delay
    GENERIC MAP ( width => 1, depth => 15, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q, xout => redist36_and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q_16_q, clk => clk, aclr => areset );

    -- expXIsMax_uid30_fpMulTest(LOGICAL,29)@0 + 1
    expXIsMax_uid30_fpMulTest_qi <= "1" WHEN expY_uid7_fpMulTest_b = cstAllOWE_uid10_fpMulTest_q ELSE "0";
    expXIsMax_uid30_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => expXIsMax_uid30_fpMulTest_qi, xout => expXIsMax_uid30_fpMulTest_q, clk => clk, aclr => areset );

    -- redist49_expXIsMax_uid30_fpMulTest_q_17(DELAY,403)
    redist49_expXIsMax_uid30_fpMulTest_q_17 : dspba_delay
    GENERIC MAP ( width => 1, depth => 16, reset_kind => "ASYNC" )
    PORT MAP ( xin => expXIsMax_uid30_fpMulTest_q, xout => redist49_expXIsMax_uid30_fpMulTest_q_17_q, clk => clk, aclr => areset );

    -- excI_y_uid33_fpMulTest(LOGICAL,32)@17 + 1
    excI_y_uid33_fpMulTest_qi <= redist49_expXIsMax_uid30_fpMulTest_q_17_q and redist36_and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q_16_q;
    excI_y_uid33_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excI_y_uid33_fpMulTest_qi, xout => excI_y_uid33_fpMulTest_q, clk => clk, aclr => areset );

    -- excZ_x_uid15_fpMulTest(LOGICAL,14)@0 + 1
    excZ_x_uid15_fpMulTest_qi <= "1" WHEN expX_uid6_fpMulTest_b = cstAllZWE_uid12_fpMulTest_q ELSE "0";
    excZ_x_uid15_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excZ_x_uid15_fpMulTest_qi, xout => excZ_x_uid15_fpMulTest_q, clk => clk, aclr => areset );

    -- redist54_excZ_x_uid15_fpMulTest_q_17(DELAY,408)
    redist54_excZ_x_uid15_fpMulTest_q_17 : dspba_delay
    GENERIC MAP ( width => 1, depth => 16, reset_kind => "ASYNC" )
    PORT MAP ( xin => excZ_x_uid15_fpMulTest_q, xout => redist54_excZ_x_uid15_fpMulTest_q_17_q, clk => clk, aclr => areset );

    -- redist55_excZ_x_uid15_fpMulTest_q_18(DELAY,409)
    redist55_excZ_x_uid15_fpMulTest_q_18 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist54_excZ_x_uid15_fpMulTest_q_17_q, xout => redist55_excZ_x_uid15_fpMulTest_q_18_q, clk => clk, aclr => areset );

    -- excXZAndExcYI_uid76_fpMulTest(LOGICAL,75)@18
    excXZAndExcYI_uid76_fpMulTest_q <= redist55_excZ_x_uid15_fpMulTest_q_18_q and excI_y_uid33_fpMulTest_q;

    -- ZeroTimesInf_uid77_fpMulTest(LOGICAL,76)@18 + 1
    ZeroTimesInf_uid77_fpMulTest_qi <= excXZAndExcYI_uid76_fpMulTest_q or excYZAndExcXI_uid75_fpMulTest_q;
    ZeroTimesInf_uid77_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => ZeroTimesInf_uid77_fpMulTest_qi, xout => ZeroTimesInf_uid77_fpMulTest_q, clk => clk, aclr => areset );

    -- fracXIsNotZero_uid32_fpMulTest(LOGICAL,31)@17
    fracXIsNotZero_uid32_fpMulTest_q <= not (redist36_and_lev1_uid188_fracXIsZero_uid31_fpMulTest_q_16_q);

    -- excN_y_uid34_fpMulTest(LOGICAL,33)@17 + 1
    excN_y_uid34_fpMulTest_qi <= redist49_expXIsMax_uid30_fpMulTest_q_17_q and fracXIsNotZero_uid32_fpMulTest_q;
    excN_y_uid34_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excN_y_uid34_fpMulTest_qi, xout => excN_y_uid34_fpMulTest_q, clk => clk, aclr => areset );

    -- redist48_excN_y_uid34_fpMulTest_q_2(DELAY,402)
    redist48_excN_y_uid34_fpMulTest_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excN_y_uid34_fpMulTest_q, xout => redist48_excN_y_uid34_fpMulTest_q_2_q, clk => clk, aclr => areset );

    -- fracXIsNotZero_uid18_fpMulTest(LOGICAL,17)@17
    fracXIsNotZero_uid18_fpMulTest_q <= not (redist37_and_lev1_uid158_fracXIsZero_uid17_fpMulTest_q_16_q);

    -- excN_x_uid20_fpMulTest(LOGICAL,19)@17 + 1
    excN_x_uid20_fpMulTest_qi <= redist53_expXIsMax_uid16_fpMulTest_q_17_q and fracXIsNotZero_uid18_fpMulTest_q;
    excN_x_uid20_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excN_x_uid20_fpMulTest_qi, xout => excN_x_uid20_fpMulTest_q, clk => clk, aclr => areset );

    -- redist52_excN_x_uid20_fpMulTest_q_2(DELAY,406)
    redist52_excN_x_uid20_fpMulTest_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excN_x_uid20_fpMulTest_q, xout => redist52_excN_x_uid20_fpMulTest_q_2_q, clk => clk, aclr => areset );

    -- excRNaN_uid78_fpMulTest(LOGICAL,77)@19
    excRNaN_uid78_fpMulTest_q <= redist52_excN_x_uid20_fpMulTest_q_2_q or redist48_excN_y_uid34_fpMulTest_q_2_q or ZeroTimesInf_uid77_fpMulTest_q;

    -- invExcRNaN_uid90_fpMulTest(LOGICAL,89)@19
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

    -- redist46_signR_uid48_fpMulTest_q_19(DELAY,400)
    redist46_signR_uid48_fpMulTest_q_19 : dspba_delay
    GENERIC MAP ( width => 1, depth => 18, reset_kind => "ASYNC" )
    PORT MAP ( xin => signR_uid48_fpMulTest_q, xout => redist46_signR_uid48_fpMulTest_q_19_q, clk => clk, aclr => areset );

    -- VCC(CONSTANT,1)
    VCC_q <= "1";

    -- signRPostExc_uid91_fpMulTest(LOGICAL,90)@19 + 1
    signRPostExc_uid91_fpMulTest_qi <= redist46_signR_uid48_fpMulTest_q_19_q and invExcRNaN_uid90_fpMulTest_q;
    signRPostExc_uid91_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => signRPostExc_uid91_fpMulTest_qi, xout => signRPostExc_uid91_fpMulTest_q, clk => clk, aclr => areset );

    -- GND(CONSTANT,0)
    GND_q <= "0";

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel0_1(BITSELECT,275)
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel0_1_b <= STD_LOGIC_VECTOR(cstZeroWF_uid11_fpMulTest_q(15 downto 0));

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_b(BITJOIN,276)@13
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_b_q <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel0_1_b & VCC_q;

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_UpperBits_for_b(CONSTANT,222)
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

    -- redist42_rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_b_1(DELAY,396)
    redist42_rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_b_1 : dspba_delay
    GENERIC MAP ( width => 26, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_b, xout => redist42_rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_b_1_q, clk => clk, aclr => areset );

    -- rightBottomX_mergedSignalTM_uid113_prod_uid47_fpMulTest(BITJOIN,112)@1
    rightBottomX_mergedSignalTM_uid113_prod_uid47_fpMulTest_q <= redist42_rightBottomX_bottomRange_uid112_prod_uid47_fpMulTest_b_1_q & GND_q;

    -- topRangeY_uid105_prod_uid47_fpMulTest(BITSELECT,104)@0
    topRangeY_uid105_prod_uid47_fpMulTest_b <= ofracY_uid43_fpMulTest_q(52 downto 26);

    -- redist43_topRangeY_uid105_prod_uid47_fpMulTest_b_1(DELAY,397)
    redist43_topRangeY_uid105_prod_uid47_fpMulTest_b_1 : dspba_delay
    GENERIC MAP ( width => 27, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => topRangeY_uid105_prod_uid47_fpMulTest_b, xout => redist43_topRangeY_uid105_prod_uid47_fpMulTest_b_1_q, clk => clk, aclr => areset );

    -- multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma(CHAINMULTADD,345)@0 + 2
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
                multSumOfTwoTS_uid119_prod_uid47_fpMulTest_cma_a0(0) <= RESIZE(UNSIGNED(redist43_topRangeY_uid105_prod_uid47_fpMulTest_b_1_q),27);
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

    -- redist38_highBBits_uid124_prod_uid47_fpMulTest_b_5(DELAY,392)
    redist38_highBBits_uid124_prod_uid47_fpMulTest_b_5 : dspba_delay
    GENERIC MAP ( width => 38, depth => 5, reset_kind => "ASYNC" )
    PORT MAP ( xin => highBBits_uid124_prod_uid47_fpMulTest_b, xout => redist38_highBBits_uid124_prod_uid47_fpMulTest_b_5_q, clk => clk, aclr => areset );

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_b(BITJOIN,221)@8
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_b_q <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_UpperBits_for_b_q & redist38_highBBits_uid124_prod_uid47_fpMulTest_b_5_q;

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b(BITSELECT,224)@8
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_b <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_b_q(16 downto 0);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_c <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_b_q(33 downto 17);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_d <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_b_q(50 downto 34);

    -- expFracRPostRounding_uid58_fpMulTest_UpperBits_for_b(CONSTANT,211)
    expFracRPostRounding_uid58_fpMulTest_UpperBits_for_b_q <= "0000000000000";

    -- sm0_uid118_prod_uid47_fpMulTest_bs1_merged_bit_select(BITSELECT,348)@0
    sm0_uid118_prod_uid47_fpMulTest_bs1_merged_bit_select_b <= topRangeX_uid104_prod_uid47_fpMulTest_b(26 downto 16);
    sm0_uid118_prod_uid47_fpMulTest_bs1_merged_bit_select_c <= topRangeX_uid104_prod_uid47_fpMulTest_b(15 downto 0);

    -- redist4_sm0_uid118_prod_uid47_fpMulTest_bs1_merged_bit_select_b_1(DELAY,358)
    redist4_sm0_uid118_prod_uid47_fpMulTest_bs1_merged_bit_select_b_1 : dspba_delay
    GENERIC MAP ( width => 11, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_bs1_merged_bit_select_b, xout => redist4_sm0_uid118_prod_uid47_fpMulTest_bs1_merged_bit_select_b_1_q, clk => clk, aclr => areset );

    -- sm0_uid118_prod_uid47_fpMulTest_bs2_merged_bit_select(BITSELECT,349)@0
    sm0_uid118_prod_uid47_fpMulTest_bs2_merged_bit_select_b <= topRangeY_uid105_prod_uid47_fpMulTest_b(26 downto 16);
    sm0_uid118_prod_uid47_fpMulTest_bs2_merged_bit_select_c <= topRangeY_uid105_prod_uid47_fpMulTest_b(15 downto 0);

    -- redist3_sm0_uid118_prod_uid47_fpMulTest_bs2_merged_bit_select_c_1(DELAY,357)
    redist3_sm0_uid118_prod_uid47_fpMulTest_bs2_merged_bit_select_c_1 : dspba_delay
    GENERIC MAP ( width => 16, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_bs2_merged_bit_select_c, xout => redist3_sm0_uid118_prod_uid47_fpMulTest_bs2_merged_bit_select_c_1_q, clk => clk, aclr => areset );

    -- sm0_uid118_prod_uid47_fpMulTest_im6_cma(CHAINMULTADD,343)@1 + 2
    sm0_uid118_prod_uid47_fpMulTest_im6_cma_reset <= areset;
    sm0_uid118_prod_uid47_fpMulTest_im6_cma_ena0 <= '1';
    sm0_uid118_prod_uid47_fpMulTest_im6_cma_ena1 <= sm0_uid118_prod_uid47_fpMulTest_im6_cma_ena0;
    sm0_uid118_prod_uid47_fpMulTest_im6_cma_p(0) <= sm0_uid118_prod_uid47_fpMulTest_im6_cma_a0(0) * sm0_uid118_prod_uid47_fpMulTest_im6_cma_c0(0);
    sm0_uid118_prod_uid47_fpMulTest_im6_cma_u(0) <= RESIZE(sm0_uid118_prod_uid47_fpMulTest_im6_cma_p(0),27);
    sm0_uid118_prod_uid47_fpMulTest_im6_cma_w(0) <= sm0_uid118_prod_uid47_fpMulTest_im6_cma_u(0);
    sm0_uid118_prod_uid47_fpMulTest_im6_cma_x(0) <= sm0_uid118_prod_uid47_fpMulTest_im6_cma_w(0);
    sm0_uid118_prod_uid47_fpMulTest_im6_cma_y(0) <= sm0_uid118_prod_uid47_fpMulTest_im6_cma_x(0);
    sm0_uid118_prod_uid47_fpMulTest_im6_cma_chainmultadd_input: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_im6_cma_a0 <= (others => (others => '0'));
            sm0_uid118_prod_uid47_fpMulTest_im6_cma_c0 <= (others => (others => '0'));
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (sm0_uid118_prod_uid47_fpMulTest_im6_cma_ena0 = '1') THEN
                sm0_uid118_prod_uid47_fpMulTest_im6_cma_a0(0) <= RESIZE(UNSIGNED(redist3_sm0_uid118_prod_uid47_fpMulTest_bs2_merged_bit_select_c_1_q),16);
                sm0_uid118_prod_uid47_fpMulTest_im6_cma_c0(0) <= RESIZE(UNSIGNED(redist4_sm0_uid118_prod_uid47_fpMulTest_bs1_merged_bit_select_b_1_q),11);
            END IF;
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_im6_cma_chainmultadd_output: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_im6_cma_s <= (others => (others => '0'));
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (sm0_uid118_prod_uid47_fpMulTest_im6_cma_ena1 = '1') THEN
                sm0_uid118_prod_uid47_fpMulTest_im6_cma_s(0) <= sm0_uid118_prod_uid47_fpMulTest_im6_cma_y(0);
            END IF;
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_im6_cma_delay : dspba_delay
    GENERIC MAP ( width => 27, depth => 0, reset_kind => "ASYNC" )
    PORT MAP ( xin => STD_LOGIC_VECTOR(sm0_uid118_prod_uid47_fpMulTest_im6_cma_s(0)(26 downto 0)), xout => sm0_uid118_prod_uid47_fpMulTest_im6_cma_qq, clk => clk, aclr => areset );
    sm0_uid118_prod_uid47_fpMulTest_im6_cma_q <= STD_LOGIC_VECTOR(sm0_uid118_prod_uid47_fpMulTest_im6_cma_qq(26 downto 0));

    -- redist6_sm0_uid118_prod_uid47_fpMulTest_im6_cma_q_1(DELAY,360)
    redist6_sm0_uid118_prod_uid47_fpMulTest_im6_cma_q_1 : dspba_delay
    GENERIC MAP ( width => 27, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_im6_cma_q, xout => redist6_sm0_uid118_prod_uid47_fpMulTest_im6_cma_q_1_q, clk => clk, aclr => areset );

    -- sm0_uid118_prod_uid47_fpMulTest_align_15(BITSHIFT,203)@4
    sm0_uid118_prod_uid47_fpMulTest_align_15_qint <= redist6_sm0_uid118_prod_uid47_fpMulTest_im6_cma_q_1_q & "0000000000000000";
    sm0_uid118_prod_uid47_fpMulTest_align_15_q <= sm0_uid118_prod_uid47_fpMulTest_align_15_qint(42 downto 0);

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitExpansion_for_b(BITJOIN,243)@4
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitExpansion_for_b_q <= expFracRPostRounding_uid58_fpMulTest_UpperBits_for_b_q & sm0_uid118_prod_uid47_fpMulTest_align_15_q;

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b(BITSELECT,246)@4
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_b <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitExpansion_for_b_q(16 downto 0);
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_c <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitExpansion_for_b_q(33 downto 17);
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_d <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitExpansion_for_b_q(50 downto 34);

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_UpperBits_for_b(CONSTANT,233)
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_UpperBits_for_b_q <= "000000000000";

    -- sm0_uid118_prod_uid47_fpMulTest_im3_cma(CHAINMULTADD,342)@0 + 2
    sm0_uid118_prod_uid47_fpMulTest_im3_cma_reset <= areset;
    sm0_uid118_prod_uid47_fpMulTest_im3_cma_ena0 <= '1';
    sm0_uid118_prod_uid47_fpMulTest_im3_cma_ena1 <= sm0_uid118_prod_uid47_fpMulTest_im3_cma_ena0;
    sm0_uid118_prod_uid47_fpMulTest_im3_cma_p(0) <= sm0_uid118_prod_uid47_fpMulTest_im3_cma_a0(0) * sm0_uid118_prod_uid47_fpMulTest_im3_cma_c0(0);
    sm0_uid118_prod_uid47_fpMulTest_im3_cma_u(0) <= RESIZE(sm0_uid118_prod_uid47_fpMulTest_im3_cma_p(0),27);
    sm0_uid118_prod_uid47_fpMulTest_im3_cma_w(0) <= sm0_uid118_prod_uid47_fpMulTest_im3_cma_u(0);
    sm0_uid118_prod_uid47_fpMulTest_im3_cma_x(0) <= sm0_uid118_prod_uid47_fpMulTest_im3_cma_w(0);
    sm0_uid118_prod_uid47_fpMulTest_im3_cma_y(0) <= sm0_uid118_prod_uid47_fpMulTest_im3_cma_x(0);
    sm0_uid118_prod_uid47_fpMulTest_im3_cma_chainmultadd_input: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_im3_cma_a0 <= (others => (others => '0'));
            sm0_uid118_prod_uid47_fpMulTest_im3_cma_c0 <= (others => (others => '0'));
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (sm0_uid118_prod_uid47_fpMulTest_im3_cma_ena0 = '1') THEN
                sm0_uid118_prod_uid47_fpMulTest_im3_cma_a0(0) <= RESIZE(UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_bs1_merged_bit_select_c),16);
                sm0_uid118_prod_uid47_fpMulTest_im3_cma_c0(0) <= RESIZE(UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_bs2_merged_bit_select_b),11);
            END IF;
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_im3_cma_chainmultadd_output: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_im3_cma_s <= (others => (others => '0'));
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (sm0_uid118_prod_uid47_fpMulTest_im3_cma_ena1 = '1') THEN
                sm0_uid118_prod_uid47_fpMulTest_im3_cma_s(0) <= sm0_uid118_prod_uid47_fpMulTest_im3_cma_y(0);
            END IF;
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_im3_cma_delay : dspba_delay
    GENERIC MAP ( width => 27, depth => 0, reset_kind => "ASYNC" )
    PORT MAP ( xin => STD_LOGIC_VECTOR(sm0_uid118_prod_uid47_fpMulTest_im3_cma_s(0)(26 downto 0)), xout => sm0_uid118_prod_uid47_fpMulTest_im3_cma_qq, clk => clk, aclr => areset );
    sm0_uid118_prod_uid47_fpMulTest_im3_cma_q <= STD_LOGIC_VECTOR(sm0_uid118_prod_uid47_fpMulTest_im3_cma_qq(26 downto 0));

    -- redist7_sm0_uid118_prod_uid47_fpMulTest_im3_cma_q_1(DELAY,361)
    redist7_sm0_uid118_prod_uid47_fpMulTest_im3_cma_q_1 : dspba_delay
    GENERIC MAP ( width => 27, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_im3_cma_q, xout => redist7_sm0_uid118_prod_uid47_fpMulTest_im3_cma_q_1_q, clk => clk, aclr => areset );

    -- sm0_uid118_prod_uid47_fpMulTest_align_13(BITSHIFT,201)@3
    sm0_uid118_prod_uid47_fpMulTest_align_13_qint <= redist7_sm0_uid118_prod_uid47_fpMulTest_im3_cma_q_1_q & "0000000000000000";
    sm0_uid118_prod_uid47_fpMulTest_align_13_q <= sm0_uid118_prod_uid47_fpMulTest_align_13_qint(42 downto 0);

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitExpansion_for_b(BITJOIN,232)@3
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitExpansion_for_b_q <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_UpperBits_for_b_q & sm0_uid118_prod_uid47_fpMulTest_align_13_q;

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b(BITSELECT,235)@3
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_b <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitExpansion_for_b_q(16 downto 0);
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_c <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitExpansion_for_b_q(33 downto 17);
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_d <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitExpansion_for_b_q(50 downto 34);

    -- sm0_uid118_prod_uid47_fpMulTest_im0_cma(CHAINMULTADD,341)@0 + 2
    sm0_uid118_prod_uid47_fpMulTest_im0_cma_reset <= areset;
    sm0_uid118_prod_uid47_fpMulTest_im0_cma_ena0 <= '1';
    sm0_uid118_prod_uid47_fpMulTest_im0_cma_ena1 <= sm0_uid118_prod_uid47_fpMulTest_im0_cma_ena0;
    sm0_uid118_prod_uid47_fpMulTest_im0_cma_p(0) <= sm0_uid118_prod_uid47_fpMulTest_im0_cma_a0(0) * sm0_uid118_prod_uid47_fpMulTest_im0_cma_c0(0);
    sm0_uid118_prod_uid47_fpMulTest_im0_cma_u(0) <= RESIZE(sm0_uid118_prod_uid47_fpMulTest_im0_cma_p(0),22);
    sm0_uid118_prod_uid47_fpMulTest_im0_cma_w(0) <= sm0_uid118_prod_uid47_fpMulTest_im0_cma_u(0);
    sm0_uid118_prod_uid47_fpMulTest_im0_cma_x(0) <= sm0_uid118_prod_uid47_fpMulTest_im0_cma_w(0);
    sm0_uid118_prod_uid47_fpMulTest_im0_cma_y(0) <= sm0_uid118_prod_uid47_fpMulTest_im0_cma_x(0);
    sm0_uid118_prod_uid47_fpMulTest_im0_cma_chainmultadd_input: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_im0_cma_a0 <= (others => (others => '0'));
            sm0_uid118_prod_uid47_fpMulTest_im0_cma_c0 <= (others => (others => '0'));
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (sm0_uid118_prod_uid47_fpMulTest_im0_cma_ena0 = '1') THEN
                sm0_uid118_prod_uid47_fpMulTest_im0_cma_a0(0) <= RESIZE(UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_bs1_merged_bit_select_b),11);
                sm0_uid118_prod_uid47_fpMulTest_im0_cma_c0(0) <= RESIZE(UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_bs2_merged_bit_select_b),11);
            END IF;
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_im0_cma_chainmultadd_output: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_im0_cma_s <= (others => (others => '0'));
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (sm0_uid118_prod_uid47_fpMulTest_im0_cma_ena1 = '1') THEN
                sm0_uid118_prod_uid47_fpMulTest_im0_cma_s(0) <= sm0_uid118_prod_uid47_fpMulTest_im0_cma_y(0);
            END IF;
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_im0_cma_delay : dspba_delay
    GENERIC MAP ( width => 22, depth => 0, reset_kind => "ASYNC" )
    PORT MAP ( xin => STD_LOGIC_VECTOR(sm0_uid118_prod_uid47_fpMulTest_im0_cma_s(0)(21 downto 0)), xout => sm0_uid118_prod_uid47_fpMulTest_im0_cma_qq, clk => clk, aclr => areset );
    sm0_uid118_prod_uid47_fpMulTest_im0_cma_q <= STD_LOGIC_VECTOR(sm0_uid118_prod_uid47_fpMulTest_im0_cma_qq(21 downto 0));

    -- redist8_sm0_uid118_prod_uid47_fpMulTest_im0_cma_q_1(DELAY,362)
    redist8_sm0_uid118_prod_uid47_fpMulTest_im0_cma_q_1 : dspba_delay
    GENERIC MAP ( width => 22, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_im0_cma_q, xout => redist8_sm0_uid118_prod_uid47_fpMulTest_im0_cma_q_1_q, clk => clk, aclr => areset );

    -- sm0_uid118_prod_uid47_fpMulTest_im9_cma(CHAINMULTADD,344)@0 + 2
    sm0_uid118_prod_uid47_fpMulTest_im9_cma_reset <= areset;
    sm0_uid118_prod_uid47_fpMulTest_im9_cma_ena0 <= '1';
    sm0_uid118_prod_uid47_fpMulTest_im9_cma_ena1 <= sm0_uid118_prod_uid47_fpMulTest_im9_cma_ena0;
    sm0_uid118_prod_uid47_fpMulTest_im9_cma_p(0) <= sm0_uid118_prod_uid47_fpMulTest_im9_cma_a0(0) * sm0_uid118_prod_uid47_fpMulTest_im9_cma_c0(0);
    sm0_uid118_prod_uid47_fpMulTest_im9_cma_u(0) <= RESIZE(sm0_uid118_prod_uid47_fpMulTest_im9_cma_p(0),32);
    sm0_uid118_prod_uid47_fpMulTest_im9_cma_w(0) <= sm0_uid118_prod_uid47_fpMulTest_im9_cma_u(0);
    sm0_uid118_prod_uid47_fpMulTest_im9_cma_x(0) <= sm0_uid118_prod_uid47_fpMulTest_im9_cma_w(0);
    sm0_uid118_prod_uid47_fpMulTest_im9_cma_y(0) <= sm0_uid118_prod_uid47_fpMulTest_im9_cma_x(0);
    sm0_uid118_prod_uid47_fpMulTest_im9_cma_chainmultadd_input: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_im9_cma_a0 <= (others => (others => '0'));
            sm0_uid118_prod_uid47_fpMulTest_im9_cma_c0 <= (others => (others => '0'));
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (sm0_uid118_prod_uid47_fpMulTest_im9_cma_ena0 = '1') THEN
                sm0_uid118_prod_uid47_fpMulTest_im9_cma_a0(0) <= RESIZE(UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_bs1_merged_bit_select_c),16);
                sm0_uid118_prod_uid47_fpMulTest_im9_cma_c0(0) <= RESIZE(UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_bs2_merged_bit_select_c),16);
            END IF;
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_im9_cma_chainmultadd_output: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_im9_cma_s <= (others => (others => '0'));
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (sm0_uid118_prod_uid47_fpMulTest_im9_cma_ena1 = '1') THEN
                sm0_uid118_prod_uid47_fpMulTest_im9_cma_s(0) <= sm0_uid118_prod_uid47_fpMulTest_im9_cma_y(0);
            END IF;
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_im9_cma_delay : dspba_delay
    GENERIC MAP ( width => 32, depth => 0, reset_kind => "ASYNC" )
    PORT MAP ( xin => STD_LOGIC_VECTOR(sm0_uid118_prod_uid47_fpMulTest_im9_cma_s(0)(31 downto 0)), xout => sm0_uid118_prod_uid47_fpMulTest_im9_cma_qq, clk => clk, aclr => areset );
    sm0_uid118_prod_uid47_fpMulTest_im9_cma_q <= STD_LOGIC_VECTOR(sm0_uid118_prod_uid47_fpMulTest_im9_cma_qq(31 downto 0));

    -- redist5_sm0_uid118_prod_uid47_fpMulTest_im9_cma_q_1(DELAY,359)
    redist5_sm0_uid118_prod_uid47_fpMulTest_im9_cma_q_1 : dspba_delay
    GENERIC MAP ( width => 32, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_im9_cma_q, xout => redist5_sm0_uid118_prod_uid47_fpMulTest_im9_cma_q_1_q, clk => clk, aclr => areset );

    -- sm0_uid118_prod_uid47_fpMulTest_join_12(BITJOIN,200)@3
    sm0_uid118_prod_uid47_fpMulTest_join_12_q <= redist8_sm0_uid118_prod_uid47_fpMulTest_im0_cma_q_1_q & redist5_sm0_uid118_prod_uid47_fpMulTest_im9_cma_q_1_q;

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitExpansion_for_a(BITJOIN,230)@3
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitExpansion_for_a_q <= GND_q & sm0_uid118_prod_uid47_fpMulTest_join_12_q;

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a(BITSELECT,234)@3
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_b <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitExpansion_for_a_q(16 downto 0);
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_c <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitExpansion_for_a_q(33 downto 17);
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_d <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitExpansion_for_a_q(50 downto 34);
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_e <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitExpansion_for_a_q(54 downto 51);

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4(ADD,236)@3 + 1
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_a <= STD_LOGIC_VECTOR("0" & sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_b);
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_b <= STD_LOGIC_VECTOR("0" & sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_b);
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_a) + UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_b));
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_c(0) <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_o(17);
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_q <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_o(16 downto 0);

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4(ADD,247)@4 + 1
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_a <= STD_LOGIC_VECTOR("0" & sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_q);
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_b <= STD_LOGIC_VECTOR("0" & sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_b);
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_a) + UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_b));
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_c(0) <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_o(17);
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_q <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_o(16 downto 0);

    -- redist21_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_c_1(DELAY,375)
    redist21_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_c_1 : dspba_delay
    GENERIC MAP ( width => 17, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_c, xout => redist21_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_c_1_q, clk => clk, aclr => areset );

    -- redist23_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_c_1(DELAY,377)
    redist23_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_c_1 : dspba_delay
    GENERIC MAP ( width => 17, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_c, xout => redist23_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_c_1_q, clk => clk, aclr => areset );

    -- redist25_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_c_1(DELAY,379)
    redist25_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_c_1 : dspba_delay
    GENERIC MAP ( width => 17, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_c, xout => redist25_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_c_1_q, clk => clk, aclr => areset );

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4(ADD,237)@4 + 1
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_cin <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p1_of_4_c;
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_a <= STD_LOGIC_VECTOR("0" & redist25_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_c_1_q) & '1';
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_b <= STD_LOGIC_VECTOR("0" & redist23_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_c_1_q) & sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_cin(0);
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_a) + UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_b));
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_c(0) <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_o(18);
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_q <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_o(17 downto 1);

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4(ADD,248)@5 + 1
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_cin <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_c;
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_a <= STD_LOGIC_VECTOR("0" & sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_q) & '1';
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_b <= STD_LOGIC_VECTOR("0" & redist21_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_c_1_q) & sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_cin(0);
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_a) + UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_b));
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_c(0) <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_o(18);
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_q <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_o(17 downto 1);

    -- redist22_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_d_2(DELAY,376)
    redist22_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_d_2 : dspba_delay
    GENERIC MAP ( width => 17, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_d, xout => redist22_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_d_2_q, clk => clk, aclr => areset );

    -- redist24_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_d_2(DELAY,378)
    redist24_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_d_2 : dspba_delay
    GENERIC MAP ( width => 17, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_d, xout => redist24_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_d_2_q, clk => clk, aclr => areset );

    -- redist26_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_d_2(DELAY,380)
    redist26_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_d_2 : dspba_delay
    GENERIC MAP ( width => 17, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_d, xout => redist26_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_d_2_q, clk => clk, aclr => areset );

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4(ADD,238)@5 + 1
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_cin <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p2_of_4_c;
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_a <= STD_LOGIC_VECTOR("0" & redist26_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_d_2_q) & '1';
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_b <= STD_LOGIC_VECTOR("0" & redist24_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_d_2_q) & sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_cin(0);
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_a) + UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_b));
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_c(0) <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_o(18);
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_q <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_o(17 downto 1);

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4(ADD,249)@6 + 1
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_cin <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_c;
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_a <= STD_LOGIC_VECTOR("0" & sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_q) & '1';
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_b <= STD_LOGIC_VECTOR("0" & redist22_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_d_2_q) & sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_cin(0);
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_a) + UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_b));
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_c(0) <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_o(18);
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_o(17 downto 1);

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_tessel3_0(BITSELECT,309)
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_tessel3_0_b <= STD_LOGIC_VECTOR(expFracRPostRounding_uid58_fpMulTest_UpperBits_for_b_q(12 downto 8));

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_tessel3_0(BITSELECT,298)
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_tessel3_0_b <= STD_LOGIC_VECTOR(sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_UpperBits_for_b_q(11 downto 8));

    -- redist27_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_e_3(DELAY,381)
    redist27_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_e_3 : dspba_delay
    GENERIC MAP ( width => 4, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_e, xout => redist27_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_e_3_q, clk => clk, aclr => areset );

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4(ADD,239)@6 + 1
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_cin <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p3_of_4_c;
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_a <= STD_LOGIC_VECTOR("0" & redist27_sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_a_e_3_q) & '1';
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_b <= STD_LOGIC_VECTOR("0" & sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_BitSelect_for_b_tessel3_0_b) & sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_cin(0);
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_a) + UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_b));
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_q <= sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_o(4 downto 1);

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_a_BitJoin_for_e(BITJOIN,308)@7
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_a_BitJoin_for_e_q <= GND_q & sm0_uid118_prod_uid47_fpMulTest_result_add_0_0_p4_of_4_q;

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4(ADD,250)@7 + 1
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_cin <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_c;
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_a <= STD_LOGIC_VECTOR("0" & sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_a_BitJoin_for_e_q) & '1';
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_b <= STD_LOGIC_VECTOR("0" & sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitSelect_for_b_tessel3_0_b) & sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_cin(0);
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_a) + UNSIGNED(sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_b));
        END IF;
    END PROCESS;
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_q <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_o(5 downto 1);

    -- redist17_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q_1(DELAY,371)
    redist17_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q_1 : dspba_delay
    GENERIC MAP ( width => 17, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q, xout => redist17_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q_1_q, clk => clk, aclr => areset );

    -- redist19_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_q_2(DELAY,373)
    redist19_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_q_2 : dspba_delay
    GENERIC MAP ( width => 17, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_q, xout => redist19_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_q_2_q, clk => clk, aclr => areset );

    -- redist20_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_q_3(DELAY,374)
    redist20_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_q_3 : dspba_delay
    GENERIC MAP ( width => 17, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_q, xout => redist20_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_q_3_q, clk => clk, aclr => areset );

    -- sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitJoin_for_q(BITJOIN,251)@8
    sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitJoin_for_q_q <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_q & redist17_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q_1_q & redist19_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_q_2_q & redist20_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_q_3_q;

    -- aboveLeftY_uid117_prod_uid47_fpMulTest(BITSELECT,116)@0
    aboveLeftY_uid117_prod_uid47_fpMulTest_in <= ofracY_uid43_fpMulTest_q(25 downto 0);
    aboveLeftY_uid117_prod_uid47_fpMulTest_b <= aboveLeftY_uid117_prod_uid47_fpMulTest_in(25 downto 21);

    -- redist40_aboveLeftY_uid117_prod_uid47_fpMulTest_b_5(DELAY,394)
    redist40_aboveLeftY_uid117_prod_uid47_fpMulTest_b_5 : dspba_delay
    GENERIC MAP ( width => 5, depth => 5, reset_kind => "ASYNC" )
    PORT MAP ( xin => aboveLeftY_uid117_prod_uid47_fpMulTest_b, xout => redist40_aboveLeftY_uid117_prod_uid47_fpMulTest_b_5_q, clk => clk, aclr => areset );

    -- aboveLeftX_uid116_prod_uid47_fpMulTest(BITSELECT,115)@0
    aboveLeftX_uid116_prod_uid47_fpMulTest_in <= ofracX_uid40_fpMulTest_q(25 downto 0);
    aboveLeftX_uid116_prod_uid47_fpMulTest_b <= aboveLeftX_uid116_prod_uid47_fpMulTest_in(25 downto 21);

    -- redist41_aboveLeftX_uid116_prod_uid47_fpMulTest_b_5(DELAY,395)
    redist41_aboveLeftX_uid116_prod_uid47_fpMulTest_b_5 : dspba_delay
    GENERIC MAP ( width => 5, depth => 5, reset_kind => "ASYNC" )
    PORT MAP ( xin => aboveLeftX_uid116_prod_uid47_fpMulTest_b, xout => redist41_aboveLeftX_uid116_prod_uid47_fpMulTest_b_5_q, clk => clk, aclr => areset );

    -- sm0_uid121_prod_uid47_fpMulTest(MULT,120)@5 + 2
    sm0_uid121_prod_uid47_fpMulTest_pr <= UNSIGNED(sm0_uid121_prod_uid47_fpMulTest_a0) * UNSIGNED(sm0_uid121_prod_uid47_fpMulTest_b0);
    sm0_uid121_prod_uid47_fpMulTest_component: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sm0_uid121_prod_uid47_fpMulTest_a0 <= (others => '0');
            sm0_uid121_prod_uid47_fpMulTest_b0 <= (others => '0');
            sm0_uid121_prod_uid47_fpMulTest_s1 <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            sm0_uid121_prod_uid47_fpMulTest_a0 <= redist41_aboveLeftX_uid116_prod_uid47_fpMulTest_b_5_q;
            sm0_uid121_prod_uid47_fpMulTest_b0 <= redist40_aboveLeftY_uid117_prod_uid47_fpMulTest_b_5_q;
            sm0_uid121_prod_uid47_fpMulTest_s1 <= STD_LOGIC_VECTOR(sm0_uid121_prod_uid47_fpMulTest_pr);
        END IF;
    END PROCESS;
    sm0_uid121_prod_uid47_fpMulTest_q <= sm0_uid121_prod_uid47_fpMulTest_s1;

    -- redist39_sm0_uid121_prod_uid47_fpMulTest_q_1(DELAY,393)
    redist39_sm0_uid121_prod_uid47_fpMulTest_q_1 : dspba_delay
    GENERIC MAP ( width => 10, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sm0_uid121_prod_uid47_fpMulTest_q, xout => redist39_sm0_uid121_prod_uid47_fpMulTest_q_1_q, clk => clk, aclr => areset );

    -- sumAb_uid122_prod_uid47_fpMulTest(BITJOIN,121)@8
    sumAb_uid122_prod_uid47_fpMulTest_q <= sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_BitJoin_for_q_q(53 downto 0) & redist39_sm0_uid121_prod_uid47_fpMulTest_q_1_q;

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_a(BITJOIN,219)@8
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_a_q <= GND_q & sumAb_uid122_prod_uid47_fpMulTest_q;

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a(BITSELECT,223)@8
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_b <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitExpansion_for_a_q(16 downto 0);

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4(ADD,225)@8 + 1
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_a <= STD_LOGIC_VECTOR("0" & lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_b);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_b <= STD_LOGIC_VECTOR("0" & lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_b);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_a) + UNSIGNED(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_b));
        END IF;
    END PROCESS;
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_c(0) <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_o(17);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_q <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_o(16 downto 0);

    -- redist30_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_c_1(DELAY,384)
    redist30_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_c_1 : dspba_delay
    GENERIC MAP ( width => 17, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_c, xout => redist30_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_c_1_q, clk => clk, aclr => areset );

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select(BITSELECT,350)@8
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_b <= STD_LOGIC_VECTOR(redist19_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_q_2_q(6 downto 0));
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_c <= STD_LOGIC_VECTOR(redist19_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p2_of_4_q_2_q(16 downto 7));

    -- redist1_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_b_1(DELAY,355)
    redist1_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_b_1 : dspba_delay
    GENERIC MAP ( width => 7, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_b, xout => redist1_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_b_1_q, clk => clk, aclr => areset );

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_0(BITSELECT,286)@8
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(redist20_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p1_of_4_q_3_q(16 downto 7));

    -- redist13_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_0_b_1(DELAY,367)
    redist13_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_0_b_1 : dspba_delay
    GENERIC MAP ( width => 10, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_0_b, xout => redist13_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_0_b_1_q, clk => clk, aclr => areset );

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_BitJoin_for_c(BITJOIN,288)@9
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_BitJoin_for_c_q <= redist1_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_b_1_q & redist13_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_0_b_1_q;

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4(ADD,226)@9 + 1
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_cin <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_c;
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_a <= STD_LOGIC_VECTOR("0" & lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_BitJoin_for_c_q) & '1';
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_b <= STD_LOGIC_VECTOR("0" & redist30_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_c_1_q) & lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_cin(0);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_a) + UNSIGNED(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_b));
        END IF;
    END PROCESS;
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_c(0) <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_o(18);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_q <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_o(17 downto 1);

    -- redist29_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_q_2(DELAY,383)
    redist29_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_q_2 : dspba_delay
    GENERIC MAP ( width => 17, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_q, xout => redist29_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_q_2_q, clk => clk, aclr => areset );

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel2_0(BITSELECT,331)@12
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel2_0_b <= STD_LOGIC_VECTOR(redist29_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_q_2_q(16 downto 1));

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel2_0(BITSELECT,317)@12
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel2_0_b <= STD_LOGIC_VECTOR(redist29_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_q_2_q(15 downto 0));

    -- redist31_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_d_2(DELAY,385)
    redist31_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_d_2 : dspba_delay
    GENERIC MAP ( width => 17, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_d, xout => redist31_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_d_2_q, clk => clk, aclr => areset );

    -- redist18_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q_3(DELAY,372)
    redist18_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q_3 : dspba_delay
    GENERIC MAP ( width => 17, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist17_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q_1_q, xout => redist18_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q_3_q, clk => clk, aclr => areset );

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel2_1_merged_bit_select(BITSELECT,351)@10
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel2_1_merged_bit_select_b <= STD_LOGIC_VECTOR(redist18_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q_3_q(6 downto 0));
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel2_1_merged_bit_select_c <= STD_LOGIC_VECTOR(redist18_sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p3_of_4_q_3_q(16 downto 7));

    -- redist2_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_c_2(DELAY,356)
    redist2_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_c_2 : dspba_delay
    GENERIC MAP ( width => 10, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_c, xout => redist2_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_c_2_q, clk => clk, aclr => areset );

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_BitJoin_for_d(BITJOIN,291)@10
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_BitJoin_for_d_q <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel2_1_merged_bit_select_b & redist2_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel1_1_merged_bit_select_c_2_q;

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4(ADD,227)@10 + 1
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_cin <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_c;
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_a <= STD_LOGIC_VECTOR("0" & lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_BitJoin_for_d_q) & '1';
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_b <= STD_LOGIC_VECTOR("0" & redist31_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_d_2_q) & lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_cin(0);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_a) + UNSIGNED(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_b));
        END IF;
    END PROCESS;
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_c(0) <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_o(18);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_q <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_o(17 downto 1);

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_tessel3_0(BITSELECT,296)
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_tessel3_0_b <= STD_LOGIC_VECTOR(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_UpperBits_for_b_q(26 downto 13));

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel3_1(BITSELECT,293)@8
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel3_1_b <= STD_LOGIC_VECTOR(sm0_uid118_prod_uid47_fpMulTest_result_add_1_0_p4_of_4_q(2 downto 0));

    -- redist12_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel3_1_b_3(DELAY,366)
    redist12_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel3_1_b_3 : dspba_delay
    GENERIC MAP ( width => 3, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel3_1_b, xout => redist12_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel3_1_b_3_q, clk => clk, aclr => areset );

    -- redist0_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel2_1_merged_bit_select_c_1(DELAY,354)
    redist0_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel2_1_merged_bit_select_c_1 : dspba_delay
    GENERIC MAP ( width => 10, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel2_1_merged_bit_select_c, xout => redist0_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel2_1_merged_bit_select_c_1_q, clk => clk, aclr => areset );

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_BitJoin_for_e(BITJOIN,295)@11
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_BitJoin_for_e_q <= GND_q & redist12_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel3_1_b_3_q & redist0_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_tessel2_1_merged_bit_select_c_1_q;

    -- lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4(ADD,228)@11 + 1
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_cin <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_c;
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_a <= STD_LOGIC_VECTOR("0" & lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_a_BitJoin_for_e_q) & '1';
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_b <= STD_LOGIC_VECTOR("0" & lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_BitSelect_for_b_tessel3_0_b) & lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_cin(0);
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_a) + UNSIGNED(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_b));
        END IF;
    END PROCESS;
    lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_q <= lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_o(14 downto 1);

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1(BITSELECT,282)@12
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b <= STD_LOGIC_VECTOR(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_q(12 downto 12));

    -- fracRPostNorm_uid53_fpMulTest_p2(MUX,256)@12 + 1
    fracRPostNorm_uid53_fpMulTest_p2_s <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b;
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

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel0_2_merged_bit_select(BITSELECT,352)@13
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel0_2_merged_bit_select_b <= STD_LOGIC_VECTOR(fracRPostNorm_uid53_fpMulTest_p2_q(8 downto 0));
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel0_2_merged_bit_select_c <= STD_LOGIC_VECTOR(fracRPostNorm_uid53_fpMulTest_p2_q(15 downto 9));

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel1_0(BITSELECT,329)@12
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel1_0_b <= STD_LOGIC_VECTOR(redist29_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_q_2_q(0 downto 0));

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel1_0(BITSELECT,315)@9
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_q(16 downto 16));

    -- redist10_fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel1_0_b_3(DELAY,364)
    redist10_fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel1_0_b_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel1_0_b, xout => redist10_fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel1_0_b_3_q, clk => clk, aclr => areset );

    -- fracRPostNorm_uid53_fpMulTest_p1(MUX,255)@12 + 1
    fracRPostNorm_uid53_fpMulTest_p1_s <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b;
    fracRPostNorm_uid53_fpMulTest_p1_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            fracRPostNorm_uid53_fpMulTest_p1_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (fracRPostNorm_uid53_fpMulTest_p1_s) IS
                WHEN "0" => fracRPostNorm_uid53_fpMulTest_p1_q <= redist10_fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel1_0_b_3_q;
                WHEN "1" => fracRPostNorm_uid53_fpMulTest_p1_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel1_0_b;
                WHEN OTHERS => fracRPostNorm_uid53_fpMulTest_p1_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel0_0(BITSELECT,327)@9
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel0_0_b <= STD_LOGIC_VECTOR(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_q(16 downto 10));

    -- redist9_fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel0_0_b_3(DELAY,363)
    redist9_fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel0_0_b_3 : dspba_delay
    GENERIC MAP ( width => 7, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel0_0_b, xout => redist9_fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel0_0_b_3_q, clk => clk, aclr => areset );

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel0_0(BITSELECT,313)@9
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel0_0_b <= STD_LOGIC_VECTOR(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p1_of_4_q(15 downto 9));

    -- redist11_fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel0_0_b_3(DELAY,365)
    redist11_fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel0_0_b_3 : dspba_delay
    GENERIC MAP ( width => 7, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel0_0_b, xout => redist11_fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel0_0_b_3_q, clk => clk, aclr => areset );

    -- fracRPostNorm_uid53_fpMulTest_p0(MUX,254)@12 + 1
    fracRPostNorm_uid53_fpMulTest_p0_s <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b;
    fracRPostNorm_uid53_fpMulTest_p0_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            fracRPostNorm_uid53_fpMulTest_p0_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (fracRPostNorm_uid53_fpMulTest_p0_s) IS
                WHEN "0" => fracRPostNorm_uid53_fpMulTest_p0_q <= redist11_fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel0_0_b_3_q;
                WHEN "1" => fracRPostNorm_uid53_fpMulTest_p0_q <= redist9_fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel0_0_b_3_q;
                WHEN OTHERS => fracRPostNorm_uid53_fpMulTest_p0_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_b(BITJOIN,265)@13
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_b_q <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel0_2_merged_bit_select_b & fracRPostNorm_uid53_fpMulTest_p1_q & fracRPostNorm_uid53_fpMulTest_p0_q;

    -- expFracRPostRounding_uid58_fpMulTest_p1_of_4(ADD,214)@13 + 1
    expFracRPostRounding_uid58_fpMulTest_p1_of_4_a <= STD_LOGIC_VECTOR("0" & expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_b_q);
    expFracRPostRounding_uid58_fpMulTest_p1_of_4_b <= STD_LOGIC_VECTOR("0" & expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_b_q);
    expFracRPostRounding_uid58_fpMulTest_p1_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expFracRPostRounding_uid58_fpMulTest_p1_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expFracRPostRounding_uid58_fpMulTest_p1_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(expFracRPostRounding_uid58_fpMulTest_p1_of_4_a) + UNSIGNED(expFracRPostRounding_uid58_fpMulTest_p1_of_4_b));
        END IF;
    END PROCESS;
    expFracRPostRounding_uid58_fpMulTest_p1_of_4_c(0) <= expFracRPostRounding_uid58_fpMulTest_p1_of_4_o(17);
    expFracRPostRounding_uid58_fpMulTest_p1_of_4_q <= expFracRPostRounding_uid58_fpMulTest_p1_of_4_o(16 downto 0);

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_0(BITSELECT,277)
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_0_b <= STD_LOGIC_VECTOR(cstZeroWF_uid11_fpMulTest_q(32 downto 16));

    -- redist28_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_q_1(DELAY,382)
    redist28_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_q_1 : dspba_delay
    GENERIC MAP ( width => 17, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_q, xout => redist28_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_q_1_q, clk => clk, aclr => areset );

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel4_0(BITSELECT,335)@12
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel4_0_b <= STD_LOGIC_VECTOR(redist28_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_q_1_q(16 downto 1));

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel4_0(BITSELECT,321)@12
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel4_0_b <= STD_LOGIC_VECTOR(redist28_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_q_1_q(15 downto 0));

    -- fracRPostNorm_uid53_fpMulTest_p4(MUX,258)@12 + 1
    fracRPostNorm_uid53_fpMulTest_p4_s <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b;
    fracRPostNorm_uid53_fpMulTest_p4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            fracRPostNorm_uid53_fpMulTest_p4_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (fracRPostNorm_uid53_fpMulTest_p4_s) IS
                WHEN "0" => fracRPostNorm_uid53_fpMulTest_p4_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel4_0_b;
                WHEN "1" => fracRPostNorm_uid53_fpMulTest_p4_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel4_0_b;
                WHEN OTHERS => fracRPostNorm_uid53_fpMulTest_p4_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel1_2_merged_bit_select(BITSELECT,353)@13
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel1_2_merged_bit_select_b <= STD_LOGIC_VECTOR(fracRPostNorm_uid53_fpMulTest_p4_q(8 downto 0));
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel1_2_merged_bit_select_c <= STD_LOGIC_VECTOR(fracRPostNorm_uid53_fpMulTest_p4_q(15 downto 9));

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel3_0(BITSELECT,333)@12
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel3_0_b <= STD_LOGIC_VECTOR(redist28_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_q_1_q(0 downto 0));

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel3_0(BITSELECT,319)@12
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel3_0_b <= STD_LOGIC_VECTOR(redist29_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p2_of_4_q_2_q(16 downto 16));

    -- fracRPostNorm_uid53_fpMulTest_p3(MUX,257)@12 + 1
    fracRPostNorm_uid53_fpMulTest_p3_s <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b;
    fracRPostNorm_uid53_fpMulTest_p3_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            fracRPostNorm_uid53_fpMulTest_p3_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (fracRPostNorm_uid53_fpMulTest_p3_s) IS
                WHEN "0" => fracRPostNorm_uid53_fpMulTest_p3_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel3_0_b;
                WHEN "1" => fracRPostNorm_uid53_fpMulTest_p3_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel3_0_b;
                WHEN OTHERS => fracRPostNorm_uid53_fpMulTest_p3_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_c(BITJOIN,269)@13
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_c_q <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel1_2_merged_bit_select_b & fracRPostNorm_uid53_fpMulTest_p3_q & expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel0_2_merged_bit_select_c;

    -- redist16_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_c_q_1(DELAY,370)
    redist16_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_c_q_1 : dspba_delay
    GENERIC MAP ( width => 17, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_c_q, xout => redist16_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_c_q_1_q, clk => clk, aclr => areset );

    -- expFracRPostRounding_uid58_fpMulTest_p2_of_4(ADD,215)@14 + 1
    expFracRPostRounding_uid58_fpMulTest_p2_of_4_cin <= expFracRPostRounding_uid58_fpMulTest_p1_of_4_c;
    expFracRPostRounding_uid58_fpMulTest_p2_of_4_a <= STD_LOGIC_VECTOR("0" & redist16_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_c_q_1_q) & '1';
    expFracRPostRounding_uid58_fpMulTest_p2_of_4_b <= STD_LOGIC_VECTOR("0" & expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel1_0_b) & expFracRPostRounding_uid58_fpMulTest_p2_of_4_cin(0);
    expFracRPostRounding_uid58_fpMulTest_p2_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expFracRPostRounding_uid58_fpMulTest_p2_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expFracRPostRounding_uid58_fpMulTest_p2_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(expFracRPostRounding_uid58_fpMulTest_p2_of_4_a) + UNSIGNED(expFracRPostRounding_uid58_fpMulTest_p2_of_4_b));
        END IF;
    END PROCESS;
    expFracRPostRounding_uid58_fpMulTest_p2_of_4_c(0) <= expFracRPostRounding_uid58_fpMulTest_p2_of_4_o(18);
    expFracRPostRounding_uid58_fpMulTest_p2_of_4_q <= expFracRPostRounding_uid58_fpMulTest_p2_of_4_o(17 downto 1);

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel2_0(BITSELECT,279)
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel2_0_b <= STD_LOGIC_VECTOR(cstZeroWF_uid11_fpMulTest_q(49 downto 33));

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel6_0(BITSELECT,339)@12
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel6_0_b <= STD_LOGIC_VECTOR(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_q(11 downto 1));

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel6_0(BITSELECT,325)@12
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel6_0_b <= STD_LOGIC_VECTOR(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_q(10 downto 0));

    -- fracRPostNorm_uid53_fpMulTest_p6(MUX,260)@12 + 1
    fracRPostNorm_uid53_fpMulTest_p6_s <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b;
    fracRPostNorm_uid53_fpMulTest_p6_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            fracRPostNorm_uid53_fpMulTest_p6_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (fracRPostNorm_uid53_fpMulTest_p6_s) IS
                WHEN "0" => fracRPostNorm_uid53_fpMulTest_p6_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel6_0_b;
                WHEN "1" => fracRPostNorm_uid53_fpMulTest_p6_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel6_0_b;
                WHEN OTHERS => fracRPostNorm_uid53_fpMulTest_p6_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel2_2(BITSELECT,272)@13
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel2_2_b <= STD_LOGIC_VECTOR(fracRPostNorm_uid53_fpMulTest_p6_q(8 downto 0));

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel5_0(BITSELECT,337)@12
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel5_0_b <= STD_LOGIC_VECTOR(lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p4_of_4_q(0 downto 0));

    -- fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel5_0(BITSELECT,323)@12
    fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel5_0_b <= STD_LOGIC_VECTOR(redist28_lev1_a0sumAHighB_uid125_prod_uid47_fpMulTest_p3_of_4_q_1_q(16 downto 16));

    -- fracRPostNorm_uid53_fpMulTest_p5(MUX,259)@12 + 1
    fracRPostNorm_uid53_fpMulTest_p5_s <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b;
    fracRPostNorm_uid53_fpMulTest_p5_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            fracRPostNorm_uid53_fpMulTest_p5_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (fracRPostNorm_uid53_fpMulTest_p5_s) IS
                WHEN "0" => fracRPostNorm_uid53_fpMulTest_p5_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_a_tessel5_0_b;
                WHEN "1" => fracRPostNorm_uid53_fpMulTest_p5_q <= fracRPostNorm_uid53_fpMulTest_BitSelect_for_b_tessel5_0_b;
                WHEN OTHERS => fracRPostNorm_uid53_fpMulTest_p5_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_d(BITJOIN,273)@13
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_d_q <= expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel2_2_b & fracRPostNorm_uid53_fpMulTest_p5_q & expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_tessel1_2_merged_bit_select_c;

    -- redist15_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_d_q_2(DELAY,369)
    redist15_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_d_q_2 : dspba_delay
    GENERIC MAP ( width => 17, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_d_q, xout => redist15_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_d_q_2_q, clk => clk, aclr => areset );

    -- expFracRPostRounding_uid58_fpMulTest_p3_of_4(ADD,216)@15 + 1
    expFracRPostRounding_uid58_fpMulTest_p3_of_4_cin <= expFracRPostRounding_uid58_fpMulTest_p2_of_4_c;
    expFracRPostRounding_uid58_fpMulTest_p3_of_4_a <= STD_LOGIC_VECTOR("0" & redist15_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_BitJoin_for_d_q_2_q) & '1';
    expFracRPostRounding_uid58_fpMulTest_p3_of_4_b <= STD_LOGIC_VECTOR("0" & expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel2_0_b) & expFracRPostRounding_uid58_fpMulTest_p3_of_4_cin(0);
    expFracRPostRounding_uid58_fpMulTest_p3_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expFracRPostRounding_uid58_fpMulTest_p3_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expFracRPostRounding_uid58_fpMulTest_p3_of_4_o <= STD_LOGIC_VECTOR(UNSIGNED(expFracRPostRounding_uid58_fpMulTest_p3_of_4_a) + UNSIGNED(expFracRPostRounding_uid58_fpMulTest_p3_of_4_b));
        END IF;
    END PROCESS;
    expFracRPostRounding_uid58_fpMulTest_p3_of_4_c(0) <= expFracRPostRounding_uid58_fpMulTest_p3_of_4_o(18);
    expFracRPostRounding_uid58_fpMulTest_p3_of_4_q <= expFracRPostRounding_uid58_fpMulTest_p3_of_4_o(17 downto 1);

    -- redist14_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b_4(DELAY,368)
    redist14_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 4, reset_kind => "ASYNC" )
    PORT MAP ( xin => expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b, xout => redist14_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b_4_q, clk => clk, aclr => areset );

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_0(BITSELECT,281)
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_0_b <= STD_LOGIC_VECTOR(cstZeroWF_uid11_fpMulTest_q(51 downto 50));

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_e(BITJOIN,285)@16
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_e_q <= expFracRPostRounding_uid58_fpMulTest_UpperBits_for_b_q & GND_q & redist14_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_1_b_4_q & expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_tessel3_0_b;

    -- expFracRPostRounding_uid58_fpMulTest_SignBit_for_a(BITSELECT,208)@13
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

    -- redist47_expSum_uid44_fpMulTest_q_13(DELAY,401)
    redist47_expSum_uid44_fpMulTest_q_13 : dspba_delay
    GENERIC MAP ( width => 12, depth => 12, reset_kind => "ASYNC" )
    PORT MAP ( xin => expSum_uid44_fpMulTest_q, xout => redist47_expSum_uid44_fpMulTest_q_13_q, clk => clk, aclr => areset );

    -- expSumMBias_uid46_fpMulTest(SUB,45)@13
    expSumMBias_uid46_fpMulTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("000" & redist47_expSum_uid44_fpMulTest_q_13_q));
    expSumMBias_uid46_fpMulTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((14 downto 13 => biasInc_uid45_fpMulTest_q(12)) & biasInc_uid45_fpMulTest_q));
    expSumMBias_uid46_fpMulTest_o <= STD_LOGIC_VECTOR(SIGNED(expSumMBias_uid46_fpMulTest_a) - SIGNED(expSumMBias_uid46_fpMulTest_b));
    expSumMBias_uid46_fpMulTest_q <= expSumMBias_uid46_fpMulTest_o(13 downto 0);

    -- fracRPostNorm_uid53_fpMulTest_BitJoin_for_q(BITJOIN,261)@13
    fracRPostNorm_uid53_fpMulTest_BitJoin_for_q_q <= fracRPostNorm_uid53_fpMulTest_p6_q & fracRPostNorm_uid53_fpMulTest_p5_q & fracRPostNorm_uid53_fpMulTest_p4_q & fracRPostNorm_uid53_fpMulTest_p3_q & fracRPostNorm_uid53_fpMulTest_p2_q & fracRPostNorm_uid53_fpMulTest_p1_q & fracRPostNorm_uid53_fpMulTest_p0_q;

    -- expFracPreRound_uid55_fpMulTest(BITJOIN,54)@13
    expFracPreRound_uid55_fpMulTest_q <= expSumMBias_uid46_fpMulTest_q & fracRPostNorm_uid53_fpMulTest_BitJoin_for_q_q;

    -- expFracRPostRounding_uid58_fpMulTest_BitExpansion_for_a(BITJOIN,207)@13
    expFracRPostRounding_uid58_fpMulTest_BitExpansion_for_a_q <= expFracRPostRounding_uid58_fpMulTest_SignBit_for_a_b & expFracPreRound_uid55_fpMulTest_q;

    -- expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a(BITSELECT,212)@13
    expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_e <= STD_LOGIC_VECTOR(expFracRPostRounding_uid58_fpMulTest_BitExpansion_for_a_q(67 downto 51));

    -- redist35_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_e_3(DELAY,389)
    redist35_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_e_3 : dspba_delay
    GENERIC MAP ( width => 17, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_e, xout => redist35_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_e_3_q, clk => clk, aclr => areset );

    -- expFracRPostRounding_uid58_fpMulTest_p4_of_4(ADD,217)@16 + 1
    expFracRPostRounding_uid58_fpMulTest_p4_of_4_cin <= expFracRPostRounding_uid58_fpMulTest_p3_of_4_c;
    expFracRPostRounding_uid58_fpMulTest_p4_of_4_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((17 downto 17 => redist35_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_e_3_q(16)) & redist35_expFracRPostRounding_uid58_fpMulTest_BitSelect_for_a_e_3_q) & '1');
    expFracRPostRounding_uid58_fpMulTest_p4_of_4_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0" & expFracRPostRounding_uid58_fpMulTest_BitSelect_for_b_BitJoin_for_e_q) & expFracRPostRounding_uid58_fpMulTest_p4_of_4_cin(0));
    expFracRPostRounding_uid58_fpMulTest_p4_of_4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expFracRPostRounding_uid58_fpMulTest_p4_of_4_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expFracRPostRounding_uid58_fpMulTest_p4_of_4_o <= STD_LOGIC_VECTOR(SIGNED(expFracRPostRounding_uid58_fpMulTest_p4_of_4_a) + SIGNED(expFracRPostRounding_uid58_fpMulTest_p4_of_4_b));
        END IF;
    END PROCESS;
    expFracRPostRounding_uid58_fpMulTest_p4_of_4_q <= expFracRPostRounding_uid58_fpMulTest_p4_of_4_o(17 downto 1);

    -- redist32_expFracRPostRounding_uid58_fpMulTest_p3_of_4_q_1(DELAY,386)
    redist32_expFracRPostRounding_uid58_fpMulTest_p3_of_4_q_1 : dspba_delay
    GENERIC MAP ( width => 17, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => expFracRPostRounding_uid58_fpMulTest_p3_of_4_q, xout => redist32_expFracRPostRounding_uid58_fpMulTest_p3_of_4_q_1_q, clk => clk, aclr => areset );

    -- redist33_expFracRPostRounding_uid58_fpMulTest_p2_of_4_q_2(DELAY,387)
    redist33_expFracRPostRounding_uid58_fpMulTest_p2_of_4_q_2 : dspba_delay
    GENERIC MAP ( width => 17, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => expFracRPostRounding_uid58_fpMulTest_p2_of_4_q, xout => redist33_expFracRPostRounding_uid58_fpMulTest_p2_of_4_q_2_q, clk => clk, aclr => areset );

    -- redist34_expFracRPostRounding_uid58_fpMulTest_p1_of_4_q_3(DELAY,388)
    redist34_expFracRPostRounding_uid58_fpMulTest_p1_of_4_q_3 : dspba_delay
    GENERIC MAP ( width => 17, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => expFracRPostRounding_uid58_fpMulTest_p1_of_4_q, xout => redist34_expFracRPostRounding_uid58_fpMulTest_p1_of_4_q_3_q, clk => clk, aclr => areset );

    -- expFracRPostRounding_uid58_fpMulTest_BitJoin_for_q(BITJOIN,218)@17
    expFracRPostRounding_uid58_fpMulTest_BitJoin_for_q_q <= expFracRPostRounding_uid58_fpMulTest_p4_of_4_q & redist32_expFracRPostRounding_uid58_fpMulTest_p3_of_4_q_1_q & redist33_expFracRPostRounding_uid58_fpMulTest_p2_of_4_q_2_q & redist34_expFracRPostRounding_uid58_fpMulTest_p1_of_4_q_3_q;

    -- expRPreExcExt_uid60_fpMulTest(BITSELECT,59)@17
    expRPreExcExt_uid60_fpMulTest_b <= STD_LOGIC_VECTOR(expFracRPostRounding_uid58_fpMulTest_BitJoin_for_q_q(67 downto 53));

    -- expRPreExc_uid61_fpMulTest(BITSELECT,60)@17
    expRPreExc_uid61_fpMulTest_in <= expRPreExcExt_uid60_fpMulTest_b(10 downto 0);
    expRPreExc_uid61_fpMulTest_b <= expRPreExc_uid61_fpMulTest_in(10 downto 0);

    -- redist44_expRPreExc_uid61_fpMulTest_b_3(DELAY,398)
    redist44_expRPreExc_uid61_fpMulTest_b_3 : dspba_delay
    GENERIC MAP ( width => 11, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => expRPreExc_uid61_fpMulTest_b, xout => redist44_expRPreExc_uid61_fpMulTest_b_3_q, clk => clk, aclr => areset );

    -- expOvf_uid64_fpMulTest(COMPARE,63)@17 + 1
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

    -- invExpXIsMax_uid35_fpMulTest(LOGICAL,34)@17
    invExpXIsMax_uid35_fpMulTest_q <= not (redist49_expXIsMax_uid30_fpMulTest_q_17_q);

    -- InvExpXIsZero_uid36_fpMulTest(LOGICAL,35)@17
    InvExpXIsZero_uid36_fpMulTest_q <= not (redist50_excZ_y_uid29_fpMulTest_q_17_q);

    -- excR_y_uid37_fpMulTest(LOGICAL,36)@17 + 1
    excR_y_uid37_fpMulTest_qi <= InvExpXIsZero_uid36_fpMulTest_q and invExpXIsMax_uid35_fpMulTest_q;
    excR_y_uid37_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excR_y_uid37_fpMulTest_qi, xout => excR_y_uid37_fpMulTest_q, clk => clk, aclr => areset );

    -- invExpXIsMax_uid21_fpMulTest(LOGICAL,20)@17
    invExpXIsMax_uid21_fpMulTest_q <= not (redist53_expXIsMax_uid16_fpMulTest_q_17_q);

    -- InvExpXIsZero_uid22_fpMulTest(LOGICAL,21)@17
    InvExpXIsZero_uid22_fpMulTest_q <= not (redist54_excZ_x_uid15_fpMulTest_q_17_q);

    -- excR_x_uid23_fpMulTest(LOGICAL,22)@17 + 1
    excR_x_uid23_fpMulTest_qi <= InvExpXIsZero_uid22_fpMulTest_q and invExpXIsMax_uid21_fpMulTest_q;
    excR_x_uid23_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excR_x_uid23_fpMulTest_qi, xout => excR_x_uid23_fpMulTest_q, clk => clk, aclr => areset );

    -- ExcROvfAndInReg_uid73_fpMulTest(LOGICAL,72)@18
    ExcROvfAndInReg_uid73_fpMulTest_q <= excR_x_uid23_fpMulTest_q and excR_y_uid37_fpMulTest_q and expOvf_uid64_fpMulTest_n;

    -- excYRAndExcXI_uid72_fpMulTest(LOGICAL,71)@18
    excYRAndExcXI_uid72_fpMulTest_q <= excR_y_uid37_fpMulTest_q and excI_x_uid19_fpMulTest_q;

    -- excXRAndExcYI_uid71_fpMulTest(LOGICAL,70)@18
    excXRAndExcYI_uid71_fpMulTest_q <= excR_x_uid23_fpMulTest_q and excI_y_uid33_fpMulTest_q;

    -- excXIAndExcYI_uid70_fpMulTest(LOGICAL,69)@18
    excXIAndExcYI_uid70_fpMulTest_q <= excI_x_uid19_fpMulTest_q and excI_y_uid33_fpMulTest_q;

    -- excRInf_uid74_fpMulTest(LOGICAL,73)@18 + 1
    excRInf_uid74_fpMulTest_qi <= excXIAndExcYI_uid70_fpMulTest_q or excXRAndExcYI_uid71_fpMulTest_q or excYRAndExcXI_uid72_fpMulTest_q or ExcROvfAndInReg_uid73_fpMulTest_q;
    excRInf_uid74_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excRInf_uid74_fpMulTest_qi, xout => excRInf_uid74_fpMulTest_q, clk => clk, aclr => areset );

    -- expUdf_uid62_fpMulTest(COMPARE,61)@17 + 1
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

    -- excZC3_uid68_fpMulTest(LOGICAL,67)@18
    excZC3_uid68_fpMulTest_q <= excR_x_uid23_fpMulTest_q and excR_y_uid37_fpMulTest_q and expUdf_uid62_fpMulTest_n;

    -- excYZAndExcXR_uid67_fpMulTest(LOGICAL,66)@18
    excYZAndExcXR_uid67_fpMulTest_q <= redist51_excZ_y_uid29_fpMulTest_q_18_q and excR_x_uid23_fpMulTest_q;

    -- excXZAndExcYR_uid66_fpMulTest(LOGICAL,65)@18
    excXZAndExcYR_uid66_fpMulTest_q <= redist55_excZ_x_uid15_fpMulTest_q_18_q and excR_y_uid37_fpMulTest_q;

    -- excXZAndExcYZ_uid65_fpMulTest(LOGICAL,64)@18
    excXZAndExcYZ_uid65_fpMulTest_q <= redist55_excZ_x_uid15_fpMulTest_q_18_q and redist51_excZ_y_uid29_fpMulTest_q_18_q;

    -- excRZero_uid69_fpMulTest(LOGICAL,68)@18 + 1
    excRZero_uid69_fpMulTest_qi <= excXZAndExcYZ_uid65_fpMulTest_q or excXZAndExcYR_uid66_fpMulTest_q or excYZAndExcXR_uid67_fpMulTest_q or excZC3_uid68_fpMulTest_q;
    excRZero_uid69_fpMulTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excRZero_uid69_fpMulTest_qi, xout => excRZero_uid69_fpMulTest_q, clk => clk, aclr => areset );

    -- concExc_uid79_fpMulTest(BITJOIN,78)@19
    concExc_uid79_fpMulTest_q <= excRNaN_uid78_fpMulTest_q & excRInf_uid74_fpMulTest_q & excRZero_uid69_fpMulTest_q;

    -- excREnc_uid80_fpMulTest(LOOKUP,79)@19 + 1
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

    -- expRPostExc_uid89_fpMulTest(MUX,88)@20
    expRPostExc_uid89_fpMulTest_s <= excREnc_uid80_fpMulTest_q;
    expRPostExc_uid89_fpMulTest_combproc: PROCESS (expRPostExc_uid89_fpMulTest_s, cstAllZWE_uid12_fpMulTest_q, redist44_expRPreExc_uid61_fpMulTest_b_3_q, cstAllOWE_uid10_fpMulTest_q)
    BEGIN
        CASE (expRPostExc_uid89_fpMulTest_s) IS
            WHEN "00" => expRPostExc_uid89_fpMulTest_q <= cstAllZWE_uid12_fpMulTest_q;
            WHEN "01" => expRPostExc_uid89_fpMulTest_q <= redist44_expRPreExc_uid61_fpMulTest_b_3_q;
            WHEN "10" => expRPostExc_uid89_fpMulTest_q <= cstAllOWE_uid10_fpMulTest_q;
            WHEN "11" => expRPostExc_uid89_fpMulTest_q <= cstAllOWE_uid10_fpMulTest_q;
            WHEN OTHERS => expRPostExc_uid89_fpMulTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- oneFracRPostExc2_uid81_fpMulTest(CONSTANT,80)
    oneFracRPostExc2_uid81_fpMulTest_q <= "0000000000000000000000000000000000000000000000000001";

    -- fracRPreExc_uid59_fpMulTest(BITSELECT,58)@17
    fracRPreExc_uid59_fpMulTest_in <= expFracRPostRounding_uid58_fpMulTest_BitJoin_for_q_q(52 downto 0);
    fracRPreExc_uid59_fpMulTest_b <= fracRPreExc_uid59_fpMulTest_in(52 downto 1);

    -- redist45_fracRPreExc_uid59_fpMulTest_b_3(DELAY,399)
    redist45_fracRPreExc_uid59_fpMulTest_b_3 : dspba_delay
    GENERIC MAP ( width => 52, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => fracRPreExc_uid59_fpMulTest_b, xout => redist45_fracRPreExc_uid59_fpMulTest_b_3_q, clk => clk, aclr => areset );

    -- fracRPostExc_uid84_fpMulTest(MUX,83)@20
    fracRPostExc_uid84_fpMulTest_s <= excREnc_uid80_fpMulTest_q;
    fracRPostExc_uid84_fpMulTest_combproc: PROCESS (fracRPostExc_uid84_fpMulTest_s, cstZeroWF_uid11_fpMulTest_q, redist45_fracRPreExc_uid59_fpMulTest_b_3_q, oneFracRPostExc2_uid81_fpMulTest_q)
    BEGIN
        CASE (fracRPostExc_uid84_fpMulTest_s) IS
            WHEN "00" => fracRPostExc_uid84_fpMulTest_q <= cstZeroWF_uid11_fpMulTest_q;
            WHEN "01" => fracRPostExc_uid84_fpMulTest_q <= redist45_fracRPreExc_uid59_fpMulTest_b_3_q;
            WHEN "10" => fracRPostExc_uid84_fpMulTest_q <= cstZeroWF_uid11_fpMulTest_q;
            WHEN "11" => fracRPostExc_uid84_fpMulTest_q <= oneFracRPostExc2_uid81_fpMulTest_q;
            WHEN OTHERS => fracRPostExc_uid84_fpMulTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- R_uid92_fpMulTest(BITJOIN,91)@20
    R_uid92_fpMulTest_q <= signRPostExc_uid91_fpMulTest_q & expRPostExc_uid89_fpMulTest_q & fracRPostExc_uid84_fpMulTest_q;

    -- xOut(GPOUT,4)@20
    q <= R_uid92_fpMulTest_q;

END normal;
