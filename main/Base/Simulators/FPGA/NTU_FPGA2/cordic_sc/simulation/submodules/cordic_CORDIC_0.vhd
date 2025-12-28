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

-- VHDL created from cordic_CORDIC_0
-- VHDL created on Tue Oct 14 06:19:41 2025


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

entity cordic_CORDIC_0 is
    port (
        a : in std_logic_vector(55 downto 0);  -- sfix56_en53
        c : out std_logic_vector(54 downto 0);  -- sfix55_en53
        s : out std_logic_vector(54 downto 0);  -- sfix55_en53
        clk : in std_logic;
        areset : in std_logic
    );
end cordic_CORDIC_0;

architecture normal of cordic_CORDIC_0 is

    attribute altera_attribute : string;
    attribute altera_attribute of normal : architecture is "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name PHYSICAL_SYNTHESIS_REGISTER_DUPLICATION ON; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007";
    
    signal GND_q : STD_LOGIC_VECTOR (0 downto 0);
    signal VCC_q : STD_LOGIC_VECTOR (0 downto 0);
    signal constantZero_uid6_sincosTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal signA_uid7_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal invSignA_uid8_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal absAE_uid9_sincosTest_a : STD_LOGIC_VECTOR (57 downto 0);
    signal absAE_uid9_sincosTest_b : STD_LOGIC_VECTOR (57 downto 0);
    signal absAE_uid9_sincosTest_o : STD_LOGIC_VECTOR (57 downto 0);
    signal absAE_uid9_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal absAE_uid9_sincosTest_q : STD_LOGIC_VECTOR (56 downto 0);
    signal absAR_uid10_sincosTest_in : STD_LOGIC_VECTOR (54 downto 0);
    signal absAR_uid10_sincosTest_b : STD_LOGIC_VECTOR (54 downto 0);
    signal cstPiO2_uid11_sincosTest_q : STD_LOGIC_VECTOR (65 downto 0);
    signal padACst_uid12_sincosTest_q : STD_LOGIC_VECTOR (11 downto 0);
    signal aPostPad_uid13_sincosTest_q : STD_LOGIC_VECTOR (66 downto 0);
    signal argMPiO2_uid14_sincosTest_a : STD_LOGIC_VECTOR (67 downto 0);
    signal argMPiO2_uid14_sincosTest_b : STD_LOGIC_VECTOR (67 downto 0);
    signal argMPiO2_uid14_sincosTest_o : STD_LOGIC_VECTOR (67 downto 0);
    signal argMPiO2_uid14_sincosTest_q : STD_LOGIC_VECTOR (67 downto 0);
    signal firstQuadrant_uid15_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal absARE_bottomRange_uid17_sincosTest_in : STD_LOGIC_VECTOR (53 downto 0);
    signal absARE_bottomRange_uid17_sincosTest_b : STD_LOGIC_VECTOR (53 downto 0);
    signal absARE_mergedSignalTM_uid18_sincosTest_q : STD_LOGIC_VECTOR (65 downto 0);
    signal argMPiO2_uid20_sincosTest_in : STD_LOGIC_VECTOR (65 downto 0);
    signal argMPiO2_uid20_sincosTest_b : STD_LOGIC_VECTOR (65 downto 0);
    signal absA_uid21_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal absA_uid21_sincosTest_q : STD_LOGIC_VECTOR (65 downto 0);
    signal cstOneOverK_uid22_sincosTest_q : STD_LOGIC_VECTOR (109 downto 0);
    signal cstArcTan2Mi_0_uid26_sincosTest_q : STD_LOGIC_VECTOR (65 downto 0);
    signal xip1E_1_uid32_sincosTest_q : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1E_1CostZeroPaddingA_uid33_sincosTest_q : STD_LOGIC_VECTOR (109 downto 0);
    signal yip1E_1NA_uid34_sincosTest_q : STD_LOGIC_VECTOR (110 downto 0);
    signal yip1E_1sumAHighB_uid35_sincosTest_a : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1E_1sumAHighB_uid35_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1E_1sumAHighB_uid35_sincosTest_o : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1E_1sumAHighB_uid35_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_1sumAHighB_uid35_sincosTest_q : STD_LOGIC_VECTOR (111 downto 0);
    signal invSignOfSelectionSignal_uid36_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_1_uid37_sincosTest_a : STD_LOGIC_VECTOR (68 downto 0);
    signal aip1E_1_uid37_sincosTest_b : STD_LOGIC_VECTOR (68 downto 0);
    signal aip1E_1_uid37_sincosTest_o : STD_LOGIC_VECTOR (68 downto 0);
    signal aip1E_1_uid37_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_1_uid37_sincosTest_q : STD_LOGIC_VECTOR (67 downto 0);
    signal xip1_1_topRange_uid39_sincosTest_in : STD_LOGIC_VECTOR (111 downto 0);
    signal xip1_1_topRange_uid39_sincosTest_b : STD_LOGIC_VECTOR (111 downto 0);
    signal xip1_1_mergedSignalTM_uid40_sincosTest_q : STD_LOGIC_VECTOR (112 downto 0);
    signal xMSB_uid42_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1_1_mergedSignalTM_uid46_sincosTest_q : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid48_sincosTest_in : STD_LOGIC_VECTOR (66 downto 0);
    signal aip1E_uid48_sincosTest_b : STD_LOGIC_VECTOR (66 downto 0);
    signal xMSB_uid49_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid51_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid54_sincosTest_b : STD_LOGIC_VECTOR (111 downto 0);
    signal twoToMiSiYip_uid55_sincosTest_b : STD_LOGIC_VECTOR (111 downto 0);
    signal cstArcTan2Mi_1_uid56_sincosTest_q : STD_LOGIC_VECTOR (64 downto 0);
    signal xip1E_2_uid58_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_2_uid58_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_2_uid58_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_2_uid58_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_2_uid58_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_2_uid59_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_2_uid59_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_2_uid59_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_2_uid59_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_2_uid59_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_2_uid61_sincosTest_a : STD_LOGIC_VECTOR (68 downto 0);
    signal aip1E_2_uid61_sincosTest_b : STD_LOGIC_VECTOR (68 downto 0);
    signal aip1E_2_uid61_sincosTest_o : STD_LOGIC_VECTOR (68 downto 0);
    signal aip1E_2_uid61_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_2_uid61_sincosTest_q : STD_LOGIC_VECTOR (67 downto 0);
    signal xip1_2_uid62_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_2_uid62_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_2_uid63_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_2_uid63_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid64_sincosTest_in : STD_LOGIC_VECTOR (65 downto 0);
    signal aip1E_uid64_sincosTest_b : STD_LOGIC_VECTOR (65 downto 0);
    signal xMSB_uid65_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid67_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid70_sincosTest_b : STD_LOGIC_VECTOR (110 downto 0);
    signal twoToMiSiYip_uid71_sincosTest_b : STD_LOGIC_VECTOR (110 downto 0);
    signal cstArcTan2Mi_2_uid72_sincosTest_q : STD_LOGIC_VECTOR (63 downto 0);
    signal xip1E_3_uid74_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_3_uid74_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_3_uid74_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_3_uid74_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_3_uid74_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_3_uid75_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_3_uid75_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_3_uid75_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_3_uid75_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_3_uid75_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_3_uid77_sincosTest_a : STD_LOGIC_VECTOR (67 downto 0);
    signal aip1E_3_uid77_sincosTest_b : STD_LOGIC_VECTOR (67 downto 0);
    signal aip1E_3_uid77_sincosTest_o : STD_LOGIC_VECTOR (67 downto 0);
    signal aip1E_3_uid77_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_3_uid77_sincosTest_q : STD_LOGIC_VECTOR (66 downto 0);
    signal xip1_3_uid78_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_3_uid78_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_3_uid79_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_3_uid79_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid80_sincosTest_in : STD_LOGIC_VECTOR (64 downto 0);
    signal aip1E_uid80_sincosTest_b : STD_LOGIC_VECTOR (64 downto 0);
    signal xMSB_uid81_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid83_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid86_sincosTest_b : STD_LOGIC_VECTOR (109 downto 0);
    signal twoToMiSiYip_uid87_sincosTest_b : STD_LOGIC_VECTOR (109 downto 0);
    signal cstArcTan2Mi_3_uid88_sincosTest_q : STD_LOGIC_VECTOR (62 downto 0);
    signal xip1E_4_uid90_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_4_uid90_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_4_uid90_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_4_uid90_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_4_uid90_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_4_uid91_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_4_uid91_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_4_uid91_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_4_uid91_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_4_uid91_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_4_uid93_sincosTest_a : STD_LOGIC_VECTOR (66 downto 0);
    signal aip1E_4_uid93_sincosTest_b : STD_LOGIC_VECTOR (66 downto 0);
    signal aip1E_4_uid93_sincosTest_o : STD_LOGIC_VECTOR (66 downto 0);
    signal aip1E_4_uid93_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_4_uid93_sincosTest_q : STD_LOGIC_VECTOR (65 downto 0);
    signal xip1_4_uid94_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_4_uid94_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_4_uid95_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_4_uid95_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid96_sincosTest_in : STD_LOGIC_VECTOR (63 downto 0);
    signal aip1E_uid96_sincosTest_b : STD_LOGIC_VECTOR (63 downto 0);
    signal xMSB_uid97_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid99_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid102_sincosTest_b : STD_LOGIC_VECTOR (108 downto 0);
    signal twoToMiSiYip_uid103_sincosTest_b : STD_LOGIC_VECTOR (108 downto 0);
    signal cstArcTan2Mi_4_uid104_sincosTest_q : STD_LOGIC_VECTOR (61 downto 0);
    signal xip1E_5_uid106_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_5_uid106_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_5_uid106_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_5_uid106_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_5_uid106_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_5_uid107_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_5_uid107_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_5_uid107_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_5_uid107_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_5_uid107_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_5_uid109_sincosTest_a : STD_LOGIC_VECTOR (65 downto 0);
    signal aip1E_5_uid109_sincosTest_b : STD_LOGIC_VECTOR (65 downto 0);
    signal aip1E_5_uid109_sincosTest_o : STD_LOGIC_VECTOR (65 downto 0);
    signal aip1E_5_uid109_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_5_uid109_sincosTest_q : STD_LOGIC_VECTOR (64 downto 0);
    signal xip1_5_uid110_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_5_uid110_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_5_uid111_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_5_uid111_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid112_sincosTest_in : STD_LOGIC_VECTOR (62 downto 0);
    signal aip1E_uid112_sincosTest_b : STD_LOGIC_VECTOR (62 downto 0);
    signal xMSB_uid113_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid115_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid118_sincosTest_b : STD_LOGIC_VECTOR (107 downto 0);
    signal twoToMiSiYip_uid119_sincosTest_b : STD_LOGIC_VECTOR (107 downto 0);
    signal cstArcTan2Mi_5_uid120_sincosTest_q : STD_LOGIC_VECTOR (60 downto 0);
    signal xip1E_6_uid122_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_6_uid122_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_6_uid122_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_6_uid122_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_6_uid122_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_6_uid123_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_6_uid123_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_6_uid123_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_6_uid123_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_6_uid123_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_6_uid125_sincosTest_a : STD_LOGIC_VECTOR (64 downto 0);
    signal aip1E_6_uid125_sincosTest_b : STD_LOGIC_VECTOR (64 downto 0);
    signal aip1E_6_uid125_sincosTest_o : STD_LOGIC_VECTOR (64 downto 0);
    signal aip1E_6_uid125_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_6_uid125_sincosTest_q : STD_LOGIC_VECTOR (63 downto 0);
    signal xip1_6_uid126_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_6_uid126_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_6_uid127_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_6_uid127_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid128_sincosTest_in : STD_LOGIC_VECTOR (61 downto 0);
    signal aip1E_uid128_sincosTest_b : STD_LOGIC_VECTOR (61 downto 0);
    signal xMSB_uid129_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid131_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid134_sincosTest_b : STD_LOGIC_VECTOR (106 downto 0);
    signal twoToMiSiYip_uid135_sincosTest_b : STD_LOGIC_VECTOR (106 downto 0);
    signal cstArcTan2Mi_6_uid136_sincosTest_q : STD_LOGIC_VECTOR (59 downto 0);
    signal xip1E_7_uid138_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_7_uid138_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_7_uid138_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_7_uid138_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_7_uid138_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_7_uid139_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_7_uid139_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_7_uid139_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_7_uid139_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_7_uid139_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_7_uid141_sincosTest_a : STD_LOGIC_VECTOR (63 downto 0);
    signal aip1E_7_uid141_sincosTest_b : STD_LOGIC_VECTOR (63 downto 0);
    signal aip1E_7_uid141_sincosTest_o : STD_LOGIC_VECTOR (63 downto 0);
    signal aip1E_7_uid141_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_7_uid141_sincosTest_q : STD_LOGIC_VECTOR (62 downto 0);
    signal xip1_7_uid142_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_7_uid142_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_7_uid143_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_7_uid143_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid144_sincosTest_in : STD_LOGIC_VECTOR (60 downto 0);
    signal aip1E_uid144_sincosTest_b : STD_LOGIC_VECTOR (60 downto 0);
    signal xMSB_uid145_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid147_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid150_sincosTest_b : STD_LOGIC_VECTOR (105 downto 0);
    signal twoToMiSiYip_uid151_sincosTest_b : STD_LOGIC_VECTOR (105 downto 0);
    signal cstArcTan2Mi_7_uid152_sincosTest_q : STD_LOGIC_VECTOR (58 downto 0);
    signal xip1E_8_uid154_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_8_uid154_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_8_uid154_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_8_uid154_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_8_uid154_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_8_uid155_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_8_uid155_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_8_uid155_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_8_uid155_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_8_uid155_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_8_uid157_sincosTest_a : STD_LOGIC_VECTOR (62 downto 0);
    signal aip1E_8_uid157_sincosTest_b : STD_LOGIC_VECTOR (62 downto 0);
    signal aip1E_8_uid157_sincosTest_o : STD_LOGIC_VECTOR (62 downto 0);
    signal aip1E_8_uid157_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_8_uid157_sincosTest_q : STD_LOGIC_VECTOR (61 downto 0);
    signal xip1_8_uid158_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_8_uid158_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_8_uid159_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_8_uid159_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid160_sincosTest_in : STD_LOGIC_VECTOR (59 downto 0);
    signal aip1E_uid160_sincosTest_b : STD_LOGIC_VECTOR (59 downto 0);
    signal xMSB_uid161_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid163_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid166_sincosTest_b : STD_LOGIC_VECTOR (104 downto 0);
    signal twoToMiSiYip_uid167_sincosTest_b : STD_LOGIC_VECTOR (104 downto 0);
    signal cstArcTan2Mi_8_uid168_sincosTest_q : STD_LOGIC_VECTOR (57 downto 0);
    signal xip1E_9_uid170_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_9_uid170_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_9_uid170_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_9_uid170_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_9_uid170_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_9_uid171_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_9_uid171_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_9_uid171_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_9_uid171_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_9_uid171_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_9_uid173_sincosTest_a : STD_LOGIC_VECTOR (61 downto 0);
    signal aip1E_9_uid173_sincosTest_b : STD_LOGIC_VECTOR (61 downto 0);
    signal aip1E_9_uid173_sincosTest_o : STD_LOGIC_VECTOR (61 downto 0);
    signal aip1E_9_uid173_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_9_uid173_sincosTest_q : STD_LOGIC_VECTOR (60 downto 0);
    signal xip1_9_uid174_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_9_uid174_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_9_uid175_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_9_uid175_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid176_sincosTest_in : STD_LOGIC_VECTOR (58 downto 0);
    signal aip1E_uid176_sincosTest_b : STD_LOGIC_VECTOR (58 downto 0);
    signal xMSB_uid177_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid179_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid182_sincosTest_b : STD_LOGIC_VECTOR (103 downto 0);
    signal twoToMiSiYip_uid183_sincosTest_b : STD_LOGIC_VECTOR (103 downto 0);
    signal cstArcTan2Mi_9_uid184_sincosTest_q : STD_LOGIC_VECTOR (56 downto 0);
    signal xip1E_10_uid186_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_10_uid186_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_10_uid186_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_10_uid186_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_10_uid186_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_10_uid187_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_10_uid187_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_10_uid187_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_10_uid187_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_10_uid187_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_10_uid189_sincosTest_a : STD_LOGIC_VECTOR (60 downto 0);
    signal aip1E_10_uid189_sincosTest_b : STD_LOGIC_VECTOR (60 downto 0);
    signal aip1E_10_uid189_sincosTest_o : STD_LOGIC_VECTOR (60 downto 0);
    signal aip1E_10_uid189_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_10_uid189_sincosTest_q : STD_LOGIC_VECTOR (59 downto 0);
    signal xip1_10_uid190_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_10_uid190_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_10_uid191_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_10_uid191_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid192_sincosTest_in : STD_LOGIC_VECTOR (57 downto 0);
    signal aip1E_uid192_sincosTest_b : STD_LOGIC_VECTOR (57 downto 0);
    signal xMSB_uid193_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid195_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid198_sincosTest_b : STD_LOGIC_VECTOR (102 downto 0);
    signal twoToMiSiYip_uid199_sincosTest_b : STD_LOGIC_VECTOR (102 downto 0);
    signal cstArcTan2Mi_10_uid200_sincosTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal xip1E_11_uid202_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_11_uid202_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_11_uid202_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_11_uid202_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_11_uid202_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_11_uid203_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_11_uid203_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_11_uid203_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_11_uid203_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_11_uid203_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_11_uid205_sincosTest_a : STD_LOGIC_VECTOR (59 downto 0);
    signal aip1E_11_uid205_sincosTest_b : STD_LOGIC_VECTOR (59 downto 0);
    signal aip1E_11_uid205_sincosTest_o : STD_LOGIC_VECTOR (59 downto 0);
    signal aip1E_11_uid205_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_11_uid205_sincosTest_q : STD_LOGIC_VECTOR (58 downto 0);
    signal xip1_11_uid206_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_11_uid206_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_11_uid207_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_11_uid207_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid208_sincosTest_in : STD_LOGIC_VECTOR (56 downto 0);
    signal aip1E_uid208_sincosTest_b : STD_LOGIC_VECTOR (56 downto 0);
    signal xMSB_uid209_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid211_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid214_sincosTest_b : STD_LOGIC_VECTOR (101 downto 0);
    signal twoToMiSiYip_uid215_sincosTest_b : STD_LOGIC_VECTOR (101 downto 0);
    signal cstArcTan2Mi_11_uid216_sincosTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal xip1E_12_uid218_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_12_uid218_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_12_uid218_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_12_uid218_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_12_uid218_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_12_uid219_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_12_uid219_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_12_uid219_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_12_uid219_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_12_uid219_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_12_uid221_sincosTest_a : STD_LOGIC_VECTOR (58 downto 0);
    signal aip1E_12_uid221_sincosTest_b : STD_LOGIC_VECTOR (58 downto 0);
    signal aip1E_12_uid221_sincosTest_o : STD_LOGIC_VECTOR (58 downto 0);
    signal aip1E_12_uid221_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_12_uid221_sincosTest_q : STD_LOGIC_VECTOR (57 downto 0);
    signal xip1_12_uid222_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_12_uid222_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_12_uid223_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_12_uid223_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid224_sincosTest_in : STD_LOGIC_VECTOR (55 downto 0);
    signal aip1E_uid224_sincosTest_b : STD_LOGIC_VECTOR (55 downto 0);
    signal xMSB_uid225_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid227_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid230_sincosTest_b : STD_LOGIC_VECTOR (100 downto 0);
    signal twoToMiSiYip_uid231_sincosTest_b : STD_LOGIC_VECTOR (100 downto 0);
    signal cstArcTan2Mi_12_uid232_sincosTest_q : STD_LOGIC_VECTOR (53 downto 0);
    signal xip1E_13_uid234_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_13_uid234_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_13_uid234_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_13_uid234_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_13_uid234_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_13_uid235_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_13_uid235_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_13_uid235_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_13_uid235_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_13_uid235_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_13_uid237_sincosTest_a : STD_LOGIC_VECTOR (57 downto 0);
    signal aip1E_13_uid237_sincosTest_b : STD_LOGIC_VECTOR (57 downto 0);
    signal aip1E_13_uid237_sincosTest_o : STD_LOGIC_VECTOR (57 downto 0);
    signal aip1E_13_uid237_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_13_uid237_sincosTest_q : STD_LOGIC_VECTOR (56 downto 0);
    signal xip1_13_uid238_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_13_uid238_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_13_uid239_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_13_uid239_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid240_sincosTest_in : STD_LOGIC_VECTOR (54 downto 0);
    signal aip1E_uid240_sincosTest_b : STD_LOGIC_VECTOR (54 downto 0);
    signal xMSB_uid241_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid243_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid246_sincosTest_b : STD_LOGIC_VECTOR (99 downto 0);
    signal twoToMiSiYip_uid247_sincosTest_b : STD_LOGIC_VECTOR (99 downto 0);
    signal cstArcTan2Mi_13_uid248_sincosTest_q : STD_LOGIC_VECTOR (52 downto 0);
    signal xip1E_14_uid250_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_14_uid250_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_14_uid250_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_14_uid250_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_14_uid250_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_14_uid251_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_14_uid251_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_14_uid251_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_14_uid251_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_14_uid251_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_14_uid253_sincosTest_a : STD_LOGIC_VECTOR (56 downto 0);
    signal aip1E_14_uid253_sincosTest_b : STD_LOGIC_VECTOR (56 downto 0);
    signal aip1E_14_uid253_sincosTest_o : STD_LOGIC_VECTOR (56 downto 0);
    signal aip1E_14_uid253_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_14_uid253_sincosTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal xip1_14_uid254_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_14_uid254_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_14_uid255_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_14_uid255_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid256_sincosTest_in : STD_LOGIC_VECTOR (53 downto 0);
    signal aip1E_uid256_sincosTest_b : STD_LOGIC_VECTOR (53 downto 0);
    signal xMSB_uid257_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid259_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid262_sincosTest_b : STD_LOGIC_VECTOR (98 downto 0);
    signal twoToMiSiYip_uid263_sincosTest_b : STD_LOGIC_VECTOR (98 downto 0);
    signal cstArcTan2Mi_14_uid264_sincosTest_q : STD_LOGIC_VECTOR (51 downto 0);
    signal xip1E_15_uid266_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_15_uid266_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_15_uid266_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_15_uid266_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_15_uid266_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_15_uid267_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_15_uid267_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_15_uid267_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_15_uid267_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_15_uid267_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_15_uid269_sincosTest_a : STD_LOGIC_VECTOR (55 downto 0);
    signal aip1E_15_uid269_sincosTest_b : STD_LOGIC_VECTOR (55 downto 0);
    signal aip1E_15_uid269_sincosTest_o : STD_LOGIC_VECTOR (55 downto 0);
    signal aip1E_15_uid269_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_15_uid269_sincosTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal xip1_15_uid270_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_15_uid270_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_15_uid271_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_15_uid271_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid272_sincosTest_in : STD_LOGIC_VECTOR (52 downto 0);
    signal aip1E_uid272_sincosTest_b : STD_LOGIC_VECTOR (52 downto 0);
    signal xMSB_uid273_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid275_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid278_sincosTest_b : STD_LOGIC_VECTOR (97 downto 0);
    signal twoToMiSiYip_uid279_sincosTest_b : STD_LOGIC_VECTOR (97 downto 0);
    signal cstArcTan2Mi_15_uid280_sincosTest_q : STD_LOGIC_VECTOR (50 downto 0);
    signal xip1E_16_uid282_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_16_uid282_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_16_uid282_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_16_uid282_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_16_uid282_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_16_uid283_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_16_uid283_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_16_uid283_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_16_uid283_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_16_uid283_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_16_uid285_sincosTest_a : STD_LOGIC_VECTOR (54 downto 0);
    signal aip1E_16_uid285_sincosTest_b : STD_LOGIC_VECTOR (54 downto 0);
    signal aip1E_16_uid285_sincosTest_o : STD_LOGIC_VECTOR (54 downto 0);
    signal aip1E_16_uid285_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_16_uid285_sincosTest_q : STD_LOGIC_VECTOR (53 downto 0);
    signal xip1_16_uid286_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_16_uid286_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_16_uid287_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_16_uid287_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid288_sincosTest_in : STD_LOGIC_VECTOR (51 downto 0);
    signal aip1E_uid288_sincosTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal xMSB_uid289_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid291_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid294_sincosTest_b : STD_LOGIC_VECTOR (96 downto 0);
    signal twoToMiSiYip_uid295_sincosTest_b : STD_LOGIC_VECTOR (96 downto 0);
    signal cstArcTan2Mi_16_uid296_sincosTest_q : STD_LOGIC_VECTOR (49 downto 0);
    signal xip1E_17_uid298_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_17_uid298_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_17_uid298_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_17_uid298_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_17_uid298_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_17_uid299_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_17_uid299_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_17_uid299_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_17_uid299_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_17_uid299_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_17_uid301_sincosTest_a : STD_LOGIC_VECTOR (53 downto 0);
    signal aip1E_17_uid301_sincosTest_b : STD_LOGIC_VECTOR (53 downto 0);
    signal aip1E_17_uid301_sincosTest_o : STD_LOGIC_VECTOR (53 downto 0);
    signal aip1E_17_uid301_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_17_uid301_sincosTest_q : STD_LOGIC_VECTOR (52 downto 0);
    signal xip1_17_uid302_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_17_uid302_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_17_uid303_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_17_uid303_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid304_sincosTest_in : STD_LOGIC_VECTOR (50 downto 0);
    signal aip1E_uid304_sincosTest_b : STD_LOGIC_VECTOR (50 downto 0);
    signal xMSB_uid305_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid307_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid310_sincosTest_b : STD_LOGIC_VECTOR (95 downto 0);
    signal twoToMiSiYip_uid311_sincosTest_b : STD_LOGIC_VECTOR (95 downto 0);
    signal cstArcTan2Mi_17_uid312_sincosTest_q : STD_LOGIC_VECTOR (48 downto 0);
    signal xip1E_18_uid314_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_18_uid314_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_18_uid314_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_18_uid314_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_18_uid314_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_18_uid315_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_18_uid315_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_18_uid315_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_18_uid315_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_18_uid315_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_18_uid317_sincosTest_a : STD_LOGIC_VECTOR (52 downto 0);
    signal aip1E_18_uid317_sincosTest_b : STD_LOGIC_VECTOR (52 downto 0);
    signal aip1E_18_uid317_sincosTest_o : STD_LOGIC_VECTOR (52 downto 0);
    signal aip1E_18_uid317_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_18_uid317_sincosTest_q : STD_LOGIC_VECTOR (51 downto 0);
    signal xip1_18_uid318_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_18_uid318_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_18_uid319_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_18_uid319_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid320_sincosTest_in : STD_LOGIC_VECTOR (49 downto 0);
    signal aip1E_uid320_sincosTest_b : STD_LOGIC_VECTOR (49 downto 0);
    signal xMSB_uid321_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid323_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid326_sincosTest_b : STD_LOGIC_VECTOR (94 downto 0);
    signal twoToMiSiYip_uid327_sincosTest_b : STD_LOGIC_VECTOR (94 downto 0);
    signal cstArcTan2Mi_18_uid328_sincosTest_q : STD_LOGIC_VECTOR (47 downto 0);
    signal xip1E_19_uid330_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_19_uid330_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_19_uid330_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_19_uid330_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_19_uid330_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_19_uid331_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_19_uid331_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_19_uid331_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_19_uid331_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_19_uid331_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_19_uid333_sincosTest_a : STD_LOGIC_VECTOR (51 downto 0);
    signal aip1E_19_uid333_sincosTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal aip1E_19_uid333_sincosTest_o : STD_LOGIC_VECTOR (51 downto 0);
    signal aip1E_19_uid333_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_19_uid333_sincosTest_q : STD_LOGIC_VECTOR (50 downto 0);
    signal xip1_19_uid334_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_19_uid334_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_19_uid335_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_19_uid335_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid336_sincosTest_in : STD_LOGIC_VECTOR (48 downto 0);
    signal aip1E_uid336_sincosTest_b : STD_LOGIC_VECTOR (48 downto 0);
    signal xMSB_uid337_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid339_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid342_sincosTest_b : STD_LOGIC_VECTOR (93 downto 0);
    signal twoToMiSiYip_uid343_sincosTest_b : STD_LOGIC_VECTOR (93 downto 0);
    signal cstArcTan2Mi_19_uid344_sincosTest_q : STD_LOGIC_VECTOR (46 downto 0);
    signal xip1E_20_uid346_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_20_uid346_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_20_uid346_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_20_uid346_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_20_uid346_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_20_uid347_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_20_uid347_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_20_uid347_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_20_uid347_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_20_uid347_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_20_uid349_sincosTest_a : STD_LOGIC_VECTOR (50 downto 0);
    signal aip1E_20_uid349_sincosTest_b : STD_LOGIC_VECTOR (50 downto 0);
    signal aip1E_20_uid349_sincosTest_o : STD_LOGIC_VECTOR (50 downto 0);
    signal aip1E_20_uid349_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_20_uid349_sincosTest_q : STD_LOGIC_VECTOR (49 downto 0);
    signal xip1_20_uid350_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_20_uid350_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_20_uid351_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_20_uid351_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid352_sincosTest_in : STD_LOGIC_VECTOR (47 downto 0);
    signal aip1E_uid352_sincosTest_b : STD_LOGIC_VECTOR (47 downto 0);
    signal xMSB_uid353_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid355_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid358_sincosTest_b : STD_LOGIC_VECTOR (92 downto 0);
    signal twoToMiSiYip_uid359_sincosTest_b : STD_LOGIC_VECTOR (92 downto 0);
    signal cstArcTan2Mi_20_uid360_sincosTest_q : STD_LOGIC_VECTOR (45 downto 0);
    signal xip1E_21_uid362_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_21_uid362_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_21_uid362_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_21_uid362_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_21_uid362_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_21_uid363_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_21_uid363_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_21_uid363_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_21_uid363_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_21_uid363_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_21_uid365_sincosTest_a : STD_LOGIC_VECTOR (49 downto 0);
    signal aip1E_21_uid365_sincosTest_b : STD_LOGIC_VECTOR (49 downto 0);
    signal aip1E_21_uid365_sincosTest_o : STD_LOGIC_VECTOR (49 downto 0);
    signal aip1E_21_uid365_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_21_uid365_sincosTest_q : STD_LOGIC_VECTOR (48 downto 0);
    signal xip1_21_uid366_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_21_uid366_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_21_uid367_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_21_uid367_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid368_sincosTest_in : STD_LOGIC_VECTOR (46 downto 0);
    signal aip1E_uid368_sincosTest_b : STD_LOGIC_VECTOR (46 downto 0);
    signal xMSB_uid369_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid371_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid374_sincosTest_b : STD_LOGIC_VECTOR (91 downto 0);
    signal twoToMiSiYip_uid375_sincosTest_b : STD_LOGIC_VECTOR (91 downto 0);
    signal cstArcTan2Mi_21_uid376_sincosTest_q : STD_LOGIC_VECTOR (44 downto 0);
    signal xip1E_22_uid378_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_22_uid378_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_22_uid378_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_22_uid378_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_22_uid378_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_22_uid379_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_22_uid379_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_22_uid379_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_22_uid379_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_22_uid379_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal aip1E_22_uid381_sincosTest_a : STD_LOGIC_VECTOR (48 downto 0);
    signal aip1E_22_uid381_sincosTest_b : STD_LOGIC_VECTOR (48 downto 0);
    signal aip1E_22_uid381_sincosTest_o : STD_LOGIC_VECTOR (48 downto 0);
    signal aip1E_22_uid381_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_22_uid381_sincosTest_q : STD_LOGIC_VECTOR (47 downto 0);
    signal xip1_22_uid382_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_22_uid382_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_22_uid383_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_22_uid383_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid384_sincosTest_in : STD_LOGIC_VECTOR (45 downto 0);
    signal aip1E_uid384_sincosTest_b : STD_LOGIC_VECTOR (45 downto 0);
    signal xMSB_uid385_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid387_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid390_sincosTest_b : STD_LOGIC_VECTOR (90 downto 0);
    signal twoToMiSiYip_uid391_sincosTest_b : STD_LOGIC_VECTOR (90 downto 0);
    signal cstArcTan2Mi_22_uid392_sincosTest_q : STD_LOGIC_VECTOR (43 downto 0);
    signal xip1E_23_uid394_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_23_uid394_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_23_uid394_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_23_uid394_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_23_uid394_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_23_uid395_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_23_uid395_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_23_uid395_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_23_uid395_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_23_uid395_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid397_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid397_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid398_sincosTest_b : STD_LOGIC_VECTOR (44 downto 0);
    signal aip1E_23high_uid399_sincosTest_a : STD_LOGIC_VECTOR (46 downto 0);
    signal aip1E_23high_uid399_sincosTest_b : STD_LOGIC_VECTOR (46 downto 0);
    signal aip1E_23high_uid399_sincosTest_o : STD_LOGIC_VECTOR (46 downto 0);
    signal aip1E_23high_uid399_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_23high_uid399_sincosTest_q : STD_LOGIC_VECTOR (45 downto 0);
    signal aip1E_23_uid400_sincosTest_q : STD_LOGIC_VECTOR (46 downto 0);
    signal xip1_23_uid401_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_23_uid401_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_23_uid402_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_23_uid402_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid403_sincosTest_in : STD_LOGIC_VECTOR (44 downto 0);
    signal aip1E_uid403_sincosTest_b : STD_LOGIC_VECTOR (44 downto 0);
    signal xMSB_uid404_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid406_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid409_sincosTest_b : STD_LOGIC_VECTOR (89 downto 0);
    signal twoToMiSiYip_uid410_sincosTest_b : STD_LOGIC_VECTOR (89 downto 0);
    signal cstArcTan2Mi_23_uid411_sincosTest_q : STD_LOGIC_VECTOR (42 downto 0);
    signal xip1E_24_uid413_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_24_uid413_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_24_uid413_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_24_uid413_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_24_uid413_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_24_uid414_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_24_uid414_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_24_uid414_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_24_uid414_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_24_uid414_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid416_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid416_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid417_sincosTest_b : STD_LOGIC_VECTOR (43 downto 0);
    signal aip1E_24high_uid418_sincosTest_a : STD_LOGIC_VECTOR (45 downto 0);
    signal aip1E_24high_uid418_sincosTest_b : STD_LOGIC_VECTOR (45 downto 0);
    signal aip1E_24high_uid418_sincosTest_o : STD_LOGIC_VECTOR (45 downto 0);
    signal aip1E_24high_uid418_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_24high_uid418_sincosTest_q : STD_LOGIC_VECTOR (44 downto 0);
    signal aip1E_24_uid419_sincosTest_q : STD_LOGIC_VECTOR (45 downto 0);
    signal xip1_24_uid420_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_24_uid420_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_24_uid421_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_24_uid421_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid422_sincosTest_in : STD_LOGIC_VECTOR (43 downto 0);
    signal aip1E_uid422_sincosTest_b : STD_LOGIC_VECTOR (43 downto 0);
    signal xMSB_uid423_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid425_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid428_sincosTest_b : STD_LOGIC_VECTOR (88 downto 0);
    signal twoToMiSiYip_uid429_sincosTest_b : STD_LOGIC_VECTOR (88 downto 0);
    signal cstArcTan2Mi_24_uid430_sincosTest_q : STD_LOGIC_VECTOR (41 downto 0);
    signal xip1E_25_uid432_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_25_uid432_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_25_uid432_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_25_uid432_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_25_uid432_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_25_uid433_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_25_uid433_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_25_uid433_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_25_uid433_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_25_uid433_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid435_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid435_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid436_sincosTest_b : STD_LOGIC_VECTOR (42 downto 0);
    signal aip1E_25high_uid437_sincosTest_a : STD_LOGIC_VECTOR (44 downto 0);
    signal aip1E_25high_uid437_sincosTest_b : STD_LOGIC_VECTOR (44 downto 0);
    signal aip1E_25high_uid437_sincosTest_o : STD_LOGIC_VECTOR (44 downto 0);
    signal aip1E_25high_uid437_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_25high_uid437_sincosTest_q : STD_LOGIC_VECTOR (43 downto 0);
    signal aip1E_25_uid438_sincosTest_q : STD_LOGIC_VECTOR (44 downto 0);
    signal xip1_25_uid439_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_25_uid439_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_25_uid440_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_25_uid440_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid441_sincosTest_in : STD_LOGIC_VECTOR (42 downto 0);
    signal aip1E_uid441_sincosTest_b : STD_LOGIC_VECTOR (42 downto 0);
    signal xMSB_uid442_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid444_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid447_sincosTest_b : STD_LOGIC_VECTOR (87 downto 0);
    signal twoToMiSiYip_uid448_sincosTest_b : STD_LOGIC_VECTOR (87 downto 0);
    signal cstArcTan2Mi_25_uid449_sincosTest_q : STD_LOGIC_VECTOR (40 downto 0);
    signal xip1E_26_uid451_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_26_uid451_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_26_uid451_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_26_uid451_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_26_uid451_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_26_uid452_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_26_uid452_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_26_uid452_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_26_uid452_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_26_uid452_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid454_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid454_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid455_sincosTest_b : STD_LOGIC_VECTOR (41 downto 0);
    signal aip1E_26high_uid456_sincosTest_a : STD_LOGIC_VECTOR (43 downto 0);
    signal aip1E_26high_uid456_sincosTest_b : STD_LOGIC_VECTOR (43 downto 0);
    signal aip1E_26high_uid456_sincosTest_o : STD_LOGIC_VECTOR (43 downto 0);
    signal aip1E_26high_uid456_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_26high_uid456_sincosTest_q : STD_LOGIC_VECTOR (42 downto 0);
    signal aip1E_26_uid457_sincosTest_q : STD_LOGIC_VECTOR (43 downto 0);
    signal xip1_26_uid458_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_26_uid458_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_26_uid459_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_26_uid459_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid460_sincosTest_in : STD_LOGIC_VECTOR (41 downto 0);
    signal aip1E_uid460_sincosTest_b : STD_LOGIC_VECTOR (41 downto 0);
    signal xMSB_uid461_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid463_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid466_sincosTest_b : STD_LOGIC_VECTOR (86 downto 0);
    signal twoToMiSiYip_uid467_sincosTest_b : STD_LOGIC_VECTOR (86 downto 0);
    signal cstArcTan2Mi_26_uid468_sincosTest_q : STD_LOGIC_VECTOR (39 downto 0);
    signal xip1E_27_uid470_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_27_uid470_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_27_uid470_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_27_uid470_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_27_uid470_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_27_uid471_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_27_uid471_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_27_uid471_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_27_uid471_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_27_uid471_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid473_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid473_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid474_sincosTest_b : STD_LOGIC_VECTOR (40 downto 0);
    signal aip1E_27high_uid475_sincosTest_a : STD_LOGIC_VECTOR (42 downto 0);
    signal aip1E_27high_uid475_sincosTest_b : STD_LOGIC_VECTOR (42 downto 0);
    signal aip1E_27high_uid475_sincosTest_o : STD_LOGIC_VECTOR (42 downto 0);
    signal aip1E_27high_uid475_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_27high_uid475_sincosTest_q : STD_LOGIC_VECTOR (41 downto 0);
    signal aip1E_27_uid476_sincosTest_q : STD_LOGIC_VECTOR (42 downto 0);
    signal xip1_27_uid477_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_27_uid477_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_27_uid478_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_27_uid478_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid479_sincosTest_in : STD_LOGIC_VECTOR (40 downto 0);
    signal aip1E_uid479_sincosTest_b : STD_LOGIC_VECTOR (40 downto 0);
    signal xMSB_uid480_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid482_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid485_sincosTest_b : STD_LOGIC_VECTOR (85 downto 0);
    signal twoToMiSiYip_uid486_sincosTest_b : STD_LOGIC_VECTOR (85 downto 0);
    signal cstArcTan2Mi_27_uid487_sincosTest_q : STD_LOGIC_VECTOR (38 downto 0);
    signal xip1E_28_uid489_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_28_uid489_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_28_uid489_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_28_uid489_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_28_uid489_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_28_uid490_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_28_uid490_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_28_uid490_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_28_uid490_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_28_uid490_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid492_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid492_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid493_sincosTest_b : STD_LOGIC_VECTOR (39 downto 0);
    signal aip1E_28high_uid494_sincosTest_a : STD_LOGIC_VECTOR (41 downto 0);
    signal aip1E_28high_uid494_sincosTest_b : STD_LOGIC_VECTOR (41 downto 0);
    signal aip1E_28high_uid494_sincosTest_o : STD_LOGIC_VECTOR (41 downto 0);
    signal aip1E_28high_uid494_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_28high_uid494_sincosTest_q : STD_LOGIC_VECTOR (40 downto 0);
    signal aip1E_28_uid495_sincosTest_q : STD_LOGIC_VECTOR (41 downto 0);
    signal xip1_28_uid496_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_28_uid496_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_28_uid497_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_28_uid497_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid498_sincosTest_in : STD_LOGIC_VECTOR (39 downto 0);
    signal aip1E_uid498_sincosTest_b : STD_LOGIC_VECTOR (39 downto 0);
    signal xMSB_uid499_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid501_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid504_sincosTest_b : STD_LOGIC_VECTOR (84 downto 0);
    signal twoToMiSiYip_uid505_sincosTest_b : STD_LOGIC_VECTOR (84 downto 0);
    signal cstArcTan2Mi_28_uid506_sincosTest_q : STD_LOGIC_VECTOR (37 downto 0);
    signal xip1E_29_uid508_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_29_uid508_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_29_uid508_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_29_uid508_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_29_uid508_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_29_uid509_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_29_uid509_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_29_uid509_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_29_uid509_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_29_uid509_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid511_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid511_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid512_sincosTest_b : STD_LOGIC_VECTOR (38 downto 0);
    signal aip1E_29high_uid513_sincosTest_a : STD_LOGIC_VECTOR (40 downto 0);
    signal aip1E_29high_uid513_sincosTest_b : STD_LOGIC_VECTOR (40 downto 0);
    signal aip1E_29high_uid513_sincosTest_o : STD_LOGIC_VECTOR (40 downto 0);
    signal aip1E_29high_uid513_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_29high_uid513_sincosTest_q : STD_LOGIC_VECTOR (39 downto 0);
    signal aip1E_29_uid514_sincosTest_q : STD_LOGIC_VECTOR (40 downto 0);
    signal xip1_29_uid515_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_29_uid515_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_29_uid516_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_29_uid516_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid517_sincosTest_in : STD_LOGIC_VECTOR (38 downto 0);
    signal aip1E_uid517_sincosTest_b : STD_LOGIC_VECTOR (38 downto 0);
    signal xMSB_uid518_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid520_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid523_sincosTest_b : STD_LOGIC_VECTOR (83 downto 0);
    signal twoToMiSiYip_uid524_sincosTest_b : STD_LOGIC_VECTOR (83 downto 0);
    signal cstArcTan2Mi_29_uid525_sincosTest_q : STD_LOGIC_VECTOR (36 downto 0);
    signal xip1E_30_uid527_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_30_uid527_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_30_uid527_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_30_uid527_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_30_uid527_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_30_uid528_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_30_uid528_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_30_uid528_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_30_uid528_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_30_uid528_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid530_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid530_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid531_sincosTest_b : STD_LOGIC_VECTOR (37 downto 0);
    signal aip1E_30high_uid532_sincosTest_a : STD_LOGIC_VECTOR (39 downto 0);
    signal aip1E_30high_uid532_sincosTest_b : STD_LOGIC_VECTOR (39 downto 0);
    signal aip1E_30high_uid532_sincosTest_o : STD_LOGIC_VECTOR (39 downto 0);
    signal aip1E_30high_uid532_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_30high_uid532_sincosTest_q : STD_LOGIC_VECTOR (38 downto 0);
    signal aip1E_30_uid533_sincosTest_q : STD_LOGIC_VECTOR (39 downto 0);
    signal xip1_30_uid534_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_30_uid534_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_30_uid535_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_30_uid535_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid536_sincosTest_in : STD_LOGIC_VECTOR (37 downto 0);
    signal aip1E_uid536_sincosTest_b : STD_LOGIC_VECTOR (37 downto 0);
    signal xMSB_uid537_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid539_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid542_sincosTest_b : STD_LOGIC_VECTOR (82 downto 0);
    signal twoToMiSiYip_uid543_sincosTest_b : STD_LOGIC_VECTOR (82 downto 0);
    signal cstArcTan2Mi_30_uid544_sincosTest_q : STD_LOGIC_VECTOR (35 downto 0);
    signal xip1E_31_uid546_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_31_uid546_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_31_uid546_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_31_uid546_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_31_uid546_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_31_uid547_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_31_uid547_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_31_uid547_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_31_uid547_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_31_uid547_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid549_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid549_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid550_sincosTest_b : STD_LOGIC_VECTOR (36 downto 0);
    signal aip1E_31high_uid551_sincosTest_a : STD_LOGIC_VECTOR (38 downto 0);
    signal aip1E_31high_uid551_sincosTest_b : STD_LOGIC_VECTOR (38 downto 0);
    signal aip1E_31high_uid551_sincosTest_o : STD_LOGIC_VECTOR (38 downto 0);
    signal aip1E_31high_uid551_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_31high_uid551_sincosTest_q : STD_LOGIC_VECTOR (37 downto 0);
    signal aip1E_31_uid552_sincosTest_q : STD_LOGIC_VECTOR (38 downto 0);
    signal xip1_31_uid553_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_31_uid553_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_31_uid554_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_31_uid554_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid555_sincosTest_in : STD_LOGIC_VECTOR (36 downto 0);
    signal aip1E_uid555_sincosTest_b : STD_LOGIC_VECTOR (36 downto 0);
    signal xMSB_uid556_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid558_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid561_sincosTest_b : STD_LOGIC_VECTOR (81 downto 0);
    signal twoToMiSiYip_uid562_sincosTest_b : STD_LOGIC_VECTOR (81 downto 0);
    signal cstArcTan2Mi_31_uid563_sincosTest_q : STD_LOGIC_VECTOR (34 downto 0);
    signal xip1E_32_uid565_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_32_uid565_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_32_uid565_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_32_uid565_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_32_uid565_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_32_uid566_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_32_uid566_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_32_uid566_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_32_uid566_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_32_uid566_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid568_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid568_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid569_sincosTest_b : STD_LOGIC_VECTOR (35 downto 0);
    signal aip1E_32high_uid570_sincosTest_a : STD_LOGIC_VECTOR (37 downto 0);
    signal aip1E_32high_uid570_sincosTest_b : STD_LOGIC_VECTOR (37 downto 0);
    signal aip1E_32high_uid570_sincosTest_o : STD_LOGIC_VECTOR (37 downto 0);
    signal aip1E_32high_uid570_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_32high_uid570_sincosTest_q : STD_LOGIC_VECTOR (36 downto 0);
    signal aip1E_32_uid571_sincosTest_q : STD_LOGIC_VECTOR (37 downto 0);
    signal xip1_32_uid572_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_32_uid572_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_32_uid573_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_32_uid573_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid574_sincosTest_in : STD_LOGIC_VECTOR (35 downto 0);
    signal aip1E_uid574_sincosTest_b : STD_LOGIC_VECTOR (35 downto 0);
    signal xMSB_uid575_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid577_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid580_sincosTest_b : STD_LOGIC_VECTOR (80 downto 0);
    signal twoToMiSiYip_uid581_sincosTest_b : STD_LOGIC_VECTOR (80 downto 0);
    signal cstArcTan2Mi_32_uid582_sincosTest_q : STD_LOGIC_VECTOR (33 downto 0);
    signal xip1E_33_uid584_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_33_uid584_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_33_uid584_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_33_uid584_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_33_uid584_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_33_uid585_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_33_uid585_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_33_uid585_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_33_uid585_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_33_uid585_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid587_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid587_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid588_sincosTest_b : STD_LOGIC_VECTOR (34 downto 0);
    signal aip1E_33high_uid589_sincosTest_a : STD_LOGIC_VECTOR (36 downto 0);
    signal aip1E_33high_uid589_sincosTest_b : STD_LOGIC_VECTOR (36 downto 0);
    signal aip1E_33high_uid589_sincosTest_o : STD_LOGIC_VECTOR (36 downto 0);
    signal aip1E_33high_uid589_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_33high_uid589_sincosTest_q : STD_LOGIC_VECTOR (35 downto 0);
    signal aip1E_33_uid590_sincosTest_q : STD_LOGIC_VECTOR (36 downto 0);
    signal xip1_33_uid591_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_33_uid591_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_33_uid592_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_33_uid592_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid593_sincosTest_in : STD_LOGIC_VECTOR (34 downto 0);
    signal aip1E_uid593_sincosTest_b : STD_LOGIC_VECTOR (34 downto 0);
    signal xMSB_uid594_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid596_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid599_sincosTest_b : STD_LOGIC_VECTOR (79 downto 0);
    signal twoToMiSiYip_uid600_sincosTest_b : STD_LOGIC_VECTOR (79 downto 0);
    signal cstArcTan2Mi_33_uid601_sincosTest_q : STD_LOGIC_VECTOR (32 downto 0);
    signal xip1E_34_uid603_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_34_uid603_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_34_uid603_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_34_uid603_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_34_uid603_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_34_uid604_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_34_uid604_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_34_uid604_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_34_uid604_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_34_uid604_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid606_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid606_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid607_sincosTest_b : STD_LOGIC_VECTOR (33 downto 0);
    signal aip1E_34high_uid608_sincosTest_a : STD_LOGIC_VECTOR (35 downto 0);
    signal aip1E_34high_uid608_sincosTest_b : STD_LOGIC_VECTOR (35 downto 0);
    signal aip1E_34high_uid608_sincosTest_o : STD_LOGIC_VECTOR (35 downto 0);
    signal aip1E_34high_uid608_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_34high_uid608_sincosTest_q : STD_LOGIC_VECTOR (34 downto 0);
    signal aip1E_34_uid609_sincosTest_q : STD_LOGIC_VECTOR (35 downto 0);
    signal xip1_34_uid610_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_34_uid610_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_34_uid611_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_34_uid611_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid612_sincosTest_in : STD_LOGIC_VECTOR (33 downto 0);
    signal aip1E_uid612_sincosTest_b : STD_LOGIC_VECTOR (33 downto 0);
    signal xMSB_uid613_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid615_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid618_sincosTest_b : STD_LOGIC_VECTOR (78 downto 0);
    signal twoToMiSiYip_uid619_sincosTest_b : STD_LOGIC_VECTOR (78 downto 0);
    signal cstArcTan2Mi_34_uid620_sincosTest_q : STD_LOGIC_VECTOR (31 downto 0);
    signal xip1E_35_uid622_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_35_uid622_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_35_uid622_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_35_uid622_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_35_uid622_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_35_uid623_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_35_uid623_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_35_uid623_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_35_uid623_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_35_uid623_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid625_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid625_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid626_sincosTest_b : STD_LOGIC_VECTOR (32 downto 0);
    signal aip1E_35high_uid627_sincosTest_a : STD_LOGIC_VECTOR (34 downto 0);
    signal aip1E_35high_uid627_sincosTest_b : STD_LOGIC_VECTOR (34 downto 0);
    signal aip1E_35high_uid627_sincosTest_o : STD_LOGIC_VECTOR (34 downto 0);
    signal aip1E_35high_uid627_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_35high_uid627_sincosTest_q : STD_LOGIC_VECTOR (33 downto 0);
    signal aip1E_35_uid628_sincosTest_q : STD_LOGIC_VECTOR (34 downto 0);
    signal xip1_35_uid629_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_35_uid629_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_35_uid630_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_35_uid630_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid631_sincosTest_in : STD_LOGIC_VECTOR (32 downto 0);
    signal aip1E_uid631_sincosTest_b : STD_LOGIC_VECTOR (32 downto 0);
    signal xMSB_uid632_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid634_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid637_sincosTest_b : STD_LOGIC_VECTOR (77 downto 0);
    signal twoToMiSiYip_uid638_sincosTest_b : STD_LOGIC_VECTOR (77 downto 0);
    signal cstArcTan2Mi_35_uid639_sincosTest_q : STD_LOGIC_VECTOR (30 downto 0);
    signal xip1E_36_uid641_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_36_uid641_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_36_uid641_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_36_uid641_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_36_uid641_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_36_uid642_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_36_uid642_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_36_uid642_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_36_uid642_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_36_uid642_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid644_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid644_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid645_sincosTest_b : STD_LOGIC_VECTOR (31 downto 0);
    signal aip1E_36high_uid646_sincosTest_a : STD_LOGIC_VECTOR (33 downto 0);
    signal aip1E_36high_uid646_sincosTest_b : STD_LOGIC_VECTOR (33 downto 0);
    signal aip1E_36high_uid646_sincosTest_o : STD_LOGIC_VECTOR (33 downto 0);
    signal aip1E_36high_uid646_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_36high_uid646_sincosTest_q : STD_LOGIC_VECTOR (32 downto 0);
    signal aip1E_36_uid647_sincosTest_q : STD_LOGIC_VECTOR (33 downto 0);
    signal xip1_36_uid648_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_36_uid648_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_36_uid649_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_36_uid649_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid650_sincosTest_in : STD_LOGIC_VECTOR (31 downto 0);
    signal aip1E_uid650_sincosTest_b : STD_LOGIC_VECTOR (31 downto 0);
    signal xMSB_uid651_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid653_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid656_sincosTest_b : STD_LOGIC_VECTOR (76 downto 0);
    signal twoToMiSiYip_uid657_sincosTest_b : STD_LOGIC_VECTOR (76 downto 0);
    signal cstArcTan2Mi_36_uid658_sincosTest_q : STD_LOGIC_VECTOR (29 downto 0);
    signal xip1E_37_uid660_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_37_uid660_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_37_uid660_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_37_uid660_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_37_uid660_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_37_uid661_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_37_uid661_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_37_uid661_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_37_uid661_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_37_uid661_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid663_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid663_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid664_sincosTest_b : STD_LOGIC_VECTOR (30 downto 0);
    signal aip1E_37high_uid665_sincosTest_a : STD_LOGIC_VECTOR (32 downto 0);
    signal aip1E_37high_uid665_sincosTest_b : STD_LOGIC_VECTOR (32 downto 0);
    signal aip1E_37high_uid665_sincosTest_o : STD_LOGIC_VECTOR (32 downto 0);
    signal aip1E_37high_uid665_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_37high_uid665_sincosTest_q : STD_LOGIC_VECTOR (31 downto 0);
    signal aip1E_37_uid666_sincosTest_q : STD_LOGIC_VECTOR (32 downto 0);
    signal xip1_37_uid667_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_37_uid667_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_37_uid668_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_37_uid668_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid669_sincosTest_in : STD_LOGIC_VECTOR (30 downto 0);
    signal aip1E_uid669_sincosTest_b : STD_LOGIC_VECTOR (30 downto 0);
    signal xMSB_uid670_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid672_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid675_sincosTest_b : STD_LOGIC_VECTOR (75 downto 0);
    signal twoToMiSiYip_uid676_sincosTest_b : STD_LOGIC_VECTOR (75 downto 0);
    signal cstArcTan2Mi_37_uid677_sincosTest_q : STD_LOGIC_VECTOR (28 downto 0);
    signal xip1E_38_uid679_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_38_uid679_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_38_uid679_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_38_uid679_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_38_uid679_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_38_uid680_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_38_uid680_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_38_uid680_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_38_uid680_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_38_uid680_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid682_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid682_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid683_sincosTest_b : STD_LOGIC_VECTOR (29 downto 0);
    signal aip1E_38high_uid684_sincosTest_a : STD_LOGIC_VECTOR (31 downto 0);
    signal aip1E_38high_uid684_sincosTest_b : STD_LOGIC_VECTOR (31 downto 0);
    signal aip1E_38high_uid684_sincosTest_o : STD_LOGIC_VECTOR (31 downto 0);
    signal aip1E_38high_uid684_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_38high_uid684_sincosTest_q : STD_LOGIC_VECTOR (30 downto 0);
    signal aip1E_38_uid685_sincosTest_q : STD_LOGIC_VECTOR (31 downto 0);
    signal xip1_38_uid686_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_38_uid686_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_38_uid687_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_38_uid687_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid688_sincosTest_in : STD_LOGIC_VECTOR (29 downto 0);
    signal aip1E_uid688_sincosTest_b : STD_LOGIC_VECTOR (29 downto 0);
    signal xMSB_uid689_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid691_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid694_sincosTest_b : STD_LOGIC_VECTOR (74 downto 0);
    signal twoToMiSiYip_uid695_sincosTest_b : STD_LOGIC_VECTOR (74 downto 0);
    signal cstArcTan2Mi_38_uid696_sincosTest_q : STD_LOGIC_VECTOR (27 downto 0);
    signal xip1E_39_uid698_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_39_uid698_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_39_uid698_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_39_uid698_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_39_uid698_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_39_uid699_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_39_uid699_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_39_uid699_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_39_uid699_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_39_uid699_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid701_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid701_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid702_sincosTest_b : STD_LOGIC_VECTOR (28 downto 0);
    signal aip1E_39high_uid703_sincosTest_a : STD_LOGIC_VECTOR (30 downto 0);
    signal aip1E_39high_uid703_sincosTest_b : STD_LOGIC_VECTOR (30 downto 0);
    signal aip1E_39high_uid703_sincosTest_o : STD_LOGIC_VECTOR (30 downto 0);
    signal aip1E_39high_uid703_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_39high_uid703_sincosTest_q : STD_LOGIC_VECTOR (29 downto 0);
    signal aip1E_39_uid704_sincosTest_q : STD_LOGIC_VECTOR (30 downto 0);
    signal xip1_39_uid705_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_39_uid705_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_39_uid706_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_39_uid706_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid707_sincosTest_in : STD_LOGIC_VECTOR (28 downto 0);
    signal aip1E_uid707_sincosTest_b : STD_LOGIC_VECTOR (28 downto 0);
    signal xMSB_uid708_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid710_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid713_sincosTest_b : STD_LOGIC_VECTOR (73 downto 0);
    signal twoToMiSiYip_uid714_sincosTest_b : STD_LOGIC_VECTOR (73 downto 0);
    signal cstArcTan2Mi_39_uid715_sincosTest_q : STD_LOGIC_VECTOR (26 downto 0);
    signal xip1E_40_uid717_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_40_uid717_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_40_uid717_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_40_uid717_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_40_uid717_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_40_uid718_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_40_uid718_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_40_uid718_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_40_uid718_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_40_uid718_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid720_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid720_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid721_sincosTest_b : STD_LOGIC_VECTOR (27 downto 0);
    signal aip1E_40high_uid722_sincosTest_a : STD_LOGIC_VECTOR (29 downto 0);
    signal aip1E_40high_uid722_sincosTest_b : STD_LOGIC_VECTOR (29 downto 0);
    signal aip1E_40high_uid722_sincosTest_o : STD_LOGIC_VECTOR (29 downto 0);
    signal aip1E_40high_uid722_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_40high_uid722_sincosTest_q : STD_LOGIC_VECTOR (28 downto 0);
    signal aip1E_40_uid723_sincosTest_q : STD_LOGIC_VECTOR (29 downto 0);
    signal xip1_40_uid724_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_40_uid724_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_40_uid725_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_40_uid725_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid726_sincosTest_in : STD_LOGIC_VECTOR (27 downto 0);
    signal aip1E_uid726_sincosTest_b : STD_LOGIC_VECTOR (27 downto 0);
    signal xMSB_uid727_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid729_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid732_sincosTest_b : STD_LOGIC_VECTOR (72 downto 0);
    signal twoToMiSiYip_uid733_sincosTest_b : STD_LOGIC_VECTOR (72 downto 0);
    signal cstArcTan2Mi_40_uid734_sincosTest_q : STD_LOGIC_VECTOR (25 downto 0);
    signal xip1E_41_uid736_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_41_uid736_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_41_uid736_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_41_uid736_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_41_uid736_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_41_uid737_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_41_uid737_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_41_uid737_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_41_uid737_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_41_uid737_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid739_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid739_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid740_sincosTest_b : STD_LOGIC_VECTOR (26 downto 0);
    signal aip1E_41high_uid741_sincosTest_a : STD_LOGIC_VECTOR (28 downto 0);
    signal aip1E_41high_uid741_sincosTest_b : STD_LOGIC_VECTOR (28 downto 0);
    signal aip1E_41high_uid741_sincosTest_o : STD_LOGIC_VECTOR (28 downto 0);
    signal aip1E_41high_uid741_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_41high_uid741_sincosTest_q : STD_LOGIC_VECTOR (27 downto 0);
    signal aip1E_41_uid742_sincosTest_q : STD_LOGIC_VECTOR (28 downto 0);
    signal xip1_41_uid743_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_41_uid743_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_41_uid744_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_41_uid744_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid745_sincosTest_in : STD_LOGIC_VECTOR (26 downto 0);
    signal aip1E_uid745_sincosTest_b : STD_LOGIC_VECTOR (26 downto 0);
    signal xMSB_uid746_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid748_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid751_sincosTest_b : STD_LOGIC_VECTOR (71 downto 0);
    signal twoToMiSiYip_uid752_sincosTest_b : STD_LOGIC_VECTOR (71 downto 0);
    signal cstArcTan2Mi_41_uid753_sincosTest_q : STD_LOGIC_VECTOR (24 downto 0);
    signal xip1E_42_uid755_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_42_uid755_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_42_uid755_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_42_uid755_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_42_uid755_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_42_uid756_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_42_uid756_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_42_uid756_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_42_uid756_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_42_uid756_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid758_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid758_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid759_sincosTest_b : STD_LOGIC_VECTOR (25 downto 0);
    signal aip1E_42high_uid760_sincosTest_a : STD_LOGIC_VECTOR (27 downto 0);
    signal aip1E_42high_uid760_sincosTest_b : STD_LOGIC_VECTOR (27 downto 0);
    signal aip1E_42high_uid760_sincosTest_o : STD_LOGIC_VECTOR (27 downto 0);
    signal aip1E_42high_uid760_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_42high_uid760_sincosTest_q : STD_LOGIC_VECTOR (26 downto 0);
    signal aip1E_42_uid761_sincosTest_q : STD_LOGIC_VECTOR (27 downto 0);
    signal xip1_42_uid762_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_42_uid762_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_42_uid763_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_42_uid763_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid764_sincosTest_in : STD_LOGIC_VECTOR (25 downto 0);
    signal aip1E_uid764_sincosTest_b : STD_LOGIC_VECTOR (25 downto 0);
    signal xMSB_uid765_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid767_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid770_sincosTest_b : STD_LOGIC_VECTOR (70 downto 0);
    signal twoToMiSiYip_uid771_sincosTest_b : STD_LOGIC_VECTOR (70 downto 0);
    signal cstArcTan2Mi_42_uid772_sincosTest_q : STD_LOGIC_VECTOR (23 downto 0);
    signal xip1E_43_uid774_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_43_uid774_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_43_uid774_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_43_uid774_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_43_uid774_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_43_uid775_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_43_uid775_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_43_uid775_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_43_uid775_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_43_uid775_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid777_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid777_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid778_sincosTest_b : STD_LOGIC_VECTOR (24 downto 0);
    signal aip1E_43high_uid779_sincosTest_a : STD_LOGIC_VECTOR (26 downto 0);
    signal aip1E_43high_uid779_sincosTest_b : STD_LOGIC_VECTOR (26 downto 0);
    signal aip1E_43high_uid779_sincosTest_o : STD_LOGIC_VECTOR (26 downto 0);
    signal aip1E_43high_uid779_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_43high_uid779_sincosTest_q : STD_LOGIC_VECTOR (25 downto 0);
    signal aip1E_43_uid780_sincosTest_q : STD_LOGIC_VECTOR (26 downto 0);
    signal xip1_43_uid781_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_43_uid781_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_43_uid782_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_43_uid782_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid783_sincosTest_in : STD_LOGIC_VECTOR (24 downto 0);
    signal aip1E_uid783_sincosTest_b : STD_LOGIC_VECTOR (24 downto 0);
    signal xMSB_uid784_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid786_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid789_sincosTest_b : STD_LOGIC_VECTOR (69 downto 0);
    signal twoToMiSiYip_uid790_sincosTest_b : STD_LOGIC_VECTOR (69 downto 0);
    signal cstArcTan2Mi_43_uid791_sincosTest_q : STD_LOGIC_VECTOR (22 downto 0);
    signal xip1E_44_uid793_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_44_uid793_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_44_uid793_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_44_uid793_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_44_uid793_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_44_uid794_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_44_uid794_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_44_uid794_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_44_uid794_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_44_uid794_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid796_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid796_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid797_sincosTest_b : STD_LOGIC_VECTOR (23 downto 0);
    signal aip1E_44high_uid798_sincosTest_a : STD_LOGIC_VECTOR (25 downto 0);
    signal aip1E_44high_uid798_sincosTest_b : STD_LOGIC_VECTOR (25 downto 0);
    signal aip1E_44high_uid798_sincosTest_o : STD_LOGIC_VECTOR (25 downto 0);
    signal aip1E_44high_uid798_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_44high_uid798_sincosTest_q : STD_LOGIC_VECTOR (24 downto 0);
    signal aip1E_44_uid799_sincosTest_q : STD_LOGIC_VECTOR (25 downto 0);
    signal xip1_44_uid800_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_44_uid800_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_44_uid801_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_44_uid801_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid802_sincosTest_in : STD_LOGIC_VECTOR (23 downto 0);
    signal aip1E_uid802_sincosTest_b : STD_LOGIC_VECTOR (23 downto 0);
    signal xMSB_uid803_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid805_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid808_sincosTest_b : STD_LOGIC_VECTOR (68 downto 0);
    signal twoToMiSiYip_uid809_sincosTest_b : STD_LOGIC_VECTOR (68 downto 0);
    signal cstArcTan2Mi_44_uid810_sincosTest_q : STD_LOGIC_VECTOR (21 downto 0);
    signal xip1E_45_uid812_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_45_uid812_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_45_uid812_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_45_uid812_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_45_uid812_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_45_uid813_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_45_uid813_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_45_uid813_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_45_uid813_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_45_uid813_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid815_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid815_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid816_sincosTest_b : STD_LOGIC_VECTOR (22 downto 0);
    signal aip1E_45high_uid817_sincosTest_a : STD_LOGIC_VECTOR (24 downto 0);
    signal aip1E_45high_uid817_sincosTest_b : STD_LOGIC_VECTOR (24 downto 0);
    signal aip1E_45high_uid817_sincosTest_o : STD_LOGIC_VECTOR (24 downto 0);
    signal aip1E_45high_uid817_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_45high_uid817_sincosTest_q : STD_LOGIC_VECTOR (23 downto 0);
    signal aip1E_45_uid818_sincosTest_q : STD_LOGIC_VECTOR (24 downto 0);
    signal xip1_45_uid819_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_45_uid819_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_45_uid820_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_45_uid820_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid821_sincosTest_in : STD_LOGIC_VECTOR (22 downto 0);
    signal aip1E_uid821_sincosTest_b : STD_LOGIC_VECTOR (22 downto 0);
    signal xMSB_uid822_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid824_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid827_sincosTest_b : STD_LOGIC_VECTOR (67 downto 0);
    signal twoToMiSiYip_uid828_sincosTest_b : STD_LOGIC_VECTOR (67 downto 0);
    signal cstArcTan2Mi_45_uid829_sincosTest_q : STD_LOGIC_VECTOR (20 downto 0);
    signal xip1E_46_uid831_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_46_uid831_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_46_uid831_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_46_uid831_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_46_uid831_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_46_uid832_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_46_uid832_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_46_uid832_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_46_uid832_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_46_uid832_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid834_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid834_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid835_sincosTest_b : STD_LOGIC_VECTOR (21 downto 0);
    signal aip1E_46high_uid836_sincosTest_a : STD_LOGIC_VECTOR (23 downto 0);
    signal aip1E_46high_uid836_sincosTest_b : STD_LOGIC_VECTOR (23 downto 0);
    signal aip1E_46high_uid836_sincosTest_o : STD_LOGIC_VECTOR (23 downto 0);
    signal aip1E_46high_uid836_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_46high_uid836_sincosTest_q : STD_LOGIC_VECTOR (22 downto 0);
    signal aip1E_46_uid837_sincosTest_q : STD_LOGIC_VECTOR (23 downto 0);
    signal xip1_46_uid838_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_46_uid838_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_46_uid839_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_46_uid839_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid840_sincosTest_in : STD_LOGIC_VECTOR (21 downto 0);
    signal aip1E_uid840_sincosTest_b : STD_LOGIC_VECTOR (21 downto 0);
    signal xMSB_uid841_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid843_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid846_sincosTest_b : STD_LOGIC_VECTOR (66 downto 0);
    signal twoToMiSiYip_uid847_sincosTest_b : STD_LOGIC_VECTOR (66 downto 0);
    signal cstArcTan2Mi_46_uid848_sincosTest_q : STD_LOGIC_VECTOR (19 downto 0);
    signal xip1E_47_uid850_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_47_uid850_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_47_uid850_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_47_uid850_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_47_uid850_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_47_uid851_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_47_uid851_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_47_uid851_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_47_uid851_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_47_uid851_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid853_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid853_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid854_sincosTest_b : STD_LOGIC_VECTOR (20 downto 0);
    signal aip1E_47high_uid855_sincosTest_a : STD_LOGIC_VECTOR (22 downto 0);
    signal aip1E_47high_uid855_sincosTest_b : STD_LOGIC_VECTOR (22 downto 0);
    signal aip1E_47high_uid855_sincosTest_o : STD_LOGIC_VECTOR (22 downto 0);
    signal aip1E_47high_uid855_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_47high_uid855_sincosTest_q : STD_LOGIC_VECTOR (21 downto 0);
    signal aip1E_47_uid856_sincosTest_q : STD_LOGIC_VECTOR (22 downto 0);
    signal xip1_47_uid857_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_47_uid857_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_47_uid858_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_47_uid858_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid859_sincosTest_in : STD_LOGIC_VECTOR (20 downto 0);
    signal aip1E_uid859_sincosTest_b : STD_LOGIC_VECTOR (20 downto 0);
    signal xMSB_uid860_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid862_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid865_sincosTest_b : STD_LOGIC_VECTOR (65 downto 0);
    signal twoToMiSiYip_uid866_sincosTest_b : STD_LOGIC_VECTOR (65 downto 0);
    signal cstArcTan2Mi_47_uid867_sincosTest_q : STD_LOGIC_VECTOR (18 downto 0);
    signal xip1E_48_uid869_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_48_uid869_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_48_uid869_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_48_uid869_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_48_uid869_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_48_uid870_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_48_uid870_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_48_uid870_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_48_uid870_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_48_uid870_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid872_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid872_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid873_sincosTest_b : STD_LOGIC_VECTOR (19 downto 0);
    signal aip1E_48high_uid874_sincosTest_a : STD_LOGIC_VECTOR (21 downto 0);
    signal aip1E_48high_uid874_sincosTest_b : STD_LOGIC_VECTOR (21 downto 0);
    signal aip1E_48high_uid874_sincosTest_o : STD_LOGIC_VECTOR (21 downto 0);
    signal aip1E_48high_uid874_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_48high_uid874_sincosTest_q : STD_LOGIC_VECTOR (20 downto 0);
    signal aip1E_48_uid875_sincosTest_q : STD_LOGIC_VECTOR (21 downto 0);
    signal xip1_48_uid876_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_48_uid876_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_48_uid877_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_48_uid877_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid878_sincosTest_in : STD_LOGIC_VECTOR (19 downto 0);
    signal aip1E_uid878_sincosTest_b : STD_LOGIC_VECTOR (19 downto 0);
    signal xMSB_uid879_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid881_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid884_sincosTest_b : STD_LOGIC_VECTOR (64 downto 0);
    signal twoToMiSiYip_uid885_sincosTest_b : STD_LOGIC_VECTOR (64 downto 0);
    signal cstArcTan2Mi_48_uid886_sincosTest_q : STD_LOGIC_VECTOR (17 downto 0);
    signal xip1E_49_uid888_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_49_uid888_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_49_uid888_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_49_uid888_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_49_uid888_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_49_uid889_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_49_uid889_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_49_uid889_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_49_uid889_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_49_uid889_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid891_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid891_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid892_sincosTest_b : STD_LOGIC_VECTOR (18 downto 0);
    signal aip1E_49high_uid893_sincosTest_a : STD_LOGIC_VECTOR (20 downto 0);
    signal aip1E_49high_uid893_sincosTest_b : STD_LOGIC_VECTOR (20 downto 0);
    signal aip1E_49high_uid893_sincosTest_o : STD_LOGIC_VECTOR (20 downto 0);
    signal aip1E_49high_uid893_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_49high_uid893_sincosTest_q : STD_LOGIC_VECTOR (19 downto 0);
    signal aip1E_49_uid894_sincosTest_q : STD_LOGIC_VECTOR (20 downto 0);
    signal xip1_49_uid895_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_49_uid895_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_49_uid896_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_49_uid896_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid897_sincosTest_in : STD_LOGIC_VECTOR (18 downto 0);
    signal aip1E_uid897_sincosTest_b : STD_LOGIC_VECTOR (18 downto 0);
    signal xMSB_uid898_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid900_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid903_sincosTest_b : STD_LOGIC_VECTOR (63 downto 0);
    signal twoToMiSiYip_uid904_sincosTest_b : STD_LOGIC_VECTOR (63 downto 0);
    signal cstArcTan2Mi_49_uid905_sincosTest_q : STD_LOGIC_VECTOR (16 downto 0);
    signal xip1E_50_uid907_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_50_uid907_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_50_uid907_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_50_uid907_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_50_uid907_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_50_uid908_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_50_uid908_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_50_uid908_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_50_uid908_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_50_uid908_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid910_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid910_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid911_sincosTest_b : STD_LOGIC_VECTOR (17 downto 0);
    signal aip1E_50high_uid912_sincosTest_a : STD_LOGIC_VECTOR (19 downto 0);
    signal aip1E_50high_uid912_sincosTest_b : STD_LOGIC_VECTOR (19 downto 0);
    signal aip1E_50high_uid912_sincosTest_o : STD_LOGIC_VECTOR (19 downto 0);
    signal aip1E_50high_uid912_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_50high_uid912_sincosTest_q : STD_LOGIC_VECTOR (18 downto 0);
    signal aip1E_50_uid913_sincosTest_q : STD_LOGIC_VECTOR (19 downto 0);
    signal xip1_50_uid914_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_50_uid914_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_50_uid915_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_50_uid915_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid916_sincosTest_in : STD_LOGIC_VECTOR (17 downto 0);
    signal aip1E_uid916_sincosTest_b : STD_LOGIC_VECTOR (17 downto 0);
    signal xMSB_uid917_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid919_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid922_sincosTest_b : STD_LOGIC_VECTOR (62 downto 0);
    signal twoToMiSiYip_uid923_sincosTest_b : STD_LOGIC_VECTOR (62 downto 0);
    signal cstArcTan2Mi_50_uid924_sincosTest_q : STD_LOGIC_VECTOR (15 downto 0);
    signal xip1E_51_uid926_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_51_uid926_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_51_uid926_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_51_uid926_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_51_uid926_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_51_uid927_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_51_uid927_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_51_uid927_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_51_uid927_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_51_uid927_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid929_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid929_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid930_sincosTest_b : STD_LOGIC_VECTOR (16 downto 0);
    signal aip1E_51high_uid931_sincosTest_a : STD_LOGIC_VECTOR (18 downto 0);
    signal aip1E_51high_uid931_sincosTest_b : STD_LOGIC_VECTOR (18 downto 0);
    signal aip1E_51high_uid931_sincosTest_o : STD_LOGIC_VECTOR (18 downto 0);
    signal aip1E_51high_uid931_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_51high_uid931_sincosTest_q : STD_LOGIC_VECTOR (17 downto 0);
    signal aip1E_51_uid932_sincosTest_q : STD_LOGIC_VECTOR (18 downto 0);
    signal xip1_51_uid933_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_51_uid933_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_51_uid934_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_51_uid934_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid935_sincosTest_in : STD_LOGIC_VECTOR (16 downto 0);
    signal aip1E_uid935_sincosTest_b : STD_LOGIC_VECTOR (16 downto 0);
    signal xMSB_uid936_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid938_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid941_sincosTest_b : STD_LOGIC_VECTOR (61 downto 0);
    signal twoToMiSiYip_uid942_sincosTest_b : STD_LOGIC_VECTOR (61 downto 0);
    signal cstArcTan2Mi_51_uid943_sincosTest_q : STD_LOGIC_VECTOR (14 downto 0);
    signal xip1E_52_uid945_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_52_uid945_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_52_uid945_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_52_uid945_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_52_uid945_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_52_uid946_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_52_uid946_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_52_uid946_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_52_uid946_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_52_uid946_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid948_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid948_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid949_sincosTest_b : STD_LOGIC_VECTOR (15 downto 0);
    signal aip1E_52high_uid950_sincosTest_a : STD_LOGIC_VECTOR (17 downto 0);
    signal aip1E_52high_uid950_sincosTest_b : STD_LOGIC_VECTOR (17 downto 0);
    signal aip1E_52high_uid950_sincosTest_o : STD_LOGIC_VECTOR (17 downto 0);
    signal aip1E_52high_uid950_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_52high_uid950_sincosTest_q : STD_LOGIC_VECTOR (16 downto 0);
    signal aip1E_52_uid951_sincosTest_q : STD_LOGIC_VECTOR (17 downto 0);
    signal xip1_52_uid952_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_52_uid952_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_52_uid953_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_52_uid953_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid954_sincosTest_in : STD_LOGIC_VECTOR (15 downto 0);
    signal aip1E_uid954_sincosTest_b : STD_LOGIC_VECTOR (15 downto 0);
    signal xMSB_uid955_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid957_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid960_sincosTest_b : STD_LOGIC_VECTOR (60 downto 0);
    signal twoToMiSiYip_uid961_sincosTest_b : STD_LOGIC_VECTOR (60 downto 0);
    signal cstArcTan2Mi_52_uid962_sincosTest_q : STD_LOGIC_VECTOR (13 downto 0);
    signal xip1E_53_uid964_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_53_uid964_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_53_uid964_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_53_uid964_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_53_uid964_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_53_uid965_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_53_uid965_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_53_uid965_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_53_uid965_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_53_uid965_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid967_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid967_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid968_sincosTest_b : STD_LOGIC_VECTOR (14 downto 0);
    signal aip1E_53high_uid969_sincosTest_a : STD_LOGIC_VECTOR (16 downto 0);
    signal aip1E_53high_uid969_sincosTest_b : STD_LOGIC_VECTOR (16 downto 0);
    signal aip1E_53high_uid969_sincosTest_o : STD_LOGIC_VECTOR (16 downto 0);
    signal aip1E_53high_uid969_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_53high_uid969_sincosTest_q : STD_LOGIC_VECTOR (15 downto 0);
    signal aip1E_53_uid970_sincosTest_q : STD_LOGIC_VECTOR (16 downto 0);
    signal xip1_53_uid971_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_53_uid971_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_53_uid972_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_53_uid972_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid973_sincosTest_in : STD_LOGIC_VECTOR (14 downto 0);
    signal aip1E_uid973_sincosTest_b : STD_LOGIC_VECTOR (14 downto 0);
    signal xMSB_uid974_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid976_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid979_sincosTest_b : STD_LOGIC_VECTOR (59 downto 0);
    signal twoToMiSiYip_uid980_sincosTest_b : STD_LOGIC_VECTOR (59 downto 0);
    signal cstArcTan2Mi_53_uid981_sincosTest_q : STD_LOGIC_VECTOR (12 downto 0);
    signal xip1E_54_uid983_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_54_uid983_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_54_uid983_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_54_uid983_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_54_uid983_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_54_uid984_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_54_uid984_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_54_uid984_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_54_uid984_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_54_uid984_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal lowRangeA_uid986_sincosTest_in : STD_LOGIC_VECTOR (0 downto 0);
    signal lowRangeA_uid986_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal highABits_uid987_sincosTest_b : STD_LOGIC_VECTOR (13 downto 0);
    signal aip1E_54high_uid988_sincosTest_a : STD_LOGIC_VECTOR (15 downto 0);
    signal aip1E_54high_uid988_sincosTest_b : STD_LOGIC_VECTOR (15 downto 0);
    signal aip1E_54high_uid988_sincosTest_o : STD_LOGIC_VECTOR (15 downto 0);
    signal aip1E_54high_uid988_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal aip1E_54high_uid988_sincosTest_q : STD_LOGIC_VECTOR (14 downto 0);
    signal aip1E_54_uid989_sincosTest_q : STD_LOGIC_VECTOR (15 downto 0);
    signal xip1_54_uid990_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_54_uid990_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_54_uid991_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_54_uid991_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal aip1E_uid992_sincosTest_in : STD_LOGIC_VECTOR (13 downto 0);
    signal aip1E_uid992_sincosTest_b : STD_LOGIC_VECTOR (13 downto 0);
    signal xMSB_uid993_sincosTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signOfSelectionSignal_uid995_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal twoToMiSiXip_uid998_sincosTest_b : STD_LOGIC_VECTOR (58 downto 0);
    signal twoToMiSiYip_uid999_sincosTest_b : STD_LOGIC_VECTOR (58 downto 0);
    signal xip1E_55_uid1002_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_55_uid1002_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_55_uid1002_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal xip1E_55_uid1002_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xip1E_55_uid1002_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal yip1E_55_uid1003_sincosTest_a : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_55_uid1003_sincosTest_b : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_55_uid1003_sincosTest_o : STD_LOGIC_VECTOR (114 downto 0);
    signal yip1E_55_uid1003_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal yip1E_55_uid1003_sincosTest_q : STD_LOGIC_VECTOR (113 downto 0);
    signal xip1_55_uid1009_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal xip1_55_uid1009_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_55_uid1010_sincosTest_in : STD_LOGIC_VECTOR (112 downto 0);
    signal yip1_55_uid1010_sincosTest_b : STD_LOGIC_VECTOR (112 downto 0);
    signal xSumPreRnd_uid1012_sincosTest_in : STD_LOGIC_VECTOR (111 downto 0);
    signal xSumPreRnd_uid1012_sincosTest_b : STD_LOGIC_VECTOR (55 downto 0);
    signal xSumPostRnd_uid1015_sincosTest_a : STD_LOGIC_VECTOR (56 downto 0);
    signal xSumPostRnd_uid1015_sincosTest_b : STD_LOGIC_VECTOR (56 downto 0);
    signal xSumPostRnd_uid1015_sincosTest_o : STD_LOGIC_VECTOR (56 downto 0);
    signal xSumPostRnd_uid1015_sincosTest_q : STD_LOGIC_VECTOR (56 downto 0);
    signal ySumPreRnd_uid1016_sincosTest_in : STD_LOGIC_VECTOR (111 downto 0);
    signal ySumPreRnd_uid1016_sincosTest_b : STD_LOGIC_VECTOR (55 downto 0);
    signal ySumPostRnd_uid1019_sincosTest_a : STD_LOGIC_VECTOR (56 downto 0);
    signal ySumPostRnd_uid1019_sincosTest_b : STD_LOGIC_VECTOR (56 downto 0);
    signal ySumPostRnd_uid1019_sincosTest_o : STD_LOGIC_VECTOR (56 downto 0);
    signal ySumPostRnd_uid1019_sincosTest_q : STD_LOGIC_VECTOR (56 downto 0);
    signal xPostExc_uid1020_sincosTest_in : STD_LOGIC_VECTOR (55 downto 0);
    signal xPostExc_uid1020_sincosTest_b : STD_LOGIC_VECTOR (54 downto 0);
    signal yPostExc_uid1021_sincosTest_in : STD_LOGIC_VECTOR (55 downto 0);
    signal yPostExc_uid1021_sincosTest_b : STD_LOGIC_VECTOR (54 downto 0);
    signal invFirstQuadrant_uid1022_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal sinNegCond2_uid1023_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal sinNegCond1_uid1024_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal sinNegCond0_uid1026_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal sinNegCond_uid1027_sincosTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal sinNegCond_uid1027_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal cstZeroForAddSub_uid1029_sincosTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal invSinNegCond_uid1030_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal sinPostNeg_uid1031_sincosTest_a : STD_LOGIC_VECTOR (56 downto 0);
    signal sinPostNeg_uid1031_sincosTest_b : STD_LOGIC_VECTOR (56 downto 0);
    signal sinPostNeg_uid1031_sincosTest_o : STD_LOGIC_VECTOR (56 downto 0);
    signal sinPostNeg_uid1031_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal sinPostNeg_uid1031_sincosTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal invCosNegCond_uid1032_sincosTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal invCosNegCond_uid1032_sincosTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal cosPostNeg_uid1033_sincosTest_a : STD_LOGIC_VECTOR (56 downto 0);
    signal cosPostNeg_uid1033_sincosTest_b : STD_LOGIC_VECTOR (56 downto 0);
    signal cosPostNeg_uid1033_sincosTest_o : STD_LOGIC_VECTOR (56 downto 0);
    signal cosPostNeg_uid1033_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal cosPostNeg_uid1033_sincosTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal xPostRR_uid1034_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xPostRR_uid1034_sincosTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal xPostRR_uid1035_sincosTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal xPostRR_uid1035_sincosTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal cos_uid1036_sincosTest_in : STD_LOGIC_VECTOR (54 downto 0);
    signal cos_uid1036_sincosTest_b : STD_LOGIC_VECTOR (54 downto 0);
    signal sin_uid1037_sincosTest_in : STD_LOGIC_VECTOR (54 downto 0);
    signal sin_uid1037_sincosTest_b : STD_LOGIC_VECTOR (54 downto 0);
    signal redist2_yPostExc_uid1021_sincosTest_b_1_q : STD_LOGIC_VECTOR (54 downto 0);
    signal redist3_xPostExc_uid1020_sincosTest_b_1_q : STD_LOGIC_VECTOR (54 downto 0);
    signal redist4_ySumPreRnd_uid1016_sincosTest_b_1_q : STD_LOGIC_VECTOR (55 downto 0);
    signal redist5_xSumPreRnd_uid1012_sincosTest_b_1_q : STD_LOGIC_VECTOR (55 downto 0);
    signal redist6_xMSB_uid993_sincosTest_b_16_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist7_yip1_54_uid991_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist8_xip1_54_uid990_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist9_xMSB_uid974_sincosTest_b_15_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist10_yip1_53_uid972_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist11_xip1_53_uid971_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist12_xMSB_uid955_sincosTest_b_14_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist13_aip1E_uid954_sincosTest_b_1_q : STD_LOGIC_VECTOR (15 downto 0);
    signal redist14_yip1_52_uid953_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist15_xip1_52_uid952_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist16_xMSB_uid936_sincosTest_b_14_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist17_yip1_51_uid934_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist18_xip1_51_uid933_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist19_xMSB_uid917_sincosTest_b_13_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist20_aip1E_uid916_sincosTest_b_1_q : STD_LOGIC_VECTOR (17 downto 0);
    signal redist21_yip1_50_uid915_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist22_xip1_50_uid914_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist23_xMSB_uid898_sincosTest_b_13_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist24_yip1_49_uid896_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist25_xip1_49_uid895_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist26_xMSB_uid879_sincosTest_b_12_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist27_aip1E_uid878_sincosTest_b_1_q : STD_LOGIC_VECTOR (19 downto 0);
    signal redist28_yip1_48_uid877_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist29_xip1_48_uid876_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist30_xMSB_uid860_sincosTest_b_12_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist31_yip1_47_uid858_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist32_xip1_47_uid857_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist33_xMSB_uid841_sincosTest_b_11_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist34_aip1E_uid840_sincosTest_b_1_q : STD_LOGIC_VECTOR (21 downto 0);
    signal redist35_yip1_46_uid839_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist36_xip1_46_uid838_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist37_xMSB_uid822_sincosTest_b_11_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist38_yip1_45_uid820_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist39_xip1_45_uid819_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist40_xMSB_uid803_sincosTest_b_10_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist41_aip1E_uid802_sincosTest_b_1_q : STD_LOGIC_VECTOR (23 downto 0);
    signal redist42_yip1_44_uid801_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist43_xip1_44_uid800_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist44_xMSB_uid784_sincosTest_b_10_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist45_yip1_43_uid782_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist46_xip1_43_uid781_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist47_xMSB_uid765_sincosTest_b_9_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist48_aip1E_uid764_sincosTest_b_1_q : STD_LOGIC_VECTOR (25 downto 0);
    signal redist49_yip1_42_uid763_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist50_xip1_42_uid762_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist51_xMSB_uid746_sincosTest_b_9_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist52_yip1_41_uid744_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist53_xip1_41_uid743_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist54_xMSB_uid727_sincosTest_b_8_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist55_aip1E_uid726_sincosTest_b_1_q : STD_LOGIC_VECTOR (27 downto 0);
    signal redist56_yip1_40_uid725_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist57_xip1_40_uid724_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist58_xMSB_uid708_sincosTest_b_8_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist59_yip1_39_uid706_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist60_xip1_39_uid705_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist61_xMSB_uid689_sincosTest_b_7_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist62_aip1E_uid688_sincosTest_b_1_q : STD_LOGIC_VECTOR (29 downto 0);
    signal redist63_yip1_38_uid687_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist64_xip1_38_uid686_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist65_xMSB_uid670_sincosTest_b_7_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist66_yip1_37_uid668_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist67_xip1_37_uid667_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist68_xMSB_uid651_sincosTest_b_6_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist69_aip1E_uid650_sincosTest_b_1_q : STD_LOGIC_VECTOR (31 downto 0);
    signal redist70_yip1_36_uid649_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist71_xip1_36_uid648_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist72_xMSB_uid632_sincosTest_b_6_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist73_yip1_35_uid630_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist74_xip1_35_uid629_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist75_xMSB_uid613_sincosTest_b_5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist76_aip1E_uid612_sincosTest_b_1_q : STD_LOGIC_VECTOR (33 downto 0);
    signal redist77_yip1_34_uid611_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist78_xip1_34_uid610_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist79_xMSB_uid594_sincosTest_b_5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist80_yip1_33_uid592_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist81_xip1_33_uid591_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist82_xMSB_uid575_sincosTest_b_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist83_aip1E_uid574_sincosTest_b_1_q : STD_LOGIC_VECTOR (35 downto 0);
    signal redist84_yip1_32_uid573_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist85_xip1_32_uid572_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist86_xMSB_uid556_sincosTest_b_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist87_yip1_31_uid554_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist88_xip1_31_uid553_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist89_xMSB_uid537_sincosTest_b_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist90_aip1E_uid536_sincosTest_b_1_q : STD_LOGIC_VECTOR (37 downto 0);
    signal redist91_yip1_30_uid535_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist92_xip1_30_uid534_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist93_xMSB_uid518_sincosTest_b_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist94_yip1_29_uid516_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist95_xip1_29_uid515_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist96_xMSB_uid499_sincosTest_b_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist97_aip1E_uid498_sincosTest_b_1_q : STD_LOGIC_VECTOR (39 downto 0);
    signal redist98_yip1_28_uid497_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist99_xip1_28_uid496_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist100_xMSB_uid480_sincosTest_b_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist101_yip1_27_uid478_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist102_xip1_27_uid477_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist103_xMSB_uid461_sincosTest_b_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist104_aip1E_uid460_sincosTest_b_1_q : STD_LOGIC_VECTOR (41 downto 0);
    signal redist105_yip1_26_uid459_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist106_xip1_26_uid458_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist107_xMSB_uid442_sincosTest_b_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist108_yip1_25_uid440_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist109_xip1_25_uid439_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist110_aip1E_uid422_sincosTest_b_1_q : STD_LOGIC_VECTOR (43 downto 0);
    signal redist111_yip1_24_uid421_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist112_xip1_24_uid420_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist113_aip1E_uid403_sincosTest_b_1_q : STD_LOGIC_VECTOR (44 downto 0);
    signal redist114_yip1_23_uid402_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist115_xip1_23_uid401_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist116_aip1E_uid384_sincosTest_b_1_q : STD_LOGIC_VECTOR (45 downto 0);
    signal redist117_yip1_22_uid383_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist118_xip1_22_uid382_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist119_aip1E_uid368_sincosTest_b_1_q : STD_LOGIC_VECTOR (46 downto 0);
    signal redist120_yip1_21_uid367_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist121_xip1_21_uid366_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist122_aip1E_uid352_sincosTest_b_1_q : STD_LOGIC_VECTOR (47 downto 0);
    signal redist123_yip1_20_uid351_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist124_xip1_20_uid350_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist125_aip1E_uid336_sincosTest_b_1_q : STD_LOGIC_VECTOR (48 downto 0);
    signal redist126_yip1_19_uid335_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist127_xip1_19_uid334_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist128_aip1E_uid320_sincosTest_b_1_q : STD_LOGIC_VECTOR (49 downto 0);
    signal redist129_yip1_18_uid319_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist130_xip1_18_uid318_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist131_aip1E_uid304_sincosTest_b_1_q : STD_LOGIC_VECTOR (50 downto 0);
    signal redist132_yip1_17_uid303_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist133_xip1_17_uid302_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist134_aip1E_uid288_sincosTest_b_1_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist135_yip1_16_uid287_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist136_xip1_16_uid286_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist137_aip1E_uid272_sincosTest_b_1_q : STD_LOGIC_VECTOR (52 downto 0);
    signal redist138_yip1_15_uid271_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist139_xip1_15_uid270_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist140_aip1E_uid256_sincosTest_b_1_q : STD_LOGIC_VECTOR (53 downto 0);
    signal redist141_yip1_14_uid255_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist142_xip1_14_uid254_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist143_aip1E_uid240_sincosTest_b_1_q : STD_LOGIC_VECTOR (54 downto 0);
    signal redist144_yip1_13_uid239_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist145_xip1_13_uid238_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist146_aip1E_uid224_sincosTest_b_1_q : STD_LOGIC_VECTOR (55 downto 0);
    signal redist147_yip1_12_uid223_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist148_xip1_12_uid222_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist149_aip1E_uid208_sincosTest_b_1_q : STD_LOGIC_VECTOR (56 downto 0);
    signal redist150_yip1_11_uid207_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist151_xip1_11_uid206_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist152_aip1E_uid192_sincosTest_b_1_q : STD_LOGIC_VECTOR (57 downto 0);
    signal redist153_yip1_10_uid191_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist154_xip1_10_uid190_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist155_aip1E_uid176_sincosTest_b_1_q : STD_LOGIC_VECTOR (58 downto 0);
    signal redist156_yip1_9_uid175_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist157_xip1_9_uid174_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist158_aip1E_uid160_sincosTest_b_1_q : STD_LOGIC_VECTOR (59 downto 0);
    signal redist159_yip1_8_uid159_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist160_xip1_8_uid158_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist161_aip1E_uid144_sincosTest_b_1_q : STD_LOGIC_VECTOR (60 downto 0);
    signal redist162_yip1_7_uid143_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist163_xip1_7_uid142_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist164_aip1E_uid128_sincosTest_b_1_q : STD_LOGIC_VECTOR (61 downto 0);
    signal redist165_yip1_6_uid127_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist166_xip1_6_uid126_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist167_aip1E_uid112_sincosTest_b_1_q : STD_LOGIC_VECTOR (62 downto 0);
    signal redist168_yip1_5_uid111_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist169_xip1_5_uid110_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist170_aip1E_uid96_sincosTest_b_1_q : STD_LOGIC_VECTOR (63 downto 0);
    signal redist171_yip1_4_uid95_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist172_xip1_4_uid94_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist173_aip1E_uid80_sincosTest_b_1_q : STD_LOGIC_VECTOR (64 downto 0);
    signal redist174_yip1_3_uid79_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist175_xip1_3_uid78_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist176_aip1E_uid64_sincosTest_b_1_q : STD_LOGIC_VECTOR (65 downto 0);
    signal redist177_yip1_2_uid63_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist178_xip1_2_uid62_sincosTest_b_1_q : STD_LOGIC_VECTOR (112 downto 0);
    signal redist179_aip1E_uid48_sincosTest_b_1_q : STD_LOGIC_VECTOR (66 downto 0);
    signal redist181_absAR_uid10_sincosTest_b_1_q : STD_LOGIC_VECTOR (54 downto 0);
    signal redist182_invSignA_uid8_sincosTest_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist183_signA_uid7_sincosTest_b_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_outputreg_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_reset0 : std_logic;
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_ia : STD_LOGIC_VECTOR (0 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_aa : STD_LOGIC_VECTOR (5 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_ab : STD_LOGIC_VECTOR (5 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_iq : STD_LOGIC_VECTOR (0 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_q : STD_LOGIC_VECTOR (5 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_i : UNSIGNED (5 downto 0);
    attribute preserve : boolean;
    attribute preserve of redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_i : signal is true;
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_eq : std_logic;
    attribute preserve of redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_eq : signal is true;
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_wraddr_q : STD_LOGIC_VECTOR (5 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_last_q : STD_LOGIC_VECTOR (6 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_cmp_b : STD_LOGIC_VECTOR (6 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_cmp_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_cmpReg_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_notEnable_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_nor_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_sticky_ena_q : STD_LOGIC_VECTOR (0 downto 0);
    attribute preserve_syn_only : boolean;
    attribute preserve_syn_only of redist0_invCosNegCond_uid1032_sincosTest_q_57_sticky_ena_q : signal is true;
    signal redist0_invCosNegCond_uid1032_sincosTest_q_57_enaAnd_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_outputreg_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_mem_reset0 : std_logic;
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_mem_ia : STD_LOGIC_VECTOR (0 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_mem_aa : STD_LOGIC_VECTOR (5 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_mem_ab : STD_LOGIC_VECTOR (5 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_mem_iq : STD_LOGIC_VECTOR (0 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_mem_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_q : STD_LOGIC_VECTOR (5 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_i : UNSIGNED (5 downto 0);
    attribute preserve of redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_i : signal is true;
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_eq : std_logic;
    attribute preserve of redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_eq : signal is true;
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_wraddr_q : STD_LOGIC_VECTOR (5 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_mem_last_q : STD_LOGIC_VECTOR (6 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_cmp_b : STD_LOGIC_VECTOR (6 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_cmp_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_cmpReg_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_notEnable_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_nor_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_sticky_ena_q : STD_LOGIC_VECTOR (0 downto 0);
    attribute preserve_syn_only of redist1_sinNegCond_uid1027_sincosTest_q_57_sticky_ena_q : signal is true;
    signal redist1_sinNegCond_uid1027_sincosTest_q_57_enaAnd_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_inputreg_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_mem_reset0 : std_logic;
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_mem_ia : STD_LOGIC_VECTOR (0 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_mem_aa : STD_LOGIC_VECTOR (5 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_mem_ab : STD_LOGIC_VECTOR (5 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_mem_iq : STD_LOGIC_VECTOR (0 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_mem_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_q : STD_LOGIC_VECTOR (5 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_i : UNSIGNED (5 downto 0);
    attribute preserve of redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_i : signal is true;
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_eq : std_logic;
    attribute preserve of redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_eq : signal is true;
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_wraddr_q : STD_LOGIC_VECTOR (5 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_mem_last_q : STD_LOGIC_VECTOR (6 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_cmp_b : STD_LOGIC_VECTOR (6 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_cmp_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_cmpReg_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_notEnable_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_nor_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_sticky_ena_q : STD_LOGIC_VECTOR (0 downto 0);
    attribute preserve_syn_only of redist180_firstQuadrant_uid15_sincosTest_b_57_sticky_ena_q : signal is true;
    signal redist180_firstQuadrant_uid15_sincosTest_b_57_enaAnd_q : STD_LOGIC_VECTOR (0 downto 0);

begin


    -- redist1_sinNegCond_uid1027_sincosTest_q_57_notEnable(LOGICAL,1239)
    redist1_sinNegCond_uid1027_sincosTest_q_57_notEnable_q <= STD_LOGIC_VECTOR(not (VCC_q));

    -- redist1_sinNegCond_uid1027_sincosTest_q_57_nor(LOGICAL,1240)
    redist1_sinNegCond_uid1027_sincosTest_q_57_nor_q <= not (redist1_sinNegCond_uid1027_sincosTest_q_57_notEnable_q or redist1_sinNegCond_uid1027_sincosTest_q_57_sticky_ena_q);

    -- redist1_sinNegCond_uid1027_sincosTest_q_57_mem_last(CONSTANT,1236)
    redist1_sinNegCond_uid1027_sincosTest_q_57_mem_last_q <= "0110100";

    -- redist1_sinNegCond_uid1027_sincosTest_q_57_cmp(LOGICAL,1237)
    redist1_sinNegCond_uid1027_sincosTest_q_57_cmp_b <= STD_LOGIC_VECTOR("0" & redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_q);
    redist1_sinNegCond_uid1027_sincosTest_q_57_cmp_q <= "1" WHEN redist1_sinNegCond_uid1027_sincosTest_q_57_mem_last_q = redist1_sinNegCond_uid1027_sincosTest_q_57_cmp_b ELSE "0";

    -- redist1_sinNegCond_uid1027_sincosTest_q_57_cmpReg(REG,1238)
    redist1_sinNegCond_uid1027_sincosTest_q_57_cmpReg_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            redist1_sinNegCond_uid1027_sincosTest_q_57_cmpReg_q <= "0";
        ELSIF (clk'EVENT AND clk = '1') THEN
            redist1_sinNegCond_uid1027_sincosTest_q_57_cmpReg_q <= STD_LOGIC_VECTOR(redist1_sinNegCond_uid1027_sincosTest_q_57_cmp_q);
        END IF;
    END PROCESS;

    -- redist1_sinNegCond_uid1027_sincosTest_q_57_sticky_ena(REG,1241)
    redist1_sinNegCond_uid1027_sincosTest_q_57_sticky_ena_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            redist1_sinNegCond_uid1027_sincosTest_q_57_sticky_ena_q <= "0";
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (redist1_sinNegCond_uid1027_sincosTest_q_57_nor_q = "1") THEN
                redist1_sinNegCond_uid1027_sincosTest_q_57_sticky_ena_q <= STD_LOGIC_VECTOR(redist1_sinNegCond_uid1027_sincosTest_q_57_cmpReg_q);
            END IF;
        END IF;
    END PROCESS;

    -- redist1_sinNegCond_uid1027_sincosTest_q_57_enaAnd(LOGICAL,1242)
    redist1_sinNegCond_uid1027_sincosTest_q_57_enaAnd_q <= redist1_sinNegCond_uid1027_sincosTest_q_57_sticky_ena_q and VCC_q;

    -- redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt(COUNTER,1234)
    -- low=0, high=53, step=1, init=0
    redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_i <= TO_UNSIGNED(0, 6);
            redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_eq <= '0';
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_i = TO_UNSIGNED(52, 6)) THEN
                redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_eq <= '1';
            ELSE
                redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_eq <= '0';
            END IF;
            IF (redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_eq = '1') THEN
                redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_i <= redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_i + 11;
            ELSE
                redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_i <= redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_i + 1;
            END IF;
        END IF;
    END PROCESS;
    redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_q <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR(RESIZE(redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_i, 6)));

    -- cstPiO2_uid11_sincosTest(CONSTANT,10)
    cstPiO2_uid11_sincosTest_q <= "110010010000111111011010101000100010000101101000110000100011010011";

    -- signA_uid7_sincosTest(BITSELECT,6)@0
    signA_uid7_sincosTest_b <= STD_LOGIC_VECTOR(a(55 downto 55));

    -- invSignA_uid8_sincosTest(LOGICAL,7)@0
    invSignA_uid8_sincosTest_q <= not (signA_uid7_sincosTest_b);

    -- constantZero_uid6_sincosTest(CONSTANT,5)
    constantZero_uid6_sincosTest_q <= "00000000000000000000000000000000000000000000000000000000";

    -- absAE_uid9_sincosTest(ADDSUB,8)@0
    absAE_uid9_sincosTest_s <= invSignA_uid8_sincosTest_q;
    absAE_uid9_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((57 downto 56 => constantZero_uid6_sincosTest_q(55)) & constantZero_uid6_sincosTest_q));
    absAE_uid9_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((57 downto 56 => a(55)) & a));
    absAE_uid9_sincosTest_combproc: PROCESS (absAE_uid9_sincosTest_a, absAE_uid9_sincosTest_b, absAE_uid9_sincosTest_s)
    BEGIN
        IF (absAE_uid9_sincosTest_s = "1") THEN
            absAE_uid9_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(absAE_uid9_sincosTest_a) + SIGNED(absAE_uid9_sincosTest_b));
        ELSE
            absAE_uid9_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(absAE_uid9_sincosTest_a) - SIGNED(absAE_uid9_sincosTest_b));
        END IF;
    END PROCESS;
    absAE_uid9_sincosTest_q <= absAE_uid9_sincosTest_o(56 downto 0);

    -- absAR_uid10_sincosTest(BITSELECT,9)@0
    absAR_uid10_sincosTest_in <= absAE_uid9_sincosTest_q(54 downto 0);
    absAR_uid10_sincosTest_b <= absAR_uid10_sincosTest_in(54 downto 0);

    -- redist181_absAR_uid10_sincosTest_b_1(DELAY,1218)
    redist181_absAR_uid10_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 55, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => absAR_uid10_sincosTest_b, xout => redist181_absAR_uid10_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- padACst_uid12_sincosTest(CONSTANT,11)
    padACst_uid12_sincosTest_q <= "000000000000";

    -- aPostPad_uid13_sincosTest(BITJOIN,12)@1
    aPostPad_uid13_sincosTest_q <= redist181_absAR_uid10_sincosTest_b_1_q & padACst_uid12_sincosTest_q;

    -- argMPiO2_uid14_sincosTest(SUB,13)@1
    argMPiO2_uid14_sincosTest_a <= STD_LOGIC_VECTOR("0" & aPostPad_uid13_sincosTest_q);
    argMPiO2_uid14_sincosTest_b <= STD_LOGIC_VECTOR("00" & cstPiO2_uid11_sincosTest_q);
    argMPiO2_uid14_sincosTest_o <= STD_LOGIC_VECTOR(UNSIGNED(argMPiO2_uid14_sincosTest_a) - UNSIGNED(argMPiO2_uid14_sincosTest_b));
    argMPiO2_uid14_sincosTest_q <= argMPiO2_uid14_sincosTest_o(67 downto 0);

    -- firstQuadrant_uid15_sincosTest(BITSELECT,14)@1
    firstQuadrant_uid15_sincosTest_b <= STD_LOGIC_VECTOR(argMPiO2_uid14_sincosTest_q(67 downto 67));

    -- invFirstQuadrant_uid1022_sincosTest(LOGICAL,1021)@1
    invFirstQuadrant_uid1022_sincosTest_q <= not (firstQuadrant_uid15_sincosTest_b);

    -- redist183_signA_uid7_sincosTest_b_1(DELAY,1220)
    redist183_signA_uid7_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => signA_uid7_sincosTest_b, xout => redist183_signA_uid7_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- sinNegCond2_uid1023_sincosTest(LOGICAL,1022)@1
    sinNegCond2_uid1023_sincosTest_q <= redist183_signA_uid7_sincosTest_b_1_q and invFirstQuadrant_uid1022_sincosTest_q;

    -- sinNegCond1_uid1024_sincosTest(LOGICAL,1023)@1
    sinNegCond1_uid1024_sincosTest_q <= redist183_signA_uid7_sincosTest_b_1_q and firstQuadrant_uid15_sincosTest_b;

    -- redist182_invSignA_uid8_sincosTest_q_1(DELAY,1219)
    redist182_invSignA_uid8_sincosTest_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => invSignA_uid8_sincosTest_q, xout => redist182_invSignA_uid8_sincosTest_q_1_q, clk => clk, aclr => areset );

    -- sinNegCond0_uid1026_sincosTest(LOGICAL,1025)@1
    sinNegCond0_uid1026_sincosTest_q <= redist182_invSignA_uid8_sincosTest_q_1_q and invFirstQuadrant_uid1022_sincosTest_q;

    -- sinNegCond_uid1027_sincosTest(LOGICAL,1026)@1 + 1
    sinNegCond_uid1027_sincosTest_qi <= sinNegCond0_uid1026_sincosTest_q or sinNegCond1_uid1024_sincosTest_q or sinNegCond2_uid1023_sincosTest_q;
    sinNegCond_uid1027_sincosTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => sinNegCond_uid1027_sincosTest_qi, xout => sinNegCond_uid1027_sincosTest_q, clk => clk, aclr => areset );

    -- redist1_sinNegCond_uid1027_sincosTest_q_57_wraddr(REG,1235)
    redist1_sinNegCond_uid1027_sincosTest_q_57_wraddr_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            redist1_sinNegCond_uid1027_sincosTest_q_57_wraddr_q <= "110101";
        ELSIF (clk'EVENT AND clk = '1') THEN
            redist1_sinNegCond_uid1027_sincosTest_q_57_wraddr_q <= STD_LOGIC_VECTOR(redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_q);
        END IF;
    END PROCESS;

    -- redist1_sinNegCond_uid1027_sincosTest_q_57_mem(DUALMEM,1233)
    redist1_sinNegCond_uid1027_sincosTest_q_57_mem_ia <= STD_LOGIC_VECTOR(sinNegCond_uid1027_sincosTest_q);
    redist1_sinNegCond_uid1027_sincosTest_q_57_mem_aa <= redist1_sinNegCond_uid1027_sincosTest_q_57_wraddr_q;
    redist1_sinNegCond_uid1027_sincosTest_q_57_mem_ab <= redist1_sinNegCond_uid1027_sincosTest_q_57_rdcnt_q;
    redist1_sinNegCond_uid1027_sincosTest_q_57_mem_reset0 <= areset;
    redist1_sinNegCond_uid1027_sincosTest_q_57_mem_dmem : altera_syncram
    GENERIC MAP (
        ram_block_type => "MLAB",
        operation_mode => "DUAL_PORT",
        width_a => 1,
        widthad_a => 6,
        numwords_a => 54,
        width_b => 1,
        widthad_b => 6,
        numwords_b => 54,
        lpm_type => "altera_syncram",
        width_byteena_a => 1,
        address_reg_b => "CLOCK0",
        indata_reg_b => "CLOCK0",
        rdcontrol_reg_b => "CLOCK0",
        byteena_reg_b => "CLOCK0",
        outdata_reg_b => "CLOCK1",
        outdata_aclr_b => "CLEAR1",
        clock_enable_input_a => "NORMAL",
        clock_enable_input_b => "NORMAL",
        clock_enable_output_b => "NORMAL",
        read_during_write_mode_mixed_ports => "DONT_CARE",
        power_up_uninitialized => "TRUE",
        intended_device_family => "Stratix V"
    )
    PORT MAP (
        clocken1 => redist1_sinNegCond_uid1027_sincosTest_q_57_enaAnd_q(0),
        clocken0 => VCC_q(0),
        clock0 => clk,
        aclr1 => redist1_sinNegCond_uid1027_sincosTest_q_57_mem_reset0,
        clock1 => clk,
        address_a => redist1_sinNegCond_uid1027_sincosTest_q_57_mem_aa,
        data_a => redist1_sinNegCond_uid1027_sincosTest_q_57_mem_ia,
        wren_a => VCC_q(0),
        address_b => redist1_sinNegCond_uid1027_sincosTest_q_57_mem_ab,
        q_b => redist1_sinNegCond_uid1027_sincosTest_q_57_mem_iq
    );
    redist1_sinNegCond_uid1027_sincosTest_q_57_mem_q <= redist1_sinNegCond_uid1027_sincosTest_q_57_mem_iq(0 downto 0);

    -- redist1_sinNegCond_uid1027_sincosTest_q_57_outputreg(DELAY,1232)
    redist1_sinNegCond_uid1027_sincosTest_q_57_outputreg : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist1_sinNegCond_uid1027_sincosTest_q_57_mem_q, xout => redist1_sinNegCond_uid1027_sincosTest_q_57_outputreg_q, clk => clk, aclr => areset );

    -- invSinNegCond_uid1030_sincosTest(LOGICAL,1029)@58
    invSinNegCond_uid1030_sincosTest_q <= not (redist1_sinNegCond_uid1027_sincosTest_q_57_outputreg_q);

    -- GND(CONSTANT,0)
    GND_q <= "0";

    -- xMSB_uid369_sincosTest(BITSELECT,368)@23
    xMSB_uid369_sincosTest_b <= STD_LOGIC_VECTOR(redist119_aip1E_uid368_sincosTest_b_1_q(46 downto 46));

    -- cstArcTan2Mi_21_uid376_sincosTest(CONSTANT,375)
    cstArcTan2Mi_21_uid376_sincosTest_q <= "011111111111111111111111111111111111111111111";

    -- xMSB_uid353_sincosTest(BITSELECT,352)@22
    xMSB_uid353_sincosTest_b <= STD_LOGIC_VECTOR(redist122_aip1E_uid352_sincosTest_b_1_q(47 downto 47));

    -- cstArcTan2Mi_20_uid360_sincosTest(CONSTANT,359)
    cstArcTan2Mi_20_uid360_sincosTest_q <= "0111111111111111111111111111111111111111110101";

    -- xMSB_uid337_sincosTest(BITSELECT,336)@21
    xMSB_uid337_sincosTest_b <= STD_LOGIC_VECTOR(redist125_aip1E_uid336_sincosTest_b_1_q(48 downto 48));

    -- cstArcTan2Mi_19_uid344_sincosTest(CONSTANT,343)
    cstArcTan2Mi_19_uid344_sincosTest_q <= "01111111111111111111111111111111111111110101011";

    -- xMSB_uid321_sincosTest(BITSELECT,320)@20
    xMSB_uid321_sincosTest_b <= STD_LOGIC_VECTOR(redist128_aip1E_uid320_sincosTest_b_1_q(49 downto 49));

    -- cstArcTan2Mi_18_uid328_sincosTest(CONSTANT,327)
    cstArcTan2Mi_18_uid328_sincosTest_q <= "011111111111111111111111111111111111110101010101";

    -- xMSB_uid305_sincosTest(BITSELECT,304)@19
    xMSB_uid305_sincosTest_b <= STD_LOGIC_VECTOR(redist131_aip1E_uid304_sincosTest_b_1_q(50 downto 50));

    -- cstArcTan2Mi_17_uid312_sincosTest(CONSTANT,311)
    cstArcTan2Mi_17_uid312_sincosTest_q <= "0111111111111111111111111111111111110101010101011";

    -- xMSB_uid289_sincosTest(BITSELECT,288)@18
    xMSB_uid289_sincosTest_b <= STD_LOGIC_VECTOR(redist134_aip1E_uid288_sincosTest_b_1_q(51 downto 51));

    -- cstArcTan2Mi_16_uid296_sincosTest(CONSTANT,295)
    cstArcTan2Mi_16_uid296_sincosTest_q <= "01111111111111111111111111111111110101010101010101";

    -- xMSB_uid273_sincosTest(BITSELECT,272)@17
    xMSB_uid273_sincosTest_b <= STD_LOGIC_VECTOR(redist137_aip1E_uid272_sincosTest_b_1_q(52 downto 52));

    -- cstArcTan2Mi_15_uid280_sincosTest(CONSTANT,279)
    cstArcTan2Mi_15_uid280_sincosTest_q <= "011111111111111111111111111111110101010101010101011";

    -- xMSB_uid257_sincosTest(BITSELECT,256)@16
    xMSB_uid257_sincosTest_b <= STD_LOGIC_VECTOR(redist140_aip1E_uid256_sincosTest_b_1_q(53 downto 53));

    -- cstArcTan2Mi_14_uid264_sincosTest(CONSTANT,263)
    cstArcTan2Mi_14_uid264_sincosTest_q <= "0111111111111111111111111111110101010101010101010101";

    -- xMSB_uid241_sincosTest(BITSELECT,240)@15
    xMSB_uid241_sincosTest_b <= STD_LOGIC_VECTOR(redist143_aip1E_uid240_sincosTest_b_1_q(54 downto 54));

    -- cstArcTan2Mi_13_uid248_sincosTest(CONSTANT,247)
    cstArcTan2Mi_13_uid248_sincosTest_q <= "01111111111111111111111111110101010101010101010101011";

    -- xMSB_uid225_sincosTest(BITSELECT,224)@14
    xMSB_uid225_sincosTest_b <= STD_LOGIC_VECTOR(redist146_aip1E_uid224_sincosTest_b_1_q(55 downto 55));

    -- cstArcTan2Mi_12_uid232_sincosTest(CONSTANT,231)
    cstArcTan2Mi_12_uid232_sincosTest_q <= "011111111111111111111111110101010101010101010101011100";

    -- xMSB_uid209_sincosTest(BITSELECT,208)@13
    xMSB_uid209_sincosTest_b <= STD_LOGIC_VECTOR(redist149_aip1E_uid208_sincosTest_b_1_q(56 downto 56));

    -- cstArcTan2Mi_11_uid216_sincosTest(CONSTANT,215)
    cstArcTan2Mi_11_uid216_sincosTest_q <= "0111111111111111111111110101010101010101010101101110111";

    -- xMSB_uid193_sincosTest(BITSELECT,192)@12
    xMSB_uid193_sincosTest_b <= STD_LOGIC_VECTOR(redist152_aip1E_uid192_sincosTest_b_1_q(57 downto 57));

    -- cstArcTan2Mi_10_uid200_sincosTest(CONSTANT,199)
    cstArcTan2Mi_10_uid200_sincosTest_q <= "01111111111111111111110101010101010101010110111011101111";

    -- xMSB_uid177_sincosTest(BITSELECT,176)@11
    xMSB_uid177_sincosTest_b <= STD_LOGIC_VECTOR(redist155_aip1E_uid176_sincosTest_b_1_q(58 downto 58));

    -- cstArcTan2Mi_9_uid184_sincosTest(CONSTANT,183)
    cstArcTan2Mi_9_uid184_sincosTest_q <= "011111111111111111110101010101010101011011101110111011101";

    -- xMSB_uid161_sincosTest(BITSELECT,160)@10
    xMSB_uid161_sincosTest_b <= STD_LOGIC_VECTOR(redist158_aip1E_uid160_sincosTest_b_1_q(59 downto 59));

    -- cstArcTan2Mi_8_uid168_sincosTest(CONSTANT,167)
    cstArcTan2Mi_8_uid168_sincosTest_q <= "0111111111111111110101010101010101101110111011101101110011";

    -- xMSB_uid145_sincosTest(BITSELECT,144)@9
    xMSB_uid145_sincosTest_b <= STD_LOGIC_VECTOR(redist161_aip1E_uid144_sincosTest_b_1_q(60 downto 60));

    -- cstArcTan2Mi_7_uid152_sincosTest(CONSTANT,151)
    cstArcTan2Mi_7_uid152_sincosTest_q <= "01111111111111110101010101010110111011101110101001011100110";

    -- xMSB_uid129_sincosTest(BITSELECT,128)@8
    xMSB_uid129_sincosTest_b <= STD_LOGIC_VECTOR(redist164_aip1E_uid128_sincosTest_b_1_q(61 downto 61));

    -- cstArcTan2Mi_6_uid136_sincosTest(CONSTANT,135)
    cstArcTan2Mi_6_uid136_sincosTest_q <= "011111111111110101010101011011101110110111001010011010101110";

    -- xMSB_uid113_sincosTest(BITSELECT,112)@7
    xMSB_uid113_sincosTest_b <= STD_LOGIC_VECTOR(redist167_aip1E_uid112_sincosTest_b_1_q(62 downto 62));

    -- cstArcTan2Mi_5_uid120_sincosTest(CONSTANT,119)
    cstArcTan2Mi_5_uid120_sincosTest_q <= "0111111111110101010101101110111010100101110110001001001010100";

    -- xMSB_uid97_sincosTest(BITSELECT,96)@6
    xMSB_uid97_sincosTest_b <= STD_LOGIC_VECTOR(redist170_aip1E_uid96_sincosTest_b_1_q(63 downto 63));

    -- cstArcTan2Mi_4_uid104_sincosTest(CONSTANT,103)
    cstArcTan2Mi_4_uid104_sincosTest_q <= "01111111110101010110111011011100101100111111011110100111000111";

    -- xMSB_uid81_sincosTest(BITSELECT,80)@5
    xMSB_uid81_sincosTest_b <= STD_LOGIC_VECTOR(redist173_aip1E_uid80_sincosTest_b_1_q(64 downto 64));

    -- cstArcTan2Mi_3_uid88_sincosTest(CONSTANT,87)
    cstArcTan2Mi_3_uid88_sincosTest_q <= "011111110101011011101010011010101011000010111101101101110001101";

    -- xMSB_uid65_sincosTest(BITSELECT,64)@4
    xMSB_uid65_sincosTest_b <= STD_LOGIC_VECTOR(redist176_aip1E_uid64_sincosTest_b_1_q(65 downto 65));

    -- cstArcTan2Mi_2_uid72_sincosTest(CONSTANT,71)
    cstArcTan2Mi_2_uid72_sincosTest_q <= "0111110101101101110101111110010010110010000000110111010110001011";

    -- xMSB_uid49_sincosTest(BITSELECT,48)@3
    xMSB_uid49_sincosTest_b <= STD_LOGIC_VECTOR(redist179_aip1E_uid48_sincosTest_b_1_q(66 downto 66));

    -- cstArcTan2Mi_1_uid56_sincosTest(CONSTANT,55)
    cstArcTan2Mi_1_uid56_sincosTest_q <= "01110110101100011001110000010101100001101110110100111101101000101";

    -- invSignOfSelectionSignal_uid36_sincosTest(LOGICAL,35)@2
    invSignOfSelectionSignal_uid36_sincosTest_q <= not (VCC_q);

    -- cstArcTan2Mi_0_uid26_sincosTest(CONSTANT,25)
    cstArcTan2Mi_0_uid26_sincosTest_q <= "011001001000011111101101010100010001000010110100011000010001101010";

    -- absARE_bottomRange_uid17_sincosTest(BITSELECT,16)@1
    absARE_bottomRange_uid17_sincosTest_in <= redist181_absAR_uid10_sincosTest_b_1_q(53 downto 0);
    absARE_bottomRange_uid17_sincosTest_b <= absARE_bottomRange_uid17_sincosTest_in(53 downto 0);

    -- absARE_mergedSignalTM_uid18_sincosTest(BITJOIN,17)@1
    absARE_mergedSignalTM_uid18_sincosTest_q <= absARE_bottomRange_uid17_sincosTest_b & padACst_uid12_sincosTest_q;

    -- argMPiO2_uid20_sincosTest(BITSELECT,19)@1
    argMPiO2_uid20_sincosTest_in <= argMPiO2_uid14_sincosTest_q(65 downto 0);
    argMPiO2_uid20_sincosTest_b <= argMPiO2_uid20_sincosTest_in(65 downto 0);

    -- absA_uid21_sincosTest(MUX,20)@1 + 1
    absA_uid21_sincosTest_s <= firstQuadrant_uid15_sincosTest_b;
    absA_uid21_sincosTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            absA_uid21_sincosTest_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (absA_uid21_sincosTest_s) IS
                WHEN "0" => absA_uid21_sincosTest_q <= argMPiO2_uid20_sincosTest_b;
                WHEN "1" => absA_uid21_sincosTest_q <= absARE_mergedSignalTM_uid18_sincosTest_q;
                WHEN OTHERS => absA_uid21_sincosTest_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- aip1E_1_uid37_sincosTest(ADDSUB,36)@2
    aip1E_1_uid37_sincosTest_s <= invSignOfSelectionSignal_uid36_sincosTest_q;
    aip1E_1_uid37_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("000" & absA_uid21_sincosTest_q));
    aip1E_1_uid37_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((68 downto 66 => cstArcTan2Mi_0_uid26_sincosTest_q(65)) & cstArcTan2Mi_0_uid26_sincosTest_q));
    aip1E_1_uid37_sincosTest_combproc: PROCESS (aip1E_1_uid37_sincosTest_a, aip1E_1_uid37_sincosTest_b, aip1E_1_uid37_sincosTest_s)
    BEGIN
        IF (aip1E_1_uid37_sincosTest_s = "1") THEN
            aip1E_1_uid37_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_1_uid37_sincosTest_a) + SIGNED(aip1E_1_uid37_sincosTest_b));
        ELSE
            aip1E_1_uid37_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_1_uid37_sincosTest_a) - SIGNED(aip1E_1_uid37_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_1_uid37_sincosTest_q <= aip1E_1_uid37_sincosTest_o(67 downto 0);

    -- aip1E_uid48_sincosTest(BITSELECT,47)@2
    aip1E_uid48_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_1_uid37_sincosTest_q(66 downto 0));
    aip1E_uid48_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid48_sincosTest_in(66 downto 0));

    -- redist179_aip1E_uid48_sincosTest_b_1(DELAY,1216)
    redist179_aip1E_uid48_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 67, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid48_sincosTest_b, xout => redist179_aip1E_uid48_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_2_uid61_sincosTest(ADDSUB,60)@3
    aip1E_2_uid61_sincosTest_s <= xMSB_uid49_sincosTest_b;
    aip1E_2_uid61_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((68 downto 67 => redist179_aip1E_uid48_sincosTest_b_1_q(66)) & redist179_aip1E_uid48_sincosTest_b_1_q));
    aip1E_2_uid61_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((68 downto 65 => cstArcTan2Mi_1_uid56_sincosTest_q(64)) & cstArcTan2Mi_1_uid56_sincosTest_q));
    aip1E_2_uid61_sincosTest_combproc: PROCESS (aip1E_2_uid61_sincosTest_a, aip1E_2_uid61_sincosTest_b, aip1E_2_uid61_sincosTest_s)
    BEGIN
        IF (aip1E_2_uid61_sincosTest_s = "1") THEN
            aip1E_2_uid61_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_2_uid61_sincosTest_a) + SIGNED(aip1E_2_uid61_sincosTest_b));
        ELSE
            aip1E_2_uid61_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_2_uid61_sincosTest_a) - SIGNED(aip1E_2_uid61_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_2_uid61_sincosTest_q <= aip1E_2_uid61_sincosTest_o(67 downto 0);

    -- aip1E_uid64_sincosTest(BITSELECT,63)@3
    aip1E_uid64_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_2_uid61_sincosTest_q(65 downto 0));
    aip1E_uid64_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid64_sincosTest_in(65 downto 0));

    -- redist176_aip1E_uid64_sincosTest_b_1(DELAY,1213)
    redist176_aip1E_uid64_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 66, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid64_sincosTest_b, xout => redist176_aip1E_uid64_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_3_uid77_sincosTest(ADDSUB,76)@4
    aip1E_3_uid77_sincosTest_s <= xMSB_uid65_sincosTest_b;
    aip1E_3_uid77_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((67 downto 66 => redist176_aip1E_uid64_sincosTest_b_1_q(65)) & redist176_aip1E_uid64_sincosTest_b_1_q));
    aip1E_3_uid77_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((67 downto 64 => cstArcTan2Mi_2_uid72_sincosTest_q(63)) & cstArcTan2Mi_2_uid72_sincosTest_q));
    aip1E_3_uid77_sincosTest_combproc: PROCESS (aip1E_3_uid77_sincosTest_a, aip1E_3_uid77_sincosTest_b, aip1E_3_uid77_sincosTest_s)
    BEGIN
        IF (aip1E_3_uid77_sincosTest_s = "1") THEN
            aip1E_3_uid77_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_3_uid77_sincosTest_a) + SIGNED(aip1E_3_uid77_sincosTest_b));
        ELSE
            aip1E_3_uid77_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_3_uid77_sincosTest_a) - SIGNED(aip1E_3_uid77_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_3_uid77_sincosTest_q <= aip1E_3_uid77_sincosTest_o(66 downto 0);

    -- aip1E_uid80_sincosTest(BITSELECT,79)@4
    aip1E_uid80_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_3_uid77_sincosTest_q(64 downto 0));
    aip1E_uid80_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid80_sincosTest_in(64 downto 0));

    -- redist173_aip1E_uid80_sincosTest_b_1(DELAY,1210)
    redist173_aip1E_uid80_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 65, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid80_sincosTest_b, xout => redist173_aip1E_uid80_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_4_uid93_sincosTest(ADDSUB,92)@5
    aip1E_4_uid93_sincosTest_s <= xMSB_uid81_sincosTest_b;
    aip1E_4_uid93_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((66 downto 65 => redist173_aip1E_uid80_sincosTest_b_1_q(64)) & redist173_aip1E_uid80_sincosTest_b_1_q));
    aip1E_4_uid93_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((66 downto 63 => cstArcTan2Mi_3_uid88_sincosTest_q(62)) & cstArcTan2Mi_3_uid88_sincosTest_q));
    aip1E_4_uid93_sincosTest_combproc: PROCESS (aip1E_4_uid93_sincosTest_a, aip1E_4_uid93_sincosTest_b, aip1E_4_uid93_sincosTest_s)
    BEGIN
        IF (aip1E_4_uid93_sincosTest_s = "1") THEN
            aip1E_4_uid93_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_4_uid93_sincosTest_a) + SIGNED(aip1E_4_uid93_sincosTest_b));
        ELSE
            aip1E_4_uid93_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_4_uid93_sincosTest_a) - SIGNED(aip1E_4_uid93_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_4_uid93_sincosTest_q <= aip1E_4_uid93_sincosTest_o(65 downto 0);

    -- aip1E_uid96_sincosTest(BITSELECT,95)@5
    aip1E_uid96_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_4_uid93_sincosTest_q(63 downto 0));
    aip1E_uid96_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid96_sincosTest_in(63 downto 0));

    -- redist170_aip1E_uid96_sincosTest_b_1(DELAY,1207)
    redist170_aip1E_uid96_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 64, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid96_sincosTest_b, xout => redist170_aip1E_uid96_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_5_uid109_sincosTest(ADDSUB,108)@6
    aip1E_5_uid109_sincosTest_s <= xMSB_uid97_sincosTest_b;
    aip1E_5_uid109_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((65 downto 64 => redist170_aip1E_uid96_sincosTest_b_1_q(63)) & redist170_aip1E_uid96_sincosTest_b_1_q));
    aip1E_5_uid109_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((65 downto 62 => cstArcTan2Mi_4_uid104_sincosTest_q(61)) & cstArcTan2Mi_4_uid104_sincosTest_q));
    aip1E_5_uid109_sincosTest_combproc: PROCESS (aip1E_5_uid109_sincosTest_a, aip1E_5_uid109_sincosTest_b, aip1E_5_uid109_sincosTest_s)
    BEGIN
        IF (aip1E_5_uid109_sincosTest_s = "1") THEN
            aip1E_5_uid109_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_5_uid109_sincosTest_a) + SIGNED(aip1E_5_uid109_sincosTest_b));
        ELSE
            aip1E_5_uid109_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_5_uid109_sincosTest_a) - SIGNED(aip1E_5_uid109_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_5_uid109_sincosTest_q <= aip1E_5_uid109_sincosTest_o(64 downto 0);

    -- aip1E_uid112_sincosTest(BITSELECT,111)@6
    aip1E_uid112_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_5_uid109_sincosTest_q(62 downto 0));
    aip1E_uid112_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid112_sincosTest_in(62 downto 0));

    -- redist167_aip1E_uid112_sincosTest_b_1(DELAY,1204)
    redist167_aip1E_uid112_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 63, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid112_sincosTest_b, xout => redist167_aip1E_uid112_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_6_uid125_sincosTest(ADDSUB,124)@7
    aip1E_6_uid125_sincosTest_s <= xMSB_uid113_sincosTest_b;
    aip1E_6_uid125_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((64 downto 63 => redist167_aip1E_uid112_sincosTest_b_1_q(62)) & redist167_aip1E_uid112_sincosTest_b_1_q));
    aip1E_6_uid125_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((64 downto 61 => cstArcTan2Mi_5_uid120_sincosTest_q(60)) & cstArcTan2Mi_5_uid120_sincosTest_q));
    aip1E_6_uid125_sincosTest_combproc: PROCESS (aip1E_6_uid125_sincosTest_a, aip1E_6_uid125_sincosTest_b, aip1E_6_uid125_sincosTest_s)
    BEGIN
        IF (aip1E_6_uid125_sincosTest_s = "1") THEN
            aip1E_6_uid125_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_6_uid125_sincosTest_a) + SIGNED(aip1E_6_uid125_sincosTest_b));
        ELSE
            aip1E_6_uid125_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_6_uid125_sincosTest_a) - SIGNED(aip1E_6_uid125_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_6_uid125_sincosTest_q <= aip1E_6_uid125_sincosTest_o(63 downto 0);

    -- aip1E_uid128_sincosTest(BITSELECT,127)@7
    aip1E_uid128_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_6_uid125_sincosTest_q(61 downto 0));
    aip1E_uid128_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid128_sincosTest_in(61 downto 0));

    -- redist164_aip1E_uid128_sincosTest_b_1(DELAY,1201)
    redist164_aip1E_uid128_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 62, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid128_sincosTest_b, xout => redist164_aip1E_uid128_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_7_uid141_sincosTest(ADDSUB,140)@8
    aip1E_7_uid141_sincosTest_s <= xMSB_uid129_sincosTest_b;
    aip1E_7_uid141_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((63 downto 62 => redist164_aip1E_uid128_sincosTest_b_1_q(61)) & redist164_aip1E_uid128_sincosTest_b_1_q));
    aip1E_7_uid141_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((63 downto 60 => cstArcTan2Mi_6_uid136_sincosTest_q(59)) & cstArcTan2Mi_6_uid136_sincosTest_q));
    aip1E_7_uid141_sincosTest_combproc: PROCESS (aip1E_7_uid141_sincosTest_a, aip1E_7_uid141_sincosTest_b, aip1E_7_uid141_sincosTest_s)
    BEGIN
        IF (aip1E_7_uid141_sincosTest_s = "1") THEN
            aip1E_7_uid141_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_7_uid141_sincosTest_a) + SIGNED(aip1E_7_uid141_sincosTest_b));
        ELSE
            aip1E_7_uid141_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_7_uid141_sincosTest_a) - SIGNED(aip1E_7_uid141_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_7_uid141_sincosTest_q <= aip1E_7_uid141_sincosTest_o(62 downto 0);

    -- aip1E_uid144_sincosTest(BITSELECT,143)@8
    aip1E_uid144_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_7_uid141_sincosTest_q(60 downto 0));
    aip1E_uid144_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid144_sincosTest_in(60 downto 0));

    -- redist161_aip1E_uid144_sincosTest_b_1(DELAY,1198)
    redist161_aip1E_uid144_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 61, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid144_sincosTest_b, xout => redist161_aip1E_uid144_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_8_uid157_sincosTest(ADDSUB,156)@9
    aip1E_8_uid157_sincosTest_s <= xMSB_uid145_sincosTest_b;
    aip1E_8_uid157_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((62 downto 61 => redist161_aip1E_uid144_sincosTest_b_1_q(60)) & redist161_aip1E_uid144_sincosTest_b_1_q));
    aip1E_8_uid157_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((62 downto 59 => cstArcTan2Mi_7_uid152_sincosTest_q(58)) & cstArcTan2Mi_7_uid152_sincosTest_q));
    aip1E_8_uid157_sincosTest_combproc: PROCESS (aip1E_8_uid157_sincosTest_a, aip1E_8_uid157_sincosTest_b, aip1E_8_uid157_sincosTest_s)
    BEGIN
        IF (aip1E_8_uid157_sincosTest_s = "1") THEN
            aip1E_8_uid157_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_8_uid157_sincosTest_a) + SIGNED(aip1E_8_uid157_sincosTest_b));
        ELSE
            aip1E_8_uid157_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_8_uid157_sincosTest_a) - SIGNED(aip1E_8_uid157_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_8_uid157_sincosTest_q <= aip1E_8_uid157_sincosTest_o(61 downto 0);

    -- aip1E_uid160_sincosTest(BITSELECT,159)@9
    aip1E_uid160_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_8_uid157_sincosTest_q(59 downto 0));
    aip1E_uid160_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid160_sincosTest_in(59 downto 0));

    -- redist158_aip1E_uid160_sincosTest_b_1(DELAY,1195)
    redist158_aip1E_uid160_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 60, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid160_sincosTest_b, xout => redist158_aip1E_uid160_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_9_uid173_sincosTest(ADDSUB,172)@10
    aip1E_9_uid173_sincosTest_s <= xMSB_uid161_sincosTest_b;
    aip1E_9_uid173_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((61 downto 60 => redist158_aip1E_uid160_sincosTest_b_1_q(59)) & redist158_aip1E_uid160_sincosTest_b_1_q));
    aip1E_9_uid173_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((61 downto 58 => cstArcTan2Mi_8_uid168_sincosTest_q(57)) & cstArcTan2Mi_8_uid168_sincosTest_q));
    aip1E_9_uid173_sincosTest_combproc: PROCESS (aip1E_9_uid173_sincosTest_a, aip1E_9_uid173_sincosTest_b, aip1E_9_uid173_sincosTest_s)
    BEGIN
        IF (aip1E_9_uid173_sincosTest_s = "1") THEN
            aip1E_9_uid173_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_9_uid173_sincosTest_a) + SIGNED(aip1E_9_uid173_sincosTest_b));
        ELSE
            aip1E_9_uid173_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_9_uid173_sincosTest_a) - SIGNED(aip1E_9_uid173_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_9_uid173_sincosTest_q <= aip1E_9_uid173_sincosTest_o(60 downto 0);

    -- aip1E_uid176_sincosTest(BITSELECT,175)@10
    aip1E_uid176_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_9_uid173_sincosTest_q(58 downto 0));
    aip1E_uid176_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid176_sincosTest_in(58 downto 0));

    -- redist155_aip1E_uid176_sincosTest_b_1(DELAY,1192)
    redist155_aip1E_uid176_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 59, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid176_sincosTest_b, xout => redist155_aip1E_uid176_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_10_uid189_sincosTest(ADDSUB,188)@11
    aip1E_10_uid189_sincosTest_s <= xMSB_uid177_sincosTest_b;
    aip1E_10_uid189_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((60 downto 59 => redist155_aip1E_uid176_sincosTest_b_1_q(58)) & redist155_aip1E_uid176_sincosTest_b_1_q));
    aip1E_10_uid189_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((60 downto 57 => cstArcTan2Mi_9_uid184_sincosTest_q(56)) & cstArcTan2Mi_9_uid184_sincosTest_q));
    aip1E_10_uid189_sincosTest_combproc: PROCESS (aip1E_10_uid189_sincosTest_a, aip1E_10_uid189_sincosTest_b, aip1E_10_uid189_sincosTest_s)
    BEGIN
        IF (aip1E_10_uid189_sincosTest_s = "1") THEN
            aip1E_10_uid189_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_10_uid189_sincosTest_a) + SIGNED(aip1E_10_uid189_sincosTest_b));
        ELSE
            aip1E_10_uid189_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_10_uid189_sincosTest_a) - SIGNED(aip1E_10_uid189_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_10_uid189_sincosTest_q <= aip1E_10_uid189_sincosTest_o(59 downto 0);

    -- aip1E_uid192_sincosTest(BITSELECT,191)@11
    aip1E_uid192_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_10_uid189_sincosTest_q(57 downto 0));
    aip1E_uid192_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid192_sincosTest_in(57 downto 0));

    -- redist152_aip1E_uid192_sincosTest_b_1(DELAY,1189)
    redist152_aip1E_uid192_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 58, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid192_sincosTest_b, xout => redist152_aip1E_uid192_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_11_uid205_sincosTest(ADDSUB,204)@12
    aip1E_11_uid205_sincosTest_s <= xMSB_uid193_sincosTest_b;
    aip1E_11_uid205_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((59 downto 58 => redist152_aip1E_uid192_sincosTest_b_1_q(57)) & redist152_aip1E_uid192_sincosTest_b_1_q));
    aip1E_11_uid205_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((59 downto 56 => cstArcTan2Mi_10_uid200_sincosTest_q(55)) & cstArcTan2Mi_10_uid200_sincosTest_q));
    aip1E_11_uid205_sincosTest_combproc: PROCESS (aip1E_11_uid205_sincosTest_a, aip1E_11_uid205_sincosTest_b, aip1E_11_uid205_sincosTest_s)
    BEGIN
        IF (aip1E_11_uid205_sincosTest_s = "1") THEN
            aip1E_11_uid205_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_11_uid205_sincosTest_a) + SIGNED(aip1E_11_uid205_sincosTest_b));
        ELSE
            aip1E_11_uid205_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_11_uid205_sincosTest_a) - SIGNED(aip1E_11_uid205_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_11_uid205_sincosTest_q <= aip1E_11_uid205_sincosTest_o(58 downto 0);

    -- aip1E_uid208_sincosTest(BITSELECT,207)@12
    aip1E_uid208_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_11_uid205_sincosTest_q(56 downto 0));
    aip1E_uid208_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid208_sincosTest_in(56 downto 0));

    -- redist149_aip1E_uid208_sincosTest_b_1(DELAY,1186)
    redist149_aip1E_uid208_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 57, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid208_sincosTest_b, xout => redist149_aip1E_uid208_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_12_uid221_sincosTest(ADDSUB,220)@13
    aip1E_12_uid221_sincosTest_s <= xMSB_uid209_sincosTest_b;
    aip1E_12_uid221_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((58 downto 57 => redist149_aip1E_uid208_sincosTest_b_1_q(56)) & redist149_aip1E_uid208_sincosTest_b_1_q));
    aip1E_12_uid221_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((58 downto 55 => cstArcTan2Mi_11_uid216_sincosTest_q(54)) & cstArcTan2Mi_11_uid216_sincosTest_q));
    aip1E_12_uid221_sincosTest_combproc: PROCESS (aip1E_12_uid221_sincosTest_a, aip1E_12_uid221_sincosTest_b, aip1E_12_uid221_sincosTest_s)
    BEGIN
        IF (aip1E_12_uid221_sincosTest_s = "1") THEN
            aip1E_12_uid221_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_12_uid221_sincosTest_a) + SIGNED(aip1E_12_uid221_sincosTest_b));
        ELSE
            aip1E_12_uid221_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_12_uid221_sincosTest_a) - SIGNED(aip1E_12_uid221_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_12_uid221_sincosTest_q <= aip1E_12_uid221_sincosTest_o(57 downto 0);

    -- aip1E_uid224_sincosTest(BITSELECT,223)@13
    aip1E_uid224_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_12_uid221_sincosTest_q(55 downto 0));
    aip1E_uid224_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid224_sincosTest_in(55 downto 0));

    -- redist146_aip1E_uid224_sincosTest_b_1(DELAY,1183)
    redist146_aip1E_uid224_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 56, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid224_sincosTest_b, xout => redist146_aip1E_uid224_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_13_uid237_sincosTest(ADDSUB,236)@14
    aip1E_13_uid237_sincosTest_s <= xMSB_uid225_sincosTest_b;
    aip1E_13_uid237_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((57 downto 56 => redist146_aip1E_uid224_sincosTest_b_1_q(55)) & redist146_aip1E_uid224_sincosTest_b_1_q));
    aip1E_13_uid237_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((57 downto 54 => cstArcTan2Mi_12_uid232_sincosTest_q(53)) & cstArcTan2Mi_12_uid232_sincosTest_q));
    aip1E_13_uid237_sincosTest_combproc: PROCESS (aip1E_13_uid237_sincosTest_a, aip1E_13_uid237_sincosTest_b, aip1E_13_uid237_sincosTest_s)
    BEGIN
        IF (aip1E_13_uid237_sincosTest_s = "1") THEN
            aip1E_13_uid237_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_13_uid237_sincosTest_a) + SIGNED(aip1E_13_uid237_sincosTest_b));
        ELSE
            aip1E_13_uid237_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_13_uid237_sincosTest_a) - SIGNED(aip1E_13_uid237_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_13_uid237_sincosTest_q <= aip1E_13_uid237_sincosTest_o(56 downto 0);

    -- aip1E_uid240_sincosTest(BITSELECT,239)@14
    aip1E_uid240_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_13_uid237_sincosTest_q(54 downto 0));
    aip1E_uid240_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid240_sincosTest_in(54 downto 0));

    -- redist143_aip1E_uid240_sincosTest_b_1(DELAY,1180)
    redist143_aip1E_uid240_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 55, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid240_sincosTest_b, xout => redist143_aip1E_uid240_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_14_uid253_sincosTest(ADDSUB,252)@15
    aip1E_14_uid253_sincosTest_s <= xMSB_uid241_sincosTest_b;
    aip1E_14_uid253_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((56 downto 55 => redist143_aip1E_uid240_sincosTest_b_1_q(54)) & redist143_aip1E_uid240_sincosTest_b_1_q));
    aip1E_14_uid253_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((56 downto 53 => cstArcTan2Mi_13_uid248_sincosTest_q(52)) & cstArcTan2Mi_13_uid248_sincosTest_q));
    aip1E_14_uid253_sincosTest_combproc: PROCESS (aip1E_14_uid253_sincosTest_a, aip1E_14_uid253_sincosTest_b, aip1E_14_uid253_sincosTest_s)
    BEGIN
        IF (aip1E_14_uid253_sincosTest_s = "1") THEN
            aip1E_14_uid253_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_14_uid253_sincosTest_a) + SIGNED(aip1E_14_uid253_sincosTest_b));
        ELSE
            aip1E_14_uid253_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_14_uid253_sincosTest_a) - SIGNED(aip1E_14_uid253_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_14_uid253_sincosTest_q <= aip1E_14_uid253_sincosTest_o(55 downto 0);

    -- aip1E_uid256_sincosTest(BITSELECT,255)@15
    aip1E_uid256_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_14_uid253_sincosTest_q(53 downto 0));
    aip1E_uid256_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid256_sincosTest_in(53 downto 0));

    -- redist140_aip1E_uid256_sincosTest_b_1(DELAY,1177)
    redist140_aip1E_uid256_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 54, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid256_sincosTest_b, xout => redist140_aip1E_uid256_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_15_uid269_sincosTest(ADDSUB,268)@16
    aip1E_15_uid269_sincosTest_s <= xMSB_uid257_sincosTest_b;
    aip1E_15_uid269_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((55 downto 54 => redist140_aip1E_uid256_sincosTest_b_1_q(53)) & redist140_aip1E_uid256_sincosTest_b_1_q));
    aip1E_15_uid269_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((55 downto 52 => cstArcTan2Mi_14_uid264_sincosTest_q(51)) & cstArcTan2Mi_14_uid264_sincosTest_q));
    aip1E_15_uid269_sincosTest_combproc: PROCESS (aip1E_15_uid269_sincosTest_a, aip1E_15_uid269_sincosTest_b, aip1E_15_uid269_sincosTest_s)
    BEGIN
        IF (aip1E_15_uid269_sincosTest_s = "1") THEN
            aip1E_15_uid269_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_15_uid269_sincosTest_a) + SIGNED(aip1E_15_uid269_sincosTest_b));
        ELSE
            aip1E_15_uid269_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_15_uid269_sincosTest_a) - SIGNED(aip1E_15_uid269_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_15_uid269_sincosTest_q <= aip1E_15_uid269_sincosTest_o(54 downto 0);

    -- aip1E_uid272_sincosTest(BITSELECT,271)@16
    aip1E_uid272_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_15_uid269_sincosTest_q(52 downto 0));
    aip1E_uid272_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid272_sincosTest_in(52 downto 0));

    -- redist137_aip1E_uid272_sincosTest_b_1(DELAY,1174)
    redist137_aip1E_uid272_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 53, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid272_sincosTest_b, xout => redist137_aip1E_uid272_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_16_uid285_sincosTest(ADDSUB,284)@17
    aip1E_16_uid285_sincosTest_s <= xMSB_uid273_sincosTest_b;
    aip1E_16_uid285_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((54 downto 53 => redist137_aip1E_uid272_sincosTest_b_1_q(52)) & redist137_aip1E_uid272_sincosTest_b_1_q));
    aip1E_16_uid285_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((54 downto 51 => cstArcTan2Mi_15_uid280_sincosTest_q(50)) & cstArcTan2Mi_15_uid280_sincosTest_q));
    aip1E_16_uid285_sincosTest_combproc: PROCESS (aip1E_16_uid285_sincosTest_a, aip1E_16_uid285_sincosTest_b, aip1E_16_uid285_sincosTest_s)
    BEGIN
        IF (aip1E_16_uid285_sincosTest_s = "1") THEN
            aip1E_16_uid285_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_16_uid285_sincosTest_a) + SIGNED(aip1E_16_uid285_sincosTest_b));
        ELSE
            aip1E_16_uid285_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_16_uid285_sincosTest_a) - SIGNED(aip1E_16_uid285_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_16_uid285_sincosTest_q <= aip1E_16_uid285_sincosTest_o(53 downto 0);

    -- aip1E_uid288_sincosTest(BITSELECT,287)@17
    aip1E_uid288_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_16_uid285_sincosTest_q(51 downto 0));
    aip1E_uid288_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid288_sincosTest_in(51 downto 0));

    -- redist134_aip1E_uid288_sincosTest_b_1(DELAY,1171)
    redist134_aip1E_uid288_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 52, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid288_sincosTest_b, xout => redist134_aip1E_uid288_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_17_uid301_sincosTest(ADDSUB,300)@18
    aip1E_17_uid301_sincosTest_s <= xMSB_uid289_sincosTest_b;
    aip1E_17_uid301_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((53 downto 52 => redist134_aip1E_uid288_sincosTest_b_1_q(51)) & redist134_aip1E_uid288_sincosTest_b_1_q));
    aip1E_17_uid301_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((53 downto 50 => cstArcTan2Mi_16_uid296_sincosTest_q(49)) & cstArcTan2Mi_16_uid296_sincosTest_q));
    aip1E_17_uid301_sincosTest_combproc: PROCESS (aip1E_17_uid301_sincosTest_a, aip1E_17_uid301_sincosTest_b, aip1E_17_uid301_sincosTest_s)
    BEGIN
        IF (aip1E_17_uid301_sincosTest_s = "1") THEN
            aip1E_17_uid301_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_17_uid301_sincosTest_a) + SIGNED(aip1E_17_uid301_sincosTest_b));
        ELSE
            aip1E_17_uid301_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_17_uid301_sincosTest_a) - SIGNED(aip1E_17_uid301_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_17_uid301_sincosTest_q <= aip1E_17_uid301_sincosTest_o(52 downto 0);

    -- aip1E_uid304_sincosTest(BITSELECT,303)@18
    aip1E_uid304_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_17_uid301_sincosTest_q(50 downto 0));
    aip1E_uid304_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid304_sincosTest_in(50 downto 0));

    -- redist131_aip1E_uid304_sincosTest_b_1(DELAY,1168)
    redist131_aip1E_uid304_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 51, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid304_sincosTest_b, xout => redist131_aip1E_uid304_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_18_uid317_sincosTest(ADDSUB,316)@19
    aip1E_18_uid317_sincosTest_s <= xMSB_uid305_sincosTest_b;
    aip1E_18_uid317_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((52 downto 51 => redist131_aip1E_uid304_sincosTest_b_1_q(50)) & redist131_aip1E_uid304_sincosTest_b_1_q));
    aip1E_18_uid317_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((52 downto 49 => cstArcTan2Mi_17_uid312_sincosTest_q(48)) & cstArcTan2Mi_17_uid312_sincosTest_q));
    aip1E_18_uid317_sincosTest_combproc: PROCESS (aip1E_18_uid317_sincosTest_a, aip1E_18_uid317_sincosTest_b, aip1E_18_uid317_sincosTest_s)
    BEGIN
        IF (aip1E_18_uid317_sincosTest_s = "1") THEN
            aip1E_18_uid317_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_18_uid317_sincosTest_a) + SIGNED(aip1E_18_uid317_sincosTest_b));
        ELSE
            aip1E_18_uid317_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_18_uid317_sincosTest_a) - SIGNED(aip1E_18_uid317_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_18_uid317_sincosTest_q <= aip1E_18_uid317_sincosTest_o(51 downto 0);

    -- aip1E_uid320_sincosTest(BITSELECT,319)@19
    aip1E_uid320_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_18_uid317_sincosTest_q(49 downto 0));
    aip1E_uid320_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid320_sincosTest_in(49 downto 0));

    -- redist128_aip1E_uid320_sincosTest_b_1(DELAY,1165)
    redist128_aip1E_uid320_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 50, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid320_sincosTest_b, xout => redist128_aip1E_uid320_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_19_uid333_sincosTest(ADDSUB,332)@20
    aip1E_19_uid333_sincosTest_s <= xMSB_uid321_sincosTest_b;
    aip1E_19_uid333_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((51 downto 50 => redist128_aip1E_uid320_sincosTest_b_1_q(49)) & redist128_aip1E_uid320_sincosTest_b_1_q));
    aip1E_19_uid333_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((51 downto 48 => cstArcTan2Mi_18_uid328_sincosTest_q(47)) & cstArcTan2Mi_18_uid328_sincosTest_q));
    aip1E_19_uid333_sincosTest_combproc: PROCESS (aip1E_19_uid333_sincosTest_a, aip1E_19_uid333_sincosTest_b, aip1E_19_uid333_sincosTest_s)
    BEGIN
        IF (aip1E_19_uid333_sincosTest_s = "1") THEN
            aip1E_19_uid333_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_19_uid333_sincosTest_a) + SIGNED(aip1E_19_uid333_sincosTest_b));
        ELSE
            aip1E_19_uid333_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_19_uid333_sincosTest_a) - SIGNED(aip1E_19_uid333_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_19_uid333_sincosTest_q <= aip1E_19_uid333_sincosTest_o(50 downto 0);

    -- aip1E_uid336_sincosTest(BITSELECT,335)@20
    aip1E_uid336_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_19_uid333_sincosTest_q(48 downto 0));
    aip1E_uid336_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid336_sincosTest_in(48 downto 0));

    -- redist125_aip1E_uid336_sincosTest_b_1(DELAY,1162)
    redist125_aip1E_uid336_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 49, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid336_sincosTest_b, xout => redist125_aip1E_uid336_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_20_uid349_sincosTest(ADDSUB,348)@21
    aip1E_20_uid349_sincosTest_s <= xMSB_uid337_sincosTest_b;
    aip1E_20_uid349_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((50 downto 49 => redist125_aip1E_uid336_sincosTest_b_1_q(48)) & redist125_aip1E_uid336_sincosTest_b_1_q));
    aip1E_20_uid349_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((50 downto 47 => cstArcTan2Mi_19_uid344_sincosTest_q(46)) & cstArcTan2Mi_19_uid344_sincosTest_q));
    aip1E_20_uid349_sincosTest_combproc: PROCESS (aip1E_20_uid349_sincosTest_a, aip1E_20_uid349_sincosTest_b, aip1E_20_uid349_sincosTest_s)
    BEGIN
        IF (aip1E_20_uid349_sincosTest_s = "1") THEN
            aip1E_20_uid349_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_20_uid349_sincosTest_a) + SIGNED(aip1E_20_uid349_sincosTest_b));
        ELSE
            aip1E_20_uid349_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_20_uid349_sincosTest_a) - SIGNED(aip1E_20_uid349_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_20_uid349_sincosTest_q <= aip1E_20_uid349_sincosTest_o(49 downto 0);

    -- aip1E_uid352_sincosTest(BITSELECT,351)@21
    aip1E_uid352_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_20_uid349_sincosTest_q(47 downto 0));
    aip1E_uid352_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid352_sincosTest_in(47 downto 0));

    -- redist122_aip1E_uid352_sincosTest_b_1(DELAY,1159)
    redist122_aip1E_uid352_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 48, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid352_sincosTest_b, xout => redist122_aip1E_uid352_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_21_uid365_sincosTest(ADDSUB,364)@22
    aip1E_21_uid365_sincosTest_s <= xMSB_uid353_sincosTest_b;
    aip1E_21_uid365_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((49 downto 48 => redist122_aip1E_uid352_sincosTest_b_1_q(47)) & redist122_aip1E_uid352_sincosTest_b_1_q));
    aip1E_21_uid365_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((49 downto 46 => cstArcTan2Mi_20_uid360_sincosTest_q(45)) & cstArcTan2Mi_20_uid360_sincosTest_q));
    aip1E_21_uid365_sincosTest_combproc: PROCESS (aip1E_21_uid365_sincosTest_a, aip1E_21_uid365_sincosTest_b, aip1E_21_uid365_sincosTest_s)
    BEGIN
        IF (aip1E_21_uid365_sincosTest_s = "1") THEN
            aip1E_21_uid365_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_21_uid365_sincosTest_a) + SIGNED(aip1E_21_uid365_sincosTest_b));
        ELSE
            aip1E_21_uid365_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_21_uid365_sincosTest_a) - SIGNED(aip1E_21_uid365_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_21_uid365_sincosTest_q <= aip1E_21_uid365_sincosTest_o(48 downto 0);

    -- aip1E_uid368_sincosTest(BITSELECT,367)@22
    aip1E_uid368_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_21_uid365_sincosTest_q(46 downto 0));
    aip1E_uid368_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid368_sincosTest_in(46 downto 0));

    -- redist119_aip1E_uid368_sincosTest_b_1(DELAY,1156)
    redist119_aip1E_uid368_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 47, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid368_sincosTest_b, xout => redist119_aip1E_uid368_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- aip1E_22_uid381_sincosTest(ADDSUB,380)@23
    aip1E_22_uid381_sincosTest_s <= xMSB_uid369_sincosTest_b;
    aip1E_22_uid381_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((48 downto 47 => redist119_aip1E_uid368_sincosTest_b_1_q(46)) & redist119_aip1E_uid368_sincosTest_b_1_q));
    aip1E_22_uid381_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((48 downto 45 => cstArcTan2Mi_21_uid376_sincosTest_q(44)) & cstArcTan2Mi_21_uid376_sincosTest_q));
    aip1E_22_uid381_sincosTest_combproc: PROCESS (aip1E_22_uid381_sincosTest_a, aip1E_22_uid381_sincosTest_b, aip1E_22_uid381_sincosTest_s)
    BEGIN
        IF (aip1E_22_uid381_sincosTest_s = "1") THEN
            aip1E_22_uid381_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_22_uid381_sincosTest_a) + SIGNED(aip1E_22_uid381_sincosTest_b));
        ELSE
            aip1E_22_uid381_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_22_uid381_sincosTest_a) - SIGNED(aip1E_22_uid381_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_22_uid381_sincosTest_q <= aip1E_22_uid381_sincosTest_o(47 downto 0);

    -- aip1E_uid384_sincosTest(BITSELECT,383)@23
    aip1E_uid384_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_22_uid381_sincosTest_q(45 downto 0));
    aip1E_uid384_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid384_sincosTest_in(45 downto 0));

    -- redist116_aip1E_uid384_sincosTest_b_1(DELAY,1153)
    redist116_aip1E_uid384_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 46, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid384_sincosTest_b, xout => redist116_aip1E_uid384_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid385_sincosTest(BITSELECT,384)@24
    xMSB_uid385_sincosTest_b <= STD_LOGIC_VECTOR(redist116_aip1E_uid384_sincosTest_b_1_q(45 downto 45));

    -- cstArcTan2Mi_22_uid392_sincosTest(CONSTANT,391)
    cstArcTan2Mi_22_uid392_sincosTest_q <= "01000000000000000000000000000000000000000000";

    -- highABits_uid398_sincosTest(BITSELECT,397)@24
    highABits_uid398_sincosTest_b <= STD_LOGIC_VECTOR(redist116_aip1E_uid384_sincosTest_b_1_q(45 downto 1));

    -- aip1E_23high_uid399_sincosTest(ADDSUB,398)@24
    aip1E_23high_uid399_sincosTest_s <= xMSB_uid385_sincosTest_b;
    aip1E_23high_uid399_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((46 downto 45 => highABits_uid398_sincosTest_b(44)) & highABits_uid398_sincosTest_b));
    aip1E_23high_uid399_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((46 downto 44 => cstArcTan2Mi_22_uid392_sincosTest_q(43)) & cstArcTan2Mi_22_uid392_sincosTest_q));
    aip1E_23high_uid399_sincosTest_combproc: PROCESS (aip1E_23high_uid399_sincosTest_a, aip1E_23high_uid399_sincosTest_b, aip1E_23high_uid399_sincosTest_s)
    BEGIN
        IF (aip1E_23high_uid399_sincosTest_s = "1") THEN
            aip1E_23high_uid399_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_23high_uid399_sincosTest_a) + SIGNED(aip1E_23high_uid399_sincosTest_b));
        ELSE
            aip1E_23high_uid399_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_23high_uid399_sincosTest_a) - SIGNED(aip1E_23high_uid399_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_23high_uid399_sincosTest_q <= aip1E_23high_uid399_sincosTest_o(45 downto 0);

    -- lowRangeA_uid397_sincosTest(BITSELECT,396)@24
    lowRangeA_uid397_sincosTest_in <= redist116_aip1E_uid384_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid397_sincosTest_b <= lowRangeA_uid397_sincosTest_in(0 downto 0);

    -- aip1E_23_uid400_sincosTest(BITJOIN,399)@24
    aip1E_23_uid400_sincosTest_q <= aip1E_23high_uid399_sincosTest_q & lowRangeA_uid397_sincosTest_b;

    -- aip1E_uid403_sincosTest(BITSELECT,402)@24
    aip1E_uid403_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_23_uid400_sincosTest_q(44 downto 0));
    aip1E_uid403_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid403_sincosTest_in(44 downto 0));

    -- redist113_aip1E_uid403_sincosTest_b_1(DELAY,1150)
    redist113_aip1E_uid403_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 45, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid403_sincosTest_b, xout => redist113_aip1E_uid403_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid404_sincosTest(BITSELECT,403)@25
    xMSB_uid404_sincosTest_b <= STD_LOGIC_VECTOR(redist113_aip1E_uid403_sincosTest_b_1_q(44 downto 44));

    -- cstArcTan2Mi_23_uid411_sincosTest(CONSTANT,410)
    cstArcTan2Mi_23_uid411_sincosTest_q <= "0100000000000000000000000000000000000000000";

    -- highABits_uid417_sincosTest(BITSELECT,416)@25
    highABits_uid417_sincosTest_b <= STD_LOGIC_VECTOR(redist113_aip1E_uid403_sincosTest_b_1_q(44 downto 1));

    -- aip1E_24high_uid418_sincosTest(ADDSUB,417)@25
    aip1E_24high_uid418_sincosTest_s <= xMSB_uid404_sincosTest_b;
    aip1E_24high_uid418_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((45 downto 44 => highABits_uid417_sincosTest_b(43)) & highABits_uid417_sincosTest_b));
    aip1E_24high_uid418_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((45 downto 43 => cstArcTan2Mi_23_uid411_sincosTest_q(42)) & cstArcTan2Mi_23_uid411_sincosTest_q));
    aip1E_24high_uid418_sincosTest_combproc: PROCESS (aip1E_24high_uid418_sincosTest_a, aip1E_24high_uid418_sincosTest_b, aip1E_24high_uid418_sincosTest_s)
    BEGIN
        IF (aip1E_24high_uid418_sincosTest_s = "1") THEN
            aip1E_24high_uid418_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_24high_uid418_sincosTest_a) + SIGNED(aip1E_24high_uid418_sincosTest_b));
        ELSE
            aip1E_24high_uid418_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_24high_uid418_sincosTest_a) - SIGNED(aip1E_24high_uid418_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_24high_uid418_sincosTest_q <= aip1E_24high_uid418_sincosTest_o(44 downto 0);

    -- lowRangeA_uid416_sincosTest(BITSELECT,415)@25
    lowRangeA_uid416_sincosTest_in <= redist113_aip1E_uid403_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid416_sincosTest_b <= lowRangeA_uid416_sincosTest_in(0 downto 0);

    -- aip1E_24_uid419_sincosTest(BITJOIN,418)@25
    aip1E_24_uid419_sincosTest_q <= aip1E_24high_uid418_sincosTest_q & lowRangeA_uid416_sincosTest_b;

    -- aip1E_uid422_sincosTest(BITSELECT,421)@25
    aip1E_uid422_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_24_uid419_sincosTest_q(43 downto 0));
    aip1E_uid422_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid422_sincosTest_in(43 downto 0));

    -- redist110_aip1E_uid422_sincosTest_b_1(DELAY,1147)
    redist110_aip1E_uid422_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 44, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid422_sincosTest_b, xout => redist110_aip1E_uid422_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid423_sincosTest(BITSELECT,422)@26
    xMSB_uid423_sincosTest_b <= STD_LOGIC_VECTOR(redist110_aip1E_uid422_sincosTest_b_1_q(43 downto 43));

    -- cstArcTan2Mi_24_uid430_sincosTest(CONSTANT,429)
    cstArcTan2Mi_24_uid430_sincosTest_q <= "010000000000000000000000000000000000000000";

    -- highABits_uid436_sincosTest(BITSELECT,435)@26
    highABits_uid436_sincosTest_b <= STD_LOGIC_VECTOR(redist110_aip1E_uid422_sincosTest_b_1_q(43 downto 1));

    -- aip1E_25high_uid437_sincosTest(ADDSUB,436)@26
    aip1E_25high_uid437_sincosTest_s <= xMSB_uid423_sincosTest_b;
    aip1E_25high_uid437_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((44 downto 43 => highABits_uid436_sincosTest_b(42)) & highABits_uid436_sincosTest_b));
    aip1E_25high_uid437_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((44 downto 42 => cstArcTan2Mi_24_uid430_sincosTest_q(41)) & cstArcTan2Mi_24_uid430_sincosTest_q));
    aip1E_25high_uid437_sincosTest_combproc: PROCESS (aip1E_25high_uid437_sincosTest_a, aip1E_25high_uid437_sincosTest_b, aip1E_25high_uid437_sincosTest_s)
    BEGIN
        IF (aip1E_25high_uid437_sincosTest_s = "1") THEN
            aip1E_25high_uid437_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_25high_uid437_sincosTest_a) + SIGNED(aip1E_25high_uid437_sincosTest_b));
        ELSE
            aip1E_25high_uid437_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_25high_uid437_sincosTest_a) - SIGNED(aip1E_25high_uid437_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_25high_uid437_sincosTest_q <= aip1E_25high_uid437_sincosTest_o(43 downto 0);

    -- lowRangeA_uid435_sincosTest(BITSELECT,434)@26
    lowRangeA_uid435_sincosTest_in <= redist110_aip1E_uid422_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid435_sincosTest_b <= lowRangeA_uid435_sincosTest_in(0 downto 0);

    -- aip1E_25_uid438_sincosTest(BITJOIN,437)@26
    aip1E_25_uid438_sincosTest_q <= aip1E_25high_uid437_sincosTest_q & lowRangeA_uid435_sincosTest_b;

    -- aip1E_uid441_sincosTest(BITSELECT,440)@26
    aip1E_uid441_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_25_uid438_sincosTest_q(42 downto 0));
    aip1E_uid441_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid441_sincosTest_in(42 downto 0));

    -- xMSB_uid442_sincosTest(BITSELECT,441)@26
    xMSB_uid442_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid441_sincosTest_b(42 downto 42));

    -- cstArcTan2Mi_25_uid449_sincosTest(CONSTANT,448)
    cstArcTan2Mi_25_uid449_sincosTest_q <= "01000000000000000000000000000000000000000";

    -- highABits_uid455_sincosTest(BITSELECT,454)@26
    highABits_uid455_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid441_sincosTest_b(42 downto 1));

    -- aip1E_26high_uid456_sincosTest(ADDSUB,455)@26
    aip1E_26high_uid456_sincosTest_s <= xMSB_uid442_sincosTest_b;
    aip1E_26high_uid456_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((43 downto 42 => highABits_uid455_sincosTest_b(41)) & highABits_uid455_sincosTest_b));
    aip1E_26high_uid456_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((43 downto 41 => cstArcTan2Mi_25_uid449_sincosTest_q(40)) & cstArcTan2Mi_25_uid449_sincosTest_q));
    aip1E_26high_uid456_sincosTest_combproc: PROCESS (aip1E_26high_uid456_sincosTest_a, aip1E_26high_uid456_sincosTest_b, aip1E_26high_uid456_sincosTest_s)
    BEGIN
        IF (aip1E_26high_uid456_sincosTest_s = "1") THEN
            aip1E_26high_uid456_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_26high_uid456_sincosTest_a) + SIGNED(aip1E_26high_uid456_sincosTest_b));
        ELSE
            aip1E_26high_uid456_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_26high_uid456_sincosTest_a) - SIGNED(aip1E_26high_uid456_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_26high_uid456_sincosTest_q <= aip1E_26high_uid456_sincosTest_o(42 downto 0);

    -- lowRangeA_uid454_sincosTest(BITSELECT,453)@26
    lowRangeA_uid454_sincosTest_in <= aip1E_uid441_sincosTest_b(0 downto 0);
    lowRangeA_uid454_sincosTest_b <= lowRangeA_uid454_sincosTest_in(0 downto 0);

    -- aip1E_26_uid457_sincosTest(BITJOIN,456)@26
    aip1E_26_uid457_sincosTest_q <= aip1E_26high_uid456_sincosTest_q & lowRangeA_uid454_sincosTest_b;

    -- aip1E_uid460_sincosTest(BITSELECT,459)@26
    aip1E_uid460_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_26_uid457_sincosTest_q(41 downto 0));
    aip1E_uid460_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid460_sincosTest_in(41 downto 0));

    -- redist104_aip1E_uid460_sincosTest_b_1(DELAY,1141)
    redist104_aip1E_uid460_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 42, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid460_sincosTest_b, xout => redist104_aip1E_uid460_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid461_sincosTest(BITSELECT,460)@27
    xMSB_uid461_sincosTest_b <= STD_LOGIC_VECTOR(redist104_aip1E_uid460_sincosTest_b_1_q(41 downto 41));

    -- cstArcTan2Mi_26_uid468_sincosTest(CONSTANT,467)
    cstArcTan2Mi_26_uid468_sincosTest_q <= "0100000000000000000000000000000000000000";

    -- highABits_uid474_sincosTest(BITSELECT,473)@27
    highABits_uid474_sincosTest_b <= STD_LOGIC_VECTOR(redist104_aip1E_uid460_sincosTest_b_1_q(41 downto 1));

    -- aip1E_27high_uid475_sincosTest(ADDSUB,474)@27
    aip1E_27high_uid475_sincosTest_s <= xMSB_uid461_sincosTest_b;
    aip1E_27high_uid475_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((42 downto 41 => highABits_uid474_sincosTest_b(40)) & highABits_uid474_sincosTest_b));
    aip1E_27high_uid475_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((42 downto 40 => cstArcTan2Mi_26_uid468_sincosTest_q(39)) & cstArcTan2Mi_26_uid468_sincosTest_q));
    aip1E_27high_uid475_sincosTest_combproc: PROCESS (aip1E_27high_uid475_sincosTest_a, aip1E_27high_uid475_sincosTest_b, aip1E_27high_uid475_sincosTest_s)
    BEGIN
        IF (aip1E_27high_uid475_sincosTest_s = "1") THEN
            aip1E_27high_uid475_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_27high_uid475_sincosTest_a) + SIGNED(aip1E_27high_uid475_sincosTest_b));
        ELSE
            aip1E_27high_uid475_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_27high_uid475_sincosTest_a) - SIGNED(aip1E_27high_uid475_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_27high_uid475_sincosTest_q <= aip1E_27high_uid475_sincosTest_o(41 downto 0);

    -- lowRangeA_uid473_sincosTest(BITSELECT,472)@27
    lowRangeA_uid473_sincosTest_in <= redist104_aip1E_uid460_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid473_sincosTest_b <= lowRangeA_uid473_sincosTest_in(0 downto 0);

    -- aip1E_27_uid476_sincosTest(BITJOIN,475)@27
    aip1E_27_uid476_sincosTest_q <= aip1E_27high_uid475_sincosTest_q & lowRangeA_uid473_sincosTest_b;

    -- aip1E_uid479_sincosTest(BITSELECT,478)@27
    aip1E_uid479_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_27_uid476_sincosTest_q(40 downto 0));
    aip1E_uid479_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid479_sincosTest_in(40 downto 0));

    -- xMSB_uid480_sincosTest(BITSELECT,479)@27
    xMSB_uid480_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid479_sincosTest_b(40 downto 40));

    -- cstArcTan2Mi_27_uid487_sincosTest(CONSTANT,486)
    cstArcTan2Mi_27_uid487_sincosTest_q <= "010000000000000000000000000000000000000";

    -- highABits_uid493_sincosTest(BITSELECT,492)@27
    highABits_uid493_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid479_sincosTest_b(40 downto 1));

    -- aip1E_28high_uid494_sincosTest(ADDSUB,493)@27
    aip1E_28high_uid494_sincosTest_s <= xMSB_uid480_sincosTest_b;
    aip1E_28high_uid494_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((41 downto 40 => highABits_uid493_sincosTest_b(39)) & highABits_uid493_sincosTest_b));
    aip1E_28high_uid494_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((41 downto 39 => cstArcTan2Mi_27_uid487_sincosTest_q(38)) & cstArcTan2Mi_27_uid487_sincosTest_q));
    aip1E_28high_uid494_sincosTest_combproc: PROCESS (aip1E_28high_uid494_sincosTest_a, aip1E_28high_uid494_sincosTest_b, aip1E_28high_uid494_sincosTest_s)
    BEGIN
        IF (aip1E_28high_uid494_sincosTest_s = "1") THEN
            aip1E_28high_uid494_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_28high_uid494_sincosTest_a) + SIGNED(aip1E_28high_uid494_sincosTest_b));
        ELSE
            aip1E_28high_uid494_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_28high_uid494_sincosTest_a) - SIGNED(aip1E_28high_uid494_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_28high_uid494_sincosTest_q <= aip1E_28high_uid494_sincosTest_o(40 downto 0);

    -- lowRangeA_uid492_sincosTest(BITSELECT,491)@27
    lowRangeA_uid492_sincosTest_in <= aip1E_uid479_sincosTest_b(0 downto 0);
    lowRangeA_uid492_sincosTest_b <= lowRangeA_uid492_sincosTest_in(0 downto 0);

    -- aip1E_28_uid495_sincosTest(BITJOIN,494)@27
    aip1E_28_uid495_sincosTest_q <= aip1E_28high_uid494_sincosTest_q & lowRangeA_uid492_sincosTest_b;

    -- aip1E_uid498_sincosTest(BITSELECT,497)@27
    aip1E_uid498_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_28_uid495_sincosTest_q(39 downto 0));
    aip1E_uid498_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid498_sincosTest_in(39 downto 0));

    -- redist97_aip1E_uid498_sincosTest_b_1(DELAY,1134)
    redist97_aip1E_uid498_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 40, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid498_sincosTest_b, xout => redist97_aip1E_uid498_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid499_sincosTest(BITSELECT,498)@28
    xMSB_uid499_sincosTest_b <= STD_LOGIC_VECTOR(redist97_aip1E_uid498_sincosTest_b_1_q(39 downto 39));

    -- cstArcTan2Mi_28_uid506_sincosTest(CONSTANT,505)
    cstArcTan2Mi_28_uid506_sincosTest_q <= "01000000000000000000000000000000000000";

    -- highABits_uid512_sincosTest(BITSELECT,511)@28
    highABits_uid512_sincosTest_b <= STD_LOGIC_VECTOR(redist97_aip1E_uid498_sincosTest_b_1_q(39 downto 1));

    -- aip1E_29high_uid513_sincosTest(ADDSUB,512)@28
    aip1E_29high_uid513_sincosTest_s <= xMSB_uid499_sincosTest_b;
    aip1E_29high_uid513_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((40 downto 39 => highABits_uid512_sincosTest_b(38)) & highABits_uid512_sincosTest_b));
    aip1E_29high_uid513_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((40 downto 38 => cstArcTan2Mi_28_uid506_sincosTest_q(37)) & cstArcTan2Mi_28_uid506_sincosTest_q));
    aip1E_29high_uid513_sincosTest_combproc: PROCESS (aip1E_29high_uid513_sincosTest_a, aip1E_29high_uid513_sincosTest_b, aip1E_29high_uid513_sincosTest_s)
    BEGIN
        IF (aip1E_29high_uid513_sincosTest_s = "1") THEN
            aip1E_29high_uid513_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_29high_uid513_sincosTest_a) + SIGNED(aip1E_29high_uid513_sincosTest_b));
        ELSE
            aip1E_29high_uid513_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_29high_uid513_sincosTest_a) - SIGNED(aip1E_29high_uid513_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_29high_uid513_sincosTest_q <= aip1E_29high_uid513_sincosTest_o(39 downto 0);

    -- lowRangeA_uid511_sincosTest(BITSELECT,510)@28
    lowRangeA_uid511_sincosTest_in <= redist97_aip1E_uid498_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid511_sincosTest_b <= lowRangeA_uid511_sincosTest_in(0 downto 0);

    -- aip1E_29_uid514_sincosTest(BITJOIN,513)@28
    aip1E_29_uid514_sincosTest_q <= aip1E_29high_uid513_sincosTest_q & lowRangeA_uid511_sincosTest_b;

    -- aip1E_uid517_sincosTest(BITSELECT,516)@28
    aip1E_uid517_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_29_uid514_sincosTest_q(38 downto 0));
    aip1E_uid517_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid517_sincosTest_in(38 downto 0));

    -- xMSB_uid518_sincosTest(BITSELECT,517)@28
    xMSB_uid518_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid517_sincosTest_b(38 downto 38));

    -- cstArcTan2Mi_29_uid525_sincosTest(CONSTANT,524)
    cstArcTan2Mi_29_uid525_sincosTest_q <= "0100000000000000000000000000000000000";

    -- highABits_uid531_sincosTest(BITSELECT,530)@28
    highABits_uid531_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid517_sincosTest_b(38 downto 1));

    -- aip1E_30high_uid532_sincosTest(ADDSUB,531)@28
    aip1E_30high_uid532_sincosTest_s <= xMSB_uid518_sincosTest_b;
    aip1E_30high_uid532_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((39 downto 38 => highABits_uid531_sincosTest_b(37)) & highABits_uid531_sincosTest_b));
    aip1E_30high_uid532_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((39 downto 37 => cstArcTan2Mi_29_uid525_sincosTest_q(36)) & cstArcTan2Mi_29_uid525_sincosTest_q));
    aip1E_30high_uid532_sincosTest_combproc: PROCESS (aip1E_30high_uid532_sincosTest_a, aip1E_30high_uid532_sincosTest_b, aip1E_30high_uid532_sincosTest_s)
    BEGIN
        IF (aip1E_30high_uid532_sincosTest_s = "1") THEN
            aip1E_30high_uid532_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_30high_uid532_sincosTest_a) + SIGNED(aip1E_30high_uid532_sincosTest_b));
        ELSE
            aip1E_30high_uid532_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_30high_uid532_sincosTest_a) - SIGNED(aip1E_30high_uid532_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_30high_uid532_sincosTest_q <= aip1E_30high_uid532_sincosTest_o(38 downto 0);

    -- lowRangeA_uid530_sincosTest(BITSELECT,529)@28
    lowRangeA_uid530_sincosTest_in <= aip1E_uid517_sincosTest_b(0 downto 0);
    lowRangeA_uid530_sincosTest_b <= lowRangeA_uid530_sincosTest_in(0 downto 0);

    -- aip1E_30_uid533_sincosTest(BITJOIN,532)@28
    aip1E_30_uid533_sincosTest_q <= aip1E_30high_uid532_sincosTest_q & lowRangeA_uid530_sincosTest_b;

    -- aip1E_uid536_sincosTest(BITSELECT,535)@28
    aip1E_uid536_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_30_uid533_sincosTest_q(37 downto 0));
    aip1E_uid536_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid536_sincosTest_in(37 downto 0));

    -- redist90_aip1E_uid536_sincosTest_b_1(DELAY,1127)
    redist90_aip1E_uid536_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 38, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid536_sincosTest_b, xout => redist90_aip1E_uid536_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid537_sincosTest(BITSELECT,536)@29
    xMSB_uid537_sincosTest_b <= STD_LOGIC_VECTOR(redist90_aip1E_uid536_sincosTest_b_1_q(37 downto 37));

    -- cstArcTan2Mi_30_uid544_sincosTest(CONSTANT,543)
    cstArcTan2Mi_30_uid544_sincosTest_q <= "010000000000000000000000000000000000";

    -- highABits_uid550_sincosTest(BITSELECT,549)@29
    highABits_uid550_sincosTest_b <= STD_LOGIC_VECTOR(redist90_aip1E_uid536_sincosTest_b_1_q(37 downto 1));

    -- aip1E_31high_uid551_sincosTest(ADDSUB,550)@29
    aip1E_31high_uid551_sincosTest_s <= xMSB_uid537_sincosTest_b;
    aip1E_31high_uid551_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((38 downto 37 => highABits_uid550_sincosTest_b(36)) & highABits_uid550_sincosTest_b));
    aip1E_31high_uid551_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((38 downto 36 => cstArcTan2Mi_30_uid544_sincosTest_q(35)) & cstArcTan2Mi_30_uid544_sincosTest_q));
    aip1E_31high_uid551_sincosTest_combproc: PROCESS (aip1E_31high_uid551_sincosTest_a, aip1E_31high_uid551_sincosTest_b, aip1E_31high_uid551_sincosTest_s)
    BEGIN
        IF (aip1E_31high_uid551_sincosTest_s = "1") THEN
            aip1E_31high_uid551_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_31high_uid551_sincosTest_a) + SIGNED(aip1E_31high_uid551_sincosTest_b));
        ELSE
            aip1E_31high_uid551_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_31high_uid551_sincosTest_a) - SIGNED(aip1E_31high_uid551_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_31high_uid551_sincosTest_q <= aip1E_31high_uid551_sincosTest_o(37 downto 0);

    -- lowRangeA_uid549_sincosTest(BITSELECT,548)@29
    lowRangeA_uid549_sincosTest_in <= redist90_aip1E_uid536_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid549_sincosTest_b <= lowRangeA_uid549_sincosTest_in(0 downto 0);

    -- aip1E_31_uid552_sincosTest(BITJOIN,551)@29
    aip1E_31_uid552_sincosTest_q <= aip1E_31high_uid551_sincosTest_q & lowRangeA_uid549_sincosTest_b;

    -- aip1E_uid555_sincosTest(BITSELECT,554)@29
    aip1E_uid555_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_31_uid552_sincosTest_q(36 downto 0));
    aip1E_uid555_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid555_sincosTest_in(36 downto 0));

    -- xMSB_uid556_sincosTest(BITSELECT,555)@29
    xMSB_uid556_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid555_sincosTest_b(36 downto 36));

    -- cstArcTan2Mi_31_uid563_sincosTest(CONSTANT,562)
    cstArcTan2Mi_31_uid563_sincosTest_q <= "01000000000000000000000000000000000";

    -- highABits_uid569_sincosTest(BITSELECT,568)@29
    highABits_uid569_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid555_sincosTest_b(36 downto 1));

    -- aip1E_32high_uid570_sincosTest(ADDSUB,569)@29
    aip1E_32high_uid570_sincosTest_s <= xMSB_uid556_sincosTest_b;
    aip1E_32high_uid570_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((37 downto 36 => highABits_uid569_sincosTest_b(35)) & highABits_uid569_sincosTest_b));
    aip1E_32high_uid570_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((37 downto 35 => cstArcTan2Mi_31_uid563_sincosTest_q(34)) & cstArcTan2Mi_31_uid563_sincosTest_q));
    aip1E_32high_uid570_sincosTest_combproc: PROCESS (aip1E_32high_uid570_sincosTest_a, aip1E_32high_uid570_sincosTest_b, aip1E_32high_uid570_sincosTest_s)
    BEGIN
        IF (aip1E_32high_uid570_sincosTest_s = "1") THEN
            aip1E_32high_uid570_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_32high_uid570_sincosTest_a) + SIGNED(aip1E_32high_uid570_sincosTest_b));
        ELSE
            aip1E_32high_uid570_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_32high_uid570_sincosTest_a) - SIGNED(aip1E_32high_uid570_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_32high_uid570_sincosTest_q <= aip1E_32high_uid570_sincosTest_o(36 downto 0);

    -- lowRangeA_uid568_sincosTest(BITSELECT,567)@29
    lowRangeA_uid568_sincosTest_in <= aip1E_uid555_sincosTest_b(0 downto 0);
    lowRangeA_uid568_sincosTest_b <= lowRangeA_uid568_sincosTest_in(0 downto 0);

    -- aip1E_32_uid571_sincosTest(BITJOIN,570)@29
    aip1E_32_uid571_sincosTest_q <= aip1E_32high_uid570_sincosTest_q & lowRangeA_uid568_sincosTest_b;

    -- aip1E_uid574_sincosTest(BITSELECT,573)@29
    aip1E_uid574_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_32_uid571_sincosTest_q(35 downto 0));
    aip1E_uid574_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid574_sincosTest_in(35 downto 0));

    -- redist83_aip1E_uid574_sincosTest_b_1(DELAY,1120)
    redist83_aip1E_uid574_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 36, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid574_sincosTest_b, xout => redist83_aip1E_uid574_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid575_sincosTest(BITSELECT,574)@30
    xMSB_uid575_sincosTest_b <= STD_LOGIC_VECTOR(redist83_aip1E_uid574_sincosTest_b_1_q(35 downto 35));

    -- cstArcTan2Mi_32_uid582_sincosTest(CONSTANT,581)
    cstArcTan2Mi_32_uid582_sincosTest_q <= "0100000000000000000000000000000000";

    -- highABits_uid588_sincosTest(BITSELECT,587)@30
    highABits_uid588_sincosTest_b <= STD_LOGIC_VECTOR(redist83_aip1E_uid574_sincosTest_b_1_q(35 downto 1));

    -- aip1E_33high_uid589_sincosTest(ADDSUB,588)@30
    aip1E_33high_uid589_sincosTest_s <= xMSB_uid575_sincosTest_b;
    aip1E_33high_uid589_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((36 downto 35 => highABits_uid588_sincosTest_b(34)) & highABits_uid588_sincosTest_b));
    aip1E_33high_uid589_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((36 downto 34 => cstArcTan2Mi_32_uid582_sincosTest_q(33)) & cstArcTan2Mi_32_uid582_sincosTest_q));
    aip1E_33high_uid589_sincosTest_combproc: PROCESS (aip1E_33high_uid589_sincosTest_a, aip1E_33high_uid589_sincosTest_b, aip1E_33high_uid589_sincosTest_s)
    BEGIN
        IF (aip1E_33high_uid589_sincosTest_s = "1") THEN
            aip1E_33high_uid589_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_33high_uid589_sincosTest_a) + SIGNED(aip1E_33high_uid589_sincosTest_b));
        ELSE
            aip1E_33high_uid589_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_33high_uid589_sincosTest_a) - SIGNED(aip1E_33high_uid589_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_33high_uid589_sincosTest_q <= aip1E_33high_uid589_sincosTest_o(35 downto 0);

    -- lowRangeA_uid587_sincosTest(BITSELECT,586)@30
    lowRangeA_uid587_sincosTest_in <= redist83_aip1E_uid574_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid587_sincosTest_b <= lowRangeA_uid587_sincosTest_in(0 downto 0);

    -- aip1E_33_uid590_sincosTest(BITJOIN,589)@30
    aip1E_33_uid590_sincosTest_q <= aip1E_33high_uid589_sincosTest_q & lowRangeA_uid587_sincosTest_b;

    -- aip1E_uid593_sincosTest(BITSELECT,592)@30
    aip1E_uid593_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_33_uid590_sincosTest_q(34 downto 0));
    aip1E_uid593_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid593_sincosTest_in(34 downto 0));

    -- xMSB_uid594_sincosTest(BITSELECT,593)@30
    xMSB_uid594_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid593_sincosTest_b(34 downto 34));

    -- cstArcTan2Mi_33_uid601_sincosTest(CONSTANT,600)
    cstArcTan2Mi_33_uid601_sincosTest_q <= "010000000000000000000000000000000";

    -- highABits_uid607_sincosTest(BITSELECT,606)@30
    highABits_uid607_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid593_sincosTest_b(34 downto 1));

    -- aip1E_34high_uid608_sincosTest(ADDSUB,607)@30
    aip1E_34high_uid608_sincosTest_s <= xMSB_uid594_sincosTest_b;
    aip1E_34high_uid608_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((35 downto 34 => highABits_uid607_sincosTest_b(33)) & highABits_uid607_sincosTest_b));
    aip1E_34high_uid608_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((35 downto 33 => cstArcTan2Mi_33_uid601_sincosTest_q(32)) & cstArcTan2Mi_33_uid601_sincosTest_q));
    aip1E_34high_uid608_sincosTest_combproc: PROCESS (aip1E_34high_uid608_sincosTest_a, aip1E_34high_uid608_sincosTest_b, aip1E_34high_uid608_sincosTest_s)
    BEGIN
        IF (aip1E_34high_uid608_sincosTest_s = "1") THEN
            aip1E_34high_uid608_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_34high_uid608_sincosTest_a) + SIGNED(aip1E_34high_uid608_sincosTest_b));
        ELSE
            aip1E_34high_uid608_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_34high_uid608_sincosTest_a) - SIGNED(aip1E_34high_uid608_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_34high_uid608_sincosTest_q <= aip1E_34high_uid608_sincosTest_o(34 downto 0);

    -- lowRangeA_uid606_sincosTest(BITSELECT,605)@30
    lowRangeA_uid606_sincosTest_in <= aip1E_uid593_sincosTest_b(0 downto 0);
    lowRangeA_uid606_sincosTest_b <= lowRangeA_uid606_sincosTest_in(0 downto 0);

    -- aip1E_34_uid609_sincosTest(BITJOIN,608)@30
    aip1E_34_uid609_sincosTest_q <= aip1E_34high_uid608_sincosTest_q & lowRangeA_uid606_sincosTest_b;

    -- aip1E_uid612_sincosTest(BITSELECT,611)@30
    aip1E_uid612_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_34_uid609_sincosTest_q(33 downto 0));
    aip1E_uid612_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid612_sincosTest_in(33 downto 0));

    -- redist76_aip1E_uid612_sincosTest_b_1(DELAY,1113)
    redist76_aip1E_uid612_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 34, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid612_sincosTest_b, xout => redist76_aip1E_uid612_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid613_sincosTest(BITSELECT,612)@31
    xMSB_uid613_sincosTest_b <= STD_LOGIC_VECTOR(redist76_aip1E_uid612_sincosTest_b_1_q(33 downto 33));

    -- cstArcTan2Mi_34_uid620_sincosTest(CONSTANT,619)
    cstArcTan2Mi_34_uid620_sincosTest_q <= "01000000000000000000000000000000";

    -- highABits_uid626_sincosTest(BITSELECT,625)@31
    highABits_uid626_sincosTest_b <= STD_LOGIC_VECTOR(redist76_aip1E_uid612_sincosTest_b_1_q(33 downto 1));

    -- aip1E_35high_uid627_sincosTest(ADDSUB,626)@31
    aip1E_35high_uid627_sincosTest_s <= xMSB_uid613_sincosTest_b;
    aip1E_35high_uid627_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((34 downto 33 => highABits_uid626_sincosTest_b(32)) & highABits_uid626_sincosTest_b));
    aip1E_35high_uid627_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((34 downto 32 => cstArcTan2Mi_34_uid620_sincosTest_q(31)) & cstArcTan2Mi_34_uid620_sincosTest_q));
    aip1E_35high_uid627_sincosTest_combproc: PROCESS (aip1E_35high_uid627_sincosTest_a, aip1E_35high_uid627_sincosTest_b, aip1E_35high_uid627_sincosTest_s)
    BEGIN
        IF (aip1E_35high_uid627_sincosTest_s = "1") THEN
            aip1E_35high_uid627_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_35high_uid627_sincosTest_a) + SIGNED(aip1E_35high_uid627_sincosTest_b));
        ELSE
            aip1E_35high_uid627_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_35high_uid627_sincosTest_a) - SIGNED(aip1E_35high_uid627_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_35high_uid627_sincosTest_q <= aip1E_35high_uid627_sincosTest_o(33 downto 0);

    -- lowRangeA_uid625_sincosTest(BITSELECT,624)@31
    lowRangeA_uid625_sincosTest_in <= redist76_aip1E_uid612_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid625_sincosTest_b <= lowRangeA_uid625_sincosTest_in(0 downto 0);

    -- aip1E_35_uid628_sincosTest(BITJOIN,627)@31
    aip1E_35_uid628_sincosTest_q <= aip1E_35high_uid627_sincosTest_q & lowRangeA_uid625_sincosTest_b;

    -- aip1E_uid631_sincosTest(BITSELECT,630)@31
    aip1E_uid631_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_35_uid628_sincosTest_q(32 downto 0));
    aip1E_uid631_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid631_sincosTest_in(32 downto 0));

    -- xMSB_uid632_sincosTest(BITSELECT,631)@31
    xMSB_uid632_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid631_sincosTest_b(32 downto 32));

    -- cstArcTan2Mi_35_uid639_sincosTest(CONSTANT,638)
    cstArcTan2Mi_35_uid639_sincosTest_q <= "0100000000000000000000000000000";

    -- highABits_uid645_sincosTest(BITSELECT,644)@31
    highABits_uid645_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid631_sincosTest_b(32 downto 1));

    -- aip1E_36high_uid646_sincosTest(ADDSUB,645)@31
    aip1E_36high_uid646_sincosTest_s <= xMSB_uid632_sincosTest_b;
    aip1E_36high_uid646_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((33 downto 32 => highABits_uid645_sincosTest_b(31)) & highABits_uid645_sincosTest_b));
    aip1E_36high_uid646_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((33 downto 31 => cstArcTan2Mi_35_uid639_sincosTest_q(30)) & cstArcTan2Mi_35_uid639_sincosTest_q));
    aip1E_36high_uid646_sincosTest_combproc: PROCESS (aip1E_36high_uid646_sincosTest_a, aip1E_36high_uid646_sincosTest_b, aip1E_36high_uid646_sincosTest_s)
    BEGIN
        IF (aip1E_36high_uid646_sincosTest_s = "1") THEN
            aip1E_36high_uid646_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_36high_uid646_sincosTest_a) + SIGNED(aip1E_36high_uid646_sincosTest_b));
        ELSE
            aip1E_36high_uid646_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_36high_uid646_sincosTest_a) - SIGNED(aip1E_36high_uid646_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_36high_uid646_sincosTest_q <= aip1E_36high_uid646_sincosTest_o(32 downto 0);

    -- lowRangeA_uid644_sincosTest(BITSELECT,643)@31
    lowRangeA_uid644_sincosTest_in <= aip1E_uid631_sincosTest_b(0 downto 0);
    lowRangeA_uid644_sincosTest_b <= lowRangeA_uid644_sincosTest_in(0 downto 0);

    -- aip1E_36_uid647_sincosTest(BITJOIN,646)@31
    aip1E_36_uid647_sincosTest_q <= aip1E_36high_uid646_sincosTest_q & lowRangeA_uid644_sincosTest_b;

    -- aip1E_uid650_sincosTest(BITSELECT,649)@31
    aip1E_uid650_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_36_uid647_sincosTest_q(31 downto 0));
    aip1E_uid650_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid650_sincosTest_in(31 downto 0));

    -- redist69_aip1E_uid650_sincosTest_b_1(DELAY,1106)
    redist69_aip1E_uid650_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 32, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid650_sincosTest_b, xout => redist69_aip1E_uid650_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid651_sincosTest(BITSELECT,650)@32
    xMSB_uid651_sincosTest_b <= STD_LOGIC_VECTOR(redist69_aip1E_uid650_sincosTest_b_1_q(31 downto 31));

    -- cstArcTan2Mi_36_uid658_sincosTest(CONSTANT,657)
    cstArcTan2Mi_36_uid658_sincosTest_q <= "010000000000000000000000000000";

    -- highABits_uid664_sincosTest(BITSELECT,663)@32
    highABits_uid664_sincosTest_b <= STD_LOGIC_VECTOR(redist69_aip1E_uid650_sincosTest_b_1_q(31 downto 1));

    -- aip1E_37high_uid665_sincosTest(ADDSUB,664)@32
    aip1E_37high_uid665_sincosTest_s <= xMSB_uid651_sincosTest_b;
    aip1E_37high_uid665_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((32 downto 31 => highABits_uid664_sincosTest_b(30)) & highABits_uid664_sincosTest_b));
    aip1E_37high_uid665_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((32 downto 30 => cstArcTan2Mi_36_uid658_sincosTest_q(29)) & cstArcTan2Mi_36_uid658_sincosTest_q));
    aip1E_37high_uid665_sincosTest_combproc: PROCESS (aip1E_37high_uid665_sincosTest_a, aip1E_37high_uid665_sincosTest_b, aip1E_37high_uid665_sincosTest_s)
    BEGIN
        IF (aip1E_37high_uid665_sincosTest_s = "1") THEN
            aip1E_37high_uid665_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_37high_uid665_sincosTest_a) + SIGNED(aip1E_37high_uid665_sincosTest_b));
        ELSE
            aip1E_37high_uid665_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_37high_uid665_sincosTest_a) - SIGNED(aip1E_37high_uid665_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_37high_uid665_sincosTest_q <= aip1E_37high_uid665_sincosTest_o(31 downto 0);

    -- lowRangeA_uid663_sincosTest(BITSELECT,662)@32
    lowRangeA_uid663_sincosTest_in <= redist69_aip1E_uid650_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid663_sincosTest_b <= lowRangeA_uid663_sincosTest_in(0 downto 0);

    -- aip1E_37_uid666_sincosTest(BITJOIN,665)@32
    aip1E_37_uid666_sincosTest_q <= aip1E_37high_uid665_sincosTest_q & lowRangeA_uid663_sincosTest_b;

    -- aip1E_uid669_sincosTest(BITSELECT,668)@32
    aip1E_uid669_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_37_uid666_sincosTest_q(30 downto 0));
    aip1E_uid669_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid669_sincosTest_in(30 downto 0));

    -- xMSB_uid670_sincosTest(BITSELECT,669)@32
    xMSB_uid670_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid669_sincosTest_b(30 downto 30));

    -- cstArcTan2Mi_37_uid677_sincosTest(CONSTANT,676)
    cstArcTan2Mi_37_uid677_sincosTest_q <= "01000000000000000000000000000";

    -- highABits_uid683_sincosTest(BITSELECT,682)@32
    highABits_uid683_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid669_sincosTest_b(30 downto 1));

    -- aip1E_38high_uid684_sincosTest(ADDSUB,683)@32
    aip1E_38high_uid684_sincosTest_s <= xMSB_uid670_sincosTest_b;
    aip1E_38high_uid684_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((31 downto 30 => highABits_uid683_sincosTest_b(29)) & highABits_uid683_sincosTest_b));
    aip1E_38high_uid684_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((31 downto 29 => cstArcTan2Mi_37_uid677_sincosTest_q(28)) & cstArcTan2Mi_37_uid677_sincosTest_q));
    aip1E_38high_uid684_sincosTest_combproc: PROCESS (aip1E_38high_uid684_sincosTest_a, aip1E_38high_uid684_sincosTest_b, aip1E_38high_uid684_sincosTest_s)
    BEGIN
        IF (aip1E_38high_uid684_sincosTest_s = "1") THEN
            aip1E_38high_uid684_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_38high_uid684_sincosTest_a) + SIGNED(aip1E_38high_uid684_sincosTest_b));
        ELSE
            aip1E_38high_uid684_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_38high_uid684_sincosTest_a) - SIGNED(aip1E_38high_uid684_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_38high_uid684_sincosTest_q <= aip1E_38high_uid684_sincosTest_o(30 downto 0);

    -- lowRangeA_uid682_sincosTest(BITSELECT,681)@32
    lowRangeA_uid682_sincosTest_in <= aip1E_uid669_sincosTest_b(0 downto 0);
    lowRangeA_uid682_sincosTest_b <= lowRangeA_uid682_sincosTest_in(0 downto 0);

    -- aip1E_38_uid685_sincosTest(BITJOIN,684)@32
    aip1E_38_uid685_sincosTest_q <= aip1E_38high_uid684_sincosTest_q & lowRangeA_uid682_sincosTest_b;

    -- aip1E_uid688_sincosTest(BITSELECT,687)@32
    aip1E_uid688_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_38_uid685_sincosTest_q(29 downto 0));
    aip1E_uid688_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid688_sincosTest_in(29 downto 0));

    -- redist62_aip1E_uid688_sincosTest_b_1(DELAY,1099)
    redist62_aip1E_uid688_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 30, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid688_sincosTest_b, xout => redist62_aip1E_uid688_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid689_sincosTest(BITSELECT,688)@33
    xMSB_uid689_sincosTest_b <= STD_LOGIC_VECTOR(redist62_aip1E_uid688_sincosTest_b_1_q(29 downto 29));

    -- cstArcTan2Mi_38_uid696_sincosTest(CONSTANT,695)
    cstArcTan2Mi_38_uid696_sincosTest_q <= "0100000000000000000000000000";

    -- highABits_uid702_sincosTest(BITSELECT,701)@33
    highABits_uid702_sincosTest_b <= STD_LOGIC_VECTOR(redist62_aip1E_uid688_sincosTest_b_1_q(29 downto 1));

    -- aip1E_39high_uid703_sincosTest(ADDSUB,702)@33
    aip1E_39high_uid703_sincosTest_s <= xMSB_uid689_sincosTest_b;
    aip1E_39high_uid703_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((30 downto 29 => highABits_uid702_sincosTest_b(28)) & highABits_uid702_sincosTest_b));
    aip1E_39high_uid703_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((30 downto 28 => cstArcTan2Mi_38_uid696_sincosTest_q(27)) & cstArcTan2Mi_38_uid696_sincosTest_q));
    aip1E_39high_uid703_sincosTest_combproc: PROCESS (aip1E_39high_uid703_sincosTest_a, aip1E_39high_uid703_sincosTest_b, aip1E_39high_uid703_sincosTest_s)
    BEGIN
        IF (aip1E_39high_uid703_sincosTest_s = "1") THEN
            aip1E_39high_uid703_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_39high_uid703_sincosTest_a) + SIGNED(aip1E_39high_uid703_sincosTest_b));
        ELSE
            aip1E_39high_uid703_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_39high_uid703_sincosTest_a) - SIGNED(aip1E_39high_uid703_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_39high_uid703_sincosTest_q <= aip1E_39high_uid703_sincosTest_o(29 downto 0);

    -- lowRangeA_uid701_sincosTest(BITSELECT,700)@33
    lowRangeA_uid701_sincosTest_in <= redist62_aip1E_uid688_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid701_sincosTest_b <= lowRangeA_uid701_sincosTest_in(0 downto 0);

    -- aip1E_39_uid704_sincosTest(BITJOIN,703)@33
    aip1E_39_uid704_sincosTest_q <= aip1E_39high_uid703_sincosTest_q & lowRangeA_uid701_sincosTest_b;

    -- aip1E_uid707_sincosTest(BITSELECT,706)@33
    aip1E_uid707_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_39_uid704_sincosTest_q(28 downto 0));
    aip1E_uid707_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid707_sincosTest_in(28 downto 0));

    -- xMSB_uid708_sincosTest(BITSELECT,707)@33
    xMSB_uid708_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid707_sincosTest_b(28 downto 28));

    -- cstArcTan2Mi_39_uid715_sincosTest(CONSTANT,714)
    cstArcTan2Mi_39_uid715_sincosTest_q <= "010000000000000000000000000";

    -- highABits_uid721_sincosTest(BITSELECT,720)@33
    highABits_uid721_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid707_sincosTest_b(28 downto 1));

    -- aip1E_40high_uid722_sincosTest(ADDSUB,721)@33
    aip1E_40high_uid722_sincosTest_s <= xMSB_uid708_sincosTest_b;
    aip1E_40high_uid722_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((29 downto 28 => highABits_uid721_sincosTest_b(27)) & highABits_uid721_sincosTest_b));
    aip1E_40high_uid722_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((29 downto 27 => cstArcTan2Mi_39_uid715_sincosTest_q(26)) & cstArcTan2Mi_39_uid715_sincosTest_q));
    aip1E_40high_uid722_sincosTest_combproc: PROCESS (aip1E_40high_uid722_sincosTest_a, aip1E_40high_uid722_sincosTest_b, aip1E_40high_uid722_sincosTest_s)
    BEGIN
        IF (aip1E_40high_uid722_sincosTest_s = "1") THEN
            aip1E_40high_uid722_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_40high_uid722_sincosTest_a) + SIGNED(aip1E_40high_uid722_sincosTest_b));
        ELSE
            aip1E_40high_uid722_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_40high_uid722_sincosTest_a) - SIGNED(aip1E_40high_uid722_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_40high_uid722_sincosTest_q <= aip1E_40high_uid722_sincosTest_o(28 downto 0);

    -- lowRangeA_uid720_sincosTest(BITSELECT,719)@33
    lowRangeA_uid720_sincosTest_in <= aip1E_uid707_sincosTest_b(0 downto 0);
    lowRangeA_uid720_sincosTest_b <= lowRangeA_uid720_sincosTest_in(0 downto 0);

    -- aip1E_40_uid723_sincosTest(BITJOIN,722)@33
    aip1E_40_uid723_sincosTest_q <= aip1E_40high_uid722_sincosTest_q & lowRangeA_uid720_sincosTest_b;

    -- aip1E_uid726_sincosTest(BITSELECT,725)@33
    aip1E_uid726_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_40_uid723_sincosTest_q(27 downto 0));
    aip1E_uid726_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid726_sincosTest_in(27 downto 0));

    -- redist55_aip1E_uid726_sincosTest_b_1(DELAY,1092)
    redist55_aip1E_uid726_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 28, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid726_sincosTest_b, xout => redist55_aip1E_uid726_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid727_sincosTest(BITSELECT,726)@34
    xMSB_uid727_sincosTest_b <= STD_LOGIC_VECTOR(redist55_aip1E_uid726_sincosTest_b_1_q(27 downto 27));

    -- cstArcTan2Mi_40_uid734_sincosTest(CONSTANT,733)
    cstArcTan2Mi_40_uid734_sincosTest_q <= "01000000000000000000000000";

    -- highABits_uid740_sincosTest(BITSELECT,739)@34
    highABits_uid740_sincosTest_b <= STD_LOGIC_VECTOR(redist55_aip1E_uid726_sincosTest_b_1_q(27 downto 1));

    -- aip1E_41high_uid741_sincosTest(ADDSUB,740)@34
    aip1E_41high_uid741_sincosTest_s <= xMSB_uid727_sincosTest_b;
    aip1E_41high_uid741_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((28 downto 27 => highABits_uid740_sincosTest_b(26)) & highABits_uid740_sincosTest_b));
    aip1E_41high_uid741_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((28 downto 26 => cstArcTan2Mi_40_uid734_sincosTest_q(25)) & cstArcTan2Mi_40_uid734_sincosTest_q));
    aip1E_41high_uid741_sincosTest_combproc: PROCESS (aip1E_41high_uid741_sincosTest_a, aip1E_41high_uid741_sincosTest_b, aip1E_41high_uid741_sincosTest_s)
    BEGIN
        IF (aip1E_41high_uid741_sincosTest_s = "1") THEN
            aip1E_41high_uid741_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_41high_uid741_sincosTest_a) + SIGNED(aip1E_41high_uid741_sincosTest_b));
        ELSE
            aip1E_41high_uid741_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_41high_uid741_sincosTest_a) - SIGNED(aip1E_41high_uid741_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_41high_uid741_sincosTest_q <= aip1E_41high_uid741_sincosTest_o(27 downto 0);

    -- lowRangeA_uid739_sincosTest(BITSELECT,738)@34
    lowRangeA_uid739_sincosTest_in <= redist55_aip1E_uid726_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid739_sincosTest_b <= lowRangeA_uid739_sincosTest_in(0 downto 0);

    -- aip1E_41_uid742_sincosTest(BITJOIN,741)@34
    aip1E_41_uid742_sincosTest_q <= aip1E_41high_uid741_sincosTest_q & lowRangeA_uid739_sincosTest_b;

    -- aip1E_uid745_sincosTest(BITSELECT,744)@34
    aip1E_uid745_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_41_uid742_sincosTest_q(26 downto 0));
    aip1E_uid745_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid745_sincosTest_in(26 downto 0));

    -- xMSB_uid746_sincosTest(BITSELECT,745)@34
    xMSB_uid746_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid745_sincosTest_b(26 downto 26));

    -- cstArcTan2Mi_41_uid753_sincosTest(CONSTANT,752)
    cstArcTan2Mi_41_uid753_sincosTest_q <= "0100000000000000000000000";

    -- highABits_uid759_sincosTest(BITSELECT,758)@34
    highABits_uid759_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid745_sincosTest_b(26 downto 1));

    -- aip1E_42high_uid760_sincosTest(ADDSUB,759)@34
    aip1E_42high_uid760_sincosTest_s <= xMSB_uid746_sincosTest_b;
    aip1E_42high_uid760_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((27 downto 26 => highABits_uid759_sincosTest_b(25)) & highABits_uid759_sincosTest_b));
    aip1E_42high_uid760_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((27 downto 25 => cstArcTan2Mi_41_uid753_sincosTest_q(24)) & cstArcTan2Mi_41_uid753_sincosTest_q));
    aip1E_42high_uid760_sincosTest_combproc: PROCESS (aip1E_42high_uid760_sincosTest_a, aip1E_42high_uid760_sincosTest_b, aip1E_42high_uid760_sincosTest_s)
    BEGIN
        IF (aip1E_42high_uid760_sincosTest_s = "1") THEN
            aip1E_42high_uid760_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_42high_uid760_sincosTest_a) + SIGNED(aip1E_42high_uid760_sincosTest_b));
        ELSE
            aip1E_42high_uid760_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_42high_uid760_sincosTest_a) - SIGNED(aip1E_42high_uid760_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_42high_uid760_sincosTest_q <= aip1E_42high_uid760_sincosTest_o(26 downto 0);

    -- lowRangeA_uid758_sincosTest(BITSELECT,757)@34
    lowRangeA_uid758_sincosTest_in <= aip1E_uid745_sincosTest_b(0 downto 0);
    lowRangeA_uid758_sincosTest_b <= lowRangeA_uid758_sincosTest_in(0 downto 0);

    -- aip1E_42_uid761_sincosTest(BITJOIN,760)@34
    aip1E_42_uid761_sincosTest_q <= aip1E_42high_uid760_sincosTest_q & lowRangeA_uid758_sincosTest_b;

    -- aip1E_uid764_sincosTest(BITSELECT,763)@34
    aip1E_uid764_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_42_uid761_sincosTest_q(25 downto 0));
    aip1E_uid764_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid764_sincosTest_in(25 downto 0));

    -- redist48_aip1E_uid764_sincosTest_b_1(DELAY,1085)
    redist48_aip1E_uid764_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 26, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid764_sincosTest_b, xout => redist48_aip1E_uid764_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid765_sincosTest(BITSELECT,764)@35
    xMSB_uid765_sincosTest_b <= STD_LOGIC_VECTOR(redist48_aip1E_uid764_sincosTest_b_1_q(25 downto 25));

    -- cstArcTan2Mi_42_uid772_sincosTest(CONSTANT,771)
    cstArcTan2Mi_42_uid772_sincosTest_q <= "010000000000000000000000";

    -- highABits_uid778_sincosTest(BITSELECT,777)@35
    highABits_uid778_sincosTest_b <= STD_LOGIC_VECTOR(redist48_aip1E_uid764_sincosTest_b_1_q(25 downto 1));

    -- aip1E_43high_uid779_sincosTest(ADDSUB,778)@35
    aip1E_43high_uid779_sincosTest_s <= xMSB_uid765_sincosTest_b;
    aip1E_43high_uid779_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((26 downto 25 => highABits_uid778_sincosTest_b(24)) & highABits_uid778_sincosTest_b));
    aip1E_43high_uid779_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((26 downto 24 => cstArcTan2Mi_42_uid772_sincosTest_q(23)) & cstArcTan2Mi_42_uid772_sincosTest_q));
    aip1E_43high_uid779_sincosTest_combproc: PROCESS (aip1E_43high_uid779_sincosTest_a, aip1E_43high_uid779_sincosTest_b, aip1E_43high_uid779_sincosTest_s)
    BEGIN
        IF (aip1E_43high_uid779_sincosTest_s = "1") THEN
            aip1E_43high_uid779_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_43high_uid779_sincosTest_a) + SIGNED(aip1E_43high_uid779_sincosTest_b));
        ELSE
            aip1E_43high_uid779_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_43high_uid779_sincosTest_a) - SIGNED(aip1E_43high_uid779_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_43high_uid779_sincosTest_q <= aip1E_43high_uid779_sincosTest_o(25 downto 0);

    -- lowRangeA_uid777_sincosTest(BITSELECT,776)@35
    lowRangeA_uid777_sincosTest_in <= redist48_aip1E_uid764_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid777_sincosTest_b <= lowRangeA_uid777_sincosTest_in(0 downto 0);

    -- aip1E_43_uid780_sincosTest(BITJOIN,779)@35
    aip1E_43_uid780_sincosTest_q <= aip1E_43high_uid779_sincosTest_q & lowRangeA_uid777_sincosTest_b;

    -- aip1E_uid783_sincosTest(BITSELECT,782)@35
    aip1E_uid783_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_43_uid780_sincosTest_q(24 downto 0));
    aip1E_uid783_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid783_sincosTest_in(24 downto 0));

    -- xMSB_uid784_sincosTest(BITSELECT,783)@35
    xMSB_uid784_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid783_sincosTest_b(24 downto 24));

    -- cstArcTan2Mi_43_uid791_sincosTest(CONSTANT,790)
    cstArcTan2Mi_43_uid791_sincosTest_q <= "01000000000000000000000";

    -- highABits_uid797_sincosTest(BITSELECT,796)@35
    highABits_uid797_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid783_sincosTest_b(24 downto 1));

    -- aip1E_44high_uid798_sincosTest(ADDSUB,797)@35
    aip1E_44high_uid798_sincosTest_s <= xMSB_uid784_sincosTest_b;
    aip1E_44high_uid798_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((25 downto 24 => highABits_uid797_sincosTest_b(23)) & highABits_uid797_sincosTest_b));
    aip1E_44high_uid798_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((25 downto 23 => cstArcTan2Mi_43_uid791_sincosTest_q(22)) & cstArcTan2Mi_43_uid791_sincosTest_q));
    aip1E_44high_uid798_sincosTest_combproc: PROCESS (aip1E_44high_uid798_sincosTest_a, aip1E_44high_uid798_sincosTest_b, aip1E_44high_uid798_sincosTest_s)
    BEGIN
        IF (aip1E_44high_uid798_sincosTest_s = "1") THEN
            aip1E_44high_uid798_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_44high_uid798_sincosTest_a) + SIGNED(aip1E_44high_uid798_sincosTest_b));
        ELSE
            aip1E_44high_uid798_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_44high_uid798_sincosTest_a) - SIGNED(aip1E_44high_uid798_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_44high_uid798_sincosTest_q <= aip1E_44high_uid798_sincosTest_o(24 downto 0);

    -- lowRangeA_uid796_sincosTest(BITSELECT,795)@35
    lowRangeA_uid796_sincosTest_in <= aip1E_uid783_sincosTest_b(0 downto 0);
    lowRangeA_uid796_sincosTest_b <= lowRangeA_uid796_sincosTest_in(0 downto 0);

    -- aip1E_44_uid799_sincosTest(BITJOIN,798)@35
    aip1E_44_uid799_sincosTest_q <= aip1E_44high_uid798_sincosTest_q & lowRangeA_uid796_sincosTest_b;

    -- aip1E_uid802_sincosTest(BITSELECT,801)@35
    aip1E_uid802_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_44_uid799_sincosTest_q(23 downto 0));
    aip1E_uid802_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid802_sincosTest_in(23 downto 0));

    -- redist41_aip1E_uid802_sincosTest_b_1(DELAY,1078)
    redist41_aip1E_uid802_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 24, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid802_sincosTest_b, xout => redist41_aip1E_uid802_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid803_sincosTest(BITSELECT,802)@36
    xMSB_uid803_sincosTest_b <= STD_LOGIC_VECTOR(redist41_aip1E_uid802_sincosTest_b_1_q(23 downto 23));

    -- cstArcTan2Mi_44_uid810_sincosTest(CONSTANT,809)
    cstArcTan2Mi_44_uid810_sincosTest_q <= "0100000000000000000000";

    -- highABits_uid816_sincosTest(BITSELECT,815)@36
    highABits_uid816_sincosTest_b <= STD_LOGIC_VECTOR(redist41_aip1E_uid802_sincosTest_b_1_q(23 downto 1));

    -- aip1E_45high_uid817_sincosTest(ADDSUB,816)@36
    aip1E_45high_uid817_sincosTest_s <= xMSB_uid803_sincosTest_b;
    aip1E_45high_uid817_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((24 downto 23 => highABits_uid816_sincosTest_b(22)) & highABits_uid816_sincosTest_b));
    aip1E_45high_uid817_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((24 downto 22 => cstArcTan2Mi_44_uid810_sincosTest_q(21)) & cstArcTan2Mi_44_uid810_sincosTest_q));
    aip1E_45high_uid817_sincosTest_combproc: PROCESS (aip1E_45high_uid817_sincosTest_a, aip1E_45high_uid817_sincosTest_b, aip1E_45high_uid817_sincosTest_s)
    BEGIN
        IF (aip1E_45high_uid817_sincosTest_s = "1") THEN
            aip1E_45high_uid817_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_45high_uid817_sincosTest_a) + SIGNED(aip1E_45high_uid817_sincosTest_b));
        ELSE
            aip1E_45high_uid817_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_45high_uid817_sincosTest_a) - SIGNED(aip1E_45high_uid817_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_45high_uid817_sincosTest_q <= aip1E_45high_uid817_sincosTest_o(23 downto 0);

    -- lowRangeA_uid815_sincosTest(BITSELECT,814)@36
    lowRangeA_uid815_sincosTest_in <= redist41_aip1E_uid802_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid815_sincosTest_b <= lowRangeA_uid815_sincosTest_in(0 downto 0);

    -- aip1E_45_uid818_sincosTest(BITJOIN,817)@36
    aip1E_45_uid818_sincosTest_q <= aip1E_45high_uid817_sincosTest_q & lowRangeA_uid815_sincosTest_b;

    -- aip1E_uid821_sincosTest(BITSELECT,820)@36
    aip1E_uid821_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_45_uid818_sincosTest_q(22 downto 0));
    aip1E_uid821_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid821_sincosTest_in(22 downto 0));

    -- xMSB_uid822_sincosTest(BITSELECT,821)@36
    xMSB_uid822_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid821_sincosTest_b(22 downto 22));

    -- cstArcTan2Mi_45_uid829_sincosTest(CONSTANT,828)
    cstArcTan2Mi_45_uid829_sincosTest_q <= "010000000000000000000";

    -- highABits_uid835_sincosTest(BITSELECT,834)@36
    highABits_uid835_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid821_sincosTest_b(22 downto 1));

    -- aip1E_46high_uid836_sincosTest(ADDSUB,835)@36
    aip1E_46high_uid836_sincosTest_s <= xMSB_uid822_sincosTest_b;
    aip1E_46high_uid836_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((23 downto 22 => highABits_uid835_sincosTest_b(21)) & highABits_uid835_sincosTest_b));
    aip1E_46high_uid836_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((23 downto 21 => cstArcTan2Mi_45_uid829_sincosTest_q(20)) & cstArcTan2Mi_45_uid829_sincosTest_q));
    aip1E_46high_uid836_sincosTest_combproc: PROCESS (aip1E_46high_uid836_sincosTest_a, aip1E_46high_uid836_sincosTest_b, aip1E_46high_uid836_sincosTest_s)
    BEGIN
        IF (aip1E_46high_uid836_sincosTest_s = "1") THEN
            aip1E_46high_uid836_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_46high_uid836_sincosTest_a) + SIGNED(aip1E_46high_uid836_sincosTest_b));
        ELSE
            aip1E_46high_uid836_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_46high_uid836_sincosTest_a) - SIGNED(aip1E_46high_uid836_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_46high_uid836_sincosTest_q <= aip1E_46high_uid836_sincosTest_o(22 downto 0);

    -- lowRangeA_uid834_sincosTest(BITSELECT,833)@36
    lowRangeA_uid834_sincosTest_in <= aip1E_uid821_sincosTest_b(0 downto 0);
    lowRangeA_uid834_sincosTest_b <= lowRangeA_uid834_sincosTest_in(0 downto 0);

    -- aip1E_46_uid837_sincosTest(BITJOIN,836)@36
    aip1E_46_uid837_sincosTest_q <= aip1E_46high_uid836_sincosTest_q & lowRangeA_uid834_sincosTest_b;

    -- aip1E_uid840_sincosTest(BITSELECT,839)@36
    aip1E_uid840_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_46_uid837_sincosTest_q(21 downto 0));
    aip1E_uid840_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid840_sincosTest_in(21 downto 0));

    -- redist34_aip1E_uid840_sincosTest_b_1(DELAY,1071)
    redist34_aip1E_uid840_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 22, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid840_sincosTest_b, xout => redist34_aip1E_uid840_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid841_sincosTest(BITSELECT,840)@37
    xMSB_uid841_sincosTest_b <= STD_LOGIC_VECTOR(redist34_aip1E_uid840_sincosTest_b_1_q(21 downto 21));

    -- cstArcTan2Mi_46_uid848_sincosTest(CONSTANT,847)
    cstArcTan2Mi_46_uid848_sincosTest_q <= "01000000000000000000";

    -- highABits_uid854_sincosTest(BITSELECT,853)@37
    highABits_uid854_sincosTest_b <= STD_LOGIC_VECTOR(redist34_aip1E_uid840_sincosTest_b_1_q(21 downto 1));

    -- aip1E_47high_uid855_sincosTest(ADDSUB,854)@37
    aip1E_47high_uid855_sincosTest_s <= xMSB_uid841_sincosTest_b;
    aip1E_47high_uid855_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((22 downto 21 => highABits_uid854_sincosTest_b(20)) & highABits_uid854_sincosTest_b));
    aip1E_47high_uid855_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((22 downto 20 => cstArcTan2Mi_46_uid848_sincosTest_q(19)) & cstArcTan2Mi_46_uid848_sincosTest_q));
    aip1E_47high_uid855_sincosTest_combproc: PROCESS (aip1E_47high_uid855_sincosTest_a, aip1E_47high_uid855_sincosTest_b, aip1E_47high_uid855_sincosTest_s)
    BEGIN
        IF (aip1E_47high_uid855_sincosTest_s = "1") THEN
            aip1E_47high_uid855_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_47high_uid855_sincosTest_a) + SIGNED(aip1E_47high_uid855_sincosTest_b));
        ELSE
            aip1E_47high_uid855_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_47high_uid855_sincosTest_a) - SIGNED(aip1E_47high_uid855_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_47high_uid855_sincosTest_q <= aip1E_47high_uid855_sincosTest_o(21 downto 0);

    -- lowRangeA_uid853_sincosTest(BITSELECT,852)@37
    lowRangeA_uid853_sincosTest_in <= redist34_aip1E_uid840_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid853_sincosTest_b <= lowRangeA_uid853_sincosTest_in(0 downto 0);

    -- aip1E_47_uid856_sincosTest(BITJOIN,855)@37
    aip1E_47_uid856_sincosTest_q <= aip1E_47high_uid855_sincosTest_q & lowRangeA_uid853_sincosTest_b;

    -- aip1E_uid859_sincosTest(BITSELECT,858)@37
    aip1E_uid859_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_47_uid856_sincosTest_q(20 downto 0));
    aip1E_uid859_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid859_sincosTest_in(20 downto 0));

    -- xMSB_uid860_sincosTest(BITSELECT,859)@37
    xMSB_uid860_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid859_sincosTest_b(20 downto 20));

    -- cstArcTan2Mi_47_uid867_sincosTest(CONSTANT,866)
    cstArcTan2Mi_47_uid867_sincosTest_q <= "0100000000000000000";

    -- highABits_uid873_sincosTest(BITSELECT,872)@37
    highABits_uid873_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid859_sincosTest_b(20 downto 1));

    -- aip1E_48high_uid874_sincosTest(ADDSUB,873)@37
    aip1E_48high_uid874_sincosTest_s <= xMSB_uid860_sincosTest_b;
    aip1E_48high_uid874_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((21 downto 20 => highABits_uid873_sincosTest_b(19)) & highABits_uid873_sincosTest_b));
    aip1E_48high_uid874_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((21 downto 19 => cstArcTan2Mi_47_uid867_sincosTest_q(18)) & cstArcTan2Mi_47_uid867_sincosTest_q));
    aip1E_48high_uid874_sincosTest_combproc: PROCESS (aip1E_48high_uid874_sincosTest_a, aip1E_48high_uid874_sincosTest_b, aip1E_48high_uid874_sincosTest_s)
    BEGIN
        IF (aip1E_48high_uid874_sincosTest_s = "1") THEN
            aip1E_48high_uid874_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_48high_uid874_sincosTest_a) + SIGNED(aip1E_48high_uid874_sincosTest_b));
        ELSE
            aip1E_48high_uid874_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_48high_uid874_sincosTest_a) - SIGNED(aip1E_48high_uid874_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_48high_uid874_sincosTest_q <= aip1E_48high_uid874_sincosTest_o(20 downto 0);

    -- lowRangeA_uid872_sincosTest(BITSELECT,871)@37
    lowRangeA_uid872_sincosTest_in <= aip1E_uid859_sincosTest_b(0 downto 0);
    lowRangeA_uid872_sincosTest_b <= lowRangeA_uid872_sincosTest_in(0 downto 0);

    -- aip1E_48_uid875_sincosTest(BITJOIN,874)@37
    aip1E_48_uid875_sincosTest_q <= aip1E_48high_uid874_sincosTest_q & lowRangeA_uid872_sincosTest_b;

    -- aip1E_uid878_sincosTest(BITSELECT,877)@37
    aip1E_uid878_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_48_uid875_sincosTest_q(19 downto 0));
    aip1E_uid878_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid878_sincosTest_in(19 downto 0));

    -- redist27_aip1E_uid878_sincosTest_b_1(DELAY,1064)
    redist27_aip1E_uid878_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 20, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid878_sincosTest_b, xout => redist27_aip1E_uid878_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid879_sincosTest(BITSELECT,878)@38
    xMSB_uid879_sincosTest_b <= STD_LOGIC_VECTOR(redist27_aip1E_uid878_sincosTest_b_1_q(19 downto 19));

    -- cstArcTan2Mi_48_uid886_sincosTest(CONSTANT,885)
    cstArcTan2Mi_48_uid886_sincosTest_q <= "010000000000000000";

    -- highABits_uid892_sincosTest(BITSELECT,891)@38
    highABits_uid892_sincosTest_b <= STD_LOGIC_VECTOR(redist27_aip1E_uid878_sincosTest_b_1_q(19 downto 1));

    -- aip1E_49high_uid893_sincosTest(ADDSUB,892)@38
    aip1E_49high_uid893_sincosTest_s <= xMSB_uid879_sincosTest_b;
    aip1E_49high_uid893_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((20 downto 19 => highABits_uid892_sincosTest_b(18)) & highABits_uid892_sincosTest_b));
    aip1E_49high_uid893_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((20 downto 18 => cstArcTan2Mi_48_uid886_sincosTest_q(17)) & cstArcTan2Mi_48_uid886_sincosTest_q));
    aip1E_49high_uid893_sincosTest_combproc: PROCESS (aip1E_49high_uid893_sincosTest_a, aip1E_49high_uid893_sincosTest_b, aip1E_49high_uid893_sincosTest_s)
    BEGIN
        IF (aip1E_49high_uid893_sincosTest_s = "1") THEN
            aip1E_49high_uid893_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_49high_uid893_sincosTest_a) + SIGNED(aip1E_49high_uid893_sincosTest_b));
        ELSE
            aip1E_49high_uid893_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_49high_uid893_sincosTest_a) - SIGNED(aip1E_49high_uid893_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_49high_uid893_sincosTest_q <= aip1E_49high_uid893_sincosTest_o(19 downto 0);

    -- lowRangeA_uid891_sincosTest(BITSELECT,890)@38
    lowRangeA_uid891_sincosTest_in <= redist27_aip1E_uid878_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid891_sincosTest_b <= lowRangeA_uid891_sincosTest_in(0 downto 0);

    -- aip1E_49_uid894_sincosTest(BITJOIN,893)@38
    aip1E_49_uid894_sincosTest_q <= aip1E_49high_uid893_sincosTest_q & lowRangeA_uid891_sincosTest_b;

    -- aip1E_uid897_sincosTest(BITSELECT,896)@38
    aip1E_uid897_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_49_uid894_sincosTest_q(18 downto 0));
    aip1E_uid897_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid897_sincosTest_in(18 downto 0));

    -- xMSB_uid898_sincosTest(BITSELECT,897)@38
    xMSB_uid898_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid897_sincosTest_b(18 downto 18));

    -- cstArcTan2Mi_49_uid905_sincosTest(CONSTANT,904)
    cstArcTan2Mi_49_uid905_sincosTest_q <= "01000000000000000";

    -- highABits_uid911_sincosTest(BITSELECT,910)@38
    highABits_uid911_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid897_sincosTest_b(18 downto 1));

    -- aip1E_50high_uid912_sincosTest(ADDSUB,911)@38
    aip1E_50high_uid912_sincosTest_s <= xMSB_uid898_sincosTest_b;
    aip1E_50high_uid912_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((19 downto 18 => highABits_uid911_sincosTest_b(17)) & highABits_uid911_sincosTest_b));
    aip1E_50high_uid912_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((19 downto 17 => cstArcTan2Mi_49_uid905_sincosTest_q(16)) & cstArcTan2Mi_49_uid905_sincosTest_q));
    aip1E_50high_uid912_sincosTest_combproc: PROCESS (aip1E_50high_uid912_sincosTest_a, aip1E_50high_uid912_sincosTest_b, aip1E_50high_uid912_sincosTest_s)
    BEGIN
        IF (aip1E_50high_uid912_sincosTest_s = "1") THEN
            aip1E_50high_uid912_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_50high_uid912_sincosTest_a) + SIGNED(aip1E_50high_uid912_sincosTest_b));
        ELSE
            aip1E_50high_uid912_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_50high_uid912_sincosTest_a) - SIGNED(aip1E_50high_uid912_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_50high_uid912_sincosTest_q <= aip1E_50high_uid912_sincosTest_o(18 downto 0);

    -- lowRangeA_uid910_sincosTest(BITSELECT,909)@38
    lowRangeA_uid910_sincosTest_in <= aip1E_uid897_sincosTest_b(0 downto 0);
    lowRangeA_uid910_sincosTest_b <= lowRangeA_uid910_sincosTest_in(0 downto 0);

    -- aip1E_50_uid913_sincosTest(BITJOIN,912)@38
    aip1E_50_uid913_sincosTest_q <= aip1E_50high_uid912_sincosTest_q & lowRangeA_uid910_sincosTest_b;

    -- aip1E_uid916_sincosTest(BITSELECT,915)@38
    aip1E_uid916_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_50_uid913_sincosTest_q(17 downto 0));
    aip1E_uid916_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid916_sincosTest_in(17 downto 0));

    -- redist20_aip1E_uid916_sincosTest_b_1(DELAY,1057)
    redist20_aip1E_uid916_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 18, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid916_sincosTest_b, xout => redist20_aip1E_uid916_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid917_sincosTest(BITSELECT,916)@39
    xMSB_uid917_sincosTest_b <= STD_LOGIC_VECTOR(redist20_aip1E_uid916_sincosTest_b_1_q(17 downto 17));

    -- cstArcTan2Mi_50_uid924_sincosTest(CONSTANT,923)
    cstArcTan2Mi_50_uid924_sincosTest_q <= "0100000000000000";

    -- highABits_uid930_sincosTest(BITSELECT,929)@39
    highABits_uid930_sincosTest_b <= STD_LOGIC_VECTOR(redist20_aip1E_uid916_sincosTest_b_1_q(17 downto 1));

    -- aip1E_51high_uid931_sincosTest(ADDSUB,930)@39
    aip1E_51high_uid931_sincosTest_s <= xMSB_uid917_sincosTest_b;
    aip1E_51high_uid931_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((18 downto 17 => highABits_uid930_sincosTest_b(16)) & highABits_uid930_sincosTest_b));
    aip1E_51high_uid931_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((18 downto 16 => cstArcTan2Mi_50_uid924_sincosTest_q(15)) & cstArcTan2Mi_50_uid924_sincosTest_q));
    aip1E_51high_uid931_sincosTest_combproc: PROCESS (aip1E_51high_uid931_sincosTest_a, aip1E_51high_uid931_sincosTest_b, aip1E_51high_uid931_sincosTest_s)
    BEGIN
        IF (aip1E_51high_uid931_sincosTest_s = "1") THEN
            aip1E_51high_uid931_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_51high_uid931_sincosTest_a) + SIGNED(aip1E_51high_uid931_sincosTest_b));
        ELSE
            aip1E_51high_uid931_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_51high_uid931_sincosTest_a) - SIGNED(aip1E_51high_uid931_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_51high_uid931_sincosTest_q <= aip1E_51high_uid931_sincosTest_o(17 downto 0);

    -- lowRangeA_uid929_sincosTest(BITSELECT,928)@39
    lowRangeA_uid929_sincosTest_in <= redist20_aip1E_uid916_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid929_sincosTest_b <= lowRangeA_uid929_sincosTest_in(0 downto 0);

    -- aip1E_51_uid932_sincosTest(BITJOIN,931)@39
    aip1E_51_uid932_sincosTest_q <= aip1E_51high_uid931_sincosTest_q & lowRangeA_uid929_sincosTest_b;

    -- aip1E_uid935_sincosTest(BITSELECT,934)@39
    aip1E_uid935_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_51_uid932_sincosTest_q(16 downto 0));
    aip1E_uid935_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid935_sincosTest_in(16 downto 0));

    -- xMSB_uid936_sincosTest(BITSELECT,935)@39
    xMSB_uid936_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid935_sincosTest_b(16 downto 16));

    -- cstArcTan2Mi_51_uid943_sincosTest(CONSTANT,942)
    cstArcTan2Mi_51_uid943_sincosTest_q <= "010000000000000";

    -- highABits_uid949_sincosTest(BITSELECT,948)@39
    highABits_uid949_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid935_sincosTest_b(16 downto 1));

    -- aip1E_52high_uid950_sincosTest(ADDSUB,949)@39
    aip1E_52high_uid950_sincosTest_s <= xMSB_uid936_sincosTest_b;
    aip1E_52high_uid950_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((17 downto 16 => highABits_uid949_sincosTest_b(15)) & highABits_uid949_sincosTest_b));
    aip1E_52high_uid950_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((17 downto 15 => cstArcTan2Mi_51_uid943_sincosTest_q(14)) & cstArcTan2Mi_51_uid943_sincosTest_q));
    aip1E_52high_uid950_sincosTest_combproc: PROCESS (aip1E_52high_uid950_sincosTest_a, aip1E_52high_uid950_sincosTest_b, aip1E_52high_uid950_sincosTest_s)
    BEGIN
        IF (aip1E_52high_uid950_sincosTest_s = "1") THEN
            aip1E_52high_uid950_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_52high_uid950_sincosTest_a) + SIGNED(aip1E_52high_uid950_sincosTest_b));
        ELSE
            aip1E_52high_uid950_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_52high_uid950_sincosTest_a) - SIGNED(aip1E_52high_uid950_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_52high_uid950_sincosTest_q <= aip1E_52high_uid950_sincosTest_o(16 downto 0);

    -- lowRangeA_uid948_sincosTest(BITSELECT,947)@39
    lowRangeA_uid948_sincosTest_in <= aip1E_uid935_sincosTest_b(0 downto 0);
    lowRangeA_uid948_sincosTest_b <= lowRangeA_uid948_sincosTest_in(0 downto 0);

    -- aip1E_52_uid951_sincosTest(BITJOIN,950)@39
    aip1E_52_uid951_sincosTest_q <= aip1E_52high_uid950_sincosTest_q & lowRangeA_uid948_sincosTest_b;

    -- aip1E_uid954_sincosTest(BITSELECT,953)@39
    aip1E_uid954_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_52_uid951_sincosTest_q(15 downto 0));
    aip1E_uid954_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid954_sincosTest_in(15 downto 0));

    -- redist13_aip1E_uid954_sincosTest_b_1(DELAY,1050)
    redist13_aip1E_uid954_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 16, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => aip1E_uid954_sincosTest_b, xout => redist13_aip1E_uid954_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xMSB_uid955_sincosTest(BITSELECT,954)@40
    xMSB_uid955_sincosTest_b <= STD_LOGIC_VECTOR(redist13_aip1E_uid954_sincosTest_b_1_q(15 downto 15));

    -- cstArcTan2Mi_52_uid962_sincosTest(CONSTANT,961)
    cstArcTan2Mi_52_uid962_sincosTest_q <= "01000000000000";

    -- highABits_uid968_sincosTest(BITSELECT,967)@40
    highABits_uid968_sincosTest_b <= STD_LOGIC_VECTOR(redist13_aip1E_uid954_sincosTest_b_1_q(15 downto 1));

    -- aip1E_53high_uid969_sincosTest(ADDSUB,968)@40
    aip1E_53high_uid969_sincosTest_s <= xMSB_uid955_sincosTest_b;
    aip1E_53high_uid969_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((16 downto 15 => highABits_uid968_sincosTest_b(14)) & highABits_uid968_sincosTest_b));
    aip1E_53high_uid969_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((16 downto 14 => cstArcTan2Mi_52_uid962_sincosTest_q(13)) & cstArcTan2Mi_52_uid962_sincosTest_q));
    aip1E_53high_uid969_sincosTest_combproc: PROCESS (aip1E_53high_uid969_sincosTest_a, aip1E_53high_uid969_sincosTest_b, aip1E_53high_uid969_sincosTest_s)
    BEGIN
        IF (aip1E_53high_uid969_sincosTest_s = "1") THEN
            aip1E_53high_uid969_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_53high_uid969_sincosTest_a) + SIGNED(aip1E_53high_uid969_sincosTest_b));
        ELSE
            aip1E_53high_uid969_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_53high_uid969_sincosTest_a) - SIGNED(aip1E_53high_uid969_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_53high_uid969_sincosTest_q <= aip1E_53high_uid969_sincosTest_o(15 downto 0);

    -- lowRangeA_uid967_sincosTest(BITSELECT,966)@40
    lowRangeA_uid967_sincosTest_in <= redist13_aip1E_uid954_sincosTest_b_1_q(0 downto 0);
    lowRangeA_uid967_sincosTest_b <= lowRangeA_uid967_sincosTest_in(0 downto 0);

    -- aip1E_53_uid970_sincosTest(BITJOIN,969)@40
    aip1E_53_uid970_sincosTest_q <= aip1E_53high_uid969_sincosTest_q & lowRangeA_uid967_sincosTest_b;

    -- aip1E_uid973_sincosTest(BITSELECT,972)@40
    aip1E_uid973_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_53_uid970_sincosTest_q(14 downto 0));
    aip1E_uid973_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid973_sincosTest_in(14 downto 0));

    -- xMSB_uid974_sincosTest(BITSELECT,973)@40
    xMSB_uid974_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid973_sincosTest_b(14 downto 14));

    -- cstArcTan2Mi_53_uid981_sincosTest(CONSTANT,980)
    cstArcTan2Mi_53_uid981_sincosTest_q <= "0100000000000";

    -- highABits_uid987_sincosTest(BITSELECT,986)@40
    highABits_uid987_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid973_sincosTest_b(14 downto 1));

    -- aip1E_54high_uid988_sincosTest(ADDSUB,987)@40
    aip1E_54high_uid988_sincosTest_s <= xMSB_uid974_sincosTest_b;
    aip1E_54high_uid988_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((15 downto 14 => highABits_uid987_sincosTest_b(13)) & highABits_uid987_sincosTest_b));
    aip1E_54high_uid988_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((15 downto 13 => cstArcTan2Mi_53_uid981_sincosTest_q(12)) & cstArcTan2Mi_53_uid981_sincosTest_q));
    aip1E_54high_uid988_sincosTest_combproc: PROCESS (aip1E_54high_uid988_sincosTest_a, aip1E_54high_uid988_sincosTest_b, aip1E_54high_uid988_sincosTest_s)
    BEGIN
        IF (aip1E_54high_uid988_sincosTest_s = "1") THEN
            aip1E_54high_uid988_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_54high_uid988_sincosTest_a) + SIGNED(aip1E_54high_uid988_sincosTest_b));
        ELSE
            aip1E_54high_uid988_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(aip1E_54high_uid988_sincosTest_a) - SIGNED(aip1E_54high_uid988_sincosTest_b));
        END IF;
    END PROCESS;
    aip1E_54high_uid988_sincosTest_q <= aip1E_54high_uid988_sincosTest_o(14 downto 0);

    -- lowRangeA_uid986_sincosTest(BITSELECT,985)@40
    lowRangeA_uid986_sincosTest_in <= aip1E_uid973_sincosTest_b(0 downto 0);
    lowRangeA_uid986_sincosTest_b <= lowRangeA_uid986_sincosTest_in(0 downto 0);

    -- aip1E_54_uid989_sincosTest(BITJOIN,988)@40
    aip1E_54_uid989_sincosTest_q <= aip1E_54high_uid988_sincosTest_q & lowRangeA_uid986_sincosTest_b;

    -- aip1E_uid992_sincosTest(BITSELECT,991)@40
    aip1E_uid992_sincosTest_in <= STD_LOGIC_VECTOR(aip1E_54_uid989_sincosTest_q(13 downto 0));
    aip1E_uid992_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid992_sincosTest_in(13 downto 0));

    -- xMSB_uid993_sincosTest(BITSELECT,992)@40
    xMSB_uid993_sincosTest_b <= STD_LOGIC_VECTOR(aip1E_uid992_sincosTest_b(13 downto 13));

    -- redist6_xMSB_uid993_sincosTest_b_16(DELAY,1043)
    redist6_xMSB_uid993_sincosTest_b_16 : dspba_delay
    GENERIC MAP ( width => 1, depth => 16, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid993_sincosTest_b, xout => redist6_xMSB_uid993_sincosTest_b_16_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid995_sincosTest(LOGICAL,994)@56
    signOfSelectionSignal_uid995_sincosTest_q <= not (redist6_xMSB_uid993_sincosTest_b_16_q);

    -- redist9_xMSB_uid974_sincosTest_b_15(DELAY,1046)
    redist9_xMSB_uid974_sincosTest_b_15 : dspba_delay
    GENERIC MAP ( width => 1, depth => 15, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid974_sincosTest_b, xout => redist9_xMSB_uid974_sincosTest_b_15_q, clk => clk, aclr => areset );

    -- redist12_xMSB_uid955_sincosTest_b_14(DELAY,1049)
    redist12_xMSB_uid955_sincosTest_b_14 : dspba_delay
    GENERIC MAP ( width => 1, depth => 14, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid955_sincosTest_b, xout => redist12_xMSB_uid955_sincosTest_b_14_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid957_sincosTest(LOGICAL,956)@54
    signOfSelectionSignal_uid957_sincosTest_q <= not (redist12_xMSB_uid955_sincosTest_b_14_q);

    -- redist16_xMSB_uid936_sincosTest_b_14(DELAY,1053)
    redist16_xMSB_uid936_sincosTest_b_14 : dspba_delay
    GENERIC MAP ( width => 1, depth => 14, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid936_sincosTest_b, xout => redist16_xMSB_uid936_sincosTest_b_14_q, clk => clk, aclr => areset );

    -- redist19_xMSB_uid917_sincosTest_b_13(DELAY,1056)
    redist19_xMSB_uid917_sincosTest_b_13 : dspba_delay
    GENERIC MAP ( width => 1, depth => 13, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid917_sincosTest_b, xout => redist19_xMSB_uid917_sincosTest_b_13_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid919_sincosTest(LOGICAL,918)@52
    signOfSelectionSignal_uid919_sincosTest_q <= not (redist19_xMSB_uid917_sincosTest_b_13_q);

    -- redist23_xMSB_uid898_sincosTest_b_13(DELAY,1060)
    redist23_xMSB_uid898_sincosTest_b_13 : dspba_delay
    GENERIC MAP ( width => 1, depth => 13, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid898_sincosTest_b, xout => redist23_xMSB_uid898_sincosTest_b_13_q, clk => clk, aclr => areset );

    -- redist26_xMSB_uid879_sincosTest_b_12(DELAY,1063)
    redist26_xMSB_uid879_sincosTest_b_12 : dspba_delay
    GENERIC MAP ( width => 1, depth => 12, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid879_sincosTest_b, xout => redist26_xMSB_uid879_sincosTest_b_12_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid881_sincosTest(LOGICAL,880)@50
    signOfSelectionSignal_uid881_sincosTest_q <= not (redist26_xMSB_uid879_sincosTest_b_12_q);

    -- redist30_xMSB_uid860_sincosTest_b_12(DELAY,1067)
    redist30_xMSB_uid860_sincosTest_b_12 : dspba_delay
    GENERIC MAP ( width => 1, depth => 12, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid860_sincosTest_b, xout => redist30_xMSB_uid860_sincosTest_b_12_q, clk => clk, aclr => areset );

    -- redist33_xMSB_uid841_sincosTest_b_11(DELAY,1070)
    redist33_xMSB_uid841_sincosTest_b_11 : dspba_delay
    GENERIC MAP ( width => 1, depth => 11, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid841_sincosTest_b, xout => redist33_xMSB_uid841_sincosTest_b_11_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid843_sincosTest(LOGICAL,842)@48
    signOfSelectionSignal_uid843_sincosTest_q <= not (redist33_xMSB_uid841_sincosTest_b_11_q);

    -- redist37_xMSB_uid822_sincosTest_b_11(DELAY,1074)
    redist37_xMSB_uid822_sincosTest_b_11 : dspba_delay
    GENERIC MAP ( width => 1, depth => 11, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid822_sincosTest_b, xout => redist37_xMSB_uid822_sincosTest_b_11_q, clk => clk, aclr => areset );

    -- redist40_xMSB_uid803_sincosTest_b_10(DELAY,1077)
    redist40_xMSB_uid803_sincosTest_b_10 : dspba_delay
    GENERIC MAP ( width => 1, depth => 10, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid803_sincosTest_b, xout => redist40_xMSB_uid803_sincosTest_b_10_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid805_sincosTest(LOGICAL,804)@46
    signOfSelectionSignal_uid805_sincosTest_q <= not (redist40_xMSB_uid803_sincosTest_b_10_q);

    -- redist44_xMSB_uid784_sincosTest_b_10(DELAY,1081)
    redist44_xMSB_uid784_sincosTest_b_10 : dspba_delay
    GENERIC MAP ( width => 1, depth => 10, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid784_sincosTest_b, xout => redist44_xMSB_uid784_sincosTest_b_10_q, clk => clk, aclr => areset );

    -- redist47_xMSB_uid765_sincosTest_b_9(DELAY,1084)
    redist47_xMSB_uid765_sincosTest_b_9 : dspba_delay
    GENERIC MAP ( width => 1, depth => 9, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid765_sincosTest_b, xout => redist47_xMSB_uid765_sincosTest_b_9_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid767_sincosTest(LOGICAL,766)@44
    signOfSelectionSignal_uid767_sincosTest_q <= not (redist47_xMSB_uid765_sincosTest_b_9_q);

    -- redist51_xMSB_uid746_sincosTest_b_9(DELAY,1088)
    redist51_xMSB_uid746_sincosTest_b_9 : dspba_delay
    GENERIC MAP ( width => 1, depth => 9, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid746_sincosTest_b, xout => redist51_xMSB_uid746_sincosTest_b_9_q, clk => clk, aclr => areset );

    -- redist54_xMSB_uid727_sincosTest_b_8(DELAY,1091)
    redist54_xMSB_uid727_sincosTest_b_8 : dspba_delay
    GENERIC MAP ( width => 1, depth => 8, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid727_sincosTest_b, xout => redist54_xMSB_uid727_sincosTest_b_8_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid729_sincosTest(LOGICAL,728)@42
    signOfSelectionSignal_uid729_sincosTest_q <= not (redist54_xMSB_uid727_sincosTest_b_8_q);

    -- redist58_xMSB_uid708_sincosTest_b_8(DELAY,1095)
    redist58_xMSB_uid708_sincosTest_b_8 : dspba_delay
    GENERIC MAP ( width => 1, depth => 8, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid708_sincosTest_b, xout => redist58_xMSB_uid708_sincosTest_b_8_q, clk => clk, aclr => areset );

    -- redist61_xMSB_uid689_sincosTest_b_7(DELAY,1098)
    redist61_xMSB_uid689_sincosTest_b_7 : dspba_delay
    GENERIC MAP ( width => 1, depth => 7, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid689_sincosTest_b, xout => redist61_xMSB_uid689_sincosTest_b_7_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid691_sincosTest(LOGICAL,690)@40
    signOfSelectionSignal_uid691_sincosTest_q <= not (redist61_xMSB_uid689_sincosTest_b_7_q);

    -- redist65_xMSB_uid670_sincosTest_b_7(DELAY,1102)
    redist65_xMSB_uid670_sincosTest_b_7 : dspba_delay
    GENERIC MAP ( width => 1, depth => 7, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid670_sincosTest_b, xout => redist65_xMSB_uid670_sincosTest_b_7_q, clk => clk, aclr => areset );

    -- redist68_xMSB_uid651_sincosTest_b_6(DELAY,1105)
    redist68_xMSB_uid651_sincosTest_b_6 : dspba_delay
    GENERIC MAP ( width => 1, depth => 6, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid651_sincosTest_b, xout => redist68_xMSB_uid651_sincosTest_b_6_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid653_sincosTest(LOGICAL,652)@38
    signOfSelectionSignal_uid653_sincosTest_q <= not (redist68_xMSB_uid651_sincosTest_b_6_q);

    -- redist72_xMSB_uid632_sincosTest_b_6(DELAY,1109)
    redist72_xMSB_uid632_sincosTest_b_6 : dspba_delay
    GENERIC MAP ( width => 1, depth => 6, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid632_sincosTest_b, xout => redist72_xMSB_uid632_sincosTest_b_6_q, clk => clk, aclr => areset );

    -- redist75_xMSB_uid613_sincosTest_b_5(DELAY,1112)
    redist75_xMSB_uid613_sincosTest_b_5 : dspba_delay
    GENERIC MAP ( width => 1, depth => 5, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid613_sincosTest_b, xout => redist75_xMSB_uid613_sincosTest_b_5_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid615_sincosTest(LOGICAL,614)@36
    signOfSelectionSignal_uid615_sincosTest_q <= not (redist75_xMSB_uid613_sincosTest_b_5_q);

    -- redist79_xMSB_uid594_sincosTest_b_5(DELAY,1116)
    redist79_xMSB_uid594_sincosTest_b_5 : dspba_delay
    GENERIC MAP ( width => 1, depth => 5, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid594_sincosTest_b, xout => redist79_xMSB_uid594_sincosTest_b_5_q, clk => clk, aclr => areset );

    -- redist82_xMSB_uid575_sincosTest_b_4(DELAY,1119)
    redist82_xMSB_uid575_sincosTest_b_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 4, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid575_sincosTest_b, xout => redist82_xMSB_uid575_sincosTest_b_4_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid577_sincosTest(LOGICAL,576)@34
    signOfSelectionSignal_uid577_sincosTest_q <= not (redist82_xMSB_uid575_sincosTest_b_4_q);

    -- redist86_xMSB_uid556_sincosTest_b_4(DELAY,1123)
    redist86_xMSB_uid556_sincosTest_b_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 4, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid556_sincosTest_b, xout => redist86_xMSB_uid556_sincosTest_b_4_q, clk => clk, aclr => areset );

    -- redist89_xMSB_uid537_sincosTest_b_3(DELAY,1126)
    redist89_xMSB_uid537_sincosTest_b_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid537_sincosTest_b, xout => redist89_xMSB_uid537_sincosTest_b_3_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid539_sincosTest(LOGICAL,538)@32
    signOfSelectionSignal_uid539_sincosTest_q <= not (redist89_xMSB_uid537_sincosTest_b_3_q);

    -- redist93_xMSB_uid518_sincosTest_b_3(DELAY,1130)
    redist93_xMSB_uid518_sincosTest_b_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid518_sincosTest_b, xout => redist93_xMSB_uid518_sincosTest_b_3_q, clk => clk, aclr => areset );

    -- redist96_xMSB_uid499_sincosTest_b_2(DELAY,1133)
    redist96_xMSB_uid499_sincosTest_b_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid499_sincosTest_b, xout => redist96_xMSB_uid499_sincosTest_b_2_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid501_sincosTest(LOGICAL,500)@30
    signOfSelectionSignal_uid501_sincosTest_q <= not (redist96_xMSB_uid499_sincosTest_b_2_q);

    -- redist100_xMSB_uid480_sincosTest_b_2(DELAY,1137)
    redist100_xMSB_uid480_sincosTest_b_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid480_sincosTest_b, xout => redist100_xMSB_uid480_sincosTest_b_2_q, clk => clk, aclr => areset );

    -- redist103_xMSB_uid461_sincosTest_b_1(DELAY,1140)
    redist103_xMSB_uid461_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid461_sincosTest_b, xout => redist103_xMSB_uid461_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid463_sincosTest(LOGICAL,462)@28
    signOfSelectionSignal_uid463_sincosTest_q <= not (redist103_xMSB_uid461_sincosTest_b_1_q);

    -- redist107_xMSB_uid442_sincosTest_b_1(DELAY,1144)
    redist107_xMSB_uid442_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xMSB_uid442_sincosTest_b, xout => redist107_xMSB_uid442_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- signOfSelectionSignal_uid425_sincosTest(LOGICAL,424)@26
    signOfSelectionSignal_uid425_sincosTest_q <= not (xMSB_uid423_sincosTest_b);

    -- signOfSelectionSignal_uid387_sincosTest(LOGICAL,386)@24
    signOfSelectionSignal_uid387_sincosTest_q <= not (xMSB_uid385_sincosTest_b);

    -- signOfSelectionSignal_uid355_sincosTest(LOGICAL,354)@22
    signOfSelectionSignal_uid355_sincosTest_q <= not (xMSB_uid353_sincosTest_b);

    -- signOfSelectionSignal_uid323_sincosTest(LOGICAL,322)@20
    signOfSelectionSignal_uid323_sincosTest_q <= not (xMSB_uid321_sincosTest_b);

    -- signOfSelectionSignal_uid291_sincosTest(LOGICAL,290)@18
    signOfSelectionSignal_uid291_sincosTest_q <= not (xMSB_uid289_sincosTest_b);

    -- signOfSelectionSignal_uid259_sincosTest(LOGICAL,258)@16
    signOfSelectionSignal_uid259_sincosTest_q <= not (xMSB_uid257_sincosTest_b);

    -- signOfSelectionSignal_uid227_sincosTest(LOGICAL,226)@14
    signOfSelectionSignal_uid227_sincosTest_q <= not (xMSB_uid225_sincosTest_b);

    -- signOfSelectionSignal_uid195_sincosTest(LOGICAL,194)@12
    signOfSelectionSignal_uid195_sincosTest_q <= not (xMSB_uid193_sincosTest_b);

    -- signOfSelectionSignal_uid163_sincosTest(LOGICAL,162)@10
    signOfSelectionSignal_uid163_sincosTest_q <= not (xMSB_uid161_sincosTest_b);

    -- signOfSelectionSignal_uid131_sincosTest(LOGICAL,130)@8
    signOfSelectionSignal_uid131_sincosTest_q <= not (xMSB_uid129_sincosTest_b);

    -- signOfSelectionSignal_uid99_sincosTest(LOGICAL,98)@6
    signOfSelectionSignal_uid99_sincosTest_q <= not (xMSB_uid97_sincosTest_b);

    -- signOfSelectionSignal_uid67_sincosTest(LOGICAL,66)@4
    signOfSelectionSignal_uid67_sincosTest_q <= not (xMSB_uid65_sincosTest_b);

    -- xMSB_uid42_sincosTest(BITSELECT,41)@3
    xMSB_uid42_sincosTest_b <= STD_LOGIC_VECTOR(yip1E_1sumAHighB_uid35_sincosTest_q(111 downto 111));

    -- cstOneOverK_uid22_sincosTest(CONSTANT,21)
    cstOneOverK_uid22_sincosTest_q <= "10011011011101001110110110101000010000110101111000000000000000000000000000000000000000000000000000000000000000";

    -- yip1E_1CostZeroPaddingA_uid33_sincosTest(CONSTANT,32)
    yip1E_1CostZeroPaddingA_uid33_sincosTest_q <= "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";

    -- yip1E_1NA_uid34_sincosTest(BITJOIN,33)@2
    yip1E_1NA_uid34_sincosTest_q <= GND_q & yip1E_1CostZeroPaddingA_uid33_sincosTest_q;

    -- yip1E_1sumAHighB_uid35_sincosTest(ADDSUB,34)@2 + 1
    yip1E_1sumAHighB_uid35_sincosTest_s <= VCC_q;
    yip1E_1sumAHighB_uid35_sincosTest_a <= STD_LOGIC_VECTOR("00" & yip1E_1NA_uid34_sincosTest_q);
    yip1E_1sumAHighB_uid35_sincosTest_b <= STD_LOGIC_VECTOR("000" & cstOneOverK_uid22_sincosTest_q);
    yip1E_1sumAHighB_uid35_sincosTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            yip1E_1sumAHighB_uid35_sincosTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (yip1E_1sumAHighB_uid35_sincosTest_s = "1") THEN
                yip1E_1sumAHighB_uid35_sincosTest_o <= STD_LOGIC_VECTOR(UNSIGNED(yip1E_1sumAHighB_uid35_sincosTest_a) + UNSIGNED(yip1E_1sumAHighB_uid35_sincosTest_b));
            ELSE
                yip1E_1sumAHighB_uid35_sincosTest_o <= STD_LOGIC_VECTOR(UNSIGNED(yip1E_1sumAHighB_uid35_sincosTest_a) - UNSIGNED(yip1E_1sumAHighB_uid35_sincosTest_b));
            END IF;
        END IF;
    END PROCESS;
    yip1E_1sumAHighB_uid35_sincosTest_q <= yip1E_1sumAHighB_uid35_sincosTest_o(111 downto 0);

    -- yip1_1_mergedSignalTM_uid46_sincosTest(BITJOIN,45)@3
    yip1_1_mergedSignalTM_uid46_sincosTest_q <= xMSB_uid42_sincosTest_b & yip1E_1sumAHighB_uid35_sincosTest_q;

    -- twoToMiSiYip_uid55_sincosTest(BITSELECT,54)@3
    twoToMiSiYip_uid55_sincosTest_b <= STD_LOGIC_VECTOR(yip1_1_mergedSignalTM_uid46_sincosTest_q(112 downto 1));

    -- xip1E_1_uid32_sincosTest(BITJOIN,31)@3
    xip1E_1_uid32_sincosTest_q <= STD_LOGIC_VECTOR((2 downto 1 => GND_q(0)) & GND_q) & cstOneOverK_uid22_sincosTest_q;

    -- xip1_1_topRange_uid39_sincosTest(BITSELECT,38)@3
    xip1_1_topRange_uid39_sincosTest_in <= xip1E_1_uid32_sincosTest_q(111 downto 0);
    xip1_1_topRange_uid39_sincosTest_b <= xip1_1_topRange_uid39_sincosTest_in(111 downto 0);

    -- xip1_1_mergedSignalTM_uid40_sincosTest(BITJOIN,39)@3
    xip1_1_mergedSignalTM_uid40_sincosTest_q <= GND_q & xip1_1_topRange_uid39_sincosTest_b;

    -- xip1E_2_uid58_sincosTest(ADDSUB,57)@3
    xip1E_2_uid58_sincosTest_s <= xMSB_uid49_sincosTest_b;
    xip1E_2_uid58_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => xip1_1_mergedSignalTM_uid40_sincosTest_q(112)) & xip1_1_mergedSignalTM_uid40_sincosTest_q));
    xip1E_2_uid58_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 112 => twoToMiSiYip_uid55_sincosTest_b(111)) & twoToMiSiYip_uid55_sincosTest_b));
    xip1E_2_uid58_sincosTest_combproc: PROCESS (xip1E_2_uid58_sincosTest_a, xip1E_2_uid58_sincosTest_b, xip1E_2_uid58_sincosTest_s)
    BEGIN
        IF (xip1E_2_uid58_sincosTest_s = "1") THEN
            xip1E_2_uid58_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_2_uid58_sincosTest_a) + SIGNED(xip1E_2_uid58_sincosTest_b));
        ELSE
            xip1E_2_uid58_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_2_uid58_sincosTest_a) - SIGNED(xip1E_2_uid58_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_2_uid58_sincosTest_q <= xip1E_2_uid58_sincosTest_o(113 downto 0);

    -- xip1_2_uid62_sincosTest(BITSELECT,61)@3
    xip1_2_uid62_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_2_uid58_sincosTest_q(112 downto 0));
    xip1_2_uid62_sincosTest_b <= STD_LOGIC_VECTOR(xip1_2_uid62_sincosTest_in(112 downto 0));

    -- redist178_xip1_2_uid62_sincosTest_b_1(DELAY,1215)
    redist178_xip1_2_uid62_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_2_uid62_sincosTest_b, xout => redist178_xip1_2_uid62_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid70_sincosTest(BITSELECT,69)@4
    twoToMiSiXip_uid70_sincosTest_b <= STD_LOGIC_VECTOR(redist178_xip1_2_uid62_sincosTest_b_1_q(112 downto 2));

    -- signOfSelectionSignal_uid51_sincosTest(LOGICAL,50)@3
    signOfSelectionSignal_uid51_sincosTest_q <= not (xMSB_uid49_sincosTest_b);

    -- twoToMiSiXip_uid54_sincosTest(BITSELECT,53)@3
    twoToMiSiXip_uid54_sincosTest_b <= STD_LOGIC_VECTOR(xip1_1_mergedSignalTM_uid40_sincosTest_q(112 downto 1));

    -- yip1E_2_uid59_sincosTest(ADDSUB,58)@3
    yip1E_2_uid59_sincosTest_s <= signOfSelectionSignal_uid51_sincosTest_q;
    yip1E_2_uid59_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => yip1_1_mergedSignalTM_uid46_sincosTest_q(112)) & yip1_1_mergedSignalTM_uid46_sincosTest_q));
    yip1E_2_uid59_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 112 => twoToMiSiXip_uid54_sincosTest_b(111)) & twoToMiSiXip_uid54_sincosTest_b));
    yip1E_2_uid59_sincosTest_combproc: PROCESS (yip1E_2_uid59_sincosTest_a, yip1E_2_uid59_sincosTest_b, yip1E_2_uid59_sincosTest_s)
    BEGIN
        IF (yip1E_2_uid59_sincosTest_s = "1") THEN
            yip1E_2_uid59_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_2_uid59_sincosTest_a) + SIGNED(yip1E_2_uid59_sincosTest_b));
        ELSE
            yip1E_2_uid59_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_2_uid59_sincosTest_a) - SIGNED(yip1E_2_uid59_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_2_uid59_sincosTest_q <= yip1E_2_uid59_sincosTest_o(113 downto 0);

    -- yip1_2_uid63_sincosTest(BITSELECT,62)@3
    yip1_2_uid63_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_2_uid59_sincosTest_q(112 downto 0));
    yip1_2_uid63_sincosTest_b <= STD_LOGIC_VECTOR(yip1_2_uid63_sincosTest_in(112 downto 0));

    -- redist177_yip1_2_uid63_sincosTest_b_1(DELAY,1214)
    redist177_yip1_2_uid63_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_2_uid63_sincosTest_b, xout => redist177_yip1_2_uid63_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_3_uid75_sincosTest(ADDSUB,74)@4
    yip1E_3_uid75_sincosTest_s <= signOfSelectionSignal_uid67_sincosTest_q;
    yip1E_3_uid75_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist177_yip1_2_uid63_sincosTest_b_1_q(112)) & redist177_yip1_2_uid63_sincosTest_b_1_q));
    yip1E_3_uid75_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 111 => twoToMiSiXip_uid70_sincosTest_b(110)) & twoToMiSiXip_uid70_sincosTest_b));
    yip1E_3_uid75_sincosTest_combproc: PROCESS (yip1E_3_uid75_sincosTest_a, yip1E_3_uid75_sincosTest_b, yip1E_3_uid75_sincosTest_s)
    BEGIN
        IF (yip1E_3_uid75_sincosTest_s = "1") THEN
            yip1E_3_uid75_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_3_uid75_sincosTest_a) + SIGNED(yip1E_3_uid75_sincosTest_b));
        ELSE
            yip1E_3_uid75_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_3_uid75_sincosTest_a) - SIGNED(yip1E_3_uid75_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_3_uid75_sincosTest_q <= yip1E_3_uid75_sincosTest_o(113 downto 0);

    -- yip1_3_uid79_sincosTest(BITSELECT,78)@4
    yip1_3_uid79_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_3_uid75_sincosTest_q(112 downto 0));
    yip1_3_uid79_sincosTest_b <= STD_LOGIC_VECTOR(yip1_3_uid79_sincosTest_in(112 downto 0));

    -- redist174_yip1_3_uid79_sincosTest_b_1(DELAY,1211)
    redist174_yip1_3_uid79_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_3_uid79_sincosTest_b, xout => redist174_yip1_3_uid79_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid87_sincosTest(BITSELECT,86)@5
    twoToMiSiYip_uid87_sincosTest_b <= STD_LOGIC_VECTOR(redist174_yip1_3_uid79_sincosTest_b_1_q(112 downto 3));

    -- twoToMiSiYip_uid71_sincosTest(BITSELECT,70)@4
    twoToMiSiYip_uid71_sincosTest_b <= STD_LOGIC_VECTOR(redist177_yip1_2_uid63_sincosTest_b_1_q(112 downto 2));

    -- xip1E_3_uid74_sincosTest(ADDSUB,73)@4
    xip1E_3_uid74_sincosTest_s <= xMSB_uid65_sincosTest_b;
    xip1E_3_uid74_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist178_xip1_2_uid62_sincosTest_b_1_q(112)) & redist178_xip1_2_uid62_sincosTest_b_1_q));
    xip1E_3_uid74_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 111 => twoToMiSiYip_uid71_sincosTest_b(110)) & twoToMiSiYip_uid71_sincosTest_b));
    xip1E_3_uid74_sincosTest_combproc: PROCESS (xip1E_3_uid74_sincosTest_a, xip1E_3_uid74_sincosTest_b, xip1E_3_uid74_sincosTest_s)
    BEGIN
        IF (xip1E_3_uid74_sincosTest_s = "1") THEN
            xip1E_3_uid74_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_3_uid74_sincosTest_a) + SIGNED(xip1E_3_uid74_sincosTest_b));
        ELSE
            xip1E_3_uid74_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_3_uid74_sincosTest_a) - SIGNED(xip1E_3_uid74_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_3_uid74_sincosTest_q <= xip1E_3_uid74_sincosTest_o(113 downto 0);

    -- xip1_3_uid78_sincosTest(BITSELECT,77)@4
    xip1_3_uid78_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_3_uid74_sincosTest_q(112 downto 0));
    xip1_3_uid78_sincosTest_b <= STD_LOGIC_VECTOR(xip1_3_uid78_sincosTest_in(112 downto 0));

    -- redist175_xip1_3_uid78_sincosTest_b_1(DELAY,1212)
    redist175_xip1_3_uid78_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_3_uid78_sincosTest_b, xout => redist175_xip1_3_uid78_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_4_uid90_sincosTest(ADDSUB,89)@5
    xip1E_4_uid90_sincosTest_s <= xMSB_uid81_sincosTest_b;
    xip1E_4_uid90_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist175_xip1_3_uid78_sincosTest_b_1_q(112)) & redist175_xip1_3_uid78_sincosTest_b_1_q));
    xip1E_4_uid90_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 110 => twoToMiSiYip_uid87_sincosTest_b(109)) & twoToMiSiYip_uid87_sincosTest_b));
    xip1E_4_uid90_sincosTest_combproc: PROCESS (xip1E_4_uid90_sincosTest_a, xip1E_4_uid90_sincosTest_b, xip1E_4_uid90_sincosTest_s)
    BEGIN
        IF (xip1E_4_uid90_sincosTest_s = "1") THEN
            xip1E_4_uid90_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_4_uid90_sincosTest_a) + SIGNED(xip1E_4_uid90_sincosTest_b));
        ELSE
            xip1E_4_uid90_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_4_uid90_sincosTest_a) - SIGNED(xip1E_4_uid90_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_4_uid90_sincosTest_q <= xip1E_4_uid90_sincosTest_o(113 downto 0);

    -- xip1_4_uid94_sincosTest(BITSELECT,93)@5
    xip1_4_uid94_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_4_uid90_sincosTest_q(112 downto 0));
    xip1_4_uid94_sincosTest_b <= STD_LOGIC_VECTOR(xip1_4_uid94_sincosTest_in(112 downto 0));

    -- redist172_xip1_4_uid94_sincosTest_b_1(DELAY,1209)
    redist172_xip1_4_uid94_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_4_uid94_sincosTest_b, xout => redist172_xip1_4_uid94_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid102_sincosTest(BITSELECT,101)@6
    twoToMiSiXip_uid102_sincosTest_b <= STD_LOGIC_VECTOR(redist172_xip1_4_uid94_sincosTest_b_1_q(112 downto 4));

    -- signOfSelectionSignal_uid83_sincosTest(LOGICAL,82)@5
    signOfSelectionSignal_uid83_sincosTest_q <= not (xMSB_uid81_sincosTest_b);

    -- twoToMiSiXip_uid86_sincosTest(BITSELECT,85)@5
    twoToMiSiXip_uid86_sincosTest_b <= STD_LOGIC_VECTOR(redist175_xip1_3_uid78_sincosTest_b_1_q(112 downto 3));

    -- yip1E_4_uid91_sincosTest(ADDSUB,90)@5
    yip1E_4_uid91_sincosTest_s <= signOfSelectionSignal_uid83_sincosTest_q;
    yip1E_4_uid91_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist174_yip1_3_uid79_sincosTest_b_1_q(112)) & redist174_yip1_3_uid79_sincosTest_b_1_q));
    yip1E_4_uid91_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 110 => twoToMiSiXip_uid86_sincosTest_b(109)) & twoToMiSiXip_uid86_sincosTest_b));
    yip1E_4_uid91_sincosTest_combproc: PROCESS (yip1E_4_uid91_sincosTest_a, yip1E_4_uid91_sincosTest_b, yip1E_4_uid91_sincosTest_s)
    BEGIN
        IF (yip1E_4_uid91_sincosTest_s = "1") THEN
            yip1E_4_uid91_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_4_uid91_sincosTest_a) + SIGNED(yip1E_4_uid91_sincosTest_b));
        ELSE
            yip1E_4_uid91_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_4_uid91_sincosTest_a) - SIGNED(yip1E_4_uid91_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_4_uid91_sincosTest_q <= yip1E_4_uid91_sincosTest_o(113 downto 0);

    -- yip1_4_uid95_sincosTest(BITSELECT,94)@5
    yip1_4_uid95_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_4_uid91_sincosTest_q(112 downto 0));
    yip1_4_uid95_sincosTest_b <= STD_LOGIC_VECTOR(yip1_4_uid95_sincosTest_in(112 downto 0));

    -- redist171_yip1_4_uid95_sincosTest_b_1(DELAY,1208)
    redist171_yip1_4_uid95_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_4_uid95_sincosTest_b, xout => redist171_yip1_4_uid95_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_5_uid107_sincosTest(ADDSUB,106)@6
    yip1E_5_uid107_sincosTest_s <= signOfSelectionSignal_uid99_sincosTest_q;
    yip1E_5_uid107_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist171_yip1_4_uid95_sincosTest_b_1_q(112)) & redist171_yip1_4_uid95_sincosTest_b_1_q));
    yip1E_5_uid107_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 109 => twoToMiSiXip_uid102_sincosTest_b(108)) & twoToMiSiXip_uid102_sincosTest_b));
    yip1E_5_uid107_sincosTest_combproc: PROCESS (yip1E_5_uid107_sincosTest_a, yip1E_5_uid107_sincosTest_b, yip1E_5_uid107_sincosTest_s)
    BEGIN
        IF (yip1E_5_uid107_sincosTest_s = "1") THEN
            yip1E_5_uid107_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_5_uid107_sincosTest_a) + SIGNED(yip1E_5_uid107_sincosTest_b));
        ELSE
            yip1E_5_uid107_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_5_uid107_sincosTest_a) - SIGNED(yip1E_5_uid107_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_5_uid107_sincosTest_q <= yip1E_5_uid107_sincosTest_o(113 downto 0);

    -- yip1_5_uid111_sincosTest(BITSELECT,110)@6
    yip1_5_uid111_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_5_uid107_sincosTest_q(112 downto 0));
    yip1_5_uid111_sincosTest_b <= STD_LOGIC_VECTOR(yip1_5_uid111_sincosTest_in(112 downto 0));

    -- redist168_yip1_5_uid111_sincosTest_b_1(DELAY,1205)
    redist168_yip1_5_uid111_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_5_uid111_sincosTest_b, xout => redist168_yip1_5_uid111_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid119_sincosTest(BITSELECT,118)@7
    twoToMiSiYip_uid119_sincosTest_b <= STD_LOGIC_VECTOR(redist168_yip1_5_uid111_sincosTest_b_1_q(112 downto 5));

    -- twoToMiSiYip_uid103_sincosTest(BITSELECT,102)@6
    twoToMiSiYip_uid103_sincosTest_b <= STD_LOGIC_VECTOR(redist171_yip1_4_uid95_sincosTest_b_1_q(112 downto 4));

    -- xip1E_5_uid106_sincosTest(ADDSUB,105)@6
    xip1E_5_uid106_sincosTest_s <= xMSB_uid97_sincosTest_b;
    xip1E_5_uid106_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist172_xip1_4_uid94_sincosTest_b_1_q(112)) & redist172_xip1_4_uid94_sincosTest_b_1_q));
    xip1E_5_uid106_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 109 => twoToMiSiYip_uid103_sincosTest_b(108)) & twoToMiSiYip_uid103_sincosTest_b));
    xip1E_5_uid106_sincosTest_combproc: PROCESS (xip1E_5_uid106_sincosTest_a, xip1E_5_uid106_sincosTest_b, xip1E_5_uid106_sincosTest_s)
    BEGIN
        IF (xip1E_5_uid106_sincosTest_s = "1") THEN
            xip1E_5_uid106_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_5_uid106_sincosTest_a) + SIGNED(xip1E_5_uid106_sincosTest_b));
        ELSE
            xip1E_5_uid106_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_5_uid106_sincosTest_a) - SIGNED(xip1E_5_uid106_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_5_uid106_sincosTest_q <= xip1E_5_uid106_sincosTest_o(113 downto 0);

    -- xip1_5_uid110_sincosTest(BITSELECT,109)@6
    xip1_5_uid110_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_5_uid106_sincosTest_q(112 downto 0));
    xip1_5_uid110_sincosTest_b <= STD_LOGIC_VECTOR(xip1_5_uid110_sincosTest_in(112 downto 0));

    -- redist169_xip1_5_uid110_sincosTest_b_1(DELAY,1206)
    redist169_xip1_5_uid110_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_5_uid110_sincosTest_b, xout => redist169_xip1_5_uid110_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_6_uid122_sincosTest(ADDSUB,121)@7
    xip1E_6_uid122_sincosTest_s <= xMSB_uid113_sincosTest_b;
    xip1E_6_uid122_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist169_xip1_5_uid110_sincosTest_b_1_q(112)) & redist169_xip1_5_uid110_sincosTest_b_1_q));
    xip1E_6_uid122_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 108 => twoToMiSiYip_uid119_sincosTest_b(107)) & twoToMiSiYip_uid119_sincosTest_b));
    xip1E_6_uid122_sincosTest_combproc: PROCESS (xip1E_6_uid122_sincosTest_a, xip1E_6_uid122_sincosTest_b, xip1E_6_uid122_sincosTest_s)
    BEGIN
        IF (xip1E_6_uid122_sincosTest_s = "1") THEN
            xip1E_6_uid122_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_6_uid122_sincosTest_a) + SIGNED(xip1E_6_uid122_sincosTest_b));
        ELSE
            xip1E_6_uid122_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_6_uid122_sincosTest_a) - SIGNED(xip1E_6_uid122_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_6_uid122_sincosTest_q <= xip1E_6_uid122_sincosTest_o(113 downto 0);

    -- xip1_6_uid126_sincosTest(BITSELECT,125)@7
    xip1_6_uid126_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_6_uid122_sincosTest_q(112 downto 0));
    xip1_6_uid126_sincosTest_b <= STD_LOGIC_VECTOR(xip1_6_uid126_sincosTest_in(112 downto 0));

    -- redist166_xip1_6_uid126_sincosTest_b_1(DELAY,1203)
    redist166_xip1_6_uid126_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_6_uid126_sincosTest_b, xout => redist166_xip1_6_uid126_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid134_sincosTest(BITSELECT,133)@8
    twoToMiSiXip_uid134_sincosTest_b <= STD_LOGIC_VECTOR(redist166_xip1_6_uid126_sincosTest_b_1_q(112 downto 6));

    -- signOfSelectionSignal_uid115_sincosTest(LOGICAL,114)@7
    signOfSelectionSignal_uid115_sincosTest_q <= not (xMSB_uid113_sincosTest_b);

    -- twoToMiSiXip_uid118_sincosTest(BITSELECT,117)@7
    twoToMiSiXip_uid118_sincosTest_b <= STD_LOGIC_VECTOR(redist169_xip1_5_uid110_sincosTest_b_1_q(112 downto 5));

    -- yip1E_6_uid123_sincosTest(ADDSUB,122)@7
    yip1E_6_uid123_sincosTest_s <= signOfSelectionSignal_uid115_sincosTest_q;
    yip1E_6_uid123_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist168_yip1_5_uid111_sincosTest_b_1_q(112)) & redist168_yip1_5_uid111_sincosTest_b_1_q));
    yip1E_6_uid123_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 108 => twoToMiSiXip_uid118_sincosTest_b(107)) & twoToMiSiXip_uid118_sincosTest_b));
    yip1E_6_uid123_sincosTest_combproc: PROCESS (yip1E_6_uid123_sincosTest_a, yip1E_6_uid123_sincosTest_b, yip1E_6_uid123_sincosTest_s)
    BEGIN
        IF (yip1E_6_uid123_sincosTest_s = "1") THEN
            yip1E_6_uid123_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_6_uid123_sincosTest_a) + SIGNED(yip1E_6_uid123_sincosTest_b));
        ELSE
            yip1E_6_uid123_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_6_uid123_sincosTest_a) - SIGNED(yip1E_6_uid123_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_6_uid123_sincosTest_q <= yip1E_6_uid123_sincosTest_o(113 downto 0);

    -- yip1_6_uid127_sincosTest(BITSELECT,126)@7
    yip1_6_uid127_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_6_uid123_sincosTest_q(112 downto 0));
    yip1_6_uid127_sincosTest_b <= STD_LOGIC_VECTOR(yip1_6_uid127_sincosTest_in(112 downto 0));

    -- redist165_yip1_6_uid127_sincosTest_b_1(DELAY,1202)
    redist165_yip1_6_uid127_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_6_uid127_sincosTest_b, xout => redist165_yip1_6_uid127_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_7_uid139_sincosTest(ADDSUB,138)@8
    yip1E_7_uid139_sincosTest_s <= signOfSelectionSignal_uid131_sincosTest_q;
    yip1E_7_uid139_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist165_yip1_6_uid127_sincosTest_b_1_q(112)) & redist165_yip1_6_uid127_sincosTest_b_1_q));
    yip1E_7_uid139_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 107 => twoToMiSiXip_uid134_sincosTest_b(106)) & twoToMiSiXip_uid134_sincosTest_b));
    yip1E_7_uid139_sincosTest_combproc: PROCESS (yip1E_7_uid139_sincosTest_a, yip1E_7_uid139_sincosTest_b, yip1E_7_uid139_sincosTest_s)
    BEGIN
        IF (yip1E_7_uid139_sincosTest_s = "1") THEN
            yip1E_7_uid139_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_7_uid139_sincosTest_a) + SIGNED(yip1E_7_uid139_sincosTest_b));
        ELSE
            yip1E_7_uid139_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_7_uid139_sincosTest_a) - SIGNED(yip1E_7_uid139_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_7_uid139_sincosTest_q <= yip1E_7_uid139_sincosTest_o(113 downto 0);

    -- yip1_7_uid143_sincosTest(BITSELECT,142)@8
    yip1_7_uid143_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_7_uid139_sincosTest_q(112 downto 0));
    yip1_7_uid143_sincosTest_b <= STD_LOGIC_VECTOR(yip1_7_uid143_sincosTest_in(112 downto 0));

    -- redist162_yip1_7_uid143_sincosTest_b_1(DELAY,1199)
    redist162_yip1_7_uid143_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_7_uid143_sincosTest_b, xout => redist162_yip1_7_uid143_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid151_sincosTest(BITSELECT,150)@9
    twoToMiSiYip_uid151_sincosTest_b <= STD_LOGIC_VECTOR(redist162_yip1_7_uid143_sincosTest_b_1_q(112 downto 7));

    -- twoToMiSiYip_uid135_sincosTest(BITSELECT,134)@8
    twoToMiSiYip_uid135_sincosTest_b <= STD_LOGIC_VECTOR(redist165_yip1_6_uid127_sincosTest_b_1_q(112 downto 6));

    -- xip1E_7_uid138_sincosTest(ADDSUB,137)@8
    xip1E_7_uid138_sincosTest_s <= xMSB_uid129_sincosTest_b;
    xip1E_7_uid138_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist166_xip1_6_uid126_sincosTest_b_1_q(112)) & redist166_xip1_6_uid126_sincosTest_b_1_q));
    xip1E_7_uid138_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 107 => twoToMiSiYip_uid135_sincosTest_b(106)) & twoToMiSiYip_uid135_sincosTest_b));
    xip1E_7_uid138_sincosTest_combproc: PROCESS (xip1E_7_uid138_sincosTest_a, xip1E_7_uid138_sincosTest_b, xip1E_7_uid138_sincosTest_s)
    BEGIN
        IF (xip1E_7_uid138_sincosTest_s = "1") THEN
            xip1E_7_uid138_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_7_uid138_sincosTest_a) + SIGNED(xip1E_7_uid138_sincosTest_b));
        ELSE
            xip1E_7_uid138_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_7_uid138_sincosTest_a) - SIGNED(xip1E_7_uid138_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_7_uid138_sincosTest_q <= xip1E_7_uid138_sincosTest_o(113 downto 0);

    -- xip1_7_uid142_sincosTest(BITSELECT,141)@8
    xip1_7_uid142_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_7_uid138_sincosTest_q(112 downto 0));
    xip1_7_uid142_sincosTest_b <= STD_LOGIC_VECTOR(xip1_7_uid142_sincosTest_in(112 downto 0));

    -- redist163_xip1_7_uid142_sincosTest_b_1(DELAY,1200)
    redist163_xip1_7_uid142_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_7_uid142_sincosTest_b, xout => redist163_xip1_7_uid142_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_8_uid154_sincosTest(ADDSUB,153)@9
    xip1E_8_uid154_sincosTest_s <= xMSB_uid145_sincosTest_b;
    xip1E_8_uid154_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist163_xip1_7_uid142_sincosTest_b_1_q(112)) & redist163_xip1_7_uid142_sincosTest_b_1_q));
    xip1E_8_uid154_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 106 => twoToMiSiYip_uid151_sincosTest_b(105)) & twoToMiSiYip_uid151_sincosTest_b));
    xip1E_8_uid154_sincosTest_combproc: PROCESS (xip1E_8_uid154_sincosTest_a, xip1E_8_uid154_sincosTest_b, xip1E_8_uid154_sincosTest_s)
    BEGIN
        IF (xip1E_8_uid154_sincosTest_s = "1") THEN
            xip1E_8_uid154_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_8_uid154_sincosTest_a) + SIGNED(xip1E_8_uid154_sincosTest_b));
        ELSE
            xip1E_8_uid154_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_8_uid154_sincosTest_a) - SIGNED(xip1E_8_uid154_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_8_uid154_sincosTest_q <= xip1E_8_uid154_sincosTest_o(113 downto 0);

    -- xip1_8_uid158_sincosTest(BITSELECT,157)@9
    xip1_8_uid158_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_8_uid154_sincosTest_q(112 downto 0));
    xip1_8_uid158_sincosTest_b <= STD_LOGIC_VECTOR(xip1_8_uid158_sincosTest_in(112 downto 0));

    -- redist160_xip1_8_uid158_sincosTest_b_1(DELAY,1197)
    redist160_xip1_8_uid158_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_8_uid158_sincosTest_b, xout => redist160_xip1_8_uid158_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid166_sincosTest(BITSELECT,165)@10
    twoToMiSiXip_uid166_sincosTest_b <= STD_LOGIC_VECTOR(redist160_xip1_8_uid158_sincosTest_b_1_q(112 downto 8));

    -- signOfSelectionSignal_uid147_sincosTest(LOGICAL,146)@9
    signOfSelectionSignal_uid147_sincosTest_q <= not (xMSB_uid145_sincosTest_b);

    -- twoToMiSiXip_uid150_sincosTest(BITSELECT,149)@9
    twoToMiSiXip_uid150_sincosTest_b <= STD_LOGIC_VECTOR(redist163_xip1_7_uid142_sincosTest_b_1_q(112 downto 7));

    -- yip1E_8_uid155_sincosTest(ADDSUB,154)@9
    yip1E_8_uid155_sincosTest_s <= signOfSelectionSignal_uid147_sincosTest_q;
    yip1E_8_uid155_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist162_yip1_7_uid143_sincosTest_b_1_q(112)) & redist162_yip1_7_uid143_sincosTest_b_1_q));
    yip1E_8_uid155_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 106 => twoToMiSiXip_uid150_sincosTest_b(105)) & twoToMiSiXip_uid150_sincosTest_b));
    yip1E_8_uid155_sincosTest_combproc: PROCESS (yip1E_8_uid155_sincosTest_a, yip1E_8_uid155_sincosTest_b, yip1E_8_uid155_sincosTest_s)
    BEGIN
        IF (yip1E_8_uid155_sincosTest_s = "1") THEN
            yip1E_8_uid155_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_8_uid155_sincosTest_a) + SIGNED(yip1E_8_uid155_sincosTest_b));
        ELSE
            yip1E_8_uid155_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_8_uid155_sincosTest_a) - SIGNED(yip1E_8_uid155_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_8_uid155_sincosTest_q <= yip1E_8_uid155_sincosTest_o(113 downto 0);

    -- yip1_8_uid159_sincosTest(BITSELECT,158)@9
    yip1_8_uid159_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_8_uid155_sincosTest_q(112 downto 0));
    yip1_8_uid159_sincosTest_b <= STD_LOGIC_VECTOR(yip1_8_uid159_sincosTest_in(112 downto 0));

    -- redist159_yip1_8_uid159_sincosTest_b_1(DELAY,1196)
    redist159_yip1_8_uid159_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_8_uid159_sincosTest_b, xout => redist159_yip1_8_uid159_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_9_uid171_sincosTest(ADDSUB,170)@10
    yip1E_9_uid171_sincosTest_s <= signOfSelectionSignal_uid163_sincosTest_q;
    yip1E_9_uid171_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist159_yip1_8_uid159_sincosTest_b_1_q(112)) & redist159_yip1_8_uid159_sincosTest_b_1_q));
    yip1E_9_uid171_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 105 => twoToMiSiXip_uid166_sincosTest_b(104)) & twoToMiSiXip_uid166_sincosTest_b));
    yip1E_9_uid171_sincosTest_combproc: PROCESS (yip1E_9_uid171_sincosTest_a, yip1E_9_uid171_sincosTest_b, yip1E_9_uid171_sincosTest_s)
    BEGIN
        IF (yip1E_9_uid171_sincosTest_s = "1") THEN
            yip1E_9_uid171_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_9_uid171_sincosTest_a) + SIGNED(yip1E_9_uid171_sincosTest_b));
        ELSE
            yip1E_9_uid171_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_9_uid171_sincosTest_a) - SIGNED(yip1E_9_uid171_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_9_uid171_sincosTest_q <= yip1E_9_uid171_sincosTest_o(113 downto 0);

    -- yip1_9_uid175_sincosTest(BITSELECT,174)@10
    yip1_9_uid175_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_9_uid171_sincosTest_q(112 downto 0));
    yip1_9_uid175_sincosTest_b <= STD_LOGIC_VECTOR(yip1_9_uid175_sincosTest_in(112 downto 0));

    -- redist156_yip1_9_uid175_sincosTest_b_1(DELAY,1193)
    redist156_yip1_9_uid175_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_9_uid175_sincosTest_b, xout => redist156_yip1_9_uid175_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid183_sincosTest(BITSELECT,182)@11
    twoToMiSiYip_uid183_sincosTest_b <= STD_LOGIC_VECTOR(redist156_yip1_9_uid175_sincosTest_b_1_q(112 downto 9));

    -- twoToMiSiYip_uid167_sincosTest(BITSELECT,166)@10
    twoToMiSiYip_uid167_sincosTest_b <= STD_LOGIC_VECTOR(redist159_yip1_8_uid159_sincosTest_b_1_q(112 downto 8));

    -- xip1E_9_uid170_sincosTest(ADDSUB,169)@10
    xip1E_9_uid170_sincosTest_s <= xMSB_uid161_sincosTest_b;
    xip1E_9_uid170_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist160_xip1_8_uid158_sincosTest_b_1_q(112)) & redist160_xip1_8_uid158_sincosTest_b_1_q));
    xip1E_9_uid170_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 105 => twoToMiSiYip_uid167_sincosTest_b(104)) & twoToMiSiYip_uid167_sincosTest_b));
    xip1E_9_uid170_sincosTest_combproc: PROCESS (xip1E_9_uid170_sincosTest_a, xip1E_9_uid170_sincosTest_b, xip1E_9_uid170_sincosTest_s)
    BEGIN
        IF (xip1E_9_uid170_sincosTest_s = "1") THEN
            xip1E_9_uid170_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_9_uid170_sincosTest_a) + SIGNED(xip1E_9_uid170_sincosTest_b));
        ELSE
            xip1E_9_uid170_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_9_uid170_sincosTest_a) - SIGNED(xip1E_9_uid170_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_9_uid170_sincosTest_q <= xip1E_9_uid170_sincosTest_o(113 downto 0);

    -- xip1_9_uid174_sincosTest(BITSELECT,173)@10
    xip1_9_uid174_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_9_uid170_sincosTest_q(112 downto 0));
    xip1_9_uid174_sincosTest_b <= STD_LOGIC_VECTOR(xip1_9_uid174_sincosTest_in(112 downto 0));

    -- redist157_xip1_9_uid174_sincosTest_b_1(DELAY,1194)
    redist157_xip1_9_uid174_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_9_uid174_sincosTest_b, xout => redist157_xip1_9_uid174_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_10_uid186_sincosTest(ADDSUB,185)@11
    xip1E_10_uid186_sincosTest_s <= xMSB_uid177_sincosTest_b;
    xip1E_10_uid186_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist157_xip1_9_uid174_sincosTest_b_1_q(112)) & redist157_xip1_9_uid174_sincosTest_b_1_q));
    xip1E_10_uid186_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 104 => twoToMiSiYip_uid183_sincosTest_b(103)) & twoToMiSiYip_uid183_sincosTest_b));
    xip1E_10_uid186_sincosTest_combproc: PROCESS (xip1E_10_uid186_sincosTest_a, xip1E_10_uid186_sincosTest_b, xip1E_10_uid186_sincosTest_s)
    BEGIN
        IF (xip1E_10_uid186_sincosTest_s = "1") THEN
            xip1E_10_uid186_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_10_uid186_sincosTest_a) + SIGNED(xip1E_10_uid186_sincosTest_b));
        ELSE
            xip1E_10_uid186_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_10_uid186_sincosTest_a) - SIGNED(xip1E_10_uid186_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_10_uid186_sincosTest_q <= xip1E_10_uid186_sincosTest_o(113 downto 0);

    -- xip1_10_uid190_sincosTest(BITSELECT,189)@11
    xip1_10_uid190_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_10_uid186_sincosTest_q(112 downto 0));
    xip1_10_uid190_sincosTest_b <= STD_LOGIC_VECTOR(xip1_10_uid190_sincosTest_in(112 downto 0));

    -- redist154_xip1_10_uid190_sincosTest_b_1(DELAY,1191)
    redist154_xip1_10_uid190_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_10_uid190_sincosTest_b, xout => redist154_xip1_10_uid190_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid198_sincosTest(BITSELECT,197)@12
    twoToMiSiXip_uid198_sincosTest_b <= STD_LOGIC_VECTOR(redist154_xip1_10_uid190_sincosTest_b_1_q(112 downto 10));

    -- signOfSelectionSignal_uid179_sincosTest(LOGICAL,178)@11
    signOfSelectionSignal_uid179_sincosTest_q <= not (xMSB_uid177_sincosTest_b);

    -- twoToMiSiXip_uid182_sincosTest(BITSELECT,181)@11
    twoToMiSiXip_uid182_sincosTest_b <= STD_LOGIC_VECTOR(redist157_xip1_9_uid174_sincosTest_b_1_q(112 downto 9));

    -- yip1E_10_uid187_sincosTest(ADDSUB,186)@11
    yip1E_10_uid187_sincosTest_s <= signOfSelectionSignal_uid179_sincosTest_q;
    yip1E_10_uid187_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist156_yip1_9_uid175_sincosTest_b_1_q(112)) & redist156_yip1_9_uid175_sincosTest_b_1_q));
    yip1E_10_uid187_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 104 => twoToMiSiXip_uid182_sincosTest_b(103)) & twoToMiSiXip_uid182_sincosTest_b));
    yip1E_10_uid187_sincosTest_combproc: PROCESS (yip1E_10_uid187_sincosTest_a, yip1E_10_uid187_sincosTest_b, yip1E_10_uid187_sincosTest_s)
    BEGIN
        IF (yip1E_10_uid187_sincosTest_s = "1") THEN
            yip1E_10_uid187_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_10_uid187_sincosTest_a) + SIGNED(yip1E_10_uid187_sincosTest_b));
        ELSE
            yip1E_10_uid187_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_10_uid187_sincosTest_a) - SIGNED(yip1E_10_uid187_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_10_uid187_sincosTest_q <= yip1E_10_uid187_sincosTest_o(113 downto 0);

    -- yip1_10_uid191_sincosTest(BITSELECT,190)@11
    yip1_10_uid191_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_10_uid187_sincosTest_q(112 downto 0));
    yip1_10_uid191_sincosTest_b <= STD_LOGIC_VECTOR(yip1_10_uid191_sincosTest_in(112 downto 0));

    -- redist153_yip1_10_uid191_sincosTest_b_1(DELAY,1190)
    redist153_yip1_10_uid191_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_10_uid191_sincosTest_b, xout => redist153_yip1_10_uid191_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_11_uid203_sincosTest(ADDSUB,202)@12
    yip1E_11_uid203_sincosTest_s <= signOfSelectionSignal_uid195_sincosTest_q;
    yip1E_11_uid203_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist153_yip1_10_uid191_sincosTest_b_1_q(112)) & redist153_yip1_10_uid191_sincosTest_b_1_q));
    yip1E_11_uid203_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 103 => twoToMiSiXip_uid198_sincosTest_b(102)) & twoToMiSiXip_uid198_sincosTest_b));
    yip1E_11_uid203_sincosTest_combproc: PROCESS (yip1E_11_uid203_sincosTest_a, yip1E_11_uid203_sincosTest_b, yip1E_11_uid203_sincosTest_s)
    BEGIN
        IF (yip1E_11_uid203_sincosTest_s = "1") THEN
            yip1E_11_uid203_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_11_uid203_sincosTest_a) + SIGNED(yip1E_11_uid203_sincosTest_b));
        ELSE
            yip1E_11_uid203_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_11_uid203_sincosTest_a) - SIGNED(yip1E_11_uid203_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_11_uid203_sincosTest_q <= yip1E_11_uid203_sincosTest_o(113 downto 0);

    -- yip1_11_uid207_sincosTest(BITSELECT,206)@12
    yip1_11_uid207_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_11_uid203_sincosTest_q(112 downto 0));
    yip1_11_uid207_sincosTest_b <= STD_LOGIC_VECTOR(yip1_11_uid207_sincosTest_in(112 downto 0));

    -- redist150_yip1_11_uid207_sincosTest_b_1(DELAY,1187)
    redist150_yip1_11_uid207_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_11_uid207_sincosTest_b, xout => redist150_yip1_11_uid207_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid215_sincosTest(BITSELECT,214)@13
    twoToMiSiYip_uid215_sincosTest_b <= STD_LOGIC_VECTOR(redist150_yip1_11_uid207_sincosTest_b_1_q(112 downto 11));

    -- twoToMiSiYip_uid199_sincosTest(BITSELECT,198)@12
    twoToMiSiYip_uid199_sincosTest_b <= STD_LOGIC_VECTOR(redist153_yip1_10_uid191_sincosTest_b_1_q(112 downto 10));

    -- xip1E_11_uid202_sincosTest(ADDSUB,201)@12
    xip1E_11_uid202_sincosTest_s <= xMSB_uid193_sincosTest_b;
    xip1E_11_uid202_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist154_xip1_10_uid190_sincosTest_b_1_q(112)) & redist154_xip1_10_uid190_sincosTest_b_1_q));
    xip1E_11_uid202_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 103 => twoToMiSiYip_uid199_sincosTest_b(102)) & twoToMiSiYip_uid199_sincosTest_b));
    xip1E_11_uid202_sincosTest_combproc: PROCESS (xip1E_11_uid202_sincosTest_a, xip1E_11_uid202_sincosTest_b, xip1E_11_uid202_sincosTest_s)
    BEGIN
        IF (xip1E_11_uid202_sincosTest_s = "1") THEN
            xip1E_11_uid202_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_11_uid202_sincosTest_a) + SIGNED(xip1E_11_uid202_sincosTest_b));
        ELSE
            xip1E_11_uid202_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_11_uid202_sincosTest_a) - SIGNED(xip1E_11_uid202_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_11_uid202_sincosTest_q <= xip1E_11_uid202_sincosTest_o(113 downto 0);

    -- xip1_11_uid206_sincosTest(BITSELECT,205)@12
    xip1_11_uid206_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_11_uid202_sincosTest_q(112 downto 0));
    xip1_11_uid206_sincosTest_b <= STD_LOGIC_VECTOR(xip1_11_uid206_sincosTest_in(112 downto 0));

    -- redist151_xip1_11_uid206_sincosTest_b_1(DELAY,1188)
    redist151_xip1_11_uid206_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_11_uid206_sincosTest_b, xout => redist151_xip1_11_uid206_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_12_uid218_sincosTest(ADDSUB,217)@13
    xip1E_12_uid218_sincosTest_s <= xMSB_uid209_sincosTest_b;
    xip1E_12_uid218_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist151_xip1_11_uid206_sincosTest_b_1_q(112)) & redist151_xip1_11_uid206_sincosTest_b_1_q));
    xip1E_12_uid218_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 102 => twoToMiSiYip_uid215_sincosTest_b(101)) & twoToMiSiYip_uid215_sincosTest_b));
    xip1E_12_uid218_sincosTest_combproc: PROCESS (xip1E_12_uid218_sincosTest_a, xip1E_12_uid218_sincosTest_b, xip1E_12_uid218_sincosTest_s)
    BEGIN
        IF (xip1E_12_uid218_sincosTest_s = "1") THEN
            xip1E_12_uid218_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_12_uid218_sincosTest_a) + SIGNED(xip1E_12_uid218_sincosTest_b));
        ELSE
            xip1E_12_uid218_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_12_uid218_sincosTest_a) - SIGNED(xip1E_12_uid218_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_12_uid218_sincosTest_q <= xip1E_12_uid218_sincosTest_o(113 downto 0);

    -- xip1_12_uid222_sincosTest(BITSELECT,221)@13
    xip1_12_uid222_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_12_uid218_sincosTest_q(112 downto 0));
    xip1_12_uid222_sincosTest_b <= STD_LOGIC_VECTOR(xip1_12_uid222_sincosTest_in(112 downto 0));

    -- redist148_xip1_12_uid222_sincosTest_b_1(DELAY,1185)
    redist148_xip1_12_uid222_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_12_uid222_sincosTest_b, xout => redist148_xip1_12_uid222_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid230_sincosTest(BITSELECT,229)@14
    twoToMiSiXip_uid230_sincosTest_b <= STD_LOGIC_VECTOR(redist148_xip1_12_uid222_sincosTest_b_1_q(112 downto 12));

    -- signOfSelectionSignal_uid211_sincosTest(LOGICAL,210)@13
    signOfSelectionSignal_uid211_sincosTest_q <= not (xMSB_uid209_sincosTest_b);

    -- twoToMiSiXip_uid214_sincosTest(BITSELECT,213)@13
    twoToMiSiXip_uid214_sincosTest_b <= STD_LOGIC_VECTOR(redist151_xip1_11_uid206_sincosTest_b_1_q(112 downto 11));

    -- yip1E_12_uid219_sincosTest(ADDSUB,218)@13
    yip1E_12_uid219_sincosTest_s <= signOfSelectionSignal_uid211_sincosTest_q;
    yip1E_12_uid219_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist150_yip1_11_uid207_sincosTest_b_1_q(112)) & redist150_yip1_11_uid207_sincosTest_b_1_q));
    yip1E_12_uid219_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 102 => twoToMiSiXip_uid214_sincosTest_b(101)) & twoToMiSiXip_uid214_sincosTest_b));
    yip1E_12_uid219_sincosTest_combproc: PROCESS (yip1E_12_uid219_sincosTest_a, yip1E_12_uid219_sincosTest_b, yip1E_12_uid219_sincosTest_s)
    BEGIN
        IF (yip1E_12_uid219_sincosTest_s = "1") THEN
            yip1E_12_uid219_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_12_uid219_sincosTest_a) + SIGNED(yip1E_12_uid219_sincosTest_b));
        ELSE
            yip1E_12_uid219_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_12_uid219_sincosTest_a) - SIGNED(yip1E_12_uid219_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_12_uid219_sincosTest_q <= yip1E_12_uid219_sincosTest_o(113 downto 0);

    -- yip1_12_uid223_sincosTest(BITSELECT,222)@13
    yip1_12_uid223_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_12_uid219_sincosTest_q(112 downto 0));
    yip1_12_uid223_sincosTest_b <= STD_LOGIC_VECTOR(yip1_12_uid223_sincosTest_in(112 downto 0));

    -- redist147_yip1_12_uid223_sincosTest_b_1(DELAY,1184)
    redist147_yip1_12_uid223_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_12_uid223_sincosTest_b, xout => redist147_yip1_12_uid223_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_13_uid235_sincosTest(ADDSUB,234)@14
    yip1E_13_uid235_sincosTest_s <= signOfSelectionSignal_uid227_sincosTest_q;
    yip1E_13_uid235_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist147_yip1_12_uid223_sincosTest_b_1_q(112)) & redist147_yip1_12_uid223_sincosTest_b_1_q));
    yip1E_13_uid235_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 101 => twoToMiSiXip_uid230_sincosTest_b(100)) & twoToMiSiXip_uid230_sincosTest_b));
    yip1E_13_uid235_sincosTest_combproc: PROCESS (yip1E_13_uid235_sincosTest_a, yip1E_13_uid235_sincosTest_b, yip1E_13_uid235_sincosTest_s)
    BEGIN
        IF (yip1E_13_uid235_sincosTest_s = "1") THEN
            yip1E_13_uid235_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_13_uid235_sincosTest_a) + SIGNED(yip1E_13_uid235_sincosTest_b));
        ELSE
            yip1E_13_uid235_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_13_uid235_sincosTest_a) - SIGNED(yip1E_13_uid235_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_13_uid235_sincosTest_q <= yip1E_13_uid235_sincosTest_o(113 downto 0);

    -- yip1_13_uid239_sincosTest(BITSELECT,238)@14
    yip1_13_uid239_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_13_uid235_sincosTest_q(112 downto 0));
    yip1_13_uid239_sincosTest_b <= STD_LOGIC_VECTOR(yip1_13_uid239_sincosTest_in(112 downto 0));

    -- redist144_yip1_13_uid239_sincosTest_b_1(DELAY,1181)
    redist144_yip1_13_uid239_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_13_uid239_sincosTest_b, xout => redist144_yip1_13_uid239_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid247_sincosTest(BITSELECT,246)@15
    twoToMiSiYip_uid247_sincosTest_b <= STD_LOGIC_VECTOR(redist144_yip1_13_uid239_sincosTest_b_1_q(112 downto 13));

    -- twoToMiSiYip_uid231_sincosTest(BITSELECT,230)@14
    twoToMiSiYip_uid231_sincosTest_b <= STD_LOGIC_VECTOR(redist147_yip1_12_uid223_sincosTest_b_1_q(112 downto 12));

    -- xip1E_13_uid234_sincosTest(ADDSUB,233)@14
    xip1E_13_uid234_sincosTest_s <= xMSB_uid225_sincosTest_b;
    xip1E_13_uid234_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist148_xip1_12_uid222_sincosTest_b_1_q(112)) & redist148_xip1_12_uid222_sincosTest_b_1_q));
    xip1E_13_uid234_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 101 => twoToMiSiYip_uid231_sincosTest_b(100)) & twoToMiSiYip_uid231_sincosTest_b));
    xip1E_13_uid234_sincosTest_combproc: PROCESS (xip1E_13_uid234_sincosTest_a, xip1E_13_uid234_sincosTest_b, xip1E_13_uid234_sincosTest_s)
    BEGIN
        IF (xip1E_13_uid234_sincosTest_s = "1") THEN
            xip1E_13_uid234_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_13_uid234_sincosTest_a) + SIGNED(xip1E_13_uid234_sincosTest_b));
        ELSE
            xip1E_13_uid234_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_13_uid234_sincosTest_a) - SIGNED(xip1E_13_uid234_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_13_uid234_sincosTest_q <= xip1E_13_uid234_sincosTest_o(113 downto 0);

    -- xip1_13_uid238_sincosTest(BITSELECT,237)@14
    xip1_13_uid238_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_13_uid234_sincosTest_q(112 downto 0));
    xip1_13_uid238_sincosTest_b <= STD_LOGIC_VECTOR(xip1_13_uid238_sincosTest_in(112 downto 0));

    -- redist145_xip1_13_uid238_sincosTest_b_1(DELAY,1182)
    redist145_xip1_13_uid238_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_13_uid238_sincosTest_b, xout => redist145_xip1_13_uid238_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_14_uid250_sincosTest(ADDSUB,249)@15
    xip1E_14_uid250_sincosTest_s <= xMSB_uid241_sincosTest_b;
    xip1E_14_uid250_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist145_xip1_13_uid238_sincosTest_b_1_q(112)) & redist145_xip1_13_uid238_sincosTest_b_1_q));
    xip1E_14_uid250_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 100 => twoToMiSiYip_uid247_sincosTest_b(99)) & twoToMiSiYip_uid247_sincosTest_b));
    xip1E_14_uid250_sincosTest_combproc: PROCESS (xip1E_14_uid250_sincosTest_a, xip1E_14_uid250_sincosTest_b, xip1E_14_uid250_sincosTest_s)
    BEGIN
        IF (xip1E_14_uid250_sincosTest_s = "1") THEN
            xip1E_14_uid250_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_14_uid250_sincosTest_a) + SIGNED(xip1E_14_uid250_sincosTest_b));
        ELSE
            xip1E_14_uid250_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_14_uid250_sincosTest_a) - SIGNED(xip1E_14_uid250_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_14_uid250_sincosTest_q <= xip1E_14_uid250_sincosTest_o(113 downto 0);

    -- xip1_14_uid254_sincosTest(BITSELECT,253)@15
    xip1_14_uid254_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_14_uid250_sincosTest_q(112 downto 0));
    xip1_14_uid254_sincosTest_b <= STD_LOGIC_VECTOR(xip1_14_uid254_sincosTest_in(112 downto 0));

    -- redist142_xip1_14_uid254_sincosTest_b_1(DELAY,1179)
    redist142_xip1_14_uid254_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_14_uid254_sincosTest_b, xout => redist142_xip1_14_uid254_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid262_sincosTest(BITSELECT,261)@16
    twoToMiSiXip_uid262_sincosTest_b <= STD_LOGIC_VECTOR(redist142_xip1_14_uid254_sincosTest_b_1_q(112 downto 14));

    -- signOfSelectionSignal_uid243_sincosTest(LOGICAL,242)@15
    signOfSelectionSignal_uid243_sincosTest_q <= not (xMSB_uid241_sincosTest_b);

    -- twoToMiSiXip_uid246_sincosTest(BITSELECT,245)@15
    twoToMiSiXip_uid246_sincosTest_b <= STD_LOGIC_VECTOR(redist145_xip1_13_uid238_sincosTest_b_1_q(112 downto 13));

    -- yip1E_14_uid251_sincosTest(ADDSUB,250)@15
    yip1E_14_uid251_sincosTest_s <= signOfSelectionSignal_uid243_sincosTest_q;
    yip1E_14_uid251_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist144_yip1_13_uid239_sincosTest_b_1_q(112)) & redist144_yip1_13_uid239_sincosTest_b_1_q));
    yip1E_14_uid251_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 100 => twoToMiSiXip_uid246_sincosTest_b(99)) & twoToMiSiXip_uid246_sincosTest_b));
    yip1E_14_uid251_sincosTest_combproc: PROCESS (yip1E_14_uid251_sincosTest_a, yip1E_14_uid251_sincosTest_b, yip1E_14_uid251_sincosTest_s)
    BEGIN
        IF (yip1E_14_uid251_sincosTest_s = "1") THEN
            yip1E_14_uid251_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_14_uid251_sincosTest_a) + SIGNED(yip1E_14_uid251_sincosTest_b));
        ELSE
            yip1E_14_uid251_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_14_uid251_sincosTest_a) - SIGNED(yip1E_14_uid251_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_14_uid251_sincosTest_q <= yip1E_14_uid251_sincosTest_o(113 downto 0);

    -- yip1_14_uid255_sincosTest(BITSELECT,254)@15
    yip1_14_uid255_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_14_uid251_sincosTest_q(112 downto 0));
    yip1_14_uid255_sincosTest_b <= STD_LOGIC_VECTOR(yip1_14_uid255_sincosTest_in(112 downto 0));

    -- redist141_yip1_14_uid255_sincosTest_b_1(DELAY,1178)
    redist141_yip1_14_uid255_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_14_uid255_sincosTest_b, xout => redist141_yip1_14_uid255_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_15_uid267_sincosTest(ADDSUB,266)@16
    yip1E_15_uid267_sincosTest_s <= signOfSelectionSignal_uid259_sincosTest_q;
    yip1E_15_uid267_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist141_yip1_14_uid255_sincosTest_b_1_q(112)) & redist141_yip1_14_uid255_sincosTest_b_1_q));
    yip1E_15_uid267_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 99 => twoToMiSiXip_uid262_sincosTest_b(98)) & twoToMiSiXip_uid262_sincosTest_b));
    yip1E_15_uid267_sincosTest_combproc: PROCESS (yip1E_15_uid267_sincosTest_a, yip1E_15_uid267_sincosTest_b, yip1E_15_uid267_sincosTest_s)
    BEGIN
        IF (yip1E_15_uid267_sincosTest_s = "1") THEN
            yip1E_15_uid267_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_15_uid267_sincosTest_a) + SIGNED(yip1E_15_uid267_sincosTest_b));
        ELSE
            yip1E_15_uid267_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_15_uid267_sincosTest_a) - SIGNED(yip1E_15_uid267_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_15_uid267_sincosTest_q <= yip1E_15_uid267_sincosTest_o(113 downto 0);

    -- yip1_15_uid271_sincosTest(BITSELECT,270)@16
    yip1_15_uid271_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_15_uid267_sincosTest_q(112 downto 0));
    yip1_15_uid271_sincosTest_b <= STD_LOGIC_VECTOR(yip1_15_uid271_sincosTest_in(112 downto 0));

    -- redist138_yip1_15_uid271_sincosTest_b_1(DELAY,1175)
    redist138_yip1_15_uid271_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_15_uid271_sincosTest_b, xout => redist138_yip1_15_uid271_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid279_sincosTest(BITSELECT,278)@17
    twoToMiSiYip_uid279_sincosTest_b <= STD_LOGIC_VECTOR(redist138_yip1_15_uid271_sincosTest_b_1_q(112 downto 15));

    -- twoToMiSiYip_uid263_sincosTest(BITSELECT,262)@16
    twoToMiSiYip_uid263_sincosTest_b <= STD_LOGIC_VECTOR(redist141_yip1_14_uid255_sincosTest_b_1_q(112 downto 14));

    -- xip1E_15_uid266_sincosTest(ADDSUB,265)@16
    xip1E_15_uid266_sincosTest_s <= xMSB_uid257_sincosTest_b;
    xip1E_15_uid266_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist142_xip1_14_uid254_sincosTest_b_1_q(112)) & redist142_xip1_14_uid254_sincosTest_b_1_q));
    xip1E_15_uid266_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 99 => twoToMiSiYip_uid263_sincosTest_b(98)) & twoToMiSiYip_uid263_sincosTest_b));
    xip1E_15_uid266_sincosTest_combproc: PROCESS (xip1E_15_uid266_sincosTest_a, xip1E_15_uid266_sincosTest_b, xip1E_15_uid266_sincosTest_s)
    BEGIN
        IF (xip1E_15_uid266_sincosTest_s = "1") THEN
            xip1E_15_uid266_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_15_uid266_sincosTest_a) + SIGNED(xip1E_15_uid266_sincosTest_b));
        ELSE
            xip1E_15_uid266_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_15_uid266_sincosTest_a) - SIGNED(xip1E_15_uid266_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_15_uid266_sincosTest_q <= xip1E_15_uid266_sincosTest_o(113 downto 0);

    -- xip1_15_uid270_sincosTest(BITSELECT,269)@16
    xip1_15_uid270_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_15_uid266_sincosTest_q(112 downto 0));
    xip1_15_uid270_sincosTest_b <= STD_LOGIC_VECTOR(xip1_15_uid270_sincosTest_in(112 downto 0));

    -- redist139_xip1_15_uid270_sincosTest_b_1(DELAY,1176)
    redist139_xip1_15_uid270_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_15_uid270_sincosTest_b, xout => redist139_xip1_15_uid270_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_16_uid282_sincosTest(ADDSUB,281)@17
    xip1E_16_uid282_sincosTest_s <= xMSB_uid273_sincosTest_b;
    xip1E_16_uid282_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist139_xip1_15_uid270_sincosTest_b_1_q(112)) & redist139_xip1_15_uid270_sincosTest_b_1_q));
    xip1E_16_uid282_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 98 => twoToMiSiYip_uid279_sincosTest_b(97)) & twoToMiSiYip_uid279_sincosTest_b));
    xip1E_16_uid282_sincosTest_combproc: PROCESS (xip1E_16_uid282_sincosTest_a, xip1E_16_uid282_sincosTest_b, xip1E_16_uid282_sincosTest_s)
    BEGIN
        IF (xip1E_16_uid282_sincosTest_s = "1") THEN
            xip1E_16_uid282_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_16_uid282_sincosTest_a) + SIGNED(xip1E_16_uid282_sincosTest_b));
        ELSE
            xip1E_16_uid282_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_16_uid282_sincosTest_a) - SIGNED(xip1E_16_uid282_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_16_uid282_sincosTest_q <= xip1E_16_uid282_sincosTest_o(113 downto 0);

    -- xip1_16_uid286_sincosTest(BITSELECT,285)@17
    xip1_16_uid286_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_16_uid282_sincosTest_q(112 downto 0));
    xip1_16_uid286_sincosTest_b <= STD_LOGIC_VECTOR(xip1_16_uid286_sincosTest_in(112 downto 0));

    -- redist136_xip1_16_uid286_sincosTest_b_1(DELAY,1173)
    redist136_xip1_16_uid286_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_16_uid286_sincosTest_b, xout => redist136_xip1_16_uid286_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid294_sincosTest(BITSELECT,293)@18
    twoToMiSiXip_uid294_sincosTest_b <= STD_LOGIC_VECTOR(redist136_xip1_16_uid286_sincosTest_b_1_q(112 downto 16));

    -- signOfSelectionSignal_uid275_sincosTest(LOGICAL,274)@17
    signOfSelectionSignal_uid275_sincosTest_q <= not (xMSB_uid273_sincosTest_b);

    -- twoToMiSiXip_uid278_sincosTest(BITSELECT,277)@17
    twoToMiSiXip_uid278_sincosTest_b <= STD_LOGIC_VECTOR(redist139_xip1_15_uid270_sincosTest_b_1_q(112 downto 15));

    -- yip1E_16_uid283_sincosTest(ADDSUB,282)@17
    yip1E_16_uid283_sincosTest_s <= signOfSelectionSignal_uid275_sincosTest_q;
    yip1E_16_uid283_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist138_yip1_15_uid271_sincosTest_b_1_q(112)) & redist138_yip1_15_uid271_sincosTest_b_1_q));
    yip1E_16_uid283_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 98 => twoToMiSiXip_uid278_sincosTest_b(97)) & twoToMiSiXip_uid278_sincosTest_b));
    yip1E_16_uid283_sincosTest_combproc: PROCESS (yip1E_16_uid283_sincosTest_a, yip1E_16_uid283_sincosTest_b, yip1E_16_uid283_sincosTest_s)
    BEGIN
        IF (yip1E_16_uid283_sincosTest_s = "1") THEN
            yip1E_16_uid283_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_16_uid283_sincosTest_a) + SIGNED(yip1E_16_uid283_sincosTest_b));
        ELSE
            yip1E_16_uid283_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_16_uid283_sincosTest_a) - SIGNED(yip1E_16_uid283_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_16_uid283_sincosTest_q <= yip1E_16_uid283_sincosTest_o(113 downto 0);

    -- yip1_16_uid287_sincosTest(BITSELECT,286)@17
    yip1_16_uid287_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_16_uid283_sincosTest_q(112 downto 0));
    yip1_16_uid287_sincosTest_b <= STD_LOGIC_VECTOR(yip1_16_uid287_sincosTest_in(112 downto 0));

    -- redist135_yip1_16_uid287_sincosTest_b_1(DELAY,1172)
    redist135_yip1_16_uid287_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_16_uid287_sincosTest_b, xout => redist135_yip1_16_uid287_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_17_uid299_sincosTest(ADDSUB,298)@18
    yip1E_17_uid299_sincosTest_s <= signOfSelectionSignal_uid291_sincosTest_q;
    yip1E_17_uid299_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist135_yip1_16_uid287_sincosTest_b_1_q(112)) & redist135_yip1_16_uid287_sincosTest_b_1_q));
    yip1E_17_uid299_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 97 => twoToMiSiXip_uid294_sincosTest_b(96)) & twoToMiSiXip_uid294_sincosTest_b));
    yip1E_17_uid299_sincosTest_combproc: PROCESS (yip1E_17_uid299_sincosTest_a, yip1E_17_uid299_sincosTest_b, yip1E_17_uid299_sincosTest_s)
    BEGIN
        IF (yip1E_17_uid299_sincosTest_s = "1") THEN
            yip1E_17_uid299_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_17_uid299_sincosTest_a) + SIGNED(yip1E_17_uid299_sincosTest_b));
        ELSE
            yip1E_17_uid299_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_17_uid299_sincosTest_a) - SIGNED(yip1E_17_uid299_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_17_uid299_sincosTest_q <= yip1E_17_uid299_sincosTest_o(113 downto 0);

    -- yip1_17_uid303_sincosTest(BITSELECT,302)@18
    yip1_17_uid303_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_17_uid299_sincosTest_q(112 downto 0));
    yip1_17_uid303_sincosTest_b <= STD_LOGIC_VECTOR(yip1_17_uid303_sincosTest_in(112 downto 0));

    -- redist132_yip1_17_uid303_sincosTest_b_1(DELAY,1169)
    redist132_yip1_17_uid303_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_17_uid303_sincosTest_b, xout => redist132_yip1_17_uid303_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid311_sincosTest(BITSELECT,310)@19
    twoToMiSiYip_uid311_sincosTest_b <= STD_LOGIC_VECTOR(redist132_yip1_17_uid303_sincosTest_b_1_q(112 downto 17));

    -- twoToMiSiYip_uid295_sincosTest(BITSELECT,294)@18
    twoToMiSiYip_uid295_sincosTest_b <= STD_LOGIC_VECTOR(redist135_yip1_16_uid287_sincosTest_b_1_q(112 downto 16));

    -- xip1E_17_uid298_sincosTest(ADDSUB,297)@18
    xip1E_17_uid298_sincosTest_s <= xMSB_uid289_sincosTest_b;
    xip1E_17_uid298_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist136_xip1_16_uid286_sincosTest_b_1_q(112)) & redist136_xip1_16_uid286_sincosTest_b_1_q));
    xip1E_17_uid298_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 97 => twoToMiSiYip_uid295_sincosTest_b(96)) & twoToMiSiYip_uid295_sincosTest_b));
    xip1E_17_uid298_sincosTest_combproc: PROCESS (xip1E_17_uid298_sincosTest_a, xip1E_17_uid298_sincosTest_b, xip1E_17_uid298_sincosTest_s)
    BEGIN
        IF (xip1E_17_uid298_sincosTest_s = "1") THEN
            xip1E_17_uid298_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_17_uid298_sincosTest_a) + SIGNED(xip1E_17_uid298_sincosTest_b));
        ELSE
            xip1E_17_uid298_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_17_uid298_sincosTest_a) - SIGNED(xip1E_17_uid298_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_17_uid298_sincosTest_q <= xip1E_17_uid298_sincosTest_o(113 downto 0);

    -- xip1_17_uid302_sincosTest(BITSELECT,301)@18
    xip1_17_uid302_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_17_uid298_sincosTest_q(112 downto 0));
    xip1_17_uid302_sincosTest_b <= STD_LOGIC_VECTOR(xip1_17_uid302_sincosTest_in(112 downto 0));

    -- redist133_xip1_17_uid302_sincosTest_b_1(DELAY,1170)
    redist133_xip1_17_uid302_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_17_uid302_sincosTest_b, xout => redist133_xip1_17_uid302_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_18_uid314_sincosTest(ADDSUB,313)@19
    xip1E_18_uid314_sincosTest_s <= xMSB_uid305_sincosTest_b;
    xip1E_18_uid314_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist133_xip1_17_uid302_sincosTest_b_1_q(112)) & redist133_xip1_17_uid302_sincosTest_b_1_q));
    xip1E_18_uid314_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 96 => twoToMiSiYip_uid311_sincosTest_b(95)) & twoToMiSiYip_uid311_sincosTest_b));
    xip1E_18_uid314_sincosTest_combproc: PROCESS (xip1E_18_uid314_sincosTest_a, xip1E_18_uid314_sincosTest_b, xip1E_18_uid314_sincosTest_s)
    BEGIN
        IF (xip1E_18_uid314_sincosTest_s = "1") THEN
            xip1E_18_uid314_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_18_uid314_sincosTest_a) + SIGNED(xip1E_18_uid314_sincosTest_b));
        ELSE
            xip1E_18_uid314_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_18_uid314_sincosTest_a) - SIGNED(xip1E_18_uid314_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_18_uid314_sincosTest_q <= xip1E_18_uid314_sincosTest_o(113 downto 0);

    -- xip1_18_uid318_sincosTest(BITSELECT,317)@19
    xip1_18_uid318_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_18_uid314_sincosTest_q(112 downto 0));
    xip1_18_uid318_sincosTest_b <= STD_LOGIC_VECTOR(xip1_18_uid318_sincosTest_in(112 downto 0));

    -- redist130_xip1_18_uid318_sincosTest_b_1(DELAY,1167)
    redist130_xip1_18_uid318_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_18_uid318_sincosTest_b, xout => redist130_xip1_18_uid318_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid326_sincosTest(BITSELECT,325)@20
    twoToMiSiXip_uid326_sincosTest_b <= STD_LOGIC_VECTOR(redist130_xip1_18_uid318_sincosTest_b_1_q(112 downto 18));

    -- signOfSelectionSignal_uid307_sincosTest(LOGICAL,306)@19
    signOfSelectionSignal_uid307_sincosTest_q <= not (xMSB_uid305_sincosTest_b);

    -- twoToMiSiXip_uid310_sincosTest(BITSELECT,309)@19
    twoToMiSiXip_uid310_sincosTest_b <= STD_LOGIC_VECTOR(redist133_xip1_17_uid302_sincosTest_b_1_q(112 downto 17));

    -- yip1E_18_uid315_sincosTest(ADDSUB,314)@19
    yip1E_18_uid315_sincosTest_s <= signOfSelectionSignal_uid307_sincosTest_q;
    yip1E_18_uid315_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist132_yip1_17_uid303_sincosTest_b_1_q(112)) & redist132_yip1_17_uid303_sincosTest_b_1_q));
    yip1E_18_uid315_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 96 => twoToMiSiXip_uid310_sincosTest_b(95)) & twoToMiSiXip_uid310_sincosTest_b));
    yip1E_18_uid315_sincosTest_combproc: PROCESS (yip1E_18_uid315_sincosTest_a, yip1E_18_uid315_sincosTest_b, yip1E_18_uid315_sincosTest_s)
    BEGIN
        IF (yip1E_18_uid315_sincosTest_s = "1") THEN
            yip1E_18_uid315_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_18_uid315_sincosTest_a) + SIGNED(yip1E_18_uid315_sincosTest_b));
        ELSE
            yip1E_18_uid315_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_18_uid315_sincosTest_a) - SIGNED(yip1E_18_uid315_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_18_uid315_sincosTest_q <= yip1E_18_uid315_sincosTest_o(113 downto 0);

    -- yip1_18_uid319_sincosTest(BITSELECT,318)@19
    yip1_18_uid319_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_18_uid315_sincosTest_q(112 downto 0));
    yip1_18_uid319_sincosTest_b <= STD_LOGIC_VECTOR(yip1_18_uid319_sincosTest_in(112 downto 0));

    -- redist129_yip1_18_uid319_sincosTest_b_1(DELAY,1166)
    redist129_yip1_18_uid319_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_18_uid319_sincosTest_b, xout => redist129_yip1_18_uid319_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_19_uid331_sincosTest(ADDSUB,330)@20
    yip1E_19_uid331_sincosTest_s <= signOfSelectionSignal_uid323_sincosTest_q;
    yip1E_19_uid331_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist129_yip1_18_uid319_sincosTest_b_1_q(112)) & redist129_yip1_18_uid319_sincosTest_b_1_q));
    yip1E_19_uid331_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 95 => twoToMiSiXip_uid326_sincosTest_b(94)) & twoToMiSiXip_uid326_sincosTest_b));
    yip1E_19_uid331_sincosTest_combproc: PROCESS (yip1E_19_uid331_sincosTest_a, yip1E_19_uid331_sincosTest_b, yip1E_19_uid331_sincosTest_s)
    BEGIN
        IF (yip1E_19_uid331_sincosTest_s = "1") THEN
            yip1E_19_uid331_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_19_uid331_sincosTest_a) + SIGNED(yip1E_19_uid331_sincosTest_b));
        ELSE
            yip1E_19_uid331_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_19_uid331_sincosTest_a) - SIGNED(yip1E_19_uid331_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_19_uid331_sincosTest_q <= yip1E_19_uid331_sincosTest_o(113 downto 0);

    -- yip1_19_uid335_sincosTest(BITSELECT,334)@20
    yip1_19_uid335_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_19_uid331_sincosTest_q(112 downto 0));
    yip1_19_uid335_sincosTest_b <= STD_LOGIC_VECTOR(yip1_19_uid335_sincosTest_in(112 downto 0));

    -- redist126_yip1_19_uid335_sincosTest_b_1(DELAY,1163)
    redist126_yip1_19_uid335_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_19_uid335_sincosTest_b, xout => redist126_yip1_19_uid335_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid343_sincosTest(BITSELECT,342)@21
    twoToMiSiYip_uid343_sincosTest_b <= STD_LOGIC_VECTOR(redist126_yip1_19_uid335_sincosTest_b_1_q(112 downto 19));

    -- twoToMiSiYip_uid327_sincosTest(BITSELECT,326)@20
    twoToMiSiYip_uid327_sincosTest_b <= STD_LOGIC_VECTOR(redist129_yip1_18_uid319_sincosTest_b_1_q(112 downto 18));

    -- xip1E_19_uid330_sincosTest(ADDSUB,329)@20
    xip1E_19_uid330_sincosTest_s <= xMSB_uid321_sincosTest_b;
    xip1E_19_uid330_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist130_xip1_18_uid318_sincosTest_b_1_q(112)) & redist130_xip1_18_uid318_sincosTest_b_1_q));
    xip1E_19_uid330_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 95 => twoToMiSiYip_uid327_sincosTest_b(94)) & twoToMiSiYip_uid327_sincosTest_b));
    xip1E_19_uid330_sincosTest_combproc: PROCESS (xip1E_19_uid330_sincosTest_a, xip1E_19_uid330_sincosTest_b, xip1E_19_uid330_sincosTest_s)
    BEGIN
        IF (xip1E_19_uid330_sincosTest_s = "1") THEN
            xip1E_19_uid330_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_19_uid330_sincosTest_a) + SIGNED(xip1E_19_uid330_sincosTest_b));
        ELSE
            xip1E_19_uid330_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_19_uid330_sincosTest_a) - SIGNED(xip1E_19_uid330_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_19_uid330_sincosTest_q <= xip1E_19_uid330_sincosTest_o(113 downto 0);

    -- xip1_19_uid334_sincosTest(BITSELECT,333)@20
    xip1_19_uid334_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_19_uid330_sincosTest_q(112 downto 0));
    xip1_19_uid334_sincosTest_b <= STD_LOGIC_VECTOR(xip1_19_uid334_sincosTest_in(112 downto 0));

    -- redist127_xip1_19_uid334_sincosTest_b_1(DELAY,1164)
    redist127_xip1_19_uid334_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_19_uid334_sincosTest_b, xout => redist127_xip1_19_uid334_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_20_uid346_sincosTest(ADDSUB,345)@21
    xip1E_20_uid346_sincosTest_s <= xMSB_uid337_sincosTest_b;
    xip1E_20_uid346_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist127_xip1_19_uid334_sincosTest_b_1_q(112)) & redist127_xip1_19_uid334_sincosTest_b_1_q));
    xip1E_20_uid346_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 94 => twoToMiSiYip_uid343_sincosTest_b(93)) & twoToMiSiYip_uid343_sincosTest_b));
    xip1E_20_uid346_sincosTest_combproc: PROCESS (xip1E_20_uid346_sincosTest_a, xip1E_20_uid346_sincosTest_b, xip1E_20_uid346_sincosTest_s)
    BEGIN
        IF (xip1E_20_uid346_sincosTest_s = "1") THEN
            xip1E_20_uid346_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_20_uid346_sincosTest_a) + SIGNED(xip1E_20_uid346_sincosTest_b));
        ELSE
            xip1E_20_uid346_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_20_uid346_sincosTest_a) - SIGNED(xip1E_20_uid346_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_20_uid346_sincosTest_q <= xip1E_20_uid346_sincosTest_o(113 downto 0);

    -- xip1_20_uid350_sincosTest(BITSELECT,349)@21
    xip1_20_uid350_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_20_uid346_sincosTest_q(112 downto 0));
    xip1_20_uid350_sincosTest_b <= STD_LOGIC_VECTOR(xip1_20_uid350_sincosTest_in(112 downto 0));

    -- redist124_xip1_20_uid350_sincosTest_b_1(DELAY,1161)
    redist124_xip1_20_uid350_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_20_uid350_sincosTest_b, xout => redist124_xip1_20_uid350_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid358_sincosTest(BITSELECT,357)@22
    twoToMiSiXip_uid358_sincosTest_b <= STD_LOGIC_VECTOR(redist124_xip1_20_uid350_sincosTest_b_1_q(112 downto 20));

    -- signOfSelectionSignal_uid339_sincosTest(LOGICAL,338)@21
    signOfSelectionSignal_uid339_sincosTest_q <= not (xMSB_uid337_sincosTest_b);

    -- twoToMiSiXip_uid342_sincosTest(BITSELECT,341)@21
    twoToMiSiXip_uid342_sincosTest_b <= STD_LOGIC_VECTOR(redist127_xip1_19_uid334_sincosTest_b_1_q(112 downto 19));

    -- yip1E_20_uid347_sincosTest(ADDSUB,346)@21
    yip1E_20_uid347_sincosTest_s <= signOfSelectionSignal_uid339_sincosTest_q;
    yip1E_20_uid347_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist126_yip1_19_uid335_sincosTest_b_1_q(112)) & redist126_yip1_19_uid335_sincosTest_b_1_q));
    yip1E_20_uid347_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 94 => twoToMiSiXip_uid342_sincosTest_b(93)) & twoToMiSiXip_uid342_sincosTest_b));
    yip1E_20_uid347_sincosTest_combproc: PROCESS (yip1E_20_uid347_sincosTest_a, yip1E_20_uid347_sincosTest_b, yip1E_20_uid347_sincosTest_s)
    BEGIN
        IF (yip1E_20_uid347_sincosTest_s = "1") THEN
            yip1E_20_uid347_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_20_uid347_sincosTest_a) + SIGNED(yip1E_20_uid347_sincosTest_b));
        ELSE
            yip1E_20_uid347_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_20_uid347_sincosTest_a) - SIGNED(yip1E_20_uid347_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_20_uid347_sincosTest_q <= yip1E_20_uid347_sincosTest_o(113 downto 0);

    -- yip1_20_uid351_sincosTest(BITSELECT,350)@21
    yip1_20_uid351_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_20_uid347_sincosTest_q(112 downto 0));
    yip1_20_uid351_sincosTest_b <= STD_LOGIC_VECTOR(yip1_20_uid351_sincosTest_in(112 downto 0));

    -- redist123_yip1_20_uid351_sincosTest_b_1(DELAY,1160)
    redist123_yip1_20_uid351_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_20_uid351_sincosTest_b, xout => redist123_yip1_20_uid351_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_21_uid363_sincosTest(ADDSUB,362)@22
    yip1E_21_uid363_sincosTest_s <= signOfSelectionSignal_uid355_sincosTest_q;
    yip1E_21_uid363_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist123_yip1_20_uid351_sincosTest_b_1_q(112)) & redist123_yip1_20_uid351_sincosTest_b_1_q));
    yip1E_21_uid363_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 93 => twoToMiSiXip_uid358_sincosTest_b(92)) & twoToMiSiXip_uid358_sincosTest_b));
    yip1E_21_uid363_sincosTest_combproc: PROCESS (yip1E_21_uid363_sincosTest_a, yip1E_21_uid363_sincosTest_b, yip1E_21_uid363_sincosTest_s)
    BEGIN
        IF (yip1E_21_uid363_sincosTest_s = "1") THEN
            yip1E_21_uid363_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_21_uid363_sincosTest_a) + SIGNED(yip1E_21_uid363_sincosTest_b));
        ELSE
            yip1E_21_uid363_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_21_uid363_sincosTest_a) - SIGNED(yip1E_21_uid363_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_21_uid363_sincosTest_q <= yip1E_21_uid363_sincosTest_o(113 downto 0);

    -- yip1_21_uid367_sincosTest(BITSELECT,366)@22
    yip1_21_uid367_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_21_uid363_sincosTest_q(112 downto 0));
    yip1_21_uid367_sincosTest_b <= STD_LOGIC_VECTOR(yip1_21_uid367_sincosTest_in(112 downto 0));

    -- redist120_yip1_21_uid367_sincosTest_b_1(DELAY,1157)
    redist120_yip1_21_uid367_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_21_uid367_sincosTest_b, xout => redist120_yip1_21_uid367_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid375_sincosTest(BITSELECT,374)@23
    twoToMiSiYip_uid375_sincosTest_b <= STD_LOGIC_VECTOR(redist120_yip1_21_uid367_sincosTest_b_1_q(112 downto 21));

    -- twoToMiSiYip_uid359_sincosTest(BITSELECT,358)@22
    twoToMiSiYip_uid359_sincosTest_b <= STD_LOGIC_VECTOR(redist123_yip1_20_uid351_sincosTest_b_1_q(112 downto 20));

    -- xip1E_21_uid362_sincosTest(ADDSUB,361)@22
    xip1E_21_uid362_sincosTest_s <= xMSB_uid353_sincosTest_b;
    xip1E_21_uid362_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist124_xip1_20_uid350_sincosTest_b_1_q(112)) & redist124_xip1_20_uid350_sincosTest_b_1_q));
    xip1E_21_uid362_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 93 => twoToMiSiYip_uid359_sincosTest_b(92)) & twoToMiSiYip_uid359_sincosTest_b));
    xip1E_21_uid362_sincosTest_combproc: PROCESS (xip1E_21_uid362_sincosTest_a, xip1E_21_uid362_sincosTest_b, xip1E_21_uid362_sincosTest_s)
    BEGIN
        IF (xip1E_21_uid362_sincosTest_s = "1") THEN
            xip1E_21_uid362_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_21_uid362_sincosTest_a) + SIGNED(xip1E_21_uid362_sincosTest_b));
        ELSE
            xip1E_21_uid362_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_21_uid362_sincosTest_a) - SIGNED(xip1E_21_uid362_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_21_uid362_sincosTest_q <= xip1E_21_uid362_sincosTest_o(113 downto 0);

    -- xip1_21_uid366_sincosTest(BITSELECT,365)@22
    xip1_21_uid366_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_21_uid362_sincosTest_q(112 downto 0));
    xip1_21_uid366_sincosTest_b <= STD_LOGIC_VECTOR(xip1_21_uid366_sincosTest_in(112 downto 0));

    -- redist121_xip1_21_uid366_sincosTest_b_1(DELAY,1158)
    redist121_xip1_21_uid366_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_21_uid366_sincosTest_b, xout => redist121_xip1_21_uid366_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_22_uid378_sincosTest(ADDSUB,377)@23
    xip1E_22_uid378_sincosTest_s <= xMSB_uid369_sincosTest_b;
    xip1E_22_uid378_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist121_xip1_21_uid366_sincosTest_b_1_q(112)) & redist121_xip1_21_uid366_sincosTest_b_1_q));
    xip1E_22_uid378_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 92 => twoToMiSiYip_uid375_sincosTest_b(91)) & twoToMiSiYip_uid375_sincosTest_b));
    xip1E_22_uid378_sincosTest_combproc: PROCESS (xip1E_22_uid378_sincosTest_a, xip1E_22_uid378_sincosTest_b, xip1E_22_uid378_sincosTest_s)
    BEGIN
        IF (xip1E_22_uid378_sincosTest_s = "1") THEN
            xip1E_22_uid378_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_22_uid378_sincosTest_a) + SIGNED(xip1E_22_uid378_sincosTest_b));
        ELSE
            xip1E_22_uid378_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_22_uid378_sincosTest_a) - SIGNED(xip1E_22_uid378_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_22_uid378_sincosTest_q <= xip1E_22_uid378_sincosTest_o(113 downto 0);

    -- xip1_22_uid382_sincosTest(BITSELECT,381)@23
    xip1_22_uid382_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_22_uid378_sincosTest_q(112 downto 0));
    xip1_22_uid382_sincosTest_b <= STD_LOGIC_VECTOR(xip1_22_uid382_sincosTest_in(112 downto 0));

    -- redist118_xip1_22_uid382_sincosTest_b_1(DELAY,1155)
    redist118_xip1_22_uid382_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_22_uid382_sincosTest_b, xout => redist118_xip1_22_uid382_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid390_sincosTest(BITSELECT,389)@24
    twoToMiSiXip_uid390_sincosTest_b <= STD_LOGIC_VECTOR(redist118_xip1_22_uid382_sincosTest_b_1_q(112 downto 22));

    -- signOfSelectionSignal_uid371_sincosTest(LOGICAL,370)@23
    signOfSelectionSignal_uid371_sincosTest_q <= not (xMSB_uid369_sincosTest_b);

    -- twoToMiSiXip_uid374_sincosTest(BITSELECT,373)@23
    twoToMiSiXip_uid374_sincosTest_b <= STD_LOGIC_VECTOR(redist121_xip1_21_uid366_sincosTest_b_1_q(112 downto 21));

    -- yip1E_22_uid379_sincosTest(ADDSUB,378)@23
    yip1E_22_uid379_sincosTest_s <= signOfSelectionSignal_uid371_sincosTest_q;
    yip1E_22_uid379_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist120_yip1_21_uid367_sincosTest_b_1_q(112)) & redist120_yip1_21_uid367_sincosTest_b_1_q));
    yip1E_22_uid379_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 92 => twoToMiSiXip_uid374_sincosTest_b(91)) & twoToMiSiXip_uid374_sincosTest_b));
    yip1E_22_uid379_sincosTest_combproc: PROCESS (yip1E_22_uid379_sincosTest_a, yip1E_22_uid379_sincosTest_b, yip1E_22_uid379_sincosTest_s)
    BEGIN
        IF (yip1E_22_uid379_sincosTest_s = "1") THEN
            yip1E_22_uid379_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_22_uid379_sincosTest_a) + SIGNED(yip1E_22_uid379_sincosTest_b));
        ELSE
            yip1E_22_uid379_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_22_uid379_sincosTest_a) - SIGNED(yip1E_22_uid379_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_22_uid379_sincosTest_q <= yip1E_22_uid379_sincosTest_o(113 downto 0);

    -- yip1_22_uid383_sincosTest(BITSELECT,382)@23
    yip1_22_uid383_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_22_uid379_sincosTest_q(112 downto 0));
    yip1_22_uid383_sincosTest_b <= STD_LOGIC_VECTOR(yip1_22_uid383_sincosTest_in(112 downto 0));

    -- redist117_yip1_22_uid383_sincosTest_b_1(DELAY,1154)
    redist117_yip1_22_uid383_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_22_uid383_sincosTest_b, xout => redist117_yip1_22_uid383_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_23_uid395_sincosTest(ADDSUB,394)@24
    yip1E_23_uid395_sincosTest_s <= signOfSelectionSignal_uid387_sincosTest_q;
    yip1E_23_uid395_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist117_yip1_22_uid383_sincosTest_b_1_q(112)) & redist117_yip1_22_uid383_sincosTest_b_1_q));
    yip1E_23_uid395_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 91 => twoToMiSiXip_uid390_sincosTest_b(90)) & twoToMiSiXip_uid390_sincosTest_b));
    yip1E_23_uid395_sincosTest_combproc: PROCESS (yip1E_23_uid395_sincosTest_a, yip1E_23_uid395_sincosTest_b, yip1E_23_uid395_sincosTest_s)
    BEGIN
        IF (yip1E_23_uid395_sincosTest_s = "1") THEN
            yip1E_23_uid395_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_23_uid395_sincosTest_a) + SIGNED(yip1E_23_uid395_sincosTest_b));
        ELSE
            yip1E_23_uid395_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_23_uid395_sincosTest_a) - SIGNED(yip1E_23_uid395_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_23_uid395_sincosTest_q <= yip1E_23_uid395_sincosTest_o(113 downto 0);

    -- yip1_23_uid402_sincosTest(BITSELECT,401)@24
    yip1_23_uid402_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_23_uid395_sincosTest_q(112 downto 0));
    yip1_23_uid402_sincosTest_b <= STD_LOGIC_VECTOR(yip1_23_uid402_sincosTest_in(112 downto 0));

    -- redist114_yip1_23_uid402_sincosTest_b_1(DELAY,1151)
    redist114_yip1_23_uid402_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_23_uid402_sincosTest_b, xout => redist114_yip1_23_uid402_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid410_sincosTest(BITSELECT,409)@25
    twoToMiSiYip_uid410_sincosTest_b <= STD_LOGIC_VECTOR(redist114_yip1_23_uid402_sincosTest_b_1_q(112 downto 23));

    -- twoToMiSiYip_uid391_sincosTest(BITSELECT,390)@24
    twoToMiSiYip_uid391_sincosTest_b <= STD_LOGIC_VECTOR(redist117_yip1_22_uid383_sincosTest_b_1_q(112 downto 22));

    -- xip1E_23_uid394_sincosTest(ADDSUB,393)@24
    xip1E_23_uid394_sincosTest_s <= xMSB_uid385_sincosTest_b;
    xip1E_23_uid394_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist118_xip1_22_uid382_sincosTest_b_1_q(112)) & redist118_xip1_22_uid382_sincosTest_b_1_q));
    xip1E_23_uid394_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 91 => twoToMiSiYip_uid391_sincosTest_b(90)) & twoToMiSiYip_uid391_sincosTest_b));
    xip1E_23_uid394_sincosTest_combproc: PROCESS (xip1E_23_uid394_sincosTest_a, xip1E_23_uid394_sincosTest_b, xip1E_23_uid394_sincosTest_s)
    BEGIN
        IF (xip1E_23_uid394_sincosTest_s = "1") THEN
            xip1E_23_uid394_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_23_uid394_sincosTest_a) + SIGNED(xip1E_23_uid394_sincosTest_b));
        ELSE
            xip1E_23_uid394_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_23_uid394_sincosTest_a) - SIGNED(xip1E_23_uid394_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_23_uid394_sincosTest_q <= xip1E_23_uid394_sincosTest_o(113 downto 0);

    -- xip1_23_uid401_sincosTest(BITSELECT,400)@24
    xip1_23_uid401_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_23_uid394_sincosTest_q(112 downto 0));
    xip1_23_uid401_sincosTest_b <= STD_LOGIC_VECTOR(xip1_23_uid401_sincosTest_in(112 downto 0));

    -- redist115_xip1_23_uid401_sincosTest_b_1(DELAY,1152)
    redist115_xip1_23_uid401_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_23_uid401_sincosTest_b, xout => redist115_xip1_23_uid401_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_24_uid413_sincosTest(ADDSUB,412)@25
    xip1E_24_uid413_sincosTest_s <= xMSB_uid404_sincosTest_b;
    xip1E_24_uid413_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist115_xip1_23_uid401_sincosTest_b_1_q(112)) & redist115_xip1_23_uid401_sincosTest_b_1_q));
    xip1E_24_uid413_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 90 => twoToMiSiYip_uid410_sincosTest_b(89)) & twoToMiSiYip_uid410_sincosTest_b));
    xip1E_24_uid413_sincosTest_combproc: PROCESS (xip1E_24_uid413_sincosTest_a, xip1E_24_uid413_sincosTest_b, xip1E_24_uid413_sincosTest_s)
    BEGIN
        IF (xip1E_24_uid413_sincosTest_s = "1") THEN
            xip1E_24_uid413_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_24_uid413_sincosTest_a) + SIGNED(xip1E_24_uid413_sincosTest_b));
        ELSE
            xip1E_24_uid413_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_24_uid413_sincosTest_a) - SIGNED(xip1E_24_uid413_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_24_uid413_sincosTest_q <= xip1E_24_uid413_sincosTest_o(113 downto 0);

    -- xip1_24_uid420_sincosTest(BITSELECT,419)@25
    xip1_24_uid420_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_24_uid413_sincosTest_q(112 downto 0));
    xip1_24_uid420_sincosTest_b <= STD_LOGIC_VECTOR(xip1_24_uid420_sincosTest_in(112 downto 0));

    -- redist112_xip1_24_uid420_sincosTest_b_1(DELAY,1149)
    redist112_xip1_24_uid420_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_24_uid420_sincosTest_b, xout => redist112_xip1_24_uid420_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid428_sincosTest(BITSELECT,427)@26
    twoToMiSiXip_uid428_sincosTest_b <= STD_LOGIC_VECTOR(redist112_xip1_24_uid420_sincosTest_b_1_q(112 downto 24));

    -- signOfSelectionSignal_uid406_sincosTest(LOGICAL,405)@25
    signOfSelectionSignal_uid406_sincosTest_q <= not (xMSB_uid404_sincosTest_b);

    -- twoToMiSiXip_uid409_sincosTest(BITSELECT,408)@25
    twoToMiSiXip_uid409_sincosTest_b <= STD_LOGIC_VECTOR(redist115_xip1_23_uid401_sincosTest_b_1_q(112 downto 23));

    -- yip1E_24_uid414_sincosTest(ADDSUB,413)@25
    yip1E_24_uid414_sincosTest_s <= signOfSelectionSignal_uid406_sincosTest_q;
    yip1E_24_uid414_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist114_yip1_23_uid402_sincosTest_b_1_q(112)) & redist114_yip1_23_uid402_sincosTest_b_1_q));
    yip1E_24_uid414_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 90 => twoToMiSiXip_uid409_sincosTest_b(89)) & twoToMiSiXip_uid409_sincosTest_b));
    yip1E_24_uid414_sincosTest_combproc: PROCESS (yip1E_24_uid414_sincosTest_a, yip1E_24_uid414_sincosTest_b, yip1E_24_uid414_sincosTest_s)
    BEGIN
        IF (yip1E_24_uid414_sincosTest_s = "1") THEN
            yip1E_24_uid414_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_24_uid414_sincosTest_a) + SIGNED(yip1E_24_uid414_sincosTest_b));
        ELSE
            yip1E_24_uid414_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_24_uid414_sincosTest_a) - SIGNED(yip1E_24_uid414_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_24_uid414_sincosTest_q <= yip1E_24_uid414_sincosTest_o(113 downto 0);

    -- yip1_24_uid421_sincosTest(BITSELECT,420)@25
    yip1_24_uid421_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_24_uid414_sincosTest_q(112 downto 0));
    yip1_24_uid421_sincosTest_b <= STD_LOGIC_VECTOR(yip1_24_uid421_sincosTest_in(112 downto 0));

    -- redist111_yip1_24_uid421_sincosTest_b_1(DELAY,1148)
    redist111_yip1_24_uid421_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_24_uid421_sincosTest_b, xout => redist111_yip1_24_uid421_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_25_uid433_sincosTest(ADDSUB,432)@26
    yip1E_25_uid433_sincosTest_s <= signOfSelectionSignal_uid425_sincosTest_q;
    yip1E_25_uid433_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist111_yip1_24_uid421_sincosTest_b_1_q(112)) & redist111_yip1_24_uid421_sincosTest_b_1_q));
    yip1E_25_uid433_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 89 => twoToMiSiXip_uid428_sincosTest_b(88)) & twoToMiSiXip_uid428_sincosTest_b));
    yip1E_25_uid433_sincosTest_combproc: PROCESS (yip1E_25_uid433_sincosTest_a, yip1E_25_uid433_sincosTest_b, yip1E_25_uid433_sincosTest_s)
    BEGIN
        IF (yip1E_25_uid433_sincosTest_s = "1") THEN
            yip1E_25_uid433_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_25_uid433_sincosTest_a) + SIGNED(yip1E_25_uid433_sincosTest_b));
        ELSE
            yip1E_25_uid433_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_25_uid433_sincosTest_a) - SIGNED(yip1E_25_uid433_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_25_uid433_sincosTest_q <= yip1E_25_uid433_sincosTest_o(113 downto 0);

    -- yip1_25_uid440_sincosTest(BITSELECT,439)@26
    yip1_25_uid440_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_25_uid433_sincosTest_q(112 downto 0));
    yip1_25_uid440_sincosTest_b <= STD_LOGIC_VECTOR(yip1_25_uid440_sincosTest_in(112 downto 0));

    -- redist108_yip1_25_uid440_sincosTest_b_1(DELAY,1145)
    redist108_yip1_25_uid440_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_25_uid440_sincosTest_b, xout => redist108_yip1_25_uid440_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid448_sincosTest(BITSELECT,447)@27
    twoToMiSiYip_uid448_sincosTest_b <= STD_LOGIC_VECTOR(redist108_yip1_25_uid440_sincosTest_b_1_q(112 downto 25));

    -- twoToMiSiYip_uid429_sincosTest(BITSELECT,428)@26
    twoToMiSiYip_uid429_sincosTest_b <= STD_LOGIC_VECTOR(redist111_yip1_24_uid421_sincosTest_b_1_q(112 downto 24));

    -- xip1E_25_uid432_sincosTest(ADDSUB,431)@26
    xip1E_25_uid432_sincosTest_s <= xMSB_uid423_sincosTest_b;
    xip1E_25_uid432_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist112_xip1_24_uid420_sincosTest_b_1_q(112)) & redist112_xip1_24_uid420_sincosTest_b_1_q));
    xip1E_25_uid432_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 89 => twoToMiSiYip_uid429_sincosTest_b(88)) & twoToMiSiYip_uid429_sincosTest_b));
    xip1E_25_uid432_sincosTest_combproc: PROCESS (xip1E_25_uid432_sincosTest_a, xip1E_25_uid432_sincosTest_b, xip1E_25_uid432_sincosTest_s)
    BEGIN
        IF (xip1E_25_uid432_sincosTest_s = "1") THEN
            xip1E_25_uid432_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_25_uid432_sincosTest_a) + SIGNED(xip1E_25_uid432_sincosTest_b));
        ELSE
            xip1E_25_uid432_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_25_uid432_sincosTest_a) - SIGNED(xip1E_25_uid432_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_25_uid432_sincosTest_q <= xip1E_25_uid432_sincosTest_o(113 downto 0);

    -- xip1_25_uid439_sincosTest(BITSELECT,438)@26
    xip1_25_uid439_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_25_uid432_sincosTest_q(112 downto 0));
    xip1_25_uid439_sincosTest_b <= STD_LOGIC_VECTOR(xip1_25_uid439_sincosTest_in(112 downto 0));

    -- redist109_xip1_25_uid439_sincosTest_b_1(DELAY,1146)
    redist109_xip1_25_uid439_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_25_uid439_sincosTest_b, xout => redist109_xip1_25_uid439_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_26_uid451_sincosTest(ADDSUB,450)@27
    xip1E_26_uid451_sincosTest_s <= redist107_xMSB_uid442_sincosTest_b_1_q;
    xip1E_26_uid451_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist109_xip1_25_uid439_sincosTest_b_1_q(112)) & redist109_xip1_25_uid439_sincosTest_b_1_q));
    xip1E_26_uid451_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 88 => twoToMiSiYip_uid448_sincosTest_b(87)) & twoToMiSiYip_uid448_sincosTest_b));
    xip1E_26_uid451_sincosTest_combproc: PROCESS (xip1E_26_uid451_sincosTest_a, xip1E_26_uid451_sincosTest_b, xip1E_26_uid451_sincosTest_s)
    BEGIN
        IF (xip1E_26_uid451_sincosTest_s = "1") THEN
            xip1E_26_uid451_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_26_uid451_sincosTest_a) + SIGNED(xip1E_26_uid451_sincosTest_b));
        ELSE
            xip1E_26_uid451_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_26_uid451_sincosTest_a) - SIGNED(xip1E_26_uid451_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_26_uid451_sincosTest_q <= xip1E_26_uid451_sincosTest_o(113 downto 0);

    -- xip1_26_uid458_sincosTest(BITSELECT,457)@27
    xip1_26_uid458_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_26_uid451_sincosTest_q(112 downto 0));
    xip1_26_uid458_sincosTest_b <= STD_LOGIC_VECTOR(xip1_26_uid458_sincosTest_in(112 downto 0));

    -- redist106_xip1_26_uid458_sincosTest_b_1(DELAY,1143)
    redist106_xip1_26_uid458_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_26_uid458_sincosTest_b, xout => redist106_xip1_26_uid458_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid466_sincosTest(BITSELECT,465)@28
    twoToMiSiXip_uid466_sincosTest_b <= STD_LOGIC_VECTOR(redist106_xip1_26_uid458_sincosTest_b_1_q(112 downto 26));

    -- signOfSelectionSignal_uid444_sincosTest(LOGICAL,443)@27
    signOfSelectionSignal_uid444_sincosTest_q <= not (redist107_xMSB_uid442_sincosTest_b_1_q);

    -- twoToMiSiXip_uid447_sincosTest(BITSELECT,446)@27
    twoToMiSiXip_uid447_sincosTest_b <= STD_LOGIC_VECTOR(redist109_xip1_25_uid439_sincosTest_b_1_q(112 downto 25));

    -- yip1E_26_uid452_sincosTest(ADDSUB,451)@27
    yip1E_26_uid452_sincosTest_s <= signOfSelectionSignal_uid444_sincosTest_q;
    yip1E_26_uid452_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist108_yip1_25_uid440_sincosTest_b_1_q(112)) & redist108_yip1_25_uid440_sincosTest_b_1_q));
    yip1E_26_uid452_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 88 => twoToMiSiXip_uid447_sincosTest_b(87)) & twoToMiSiXip_uid447_sincosTest_b));
    yip1E_26_uid452_sincosTest_combproc: PROCESS (yip1E_26_uid452_sincosTest_a, yip1E_26_uid452_sincosTest_b, yip1E_26_uid452_sincosTest_s)
    BEGIN
        IF (yip1E_26_uid452_sincosTest_s = "1") THEN
            yip1E_26_uid452_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_26_uid452_sincosTest_a) + SIGNED(yip1E_26_uid452_sincosTest_b));
        ELSE
            yip1E_26_uid452_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_26_uid452_sincosTest_a) - SIGNED(yip1E_26_uid452_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_26_uid452_sincosTest_q <= yip1E_26_uid452_sincosTest_o(113 downto 0);

    -- yip1_26_uid459_sincosTest(BITSELECT,458)@27
    yip1_26_uid459_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_26_uid452_sincosTest_q(112 downto 0));
    yip1_26_uid459_sincosTest_b <= STD_LOGIC_VECTOR(yip1_26_uid459_sincosTest_in(112 downto 0));

    -- redist105_yip1_26_uid459_sincosTest_b_1(DELAY,1142)
    redist105_yip1_26_uid459_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_26_uid459_sincosTest_b, xout => redist105_yip1_26_uid459_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_27_uid471_sincosTest(ADDSUB,470)@28
    yip1E_27_uid471_sincosTest_s <= signOfSelectionSignal_uid463_sincosTest_q;
    yip1E_27_uid471_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist105_yip1_26_uid459_sincosTest_b_1_q(112)) & redist105_yip1_26_uid459_sincosTest_b_1_q));
    yip1E_27_uid471_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 87 => twoToMiSiXip_uid466_sincosTest_b(86)) & twoToMiSiXip_uid466_sincosTest_b));
    yip1E_27_uid471_sincosTest_combproc: PROCESS (yip1E_27_uid471_sincosTest_a, yip1E_27_uid471_sincosTest_b, yip1E_27_uid471_sincosTest_s)
    BEGIN
        IF (yip1E_27_uid471_sincosTest_s = "1") THEN
            yip1E_27_uid471_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_27_uid471_sincosTest_a) + SIGNED(yip1E_27_uid471_sincosTest_b));
        ELSE
            yip1E_27_uid471_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_27_uid471_sincosTest_a) - SIGNED(yip1E_27_uid471_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_27_uid471_sincosTest_q <= yip1E_27_uid471_sincosTest_o(113 downto 0);

    -- yip1_27_uid478_sincosTest(BITSELECT,477)@28
    yip1_27_uid478_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_27_uid471_sincosTest_q(112 downto 0));
    yip1_27_uid478_sincosTest_b <= STD_LOGIC_VECTOR(yip1_27_uid478_sincosTest_in(112 downto 0));

    -- redist101_yip1_27_uid478_sincosTest_b_1(DELAY,1138)
    redist101_yip1_27_uid478_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_27_uid478_sincosTest_b, xout => redist101_yip1_27_uid478_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid486_sincosTest(BITSELECT,485)@29
    twoToMiSiYip_uid486_sincosTest_b <= STD_LOGIC_VECTOR(redist101_yip1_27_uid478_sincosTest_b_1_q(112 downto 27));

    -- twoToMiSiYip_uid467_sincosTest(BITSELECT,466)@28
    twoToMiSiYip_uid467_sincosTest_b <= STD_LOGIC_VECTOR(redist105_yip1_26_uid459_sincosTest_b_1_q(112 downto 26));

    -- xip1E_27_uid470_sincosTest(ADDSUB,469)@28
    xip1E_27_uid470_sincosTest_s <= redist103_xMSB_uid461_sincosTest_b_1_q;
    xip1E_27_uid470_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist106_xip1_26_uid458_sincosTest_b_1_q(112)) & redist106_xip1_26_uid458_sincosTest_b_1_q));
    xip1E_27_uid470_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 87 => twoToMiSiYip_uid467_sincosTest_b(86)) & twoToMiSiYip_uid467_sincosTest_b));
    xip1E_27_uid470_sincosTest_combproc: PROCESS (xip1E_27_uid470_sincosTest_a, xip1E_27_uid470_sincosTest_b, xip1E_27_uid470_sincosTest_s)
    BEGIN
        IF (xip1E_27_uid470_sincosTest_s = "1") THEN
            xip1E_27_uid470_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_27_uid470_sincosTest_a) + SIGNED(xip1E_27_uid470_sincosTest_b));
        ELSE
            xip1E_27_uid470_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_27_uid470_sincosTest_a) - SIGNED(xip1E_27_uid470_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_27_uid470_sincosTest_q <= xip1E_27_uid470_sincosTest_o(113 downto 0);

    -- xip1_27_uid477_sincosTest(BITSELECT,476)@28
    xip1_27_uid477_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_27_uid470_sincosTest_q(112 downto 0));
    xip1_27_uid477_sincosTest_b <= STD_LOGIC_VECTOR(xip1_27_uid477_sincosTest_in(112 downto 0));

    -- redist102_xip1_27_uid477_sincosTest_b_1(DELAY,1139)
    redist102_xip1_27_uid477_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_27_uid477_sincosTest_b, xout => redist102_xip1_27_uid477_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_28_uid489_sincosTest(ADDSUB,488)@29
    xip1E_28_uid489_sincosTest_s <= redist100_xMSB_uid480_sincosTest_b_2_q;
    xip1E_28_uid489_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist102_xip1_27_uid477_sincosTest_b_1_q(112)) & redist102_xip1_27_uid477_sincosTest_b_1_q));
    xip1E_28_uid489_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 86 => twoToMiSiYip_uid486_sincosTest_b(85)) & twoToMiSiYip_uid486_sincosTest_b));
    xip1E_28_uid489_sincosTest_combproc: PROCESS (xip1E_28_uid489_sincosTest_a, xip1E_28_uid489_sincosTest_b, xip1E_28_uid489_sincosTest_s)
    BEGIN
        IF (xip1E_28_uid489_sincosTest_s = "1") THEN
            xip1E_28_uid489_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_28_uid489_sincosTest_a) + SIGNED(xip1E_28_uid489_sincosTest_b));
        ELSE
            xip1E_28_uid489_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_28_uid489_sincosTest_a) - SIGNED(xip1E_28_uid489_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_28_uid489_sincosTest_q <= xip1E_28_uid489_sincosTest_o(113 downto 0);

    -- xip1_28_uid496_sincosTest(BITSELECT,495)@29
    xip1_28_uid496_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_28_uid489_sincosTest_q(112 downto 0));
    xip1_28_uid496_sincosTest_b <= STD_LOGIC_VECTOR(xip1_28_uid496_sincosTest_in(112 downto 0));

    -- redist99_xip1_28_uid496_sincosTest_b_1(DELAY,1136)
    redist99_xip1_28_uid496_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_28_uid496_sincosTest_b, xout => redist99_xip1_28_uid496_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid504_sincosTest(BITSELECT,503)@30
    twoToMiSiXip_uid504_sincosTest_b <= STD_LOGIC_VECTOR(redist99_xip1_28_uid496_sincosTest_b_1_q(112 downto 28));

    -- signOfSelectionSignal_uid482_sincosTest(LOGICAL,481)@29
    signOfSelectionSignal_uid482_sincosTest_q <= not (redist100_xMSB_uid480_sincosTest_b_2_q);

    -- twoToMiSiXip_uid485_sincosTest(BITSELECT,484)@29
    twoToMiSiXip_uid485_sincosTest_b <= STD_LOGIC_VECTOR(redist102_xip1_27_uid477_sincosTest_b_1_q(112 downto 27));

    -- yip1E_28_uid490_sincosTest(ADDSUB,489)@29
    yip1E_28_uid490_sincosTest_s <= signOfSelectionSignal_uid482_sincosTest_q;
    yip1E_28_uid490_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist101_yip1_27_uid478_sincosTest_b_1_q(112)) & redist101_yip1_27_uid478_sincosTest_b_1_q));
    yip1E_28_uid490_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 86 => twoToMiSiXip_uid485_sincosTest_b(85)) & twoToMiSiXip_uid485_sincosTest_b));
    yip1E_28_uid490_sincosTest_combproc: PROCESS (yip1E_28_uid490_sincosTest_a, yip1E_28_uid490_sincosTest_b, yip1E_28_uid490_sincosTest_s)
    BEGIN
        IF (yip1E_28_uid490_sincosTest_s = "1") THEN
            yip1E_28_uid490_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_28_uid490_sincosTest_a) + SIGNED(yip1E_28_uid490_sincosTest_b));
        ELSE
            yip1E_28_uid490_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_28_uid490_sincosTest_a) - SIGNED(yip1E_28_uid490_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_28_uid490_sincosTest_q <= yip1E_28_uid490_sincosTest_o(113 downto 0);

    -- yip1_28_uid497_sincosTest(BITSELECT,496)@29
    yip1_28_uid497_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_28_uid490_sincosTest_q(112 downto 0));
    yip1_28_uid497_sincosTest_b <= STD_LOGIC_VECTOR(yip1_28_uid497_sincosTest_in(112 downto 0));

    -- redist98_yip1_28_uid497_sincosTest_b_1(DELAY,1135)
    redist98_yip1_28_uid497_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_28_uid497_sincosTest_b, xout => redist98_yip1_28_uid497_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_29_uid509_sincosTest(ADDSUB,508)@30
    yip1E_29_uid509_sincosTest_s <= signOfSelectionSignal_uid501_sincosTest_q;
    yip1E_29_uid509_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist98_yip1_28_uid497_sincosTest_b_1_q(112)) & redist98_yip1_28_uid497_sincosTest_b_1_q));
    yip1E_29_uid509_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 85 => twoToMiSiXip_uid504_sincosTest_b(84)) & twoToMiSiXip_uid504_sincosTest_b));
    yip1E_29_uid509_sincosTest_combproc: PROCESS (yip1E_29_uid509_sincosTest_a, yip1E_29_uid509_sincosTest_b, yip1E_29_uid509_sincosTest_s)
    BEGIN
        IF (yip1E_29_uid509_sincosTest_s = "1") THEN
            yip1E_29_uid509_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_29_uid509_sincosTest_a) + SIGNED(yip1E_29_uid509_sincosTest_b));
        ELSE
            yip1E_29_uid509_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_29_uid509_sincosTest_a) - SIGNED(yip1E_29_uid509_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_29_uid509_sincosTest_q <= yip1E_29_uid509_sincosTest_o(113 downto 0);

    -- yip1_29_uid516_sincosTest(BITSELECT,515)@30
    yip1_29_uid516_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_29_uid509_sincosTest_q(112 downto 0));
    yip1_29_uid516_sincosTest_b <= STD_LOGIC_VECTOR(yip1_29_uid516_sincosTest_in(112 downto 0));

    -- redist94_yip1_29_uid516_sincosTest_b_1(DELAY,1131)
    redist94_yip1_29_uid516_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_29_uid516_sincosTest_b, xout => redist94_yip1_29_uid516_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid524_sincosTest(BITSELECT,523)@31
    twoToMiSiYip_uid524_sincosTest_b <= STD_LOGIC_VECTOR(redist94_yip1_29_uid516_sincosTest_b_1_q(112 downto 29));

    -- twoToMiSiYip_uid505_sincosTest(BITSELECT,504)@30
    twoToMiSiYip_uid505_sincosTest_b <= STD_LOGIC_VECTOR(redist98_yip1_28_uid497_sincosTest_b_1_q(112 downto 28));

    -- xip1E_29_uid508_sincosTest(ADDSUB,507)@30
    xip1E_29_uid508_sincosTest_s <= redist96_xMSB_uid499_sincosTest_b_2_q;
    xip1E_29_uid508_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist99_xip1_28_uid496_sincosTest_b_1_q(112)) & redist99_xip1_28_uid496_sincosTest_b_1_q));
    xip1E_29_uid508_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 85 => twoToMiSiYip_uid505_sincosTest_b(84)) & twoToMiSiYip_uid505_sincosTest_b));
    xip1E_29_uid508_sincosTest_combproc: PROCESS (xip1E_29_uid508_sincosTest_a, xip1E_29_uid508_sincosTest_b, xip1E_29_uid508_sincosTest_s)
    BEGIN
        IF (xip1E_29_uid508_sincosTest_s = "1") THEN
            xip1E_29_uid508_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_29_uid508_sincosTest_a) + SIGNED(xip1E_29_uid508_sincosTest_b));
        ELSE
            xip1E_29_uid508_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_29_uid508_sincosTest_a) - SIGNED(xip1E_29_uid508_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_29_uid508_sincosTest_q <= xip1E_29_uid508_sincosTest_o(113 downto 0);

    -- xip1_29_uid515_sincosTest(BITSELECT,514)@30
    xip1_29_uid515_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_29_uid508_sincosTest_q(112 downto 0));
    xip1_29_uid515_sincosTest_b <= STD_LOGIC_VECTOR(xip1_29_uid515_sincosTest_in(112 downto 0));

    -- redist95_xip1_29_uid515_sincosTest_b_1(DELAY,1132)
    redist95_xip1_29_uid515_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_29_uid515_sincosTest_b, xout => redist95_xip1_29_uid515_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_30_uid527_sincosTest(ADDSUB,526)@31
    xip1E_30_uid527_sincosTest_s <= redist93_xMSB_uid518_sincosTest_b_3_q;
    xip1E_30_uid527_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist95_xip1_29_uid515_sincosTest_b_1_q(112)) & redist95_xip1_29_uid515_sincosTest_b_1_q));
    xip1E_30_uid527_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 84 => twoToMiSiYip_uid524_sincosTest_b(83)) & twoToMiSiYip_uid524_sincosTest_b));
    xip1E_30_uid527_sincosTest_combproc: PROCESS (xip1E_30_uid527_sincosTest_a, xip1E_30_uid527_sincosTest_b, xip1E_30_uid527_sincosTest_s)
    BEGIN
        IF (xip1E_30_uid527_sincosTest_s = "1") THEN
            xip1E_30_uid527_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_30_uid527_sincosTest_a) + SIGNED(xip1E_30_uid527_sincosTest_b));
        ELSE
            xip1E_30_uid527_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_30_uid527_sincosTest_a) - SIGNED(xip1E_30_uid527_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_30_uid527_sincosTest_q <= xip1E_30_uid527_sincosTest_o(113 downto 0);

    -- xip1_30_uid534_sincosTest(BITSELECT,533)@31
    xip1_30_uid534_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_30_uid527_sincosTest_q(112 downto 0));
    xip1_30_uid534_sincosTest_b <= STD_LOGIC_VECTOR(xip1_30_uid534_sincosTest_in(112 downto 0));

    -- redist92_xip1_30_uid534_sincosTest_b_1(DELAY,1129)
    redist92_xip1_30_uid534_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_30_uid534_sincosTest_b, xout => redist92_xip1_30_uid534_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid542_sincosTest(BITSELECT,541)@32
    twoToMiSiXip_uid542_sincosTest_b <= STD_LOGIC_VECTOR(redist92_xip1_30_uid534_sincosTest_b_1_q(112 downto 30));

    -- signOfSelectionSignal_uid520_sincosTest(LOGICAL,519)@31
    signOfSelectionSignal_uid520_sincosTest_q <= not (redist93_xMSB_uid518_sincosTest_b_3_q);

    -- twoToMiSiXip_uid523_sincosTest(BITSELECT,522)@31
    twoToMiSiXip_uid523_sincosTest_b <= STD_LOGIC_VECTOR(redist95_xip1_29_uid515_sincosTest_b_1_q(112 downto 29));

    -- yip1E_30_uid528_sincosTest(ADDSUB,527)@31
    yip1E_30_uid528_sincosTest_s <= signOfSelectionSignal_uid520_sincosTest_q;
    yip1E_30_uid528_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist94_yip1_29_uid516_sincosTest_b_1_q(112)) & redist94_yip1_29_uid516_sincosTest_b_1_q));
    yip1E_30_uid528_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 84 => twoToMiSiXip_uid523_sincosTest_b(83)) & twoToMiSiXip_uid523_sincosTest_b));
    yip1E_30_uid528_sincosTest_combproc: PROCESS (yip1E_30_uid528_sincosTest_a, yip1E_30_uid528_sincosTest_b, yip1E_30_uid528_sincosTest_s)
    BEGIN
        IF (yip1E_30_uid528_sincosTest_s = "1") THEN
            yip1E_30_uid528_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_30_uid528_sincosTest_a) + SIGNED(yip1E_30_uid528_sincosTest_b));
        ELSE
            yip1E_30_uid528_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_30_uid528_sincosTest_a) - SIGNED(yip1E_30_uid528_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_30_uid528_sincosTest_q <= yip1E_30_uid528_sincosTest_o(113 downto 0);

    -- yip1_30_uid535_sincosTest(BITSELECT,534)@31
    yip1_30_uid535_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_30_uid528_sincosTest_q(112 downto 0));
    yip1_30_uid535_sincosTest_b <= STD_LOGIC_VECTOR(yip1_30_uid535_sincosTest_in(112 downto 0));

    -- redist91_yip1_30_uid535_sincosTest_b_1(DELAY,1128)
    redist91_yip1_30_uid535_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_30_uid535_sincosTest_b, xout => redist91_yip1_30_uid535_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_31_uid547_sincosTest(ADDSUB,546)@32
    yip1E_31_uid547_sincosTest_s <= signOfSelectionSignal_uid539_sincosTest_q;
    yip1E_31_uid547_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist91_yip1_30_uid535_sincosTest_b_1_q(112)) & redist91_yip1_30_uid535_sincosTest_b_1_q));
    yip1E_31_uid547_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 83 => twoToMiSiXip_uid542_sincosTest_b(82)) & twoToMiSiXip_uid542_sincosTest_b));
    yip1E_31_uid547_sincosTest_combproc: PROCESS (yip1E_31_uid547_sincosTest_a, yip1E_31_uid547_sincosTest_b, yip1E_31_uid547_sincosTest_s)
    BEGIN
        IF (yip1E_31_uid547_sincosTest_s = "1") THEN
            yip1E_31_uid547_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_31_uid547_sincosTest_a) + SIGNED(yip1E_31_uid547_sincosTest_b));
        ELSE
            yip1E_31_uid547_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_31_uid547_sincosTest_a) - SIGNED(yip1E_31_uid547_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_31_uid547_sincosTest_q <= yip1E_31_uid547_sincosTest_o(113 downto 0);

    -- yip1_31_uid554_sincosTest(BITSELECT,553)@32
    yip1_31_uid554_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_31_uid547_sincosTest_q(112 downto 0));
    yip1_31_uid554_sincosTest_b <= STD_LOGIC_VECTOR(yip1_31_uid554_sincosTest_in(112 downto 0));

    -- redist87_yip1_31_uid554_sincosTest_b_1(DELAY,1124)
    redist87_yip1_31_uid554_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_31_uid554_sincosTest_b, xout => redist87_yip1_31_uid554_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid562_sincosTest(BITSELECT,561)@33
    twoToMiSiYip_uid562_sincosTest_b <= STD_LOGIC_VECTOR(redist87_yip1_31_uid554_sincosTest_b_1_q(112 downto 31));

    -- twoToMiSiYip_uid543_sincosTest(BITSELECT,542)@32
    twoToMiSiYip_uid543_sincosTest_b <= STD_LOGIC_VECTOR(redist91_yip1_30_uid535_sincosTest_b_1_q(112 downto 30));

    -- xip1E_31_uid546_sincosTest(ADDSUB,545)@32
    xip1E_31_uid546_sincosTest_s <= redist89_xMSB_uid537_sincosTest_b_3_q;
    xip1E_31_uid546_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist92_xip1_30_uid534_sincosTest_b_1_q(112)) & redist92_xip1_30_uid534_sincosTest_b_1_q));
    xip1E_31_uid546_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 83 => twoToMiSiYip_uid543_sincosTest_b(82)) & twoToMiSiYip_uid543_sincosTest_b));
    xip1E_31_uid546_sincosTest_combproc: PROCESS (xip1E_31_uid546_sincosTest_a, xip1E_31_uid546_sincosTest_b, xip1E_31_uid546_sincosTest_s)
    BEGIN
        IF (xip1E_31_uid546_sincosTest_s = "1") THEN
            xip1E_31_uid546_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_31_uid546_sincosTest_a) + SIGNED(xip1E_31_uid546_sincosTest_b));
        ELSE
            xip1E_31_uid546_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_31_uid546_sincosTest_a) - SIGNED(xip1E_31_uid546_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_31_uid546_sincosTest_q <= xip1E_31_uid546_sincosTest_o(113 downto 0);

    -- xip1_31_uid553_sincosTest(BITSELECT,552)@32
    xip1_31_uid553_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_31_uid546_sincosTest_q(112 downto 0));
    xip1_31_uid553_sincosTest_b <= STD_LOGIC_VECTOR(xip1_31_uid553_sincosTest_in(112 downto 0));

    -- redist88_xip1_31_uid553_sincosTest_b_1(DELAY,1125)
    redist88_xip1_31_uid553_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_31_uid553_sincosTest_b, xout => redist88_xip1_31_uid553_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_32_uid565_sincosTest(ADDSUB,564)@33
    xip1E_32_uid565_sincosTest_s <= redist86_xMSB_uid556_sincosTest_b_4_q;
    xip1E_32_uid565_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist88_xip1_31_uid553_sincosTest_b_1_q(112)) & redist88_xip1_31_uid553_sincosTest_b_1_q));
    xip1E_32_uid565_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 82 => twoToMiSiYip_uid562_sincosTest_b(81)) & twoToMiSiYip_uid562_sincosTest_b));
    xip1E_32_uid565_sincosTest_combproc: PROCESS (xip1E_32_uid565_sincosTest_a, xip1E_32_uid565_sincosTest_b, xip1E_32_uid565_sincosTest_s)
    BEGIN
        IF (xip1E_32_uid565_sincosTest_s = "1") THEN
            xip1E_32_uid565_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_32_uid565_sincosTest_a) + SIGNED(xip1E_32_uid565_sincosTest_b));
        ELSE
            xip1E_32_uid565_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_32_uid565_sincosTest_a) - SIGNED(xip1E_32_uid565_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_32_uid565_sincosTest_q <= xip1E_32_uid565_sincosTest_o(113 downto 0);

    -- xip1_32_uid572_sincosTest(BITSELECT,571)@33
    xip1_32_uid572_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_32_uid565_sincosTest_q(112 downto 0));
    xip1_32_uid572_sincosTest_b <= STD_LOGIC_VECTOR(xip1_32_uid572_sincosTest_in(112 downto 0));

    -- redist85_xip1_32_uid572_sincosTest_b_1(DELAY,1122)
    redist85_xip1_32_uid572_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_32_uid572_sincosTest_b, xout => redist85_xip1_32_uid572_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid580_sincosTest(BITSELECT,579)@34
    twoToMiSiXip_uid580_sincosTest_b <= STD_LOGIC_VECTOR(redist85_xip1_32_uid572_sincosTest_b_1_q(112 downto 32));

    -- signOfSelectionSignal_uid558_sincosTest(LOGICAL,557)@33
    signOfSelectionSignal_uid558_sincosTest_q <= not (redist86_xMSB_uid556_sincosTest_b_4_q);

    -- twoToMiSiXip_uid561_sincosTest(BITSELECT,560)@33
    twoToMiSiXip_uid561_sincosTest_b <= STD_LOGIC_VECTOR(redist88_xip1_31_uid553_sincosTest_b_1_q(112 downto 31));

    -- yip1E_32_uid566_sincosTest(ADDSUB,565)@33
    yip1E_32_uid566_sincosTest_s <= signOfSelectionSignal_uid558_sincosTest_q;
    yip1E_32_uid566_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist87_yip1_31_uid554_sincosTest_b_1_q(112)) & redist87_yip1_31_uid554_sincosTest_b_1_q));
    yip1E_32_uid566_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 82 => twoToMiSiXip_uid561_sincosTest_b(81)) & twoToMiSiXip_uid561_sincosTest_b));
    yip1E_32_uid566_sincosTest_combproc: PROCESS (yip1E_32_uid566_sincosTest_a, yip1E_32_uid566_sincosTest_b, yip1E_32_uid566_sincosTest_s)
    BEGIN
        IF (yip1E_32_uid566_sincosTest_s = "1") THEN
            yip1E_32_uid566_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_32_uid566_sincosTest_a) + SIGNED(yip1E_32_uid566_sincosTest_b));
        ELSE
            yip1E_32_uid566_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_32_uid566_sincosTest_a) - SIGNED(yip1E_32_uid566_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_32_uid566_sincosTest_q <= yip1E_32_uid566_sincosTest_o(113 downto 0);

    -- yip1_32_uid573_sincosTest(BITSELECT,572)@33
    yip1_32_uid573_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_32_uid566_sincosTest_q(112 downto 0));
    yip1_32_uid573_sincosTest_b <= STD_LOGIC_VECTOR(yip1_32_uid573_sincosTest_in(112 downto 0));

    -- redist84_yip1_32_uid573_sincosTest_b_1(DELAY,1121)
    redist84_yip1_32_uid573_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_32_uid573_sincosTest_b, xout => redist84_yip1_32_uid573_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_33_uid585_sincosTest(ADDSUB,584)@34
    yip1E_33_uid585_sincosTest_s <= signOfSelectionSignal_uid577_sincosTest_q;
    yip1E_33_uid585_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist84_yip1_32_uid573_sincosTest_b_1_q(112)) & redist84_yip1_32_uid573_sincosTest_b_1_q));
    yip1E_33_uid585_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 81 => twoToMiSiXip_uid580_sincosTest_b(80)) & twoToMiSiXip_uid580_sincosTest_b));
    yip1E_33_uid585_sincosTest_combproc: PROCESS (yip1E_33_uid585_sincosTest_a, yip1E_33_uid585_sincosTest_b, yip1E_33_uid585_sincosTest_s)
    BEGIN
        IF (yip1E_33_uid585_sincosTest_s = "1") THEN
            yip1E_33_uid585_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_33_uid585_sincosTest_a) + SIGNED(yip1E_33_uid585_sincosTest_b));
        ELSE
            yip1E_33_uid585_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_33_uid585_sincosTest_a) - SIGNED(yip1E_33_uid585_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_33_uid585_sincosTest_q <= yip1E_33_uid585_sincosTest_o(113 downto 0);

    -- yip1_33_uid592_sincosTest(BITSELECT,591)@34
    yip1_33_uid592_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_33_uid585_sincosTest_q(112 downto 0));
    yip1_33_uid592_sincosTest_b <= STD_LOGIC_VECTOR(yip1_33_uid592_sincosTest_in(112 downto 0));

    -- redist80_yip1_33_uid592_sincosTest_b_1(DELAY,1117)
    redist80_yip1_33_uid592_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_33_uid592_sincosTest_b, xout => redist80_yip1_33_uid592_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid600_sincosTest(BITSELECT,599)@35
    twoToMiSiYip_uid600_sincosTest_b <= STD_LOGIC_VECTOR(redist80_yip1_33_uid592_sincosTest_b_1_q(112 downto 33));

    -- twoToMiSiYip_uid581_sincosTest(BITSELECT,580)@34
    twoToMiSiYip_uid581_sincosTest_b <= STD_LOGIC_VECTOR(redist84_yip1_32_uid573_sincosTest_b_1_q(112 downto 32));

    -- xip1E_33_uid584_sincosTest(ADDSUB,583)@34
    xip1E_33_uid584_sincosTest_s <= redist82_xMSB_uid575_sincosTest_b_4_q;
    xip1E_33_uid584_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist85_xip1_32_uid572_sincosTest_b_1_q(112)) & redist85_xip1_32_uid572_sincosTest_b_1_q));
    xip1E_33_uid584_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 81 => twoToMiSiYip_uid581_sincosTest_b(80)) & twoToMiSiYip_uid581_sincosTest_b));
    xip1E_33_uid584_sincosTest_combproc: PROCESS (xip1E_33_uid584_sincosTest_a, xip1E_33_uid584_sincosTest_b, xip1E_33_uid584_sincosTest_s)
    BEGIN
        IF (xip1E_33_uid584_sincosTest_s = "1") THEN
            xip1E_33_uid584_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_33_uid584_sincosTest_a) + SIGNED(xip1E_33_uid584_sincosTest_b));
        ELSE
            xip1E_33_uid584_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_33_uid584_sincosTest_a) - SIGNED(xip1E_33_uid584_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_33_uid584_sincosTest_q <= xip1E_33_uid584_sincosTest_o(113 downto 0);

    -- xip1_33_uid591_sincosTest(BITSELECT,590)@34
    xip1_33_uid591_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_33_uid584_sincosTest_q(112 downto 0));
    xip1_33_uid591_sincosTest_b <= STD_LOGIC_VECTOR(xip1_33_uid591_sincosTest_in(112 downto 0));

    -- redist81_xip1_33_uid591_sincosTest_b_1(DELAY,1118)
    redist81_xip1_33_uid591_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_33_uid591_sincosTest_b, xout => redist81_xip1_33_uid591_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_34_uid603_sincosTest(ADDSUB,602)@35
    xip1E_34_uid603_sincosTest_s <= redist79_xMSB_uid594_sincosTest_b_5_q;
    xip1E_34_uid603_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist81_xip1_33_uid591_sincosTest_b_1_q(112)) & redist81_xip1_33_uid591_sincosTest_b_1_q));
    xip1E_34_uid603_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 80 => twoToMiSiYip_uid600_sincosTest_b(79)) & twoToMiSiYip_uid600_sincosTest_b));
    xip1E_34_uid603_sincosTest_combproc: PROCESS (xip1E_34_uid603_sincosTest_a, xip1E_34_uid603_sincosTest_b, xip1E_34_uid603_sincosTest_s)
    BEGIN
        IF (xip1E_34_uid603_sincosTest_s = "1") THEN
            xip1E_34_uid603_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_34_uid603_sincosTest_a) + SIGNED(xip1E_34_uid603_sincosTest_b));
        ELSE
            xip1E_34_uid603_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_34_uid603_sincosTest_a) - SIGNED(xip1E_34_uid603_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_34_uid603_sincosTest_q <= xip1E_34_uid603_sincosTest_o(113 downto 0);

    -- xip1_34_uid610_sincosTest(BITSELECT,609)@35
    xip1_34_uid610_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_34_uid603_sincosTest_q(112 downto 0));
    xip1_34_uid610_sincosTest_b <= STD_LOGIC_VECTOR(xip1_34_uid610_sincosTest_in(112 downto 0));

    -- redist78_xip1_34_uid610_sincosTest_b_1(DELAY,1115)
    redist78_xip1_34_uid610_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_34_uid610_sincosTest_b, xout => redist78_xip1_34_uid610_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid618_sincosTest(BITSELECT,617)@36
    twoToMiSiXip_uid618_sincosTest_b <= STD_LOGIC_VECTOR(redist78_xip1_34_uid610_sincosTest_b_1_q(112 downto 34));

    -- signOfSelectionSignal_uid596_sincosTest(LOGICAL,595)@35
    signOfSelectionSignal_uid596_sincosTest_q <= not (redist79_xMSB_uid594_sincosTest_b_5_q);

    -- twoToMiSiXip_uid599_sincosTest(BITSELECT,598)@35
    twoToMiSiXip_uid599_sincosTest_b <= STD_LOGIC_VECTOR(redist81_xip1_33_uid591_sincosTest_b_1_q(112 downto 33));

    -- yip1E_34_uid604_sincosTest(ADDSUB,603)@35
    yip1E_34_uid604_sincosTest_s <= signOfSelectionSignal_uid596_sincosTest_q;
    yip1E_34_uid604_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist80_yip1_33_uid592_sincosTest_b_1_q(112)) & redist80_yip1_33_uid592_sincosTest_b_1_q));
    yip1E_34_uid604_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 80 => twoToMiSiXip_uid599_sincosTest_b(79)) & twoToMiSiXip_uid599_sincosTest_b));
    yip1E_34_uid604_sincosTest_combproc: PROCESS (yip1E_34_uid604_sincosTest_a, yip1E_34_uid604_sincosTest_b, yip1E_34_uid604_sincosTest_s)
    BEGIN
        IF (yip1E_34_uid604_sincosTest_s = "1") THEN
            yip1E_34_uid604_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_34_uid604_sincosTest_a) + SIGNED(yip1E_34_uid604_sincosTest_b));
        ELSE
            yip1E_34_uid604_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_34_uid604_sincosTest_a) - SIGNED(yip1E_34_uid604_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_34_uid604_sincosTest_q <= yip1E_34_uid604_sincosTest_o(113 downto 0);

    -- yip1_34_uid611_sincosTest(BITSELECT,610)@35
    yip1_34_uid611_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_34_uid604_sincosTest_q(112 downto 0));
    yip1_34_uid611_sincosTest_b <= STD_LOGIC_VECTOR(yip1_34_uid611_sincosTest_in(112 downto 0));

    -- redist77_yip1_34_uid611_sincosTest_b_1(DELAY,1114)
    redist77_yip1_34_uid611_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_34_uid611_sincosTest_b, xout => redist77_yip1_34_uid611_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_35_uid623_sincosTest(ADDSUB,622)@36
    yip1E_35_uid623_sincosTest_s <= signOfSelectionSignal_uid615_sincosTest_q;
    yip1E_35_uid623_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist77_yip1_34_uid611_sincosTest_b_1_q(112)) & redist77_yip1_34_uid611_sincosTest_b_1_q));
    yip1E_35_uid623_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 79 => twoToMiSiXip_uid618_sincosTest_b(78)) & twoToMiSiXip_uid618_sincosTest_b));
    yip1E_35_uid623_sincosTest_combproc: PROCESS (yip1E_35_uid623_sincosTest_a, yip1E_35_uid623_sincosTest_b, yip1E_35_uid623_sincosTest_s)
    BEGIN
        IF (yip1E_35_uid623_sincosTest_s = "1") THEN
            yip1E_35_uid623_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_35_uid623_sincosTest_a) + SIGNED(yip1E_35_uid623_sincosTest_b));
        ELSE
            yip1E_35_uid623_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_35_uid623_sincosTest_a) - SIGNED(yip1E_35_uid623_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_35_uid623_sincosTest_q <= yip1E_35_uid623_sincosTest_o(113 downto 0);

    -- yip1_35_uid630_sincosTest(BITSELECT,629)@36
    yip1_35_uid630_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_35_uid623_sincosTest_q(112 downto 0));
    yip1_35_uid630_sincosTest_b <= STD_LOGIC_VECTOR(yip1_35_uid630_sincosTest_in(112 downto 0));

    -- redist73_yip1_35_uid630_sincosTest_b_1(DELAY,1110)
    redist73_yip1_35_uid630_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_35_uid630_sincosTest_b, xout => redist73_yip1_35_uid630_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid638_sincosTest(BITSELECT,637)@37
    twoToMiSiYip_uid638_sincosTest_b <= STD_LOGIC_VECTOR(redist73_yip1_35_uid630_sincosTest_b_1_q(112 downto 35));

    -- twoToMiSiYip_uid619_sincosTest(BITSELECT,618)@36
    twoToMiSiYip_uid619_sincosTest_b <= STD_LOGIC_VECTOR(redist77_yip1_34_uid611_sincosTest_b_1_q(112 downto 34));

    -- xip1E_35_uid622_sincosTest(ADDSUB,621)@36
    xip1E_35_uid622_sincosTest_s <= redist75_xMSB_uid613_sincosTest_b_5_q;
    xip1E_35_uid622_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist78_xip1_34_uid610_sincosTest_b_1_q(112)) & redist78_xip1_34_uid610_sincosTest_b_1_q));
    xip1E_35_uid622_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 79 => twoToMiSiYip_uid619_sincosTest_b(78)) & twoToMiSiYip_uid619_sincosTest_b));
    xip1E_35_uid622_sincosTest_combproc: PROCESS (xip1E_35_uid622_sincosTest_a, xip1E_35_uid622_sincosTest_b, xip1E_35_uid622_sincosTest_s)
    BEGIN
        IF (xip1E_35_uid622_sincosTest_s = "1") THEN
            xip1E_35_uid622_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_35_uid622_sincosTest_a) + SIGNED(xip1E_35_uid622_sincosTest_b));
        ELSE
            xip1E_35_uid622_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_35_uid622_sincosTest_a) - SIGNED(xip1E_35_uid622_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_35_uid622_sincosTest_q <= xip1E_35_uid622_sincosTest_o(113 downto 0);

    -- xip1_35_uid629_sincosTest(BITSELECT,628)@36
    xip1_35_uid629_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_35_uid622_sincosTest_q(112 downto 0));
    xip1_35_uid629_sincosTest_b <= STD_LOGIC_VECTOR(xip1_35_uid629_sincosTest_in(112 downto 0));

    -- redist74_xip1_35_uid629_sincosTest_b_1(DELAY,1111)
    redist74_xip1_35_uid629_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_35_uid629_sincosTest_b, xout => redist74_xip1_35_uid629_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_36_uid641_sincosTest(ADDSUB,640)@37
    xip1E_36_uid641_sincosTest_s <= redist72_xMSB_uid632_sincosTest_b_6_q;
    xip1E_36_uid641_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist74_xip1_35_uid629_sincosTest_b_1_q(112)) & redist74_xip1_35_uid629_sincosTest_b_1_q));
    xip1E_36_uid641_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 78 => twoToMiSiYip_uid638_sincosTest_b(77)) & twoToMiSiYip_uid638_sincosTest_b));
    xip1E_36_uid641_sincosTest_combproc: PROCESS (xip1E_36_uid641_sincosTest_a, xip1E_36_uid641_sincosTest_b, xip1E_36_uid641_sincosTest_s)
    BEGIN
        IF (xip1E_36_uid641_sincosTest_s = "1") THEN
            xip1E_36_uid641_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_36_uid641_sincosTest_a) + SIGNED(xip1E_36_uid641_sincosTest_b));
        ELSE
            xip1E_36_uid641_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_36_uid641_sincosTest_a) - SIGNED(xip1E_36_uid641_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_36_uid641_sincosTest_q <= xip1E_36_uid641_sincosTest_o(113 downto 0);

    -- xip1_36_uid648_sincosTest(BITSELECT,647)@37
    xip1_36_uid648_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_36_uid641_sincosTest_q(112 downto 0));
    xip1_36_uid648_sincosTest_b <= STD_LOGIC_VECTOR(xip1_36_uid648_sincosTest_in(112 downto 0));

    -- redist71_xip1_36_uid648_sincosTest_b_1(DELAY,1108)
    redist71_xip1_36_uid648_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_36_uid648_sincosTest_b, xout => redist71_xip1_36_uid648_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid656_sincosTest(BITSELECT,655)@38
    twoToMiSiXip_uid656_sincosTest_b <= STD_LOGIC_VECTOR(redist71_xip1_36_uid648_sincosTest_b_1_q(112 downto 36));

    -- signOfSelectionSignal_uid634_sincosTest(LOGICAL,633)@37
    signOfSelectionSignal_uid634_sincosTest_q <= not (redist72_xMSB_uid632_sincosTest_b_6_q);

    -- twoToMiSiXip_uid637_sincosTest(BITSELECT,636)@37
    twoToMiSiXip_uid637_sincosTest_b <= STD_LOGIC_VECTOR(redist74_xip1_35_uid629_sincosTest_b_1_q(112 downto 35));

    -- yip1E_36_uid642_sincosTest(ADDSUB,641)@37
    yip1E_36_uid642_sincosTest_s <= signOfSelectionSignal_uid634_sincosTest_q;
    yip1E_36_uid642_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist73_yip1_35_uid630_sincosTest_b_1_q(112)) & redist73_yip1_35_uid630_sincosTest_b_1_q));
    yip1E_36_uid642_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 78 => twoToMiSiXip_uid637_sincosTest_b(77)) & twoToMiSiXip_uid637_sincosTest_b));
    yip1E_36_uid642_sincosTest_combproc: PROCESS (yip1E_36_uid642_sincosTest_a, yip1E_36_uid642_sincosTest_b, yip1E_36_uid642_sincosTest_s)
    BEGIN
        IF (yip1E_36_uid642_sincosTest_s = "1") THEN
            yip1E_36_uid642_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_36_uid642_sincosTest_a) + SIGNED(yip1E_36_uid642_sincosTest_b));
        ELSE
            yip1E_36_uid642_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_36_uid642_sincosTest_a) - SIGNED(yip1E_36_uid642_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_36_uid642_sincosTest_q <= yip1E_36_uid642_sincosTest_o(113 downto 0);

    -- yip1_36_uid649_sincosTest(BITSELECT,648)@37
    yip1_36_uid649_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_36_uid642_sincosTest_q(112 downto 0));
    yip1_36_uid649_sincosTest_b <= STD_LOGIC_VECTOR(yip1_36_uid649_sincosTest_in(112 downto 0));

    -- redist70_yip1_36_uid649_sincosTest_b_1(DELAY,1107)
    redist70_yip1_36_uid649_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_36_uid649_sincosTest_b, xout => redist70_yip1_36_uid649_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_37_uid661_sincosTest(ADDSUB,660)@38
    yip1E_37_uid661_sincosTest_s <= signOfSelectionSignal_uid653_sincosTest_q;
    yip1E_37_uid661_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist70_yip1_36_uid649_sincosTest_b_1_q(112)) & redist70_yip1_36_uid649_sincosTest_b_1_q));
    yip1E_37_uid661_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 77 => twoToMiSiXip_uid656_sincosTest_b(76)) & twoToMiSiXip_uid656_sincosTest_b));
    yip1E_37_uid661_sincosTest_combproc: PROCESS (yip1E_37_uid661_sincosTest_a, yip1E_37_uid661_sincosTest_b, yip1E_37_uid661_sincosTest_s)
    BEGIN
        IF (yip1E_37_uid661_sincosTest_s = "1") THEN
            yip1E_37_uid661_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_37_uid661_sincosTest_a) + SIGNED(yip1E_37_uid661_sincosTest_b));
        ELSE
            yip1E_37_uid661_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_37_uid661_sincosTest_a) - SIGNED(yip1E_37_uid661_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_37_uid661_sincosTest_q <= yip1E_37_uid661_sincosTest_o(113 downto 0);

    -- yip1_37_uid668_sincosTest(BITSELECT,667)@38
    yip1_37_uid668_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_37_uid661_sincosTest_q(112 downto 0));
    yip1_37_uid668_sincosTest_b <= STD_LOGIC_VECTOR(yip1_37_uid668_sincosTest_in(112 downto 0));

    -- redist66_yip1_37_uid668_sincosTest_b_1(DELAY,1103)
    redist66_yip1_37_uid668_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_37_uid668_sincosTest_b, xout => redist66_yip1_37_uid668_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid676_sincosTest(BITSELECT,675)@39
    twoToMiSiYip_uid676_sincosTest_b <= STD_LOGIC_VECTOR(redist66_yip1_37_uid668_sincosTest_b_1_q(112 downto 37));

    -- twoToMiSiYip_uid657_sincosTest(BITSELECT,656)@38
    twoToMiSiYip_uid657_sincosTest_b <= STD_LOGIC_VECTOR(redist70_yip1_36_uid649_sincosTest_b_1_q(112 downto 36));

    -- xip1E_37_uid660_sincosTest(ADDSUB,659)@38
    xip1E_37_uid660_sincosTest_s <= redist68_xMSB_uid651_sincosTest_b_6_q;
    xip1E_37_uid660_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist71_xip1_36_uid648_sincosTest_b_1_q(112)) & redist71_xip1_36_uid648_sincosTest_b_1_q));
    xip1E_37_uid660_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 77 => twoToMiSiYip_uid657_sincosTest_b(76)) & twoToMiSiYip_uid657_sincosTest_b));
    xip1E_37_uid660_sincosTest_combproc: PROCESS (xip1E_37_uid660_sincosTest_a, xip1E_37_uid660_sincosTest_b, xip1E_37_uid660_sincosTest_s)
    BEGIN
        IF (xip1E_37_uid660_sincosTest_s = "1") THEN
            xip1E_37_uid660_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_37_uid660_sincosTest_a) + SIGNED(xip1E_37_uid660_sincosTest_b));
        ELSE
            xip1E_37_uid660_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_37_uid660_sincosTest_a) - SIGNED(xip1E_37_uid660_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_37_uid660_sincosTest_q <= xip1E_37_uid660_sincosTest_o(113 downto 0);

    -- xip1_37_uid667_sincosTest(BITSELECT,666)@38
    xip1_37_uid667_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_37_uid660_sincosTest_q(112 downto 0));
    xip1_37_uid667_sincosTest_b <= STD_LOGIC_VECTOR(xip1_37_uid667_sincosTest_in(112 downto 0));

    -- redist67_xip1_37_uid667_sincosTest_b_1(DELAY,1104)
    redist67_xip1_37_uid667_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_37_uid667_sincosTest_b, xout => redist67_xip1_37_uid667_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_38_uid679_sincosTest(ADDSUB,678)@39
    xip1E_38_uid679_sincosTest_s <= redist65_xMSB_uid670_sincosTest_b_7_q;
    xip1E_38_uid679_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist67_xip1_37_uid667_sincosTest_b_1_q(112)) & redist67_xip1_37_uid667_sincosTest_b_1_q));
    xip1E_38_uid679_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 76 => twoToMiSiYip_uid676_sincosTest_b(75)) & twoToMiSiYip_uid676_sincosTest_b));
    xip1E_38_uid679_sincosTest_combproc: PROCESS (xip1E_38_uid679_sincosTest_a, xip1E_38_uid679_sincosTest_b, xip1E_38_uid679_sincosTest_s)
    BEGIN
        IF (xip1E_38_uid679_sincosTest_s = "1") THEN
            xip1E_38_uid679_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_38_uid679_sincosTest_a) + SIGNED(xip1E_38_uid679_sincosTest_b));
        ELSE
            xip1E_38_uid679_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_38_uid679_sincosTest_a) - SIGNED(xip1E_38_uid679_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_38_uid679_sincosTest_q <= xip1E_38_uid679_sincosTest_o(113 downto 0);

    -- xip1_38_uid686_sincosTest(BITSELECT,685)@39
    xip1_38_uid686_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_38_uid679_sincosTest_q(112 downto 0));
    xip1_38_uid686_sincosTest_b <= STD_LOGIC_VECTOR(xip1_38_uid686_sincosTest_in(112 downto 0));

    -- redist64_xip1_38_uid686_sincosTest_b_1(DELAY,1101)
    redist64_xip1_38_uid686_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_38_uid686_sincosTest_b, xout => redist64_xip1_38_uid686_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid694_sincosTest(BITSELECT,693)@40
    twoToMiSiXip_uid694_sincosTest_b <= STD_LOGIC_VECTOR(redist64_xip1_38_uid686_sincosTest_b_1_q(112 downto 38));

    -- signOfSelectionSignal_uid672_sincosTest(LOGICAL,671)@39
    signOfSelectionSignal_uid672_sincosTest_q <= not (redist65_xMSB_uid670_sincosTest_b_7_q);

    -- twoToMiSiXip_uid675_sincosTest(BITSELECT,674)@39
    twoToMiSiXip_uid675_sincosTest_b <= STD_LOGIC_VECTOR(redist67_xip1_37_uid667_sincosTest_b_1_q(112 downto 37));

    -- yip1E_38_uid680_sincosTest(ADDSUB,679)@39
    yip1E_38_uid680_sincosTest_s <= signOfSelectionSignal_uid672_sincosTest_q;
    yip1E_38_uid680_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist66_yip1_37_uid668_sincosTest_b_1_q(112)) & redist66_yip1_37_uid668_sincosTest_b_1_q));
    yip1E_38_uid680_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 76 => twoToMiSiXip_uid675_sincosTest_b(75)) & twoToMiSiXip_uid675_sincosTest_b));
    yip1E_38_uid680_sincosTest_combproc: PROCESS (yip1E_38_uid680_sincosTest_a, yip1E_38_uid680_sincosTest_b, yip1E_38_uid680_sincosTest_s)
    BEGIN
        IF (yip1E_38_uid680_sincosTest_s = "1") THEN
            yip1E_38_uid680_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_38_uid680_sincosTest_a) + SIGNED(yip1E_38_uid680_sincosTest_b));
        ELSE
            yip1E_38_uid680_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_38_uid680_sincosTest_a) - SIGNED(yip1E_38_uid680_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_38_uid680_sincosTest_q <= yip1E_38_uid680_sincosTest_o(113 downto 0);

    -- yip1_38_uid687_sincosTest(BITSELECT,686)@39
    yip1_38_uid687_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_38_uid680_sincosTest_q(112 downto 0));
    yip1_38_uid687_sincosTest_b <= STD_LOGIC_VECTOR(yip1_38_uid687_sincosTest_in(112 downto 0));

    -- redist63_yip1_38_uid687_sincosTest_b_1(DELAY,1100)
    redist63_yip1_38_uid687_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_38_uid687_sincosTest_b, xout => redist63_yip1_38_uid687_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_39_uid699_sincosTest(ADDSUB,698)@40
    yip1E_39_uid699_sincosTest_s <= signOfSelectionSignal_uid691_sincosTest_q;
    yip1E_39_uid699_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist63_yip1_38_uid687_sincosTest_b_1_q(112)) & redist63_yip1_38_uid687_sincosTest_b_1_q));
    yip1E_39_uid699_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 75 => twoToMiSiXip_uid694_sincosTest_b(74)) & twoToMiSiXip_uid694_sincosTest_b));
    yip1E_39_uid699_sincosTest_combproc: PROCESS (yip1E_39_uid699_sincosTest_a, yip1E_39_uid699_sincosTest_b, yip1E_39_uid699_sincosTest_s)
    BEGIN
        IF (yip1E_39_uid699_sincosTest_s = "1") THEN
            yip1E_39_uid699_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_39_uid699_sincosTest_a) + SIGNED(yip1E_39_uid699_sincosTest_b));
        ELSE
            yip1E_39_uid699_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_39_uid699_sincosTest_a) - SIGNED(yip1E_39_uid699_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_39_uid699_sincosTest_q <= yip1E_39_uid699_sincosTest_o(113 downto 0);

    -- yip1_39_uid706_sincosTest(BITSELECT,705)@40
    yip1_39_uid706_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_39_uid699_sincosTest_q(112 downto 0));
    yip1_39_uid706_sincosTest_b <= STD_LOGIC_VECTOR(yip1_39_uid706_sincosTest_in(112 downto 0));

    -- redist59_yip1_39_uid706_sincosTest_b_1(DELAY,1096)
    redist59_yip1_39_uid706_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_39_uid706_sincosTest_b, xout => redist59_yip1_39_uid706_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid714_sincosTest(BITSELECT,713)@41
    twoToMiSiYip_uid714_sincosTest_b <= STD_LOGIC_VECTOR(redist59_yip1_39_uid706_sincosTest_b_1_q(112 downto 39));

    -- twoToMiSiYip_uid695_sincosTest(BITSELECT,694)@40
    twoToMiSiYip_uid695_sincosTest_b <= STD_LOGIC_VECTOR(redist63_yip1_38_uid687_sincosTest_b_1_q(112 downto 38));

    -- xip1E_39_uid698_sincosTest(ADDSUB,697)@40
    xip1E_39_uid698_sincosTest_s <= redist61_xMSB_uid689_sincosTest_b_7_q;
    xip1E_39_uid698_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist64_xip1_38_uid686_sincosTest_b_1_q(112)) & redist64_xip1_38_uid686_sincosTest_b_1_q));
    xip1E_39_uid698_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 75 => twoToMiSiYip_uid695_sincosTest_b(74)) & twoToMiSiYip_uid695_sincosTest_b));
    xip1E_39_uid698_sincosTest_combproc: PROCESS (xip1E_39_uid698_sincosTest_a, xip1E_39_uid698_sincosTest_b, xip1E_39_uid698_sincosTest_s)
    BEGIN
        IF (xip1E_39_uid698_sincosTest_s = "1") THEN
            xip1E_39_uid698_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_39_uid698_sincosTest_a) + SIGNED(xip1E_39_uid698_sincosTest_b));
        ELSE
            xip1E_39_uid698_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_39_uid698_sincosTest_a) - SIGNED(xip1E_39_uid698_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_39_uid698_sincosTest_q <= xip1E_39_uid698_sincosTest_o(113 downto 0);

    -- xip1_39_uid705_sincosTest(BITSELECT,704)@40
    xip1_39_uid705_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_39_uid698_sincosTest_q(112 downto 0));
    xip1_39_uid705_sincosTest_b <= STD_LOGIC_VECTOR(xip1_39_uid705_sincosTest_in(112 downto 0));

    -- redist60_xip1_39_uid705_sincosTest_b_1(DELAY,1097)
    redist60_xip1_39_uid705_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_39_uid705_sincosTest_b, xout => redist60_xip1_39_uid705_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_40_uid717_sincosTest(ADDSUB,716)@41
    xip1E_40_uid717_sincosTest_s <= redist58_xMSB_uid708_sincosTest_b_8_q;
    xip1E_40_uid717_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist60_xip1_39_uid705_sincosTest_b_1_q(112)) & redist60_xip1_39_uid705_sincosTest_b_1_q));
    xip1E_40_uid717_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 74 => twoToMiSiYip_uid714_sincosTest_b(73)) & twoToMiSiYip_uid714_sincosTest_b));
    xip1E_40_uid717_sincosTest_combproc: PROCESS (xip1E_40_uid717_sincosTest_a, xip1E_40_uid717_sincosTest_b, xip1E_40_uid717_sincosTest_s)
    BEGIN
        IF (xip1E_40_uid717_sincosTest_s = "1") THEN
            xip1E_40_uid717_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_40_uid717_sincosTest_a) + SIGNED(xip1E_40_uid717_sincosTest_b));
        ELSE
            xip1E_40_uid717_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_40_uid717_sincosTest_a) - SIGNED(xip1E_40_uid717_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_40_uid717_sincosTest_q <= xip1E_40_uid717_sincosTest_o(113 downto 0);

    -- xip1_40_uid724_sincosTest(BITSELECT,723)@41
    xip1_40_uid724_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_40_uid717_sincosTest_q(112 downto 0));
    xip1_40_uid724_sincosTest_b <= STD_LOGIC_VECTOR(xip1_40_uid724_sincosTest_in(112 downto 0));

    -- redist57_xip1_40_uid724_sincosTest_b_1(DELAY,1094)
    redist57_xip1_40_uid724_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_40_uid724_sincosTest_b, xout => redist57_xip1_40_uid724_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid732_sincosTest(BITSELECT,731)@42
    twoToMiSiXip_uid732_sincosTest_b <= STD_LOGIC_VECTOR(redist57_xip1_40_uid724_sincosTest_b_1_q(112 downto 40));

    -- signOfSelectionSignal_uid710_sincosTest(LOGICAL,709)@41
    signOfSelectionSignal_uid710_sincosTest_q <= not (redist58_xMSB_uid708_sincosTest_b_8_q);

    -- twoToMiSiXip_uid713_sincosTest(BITSELECT,712)@41
    twoToMiSiXip_uid713_sincosTest_b <= STD_LOGIC_VECTOR(redist60_xip1_39_uid705_sincosTest_b_1_q(112 downto 39));

    -- yip1E_40_uid718_sincosTest(ADDSUB,717)@41
    yip1E_40_uid718_sincosTest_s <= signOfSelectionSignal_uid710_sincosTest_q;
    yip1E_40_uid718_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist59_yip1_39_uid706_sincosTest_b_1_q(112)) & redist59_yip1_39_uid706_sincosTest_b_1_q));
    yip1E_40_uid718_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 74 => twoToMiSiXip_uid713_sincosTest_b(73)) & twoToMiSiXip_uid713_sincosTest_b));
    yip1E_40_uid718_sincosTest_combproc: PROCESS (yip1E_40_uid718_sincosTest_a, yip1E_40_uid718_sincosTest_b, yip1E_40_uid718_sincosTest_s)
    BEGIN
        IF (yip1E_40_uid718_sincosTest_s = "1") THEN
            yip1E_40_uid718_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_40_uid718_sincosTest_a) + SIGNED(yip1E_40_uid718_sincosTest_b));
        ELSE
            yip1E_40_uid718_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_40_uid718_sincosTest_a) - SIGNED(yip1E_40_uid718_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_40_uid718_sincosTest_q <= yip1E_40_uid718_sincosTest_o(113 downto 0);

    -- yip1_40_uid725_sincosTest(BITSELECT,724)@41
    yip1_40_uid725_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_40_uid718_sincosTest_q(112 downto 0));
    yip1_40_uid725_sincosTest_b <= STD_LOGIC_VECTOR(yip1_40_uid725_sincosTest_in(112 downto 0));

    -- redist56_yip1_40_uid725_sincosTest_b_1(DELAY,1093)
    redist56_yip1_40_uid725_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_40_uid725_sincosTest_b, xout => redist56_yip1_40_uid725_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_41_uid737_sincosTest(ADDSUB,736)@42
    yip1E_41_uid737_sincosTest_s <= signOfSelectionSignal_uid729_sincosTest_q;
    yip1E_41_uid737_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist56_yip1_40_uid725_sincosTest_b_1_q(112)) & redist56_yip1_40_uid725_sincosTest_b_1_q));
    yip1E_41_uid737_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 73 => twoToMiSiXip_uid732_sincosTest_b(72)) & twoToMiSiXip_uid732_sincosTest_b));
    yip1E_41_uid737_sincosTest_combproc: PROCESS (yip1E_41_uid737_sincosTest_a, yip1E_41_uid737_sincosTest_b, yip1E_41_uid737_sincosTest_s)
    BEGIN
        IF (yip1E_41_uid737_sincosTest_s = "1") THEN
            yip1E_41_uid737_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_41_uid737_sincosTest_a) + SIGNED(yip1E_41_uid737_sincosTest_b));
        ELSE
            yip1E_41_uid737_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_41_uid737_sincosTest_a) - SIGNED(yip1E_41_uid737_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_41_uid737_sincosTest_q <= yip1E_41_uid737_sincosTest_o(113 downto 0);

    -- yip1_41_uid744_sincosTest(BITSELECT,743)@42
    yip1_41_uid744_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_41_uid737_sincosTest_q(112 downto 0));
    yip1_41_uid744_sincosTest_b <= STD_LOGIC_VECTOR(yip1_41_uid744_sincosTest_in(112 downto 0));

    -- redist52_yip1_41_uid744_sincosTest_b_1(DELAY,1089)
    redist52_yip1_41_uid744_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_41_uid744_sincosTest_b, xout => redist52_yip1_41_uid744_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid752_sincosTest(BITSELECT,751)@43
    twoToMiSiYip_uid752_sincosTest_b <= STD_LOGIC_VECTOR(redist52_yip1_41_uid744_sincosTest_b_1_q(112 downto 41));

    -- twoToMiSiYip_uid733_sincosTest(BITSELECT,732)@42
    twoToMiSiYip_uid733_sincosTest_b <= STD_LOGIC_VECTOR(redist56_yip1_40_uid725_sincosTest_b_1_q(112 downto 40));

    -- xip1E_41_uid736_sincosTest(ADDSUB,735)@42
    xip1E_41_uid736_sincosTest_s <= redist54_xMSB_uid727_sincosTest_b_8_q;
    xip1E_41_uid736_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist57_xip1_40_uid724_sincosTest_b_1_q(112)) & redist57_xip1_40_uid724_sincosTest_b_1_q));
    xip1E_41_uid736_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 73 => twoToMiSiYip_uid733_sincosTest_b(72)) & twoToMiSiYip_uid733_sincosTest_b));
    xip1E_41_uid736_sincosTest_combproc: PROCESS (xip1E_41_uid736_sincosTest_a, xip1E_41_uid736_sincosTest_b, xip1E_41_uid736_sincosTest_s)
    BEGIN
        IF (xip1E_41_uid736_sincosTest_s = "1") THEN
            xip1E_41_uid736_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_41_uid736_sincosTest_a) + SIGNED(xip1E_41_uid736_sincosTest_b));
        ELSE
            xip1E_41_uid736_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_41_uid736_sincosTest_a) - SIGNED(xip1E_41_uid736_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_41_uid736_sincosTest_q <= xip1E_41_uid736_sincosTest_o(113 downto 0);

    -- xip1_41_uid743_sincosTest(BITSELECT,742)@42
    xip1_41_uid743_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_41_uid736_sincosTest_q(112 downto 0));
    xip1_41_uid743_sincosTest_b <= STD_LOGIC_VECTOR(xip1_41_uid743_sincosTest_in(112 downto 0));

    -- redist53_xip1_41_uid743_sincosTest_b_1(DELAY,1090)
    redist53_xip1_41_uid743_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_41_uid743_sincosTest_b, xout => redist53_xip1_41_uid743_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_42_uid755_sincosTest(ADDSUB,754)@43
    xip1E_42_uid755_sincosTest_s <= redist51_xMSB_uid746_sincosTest_b_9_q;
    xip1E_42_uid755_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist53_xip1_41_uid743_sincosTest_b_1_q(112)) & redist53_xip1_41_uid743_sincosTest_b_1_q));
    xip1E_42_uid755_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 72 => twoToMiSiYip_uid752_sincosTest_b(71)) & twoToMiSiYip_uid752_sincosTest_b));
    xip1E_42_uid755_sincosTest_combproc: PROCESS (xip1E_42_uid755_sincosTest_a, xip1E_42_uid755_sincosTest_b, xip1E_42_uid755_sincosTest_s)
    BEGIN
        IF (xip1E_42_uid755_sincosTest_s = "1") THEN
            xip1E_42_uid755_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_42_uid755_sincosTest_a) + SIGNED(xip1E_42_uid755_sincosTest_b));
        ELSE
            xip1E_42_uid755_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_42_uid755_sincosTest_a) - SIGNED(xip1E_42_uid755_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_42_uid755_sincosTest_q <= xip1E_42_uid755_sincosTest_o(113 downto 0);

    -- xip1_42_uid762_sincosTest(BITSELECT,761)@43
    xip1_42_uid762_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_42_uid755_sincosTest_q(112 downto 0));
    xip1_42_uid762_sincosTest_b <= STD_LOGIC_VECTOR(xip1_42_uid762_sincosTest_in(112 downto 0));

    -- redist50_xip1_42_uid762_sincosTest_b_1(DELAY,1087)
    redist50_xip1_42_uid762_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_42_uid762_sincosTest_b, xout => redist50_xip1_42_uid762_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid770_sincosTest(BITSELECT,769)@44
    twoToMiSiXip_uid770_sincosTest_b <= STD_LOGIC_VECTOR(redist50_xip1_42_uid762_sincosTest_b_1_q(112 downto 42));

    -- signOfSelectionSignal_uid748_sincosTest(LOGICAL,747)@43
    signOfSelectionSignal_uid748_sincosTest_q <= not (redist51_xMSB_uid746_sincosTest_b_9_q);

    -- twoToMiSiXip_uid751_sincosTest(BITSELECT,750)@43
    twoToMiSiXip_uid751_sincosTest_b <= STD_LOGIC_VECTOR(redist53_xip1_41_uid743_sincosTest_b_1_q(112 downto 41));

    -- yip1E_42_uid756_sincosTest(ADDSUB,755)@43
    yip1E_42_uid756_sincosTest_s <= signOfSelectionSignal_uid748_sincosTest_q;
    yip1E_42_uid756_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist52_yip1_41_uid744_sincosTest_b_1_q(112)) & redist52_yip1_41_uid744_sincosTest_b_1_q));
    yip1E_42_uid756_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 72 => twoToMiSiXip_uid751_sincosTest_b(71)) & twoToMiSiXip_uid751_sincosTest_b));
    yip1E_42_uid756_sincosTest_combproc: PROCESS (yip1E_42_uid756_sincosTest_a, yip1E_42_uid756_sincosTest_b, yip1E_42_uid756_sincosTest_s)
    BEGIN
        IF (yip1E_42_uid756_sincosTest_s = "1") THEN
            yip1E_42_uid756_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_42_uid756_sincosTest_a) + SIGNED(yip1E_42_uid756_sincosTest_b));
        ELSE
            yip1E_42_uid756_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_42_uid756_sincosTest_a) - SIGNED(yip1E_42_uid756_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_42_uid756_sincosTest_q <= yip1E_42_uid756_sincosTest_o(113 downto 0);

    -- yip1_42_uid763_sincosTest(BITSELECT,762)@43
    yip1_42_uid763_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_42_uid756_sincosTest_q(112 downto 0));
    yip1_42_uid763_sincosTest_b <= STD_LOGIC_VECTOR(yip1_42_uid763_sincosTest_in(112 downto 0));

    -- redist49_yip1_42_uid763_sincosTest_b_1(DELAY,1086)
    redist49_yip1_42_uid763_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_42_uid763_sincosTest_b, xout => redist49_yip1_42_uid763_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_43_uid775_sincosTest(ADDSUB,774)@44
    yip1E_43_uid775_sincosTest_s <= signOfSelectionSignal_uid767_sincosTest_q;
    yip1E_43_uid775_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist49_yip1_42_uid763_sincosTest_b_1_q(112)) & redist49_yip1_42_uid763_sincosTest_b_1_q));
    yip1E_43_uid775_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 71 => twoToMiSiXip_uid770_sincosTest_b(70)) & twoToMiSiXip_uid770_sincosTest_b));
    yip1E_43_uid775_sincosTest_combproc: PROCESS (yip1E_43_uid775_sincosTest_a, yip1E_43_uid775_sincosTest_b, yip1E_43_uid775_sincosTest_s)
    BEGIN
        IF (yip1E_43_uid775_sincosTest_s = "1") THEN
            yip1E_43_uid775_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_43_uid775_sincosTest_a) + SIGNED(yip1E_43_uid775_sincosTest_b));
        ELSE
            yip1E_43_uid775_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_43_uid775_sincosTest_a) - SIGNED(yip1E_43_uid775_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_43_uid775_sincosTest_q <= yip1E_43_uid775_sincosTest_o(113 downto 0);

    -- yip1_43_uid782_sincosTest(BITSELECT,781)@44
    yip1_43_uid782_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_43_uid775_sincosTest_q(112 downto 0));
    yip1_43_uid782_sincosTest_b <= STD_LOGIC_VECTOR(yip1_43_uid782_sincosTest_in(112 downto 0));

    -- redist45_yip1_43_uid782_sincosTest_b_1(DELAY,1082)
    redist45_yip1_43_uid782_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_43_uid782_sincosTest_b, xout => redist45_yip1_43_uid782_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid790_sincosTest(BITSELECT,789)@45
    twoToMiSiYip_uid790_sincosTest_b <= STD_LOGIC_VECTOR(redist45_yip1_43_uid782_sincosTest_b_1_q(112 downto 43));

    -- twoToMiSiYip_uid771_sincosTest(BITSELECT,770)@44
    twoToMiSiYip_uid771_sincosTest_b <= STD_LOGIC_VECTOR(redist49_yip1_42_uid763_sincosTest_b_1_q(112 downto 42));

    -- xip1E_43_uid774_sincosTest(ADDSUB,773)@44
    xip1E_43_uid774_sincosTest_s <= redist47_xMSB_uid765_sincosTest_b_9_q;
    xip1E_43_uid774_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist50_xip1_42_uid762_sincosTest_b_1_q(112)) & redist50_xip1_42_uid762_sincosTest_b_1_q));
    xip1E_43_uid774_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 71 => twoToMiSiYip_uid771_sincosTest_b(70)) & twoToMiSiYip_uid771_sincosTest_b));
    xip1E_43_uid774_sincosTest_combproc: PROCESS (xip1E_43_uid774_sincosTest_a, xip1E_43_uid774_sincosTest_b, xip1E_43_uid774_sincosTest_s)
    BEGIN
        IF (xip1E_43_uid774_sincosTest_s = "1") THEN
            xip1E_43_uid774_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_43_uid774_sincosTest_a) + SIGNED(xip1E_43_uid774_sincosTest_b));
        ELSE
            xip1E_43_uid774_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_43_uid774_sincosTest_a) - SIGNED(xip1E_43_uid774_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_43_uid774_sincosTest_q <= xip1E_43_uid774_sincosTest_o(113 downto 0);

    -- xip1_43_uid781_sincosTest(BITSELECT,780)@44
    xip1_43_uid781_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_43_uid774_sincosTest_q(112 downto 0));
    xip1_43_uid781_sincosTest_b <= STD_LOGIC_VECTOR(xip1_43_uid781_sincosTest_in(112 downto 0));

    -- redist46_xip1_43_uid781_sincosTest_b_1(DELAY,1083)
    redist46_xip1_43_uid781_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_43_uid781_sincosTest_b, xout => redist46_xip1_43_uid781_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_44_uid793_sincosTest(ADDSUB,792)@45
    xip1E_44_uid793_sincosTest_s <= redist44_xMSB_uid784_sincosTest_b_10_q;
    xip1E_44_uid793_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist46_xip1_43_uid781_sincosTest_b_1_q(112)) & redist46_xip1_43_uid781_sincosTest_b_1_q));
    xip1E_44_uid793_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 70 => twoToMiSiYip_uid790_sincosTest_b(69)) & twoToMiSiYip_uid790_sincosTest_b));
    xip1E_44_uid793_sincosTest_combproc: PROCESS (xip1E_44_uid793_sincosTest_a, xip1E_44_uid793_sincosTest_b, xip1E_44_uid793_sincosTest_s)
    BEGIN
        IF (xip1E_44_uid793_sincosTest_s = "1") THEN
            xip1E_44_uid793_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_44_uid793_sincosTest_a) + SIGNED(xip1E_44_uid793_sincosTest_b));
        ELSE
            xip1E_44_uid793_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_44_uid793_sincosTest_a) - SIGNED(xip1E_44_uid793_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_44_uid793_sincosTest_q <= xip1E_44_uid793_sincosTest_o(113 downto 0);

    -- xip1_44_uid800_sincosTest(BITSELECT,799)@45
    xip1_44_uid800_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_44_uid793_sincosTest_q(112 downto 0));
    xip1_44_uid800_sincosTest_b <= STD_LOGIC_VECTOR(xip1_44_uid800_sincosTest_in(112 downto 0));

    -- redist43_xip1_44_uid800_sincosTest_b_1(DELAY,1080)
    redist43_xip1_44_uid800_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_44_uid800_sincosTest_b, xout => redist43_xip1_44_uid800_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid808_sincosTest(BITSELECT,807)@46
    twoToMiSiXip_uid808_sincosTest_b <= STD_LOGIC_VECTOR(redist43_xip1_44_uid800_sincosTest_b_1_q(112 downto 44));

    -- signOfSelectionSignal_uid786_sincosTest(LOGICAL,785)@45
    signOfSelectionSignal_uid786_sincosTest_q <= not (redist44_xMSB_uid784_sincosTest_b_10_q);

    -- twoToMiSiXip_uid789_sincosTest(BITSELECT,788)@45
    twoToMiSiXip_uid789_sincosTest_b <= STD_LOGIC_VECTOR(redist46_xip1_43_uid781_sincosTest_b_1_q(112 downto 43));

    -- yip1E_44_uid794_sincosTest(ADDSUB,793)@45
    yip1E_44_uid794_sincosTest_s <= signOfSelectionSignal_uid786_sincosTest_q;
    yip1E_44_uid794_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist45_yip1_43_uid782_sincosTest_b_1_q(112)) & redist45_yip1_43_uid782_sincosTest_b_1_q));
    yip1E_44_uid794_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 70 => twoToMiSiXip_uid789_sincosTest_b(69)) & twoToMiSiXip_uid789_sincosTest_b));
    yip1E_44_uid794_sincosTest_combproc: PROCESS (yip1E_44_uid794_sincosTest_a, yip1E_44_uid794_sincosTest_b, yip1E_44_uid794_sincosTest_s)
    BEGIN
        IF (yip1E_44_uid794_sincosTest_s = "1") THEN
            yip1E_44_uid794_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_44_uid794_sincosTest_a) + SIGNED(yip1E_44_uid794_sincosTest_b));
        ELSE
            yip1E_44_uid794_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_44_uid794_sincosTest_a) - SIGNED(yip1E_44_uid794_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_44_uid794_sincosTest_q <= yip1E_44_uid794_sincosTest_o(113 downto 0);

    -- yip1_44_uid801_sincosTest(BITSELECT,800)@45
    yip1_44_uid801_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_44_uid794_sincosTest_q(112 downto 0));
    yip1_44_uid801_sincosTest_b <= STD_LOGIC_VECTOR(yip1_44_uid801_sincosTest_in(112 downto 0));

    -- redist42_yip1_44_uid801_sincosTest_b_1(DELAY,1079)
    redist42_yip1_44_uid801_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_44_uid801_sincosTest_b, xout => redist42_yip1_44_uid801_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_45_uid813_sincosTest(ADDSUB,812)@46
    yip1E_45_uid813_sincosTest_s <= signOfSelectionSignal_uid805_sincosTest_q;
    yip1E_45_uid813_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist42_yip1_44_uid801_sincosTest_b_1_q(112)) & redist42_yip1_44_uid801_sincosTest_b_1_q));
    yip1E_45_uid813_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 69 => twoToMiSiXip_uid808_sincosTest_b(68)) & twoToMiSiXip_uid808_sincosTest_b));
    yip1E_45_uid813_sincosTest_combproc: PROCESS (yip1E_45_uid813_sincosTest_a, yip1E_45_uid813_sincosTest_b, yip1E_45_uid813_sincosTest_s)
    BEGIN
        IF (yip1E_45_uid813_sincosTest_s = "1") THEN
            yip1E_45_uid813_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_45_uid813_sincosTest_a) + SIGNED(yip1E_45_uid813_sincosTest_b));
        ELSE
            yip1E_45_uid813_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_45_uid813_sincosTest_a) - SIGNED(yip1E_45_uid813_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_45_uid813_sincosTest_q <= yip1E_45_uid813_sincosTest_o(113 downto 0);

    -- yip1_45_uid820_sincosTest(BITSELECT,819)@46
    yip1_45_uid820_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_45_uid813_sincosTest_q(112 downto 0));
    yip1_45_uid820_sincosTest_b <= STD_LOGIC_VECTOR(yip1_45_uid820_sincosTest_in(112 downto 0));

    -- redist38_yip1_45_uid820_sincosTest_b_1(DELAY,1075)
    redist38_yip1_45_uid820_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_45_uid820_sincosTest_b, xout => redist38_yip1_45_uid820_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid828_sincosTest(BITSELECT,827)@47
    twoToMiSiYip_uid828_sincosTest_b <= STD_LOGIC_VECTOR(redist38_yip1_45_uid820_sincosTest_b_1_q(112 downto 45));

    -- twoToMiSiYip_uid809_sincosTest(BITSELECT,808)@46
    twoToMiSiYip_uid809_sincosTest_b <= STD_LOGIC_VECTOR(redist42_yip1_44_uid801_sincosTest_b_1_q(112 downto 44));

    -- xip1E_45_uid812_sincosTest(ADDSUB,811)@46
    xip1E_45_uid812_sincosTest_s <= redist40_xMSB_uid803_sincosTest_b_10_q;
    xip1E_45_uid812_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist43_xip1_44_uid800_sincosTest_b_1_q(112)) & redist43_xip1_44_uid800_sincosTest_b_1_q));
    xip1E_45_uid812_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 69 => twoToMiSiYip_uid809_sincosTest_b(68)) & twoToMiSiYip_uid809_sincosTest_b));
    xip1E_45_uid812_sincosTest_combproc: PROCESS (xip1E_45_uid812_sincosTest_a, xip1E_45_uid812_sincosTest_b, xip1E_45_uid812_sincosTest_s)
    BEGIN
        IF (xip1E_45_uid812_sincosTest_s = "1") THEN
            xip1E_45_uid812_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_45_uid812_sincosTest_a) + SIGNED(xip1E_45_uid812_sincosTest_b));
        ELSE
            xip1E_45_uid812_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_45_uid812_sincosTest_a) - SIGNED(xip1E_45_uid812_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_45_uid812_sincosTest_q <= xip1E_45_uid812_sincosTest_o(113 downto 0);

    -- xip1_45_uid819_sincosTest(BITSELECT,818)@46
    xip1_45_uid819_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_45_uid812_sincosTest_q(112 downto 0));
    xip1_45_uid819_sincosTest_b <= STD_LOGIC_VECTOR(xip1_45_uid819_sincosTest_in(112 downto 0));

    -- redist39_xip1_45_uid819_sincosTest_b_1(DELAY,1076)
    redist39_xip1_45_uid819_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_45_uid819_sincosTest_b, xout => redist39_xip1_45_uid819_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_46_uid831_sincosTest(ADDSUB,830)@47
    xip1E_46_uid831_sincosTest_s <= redist37_xMSB_uid822_sincosTest_b_11_q;
    xip1E_46_uid831_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist39_xip1_45_uid819_sincosTest_b_1_q(112)) & redist39_xip1_45_uid819_sincosTest_b_1_q));
    xip1E_46_uid831_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 68 => twoToMiSiYip_uid828_sincosTest_b(67)) & twoToMiSiYip_uid828_sincosTest_b));
    xip1E_46_uid831_sincosTest_combproc: PROCESS (xip1E_46_uid831_sincosTest_a, xip1E_46_uid831_sincosTest_b, xip1E_46_uid831_sincosTest_s)
    BEGIN
        IF (xip1E_46_uid831_sincosTest_s = "1") THEN
            xip1E_46_uid831_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_46_uid831_sincosTest_a) + SIGNED(xip1E_46_uid831_sincosTest_b));
        ELSE
            xip1E_46_uid831_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_46_uid831_sincosTest_a) - SIGNED(xip1E_46_uid831_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_46_uid831_sincosTest_q <= xip1E_46_uid831_sincosTest_o(113 downto 0);

    -- xip1_46_uid838_sincosTest(BITSELECT,837)@47
    xip1_46_uid838_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_46_uid831_sincosTest_q(112 downto 0));
    xip1_46_uid838_sincosTest_b <= STD_LOGIC_VECTOR(xip1_46_uid838_sincosTest_in(112 downto 0));

    -- redist36_xip1_46_uid838_sincosTest_b_1(DELAY,1073)
    redist36_xip1_46_uid838_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_46_uid838_sincosTest_b, xout => redist36_xip1_46_uid838_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid846_sincosTest(BITSELECT,845)@48
    twoToMiSiXip_uid846_sincosTest_b <= STD_LOGIC_VECTOR(redist36_xip1_46_uid838_sincosTest_b_1_q(112 downto 46));

    -- signOfSelectionSignal_uid824_sincosTest(LOGICAL,823)@47
    signOfSelectionSignal_uid824_sincosTest_q <= not (redist37_xMSB_uid822_sincosTest_b_11_q);

    -- twoToMiSiXip_uid827_sincosTest(BITSELECT,826)@47
    twoToMiSiXip_uid827_sincosTest_b <= STD_LOGIC_VECTOR(redist39_xip1_45_uid819_sincosTest_b_1_q(112 downto 45));

    -- yip1E_46_uid832_sincosTest(ADDSUB,831)@47
    yip1E_46_uid832_sincosTest_s <= signOfSelectionSignal_uid824_sincosTest_q;
    yip1E_46_uid832_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist38_yip1_45_uid820_sincosTest_b_1_q(112)) & redist38_yip1_45_uid820_sincosTest_b_1_q));
    yip1E_46_uid832_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 68 => twoToMiSiXip_uid827_sincosTest_b(67)) & twoToMiSiXip_uid827_sincosTest_b));
    yip1E_46_uid832_sincosTest_combproc: PROCESS (yip1E_46_uid832_sincosTest_a, yip1E_46_uid832_sincosTest_b, yip1E_46_uid832_sincosTest_s)
    BEGIN
        IF (yip1E_46_uid832_sincosTest_s = "1") THEN
            yip1E_46_uid832_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_46_uid832_sincosTest_a) + SIGNED(yip1E_46_uid832_sincosTest_b));
        ELSE
            yip1E_46_uid832_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_46_uid832_sincosTest_a) - SIGNED(yip1E_46_uid832_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_46_uid832_sincosTest_q <= yip1E_46_uid832_sincosTest_o(113 downto 0);

    -- yip1_46_uid839_sincosTest(BITSELECT,838)@47
    yip1_46_uid839_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_46_uid832_sincosTest_q(112 downto 0));
    yip1_46_uid839_sincosTest_b <= STD_LOGIC_VECTOR(yip1_46_uid839_sincosTest_in(112 downto 0));

    -- redist35_yip1_46_uid839_sincosTest_b_1(DELAY,1072)
    redist35_yip1_46_uid839_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_46_uid839_sincosTest_b, xout => redist35_yip1_46_uid839_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_47_uid851_sincosTest(ADDSUB,850)@48
    yip1E_47_uid851_sincosTest_s <= signOfSelectionSignal_uid843_sincosTest_q;
    yip1E_47_uid851_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist35_yip1_46_uid839_sincosTest_b_1_q(112)) & redist35_yip1_46_uid839_sincosTest_b_1_q));
    yip1E_47_uid851_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 67 => twoToMiSiXip_uid846_sincosTest_b(66)) & twoToMiSiXip_uid846_sincosTest_b));
    yip1E_47_uid851_sincosTest_combproc: PROCESS (yip1E_47_uid851_sincosTest_a, yip1E_47_uid851_sincosTest_b, yip1E_47_uid851_sincosTest_s)
    BEGIN
        IF (yip1E_47_uid851_sincosTest_s = "1") THEN
            yip1E_47_uid851_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_47_uid851_sincosTest_a) + SIGNED(yip1E_47_uid851_sincosTest_b));
        ELSE
            yip1E_47_uid851_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_47_uid851_sincosTest_a) - SIGNED(yip1E_47_uid851_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_47_uid851_sincosTest_q <= yip1E_47_uid851_sincosTest_o(113 downto 0);

    -- yip1_47_uid858_sincosTest(BITSELECT,857)@48
    yip1_47_uid858_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_47_uid851_sincosTest_q(112 downto 0));
    yip1_47_uid858_sincosTest_b <= STD_LOGIC_VECTOR(yip1_47_uid858_sincosTest_in(112 downto 0));

    -- redist31_yip1_47_uid858_sincosTest_b_1(DELAY,1068)
    redist31_yip1_47_uid858_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_47_uid858_sincosTest_b, xout => redist31_yip1_47_uid858_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid866_sincosTest(BITSELECT,865)@49
    twoToMiSiYip_uid866_sincosTest_b <= STD_LOGIC_VECTOR(redist31_yip1_47_uid858_sincosTest_b_1_q(112 downto 47));

    -- twoToMiSiYip_uid847_sincosTest(BITSELECT,846)@48
    twoToMiSiYip_uid847_sincosTest_b <= STD_LOGIC_VECTOR(redist35_yip1_46_uid839_sincosTest_b_1_q(112 downto 46));

    -- xip1E_47_uid850_sincosTest(ADDSUB,849)@48
    xip1E_47_uid850_sincosTest_s <= redist33_xMSB_uid841_sincosTest_b_11_q;
    xip1E_47_uid850_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist36_xip1_46_uid838_sincosTest_b_1_q(112)) & redist36_xip1_46_uid838_sincosTest_b_1_q));
    xip1E_47_uid850_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 67 => twoToMiSiYip_uid847_sincosTest_b(66)) & twoToMiSiYip_uid847_sincosTest_b));
    xip1E_47_uid850_sincosTest_combproc: PROCESS (xip1E_47_uid850_sincosTest_a, xip1E_47_uid850_sincosTest_b, xip1E_47_uid850_sincosTest_s)
    BEGIN
        IF (xip1E_47_uid850_sincosTest_s = "1") THEN
            xip1E_47_uid850_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_47_uid850_sincosTest_a) + SIGNED(xip1E_47_uid850_sincosTest_b));
        ELSE
            xip1E_47_uid850_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_47_uid850_sincosTest_a) - SIGNED(xip1E_47_uid850_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_47_uid850_sincosTest_q <= xip1E_47_uid850_sincosTest_o(113 downto 0);

    -- xip1_47_uid857_sincosTest(BITSELECT,856)@48
    xip1_47_uid857_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_47_uid850_sincosTest_q(112 downto 0));
    xip1_47_uid857_sincosTest_b <= STD_LOGIC_VECTOR(xip1_47_uid857_sincosTest_in(112 downto 0));

    -- redist32_xip1_47_uid857_sincosTest_b_1(DELAY,1069)
    redist32_xip1_47_uid857_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_47_uid857_sincosTest_b, xout => redist32_xip1_47_uid857_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_48_uid869_sincosTest(ADDSUB,868)@49
    xip1E_48_uid869_sincosTest_s <= redist30_xMSB_uid860_sincosTest_b_12_q;
    xip1E_48_uid869_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist32_xip1_47_uid857_sincosTest_b_1_q(112)) & redist32_xip1_47_uid857_sincosTest_b_1_q));
    xip1E_48_uid869_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 66 => twoToMiSiYip_uid866_sincosTest_b(65)) & twoToMiSiYip_uid866_sincosTest_b));
    xip1E_48_uid869_sincosTest_combproc: PROCESS (xip1E_48_uid869_sincosTest_a, xip1E_48_uid869_sincosTest_b, xip1E_48_uid869_sincosTest_s)
    BEGIN
        IF (xip1E_48_uid869_sincosTest_s = "1") THEN
            xip1E_48_uid869_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_48_uid869_sincosTest_a) + SIGNED(xip1E_48_uid869_sincosTest_b));
        ELSE
            xip1E_48_uid869_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_48_uid869_sincosTest_a) - SIGNED(xip1E_48_uid869_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_48_uid869_sincosTest_q <= xip1E_48_uid869_sincosTest_o(113 downto 0);

    -- xip1_48_uid876_sincosTest(BITSELECT,875)@49
    xip1_48_uid876_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_48_uid869_sincosTest_q(112 downto 0));
    xip1_48_uid876_sincosTest_b <= STD_LOGIC_VECTOR(xip1_48_uid876_sincosTest_in(112 downto 0));

    -- redist29_xip1_48_uid876_sincosTest_b_1(DELAY,1066)
    redist29_xip1_48_uid876_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_48_uid876_sincosTest_b, xout => redist29_xip1_48_uid876_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid884_sincosTest(BITSELECT,883)@50
    twoToMiSiXip_uid884_sincosTest_b <= STD_LOGIC_VECTOR(redist29_xip1_48_uid876_sincosTest_b_1_q(112 downto 48));

    -- signOfSelectionSignal_uid862_sincosTest(LOGICAL,861)@49
    signOfSelectionSignal_uid862_sincosTest_q <= not (redist30_xMSB_uid860_sincosTest_b_12_q);

    -- twoToMiSiXip_uid865_sincosTest(BITSELECT,864)@49
    twoToMiSiXip_uid865_sincosTest_b <= STD_LOGIC_VECTOR(redist32_xip1_47_uid857_sincosTest_b_1_q(112 downto 47));

    -- yip1E_48_uid870_sincosTest(ADDSUB,869)@49
    yip1E_48_uid870_sincosTest_s <= signOfSelectionSignal_uid862_sincosTest_q;
    yip1E_48_uid870_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist31_yip1_47_uid858_sincosTest_b_1_q(112)) & redist31_yip1_47_uid858_sincosTest_b_1_q));
    yip1E_48_uid870_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 66 => twoToMiSiXip_uid865_sincosTest_b(65)) & twoToMiSiXip_uid865_sincosTest_b));
    yip1E_48_uid870_sincosTest_combproc: PROCESS (yip1E_48_uid870_sincosTest_a, yip1E_48_uid870_sincosTest_b, yip1E_48_uid870_sincosTest_s)
    BEGIN
        IF (yip1E_48_uid870_sincosTest_s = "1") THEN
            yip1E_48_uid870_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_48_uid870_sincosTest_a) + SIGNED(yip1E_48_uid870_sincosTest_b));
        ELSE
            yip1E_48_uid870_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_48_uid870_sincosTest_a) - SIGNED(yip1E_48_uid870_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_48_uid870_sincosTest_q <= yip1E_48_uid870_sincosTest_o(113 downto 0);

    -- yip1_48_uid877_sincosTest(BITSELECT,876)@49
    yip1_48_uid877_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_48_uid870_sincosTest_q(112 downto 0));
    yip1_48_uid877_sincosTest_b <= STD_LOGIC_VECTOR(yip1_48_uid877_sincosTest_in(112 downto 0));

    -- redist28_yip1_48_uid877_sincosTest_b_1(DELAY,1065)
    redist28_yip1_48_uid877_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_48_uid877_sincosTest_b, xout => redist28_yip1_48_uid877_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_49_uid889_sincosTest(ADDSUB,888)@50
    yip1E_49_uid889_sincosTest_s <= signOfSelectionSignal_uid881_sincosTest_q;
    yip1E_49_uid889_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist28_yip1_48_uid877_sincosTest_b_1_q(112)) & redist28_yip1_48_uid877_sincosTest_b_1_q));
    yip1E_49_uid889_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 65 => twoToMiSiXip_uid884_sincosTest_b(64)) & twoToMiSiXip_uid884_sincosTest_b));
    yip1E_49_uid889_sincosTest_combproc: PROCESS (yip1E_49_uid889_sincosTest_a, yip1E_49_uid889_sincosTest_b, yip1E_49_uid889_sincosTest_s)
    BEGIN
        IF (yip1E_49_uid889_sincosTest_s = "1") THEN
            yip1E_49_uid889_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_49_uid889_sincosTest_a) + SIGNED(yip1E_49_uid889_sincosTest_b));
        ELSE
            yip1E_49_uid889_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_49_uid889_sincosTest_a) - SIGNED(yip1E_49_uid889_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_49_uid889_sincosTest_q <= yip1E_49_uid889_sincosTest_o(113 downto 0);

    -- yip1_49_uid896_sincosTest(BITSELECT,895)@50
    yip1_49_uid896_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_49_uid889_sincosTest_q(112 downto 0));
    yip1_49_uid896_sincosTest_b <= STD_LOGIC_VECTOR(yip1_49_uid896_sincosTest_in(112 downto 0));

    -- redist24_yip1_49_uid896_sincosTest_b_1(DELAY,1061)
    redist24_yip1_49_uid896_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_49_uid896_sincosTest_b, xout => redist24_yip1_49_uid896_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid904_sincosTest(BITSELECT,903)@51
    twoToMiSiYip_uid904_sincosTest_b <= STD_LOGIC_VECTOR(redist24_yip1_49_uid896_sincosTest_b_1_q(112 downto 49));

    -- twoToMiSiYip_uid885_sincosTest(BITSELECT,884)@50
    twoToMiSiYip_uid885_sincosTest_b <= STD_LOGIC_VECTOR(redist28_yip1_48_uid877_sincosTest_b_1_q(112 downto 48));

    -- xip1E_49_uid888_sincosTest(ADDSUB,887)@50
    xip1E_49_uid888_sincosTest_s <= redist26_xMSB_uid879_sincosTest_b_12_q;
    xip1E_49_uid888_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist29_xip1_48_uid876_sincosTest_b_1_q(112)) & redist29_xip1_48_uid876_sincosTest_b_1_q));
    xip1E_49_uid888_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 65 => twoToMiSiYip_uid885_sincosTest_b(64)) & twoToMiSiYip_uid885_sincosTest_b));
    xip1E_49_uid888_sincosTest_combproc: PROCESS (xip1E_49_uid888_sincosTest_a, xip1E_49_uid888_sincosTest_b, xip1E_49_uid888_sincosTest_s)
    BEGIN
        IF (xip1E_49_uid888_sincosTest_s = "1") THEN
            xip1E_49_uid888_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_49_uid888_sincosTest_a) + SIGNED(xip1E_49_uid888_sincosTest_b));
        ELSE
            xip1E_49_uid888_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_49_uid888_sincosTest_a) - SIGNED(xip1E_49_uid888_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_49_uid888_sincosTest_q <= xip1E_49_uid888_sincosTest_o(113 downto 0);

    -- xip1_49_uid895_sincosTest(BITSELECT,894)@50
    xip1_49_uid895_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_49_uid888_sincosTest_q(112 downto 0));
    xip1_49_uid895_sincosTest_b <= STD_LOGIC_VECTOR(xip1_49_uid895_sincosTest_in(112 downto 0));

    -- redist25_xip1_49_uid895_sincosTest_b_1(DELAY,1062)
    redist25_xip1_49_uid895_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_49_uid895_sincosTest_b, xout => redist25_xip1_49_uid895_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_50_uid907_sincosTest(ADDSUB,906)@51
    xip1E_50_uid907_sincosTest_s <= redist23_xMSB_uid898_sincosTest_b_13_q;
    xip1E_50_uid907_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist25_xip1_49_uid895_sincosTest_b_1_q(112)) & redist25_xip1_49_uid895_sincosTest_b_1_q));
    xip1E_50_uid907_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 64 => twoToMiSiYip_uid904_sincosTest_b(63)) & twoToMiSiYip_uid904_sincosTest_b));
    xip1E_50_uid907_sincosTest_combproc: PROCESS (xip1E_50_uid907_sincosTest_a, xip1E_50_uid907_sincosTest_b, xip1E_50_uid907_sincosTest_s)
    BEGIN
        IF (xip1E_50_uid907_sincosTest_s = "1") THEN
            xip1E_50_uid907_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_50_uid907_sincosTest_a) + SIGNED(xip1E_50_uid907_sincosTest_b));
        ELSE
            xip1E_50_uid907_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_50_uid907_sincosTest_a) - SIGNED(xip1E_50_uid907_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_50_uid907_sincosTest_q <= xip1E_50_uid907_sincosTest_o(113 downto 0);

    -- xip1_50_uid914_sincosTest(BITSELECT,913)@51
    xip1_50_uid914_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_50_uid907_sincosTest_q(112 downto 0));
    xip1_50_uid914_sincosTest_b <= STD_LOGIC_VECTOR(xip1_50_uid914_sincosTest_in(112 downto 0));

    -- redist22_xip1_50_uid914_sincosTest_b_1(DELAY,1059)
    redist22_xip1_50_uid914_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_50_uid914_sincosTest_b, xout => redist22_xip1_50_uid914_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid922_sincosTest(BITSELECT,921)@52
    twoToMiSiXip_uid922_sincosTest_b <= STD_LOGIC_VECTOR(redist22_xip1_50_uid914_sincosTest_b_1_q(112 downto 50));

    -- signOfSelectionSignal_uid900_sincosTest(LOGICAL,899)@51
    signOfSelectionSignal_uid900_sincosTest_q <= not (redist23_xMSB_uid898_sincosTest_b_13_q);

    -- twoToMiSiXip_uid903_sincosTest(BITSELECT,902)@51
    twoToMiSiXip_uid903_sincosTest_b <= STD_LOGIC_VECTOR(redist25_xip1_49_uid895_sincosTest_b_1_q(112 downto 49));

    -- yip1E_50_uid908_sincosTest(ADDSUB,907)@51
    yip1E_50_uid908_sincosTest_s <= signOfSelectionSignal_uid900_sincosTest_q;
    yip1E_50_uid908_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist24_yip1_49_uid896_sincosTest_b_1_q(112)) & redist24_yip1_49_uid896_sincosTest_b_1_q));
    yip1E_50_uid908_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 64 => twoToMiSiXip_uid903_sincosTest_b(63)) & twoToMiSiXip_uid903_sincosTest_b));
    yip1E_50_uid908_sincosTest_combproc: PROCESS (yip1E_50_uid908_sincosTest_a, yip1E_50_uid908_sincosTest_b, yip1E_50_uid908_sincosTest_s)
    BEGIN
        IF (yip1E_50_uid908_sincosTest_s = "1") THEN
            yip1E_50_uid908_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_50_uid908_sincosTest_a) + SIGNED(yip1E_50_uid908_sincosTest_b));
        ELSE
            yip1E_50_uid908_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_50_uid908_sincosTest_a) - SIGNED(yip1E_50_uid908_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_50_uid908_sincosTest_q <= yip1E_50_uid908_sincosTest_o(113 downto 0);

    -- yip1_50_uid915_sincosTest(BITSELECT,914)@51
    yip1_50_uid915_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_50_uid908_sincosTest_q(112 downto 0));
    yip1_50_uid915_sincosTest_b <= STD_LOGIC_VECTOR(yip1_50_uid915_sincosTest_in(112 downto 0));

    -- redist21_yip1_50_uid915_sincosTest_b_1(DELAY,1058)
    redist21_yip1_50_uid915_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_50_uid915_sincosTest_b, xout => redist21_yip1_50_uid915_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_51_uid927_sincosTest(ADDSUB,926)@52
    yip1E_51_uid927_sincosTest_s <= signOfSelectionSignal_uid919_sincosTest_q;
    yip1E_51_uid927_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist21_yip1_50_uid915_sincosTest_b_1_q(112)) & redist21_yip1_50_uid915_sincosTest_b_1_q));
    yip1E_51_uid927_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 63 => twoToMiSiXip_uid922_sincosTest_b(62)) & twoToMiSiXip_uid922_sincosTest_b));
    yip1E_51_uid927_sincosTest_combproc: PROCESS (yip1E_51_uid927_sincosTest_a, yip1E_51_uid927_sincosTest_b, yip1E_51_uid927_sincosTest_s)
    BEGIN
        IF (yip1E_51_uid927_sincosTest_s = "1") THEN
            yip1E_51_uid927_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_51_uid927_sincosTest_a) + SIGNED(yip1E_51_uid927_sincosTest_b));
        ELSE
            yip1E_51_uid927_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_51_uid927_sincosTest_a) - SIGNED(yip1E_51_uid927_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_51_uid927_sincosTest_q <= yip1E_51_uid927_sincosTest_o(113 downto 0);

    -- yip1_51_uid934_sincosTest(BITSELECT,933)@52
    yip1_51_uid934_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_51_uid927_sincosTest_q(112 downto 0));
    yip1_51_uid934_sincosTest_b <= STD_LOGIC_VECTOR(yip1_51_uid934_sincosTest_in(112 downto 0));

    -- redist17_yip1_51_uid934_sincosTest_b_1(DELAY,1054)
    redist17_yip1_51_uid934_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_51_uid934_sincosTest_b, xout => redist17_yip1_51_uid934_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid942_sincosTest(BITSELECT,941)@53
    twoToMiSiYip_uid942_sincosTest_b <= STD_LOGIC_VECTOR(redist17_yip1_51_uid934_sincosTest_b_1_q(112 downto 51));

    -- twoToMiSiYip_uid923_sincosTest(BITSELECT,922)@52
    twoToMiSiYip_uid923_sincosTest_b <= STD_LOGIC_VECTOR(redist21_yip1_50_uid915_sincosTest_b_1_q(112 downto 50));

    -- xip1E_51_uid926_sincosTest(ADDSUB,925)@52
    xip1E_51_uid926_sincosTest_s <= redist19_xMSB_uid917_sincosTest_b_13_q;
    xip1E_51_uid926_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist22_xip1_50_uid914_sincosTest_b_1_q(112)) & redist22_xip1_50_uid914_sincosTest_b_1_q));
    xip1E_51_uid926_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 63 => twoToMiSiYip_uid923_sincosTest_b(62)) & twoToMiSiYip_uid923_sincosTest_b));
    xip1E_51_uid926_sincosTest_combproc: PROCESS (xip1E_51_uid926_sincosTest_a, xip1E_51_uid926_sincosTest_b, xip1E_51_uid926_sincosTest_s)
    BEGIN
        IF (xip1E_51_uid926_sincosTest_s = "1") THEN
            xip1E_51_uid926_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_51_uid926_sincosTest_a) + SIGNED(xip1E_51_uid926_sincosTest_b));
        ELSE
            xip1E_51_uid926_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_51_uid926_sincosTest_a) - SIGNED(xip1E_51_uid926_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_51_uid926_sincosTest_q <= xip1E_51_uid926_sincosTest_o(113 downto 0);

    -- xip1_51_uid933_sincosTest(BITSELECT,932)@52
    xip1_51_uid933_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_51_uid926_sincosTest_q(112 downto 0));
    xip1_51_uid933_sincosTest_b <= STD_LOGIC_VECTOR(xip1_51_uid933_sincosTest_in(112 downto 0));

    -- redist18_xip1_51_uid933_sincosTest_b_1(DELAY,1055)
    redist18_xip1_51_uid933_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_51_uid933_sincosTest_b, xout => redist18_xip1_51_uid933_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_52_uid945_sincosTest(ADDSUB,944)@53
    xip1E_52_uid945_sincosTest_s <= redist16_xMSB_uid936_sincosTest_b_14_q;
    xip1E_52_uid945_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist18_xip1_51_uid933_sincosTest_b_1_q(112)) & redist18_xip1_51_uid933_sincosTest_b_1_q));
    xip1E_52_uid945_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 62 => twoToMiSiYip_uid942_sincosTest_b(61)) & twoToMiSiYip_uid942_sincosTest_b));
    xip1E_52_uid945_sincosTest_combproc: PROCESS (xip1E_52_uid945_sincosTest_a, xip1E_52_uid945_sincosTest_b, xip1E_52_uid945_sincosTest_s)
    BEGIN
        IF (xip1E_52_uid945_sincosTest_s = "1") THEN
            xip1E_52_uid945_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_52_uid945_sincosTest_a) + SIGNED(xip1E_52_uid945_sincosTest_b));
        ELSE
            xip1E_52_uid945_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_52_uid945_sincosTest_a) - SIGNED(xip1E_52_uid945_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_52_uid945_sincosTest_q <= xip1E_52_uid945_sincosTest_o(113 downto 0);

    -- xip1_52_uid952_sincosTest(BITSELECT,951)@53
    xip1_52_uid952_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_52_uid945_sincosTest_q(112 downto 0));
    xip1_52_uid952_sincosTest_b <= STD_LOGIC_VECTOR(xip1_52_uid952_sincosTest_in(112 downto 0));

    -- redist15_xip1_52_uid952_sincosTest_b_1(DELAY,1052)
    redist15_xip1_52_uid952_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_52_uid952_sincosTest_b, xout => redist15_xip1_52_uid952_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid960_sincosTest(BITSELECT,959)@54
    twoToMiSiXip_uid960_sincosTest_b <= STD_LOGIC_VECTOR(redist15_xip1_52_uid952_sincosTest_b_1_q(112 downto 52));

    -- signOfSelectionSignal_uid938_sincosTest(LOGICAL,937)@53
    signOfSelectionSignal_uid938_sincosTest_q <= not (redist16_xMSB_uid936_sincosTest_b_14_q);

    -- twoToMiSiXip_uid941_sincosTest(BITSELECT,940)@53
    twoToMiSiXip_uid941_sincosTest_b <= STD_LOGIC_VECTOR(redist18_xip1_51_uid933_sincosTest_b_1_q(112 downto 51));

    -- yip1E_52_uid946_sincosTest(ADDSUB,945)@53
    yip1E_52_uid946_sincosTest_s <= signOfSelectionSignal_uid938_sincosTest_q;
    yip1E_52_uid946_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist17_yip1_51_uid934_sincosTest_b_1_q(112)) & redist17_yip1_51_uid934_sincosTest_b_1_q));
    yip1E_52_uid946_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 62 => twoToMiSiXip_uid941_sincosTest_b(61)) & twoToMiSiXip_uid941_sincosTest_b));
    yip1E_52_uid946_sincosTest_combproc: PROCESS (yip1E_52_uid946_sincosTest_a, yip1E_52_uid946_sincosTest_b, yip1E_52_uid946_sincosTest_s)
    BEGIN
        IF (yip1E_52_uid946_sincosTest_s = "1") THEN
            yip1E_52_uid946_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_52_uid946_sincosTest_a) + SIGNED(yip1E_52_uid946_sincosTest_b));
        ELSE
            yip1E_52_uid946_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_52_uid946_sincosTest_a) - SIGNED(yip1E_52_uid946_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_52_uid946_sincosTest_q <= yip1E_52_uid946_sincosTest_o(113 downto 0);

    -- yip1_52_uid953_sincosTest(BITSELECT,952)@53
    yip1_52_uid953_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_52_uid946_sincosTest_q(112 downto 0));
    yip1_52_uid953_sincosTest_b <= STD_LOGIC_VECTOR(yip1_52_uid953_sincosTest_in(112 downto 0));

    -- redist14_yip1_52_uid953_sincosTest_b_1(DELAY,1051)
    redist14_yip1_52_uid953_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_52_uid953_sincosTest_b, xout => redist14_yip1_52_uid953_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_53_uid965_sincosTest(ADDSUB,964)@54
    yip1E_53_uid965_sincosTest_s <= signOfSelectionSignal_uid957_sincosTest_q;
    yip1E_53_uid965_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist14_yip1_52_uid953_sincosTest_b_1_q(112)) & redist14_yip1_52_uid953_sincosTest_b_1_q));
    yip1E_53_uid965_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 61 => twoToMiSiXip_uid960_sincosTest_b(60)) & twoToMiSiXip_uid960_sincosTest_b));
    yip1E_53_uid965_sincosTest_combproc: PROCESS (yip1E_53_uid965_sincosTest_a, yip1E_53_uid965_sincosTest_b, yip1E_53_uid965_sincosTest_s)
    BEGIN
        IF (yip1E_53_uid965_sincosTest_s = "1") THEN
            yip1E_53_uid965_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_53_uid965_sincosTest_a) + SIGNED(yip1E_53_uid965_sincosTest_b));
        ELSE
            yip1E_53_uid965_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_53_uid965_sincosTest_a) - SIGNED(yip1E_53_uid965_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_53_uid965_sincosTest_q <= yip1E_53_uid965_sincosTest_o(113 downto 0);

    -- yip1_53_uid972_sincosTest(BITSELECT,971)@54
    yip1_53_uid972_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_53_uid965_sincosTest_q(112 downto 0));
    yip1_53_uid972_sincosTest_b <= STD_LOGIC_VECTOR(yip1_53_uid972_sincosTest_in(112 downto 0));

    -- redist10_yip1_53_uid972_sincosTest_b_1(DELAY,1047)
    redist10_yip1_53_uid972_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_53_uid972_sincosTest_b, xout => redist10_yip1_53_uid972_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid980_sincosTest(BITSELECT,979)@55
    twoToMiSiYip_uid980_sincosTest_b <= STD_LOGIC_VECTOR(redist10_yip1_53_uid972_sincosTest_b_1_q(112 downto 53));

    -- twoToMiSiYip_uid961_sincosTest(BITSELECT,960)@54
    twoToMiSiYip_uid961_sincosTest_b <= STD_LOGIC_VECTOR(redist14_yip1_52_uid953_sincosTest_b_1_q(112 downto 52));

    -- xip1E_53_uid964_sincosTest(ADDSUB,963)@54
    xip1E_53_uid964_sincosTest_s <= redist12_xMSB_uid955_sincosTest_b_14_q;
    xip1E_53_uid964_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist15_xip1_52_uid952_sincosTest_b_1_q(112)) & redist15_xip1_52_uid952_sincosTest_b_1_q));
    xip1E_53_uid964_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 61 => twoToMiSiYip_uid961_sincosTest_b(60)) & twoToMiSiYip_uid961_sincosTest_b));
    xip1E_53_uid964_sincosTest_combproc: PROCESS (xip1E_53_uid964_sincosTest_a, xip1E_53_uid964_sincosTest_b, xip1E_53_uid964_sincosTest_s)
    BEGIN
        IF (xip1E_53_uid964_sincosTest_s = "1") THEN
            xip1E_53_uid964_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_53_uid964_sincosTest_a) + SIGNED(xip1E_53_uid964_sincosTest_b));
        ELSE
            xip1E_53_uid964_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_53_uid964_sincosTest_a) - SIGNED(xip1E_53_uid964_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_53_uid964_sincosTest_q <= xip1E_53_uid964_sincosTest_o(113 downto 0);

    -- xip1_53_uid971_sincosTest(BITSELECT,970)@54
    xip1_53_uid971_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_53_uid964_sincosTest_q(112 downto 0));
    xip1_53_uid971_sincosTest_b <= STD_LOGIC_VECTOR(xip1_53_uid971_sincosTest_in(112 downto 0));

    -- redist11_xip1_53_uid971_sincosTest_b_1(DELAY,1048)
    redist11_xip1_53_uid971_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_53_uid971_sincosTest_b, xout => redist11_xip1_53_uid971_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xip1E_54_uid983_sincosTest(ADDSUB,982)@55
    xip1E_54_uid983_sincosTest_s <= redist9_xMSB_uid974_sincosTest_b_15_q;
    xip1E_54_uid983_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist11_xip1_53_uid971_sincosTest_b_1_q(112)) & redist11_xip1_53_uid971_sincosTest_b_1_q));
    xip1E_54_uid983_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 60 => twoToMiSiYip_uid980_sincosTest_b(59)) & twoToMiSiYip_uid980_sincosTest_b));
    xip1E_54_uid983_sincosTest_combproc: PROCESS (xip1E_54_uid983_sincosTest_a, xip1E_54_uid983_sincosTest_b, xip1E_54_uid983_sincosTest_s)
    BEGIN
        IF (xip1E_54_uid983_sincosTest_s = "1") THEN
            xip1E_54_uid983_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_54_uid983_sincosTest_a) + SIGNED(xip1E_54_uid983_sincosTest_b));
        ELSE
            xip1E_54_uid983_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_54_uid983_sincosTest_a) - SIGNED(xip1E_54_uid983_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_54_uid983_sincosTest_q <= xip1E_54_uid983_sincosTest_o(113 downto 0);

    -- xip1_54_uid990_sincosTest(BITSELECT,989)@55
    xip1_54_uid990_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_54_uid983_sincosTest_q(112 downto 0));
    xip1_54_uid990_sincosTest_b <= STD_LOGIC_VECTOR(xip1_54_uid990_sincosTest_in(112 downto 0));

    -- redist8_xip1_54_uid990_sincosTest_b_1(DELAY,1045)
    redist8_xip1_54_uid990_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xip1_54_uid990_sincosTest_b, xout => redist8_xip1_54_uid990_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- twoToMiSiXip_uid998_sincosTest(BITSELECT,997)@56
    twoToMiSiXip_uid998_sincosTest_b <= STD_LOGIC_VECTOR(redist8_xip1_54_uid990_sincosTest_b_1_q(112 downto 54));

    -- signOfSelectionSignal_uid976_sincosTest(LOGICAL,975)@55
    signOfSelectionSignal_uid976_sincosTest_q <= not (redist9_xMSB_uid974_sincosTest_b_15_q);

    -- twoToMiSiXip_uid979_sincosTest(BITSELECT,978)@55
    twoToMiSiXip_uid979_sincosTest_b <= STD_LOGIC_VECTOR(redist11_xip1_53_uid971_sincosTest_b_1_q(112 downto 53));

    -- yip1E_54_uid984_sincosTest(ADDSUB,983)@55
    yip1E_54_uid984_sincosTest_s <= signOfSelectionSignal_uid976_sincosTest_q;
    yip1E_54_uid984_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist10_yip1_53_uid972_sincosTest_b_1_q(112)) & redist10_yip1_53_uid972_sincosTest_b_1_q));
    yip1E_54_uid984_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 60 => twoToMiSiXip_uid979_sincosTest_b(59)) & twoToMiSiXip_uid979_sincosTest_b));
    yip1E_54_uid984_sincosTest_combproc: PROCESS (yip1E_54_uid984_sincosTest_a, yip1E_54_uid984_sincosTest_b, yip1E_54_uid984_sincosTest_s)
    BEGIN
        IF (yip1E_54_uid984_sincosTest_s = "1") THEN
            yip1E_54_uid984_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_54_uid984_sincosTest_a) + SIGNED(yip1E_54_uid984_sincosTest_b));
        ELSE
            yip1E_54_uid984_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_54_uid984_sincosTest_a) - SIGNED(yip1E_54_uid984_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_54_uid984_sincosTest_q <= yip1E_54_uid984_sincosTest_o(113 downto 0);

    -- yip1_54_uid991_sincosTest(BITSELECT,990)@55
    yip1_54_uid991_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_54_uid984_sincosTest_q(112 downto 0));
    yip1_54_uid991_sincosTest_b <= STD_LOGIC_VECTOR(yip1_54_uid991_sincosTest_in(112 downto 0));

    -- redist7_yip1_54_uid991_sincosTest_b_1(DELAY,1044)
    redist7_yip1_54_uid991_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 113, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yip1_54_uid991_sincosTest_b, xout => redist7_yip1_54_uid991_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- yip1E_55_uid1003_sincosTest(ADDSUB,1002)@56
    yip1E_55_uid1003_sincosTest_s <= signOfSelectionSignal_uid995_sincosTest_q;
    yip1E_55_uid1003_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist7_yip1_54_uid991_sincosTest_b_1_q(112)) & redist7_yip1_54_uid991_sincosTest_b_1_q));
    yip1E_55_uid1003_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 59 => twoToMiSiXip_uid998_sincosTest_b(58)) & twoToMiSiXip_uid998_sincosTest_b));
    yip1E_55_uid1003_sincosTest_combproc: PROCESS (yip1E_55_uid1003_sincosTest_a, yip1E_55_uid1003_sincosTest_b, yip1E_55_uid1003_sincosTest_s)
    BEGIN
        IF (yip1E_55_uid1003_sincosTest_s = "1") THEN
            yip1E_55_uid1003_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_55_uid1003_sincosTest_a) + SIGNED(yip1E_55_uid1003_sincosTest_b));
        ELSE
            yip1E_55_uid1003_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(yip1E_55_uid1003_sincosTest_a) - SIGNED(yip1E_55_uid1003_sincosTest_b));
        END IF;
    END PROCESS;
    yip1E_55_uid1003_sincosTest_q <= yip1E_55_uid1003_sincosTest_o(113 downto 0);

    -- yip1_55_uid1010_sincosTest(BITSELECT,1009)@56
    yip1_55_uid1010_sincosTest_in <= STD_LOGIC_VECTOR(yip1E_55_uid1003_sincosTest_q(112 downto 0));
    yip1_55_uid1010_sincosTest_b <= STD_LOGIC_VECTOR(yip1_55_uid1010_sincosTest_in(112 downto 0));

    -- ySumPreRnd_uid1016_sincosTest(BITSELECT,1015)@56
    ySumPreRnd_uid1016_sincosTest_in <= yip1_55_uid1010_sincosTest_b(111 downto 0);
    ySumPreRnd_uid1016_sincosTest_b <= ySumPreRnd_uid1016_sincosTest_in(111 downto 56);

    -- redist4_ySumPreRnd_uid1016_sincosTest_b_1(DELAY,1041)
    redist4_ySumPreRnd_uid1016_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 56, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => ySumPreRnd_uid1016_sincosTest_b, xout => redist4_ySumPreRnd_uid1016_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- ySumPostRnd_uid1019_sincosTest(ADD,1018)@57
    ySumPostRnd_uid1019_sincosTest_a <= STD_LOGIC_VECTOR("0" & redist4_ySumPreRnd_uid1016_sincosTest_b_1_q);
    ySumPostRnd_uid1019_sincosTest_b <= STD_LOGIC_VECTOR("00000000000000000000000000000000000000000000000000000000" & VCC_q);
    ySumPostRnd_uid1019_sincosTest_o <= STD_LOGIC_VECTOR(UNSIGNED(ySumPostRnd_uid1019_sincosTest_a) + UNSIGNED(ySumPostRnd_uid1019_sincosTest_b));
    ySumPostRnd_uid1019_sincosTest_q <= ySumPostRnd_uid1019_sincosTest_o(56 downto 0);

    -- yPostExc_uid1021_sincosTest(BITSELECT,1020)@57
    yPostExc_uid1021_sincosTest_in <= STD_LOGIC_VECTOR(ySumPostRnd_uid1019_sincosTest_q(55 downto 0));
    yPostExc_uid1021_sincosTest_b <= STD_LOGIC_VECTOR(yPostExc_uid1021_sincosTest_in(55 downto 1));

    -- redist2_yPostExc_uid1021_sincosTest_b_1(DELAY,1039)
    redist2_yPostExc_uid1021_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 55, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yPostExc_uid1021_sincosTest_b, xout => redist2_yPostExc_uid1021_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- cstZeroForAddSub_uid1029_sincosTest(CONSTANT,1028)
    cstZeroForAddSub_uid1029_sincosTest_q <= "0000000000000000000000000000000000000000000000000000000";

    -- sinPostNeg_uid1031_sincosTest(ADDSUB,1030)@58
    sinPostNeg_uid1031_sincosTest_s <= invSinNegCond_uid1030_sincosTest_q;
    sinPostNeg_uid1031_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((56 downto 55 => cstZeroForAddSub_uid1029_sincosTest_q(54)) & cstZeroForAddSub_uid1029_sincosTest_q));
    sinPostNeg_uid1031_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((56 downto 55 => redist2_yPostExc_uid1021_sincosTest_b_1_q(54)) & redist2_yPostExc_uid1021_sincosTest_b_1_q));
    sinPostNeg_uid1031_sincosTest_combproc: PROCESS (sinPostNeg_uid1031_sincosTest_a, sinPostNeg_uid1031_sincosTest_b, sinPostNeg_uid1031_sincosTest_s)
    BEGIN
        IF (sinPostNeg_uid1031_sincosTest_s = "1") THEN
            sinPostNeg_uid1031_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(sinPostNeg_uid1031_sincosTest_a) + SIGNED(sinPostNeg_uid1031_sincosTest_b));
        ELSE
            sinPostNeg_uid1031_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(sinPostNeg_uid1031_sincosTest_a) - SIGNED(sinPostNeg_uid1031_sincosTest_b));
        END IF;
    END PROCESS;
    sinPostNeg_uid1031_sincosTest_q <= sinPostNeg_uid1031_sincosTest_o(55 downto 0);

    -- redist0_invCosNegCond_uid1032_sincosTest_q_57_notEnable(LOGICAL,1228)
    redist0_invCosNegCond_uid1032_sincosTest_q_57_notEnable_q <= STD_LOGIC_VECTOR(not (VCC_q));

    -- redist0_invCosNegCond_uid1032_sincosTest_q_57_nor(LOGICAL,1229)
    redist0_invCosNegCond_uid1032_sincosTest_q_57_nor_q <= not (redist0_invCosNegCond_uid1032_sincosTest_q_57_notEnable_q or redist0_invCosNegCond_uid1032_sincosTest_q_57_sticky_ena_q);

    -- redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_last(CONSTANT,1225)
    redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_last_q <= "0110100";

    -- redist0_invCosNegCond_uid1032_sincosTest_q_57_cmp(LOGICAL,1226)
    redist0_invCosNegCond_uid1032_sincosTest_q_57_cmp_b <= STD_LOGIC_VECTOR("0" & redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_q);
    redist0_invCosNegCond_uid1032_sincosTest_q_57_cmp_q <= "1" WHEN redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_last_q = redist0_invCosNegCond_uid1032_sincosTest_q_57_cmp_b ELSE "0";

    -- redist0_invCosNegCond_uid1032_sincosTest_q_57_cmpReg(REG,1227)
    redist0_invCosNegCond_uid1032_sincosTest_q_57_cmpReg_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            redist0_invCosNegCond_uid1032_sincosTest_q_57_cmpReg_q <= "0";
        ELSIF (clk'EVENT AND clk = '1') THEN
            redist0_invCosNegCond_uid1032_sincosTest_q_57_cmpReg_q <= STD_LOGIC_VECTOR(redist0_invCosNegCond_uid1032_sincosTest_q_57_cmp_q);
        END IF;
    END PROCESS;

    -- redist0_invCosNegCond_uid1032_sincosTest_q_57_sticky_ena(REG,1230)
    redist0_invCosNegCond_uid1032_sincosTest_q_57_sticky_ena_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            redist0_invCosNegCond_uid1032_sincosTest_q_57_sticky_ena_q <= "0";
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (redist0_invCosNegCond_uid1032_sincosTest_q_57_nor_q = "1") THEN
                redist0_invCosNegCond_uid1032_sincosTest_q_57_sticky_ena_q <= STD_LOGIC_VECTOR(redist0_invCosNegCond_uid1032_sincosTest_q_57_cmpReg_q);
            END IF;
        END IF;
    END PROCESS;

    -- redist0_invCosNegCond_uid1032_sincosTest_q_57_enaAnd(LOGICAL,1231)
    redist0_invCosNegCond_uid1032_sincosTest_q_57_enaAnd_q <= redist0_invCosNegCond_uid1032_sincosTest_q_57_sticky_ena_q and VCC_q;

    -- redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt(COUNTER,1223)
    -- low=0, high=53, step=1, init=0
    redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_i <= TO_UNSIGNED(0, 6);
            redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_eq <= '0';
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_i = TO_UNSIGNED(52, 6)) THEN
                redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_eq <= '1';
            ELSE
                redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_eq <= '0';
            END IF;
            IF (redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_eq = '1') THEN
                redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_i <= redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_i + 11;
            ELSE
                redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_i <= redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_i + 1;
            END IF;
        END IF;
    END PROCESS;
    redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_q <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR(RESIZE(redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_i, 6)));

    -- invCosNegCond_uid1032_sincosTest(LOGICAL,1031)@1 + 1
    invCosNegCond_uid1032_sincosTest_qi <= not (sinNegCond2_uid1023_sincosTest_q);
    invCosNegCond_uid1032_sincosTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => invCosNegCond_uid1032_sincosTest_qi, xout => invCosNegCond_uid1032_sincosTest_q, clk => clk, aclr => areset );

    -- redist0_invCosNegCond_uid1032_sincosTest_q_57_wraddr(REG,1224)
    redist0_invCosNegCond_uid1032_sincosTest_q_57_wraddr_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            redist0_invCosNegCond_uid1032_sincosTest_q_57_wraddr_q <= "110101";
        ELSIF (clk'EVENT AND clk = '1') THEN
            redist0_invCosNegCond_uid1032_sincosTest_q_57_wraddr_q <= STD_LOGIC_VECTOR(redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_q);
        END IF;
    END PROCESS;

    -- redist0_invCosNegCond_uid1032_sincosTest_q_57_mem(DUALMEM,1222)
    redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_ia <= STD_LOGIC_VECTOR(invCosNegCond_uid1032_sincosTest_q);
    redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_aa <= redist0_invCosNegCond_uid1032_sincosTest_q_57_wraddr_q;
    redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_ab <= redist0_invCosNegCond_uid1032_sincosTest_q_57_rdcnt_q;
    redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_reset0 <= areset;
    redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_dmem : altera_syncram
    GENERIC MAP (
        ram_block_type => "MLAB",
        operation_mode => "DUAL_PORT",
        width_a => 1,
        widthad_a => 6,
        numwords_a => 54,
        width_b => 1,
        widthad_b => 6,
        numwords_b => 54,
        lpm_type => "altera_syncram",
        width_byteena_a => 1,
        address_reg_b => "CLOCK0",
        indata_reg_b => "CLOCK0",
        rdcontrol_reg_b => "CLOCK0",
        byteena_reg_b => "CLOCK0",
        outdata_reg_b => "CLOCK1",
        outdata_aclr_b => "CLEAR1",
        clock_enable_input_a => "NORMAL",
        clock_enable_input_b => "NORMAL",
        clock_enable_output_b => "NORMAL",
        read_during_write_mode_mixed_ports => "DONT_CARE",
        power_up_uninitialized => "TRUE",
        intended_device_family => "Stratix V"
    )
    PORT MAP (
        clocken1 => redist0_invCosNegCond_uid1032_sincosTest_q_57_enaAnd_q(0),
        clocken0 => VCC_q(0),
        clock0 => clk,
        aclr1 => redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_reset0,
        clock1 => clk,
        address_a => redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_aa,
        data_a => redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_ia,
        wren_a => VCC_q(0),
        address_b => redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_ab,
        q_b => redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_iq
    );
    redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_q <= redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_iq(0 downto 0);

    -- redist0_invCosNegCond_uid1032_sincosTest_q_57_outputreg(DELAY,1221)
    redist0_invCosNegCond_uid1032_sincosTest_q_57_outputreg : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist0_invCosNegCond_uid1032_sincosTest_q_57_mem_q, xout => redist0_invCosNegCond_uid1032_sincosTest_q_57_outputreg_q, clk => clk, aclr => areset );

    -- twoToMiSiYip_uid999_sincosTest(BITSELECT,998)@56
    twoToMiSiYip_uid999_sincosTest_b <= STD_LOGIC_VECTOR(redist7_yip1_54_uid991_sincosTest_b_1_q(112 downto 54));

    -- xip1E_55_uid1002_sincosTest(ADDSUB,1001)@56
    xip1E_55_uid1002_sincosTest_s <= redist6_xMSB_uid993_sincosTest_b_16_q;
    xip1E_55_uid1002_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 113 => redist8_xip1_54_uid990_sincosTest_b_1_q(112)) & redist8_xip1_54_uid990_sincosTest_b_1_q));
    xip1E_55_uid1002_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((114 downto 59 => twoToMiSiYip_uid999_sincosTest_b(58)) & twoToMiSiYip_uid999_sincosTest_b));
    xip1E_55_uid1002_sincosTest_combproc: PROCESS (xip1E_55_uid1002_sincosTest_a, xip1E_55_uid1002_sincosTest_b, xip1E_55_uid1002_sincosTest_s)
    BEGIN
        IF (xip1E_55_uid1002_sincosTest_s = "1") THEN
            xip1E_55_uid1002_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_55_uid1002_sincosTest_a) + SIGNED(xip1E_55_uid1002_sincosTest_b));
        ELSE
            xip1E_55_uid1002_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(xip1E_55_uid1002_sincosTest_a) - SIGNED(xip1E_55_uid1002_sincosTest_b));
        END IF;
    END PROCESS;
    xip1E_55_uid1002_sincosTest_q <= xip1E_55_uid1002_sincosTest_o(113 downto 0);

    -- xip1_55_uid1009_sincosTest(BITSELECT,1008)@56
    xip1_55_uid1009_sincosTest_in <= STD_LOGIC_VECTOR(xip1E_55_uid1002_sincosTest_q(112 downto 0));
    xip1_55_uid1009_sincosTest_b <= STD_LOGIC_VECTOR(xip1_55_uid1009_sincosTest_in(112 downto 0));

    -- xSumPreRnd_uid1012_sincosTest(BITSELECT,1011)@56
    xSumPreRnd_uid1012_sincosTest_in <= xip1_55_uid1009_sincosTest_b(111 downto 0);
    xSumPreRnd_uid1012_sincosTest_b <= xSumPreRnd_uid1012_sincosTest_in(111 downto 56);

    -- redist5_xSumPreRnd_uid1012_sincosTest_b_1(DELAY,1042)
    redist5_xSumPreRnd_uid1012_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 56, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xSumPreRnd_uid1012_sincosTest_b, xout => redist5_xSumPreRnd_uid1012_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- xSumPostRnd_uid1015_sincosTest(ADD,1014)@57
    xSumPostRnd_uid1015_sincosTest_a <= STD_LOGIC_VECTOR("0" & redist5_xSumPreRnd_uid1012_sincosTest_b_1_q);
    xSumPostRnd_uid1015_sincosTest_b <= STD_LOGIC_VECTOR("00000000000000000000000000000000000000000000000000000000" & VCC_q);
    xSumPostRnd_uid1015_sincosTest_o <= STD_LOGIC_VECTOR(UNSIGNED(xSumPostRnd_uid1015_sincosTest_a) + UNSIGNED(xSumPostRnd_uid1015_sincosTest_b));
    xSumPostRnd_uid1015_sincosTest_q <= xSumPostRnd_uid1015_sincosTest_o(56 downto 0);

    -- xPostExc_uid1020_sincosTest(BITSELECT,1019)@57
    xPostExc_uid1020_sincosTest_in <= STD_LOGIC_VECTOR(xSumPostRnd_uid1015_sincosTest_q(55 downto 0));
    xPostExc_uid1020_sincosTest_b <= STD_LOGIC_VECTOR(xPostExc_uid1020_sincosTest_in(55 downto 1));

    -- redist3_xPostExc_uid1020_sincosTest_b_1(DELAY,1040)
    redist3_xPostExc_uid1020_sincosTest_b_1 : dspba_delay
    GENERIC MAP ( width => 55, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xPostExc_uid1020_sincosTest_b, xout => redist3_xPostExc_uid1020_sincosTest_b_1_q, clk => clk, aclr => areset );

    -- cosPostNeg_uid1033_sincosTest(ADDSUB,1032)@58
    cosPostNeg_uid1033_sincosTest_s <= redist0_invCosNegCond_uid1032_sincosTest_q_57_outputreg_q;
    cosPostNeg_uid1033_sincosTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((56 downto 55 => cstZeroForAddSub_uid1029_sincosTest_q(54)) & cstZeroForAddSub_uid1029_sincosTest_q));
    cosPostNeg_uid1033_sincosTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((56 downto 55 => redist3_xPostExc_uid1020_sincosTest_b_1_q(54)) & redist3_xPostExc_uid1020_sincosTest_b_1_q));
    cosPostNeg_uid1033_sincosTest_combproc: PROCESS (cosPostNeg_uid1033_sincosTest_a, cosPostNeg_uid1033_sincosTest_b, cosPostNeg_uid1033_sincosTest_s)
    BEGIN
        IF (cosPostNeg_uid1033_sincosTest_s = "1") THEN
            cosPostNeg_uid1033_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(cosPostNeg_uid1033_sincosTest_a) + SIGNED(cosPostNeg_uid1033_sincosTest_b));
        ELSE
            cosPostNeg_uid1033_sincosTest_o <= STD_LOGIC_VECTOR(SIGNED(cosPostNeg_uid1033_sincosTest_a) - SIGNED(cosPostNeg_uid1033_sincosTest_b));
        END IF;
    END PROCESS;
    cosPostNeg_uid1033_sincosTest_q <= cosPostNeg_uid1033_sincosTest_o(55 downto 0);

    -- redist180_firstQuadrant_uid15_sincosTest_b_57_notEnable(LOGICAL,1250)
    redist180_firstQuadrant_uid15_sincosTest_b_57_notEnable_q <= STD_LOGIC_VECTOR(not (VCC_q));

    -- redist180_firstQuadrant_uid15_sincosTest_b_57_nor(LOGICAL,1251)
    redist180_firstQuadrant_uid15_sincosTest_b_57_nor_q <= not (redist180_firstQuadrant_uid15_sincosTest_b_57_notEnable_q or redist180_firstQuadrant_uid15_sincosTest_b_57_sticky_ena_q);

    -- redist180_firstQuadrant_uid15_sincosTest_b_57_mem_last(CONSTANT,1247)
    redist180_firstQuadrant_uid15_sincosTest_b_57_mem_last_q <= "0110101";

    -- redist180_firstQuadrant_uid15_sincosTest_b_57_cmp(LOGICAL,1248)
    redist180_firstQuadrant_uid15_sincosTest_b_57_cmp_b <= STD_LOGIC_VECTOR("0" & redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_q);
    redist180_firstQuadrant_uid15_sincosTest_b_57_cmp_q <= "1" WHEN redist180_firstQuadrant_uid15_sincosTest_b_57_mem_last_q = redist180_firstQuadrant_uid15_sincosTest_b_57_cmp_b ELSE "0";

    -- redist180_firstQuadrant_uid15_sincosTest_b_57_cmpReg(REG,1249)
    redist180_firstQuadrant_uid15_sincosTest_b_57_cmpReg_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            redist180_firstQuadrant_uid15_sincosTest_b_57_cmpReg_q <= "0";
        ELSIF (clk'EVENT AND clk = '1') THEN
            redist180_firstQuadrant_uid15_sincosTest_b_57_cmpReg_q <= STD_LOGIC_VECTOR(redist180_firstQuadrant_uid15_sincosTest_b_57_cmp_q);
        END IF;
    END PROCESS;

    -- redist180_firstQuadrant_uid15_sincosTest_b_57_sticky_ena(REG,1252)
    redist180_firstQuadrant_uid15_sincosTest_b_57_sticky_ena_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            redist180_firstQuadrant_uid15_sincosTest_b_57_sticky_ena_q <= "0";
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (redist180_firstQuadrant_uid15_sincosTest_b_57_nor_q = "1") THEN
                redist180_firstQuadrant_uid15_sincosTest_b_57_sticky_ena_q <= STD_LOGIC_VECTOR(redist180_firstQuadrant_uid15_sincosTest_b_57_cmpReg_q);
            END IF;
        END IF;
    END PROCESS;

    -- redist180_firstQuadrant_uid15_sincosTest_b_57_enaAnd(LOGICAL,1253)
    redist180_firstQuadrant_uid15_sincosTest_b_57_enaAnd_q <= redist180_firstQuadrant_uid15_sincosTest_b_57_sticky_ena_q and VCC_q;

    -- redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt(COUNTER,1245)
    -- low=0, high=54, step=1, init=0
    redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_i <= TO_UNSIGNED(0, 6);
            redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_eq <= '0';
        ELSIF (clk'EVENT AND clk = '1') THEN
            IF (redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_i = TO_UNSIGNED(53, 6)) THEN
                redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_eq <= '1';
            ELSE
                redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_eq <= '0';
            END IF;
            IF (redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_eq = '1') THEN
                redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_i <= redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_i + 10;
            ELSE
                redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_i <= redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_i + 1;
            END IF;
        END IF;
    END PROCESS;
    redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_q <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR(RESIZE(redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_i, 6)));

    -- redist180_firstQuadrant_uid15_sincosTest_b_57_inputreg(DELAY,1243)
    redist180_firstQuadrant_uid15_sincosTest_b_57_inputreg : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => firstQuadrant_uid15_sincosTest_b, xout => redist180_firstQuadrant_uid15_sincosTest_b_57_inputreg_q, clk => clk, aclr => areset );

    -- redist180_firstQuadrant_uid15_sincosTest_b_57_wraddr(REG,1246)
    redist180_firstQuadrant_uid15_sincosTest_b_57_wraddr_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            redist180_firstQuadrant_uid15_sincosTest_b_57_wraddr_q <= "110110";
        ELSIF (clk'EVENT AND clk = '1') THEN
            redist180_firstQuadrant_uid15_sincosTest_b_57_wraddr_q <= STD_LOGIC_VECTOR(redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_q);
        END IF;
    END PROCESS;

    -- redist180_firstQuadrant_uid15_sincosTest_b_57_mem(DUALMEM,1244)
    redist180_firstQuadrant_uid15_sincosTest_b_57_mem_ia <= STD_LOGIC_VECTOR(redist180_firstQuadrant_uid15_sincosTest_b_57_inputreg_q);
    redist180_firstQuadrant_uid15_sincosTest_b_57_mem_aa <= redist180_firstQuadrant_uid15_sincosTest_b_57_wraddr_q;
    redist180_firstQuadrant_uid15_sincosTest_b_57_mem_ab <= redist180_firstQuadrant_uid15_sincosTest_b_57_rdcnt_q;
    redist180_firstQuadrant_uid15_sincosTest_b_57_mem_reset0 <= areset;
    redist180_firstQuadrant_uid15_sincosTest_b_57_mem_dmem : altera_syncram
    GENERIC MAP (
        ram_block_type => "MLAB",
        operation_mode => "DUAL_PORT",
        width_a => 1,
        widthad_a => 6,
        numwords_a => 55,
        width_b => 1,
        widthad_b => 6,
        numwords_b => 55,
        lpm_type => "altera_syncram",
        width_byteena_a => 1,
        address_reg_b => "CLOCK0",
        indata_reg_b => "CLOCK0",
        rdcontrol_reg_b => "CLOCK0",
        byteena_reg_b => "CLOCK0",
        outdata_reg_b => "CLOCK1",
        outdata_aclr_b => "CLEAR1",
        clock_enable_input_a => "NORMAL",
        clock_enable_input_b => "NORMAL",
        clock_enable_output_b => "NORMAL",
        read_during_write_mode_mixed_ports => "DONT_CARE",
        power_up_uninitialized => "TRUE",
        intended_device_family => "Stratix V"
    )
    PORT MAP (
        clocken1 => redist180_firstQuadrant_uid15_sincosTest_b_57_enaAnd_q(0),
        clocken0 => VCC_q(0),
        clock0 => clk,
        aclr1 => redist180_firstQuadrant_uid15_sincosTest_b_57_mem_reset0,
        clock1 => clk,
        address_a => redist180_firstQuadrant_uid15_sincosTest_b_57_mem_aa,
        data_a => redist180_firstQuadrant_uid15_sincosTest_b_57_mem_ia,
        wren_a => VCC_q(0),
        address_b => redist180_firstQuadrant_uid15_sincosTest_b_57_mem_ab,
        q_b => redist180_firstQuadrant_uid15_sincosTest_b_57_mem_iq
    );
    redist180_firstQuadrant_uid15_sincosTest_b_57_mem_q <= redist180_firstQuadrant_uid15_sincosTest_b_57_mem_iq(0 downto 0);

    -- VCC(CONSTANT,1)
    VCC_q <= "1";

    -- xPostRR_uid1035_sincosTest(MUX,1034)@58
    xPostRR_uid1035_sincosTest_s <= redist180_firstQuadrant_uid15_sincosTest_b_57_mem_q;
    xPostRR_uid1035_sincosTest_combproc: PROCESS (xPostRR_uid1035_sincosTest_s, cosPostNeg_uid1033_sincosTest_q, sinPostNeg_uid1031_sincosTest_q)
    BEGIN
        CASE (xPostRR_uid1035_sincosTest_s) IS
            WHEN "0" => xPostRR_uid1035_sincosTest_q <= cosPostNeg_uid1033_sincosTest_q;
            WHEN "1" => xPostRR_uid1035_sincosTest_q <= sinPostNeg_uid1031_sincosTest_q;
            WHEN OTHERS => xPostRR_uid1035_sincosTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- sin_uid1037_sincosTest(BITSELECT,1036)@58
    sin_uid1037_sincosTest_in <= STD_LOGIC_VECTOR(xPostRR_uid1035_sincosTest_q(54 downto 0));
    sin_uid1037_sincosTest_b <= STD_LOGIC_VECTOR(sin_uid1037_sincosTest_in(54 downto 0));

    -- xPostRR_uid1034_sincosTest(MUX,1033)@58
    xPostRR_uid1034_sincosTest_s <= redist180_firstQuadrant_uid15_sincosTest_b_57_mem_q;
    xPostRR_uid1034_sincosTest_combproc: PROCESS (xPostRR_uid1034_sincosTest_s, sinPostNeg_uid1031_sincosTest_q, cosPostNeg_uid1033_sincosTest_q)
    BEGIN
        CASE (xPostRR_uid1034_sincosTest_s) IS
            WHEN "0" => xPostRR_uid1034_sincosTest_q <= sinPostNeg_uid1031_sincosTest_q;
            WHEN "1" => xPostRR_uid1034_sincosTest_q <= cosPostNeg_uid1033_sincosTest_q;
            WHEN OTHERS => xPostRR_uid1034_sincosTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- cos_uid1036_sincosTest(BITSELECT,1035)@58
    cos_uid1036_sincosTest_in <= STD_LOGIC_VECTOR(xPostRR_uid1034_sincosTest_q(54 downto 0));
    cos_uid1036_sincosTest_b <= STD_LOGIC_VECTOR(cos_uid1036_sincosTest_in(54 downto 0));

    -- xOut(GPOUT,4)@58
    c <= cos_uid1036_sincosTest_b;
    s <= sin_uid1037_sincosTest_b;

END normal;
