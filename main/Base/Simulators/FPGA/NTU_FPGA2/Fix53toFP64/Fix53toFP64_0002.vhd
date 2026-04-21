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

-- VHDL created from Fix53toFP64_0002
-- VHDL created on Sat Apr 18 01:27:25 2026


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

entity Fix53toFP64_0002 is
    port (
        a : in std_logic_vector(54 downto 0);  -- sfix55_en53
        q : out std_logic_vector(63 downto 0);  -- float64_m52
        clk : in std_logic;
        areset : in std_logic
    );
end Fix53toFP64_0002;

architecture normal of Fix53toFP64_0002 is

    attribute altera_attribute : string;
    attribute altera_attribute of normal : architecture is "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name PHYSICAL_SYNTHESIS_REGISTER_DUPLICATION ON; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007";
    
    signal GND_q : STD_LOGIC_VECTOR (0 downto 0);
    signal VCC_q : STD_LOGIC_VECTOR (0 downto 0);
    signal signX_uid6_fxpToFPTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal xXorSign_uid7_fxpToFPTest_b : STD_LOGIC_VECTOR (54 downto 0);
    signal xXorSign_uid7_fxpToFPTest_qi : STD_LOGIC_VECTOR (54 downto 0);
    signal xXorSign_uid7_fxpToFPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal y_uid9_fxpToFPTest_in : STD_LOGIC_VECTOR (54 downto 0);
    signal y_uid9_fxpToFPTest_b : STD_LOGIC_VECTOR (54 downto 0);
    signal maxCount_uid11_fxpToFPTest_q : STD_LOGIC_VECTOR (5 downto 0);
    signal inIsZero_uid12_fxpToFPTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal inIsZero_uid12_fxpToFPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal msbIn_uid13_fxpToFPTest_q : STD_LOGIC_VECTOR (10 downto 0);
    signal expPreRnd_uid14_fxpToFPTest_a : STD_LOGIC_VECTOR (11 downto 0);
    signal expPreRnd_uid14_fxpToFPTest_b : STD_LOGIC_VECTOR (11 downto 0);
    signal expPreRnd_uid14_fxpToFPTest_o : STD_LOGIC_VECTOR (11 downto 0);
    signal expPreRnd_uid14_fxpToFPTest_q : STD_LOGIC_VECTOR (11 downto 0);
    signal expFracRnd_uid16_fxpToFPTest_q : STD_LOGIC_VECTOR (64 downto 0);
    signal nr_uid20_fxpToFPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal rnd_uid21_fxpToFPTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal rnd_uid21_fxpToFPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracR_uid24_fxpToFPTest_in : STD_LOGIC_VECTOR (52 downto 0);
    signal fracR_uid24_fxpToFPTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal expR_uid25_fxpToFPTest_b : STD_LOGIC_VECTOR (12 downto 0);
    signal udf_uid26_fxpToFPTest_a : STD_LOGIC_VECTOR (14 downto 0);
    signal udf_uid26_fxpToFPTest_b : STD_LOGIC_VECTOR (14 downto 0);
    signal udf_uid26_fxpToFPTest_o : STD_LOGIC_VECTOR (14 downto 0);
    signal udf_uid26_fxpToFPTest_n : STD_LOGIC_VECTOR (0 downto 0);
    signal expInf_uid27_fxpToFPTest_q : STD_LOGIC_VECTOR (10 downto 0);
    signal ovf_uid28_fxpToFPTest_a : STD_LOGIC_VECTOR (14 downto 0);
    signal ovf_uid28_fxpToFPTest_b : STD_LOGIC_VECTOR (14 downto 0);
    signal ovf_uid28_fxpToFPTest_o : STD_LOGIC_VECTOR (14 downto 0);
    signal ovf_uid28_fxpToFPTest_n : STD_LOGIC_VECTOR (0 downto 0);
    signal excSelector_uid29_fxpToFPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracZ_uid30_fxpToFPTest_q : STD_LOGIC_VECTOR (51 downto 0);
    signal fracRPostExc_uid31_fxpToFPTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostExc_uid31_fxpToFPTest_q : STD_LOGIC_VECTOR (51 downto 0);
    signal udfOrInZero_uid32_fxpToFPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excSelector_uid33_fxpToFPTest_q : STD_LOGIC_VECTOR (1 downto 0);
    signal expZ_uid36_fxpToFPTest_q : STD_LOGIC_VECTOR (10 downto 0);
    signal expR_uid37_fxpToFPTest_in : STD_LOGIC_VECTOR (10 downto 0);
    signal expR_uid37_fxpToFPTest_b : STD_LOGIC_VECTOR (10 downto 0);
    signal expRPostExc_uid38_fxpToFPTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal expRPostExc_uid38_fxpToFPTest_q : STD_LOGIC_VECTOR (10 downto 0);
    signal outRes_uid39_fxpToFPTest_q : STD_LOGIC_VECTOR (63 downto 0);
    signal zs_uid41_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (31 downto 0);
    signal rVStage_uid42_lzcShifterZ1_uid10_fxpToFPTest_b : STD_LOGIC_VECTOR (31 downto 0);
    signal vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal zs_uid48_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (15 downto 0);
    signal rVStage_uid49_lzcShifterZ1_uid10_fxpToFPTest_b : STD_LOGIC_VECTOR (15 downto 0);
    signal vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal zs_uid55_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (7 downto 0);
    signal rVStage_uid56_lzcShifterZ1_uid10_fxpToFPTest_b : STD_LOGIC_VECTOR (7 downto 0);
    signal vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal zs_uid62_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (3 downto 0);
    signal rVStage_uid63_lzcShifterZ1_uid10_fxpToFPTest_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal zs_uid69_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (1 downto 0);
    signal rVStage_uid70_lzcShifterZ1_uid10_fxpToFPTest_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal rVStage_uid77_lzcShifterZ1_uid10_fxpToFPTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (5 downto 0);
    signal vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest_a : STD_LOGIC_VECTOR (7 downto 0);
    signal vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest_b : STD_LOGIC_VECTOR (7 downto 0);
    signal vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest_o : STD_LOGIC_VECTOR (7 downto 0);
    signal vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest_c : STD_LOGIC_VECTOR (0 downto 0);
    signal vCountFinal_uid87_lzcShifterZ1_uid10_fxpToFPTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vCountFinal_uid87_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (5 downto 0);
    signal yE_uid8_fxpToFPTest_BitExpansion_for_a_q : STD_LOGIC_VECTOR (55 downto 0);
    signal yE_uid8_fxpToFPTest_BitExpansion_for_b_q : STD_LOGIC_VECTOR (55 downto 0);
    signal yE_uid8_fxpToFPTest_UpperBits_for_b_q : STD_LOGIC_VECTOR (54 downto 0);
    signal yE_uid8_fxpToFPTest_BitSelect_for_a_b : STD_LOGIC_VECTOR (32 downto 0);
    signal yE_uid8_fxpToFPTest_BitSelect_for_a_c : STD_LOGIC_VECTOR (22 downto 0);
    signal yE_uid8_fxpToFPTest_BitSelect_for_b_b : STD_LOGIC_VECTOR (32 downto 0);
    signal yE_uid8_fxpToFPTest_p1_of_2_a : STD_LOGIC_VECTOR (33 downto 0);
    signal yE_uid8_fxpToFPTest_p1_of_2_b : STD_LOGIC_VECTOR (33 downto 0);
    signal yE_uid8_fxpToFPTest_p1_of_2_o : STD_LOGIC_VECTOR (33 downto 0);
    signal yE_uid8_fxpToFPTest_p1_of_2_c : STD_LOGIC_VECTOR (0 downto 0);
    signal yE_uid8_fxpToFPTest_p1_of_2_q : STD_LOGIC_VECTOR (32 downto 0);
    signal yE_uid8_fxpToFPTest_p2_of_2_a : STD_LOGIC_VECTOR (24 downto 0);
    signal yE_uid8_fxpToFPTest_p2_of_2_b : STD_LOGIC_VECTOR (24 downto 0);
    signal yE_uid8_fxpToFPTest_p2_of_2_o : STD_LOGIC_VECTOR (24 downto 0);
    signal yE_uid8_fxpToFPTest_p2_of_2_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal yE_uid8_fxpToFPTest_p2_of_2_q : STD_LOGIC_VECTOR (22 downto 0);
    signal yE_uid8_fxpToFPTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (55 downto 0);
    signal expFracR_uid23_fxpToFPTest_BitExpansion_for_a_q : STD_LOGIC_VECTOR (65 downto 0);
    signal expFracR_uid23_fxpToFPTest_BitExpansion_for_b_q : STD_LOGIC_VECTOR (65 downto 0);
    signal expFracR_uid23_fxpToFPTest_UpperBits_for_b_q : STD_LOGIC_VECTOR (64 downto 0);
    signal expFracR_uid23_fxpToFPTest_BitSelect_for_a_c : STD_LOGIC_VECTOR (32 downto 0);
    signal expFracR_uid23_fxpToFPTest_BitSelect_for_b_b : STD_LOGIC_VECTOR (32 downto 0);
    signal expFracR_uid23_fxpToFPTest_p1_of_2_a : STD_LOGIC_VECTOR (33 downto 0);
    signal expFracR_uid23_fxpToFPTest_p1_of_2_b : STD_LOGIC_VECTOR (33 downto 0);
    signal expFracR_uid23_fxpToFPTest_p1_of_2_o : STD_LOGIC_VECTOR (33 downto 0);
    signal expFracR_uid23_fxpToFPTest_p1_of_2_c : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracR_uid23_fxpToFPTest_p1_of_2_q : STD_LOGIC_VECTOR (32 downto 0);
    signal expFracR_uid23_fxpToFPTest_p2_of_2_a : STD_LOGIC_VECTOR (34 downto 0);
    signal expFracR_uid23_fxpToFPTest_p2_of_2_b : STD_LOGIC_VECTOR (34 downto 0);
    signal expFracR_uid23_fxpToFPTest_p2_of_2_o : STD_LOGIC_VECTOR (34 downto 0);
    signal expFracR_uid23_fxpToFPTest_p2_of_2_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracR_uid23_fxpToFPTest_p2_of_2_q : STD_LOGIC_VECTOR (32 downto 0);
    signal expFracR_uid23_fxpToFPTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (65 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q : STD_LOGIC_VECTOR (31 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q : STD_LOGIC_VECTOR (21 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (54 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q : STD_LOGIC_VECTOR (15 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q : STD_LOGIC_VECTOR (15 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q : STD_LOGIC_VECTOR (14 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_q : STD_LOGIC_VECTOR (5 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (54 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q : STD_LOGIC_VECTOR (7 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q : STD_LOGIC_VECTOR (7 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q : STD_LOGIC_VECTOR (7 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q : STD_LOGIC_VECTOR (7 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q : STD_LOGIC_VECTOR (6 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q : STD_LOGIC_VECTOR (6 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_q : STD_LOGIC_VECTOR (5 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (54 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_q : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_q : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_q : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_q : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_q : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_q : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_q : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p18_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p18_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p19_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p19_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (54 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p36_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p36_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p37_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p37_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p38_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p38_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (54 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (54 downto 0);
    signal yE_uid8_fxpToFPTest_BitSelect_for_b_tessel1_0_b : STD_LOGIC_VECTOR (22 downto 0);
    signal expFracR_uid23_fxpToFPTest_BitSelect_for_a_BitJoin_for_b_q : STD_LOGIC_VECTOR (32 downto 0);
    signal expFracR_uid23_fxpToFPTest_BitSelect_for_b_tessel1_0_b : STD_LOGIC_VECTOR (32 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b : STD_LOGIC_VECTOR (31 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b : STD_LOGIC_VECTOR (21 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel1_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b : STD_LOGIC_VECTOR (21 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b : STD_LOGIC_VECTOR (15 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b : STD_LOGIC_VECTOR (15 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b : STD_LOGIC_VECTOR (14 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b : STD_LOGIC_VECTOR (5 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel3_0_b : STD_LOGIC_VECTOR (14 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel5_0_b : STD_LOGIC_VECTOR (5 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b : STD_LOGIC_VECTOR (7 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b : STD_LOGIC_VECTOR (7 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b : STD_LOGIC_VECTOR (7 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b : STD_LOGIC_VECTOR (7 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b : STD_LOGIC_VECTOR (6 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b : STD_LOGIC_VECTOR (6 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel4_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel5_0_b : STD_LOGIC_VECTOR (6 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0_b : STD_LOGIC_VECTOR (5 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel8_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0_b : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel19_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel21_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel22_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel23_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel25_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel26_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel27_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel29_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel30_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel33_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel34_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel35_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel16_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel17_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel20_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel21_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel22_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel23_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel24_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel25_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel26_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel27_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel28_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel29_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel30_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal l_uid17_fxpToFPTest_merged_bit_select_b : STD_LOGIC_VECTOR (0 downto 0);
    signal l_uid17_fxpToFPTest_merged_bit_select_c : STD_LOGIC_VECTOR (0 downto 0);
    signal l_uid17_fxpToFPTest_merged_bit_select_d : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel37_0_merged_bit_select_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel37_0_merged_bit_select_c : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRnd_uid15_fxpToFPTest_merged_bit_select_in : STD_LOGIC_VECTOR (53 downto 0);
    signal fracRnd_uid15_fxpToFPTest_merged_bit_select_b : STD_LOGIC_VECTOR (52 downto 0);
    signal fracRnd_uid15_fxpToFPTest_merged_bit_select_c : STD_LOGIC_VECTOR (0 downto 0);
    signal redist0_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist1_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist2_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist3_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist4_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist5_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist6_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist7_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist8_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist9_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist10_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist11_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist12_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist13_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist14_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist15_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist16_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist17_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist18_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist19_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist20_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist21_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist22_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist23_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist24_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist25_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist26_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist27_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist28_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist29_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist30_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist31_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist32_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist33_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist34_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist35_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist36_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist37_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist38_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist39_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist40_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist41_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist42_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist43_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist44_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist45_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist46_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist47_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist48_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist49_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist50_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist51_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist52_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist53_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist54_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist55_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist56_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist57_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist58_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist59_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist60_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist61_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist62_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist63_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist64_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist65_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist66_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist67_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist68_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist69_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist70_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist71_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist72_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist73_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist74_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist75_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist76_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist77_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist78_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist79_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist80_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist81_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist82_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist83_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist84_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist85_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist86_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist87_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist88_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist89_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_q_1_q : STD_LOGIC_VECTOR (5 downto 0);
    signal redist90_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist91_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1_q : STD_LOGIC_VECTOR (6 downto 0);
    signal redist92_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist93_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1_q : STD_LOGIC_VECTOR (6 downto 0);
    signal redist94_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist95_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q : STD_LOGIC_VECTOR (7 downto 0);
    signal redist96_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q : STD_LOGIC_VECTOR (7 downto 0);
    signal redist97_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q : STD_LOGIC_VECTOR (7 downto 0);
    signal redist98_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q : STD_LOGIC_VECTOR (7 downto 0);
    signal redist99_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_q_2_q : STD_LOGIC_VECTOR (5 downto 0);
    signal redist100_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist101_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_2_q : STD_LOGIC_VECTOR (14 downto 0);
    signal redist102_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist103_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q : STD_LOGIC_VECTOR (15 downto 0);
    signal redist104_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q : STD_LOGIC_VECTOR (15 downto 0);
    signal redist105_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist106_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_2_q : STD_LOGIC_VECTOR (31 downto 0);
    signal redist107_expFracR_uid23_fxpToFPTest_p1_of_2_q_1_q : STD_LOGIC_VECTOR (32 downto 0);
    signal redist108_expFracR_uid23_fxpToFPTest_BitSelect_for_a_c_2_q : STD_LOGIC_VECTOR (32 downto 0);
    signal redist109_yE_uid8_fxpToFPTest_p1_of_2_q_1_q : STD_LOGIC_VECTOR (32 downto 0);
    signal redist110_yE_uid8_fxpToFPTest_BitSelect_for_a_c_1_q : STD_LOGIC_VECTOR (22 downto 0);
    signal redist111_vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q_1_q : STD_LOGIC_VECTOR (5 downto 0);
    signal redist112_vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist113_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist114_vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist115_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist116_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist117_vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q_5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist118_expR_uid37_fxpToFPTest_b_1_q : STD_LOGIC_VECTOR (10 downto 0);
    signal redist119_fracR_uid24_fxpToFPTest_b_1_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist120_inIsZero_uid12_fxpToFPTest_q_5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist121_signX_uid6_fxpToFPTest_b_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist122_signX_uid6_fxpToFPTest_b_15_q : STD_LOGIC_VECTOR (0 downto 0);

begin


    -- VCC(CONSTANT,1)
    VCC_q <= "1";

    -- signX_uid6_fxpToFPTest(BITSELECT,5)@0
    signX_uid6_fxpToFPTest_b <= STD_LOGIC_VECTOR(a(54 downto 54));

    -- redist121_signX_uid6_fxpToFPTest_b_1(DELAY,953)
    redist121_signX_uid6_fxpToFPTest_b_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => signX_uid6_fxpToFPTest_b, xout => redist121_signX_uid6_fxpToFPTest_b_1_q, clk => clk, aclr => areset );

    -- redist122_signX_uid6_fxpToFPTest_b_15(DELAY,954)
    redist122_signX_uid6_fxpToFPTest_b_15 : dspba_delay
    GENERIC MAP ( width => 1, depth => 14, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist121_signX_uid6_fxpToFPTest_b_1_q, xout => redist122_signX_uid6_fxpToFPTest_b_15_q, clk => clk, aclr => areset );

    -- expInf_uid27_fxpToFPTest(CONSTANT,26)
    expInf_uid27_fxpToFPTest_q <= "11111111111";

    -- expZ_uid36_fxpToFPTest(CONSTANT,35)
    expZ_uid36_fxpToFPTest_q <= "00000000000";

    -- GND(CONSTANT,0)
    GND_q <= "0";

    -- expFracR_uid23_fxpToFPTest_UpperBits_for_b(CONSTANT,102)
    expFracR_uid23_fxpToFPTest_UpperBits_for_b_q <= "00000000000000000000000000000000000000000000000000000000000000000";

    -- zs_uid41_lzcShifterZ1_uid10_fxpToFPTest(CONSTANT,40)
    zs_uid41_lzcShifterZ1_uid10_fxpToFPTest_q <= "00000000000000000000000000000000";

    -- yE_uid8_fxpToFPTest_UpperBits_for_b(CONSTANT,92)
    yE_uid8_fxpToFPTest_UpperBits_for_b_q <= "0000000000000000000000000000000000000000000000000000000";

    -- yE_uid8_fxpToFPTest_BitExpansion_for_b(BITJOIN,91)@1
    yE_uid8_fxpToFPTest_BitExpansion_for_b_q <= yE_uid8_fxpToFPTest_UpperBits_for_b_q & redist121_signX_uid6_fxpToFPTest_b_1_q;

    -- yE_uid8_fxpToFPTest_BitSelect_for_b(BITSELECT,94)@1
    yE_uid8_fxpToFPTest_BitSelect_for_b_b <= yE_uid8_fxpToFPTest_BitExpansion_for_b_q(32 downto 0);

    -- xXorSign_uid7_fxpToFPTest(LOGICAL,6)@0 + 1
    xXorSign_uid7_fxpToFPTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((54 downto 1 => signX_uid6_fxpToFPTest_b(0)) & signX_uid6_fxpToFPTest_b));
    xXorSign_uid7_fxpToFPTest_qi <= a xor xXorSign_uid7_fxpToFPTest_b;
    xXorSign_uid7_fxpToFPTest_delay : dspba_delay
    GENERIC MAP ( width => 55, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xXorSign_uid7_fxpToFPTest_qi, xout => xXorSign_uid7_fxpToFPTest_q, clk => clk, aclr => areset );

    -- yE_uid8_fxpToFPTest_BitExpansion_for_a(BITJOIN,89)@1
    yE_uid8_fxpToFPTest_BitExpansion_for_a_q <= GND_q & xXorSign_uid7_fxpToFPTest_q;

    -- yE_uid8_fxpToFPTest_BitSelect_for_a(BITSELECT,93)@1
    yE_uid8_fxpToFPTest_BitSelect_for_a_b <= yE_uid8_fxpToFPTest_BitExpansion_for_a_q(32 downto 0);
    yE_uid8_fxpToFPTest_BitSelect_for_a_c <= yE_uid8_fxpToFPTest_BitExpansion_for_a_q(55 downto 33);

    -- yE_uid8_fxpToFPTest_p1_of_2(ADD,95)@1 + 1
    yE_uid8_fxpToFPTest_p1_of_2_a <= STD_LOGIC_VECTOR("0" & yE_uid8_fxpToFPTest_BitSelect_for_a_b);
    yE_uid8_fxpToFPTest_p1_of_2_b <= STD_LOGIC_VECTOR("0" & yE_uid8_fxpToFPTest_BitSelect_for_b_b);
    yE_uid8_fxpToFPTest_p1_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            yE_uid8_fxpToFPTest_p1_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            yE_uid8_fxpToFPTest_p1_of_2_o <= STD_LOGIC_VECTOR(UNSIGNED(yE_uid8_fxpToFPTest_p1_of_2_a) + UNSIGNED(yE_uid8_fxpToFPTest_p1_of_2_b));
        END IF;
    END PROCESS;
    yE_uid8_fxpToFPTest_p1_of_2_c(0) <= yE_uid8_fxpToFPTest_p1_of_2_o(33);
    yE_uid8_fxpToFPTest_p1_of_2_q <= yE_uid8_fxpToFPTest_p1_of_2_o(32 downto 0);

    -- redist109_yE_uid8_fxpToFPTest_p1_of_2_q_1(DELAY,941)
    redist109_yE_uid8_fxpToFPTest_p1_of_2_q_1 : dspba_delay
    GENERIC MAP ( width => 33, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yE_uid8_fxpToFPTest_p1_of_2_q, xout => redist109_yE_uid8_fxpToFPTest_p1_of_2_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0(BITSELECT,297)@3
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b <= STD_LOGIC_VECTOR(redist109_yE_uid8_fxpToFPTest_p1_of_2_q_1_q(31 downto 0));

    -- yE_uid8_fxpToFPTest_BitSelect_for_b_tessel1_0(BITSELECT,259)
    yE_uid8_fxpToFPTest_BitSelect_for_b_tessel1_0_b <= STD_LOGIC_VECTOR(yE_uid8_fxpToFPTest_UpperBits_for_b_q(54 downto 32));

    -- redist110_yE_uid8_fxpToFPTest_BitSelect_for_a_c_1(DELAY,942)
    redist110_yE_uid8_fxpToFPTest_BitSelect_for_a_c_1 : dspba_delay
    GENERIC MAP ( width => 23, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yE_uid8_fxpToFPTest_BitSelect_for_a_c, xout => redist110_yE_uid8_fxpToFPTest_BitSelect_for_a_c_1_q, clk => clk, aclr => areset );

    -- yE_uid8_fxpToFPTest_p2_of_2(ADD,96)@2 + 1
    yE_uid8_fxpToFPTest_p2_of_2_cin <= yE_uid8_fxpToFPTest_p1_of_2_c;
    yE_uid8_fxpToFPTest_p2_of_2_a <= STD_LOGIC_VECTOR("0" & redist110_yE_uid8_fxpToFPTest_BitSelect_for_a_c_1_q) & '1';
    yE_uid8_fxpToFPTest_p2_of_2_b <= STD_LOGIC_VECTOR("0" & yE_uid8_fxpToFPTest_BitSelect_for_b_tessel1_0_b) & yE_uid8_fxpToFPTest_p2_of_2_cin(0);
    yE_uid8_fxpToFPTest_p2_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            yE_uid8_fxpToFPTest_p2_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            yE_uid8_fxpToFPTest_p2_of_2_o <= STD_LOGIC_VECTOR(UNSIGNED(yE_uid8_fxpToFPTest_p2_of_2_a) + UNSIGNED(yE_uid8_fxpToFPTest_p2_of_2_b));
        END IF;
    END PROCESS;
    yE_uid8_fxpToFPTest_p2_of_2_q <= yE_uid8_fxpToFPTest_p2_of_2_o(23 downto 1);

    -- yE_uid8_fxpToFPTest_BitJoin_for_q(BITJOIN,97)@3
    yE_uid8_fxpToFPTest_BitJoin_for_q_q <= yE_uid8_fxpToFPTest_p2_of_2_q & redist109_yE_uid8_fxpToFPTest_p1_of_2_q_1_q;

    -- y_uid9_fxpToFPTest(BITSELECT,8)@3
    y_uid9_fxpToFPTest_in <= STD_LOGIC_VECTOR(yE_uid8_fxpToFPTest_BitJoin_for_q_q(54 downto 0));
    y_uid9_fxpToFPTest_b <= STD_LOGIC_VECTOR(y_uid9_fxpToFPTest_in(54 downto 0));

    -- rVStage_uid42_lzcShifterZ1_uid10_fxpToFPTest(BITSELECT,41)@3
    rVStage_uid42_lzcShifterZ1_uid10_fxpToFPTest_b <= y_uid9_fxpToFPTest_b(54 downto 23);

    -- vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,42)@3
    vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q <= "1" WHEN rVStage_uid42_lzcShifterZ1_uid10_fxpToFPTest_b = zs_uid41_lzcShifterZ1_uid10_fxpToFPTest_q ELSE "0";

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0(MUX,110)@3 + 1
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_s <= vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_s) IS
                WHEN "0" => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
                WHEN "1" => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= zs_uid41_lzcShifterZ1_uid10_fxpToFPTest_q;
                WHEN OTHERS => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel3_0(BITSELECT,327)@4
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel3_0_b <= STD_LOGIC_VECTOR(vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q(31 downto 17));

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0(BITSELECT,307)@3
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b <= STD_LOGIC_VECTOR(redist109_yE_uid8_fxpToFPTest_p1_of_2_q_1_q(22 downto 1));

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0(BITSELECT,301)@3
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b <= STD_LOGIC_VECTOR(yE_uid8_fxpToFPTest_p2_of_2_q(21 downto 0));

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2(MUX,112)@3 + 1
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_s <= vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_s) IS
                WHEN "0" => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b;
                WHEN "1" => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b;
                WHEN OTHERS => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0(BITSELECT,315)@4
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b <= STD_LOGIC_VECTOR(vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q(14 downto 0));

    -- zs_uid48_lzcShifterZ1_uid10_fxpToFPTest(CONSTANT,47)
    zs_uid48_lzcShifterZ1_uid10_fxpToFPTest_q <= "0000000000000000";

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel1_0(BITSELECT,305)@3
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel1_0_b <= STD_LOGIC_VECTOR(redist109_yE_uid8_fxpToFPTest_p1_of_2_q_1_q(0 downto 0));

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0(BITSELECT,299)@3
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(redist109_yE_uid8_fxpToFPTest_p1_of_2_q_1_q(32 downto 32));

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1(MUX,111)@3 + 1
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_s <= vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_s) IS
                WHEN "0" => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
                WHEN "1" => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel1_0_b;
                WHEN OTHERS => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q(BITJOIN,113)@4
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q <= vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q & vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q & vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q;

    -- rVStage_uid49_lzcShifterZ1_uid10_fxpToFPTest(BITSELECT,48)@4
    rVStage_uid49_lzcShifterZ1_uid10_fxpToFPTest_b <= vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q(54 downto 39);

    -- vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,49)@4
    vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q <= "1" WHEN rVStage_uid49_lzcShifterZ1_uid10_fxpToFPTest_b = zs_uid48_lzcShifterZ1_uid10_fxpToFPTest_q ELSE "0";

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3(MUX,119)@4 + 1
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_s <= vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_s) IS
                WHEN "0" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b;
                WHEN "1" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel3_0_b;
                WHEN OTHERS => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist101_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_2(DELAY,933)
    redist101_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_2 : dspba_delay
    GENERIC MAP ( width => 15, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q, xout => redist101_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_2_q, clk => clk, aclr => areset );

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0(BITSELECT,343)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b <= STD_LOGIC_VECTOR(redist101_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_2_q(6 downto 0));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0(BITSELECT,347)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b <= STD_LOGIC_VECTOR(redist101_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_2_q(14 downto 8));

    -- zs_uid55_lzcShifterZ1_uid10_fxpToFPTest(CONSTANT,54)
    zs_uid55_lzcShifterZ1_uid10_fxpToFPTest_q <= "00000000";

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel5_0(BITSELECT,331)@4
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel5_0_b <= STD_LOGIC_VECTOR(vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q(5 downto 0));

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0(BITSELECT,319)@4
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b <= STD_LOGIC_VECTOR(vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q(21 downto 16));

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5(MUX,121)@4 + 1
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_s <= vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_s) IS
                WHEN "0" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b;
                WHEN "1" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel5_0_b;
                WHEN OTHERS => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist105_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_2(DELAY,937)
    redist105_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q, xout => redist105_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_2_q, clk => clk, aclr => areset );

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0(BITSELECT,317)@4
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b <= STD_LOGIC_VECTOR(vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q(15 downto 15));

    -- redist0_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b_1(DELAY,832)
    redist0_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b, xout => redist0_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b_1_q, clk => clk, aclr => areset );

    -- redist115_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_1(DELAY,947)
    redist115_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q, xout => redist115_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4(MUX,120)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_s <= redist115_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_combproc: PROCESS (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_s, redist0_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b_1_q, redist105_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_2_q)
    BEGIN
        CASE (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_s) IS
            WHEN "0" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= redist0_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b_1_q;
            WHEN "1" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= redist105_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_2_q;
            WHEN OTHERS => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist106_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_2(DELAY,938)
    redist106_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_2 : dspba_delay
    GENERIC MAP ( width => 32, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q, xout => redist106_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_2_q, clk => clk, aclr => areset );

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0(BITSELECT,325)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b <= STD_LOGIC_VECTOR(redist106_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_2_q(16 downto 16));

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2(MUX,118)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_s <= redist115_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_combproc: PROCESS (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_s, redist105_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_2_q, vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b)
    BEGIN
        CASE (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_s) IS
            WHEN "0" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= redist105_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_2_q;
            WHEN "1" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b;
            WHEN OTHERS => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0(BITSELECT,309)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b <= STD_LOGIC_VECTOR(redist106_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_2_q(15 downto 0));

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0(BITSELECT,311)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(redist106_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_2_q(31 downto 16));

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1(MUX,117)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_s <= redist115_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_combproc: PROCESS (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_s, vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b, vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b)
    BEGIN
        CASE (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_s) IS
            WHEN "0" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
            WHEN "1" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
            WHEN OTHERS => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0(MUX,116)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_s <= redist115_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_combproc: PROCESS (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_s, vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b, zs_uid48_lzcShifterZ1_uid10_fxpToFPTest_q)
    BEGIN
        CASE (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_s) IS
            WHEN "0" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
            WHEN "1" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= zs_uid48_lzcShifterZ1_uid10_fxpToFPTest_q;
            WHEN OTHERS => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q(BITJOIN,122)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_q & vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q & vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q & vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q & vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q & vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q;

    -- rVStage_uid56_lzcShifterZ1_uid10_fxpToFPTest(BITSELECT,55)@5
    rVStage_uid56_lzcShifterZ1_uid10_fxpToFPTest_b <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q(54 downto 47);

    -- vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,56)@5 + 1
    vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_qi <= "1" WHEN rVStage_uid56_lzcShifterZ1_uid10_fxpToFPTest_b = zs_uid55_lzcShifterZ1_uid10_fxpToFPTest_q ELSE "0";
    vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_qi, xout => vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q, clk => clk, aclr => areset );

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7(MUX,132)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_s, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist91_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1(DELAY,923)
    redist91_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1 : dspba_delay
    GENERIC MAP ( width => 7, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q, xout => redist91_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0(BITSELECT,403)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b <= STD_LOGIC_VECTOR(redist91_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1_q(6 downto 4));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0(BITSELECT,371)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0_b <= STD_LOGIC_VECTOR(redist101_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_2_q(13 downto 8));

    -- redist99_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_q_2(DELAY,931)
    redist99_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_q_2 : dspba_delay
    GENERIC MAP ( width => 6, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_q, xout => redist99_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_q_2_q, clk => clk, aclr => areset );

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9(MUX,134)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_s, redist99_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_q_2_q, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= redist99_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p5_q_2_q;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist89_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_q_1(DELAY,921)
    redist89_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_q_1 : dspba_delay
    GENERIC MAP ( width => 6, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_q, xout => redist89_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0(BITSELECT,407)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b <= STD_LOGIC_VECTOR(redist89_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_q_1_q(2 downto 0));

    -- zs_uid62_lzcShifterZ1_uid10_fxpToFPTest(CONSTANT,61)
    zs_uid62_lzcShifterZ1_uid10_fxpToFPTest_q <= "0000";

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0(BITSELECT,345)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b <= STD_LOGIC_VECTOR(redist101_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_2_q(7 downto 7));

    -- redist100_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1(DELAY,932)
    redist100_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q, xout => redist100_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8(MUX,133)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_s, redist100_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= redist100_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist102_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1(DELAY,934)
    redist102_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q, xout => redist102_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6(MUX,131)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_s, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b, redist102_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= redist102_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist103_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1(DELAY,935)
    redist103_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1 : dspba_delay
    GENERIC MAP ( width => 16, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q, xout => redist103_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel5_0(BITSELECT,363)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel5_0_b <= STD_LOGIC_VECTOR(redist103_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q(15 downto 9));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5(MUX,130)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_s, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel5_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel5_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel4_0(BITSELECT,361)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel4_0_b <= STD_LOGIC_VECTOR(redist103_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q(8 downto 8));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4(MUX,129)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_s, redist102_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel4_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= redist102_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel4_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0(BITSELECT,337)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b <= STD_LOGIC_VECTOR(redist103_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q(7 downto 0));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0(BITSELECT,339)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b <= STD_LOGIC_VECTOR(redist103_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q(15 downto 8));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3(MUX,128)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_s, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist104_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1(DELAY,936)
    redist104_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1 : dspba_delay
    GENERIC MAP ( width => 16, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q, xout => redist104_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0(BITSELECT,335)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(redist104_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q(15 downto 8));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2(MUX,127)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_s, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0(BITSELECT,333)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b <= STD_LOGIC_VECTOR(redist104_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q(7 downto 0));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1(MUX,126)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_s, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0(MUX,125)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_s, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b, zs_uid55_lzcShifterZ1_uid10_fxpToFPTest_q)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= zs_uid55_lzcShifterZ1_uid10_fxpToFPTest_q;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q(BITJOIN,135)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q;

    -- rVStage_uid63_lzcShifterZ1_uid10_fxpToFPTest(BITSELECT,62)@6
    rVStage_uid63_lzcShifterZ1_uid10_fxpToFPTest_b <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q(54 downto 51);

    -- vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,63)@6 + 1
    vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_qi <= "1" WHEN rVStage_uid63_lzcShifterZ1_uid10_fxpToFPTest_b = zs_uid62_lzcShifterZ1_uid10_fxpToFPTest_q ELSE "0";
    vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_qi, xout => vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17(MUX,155)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel35_0(BITSELECT,523)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel35_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q(2 downto 2));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel19_0(BITSELECT,451)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel19_0_b <= STD_LOGIC_VECTOR(redist89_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_q_1_q(1 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0(BITSELECT,411)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b <= STD_LOGIC_VECTOR(redist89_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_q_1_q(5 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p19(MUX,157)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p19_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p19_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p19_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel19_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p19_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel19_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel37_0_merged_bit_select(BITSELECT,830)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel37_0_merged_bit_select_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p19_q(0 downto 0));
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel37_0_merged_bit_select_c <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p19_q(1 downto 1));

    -- zs_uid69_lzcShifterZ1_uid10_fxpToFPTest(CONSTANT,68)
    zs_uid69_lzcShifterZ1_uid10_fxpToFPTest_q <= "00";

    -- redist90_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q_1(DELAY,922)
    redist90_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q, xout => redist90_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0(BITSELECT,409)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b <= STD_LOGIC_VECTOR(redist89_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p9_q_1_q(3 downto 3));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p18(MUX,156)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p18_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p18_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p18_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b, redist90_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q_1_q)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p18_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= redist90_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q_1_q;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0(BITSELECT,401)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b <= STD_LOGIC_VECTOR(redist91_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1_q(3 downto 3));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16(MUX,154)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_s, redist90_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q_1_q, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= redist90_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q_1_q;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0(BITSELECT,399)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b <= STD_LOGIC_VECTOR(redist91_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1_q(2 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15(MUX,153)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist92_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1(DELAY,924)
    redist92_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q, xout => redist92_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14(MUX,152)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b, redist92_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1_q)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_q <= redist92_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1_q;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist93_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1(DELAY,925)
    redist93_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1 : dspba_delay
    GENERIC MAP ( width => 7, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q, xout => redist93_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0(BITSELECT,395)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b <= STD_LOGIC_VECTOR(redist93_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1_q(6 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13(MUX,151)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0(BITSELECT,393)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b <= STD_LOGIC_VECTOR(redist93_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1_q(3 downto 3));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12(MUX,150)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_s, redist92_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1_q, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_q <= redist92_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1_q;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0(BITSELECT,391)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b <= STD_LOGIC_VECTOR(redist93_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1_q(2 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11(MUX,149)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist94_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1(DELAY,926)
    redist94_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q, xout => redist94_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10(MUX,148)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b, redist94_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_q <= redist94_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist95_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1(DELAY,927)
    redist95_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1 : dspba_delay
    GENERIC MAP ( width => 8, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q, xout => redist95_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0(BITSELECT,431)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0_b <= STD_LOGIC_VECTOR(redist95_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q(7 downto 5));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9(MUX,147)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel8_0(BITSELECT,429)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel8_0_b <= STD_LOGIC_VECTOR(redist95_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q(4 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8(MUX,146)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_s, redist94_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel8_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= redist94_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel8_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0(BITSELECT,385)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b <= STD_LOGIC_VECTOR(redist95_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q(3 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0(BITSELECT,387)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b <= STD_LOGIC_VECTOR(redist95_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q(7 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7(MUX,145)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist96_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1(DELAY,928)
    redist96_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1 : dspba_delay
    GENERIC MAP ( width => 8, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q, xout => redist96_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0(BITSELECT,383)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b <= STD_LOGIC_VECTOR(redist96_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q(7 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6(MUX,144)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0(BITSELECT,381)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b <= STD_LOGIC_VECTOR(redist96_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q(3 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5(MUX,143)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist97_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1(DELAY,929)
    redist97_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1 : dspba_delay
    GENERIC MAP ( width => 8, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q, xout => redist97_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0(BITSELECT,379)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b <= STD_LOGIC_VECTOR(redist97_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q(7 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4(MUX,142)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0(BITSELECT,377)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b <= STD_LOGIC_VECTOR(redist97_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q(3 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3(MUX,141)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist98_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1(DELAY,930)
    redist98_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1 : dspba_delay
    GENERIC MAP ( width => 8, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q, xout => redist98_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0(BITSELECT,375)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(redist98_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q(7 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2(MUX,140)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0(BITSELECT,373)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b <= STD_LOGIC_VECTOR(redist98_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q(3 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1(MUX,139)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0(MUX,138)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b, zs_uid62_lzcShifterZ1_uid10_fxpToFPTest_q)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= zs_uid62_lzcShifterZ1_uid10_fxpToFPTest_q;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q(BITJOIN,158)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p19_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p18_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_q;

    -- rVStage_uid70_lzcShifterZ1_uid10_fxpToFPTest(BITSELECT,69)@7
    rVStage_uid70_lzcShifterZ1_uid10_fxpToFPTest_b <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q(54 downto 53);

    -- vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,70)@7
    vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q <= "1" WHEN rVStage_uid70_lzcShifterZ1_uid10_fxpToFPTest_b = zs_uid69_lzcShifterZ1_uid10_fxpToFPTest_q ELSE "0";

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p37(MUX,198)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p37_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p37_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p37_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p37_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p37_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel37_0_merged_bit_select_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p37_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel35_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p37_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p38(MUX,199)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p38_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p38_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p38_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p38_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p38_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel37_0_merged_bit_select_c;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p38_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p18_q;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p38_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel34_0(BITSELECT,521)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel34_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q(1 downto 1));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p36(MUX,197)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p36_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p36_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p36_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p36_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p36_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p18_q;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p36_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel34_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p36_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel33_0(BITSELECT,519)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel33_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q(0 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35(MUX,196)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel35_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel33_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34(MUX,195)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel34_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_q;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0(BITSELECT,515)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q(2 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33(MUX,194)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel33_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel30_0(BITSELECT,513)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel30_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q(1 downto 1));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32(MUX,193)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_q;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel30_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel29_0(BITSELECT,511)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel29_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q(0 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31(MUX,192)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel29_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30(MUX,191)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel30_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_q;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel27_0(BITSELECT,507)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel27_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q(2 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29(MUX,190)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel29_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel27_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel26_0(BITSELECT,505)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel26_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q(1 downto 1));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28(MUX,189)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_q;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel26_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel25_0(BITSELECT,503)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel25_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q(0 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27(MUX,188)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel27_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel25_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26(MUX,187)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel26_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_q;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel23_0(BITSELECT,499)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel23_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q(2 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25(MUX,186)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel25_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel23_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel22_0(BITSELECT,497)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel22_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q(1 downto 1));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24(MUX,185)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_q;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel22_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel21_0(BITSELECT,495)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel21_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q(0 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23(MUX,184)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel23_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel21_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22(MUX,183)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel22_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_q;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0(BITSELECT,491)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q(2 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21(MUX,182)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel21_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0(BITSELECT,489)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q(1 downto 1));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20(MUX,181)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_q;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0(BITSELECT,487)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q(0 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19(MUX,180)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18(MUX,179)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_q;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel17_0(BITSELECT,565)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel17_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q(3 downto 3));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17(MUX,178)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel17_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel16_0(BITSELECT,563)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel16_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q(2 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16(MUX,177)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_q;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel16_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0(BITSELECT,481)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0(BITSELECT,483)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15(MUX,176)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0(BITSELECT,479)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14(MUX,175)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0(BITSELECT,477)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13(MUX,174)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0(BITSELECT,475)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12(MUX,173)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0(BITSELECT,473)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11(MUX,172)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0(BITSELECT,471)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10(MUX,171)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0(BITSELECT,469)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9(MUX,170)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0(BITSELECT,467)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8(MUX,169)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0(BITSELECT,465)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7(MUX,168)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0(BITSELECT,463)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6(MUX,167)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0(BITSELECT,461)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5(MUX,166)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0(BITSELECT,459)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4(MUX,165)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0(BITSELECT,457)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3(MUX,164)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0(BITSELECT,455)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2(MUX,163)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0(BITSELECT,453)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1(MUX,162)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0(MUX,161)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= zs_uid69_lzcShifterZ1_uid10_fxpToFPTest_q;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q(BITJOIN,200)@8
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p38_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p37_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p36_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_q;

    -- rVStage_uid77_lzcShifterZ1_uid10_fxpToFPTest(BITSELECT,76)@8
    rVStage_uid77_lzcShifterZ1_uid10_fxpToFPTest_b <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q(54 downto 54);

    -- vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,77)@8
    vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q <= "1" WHEN rVStage_uid77_lzcShifterZ1_uid10_fxpToFPTest_b = GND_q ELSE "0";

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54(MUX,257)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p38_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p37_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist1_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q_3(DELAY,833)
    redist1_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q, xout => redist1_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53(MUX,256)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p37_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p36_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist2_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q_3(DELAY,834)
    redist2_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q, xout => redist2_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52(MUX,255)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p36_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist3_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q_3(DELAY,835)
    redist3_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q, xout => redist3_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51(MUX,254)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist4_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q_3(DELAY,836)
    redist4_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q, xout => redist4_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50(MUX,253)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist5_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q_3(DELAY,837)
    redist5_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q, xout => redist5_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49(MUX,252)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist6_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q_3(DELAY,838)
    redist6_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q, xout => redist6_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48(MUX,251)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist7_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q_3(DELAY,839)
    redist7_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q, xout => redist7_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47(MUX,250)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist8_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q_3(DELAY,840)
    redist8_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q, xout => redist8_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46(MUX,249)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist9_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q_3(DELAY,841)
    redist9_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q, xout => redist9_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45(MUX,248)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist10_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q_3(DELAY,842)
    redist10_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q, xout => redist10_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44(MUX,247)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist11_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q_3(DELAY,843)
    redist11_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q, xout => redist11_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43(MUX,246)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist12_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q_3(DELAY,844)
    redist12_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q, xout => redist12_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42(MUX,245)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist13_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q_3(DELAY,845)
    redist13_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q, xout => redist13_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41(MUX,244)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist14_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q_3(DELAY,846)
    redist14_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q, xout => redist14_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40(MUX,243)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist15_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q_3(DELAY,847)
    redist15_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q, xout => redist15_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39(MUX,242)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist16_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_3(DELAY,848)
    redist16_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q, xout => redist16_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38(MUX,241)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist17_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_3(DELAY,849)
    redist17_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q, xout => redist17_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37(MUX,240)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist18_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_3(DELAY,850)
    redist18_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q, xout => redist18_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36(MUX,239)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist19_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_3(DELAY,851)
    redist19_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q, xout => redist19_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35(MUX,238)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist20_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_3(DELAY,852)
    redist20_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q, xout => redist20_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34(MUX,237)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist21_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_3(DELAY,853)
    redist21_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q, xout => redist21_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33(MUX,236)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist22_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_3(DELAY,854)
    redist22_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q, xout => redist22_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0(BITSELECT,671)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32(MUX,235)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist24_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_3(DELAY,856)
    redist24_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q, xout => redist24_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel30_0(BITSELECT,669)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel30_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31(MUX,234)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel30_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist26_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_3(DELAY,858)
    redist26_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q, xout => redist26_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel29_0(BITSELECT,667)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel29_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30(MUX,233)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel30_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel29_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist28_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_3(DELAY,860)
    redist28_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q, xout => redist28_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel28_0(BITSELECT,665)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel28_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29(MUX,232)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel29_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel28_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist30_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_3(DELAY,862)
    redist30_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q, xout => redist30_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel27_0(BITSELECT,663)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel27_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28(MUX,231)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel28_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel27_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist32_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_3(DELAY,864)
    redist32_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q, xout => redist32_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel26_0(BITSELECT,661)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel26_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27(MUX,230)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel27_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel26_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist34_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_3(DELAY,866)
    redist34_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q, xout => redist34_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel25_0(BITSELECT,659)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel25_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26(MUX,229)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel26_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel25_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist36_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_3(DELAY,868)
    redist36_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q, xout => redist36_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel24_0(BITSELECT,657)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel24_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25(MUX,228)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel25_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel24_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist38_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_3(DELAY,870)
    redist38_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q, xout => redist38_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel23_0(BITSELECT,655)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel23_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24(MUX,227)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel24_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel23_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist40_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_3(DELAY,872)
    redist40_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q, xout => redist40_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel22_0(BITSELECT,653)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel22_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23(MUX,226)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel23_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel22_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist42_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_3(DELAY,874)
    redist42_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q, xout => redist42_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel21_0(BITSELECT,651)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel21_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22(MUX,225)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel22_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel21_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist44_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_3(DELAY,876)
    redist44_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q, xout => redist44_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel20_0(BITSELECT,649)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel20_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21(MUX,224)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel21_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel20_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist46_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_3(DELAY,878)
    redist46_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q, xout => redist46_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0(BITSELECT,647)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20(MUX,223)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel20_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist48_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_3(DELAY,880)
    redist48_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q, xout => redist48_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0(BITSELECT,645)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19(MUX,222)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist50_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_3(DELAY,882)
    redist50_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q, xout => redist50_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0(BITSELECT,643)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18(MUX,221)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist52_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_3(DELAY,884)
    redist52_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q, xout => redist52_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0(BITSELECT,641)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17(MUX,220)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist54_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_3(DELAY,886)
    redist54_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q, xout => redist54_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0(BITSELECT,639)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16(MUX,219)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist56_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_3(DELAY,888)
    redist56_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q, xout => redist56_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0(BITSELECT,637)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15(MUX,218)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist58_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_3(DELAY,890)
    redist58_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q, xout => redist58_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0(BITSELECT,635)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14(MUX,217)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist60_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_3(DELAY,892)
    redist60_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q, xout => redist60_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0(BITSELECT,633)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13(MUX,216)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist62_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_3(DELAY,894)
    redist62_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q, xout => redist62_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0(BITSELECT,631)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12(MUX,215)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist64_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_3(DELAY,896)
    redist64_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q, xout => redist64_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0(BITSELECT,629)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11(MUX,214)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist66_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_3(DELAY,898)
    redist66_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q, xout => redist66_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0(BITSELECT,627)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10(MUX,213)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist68_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_3(DELAY,900)
    redist68_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q, xout => redist68_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0(BITSELECT,625)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9(MUX,212)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist70_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_3(DELAY,902)
    redist70_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q, xout => redist70_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0(BITSELECT,623)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8(MUX,211)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist72_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_3(DELAY,904)
    redist72_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q, xout => redist72_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0(BITSELECT,621)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7(MUX,210)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist74_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_3(DELAY,906)
    redist74_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q, xout => redist74_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0(BITSELECT,619)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6(MUX,209)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist76_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_3(DELAY,908)
    redist76_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q, xout => redist76_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0(BITSELECT,617)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5(MUX,208)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist78_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_3(DELAY,910)
    redist78_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q, xout => redist78_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0(BITSELECT,615)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4(MUX,207)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist80_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_3(DELAY,912)
    redist80_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q, xout => redist80_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0(BITSELECT,613)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3(MUX,206)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist82_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_3(DELAY,914)
    redist82_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q, xout => redist82_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0(BITSELECT,611)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2(MUX,205)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist84_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_3(DELAY,916)
    redist84_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q, xout => redist84_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0(BITSELECT,609)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1(MUX,204)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist86_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_3(DELAY,918)
    redist86_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q, xout => redist86_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0(MUX,203)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= GND_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist88_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q_3(DELAY,920)
    redist88_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q, xout => redist88_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q(BITJOIN,258)@11
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q <= redist1_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q_3_q & redist2_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q_3_q & redist3_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q_3_q & redist4_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q_3_q & redist5_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q_3_q & redist6_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q_3_q & redist7_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q_3_q & redist8_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q_3_q & redist9_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q_3_q & redist10_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q_3_q & redist11_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q_3_q & redist12_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q_3_q & redist13_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q_3_q & redist14_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q_3_q & redist15_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q_3_q & redist16_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_3_q & redist17_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_3_q & redist18_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_3_q & redist19_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_3_q & redist20_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_3_q & redist21_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_3_q & redist22_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_3_q & redist24_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_3_q & redist26_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_3_q & redist28_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_3_q & redist30_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_3_q & redist32_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_3_q & redist34_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_3_q & redist36_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_3_q & redist38_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_3_q & redist40_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_3_q & redist42_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_3_q & redist44_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_3_q & redist46_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_3_q & redist48_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_3_q & redist50_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_3_q & redist52_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_3_q & redist54_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_3_q & redist56_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_3_q & redist58_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_3_q & redist60_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_3_q & redist62_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_3_q & redist64_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_3_q & redist66_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_3_q & redist68_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_3_q & redist70_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_3_q & redist72_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_3_q & redist74_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_3_q & redist76_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_3_q & redist78_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_3_q & redist80_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_3_q & redist82_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_3_q & redist84_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_3_q & redist86_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_3_q & redist88_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q_3_q;

    -- fracRnd_uid15_fxpToFPTest_merged_bit_select(BITSELECT,831)@11
    fracRnd_uid15_fxpToFPTest_merged_bit_select_in <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q(53 downto 0);
    fracRnd_uid15_fxpToFPTest_merged_bit_select_b <= fracRnd_uid15_fxpToFPTest_merged_bit_select_in(53 downto 1);
    fracRnd_uid15_fxpToFPTest_merged_bit_select_c <= fracRnd_uid15_fxpToFPTest_merged_bit_select_in(0 downto 0);

    -- nr_uid20_fxpToFPTest(LOGICAL,19)@11
    nr_uid20_fxpToFPTest_q <= not (l_uid17_fxpToFPTest_merged_bit_select_c);

    -- maxCount_uid11_fxpToFPTest(CONSTANT,10)
    maxCount_uid11_fxpToFPTest_q <= "110111";

    -- redist117_vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q_5(DELAY,949)
    redist117_vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q_5 : dspba_delay
    GENERIC MAP ( width => 1, depth => 5, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q, xout => redist117_vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q_5_q, clk => clk, aclr => areset );

    -- redist116_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_4(DELAY,948)
    redist116_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist115_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_1_q, xout => redist116_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_4_q, clk => clk, aclr => areset );

    -- redist114_vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q_3(DELAY,946)
    redist114_vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q, xout => redist114_vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q_3_q, clk => clk, aclr => areset );

    -- redist113_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_2(DELAY,945)
    redist113_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q, xout => redist113_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_2_q, clk => clk, aclr => areset );

    -- redist112_vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q_1(DELAY,944)
    redist112_vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q, xout => redist112_vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q_1_q, clk => clk, aclr => areset );

    -- vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest(BITJOIN,82)@8
    vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q <= redist117_vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q_5_q & redist116_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_4_q & redist114_vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q_3_q & redist113_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_2_q & redist112_vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q_1_q & vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;

    -- redist111_vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q_1(DELAY,943)
    redist111_vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q_1 : dspba_delay
    GENERIC MAP ( width => 6, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q, xout => redist111_vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q_1_q, clk => clk, aclr => areset );

    -- vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest(COMPARE,84)@8 + 1
    vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest_a <= STD_LOGIC_VECTOR("00" & maxCount_uid11_fxpToFPTest_q);
    vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest_b <= STD_LOGIC_VECTOR("00" & vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q);
    vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest_o <= STD_LOGIC_VECTOR(UNSIGNED(vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest_a) - UNSIGNED(vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest_b));
        END IF;
    END PROCESS;
    vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest_c(0) <= vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest_o(7);

    -- vCountFinal_uid87_lzcShifterZ1_uid10_fxpToFPTest(MUX,86)@9 + 1
    vCountFinal_uid87_lzcShifterZ1_uid10_fxpToFPTest_s <= vCountBig_uid85_lzcShifterZ1_uid10_fxpToFPTest_c;
    vCountFinal_uid87_lzcShifterZ1_uid10_fxpToFPTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vCountFinal_uid87_lzcShifterZ1_uid10_fxpToFPTest_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vCountFinal_uid87_lzcShifterZ1_uid10_fxpToFPTest_s) IS
                WHEN "0" => vCountFinal_uid87_lzcShifterZ1_uid10_fxpToFPTest_q <= redist111_vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
                WHEN "1" => vCountFinal_uid87_lzcShifterZ1_uid10_fxpToFPTest_q <= maxCount_uid11_fxpToFPTest_q;
                WHEN OTHERS => vCountFinal_uid87_lzcShifterZ1_uid10_fxpToFPTest_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- msbIn_uid13_fxpToFPTest(CONSTANT,12)
    msbIn_uid13_fxpToFPTest_q <= "10000000000";

    -- expPreRnd_uid14_fxpToFPTest(SUB,13)@10 + 1
    expPreRnd_uid14_fxpToFPTest_a <= STD_LOGIC_VECTOR("0" & msbIn_uid13_fxpToFPTest_q);
    expPreRnd_uid14_fxpToFPTest_b <= STD_LOGIC_VECTOR("000000" & vCountFinal_uid87_lzcShifterZ1_uid10_fxpToFPTest_q);
    expPreRnd_uid14_fxpToFPTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expPreRnd_uid14_fxpToFPTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expPreRnd_uid14_fxpToFPTest_o <= STD_LOGIC_VECTOR(UNSIGNED(expPreRnd_uid14_fxpToFPTest_a) - UNSIGNED(expPreRnd_uid14_fxpToFPTest_b));
        END IF;
    END PROCESS;
    expPreRnd_uid14_fxpToFPTest_q <= expPreRnd_uid14_fxpToFPTest_o(11 downto 0);

    -- expFracRnd_uid16_fxpToFPTest(BITJOIN,15)@11
    expFracRnd_uid16_fxpToFPTest_q <= expPreRnd_uid14_fxpToFPTest_q & fracRnd_uid15_fxpToFPTest_merged_bit_select_b;

    -- l_uid17_fxpToFPTest_merged_bit_select(BITSELECT,829)@11
    l_uid17_fxpToFPTest_merged_bit_select_b <= STD_LOGIC_VECTOR(expFracRnd_uid16_fxpToFPTest_q(1 downto 1));
    l_uid17_fxpToFPTest_merged_bit_select_c <= STD_LOGIC_VECTOR(expFracRnd_uid16_fxpToFPTest_q(0 downto 0));
    l_uid17_fxpToFPTest_merged_bit_select_d <= STD_LOGIC_VECTOR(expFracRnd_uid16_fxpToFPTest_q(64 downto 64));

    -- rnd_uid21_fxpToFPTest(LOGICAL,20)@11 + 1
    rnd_uid21_fxpToFPTest_qi <= l_uid17_fxpToFPTest_merged_bit_select_b or nr_uid20_fxpToFPTest_q or fracRnd_uid15_fxpToFPTest_merged_bit_select_c;
    rnd_uid21_fxpToFPTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => rnd_uid21_fxpToFPTest_qi, xout => rnd_uid21_fxpToFPTest_q, clk => clk, aclr => areset );

    -- expFracR_uid23_fxpToFPTest_BitExpansion_for_b(BITJOIN,101)@12
    expFracR_uid23_fxpToFPTest_BitExpansion_for_b_q <= expFracR_uid23_fxpToFPTest_UpperBits_for_b_q & rnd_uid21_fxpToFPTest_q;

    -- expFracR_uid23_fxpToFPTest_BitSelect_for_b(BITSELECT,104)@12
    expFracR_uid23_fxpToFPTest_BitSelect_for_b_b <= expFracR_uid23_fxpToFPTest_BitExpansion_for_b_q(32 downto 0);

    -- redist23_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_4(DELAY,855)
    redist23_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist22_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_3_q, xout => redist23_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_4_q, clk => clk, aclr => areset );

    -- redist25_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_4(DELAY,857)
    redist25_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist24_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_3_q, xout => redist25_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_4_q, clk => clk, aclr => areset );

    -- redist27_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_4(DELAY,859)
    redist27_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist26_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_3_q, xout => redist27_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_4_q, clk => clk, aclr => areset );

    -- redist29_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_4(DELAY,861)
    redist29_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist28_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_3_q, xout => redist29_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_4_q, clk => clk, aclr => areset );

    -- redist31_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_4(DELAY,863)
    redist31_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist30_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_3_q, xout => redist31_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_4_q, clk => clk, aclr => areset );

    -- redist33_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_4(DELAY,865)
    redist33_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist32_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_3_q, xout => redist33_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_4_q, clk => clk, aclr => areset );

    -- redist35_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_4(DELAY,867)
    redist35_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist34_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_3_q, xout => redist35_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_4_q, clk => clk, aclr => areset );

    -- redist37_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_4(DELAY,869)
    redist37_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist36_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_3_q, xout => redist37_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_4_q, clk => clk, aclr => areset );

    -- redist39_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_4(DELAY,871)
    redist39_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist38_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_3_q, xout => redist39_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_4_q, clk => clk, aclr => areset );

    -- redist41_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_4(DELAY,873)
    redist41_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist40_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_3_q, xout => redist41_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_4_q, clk => clk, aclr => areset );

    -- redist43_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_4(DELAY,875)
    redist43_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist42_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_3_q, xout => redist43_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_4_q, clk => clk, aclr => areset );

    -- redist45_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_4(DELAY,877)
    redist45_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist44_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_3_q, xout => redist45_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_4_q, clk => clk, aclr => areset );

    -- redist47_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_4(DELAY,879)
    redist47_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist46_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_3_q, xout => redist47_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_4_q, clk => clk, aclr => areset );

    -- redist49_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_4(DELAY,881)
    redist49_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist48_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_3_q, xout => redist49_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_4_q, clk => clk, aclr => areset );

    -- redist51_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_4(DELAY,883)
    redist51_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist50_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_3_q, xout => redist51_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_4_q, clk => clk, aclr => areset );

    -- redist53_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_4(DELAY,885)
    redist53_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist52_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_3_q, xout => redist53_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_4_q, clk => clk, aclr => areset );

    -- redist55_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_4(DELAY,887)
    redist55_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist54_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_3_q, xout => redist55_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_4_q, clk => clk, aclr => areset );

    -- redist57_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_4(DELAY,889)
    redist57_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist56_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_3_q, xout => redist57_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_4_q, clk => clk, aclr => areset );

    -- redist59_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_4(DELAY,891)
    redist59_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist58_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_3_q, xout => redist59_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_4_q, clk => clk, aclr => areset );

    -- redist61_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_4(DELAY,893)
    redist61_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist60_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_3_q, xout => redist61_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_4_q, clk => clk, aclr => areset );

    -- redist63_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_4(DELAY,895)
    redist63_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist62_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_3_q, xout => redist63_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_4_q, clk => clk, aclr => areset );

    -- redist65_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_4(DELAY,897)
    redist65_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist64_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_3_q, xout => redist65_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_4_q, clk => clk, aclr => areset );

    -- redist67_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_4(DELAY,899)
    redist67_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist66_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_3_q, xout => redist67_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_4_q, clk => clk, aclr => areset );

    -- redist69_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_4(DELAY,901)
    redist69_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist68_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_3_q, xout => redist69_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_4_q, clk => clk, aclr => areset );

    -- redist71_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_4(DELAY,903)
    redist71_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist70_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_3_q, xout => redist71_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_4_q, clk => clk, aclr => areset );

    -- redist73_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_4(DELAY,905)
    redist73_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist72_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_3_q, xout => redist73_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_4_q, clk => clk, aclr => areset );

    -- redist75_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_4(DELAY,907)
    redist75_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist74_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_3_q, xout => redist75_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_4_q, clk => clk, aclr => areset );

    -- redist77_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_4(DELAY,909)
    redist77_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist76_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_3_q, xout => redist77_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_4_q, clk => clk, aclr => areset );

    -- redist79_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_4(DELAY,911)
    redist79_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist78_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_3_q, xout => redist79_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_4_q, clk => clk, aclr => areset );

    -- redist81_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_4(DELAY,913)
    redist81_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist80_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_3_q, xout => redist81_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_4_q, clk => clk, aclr => areset );

    -- redist83_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_4(DELAY,915)
    redist83_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist82_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_3_q, xout => redist83_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_4_q, clk => clk, aclr => areset );

    -- redist85_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_4(DELAY,917)
    redist85_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist84_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_3_q, xout => redist85_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_4_q, clk => clk, aclr => areset );

    -- redist87_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_4(DELAY,919)
    redist87_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist86_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_3_q, xout => redist87_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_4_q, clk => clk, aclr => areset );

    -- expFracR_uid23_fxpToFPTest_BitSelect_for_a_BitJoin_for_b(BITJOIN,294)@12
    expFracR_uid23_fxpToFPTest_BitSelect_for_a_BitJoin_for_b_q <= redist23_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_4_q & redist25_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_4_q & redist27_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_4_q & redist29_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_4_q & redist31_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_4_q & redist33_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_4_q & redist35_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_4_q & redist37_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_4_q & redist39_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_4_q & redist41_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_4_q & redist43_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_4_q & redist45_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_4_q & redist47_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_4_q & redist49_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_4_q & redist51_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_4_q & redist53_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_4_q & redist55_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_4_q & redist57_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_4_q & redist59_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_4_q & redist61_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_4_q & redist63_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_4_q & redist65_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_4_q & redist67_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_4_q & redist69_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_4_q & redist71_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_4_q & redist73_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_4_q & redist75_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_4_q & redist77_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_4_q & redist79_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_4_q & redist81_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_4_q & redist83_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_4_q & redist85_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_4_q & redist87_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_4_q;

    -- expFracR_uid23_fxpToFPTest_p1_of_2(ADD,105)@12 + 1
    expFracR_uid23_fxpToFPTest_p1_of_2_a <= STD_LOGIC_VECTOR("0" & expFracR_uid23_fxpToFPTest_BitSelect_for_a_BitJoin_for_b_q);
    expFracR_uid23_fxpToFPTest_p1_of_2_b <= STD_LOGIC_VECTOR("0" & expFracR_uid23_fxpToFPTest_BitSelect_for_b_b);
    expFracR_uid23_fxpToFPTest_p1_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expFracR_uid23_fxpToFPTest_p1_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expFracR_uid23_fxpToFPTest_p1_of_2_o <= STD_LOGIC_VECTOR(UNSIGNED(expFracR_uid23_fxpToFPTest_p1_of_2_a) + UNSIGNED(expFracR_uid23_fxpToFPTest_p1_of_2_b));
        END IF;
    END PROCESS;
    expFracR_uid23_fxpToFPTest_p1_of_2_c(0) <= expFracR_uid23_fxpToFPTest_p1_of_2_o(33);
    expFracR_uid23_fxpToFPTest_p1_of_2_q <= expFracR_uid23_fxpToFPTest_p1_of_2_o(32 downto 0);

    -- expFracR_uid23_fxpToFPTest_BitSelect_for_b_tessel1_0(BITSELECT,295)
    expFracR_uid23_fxpToFPTest_BitSelect_for_b_tessel1_0_b <= STD_LOGIC_VECTOR(expFracR_uid23_fxpToFPTest_UpperBits_for_b_q(64 downto 32));

    -- expFracR_uid23_fxpToFPTest_BitExpansion_for_a(BITJOIN,98)@11
    expFracR_uid23_fxpToFPTest_BitExpansion_for_a_q <= l_uid17_fxpToFPTest_merged_bit_select_d & expFracRnd_uid16_fxpToFPTest_q;

    -- expFracR_uid23_fxpToFPTest_BitSelect_for_a(BITSELECT,103)@11
    expFracR_uid23_fxpToFPTest_BitSelect_for_a_c <= STD_LOGIC_VECTOR(expFracR_uid23_fxpToFPTest_BitExpansion_for_a_q(65 downto 33));

    -- redist108_expFracR_uid23_fxpToFPTest_BitSelect_for_a_c_2(DELAY,940)
    redist108_expFracR_uid23_fxpToFPTest_BitSelect_for_a_c_2 : dspba_delay
    GENERIC MAP ( width => 33, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => expFracR_uid23_fxpToFPTest_BitSelect_for_a_c, xout => redist108_expFracR_uid23_fxpToFPTest_BitSelect_for_a_c_2_q, clk => clk, aclr => areset );

    -- expFracR_uid23_fxpToFPTest_p2_of_2(ADD,106)@13 + 1
    expFracR_uid23_fxpToFPTest_p2_of_2_cin <= expFracR_uid23_fxpToFPTest_p1_of_2_c;
    expFracR_uid23_fxpToFPTest_p2_of_2_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((33 downto 33 => redist108_expFracR_uid23_fxpToFPTest_BitSelect_for_a_c_2_q(32)) & redist108_expFracR_uid23_fxpToFPTest_BitSelect_for_a_c_2_q) & '1');
    expFracR_uid23_fxpToFPTest_p2_of_2_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0" & expFracR_uid23_fxpToFPTest_BitSelect_for_b_tessel1_0_b) & expFracR_uid23_fxpToFPTest_p2_of_2_cin(0));
    expFracR_uid23_fxpToFPTest_p2_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expFracR_uid23_fxpToFPTest_p2_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expFracR_uid23_fxpToFPTest_p2_of_2_o <= STD_LOGIC_VECTOR(SIGNED(expFracR_uid23_fxpToFPTest_p2_of_2_a) + SIGNED(expFracR_uid23_fxpToFPTest_p2_of_2_b));
        END IF;
    END PROCESS;
    expFracR_uid23_fxpToFPTest_p2_of_2_q <= expFracR_uid23_fxpToFPTest_p2_of_2_o(33 downto 1);

    -- redist107_expFracR_uid23_fxpToFPTest_p1_of_2_q_1(DELAY,939)
    redist107_expFracR_uid23_fxpToFPTest_p1_of_2_q_1 : dspba_delay
    GENERIC MAP ( width => 33, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => expFracR_uid23_fxpToFPTest_p1_of_2_q, xout => redist107_expFracR_uid23_fxpToFPTest_p1_of_2_q_1_q, clk => clk, aclr => areset );

    -- expFracR_uid23_fxpToFPTest_BitJoin_for_q(BITJOIN,107)@14
    expFracR_uid23_fxpToFPTest_BitJoin_for_q_q <= expFracR_uid23_fxpToFPTest_p2_of_2_q & redist107_expFracR_uid23_fxpToFPTest_p1_of_2_q_1_q;

    -- expR_uid25_fxpToFPTest(BITSELECT,24)@14
    expR_uid25_fxpToFPTest_b <= STD_LOGIC_VECTOR(expFracR_uid23_fxpToFPTest_BitJoin_for_q_q(65 downto 53));

    -- expR_uid37_fxpToFPTest(BITSELECT,36)@14
    expR_uid37_fxpToFPTest_in <= expR_uid25_fxpToFPTest_b(10 downto 0);
    expR_uid37_fxpToFPTest_b <= expR_uid37_fxpToFPTest_in(10 downto 0);

    -- redist118_expR_uid37_fxpToFPTest_b_1(DELAY,950)
    redist118_expR_uid37_fxpToFPTest_b_1 : dspba_delay
    GENERIC MAP ( width => 11, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => expR_uid37_fxpToFPTest_b, xout => redist118_expR_uid37_fxpToFPTest_b_1_q, clk => clk, aclr => areset );

    -- ovf_uid28_fxpToFPTest(COMPARE,27)@14 + 1
    ovf_uid28_fxpToFPTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((14 downto 13 => expR_uid25_fxpToFPTest_b(12)) & expR_uid25_fxpToFPTest_b));
    ovf_uid28_fxpToFPTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0000" & expInf_uid27_fxpToFPTest_q));
    ovf_uid28_fxpToFPTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            ovf_uid28_fxpToFPTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            ovf_uid28_fxpToFPTest_o <= STD_LOGIC_VECTOR(SIGNED(ovf_uid28_fxpToFPTest_a) - SIGNED(ovf_uid28_fxpToFPTest_b));
        END IF;
    END PROCESS;
    ovf_uid28_fxpToFPTest_n(0) <= not (ovf_uid28_fxpToFPTest_o(14));

    -- inIsZero_uid12_fxpToFPTest(LOGICAL,11)@10 + 1
    inIsZero_uid12_fxpToFPTest_qi <= "1" WHEN vCountFinal_uid87_lzcShifterZ1_uid10_fxpToFPTest_q = maxCount_uid11_fxpToFPTest_q ELSE "0";
    inIsZero_uid12_fxpToFPTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => inIsZero_uid12_fxpToFPTest_qi, xout => inIsZero_uid12_fxpToFPTest_q, clk => clk, aclr => areset );

    -- redist120_inIsZero_uid12_fxpToFPTest_q_5(DELAY,952)
    redist120_inIsZero_uid12_fxpToFPTest_q_5 : dspba_delay
    GENERIC MAP ( width => 1, depth => 4, reset_kind => "ASYNC" )
    PORT MAP ( xin => inIsZero_uid12_fxpToFPTest_q, xout => redist120_inIsZero_uid12_fxpToFPTest_q_5_q, clk => clk, aclr => areset );

    -- udf_uid26_fxpToFPTest(COMPARE,25)@14 + 1
    udf_uid26_fxpToFPTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("00000000000000" & GND_q));
    udf_uid26_fxpToFPTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((14 downto 13 => expR_uid25_fxpToFPTest_b(12)) & expR_uid25_fxpToFPTest_b));
    udf_uid26_fxpToFPTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            udf_uid26_fxpToFPTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            udf_uid26_fxpToFPTest_o <= STD_LOGIC_VECTOR(SIGNED(udf_uid26_fxpToFPTest_a) - SIGNED(udf_uid26_fxpToFPTest_b));
        END IF;
    END PROCESS;
    udf_uid26_fxpToFPTest_n(0) <= not (udf_uid26_fxpToFPTest_o(14));

    -- udfOrInZero_uid32_fxpToFPTest(LOGICAL,31)@15
    udfOrInZero_uid32_fxpToFPTest_q <= udf_uid26_fxpToFPTest_n or redist120_inIsZero_uid12_fxpToFPTest_q_5_q;

    -- excSelector_uid33_fxpToFPTest(BITJOIN,32)@15
    excSelector_uid33_fxpToFPTest_q <= ovf_uid28_fxpToFPTest_n & udfOrInZero_uid32_fxpToFPTest_q;

    -- expRPostExc_uid38_fxpToFPTest(MUX,37)@15
    expRPostExc_uid38_fxpToFPTest_s <= excSelector_uid33_fxpToFPTest_q;
    expRPostExc_uid38_fxpToFPTest_combproc: PROCESS (expRPostExc_uid38_fxpToFPTest_s, redist118_expR_uid37_fxpToFPTest_b_1_q, expZ_uid36_fxpToFPTest_q, expInf_uid27_fxpToFPTest_q)
    BEGIN
        CASE (expRPostExc_uid38_fxpToFPTest_s) IS
            WHEN "00" => expRPostExc_uid38_fxpToFPTest_q <= redist118_expR_uid37_fxpToFPTest_b_1_q;
            WHEN "01" => expRPostExc_uid38_fxpToFPTest_q <= expZ_uid36_fxpToFPTest_q;
            WHEN "10" => expRPostExc_uid38_fxpToFPTest_q <= expInf_uid27_fxpToFPTest_q;
            WHEN "11" => expRPostExc_uid38_fxpToFPTest_q <= expInf_uid27_fxpToFPTest_q;
            WHEN OTHERS => expRPostExc_uid38_fxpToFPTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- fracZ_uid30_fxpToFPTest(CONSTANT,29)
    fracZ_uid30_fxpToFPTest_q <= "0000000000000000000000000000000000000000000000000000";

    -- fracR_uid24_fxpToFPTest(BITSELECT,23)@14
    fracR_uid24_fxpToFPTest_in <= expFracR_uid23_fxpToFPTest_BitJoin_for_q_q(52 downto 0);
    fracR_uid24_fxpToFPTest_b <= fracR_uid24_fxpToFPTest_in(52 downto 1);

    -- redist119_fracR_uid24_fxpToFPTest_b_1(DELAY,951)
    redist119_fracR_uid24_fxpToFPTest_b_1 : dspba_delay
    GENERIC MAP ( width => 52, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => fracR_uid24_fxpToFPTest_b, xout => redist119_fracR_uid24_fxpToFPTest_b_1_q, clk => clk, aclr => areset );

    -- excSelector_uid29_fxpToFPTest(LOGICAL,28)@15
    excSelector_uid29_fxpToFPTest_q <= redist120_inIsZero_uid12_fxpToFPTest_q_5_q or ovf_uid28_fxpToFPTest_n or udf_uid26_fxpToFPTest_n;

    -- fracRPostExc_uid31_fxpToFPTest(MUX,30)@15
    fracRPostExc_uid31_fxpToFPTest_s <= excSelector_uid29_fxpToFPTest_q;
    fracRPostExc_uid31_fxpToFPTest_combproc: PROCESS (fracRPostExc_uid31_fxpToFPTest_s, redist119_fracR_uid24_fxpToFPTest_b_1_q, fracZ_uid30_fxpToFPTest_q)
    BEGIN
        CASE (fracRPostExc_uid31_fxpToFPTest_s) IS
            WHEN "0" => fracRPostExc_uid31_fxpToFPTest_q <= redist119_fracR_uid24_fxpToFPTest_b_1_q;
            WHEN "1" => fracRPostExc_uid31_fxpToFPTest_q <= fracZ_uid30_fxpToFPTest_q;
            WHEN OTHERS => fracRPostExc_uid31_fxpToFPTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- outRes_uid39_fxpToFPTest(BITJOIN,38)@15
    outRes_uid39_fxpToFPTest_q <= redist122_signX_uid6_fxpToFPTest_b_15_q & expRPostExc_uid38_fxpToFPTest_q & fracRPostExc_uid31_fxpToFPTest_q;

    -- xOut(GPOUT,4)@15
    q <= outRes_uid39_fxpToFPTest_q;

END normal;
