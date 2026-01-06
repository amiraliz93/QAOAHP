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

-- VHDL created from Fix53toFP64_0002
-- VHDL created on Wed Jan 07 02:54:41 2026


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
    signal vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal zs_uid48_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (15 downto 0);
    signal rVStage_uid49_lzcShifterZ1_uid10_fxpToFPTest_b : STD_LOGIC_VECTOR (15 downto 0);
    signal vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal zs_uid55_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (7 downto 0);
    signal rVStage_uid56_lzcShifterZ1_uid10_fxpToFPTest_b : STD_LOGIC_VECTOR (7 downto 0);
    signal vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal zs_uid62_lzcShifterZ1_uid10_fxpToFPTest_q : STD_LOGIC_VECTOR (3 downto 0);
    signal rVStage_uid63_lzcShifterZ1_uid10_fxpToFPTest_b : STD_LOGIC_VECTOR (3 downto 0);
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
    signal yE_uid8_fxpToFPTest_BitSelect_for_a_b : STD_LOGIC_VECTOR (38 downto 0);
    signal yE_uid8_fxpToFPTest_BitSelect_for_a_c : STD_LOGIC_VECTOR (16 downto 0);
    signal yE_uid8_fxpToFPTest_BitSelect_for_b_b : STD_LOGIC_VECTOR (38 downto 0);
    signal yE_uid8_fxpToFPTest_p1_of_2_a : STD_LOGIC_VECTOR (39 downto 0);
    signal yE_uid8_fxpToFPTest_p1_of_2_b : STD_LOGIC_VECTOR (39 downto 0);
    signal yE_uid8_fxpToFPTest_p1_of_2_o : STD_LOGIC_VECTOR (39 downto 0);
    signal yE_uid8_fxpToFPTest_p1_of_2_c : STD_LOGIC_VECTOR (0 downto 0);
    signal yE_uid8_fxpToFPTest_p1_of_2_q : STD_LOGIC_VECTOR (38 downto 0);
    signal yE_uid8_fxpToFPTest_p2_of_2_a : STD_LOGIC_VECTOR (18 downto 0);
    signal yE_uid8_fxpToFPTest_p2_of_2_b : STD_LOGIC_VECTOR (18 downto 0);
    signal yE_uid8_fxpToFPTest_p2_of_2_o : STD_LOGIC_VECTOR (18 downto 0);
    signal yE_uid8_fxpToFPTest_p2_of_2_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal yE_uid8_fxpToFPTest_p2_of_2_q : STD_LOGIC_VECTOR (16 downto 0);
    signal yE_uid8_fxpToFPTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (55 downto 0);
    signal expFracR_uid23_fxpToFPTest_BitExpansion_for_a_q : STD_LOGIC_VECTOR (65 downto 0);
    signal expFracR_uid23_fxpToFPTest_BitExpansion_for_b_q : STD_LOGIC_VECTOR (65 downto 0);
    signal expFracR_uid23_fxpToFPTest_UpperBits_for_b_q : STD_LOGIC_VECTOR (64 downto 0);
    signal expFracR_uid23_fxpToFPTest_BitSelect_for_a_c : STD_LOGIC_VECTOR (26 downto 0);
    signal expFracR_uid23_fxpToFPTest_BitSelect_for_b_b : STD_LOGIC_VECTOR (38 downto 0);
    signal expFracR_uid23_fxpToFPTest_p1_of_2_a : STD_LOGIC_VECTOR (39 downto 0);
    signal expFracR_uid23_fxpToFPTest_p1_of_2_b : STD_LOGIC_VECTOR (39 downto 0);
    signal expFracR_uid23_fxpToFPTest_p1_of_2_o : STD_LOGIC_VECTOR (39 downto 0);
    signal expFracR_uid23_fxpToFPTest_p1_of_2_c : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracR_uid23_fxpToFPTest_p1_of_2_q : STD_LOGIC_VECTOR (38 downto 0);
    signal expFracR_uid23_fxpToFPTest_p2_of_2_a : STD_LOGIC_VECTOR (28 downto 0);
    signal expFracR_uid23_fxpToFPTest_p2_of_2_b : STD_LOGIC_VECTOR (28 downto 0);
    signal expFracR_uid23_fxpToFPTest_p2_of_2_o : STD_LOGIC_VECTOR (28 downto 0);
    signal expFracR_uid23_fxpToFPTest_p2_of_2_cin : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracR_uid23_fxpToFPTest_p2_of_2_q : STD_LOGIC_VECTOR (26 downto 0);
    signal expFracR_uid23_fxpToFPTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (65 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q : STD_LOGIC_VECTOR (31 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q : STD_LOGIC_VECTOR (6 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q : STD_LOGIC_VECTOR (15 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q : STD_LOGIC_VECTOR (54 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q : STD_LOGIC_VECTOR (15 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q : STD_LOGIC_VECTOR (15 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q : STD_LOGIC_VECTOR (6 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q : STD_LOGIC_VECTOR (8 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q : STD_LOGIC_VECTOR (6 downto 0);
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
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q : STD_LOGIC_VECTOR (6 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q : STD_LOGIC_VECTOR (6 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q : STD_LOGIC_VECTOR (6 downto 0);
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
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_q : STD_LOGIC_VECTOR (3 downto 0);
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
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q : STD_LOGIC_VECTOR (1 downto 0);
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
    signal yE_uid8_fxpToFPTest_BitSelect_for_b_tessel1_0_b : STD_LOGIC_VECTOR (16 downto 0);
    signal expFracR_uid23_fxpToFPTest_BitSelect_for_a_BitJoin_for_b_q : STD_LOGIC_VECTOR (38 downto 0);
    signal expFracR_uid23_fxpToFPTest_BitSelect_for_b_tessel1_0_b : STD_LOGIC_VECTOR (26 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b : STD_LOGIC_VECTOR (31 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b : STD_LOGIC_VECTOR (6 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b : STD_LOGIC_VECTOR (15 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel1_0_b : STD_LOGIC_VECTOR (6 downto 0);
    signal vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b : STD_LOGIC_VECTOR (15 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b : STD_LOGIC_VECTOR (15 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b : STD_LOGIC_VECTOR (15 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b : STD_LOGIC_VECTOR (6 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel3_0_b : STD_LOGIC_VECTOR (8 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b : STD_LOGIC_VECTOR (7 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b : STD_LOGIC_VECTOR (7 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b : STD_LOGIC_VECTOR (7 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b : STD_LOGIC_VECTOR (7 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b : STD_LOGIC_VECTOR (6 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel4_0_b : STD_LOGIC_VECTOR (6 downto 0);
    signal vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel5_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0_b : STD_LOGIC_VECTOR (3 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0_b : STD_LOGIC_VECTOR (2 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel10_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel15_0_b : STD_LOGIC_VECTOR (2 downto 0);
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
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b : STD_LOGIC_VECTOR (1 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b : STD_LOGIC_VECTOR (1 downto 0);
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
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel19_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel20_0_b : STD_LOGIC_VECTOR (0 downto 0);
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
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel32_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel33_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel34_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel35_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel36_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel37_0_b : STD_LOGIC_VECTOR (0 downto 0);
    signal l_uid17_fxpToFPTest_merged_bit_select_b : STD_LOGIC_VECTOR (0 downto 0);
    signal l_uid17_fxpToFPTest_merged_bit_select_c : STD_LOGIC_VECTOR (0 downto 0);
    signal l_uid17_fxpToFPTest_merged_bit_select_d : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_merged_bit_select_b : STD_LOGIC_VECTOR (8 downto 0);
    signal vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_merged_bit_select_c : STD_LOGIC_VECTOR (6 downto 0);
    signal fracRnd_uid15_fxpToFPTest_merged_bit_select_in : STD_LOGIC_VECTOR (53 downto 0);
    signal fracRnd_uid15_fxpToFPTest_merged_bit_select_b : STD_LOGIC_VECTOR (52 downto 0);
    signal fracRnd_uid15_fxpToFPTest_merged_bit_select_c : STD_LOGIC_VECTOR (0 downto 0);
    signal redist0_vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist1_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b_1_q : STD_LOGIC_VECTOR (15 downto 0);
    signal redist2_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist3_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist4_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist5_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist6_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist7_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist8_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist9_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist10_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist11_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist12_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist13_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist14_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist15_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist16_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist17_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist18_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist19_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist20_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist21_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist22_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist23_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist24_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist25_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist26_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist27_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist28_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist29_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist30_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist31_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist32_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist33_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist34_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist35_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist36_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist37_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist38_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist39_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist40_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist41_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist42_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist43_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist44_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist45_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist46_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist47_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist48_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist49_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist50_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist51_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist52_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist53_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist54_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist55_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist56_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist57_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist58_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist59_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist60_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist61_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist62_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist63_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist64_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist65_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist66_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist67_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist68_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist69_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist70_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist71_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist72_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist73_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist74_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist75_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist76_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist77_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist78_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist79_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist80_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist81_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist82_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist83_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist84_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist85_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist86_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist87_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist88_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist89_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist90_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist91_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist92_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist93_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist94_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist95_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist96_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist97_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1_q : STD_LOGIC_VECTOR (6 downto 0);
    signal redist98_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist99_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q : STD_LOGIC_VECTOR (6 downto 0);
    signal redist100_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q : STD_LOGIC_VECTOR (7 downto 0);
    signal redist101_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q : STD_LOGIC_VECTOR (7 downto 0);
    signal redist102_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q : STD_LOGIC_VECTOR (7 downto 0);
    signal redist103_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q : STD_LOGIC_VECTOR (7 downto 0);
    signal redist104_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q : STD_LOGIC_VECTOR (6 downto 0);
    signal redist105_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q : STD_LOGIC_VECTOR (8 downto 0);
    signal redist106_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q : STD_LOGIC_VECTOR (6 downto 0);
    signal redist107_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q : STD_LOGIC_VECTOR (15 downto 0);
    signal redist108_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q : STD_LOGIC_VECTOR (15 downto 0);
    signal redist109_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q : STD_LOGIC_VECTOR (15 downto 0);
    signal redist110_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q : STD_LOGIC_VECTOR (6 downto 0);
    signal redist111_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q : STD_LOGIC_VECTOR (31 downto 0);
    signal redist112_expFracR_uid23_fxpToFPTest_p1_of_2_q_1_q : STD_LOGIC_VECTOR (38 downto 0);
    signal redist113_expFracR_uid23_fxpToFPTest_BitSelect_for_a_c_2_q : STD_LOGIC_VECTOR (26 downto 0);
    signal redist114_yE_uid8_fxpToFPTest_p1_of_2_q_1_q : STD_LOGIC_VECTOR (38 downto 0);
    signal redist115_yE_uid8_fxpToFPTest_p1_of_2_q_2_q : STD_LOGIC_VECTOR (38 downto 0);
    signal redist116_yE_uid8_fxpToFPTest_BitSelect_for_a_c_1_q : STD_LOGIC_VECTOR (16 downto 0);
    signal redist117_vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q_1_q : STD_LOGIC_VECTOR (5 downto 0);
    signal redist118_vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist120_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist121_vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist122_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist123_vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q_5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist124_expR_uid37_fxpToFPTest_b_1_q : STD_LOGIC_VECTOR (10 downto 0);
    signal redist125_fracR_uid24_fxpToFPTest_b_1_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist126_inIsZero_uid12_fxpToFPTest_q_5_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist127_signX_uid6_fxpToFPTest_b_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist128_signX_uid6_fxpToFPTest_b_15_q : STD_LOGIC_VECTOR (0 downto 0);

begin


    -- VCC(CONSTANT,1)
    VCC_q <= "1";

    -- signX_uid6_fxpToFPTest(BITSELECT,5)@0
    signX_uid6_fxpToFPTest_b <= STD_LOGIC_VECTOR(a(54 downto 54));

    -- redist127_signX_uid6_fxpToFPTest_b_1(DELAY,930)
    redist127_signX_uid6_fxpToFPTest_b_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => signX_uid6_fxpToFPTest_b, xout => redist127_signX_uid6_fxpToFPTest_b_1_q, clk => clk, aclr => areset );

    -- redist128_signX_uid6_fxpToFPTest_b_15(DELAY,931)
    redist128_signX_uid6_fxpToFPTest_b_15 : dspba_delay
    GENERIC MAP ( width => 1, depth => 14, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist127_signX_uid6_fxpToFPTest_b_1_q, xout => redist128_signX_uid6_fxpToFPTest_b_15_q, clk => clk, aclr => areset );

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
    yE_uid8_fxpToFPTest_BitExpansion_for_b_q <= yE_uid8_fxpToFPTest_UpperBits_for_b_q & redist127_signX_uid6_fxpToFPTest_b_1_q;

    -- yE_uid8_fxpToFPTest_BitSelect_for_b(BITSELECT,94)@1
    yE_uid8_fxpToFPTest_BitSelect_for_b_b <= yE_uid8_fxpToFPTest_BitExpansion_for_b_q(38 downto 0);

    -- xXorSign_uid7_fxpToFPTest(LOGICAL,6)@0 + 1
    xXorSign_uid7_fxpToFPTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((54 downto 1 => signX_uid6_fxpToFPTest_b(0)) & signX_uid6_fxpToFPTest_b));
    xXorSign_uid7_fxpToFPTest_qi <= a xor xXorSign_uid7_fxpToFPTest_b;
    xXorSign_uid7_fxpToFPTest_delay : dspba_delay
    GENERIC MAP ( width => 55, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => xXorSign_uid7_fxpToFPTest_qi, xout => xXorSign_uid7_fxpToFPTest_q, clk => clk, aclr => areset );

    -- yE_uid8_fxpToFPTest_BitExpansion_for_a(BITJOIN,89)@1
    yE_uid8_fxpToFPTest_BitExpansion_for_a_q <= GND_q & xXorSign_uid7_fxpToFPTest_q;

    -- yE_uid8_fxpToFPTest_BitSelect_for_a(BITSELECT,93)@1
    yE_uid8_fxpToFPTest_BitSelect_for_a_b <= yE_uid8_fxpToFPTest_BitExpansion_for_a_q(38 downto 0);
    yE_uid8_fxpToFPTest_BitSelect_for_a_c <= yE_uid8_fxpToFPTest_BitExpansion_for_a_q(55 downto 39);

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
    yE_uid8_fxpToFPTest_p1_of_2_c(0) <= yE_uid8_fxpToFPTest_p1_of_2_o(39);
    yE_uid8_fxpToFPTest_p1_of_2_q <= yE_uid8_fxpToFPTest_p1_of_2_o(38 downto 0);

    -- redist114_yE_uid8_fxpToFPTest_p1_of_2_q_1(DELAY,917)
    redist114_yE_uid8_fxpToFPTest_p1_of_2_q_1 : dspba_delay
    GENERIC MAP ( width => 39, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yE_uid8_fxpToFPTest_p1_of_2_q, xout => redist114_yE_uid8_fxpToFPTest_p1_of_2_q_1_q, clk => clk, aclr => areset );

    -- redist115_yE_uid8_fxpToFPTest_p1_of_2_q_2(DELAY,918)
    redist115_yE_uid8_fxpToFPTest_p1_of_2_q_2 : dspba_delay
    GENERIC MAP ( width => 39, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist114_yE_uid8_fxpToFPTest_p1_of_2_q_1_q, xout => redist115_yE_uid8_fxpToFPTest_p1_of_2_q_2_q, clk => clk, aclr => areset );

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0(BITSELECT,296)@4
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b <= STD_LOGIC_VECTOR(redist115_yE_uid8_fxpToFPTest_p1_of_2_q_2_q(31 downto 0));

    -- yE_uid8_fxpToFPTest_BitSelect_for_b_tessel1_0(BITSELECT,252)
    yE_uid8_fxpToFPTest_BitSelect_for_b_tessel1_0_b <= STD_LOGIC_VECTOR(yE_uid8_fxpToFPTest_UpperBits_for_b_q(54 downto 38));

    -- redist116_yE_uid8_fxpToFPTest_BitSelect_for_a_c_1(DELAY,919)
    redist116_yE_uid8_fxpToFPTest_BitSelect_for_a_c_1 : dspba_delay
    GENERIC MAP ( width => 17, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => yE_uid8_fxpToFPTest_BitSelect_for_a_c, xout => redist116_yE_uid8_fxpToFPTest_BitSelect_for_a_c_1_q, clk => clk, aclr => areset );

    -- yE_uid8_fxpToFPTest_p2_of_2(ADD,96)@2 + 1
    yE_uid8_fxpToFPTest_p2_of_2_cin <= yE_uid8_fxpToFPTest_p1_of_2_c;
    yE_uid8_fxpToFPTest_p2_of_2_a <= STD_LOGIC_VECTOR("0" & redist116_yE_uid8_fxpToFPTest_BitSelect_for_a_c_1_q) & '1';
    yE_uid8_fxpToFPTest_p2_of_2_b <= STD_LOGIC_VECTOR("0" & yE_uid8_fxpToFPTest_BitSelect_for_b_tessel1_0_b) & yE_uid8_fxpToFPTest_p2_of_2_cin(0);
    yE_uid8_fxpToFPTest_p2_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            yE_uid8_fxpToFPTest_p2_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            yE_uid8_fxpToFPTest_p2_of_2_o <= STD_LOGIC_VECTOR(UNSIGNED(yE_uid8_fxpToFPTest_p2_of_2_a) + UNSIGNED(yE_uid8_fxpToFPTest_p2_of_2_b));
        END IF;
    END PROCESS;
    yE_uid8_fxpToFPTest_p2_of_2_q <= yE_uid8_fxpToFPTest_p2_of_2_o(17 downto 1);

    -- yE_uid8_fxpToFPTest_BitJoin_for_q(BITJOIN,97)@3
    yE_uid8_fxpToFPTest_BitJoin_for_q_q <= yE_uid8_fxpToFPTest_p2_of_2_q & redist114_yE_uid8_fxpToFPTest_p1_of_2_q_1_q;

    -- y_uid9_fxpToFPTest(BITSELECT,8)@3
    y_uid9_fxpToFPTest_in <= STD_LOGIC_VECTOR(yE_uid8_fxpToFPTest_BitJoin_for_q_q(54 downto 0));
    y_uid9_fxpToFPTest_b <= STD_LOGIC_VECTOR(y_uid9_fxpToFPTest_in(54 downto 0));

    -- rVStage_uid42_lzcShifterZ1_uid10_fxpToFPTest(BITSELECT,41)@3
    rVStage_uid42_lzcShifterZ1_uid10_fxpToFPTest_b <= y_uid9_fxpToFPTest_b(54 downto 23);

    -- vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,42)@3 + 1
    vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_qi <= "1" WHEN rVStage_uid42_lzcShifterZ1_uid10_fxpToFPTest_b = zs_uid41_lzcShifterZ1_uid10_fxpToFPTest_q ELSE "0";
    vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_qi, xout => vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q, clk => clk, aclr => areset );

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0(MUX,110)@4
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_s <= vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_combproc: PROCESS (vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_s, vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b, zs_uid41_lzcShifterZ1_uid10_fxpToFPTest_q)
    BEGIN
        CASE (vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_s) IS
            WHEN "0" => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
            WHEN "1" => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= zs_uid41_lzcShifterZ1_uid10_fxpToFPTest_q;
            WHEN OTHERS => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist111_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1(DELAY,914)
    redist111_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1 : dspba_delay
    GENERIC MAP ( width => 32, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q, xout => redist111_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel3_0(BITSELECT,324)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel3_0_b <= STD_LOGIC_VECTOR(redist111_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q(31 downto 23));

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0(BITSELECT,306)@4
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b <= STD_LOGIC_VECTOR(redist115_yE_uid8_fxpToFPTest_p1_of_2_q_2_q(22 downto 7));

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0(BITSELECT,300)@3
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b <= STD_LOGIC_VECTOR(yE_uid8_fxpToFPTest_p2_of_2_q(15 downto 0));

    -- redist1_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b_1(DELAY,804)
    redist1_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b_1 : dspba_delay
    GENERIC MAP ( width => 16, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b, xout => redist1_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b_1_q, clk => clk, aclr => areset );

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2(MUX,112)@4
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_s <= vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_combproc: PROCESS (vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_s, redist1_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b_1_q, vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b)
    BEGIN
        CASE (vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_s) IS
            WHEN "0" => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= redist1_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b_1_q;
            WHEN "1" => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b;
            WHEN OTHERS => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist109_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1(DELAY,912)
    redist109_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1 : dspba_delay
    GENERIC MAP ( width => 16, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q, xout => redist109_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_merged_bit_select(BITSELECT,801)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_merged_bit_select_b <= STD_LOGIC_VECTOR(redist109_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q(8 downto 0));
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_merged_bit_select_c <= STD_LOGIC_VECTOR(redist109_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q(15 downto 9));

    -- zs_uid48_lzcShifterZ1_uid10_fxpToFPTest(CONSTANT,47)
    zs_uid48_lzcShifterZ1_uid10_fxpToFPTest_q <= "0000000000000000";

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel1_0(BITSELECT,304)@4
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel1_0_b <= STD_LOGIC_VECTOR(redist115_yE_uid8_fxpToFPTest_p1_of_2_q_2_q(6 downto 0));

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0(BITSELECT,298)@4
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(redist115_yE_uid8_fxpToFPTest_p1_of_2_q_2_q(38 downto 32));

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1(MUX,111)@4
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_s <= vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_combproc: PROCESS (vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_s, vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b, vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel1_0_b)
    BEGIN
        CASE (vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_s) IS
            WHEN "0" => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
            WHEN "1" => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel1_0_b;
            WHEN OTHERS => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q(BITJOIN,113)@4
    vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q <= vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p2_q & vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q & vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q;

    -- rVStage_uid49_lzcShifterZ1_uid10_fxpToFPTest(BITSELECT,48)@4
    rVStage_uid49_lzcShifterZ1_uid10_fxpToFPTest_b <= vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q(54 downto 39);

    -- vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,49)@4 + 1
    vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_qi <= "1" WHEN rVStage_uid49_lzcShifterZ1_uid10_fxpToFPTest_b = zs_uid48_lzcShifterZ1_uid10_fxpToFPTest_q ELSE "0";
    vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_qi, xout => vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q, clk => clk, aclr => areset );

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3(MUX,119)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_s <= vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_combproc: PROCESS (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_s, vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_merged_bit_select_b, vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel3_0_b)
    BEGIN
        CASE (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_s) IS
            WHEN "0" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_merged_bit_select_b;
            WHEN "1" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel3_0_b;
            WHEN OTHERS => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist105_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1(DELAY,908)
    redist105_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1 : dspba_delay
    GENERIC MAP ( width => 9, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q, xout => redist105_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0(BITSELECT,338)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b <= STD_LOGIC_VECTOR(redist105_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q(0 downto 0));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0(BITSELECT,342)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b <= STD_LOGIC_VECTOR(redist105_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q(8 downto 8));

    -- zs_uid55_lzcShifterZ1_uid10_fxpToFPTest(CONSTANT,54)
    zs_uid55_lzcShifterZ1_uid10_fxpToFPTest_q <= "00000000";

    -- redist110_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1(DELAY,913)
    redist110_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1 : dspba_delay
    GENERIC MAP ( width => 7, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q, xout => redist110_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4(MUX,120)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_s <= vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_combproc: PROCESS (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_s, vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_merged_bit_select_c, redist110_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q)
    BEGIN
        CASE (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_s) IS
            WHEN "0" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_merged_bit_select_c;
            WHEN "1" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= redist110_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q;
            WHEN OTHERS => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0(BITSELECT,322)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b <= STD_LOGIC_VECTOR(redist111_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q(22 downto 16));

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2(MUX,118)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_s <= vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_combproc: PROCESS (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_s, redist110_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q, vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b)
    BEGIN
        CASE (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_s) IS
            WHEN "0" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= redist110_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q;
            WHEN "1" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel2_0_b;
            WHEN OTHERS => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0(BITSELECT,308)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b <= STD_LOGIC_VECTOR(redist111_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q(15 downto 0));

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0(BITSELECT,310)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(redist111_vStagei_uid47_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q(31 downto 16));

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1(MUX,117)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_s <= vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_combproc: PROCESS (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_s, vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b, vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b)
    BEGIN
        CASE (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_s) IS
            WHEN "0" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
            WHEN "1" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
            WHEN OTHERS => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0(MUX,116)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_s <= vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_combproc: PROCESS (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_s, vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b, zs_uid48_lzcShifterZ1_uid10_fxpToFPTest_q)
    BEGIN
        CASE (vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_s) IS
            WHEN "0" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
            WHEN "1" => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= zs_uid48_lzcShifterZ1_uid10_fxpToFPTest_q;
            WHEN OTHERS => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q(BITJOIN,121)@5
    vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q & vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q & vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q & vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q & vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q;

    -- rVStage_uid56_lzcShifterZ1_uid10_fxpToFPTest(BITSELECT,55)@5
    rVStage_uid56_lzcShifterZ1_uid10_fxpToFPTest_b <= vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q(54 downto 47);

    -- vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,56)@5 + 1
    vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_qi <= "1" WHEN rVStage_uid56_lzcShifterZ1_uid10_fxpToFPTest_b = zs_uid55_lzcShifterZ1_uid10_fxpToFPTest_q ELSE "0";
    vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_qi, xout => vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q, clk => clk, aclr => areset );

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7(MUX,131)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_s, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist96_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1(DELAY,899)
    redist96_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q, xout => redist96_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0(BITSELECT,340)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b <= STD_LOGIC_VECTOR(redist105_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q(7 downto 1));

    -- redist104_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1(DELAY,907)
    redist104_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1 : dspba_delay
    GENERIC MAP ( width => 7, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q, xout => redist104_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8(MUX,132)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_s, redist104_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= redist104_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0(BITSELECT,396)@6
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b <= STD_LOGIC_VECTOR(vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q(3 downto 3));

    -- redist0_vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b_1(DELAY,803)
    redist0_vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b, xout => redist0_vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b_1_q, clk => clk, aclr => areset );

    -- zs_uid62_lzcShifterZ1_uid10_fxpToFPTest(CONSTANT,61)
    zs_uid62_lzcShifterZ1_uid10_fxpToFPTest_q <= "0000";

    -- redist106_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1(DELAY,909)
    redist106_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1 : dspba_delay
    GENERIC MAP ( width => 7, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q, xout => redist106_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6(MUX,130)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_s, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b, redist106_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= redist106_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist107_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1(DELAY,910)
    redist107_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1 : dspba_delay
    GENERIC MAP ( width => 16, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q, xout => redist107_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel5_0(BITSELECT,356)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel5_0_b <= STD_LOGIC_VECTOR(redist107_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q(15 downto 15));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5(MUX,129)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_s, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel5_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel5_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel4_0(BITSELECT,354)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel4_0_b <= STD_LOGIC_VECTOR(redist107_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q(14 downto 8));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4(MUX,128)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_s, redist106_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel4_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= redist106_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel4_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0(BITSELECT,332)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b <= STD_LOGIC_VECTOR(redist107_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q(7 downto 0));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0(BITSELECT,334)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b <= STD_LOGIC_VECTOR(redist107_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q(15 downto 8));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3(MUX,127)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_s, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist108_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1(DELAY,911)
    redist108_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1 : dspba_delay
    GENERIC MAP ( width => 16, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q, xout => redist108_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0(BITSELECT,330)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(redist108_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q(15 downto 8));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2(MUX,126)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_s, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0(BITSELECT,328)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b <= STD_LOGIC_VECTOR(redist108_vStagei_uid54_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q(7 downto 0));

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1(MUX,125)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_s, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0(MUX,124)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_s <= vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_combproc: PROCESS (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_s, vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b, zs_uid55_lzcShifterZ1_uid10_fxpToFPTest_q)
    BEGIN
        CASE (vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_s) IS
            WHEN "0" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
            WHEN "1" => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= zs_uid55_lzcShifterZ1_uid10_fxpToFPTest_q;
            WHEN OTHERS => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q(BITJOIN,133)@6
    vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q & vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q;

    -- rVStage_uid63_lzcShifterZ1_uid10_fxpToFPTest(BITSELECT,62)@6
    rVStage_uid63_lzcShifterZ1_uid10_fxpToFPTest_b <= vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q(54 downto 51);

    -- vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,63)@6
    vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q <= "1" WHEN rVStage_uid63_lzcShifterZ1_uid10_fxpToFPTest_b = zs_uid62_lzcShifterZ1_uid10_fxpToFPTest_q ELSE "0";

    -- redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1(DELAY,922)
    redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q, xout => redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16(MUX,152)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_s, redist0_vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b_1_q, redist96_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1_q)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= redist0_vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b_1_q;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= redist96_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1_q;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0(BITSELECT,394)@6
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b <= STD_LOGIC_VECTOR(vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q(2 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0(BITSELECT,398)@6
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b <= STD_LOGIC_VECTOR(vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p8_q(6 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17(MUX,153)@6 + 1
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_s) IS
                WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b;
                WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b;
                WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel34_0(BITSELECT,504)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel34_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q(1 downto 1));

    -- zs_uid69_lzcShifterZ1_uid10_fxpToFPTest(CONSTANT,68)
    zs_uid69_lzcShifterZ1_uid10_fxpToFPTest_q <= "00";

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel15_0(BITSELECT,430)@6
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel15_0_b <= STD_LOGIC_VECTOR(vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q(6 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15(MUX,151)@6 + 1
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_s <= vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_s) IS
                WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b;
                WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel15_0_b;
                WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist97_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1(DELAY,900)
    redist97_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1 : dspba_delay
    GENERIC MAP ( width => 7, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q, xout => redist97_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0(BITSELECT,388)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b <= STD_LOGIC_VECTOR(redist97_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1_q(3 downto 3));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14(MUX,150)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_s, redist96_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1_q, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_q <= redist96_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p7_q_1_q;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0(BITSELECT,386)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b <= STD_LOGIC_VECTOR(redist97_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1_q(2 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0(BITSELECT,390)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b <= STD_LOGIC_VECTOR(redist97_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p6_q_1_q(6 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13(MUX,149)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist98_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1(DELAY,901)
    redist98_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q, xout => redist98_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12(MUX,148)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b, redist98_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1_q)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_q <= redist98_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1_q;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist99_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1(DELAY,902)
    redist99_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1 : dspba_delay
    GENERIC MAP ( width => 7, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q, xout => redist99_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0(BITSELECT,382)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b <= STD_LOGIC_VECTOR(redist99_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q(6 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11(MUX,147)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel10_0(BITSELECT,420)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel10_0_b <= STD_LOGIC_VECTOR(redist99_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q(3 downto 3));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10(MUX,146)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_s, redist98_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1_q, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel10_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_q <= redist98_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p5_q_1_q;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel10_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0(BITSELECT,418)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0_b <= STD_LOGIC_VECTOR(redist99_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q(2 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9(MUX,145)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel9_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist100_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1(DELAY,903)
    redist100_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1 : dspba_delay
    GENERIC MAP ( width => 8, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q, xout => redist100_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0(BITSELECT,378)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b <= STD_LOGIC_VECTOR(redist100_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q(7 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0(BITSELECT,380)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0_b <= STD_LOGIC_VECTOR(redist99_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p4_q_1_q(3 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8(MUX,144)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0(BITSELECT,376)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b <= STD_LOGIC_VECTOR(redist100_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p3_q_1_q(3 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7(MUX,143)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist101_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1(DELAY,904)
    redist101_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1 : dspba_delay
    GENERIC MAP ( width => 8, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q, xout => redist101_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0(BITSELECT,374)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b <= STD_LOGIC_VECTOR(redist101_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q(7 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6(MUX,142)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0(BITSELECT,372)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b <= STD_LOGIC_VECTOR(redist101_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p2_q_1_q(3 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5(MUX,141)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist102_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1(DELAY,905)
    redist102_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1 : dspba_delay
    GENERIC MAP ( width => 8, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q, xout => redist102_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0(BITSELECT,370)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b <= STD_LOGIC_VECTOR(redist102_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q(7 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4(MUX,140)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0(BITSELECT,368)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b <= STD_LOGIC_VECTOR(redist102_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p1_q_1_q(3 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3(MUX,139)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist103_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1(DELAY,906)
    redist103_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1 : dspba_delay
    GENERIC MAP ( width => 8, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q, xout => redist103_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q, clk => clk, aclr => areset );

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0(BITSELECT,366)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(redist103_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q(7 downto 4));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2(MUX,138)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0(BITSELECT,364)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b <= STD_LOGIC_VECTOR(redist103_vStagei_uid61_lzcShifterZ1_uid10_fxpToFPTest_p0_q_1_q(3 downto 0));

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1(MUX,137)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0(MUX,136)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_s <= redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_combproc: PROCESS (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_s, vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b, zs_uid62_lzcShifterZ1_uid10_fxpToFPTest_q)
    BEGIN
        CASE (vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_s) IS
            WHEN "0" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b;
            WHEN "1" => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= zs_uid62_lzcShifterZ1_uid10_fxpToFPTest_q;
            WHEN OTHERS => vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q(BITJOIN,154)@7
    vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p16_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p14_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p12_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_q & vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_q;

    -- rVStage_uid70_lzcShifterZ1_uid10_fxpToFPTest(BITSELECT,69)@7
    rVStage_uid70_lzcShifterZ1_uid10_fxpToFPTest_b <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q(54 downto 53);

    -- vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,70)@7
    vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q <= "1" WHEN rVStage_uid70_lzcShifterZ1_uid10_fxpToFPTest_b = zs_uid69_lzcShifterZ1_uid10_fxpToFPTest_q ELSE "0";

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34(MUX,191)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel33_0(BITSELECT,502)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel33_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q(0 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel35_0(BITSELECT,506)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel35_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p17_q(2 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35(MUX,192)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0(BITSELECT,498)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q(2 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33(MUX,190)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel30_0(BITSELECT,496)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel30_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q(1 downto 1));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32(MUX,189)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel29_0(BITSELECT,494)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel29_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p15_q(0 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31(MUX,188)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30(MUX,187)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel27_0(BITSELECT,490)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel27_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q(2 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29(MUX,186)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel26_0(BITSELECT,488)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel26_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q(1 downto 1));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28(MUX,185)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel25_0(BITSELECT,486)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel25_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p13_q(0 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27(MUX,184)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26(MUX,183)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel23_0(BITSELECT,482)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel23_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q(2 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25(MUX,182)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel22_0(BITSELECT,480)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel22_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q(1 downto 1));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24(MUX,181)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel21_0(BITSELECT,478)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel21_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p11_q(0 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23(MUX,180)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22(MUX,179)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0(BITSELECT,474)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q(2 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21(MUX,178)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel20_0(BITSELECT,548)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel20_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q(1 downto 1));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20(MUX,177)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q <= vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p10_q;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel20_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel19_0(BITSELECT,546)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel19_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q(0 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19(MUX,176)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_b_tessel19_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0(BITSELECT,470)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0(BITSELECT,472)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p9_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18(MUX,175)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0(BITSELECT,468)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p8_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17(MUX,174)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0(BITSELECT,466)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16(MUX,173)@7 + 1
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_s <= vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_s) IS
                WHEN "0" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b;
                WHEN "1" => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b;
                WHEN OTHERS => vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0(BITSELECT,464)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p7_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15(MUX,172)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0(BITSELECT,462)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14(MUX,171)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0(BITSELECT,460)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p6_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13(MUX,170)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0(BITSELECT,458)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12(MUX,169)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0(BITSELECT,456)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p5_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11(MUX,168)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0(BITSELECT,454)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10(MUX,167)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0(BITSELECT,452)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p4_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9(MUX,166)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0(BITSELECT,450)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8(MUX,165)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0(BITSELECT,448)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p3_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7(MUX,164)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0(BITSELECT,446)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6(MUX,163)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0(BITSELECT,444)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p2_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5(MUX,162)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0(BITSELECT,442)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4(MUX,161)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0(BITSELECT,440)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p1_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3(MUX,160)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0(BITSELECT,438)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_q(3 downto 2));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2(MUX,159)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0(BITSELECT,436)@7
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b <= STD_LOGIC_VECTOR(vStagei_uid68_lzcShifterZ1_uid10_fxpToFPTest_p0_q(1 downto 0));

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1(MUX,158)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0(MUX,157)@7 + 1
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

    -- vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q(BITJOIN,193)@8
    vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_q & vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_q;

    -- rVStage_uid77_lzcShifterZ1_uid10_fxpToFPTest(BITSELECT,76)@8
    rVStage_uid77_lzcShifterZ1_uid10_fxpToFPTest_b <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q(54 downto 54);

    -- vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,77)@8
    vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q <= "1" WHEN rVStage_uid77_lzcShifterZ1_uid10_fxpToFPTest_b = GND_q ELSE "0";

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54(MUX,250)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p35_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist2_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q_3(DELAY,805)
    redist2_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q, xout => redist2_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53(MUX,249)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p34_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist3_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q_3(DELAY,806)
    redist3_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q, xout => redist3_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52(MUX,248)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p33_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist4_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q_3(DELAY,807)
    redist4_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q, xout => redist4_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51(MUX,247)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p32_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist5_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q_3(DELAY,808)
    redist5_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q, xout => redist5_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50(MUX,246)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p31_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist6_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q_3(DELAY,809)
    redist6_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q, xout => redist6_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49(MUX,245)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p30_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist7_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q_3(DELAY,810)
    redist7_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q, xout => redist7_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48(MUX,244)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p29_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist8_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q_3(DELAY,811)
    redist8_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q, xout => redist8_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47(MUX,243)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p28_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist9_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q_3(DELAY,812)
    redist9_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q, xout => redist9_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46(MUX,242)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p27_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist10_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q_3(DELAY,813)
    redist10_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q, xout => redist10_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45(MUX,241)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p26_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist11_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q_3(DELAY,814)
    redist11_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q, xout => redist11_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44(MUX,240)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p25_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist12_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q_3(DELAY,815)
    redist12_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q, xout => redist12_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43(MUX,239)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p24_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist13_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q_3(DELAY,816)
    redist13_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q, xout => redist13_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42(MUX,238)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p23_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist14_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q_3(DELAY,817)
    redist14_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q, xout => redist14_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41(MUX,237)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p22_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist15_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q_3(DELAY,818)
    redist15_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q, xout => redist15_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40(MUX,236)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p21_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist16_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q_3(DELAY,819)
    redist16_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q, xout => redist16_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39(MUX,235)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p20_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist17_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_3(DELAY,820)
    redist17_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q, xout => redist17_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel37_0(BITSELECT,654)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel37_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38(MUX,234)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q <= vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p19_q;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel37_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist19_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_3(DELAY,822)
    redist19_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q, xout => redist19_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel36_0(BITSELECT,652)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel36_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p18_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37(MUX,233)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel37_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel36_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist21_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_3(DELAY,824)
    redist21_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q, xout => redist21_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel35_0(BITSELECT,650)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel35_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36(MUX,232)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel36_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel35_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist23_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_3(DELAY,826)
    redist23_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q, xout => redist23_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel34_0(BITSELECT,648)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel34_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p17_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35(MUX,231)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel35_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel34_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist25_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_3(DELAY,828)
    redist25_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q, xout => redist25_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel33_0(BITSELECT,646)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel33_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34(MUX,230)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel34_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel33_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist27_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_3(DELAY,830)
    redist27_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q, xout => redist27_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel32_0(BITSELECT,644)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel32_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p16_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33(MUX,229)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel33_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel32_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist29_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_3(DELAY,832)
    redist29_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q, xout => redist29_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0(BITSELECT,642)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32(MUX,228)@8 + 1
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_s <= vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_s) IS
                WHEN "0" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel32_0_b;
                WHEN "1" => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel31_0_b;
                WHEN OTHERS => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist31_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_3(DELAY,834)
    redist31_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q, xout => redist31_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel30_0(BITSELECT,640)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel30_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p15_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31(MUX,227)@8 + 1
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

    -- redist33_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_3(DELAY,836)
    redist33_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q, xout => redist33_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel29_0(BITSELECT,638)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel29_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30(MUX,226)@8 + 1
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

    -- redist35_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_3(DELAY,838)
    redist35_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q, xout => redist35_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel28_0(BITSELECT,636)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel28_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p14_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29(MUX,225)@8 + 1
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

    -- redist37_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_3(DELAY,840)
    redist37_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q, xout => redist37_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel27_0(BITSELECT,634)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel27_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28(MUX,224)@8 + 1
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

    -- redist39_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_3(DELAY,842)
    redist39_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q, xout => redist39_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel26_0(BITSELECT,632)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel26_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p13_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27(MUX,223)@8 + 1
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

    -- redist41_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_3(DELAY,844)
    redist41_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q, xout => redist41_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel25_0(BITSELECT,630)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel25_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26(MUX,222)@8 + 1
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

    -- redist43_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_3(DELAY,846)
    redist43_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q, xout => redist43_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel24_0(BITSELECT,628)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel24_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p12_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25(MUX,221)@8 + 1
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

    -- redist45_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_3(DELAY,848)
    redist45_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q, xout => redist45_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel23_0(BITSELECT,626)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel23_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24(MUX,220)@8 + 1
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

    -- redist47_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_3(DELAY,850)
    redist47_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q, xout => redist47_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel22_0(BITSELECT,624)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel22_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p11_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23(MUX,219)@8 + 1
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

    -- redist49_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_3(DELAY,852)
    redist49_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q, xout => redist49_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel21_0(BITSELECT,622)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel21_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22(MUX,218)@8 + 1
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

    -- redist51_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_3(DELAY,854)
    redist51_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q, xout => redist51_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel20_0(BITSELECT,620)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel20_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p10_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21(MUX,217)@8 + 1
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

    -- redist53_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_3(DELAY,856)
    redist53_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q, xout => redist53_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0(BITSELECT,618)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel19_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20(MUX,216)@8 + 1
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

    -- redist55_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_3(DELAY,858)
    redist55_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q, xout => redist55_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0(BITSELECT,616)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel18_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p9_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19(MUX,215)@8 + 1
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

    -- redist57_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_3(DELAY,860)
    redist57_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q, xout => redist57_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0(BITSELECT,614)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel17_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18(MUX,214)@8 + 1
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

    -- redist59_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_3(DELAY,862)
    redist59_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q, xout => redist59_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0(BITSELECT,612)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel16_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p8_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17(MUX,213)@8 + 1
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

    -- redist61_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_3(DELAY,864)
    redist61_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q, xout => redist61_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0(BITSELECT,610)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel15_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16(MUX,212)@8 + 1
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

    -- redist63_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_3(DELAY,866)
    redist63_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q, xout => redist63_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0(BITSELECT,608)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel14_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p7_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15(MUX,211)@8 + 1
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

    -- redist65_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_3(DELAY,868)
    redist65_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q, xout => redist65_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0(BITSELECT,606)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel13_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14(MUX,210)@8 + 1
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

    -- redist67_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_3(DELAY,870)
    redist67_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q, xout => redist67_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0(BITSELECT,604)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel12_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p6_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13(MUX,209)@8 + 1
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

    -- redist69_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_3(DELAY,872)
    redist69_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q, xout => redist69_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0(BITSELECT,602)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel11_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12(MUX,208)@8 + 1
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

    -- redist71_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_3(DELAY,874)
    redist71_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q, xout => redist71_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0(BITSELECT,600)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel10_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p5_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11(MUX,207)@8 + 1
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

    -- redist73_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_3(DELAY,876)
    redist73_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q, xout => redist73_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0(BITSELECT,598)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel9_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10(MUX,206)@8 + 1
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

    -- redist75_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_3(DELAY,878)
    redist75_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q, xout => redist75_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0(BITSELECT,596)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel8_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p4_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9(MUX,205)@8 + 1
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

    -- redist77_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_3(DELAY,880)
    redist77_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q, xout => redist77_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0(BITSELECT,594)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel7_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8(MUX,204)@8 + 1
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

    -- redist79_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_3(DELAY,882)
    redist79_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q, xout => redist79_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0(BITSELECT,592)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel6_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p3_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7(MUX,203)@8 + 1
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

    -- redist81_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_3(DELAY,884)
    redist81_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q, xout => redist81_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0(BITSELECT,590)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel5_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6(MUX,202)@8 + 1
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

    -- redist83_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_3(DELAY,886)
    redist83_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q, xout => redist83_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0(BITSELECT,588)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel4_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p2_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5(MUX,201)@8 + 1
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

    -- redist85_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_3(DELAY,888)
    redist85_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q, xout => redist85_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0(BITSELECT,586)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel3_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4(MUX,200)@8 + 1
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

    -- redist87_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_3(DELAY,890)
    redist87_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q, xout => redist87_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0(BITSELECT,584)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel2_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p1_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3(MUX,199)@8 + 1
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

    -- redist89_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_3(DELAY,892)
    redist89_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q, xout => redist89_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0(BITSELECT,582)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel1_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_q(1 downto 1));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2(MUX,198)@8 + 1
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

    -- redist91_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_3(DELAY,894)
    redist91_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q, xout => redist91_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0(BITSELECT,580)@8
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitSelect_for_a_tessel0_0_b <= STD_LOGIC_VECTOR(vStagei_uid75_lzcShifterZ1_uid10_fxpToFPTest_p0_q(0 downto 0));

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1(MUX,197)@8 + 1
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

    -- redist93_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_3(DELAY,896)
    redist93_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q, xout => redist93_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0(MUX,196)@8 + 1
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

    -- redist95_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q_3(DELAY,898)
    redist95_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q, xout => redist95_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q_3_q, clk => clk, aclr => areset );

    -- vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q(BITJOIN,251)@11
    vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q <= redist2_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p54_q_3_q & redist3_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p53_q_3_q & redist4_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p52_q_3_q & redist5_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p51_q_3_q & redist6_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p50_q_3_q & redist7_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p49_q_3_q & redist8_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p48_q_3_q & redist9_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p47_q_3_q & redist10_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p46_q_3_q & redist11_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p45_q_3_q & redist12_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p44_q_3_q & redist13_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p43_q_3_q & redist14_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p42_q_3_q & redist15_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p41_q_3_q & redist16_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p40_q_3_q & redist17_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_3_q & redist19_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_3_q & redist21_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_3_q & redist23_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_3_q & redist25_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_3_q & redist27_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_3_q & redist29_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_3_q & redist31_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_3_q & redist33_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_3_q & redist35_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_3_q & redist37_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_3_q & redist39_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_3_q & redist41_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_3_q & redist43_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_3_q & redist45_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_3_q & redist47_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_3_q & redist49_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_3_q & redist51_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_3_q & redist53_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_3_q & redist55_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_3_q & redist57_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_3_q & redist59_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_3_q & redist61_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_3_q & redist63_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_3_q & redist65_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_3_q & redist67_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_3_q & redist69_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_3_q & redist71_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_3_q & redist73_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_3_q & redist75_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_3_q & redist77_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_3_q & redist79_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_3_q & redist81_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_3_q & redist83_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_3_q & redist85_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_3_q & redist87_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_3_q & redist89_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_3_q & redist91_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_3_q & redist93_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_3_q & redist95_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p0_q_3_q;

    -- fracRnd_uid15_fxpToFPTest_merged_bit_select(BITSELECT,802)@11
    fracRnd_uid15_fxpToFPTest_merged_bit_select_in <= vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_BitJoin_for_q_q(53 downto 0);
    fracRnd_uid15_fxpToFPTest_merged_bit_select_b <= fracRnd_uid15_fxpToFPTest_merged_bit_select_in(53 downto 1);
    fracRnd_uid15_fxpToFPTest_merged_bit_select_c <= fracRnd_uid15_fxpToFPTest_merged_bit_select_in(0 downto 0);

    -- nr_uid20_fxpToFPTest(LOGICAL,19)@11
    nr_uid20_fxpToFPTest_q <= not (l_uid17_fxpToFPTest_merged_bit_select_c);

    -- maxCount_uid11_fxpToFPTest(CONSTANT,10)
    maxCount_uid11_fxpToFPTest_q <= "110111";

    -- redist123_vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q_5(DELAY,926)
    redist123_vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q_5 : dspba_delay
    GENERIC MAP ( width => 1, depth => 4, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q, xout => redist123_vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q_5_q, clk => clk, aclr => areset );

    -- redist122_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_4(DELAY,925)
    redist122_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 3, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q, xout => redist122_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_4_q, clk => clk, aclr => areset );

    -- redist121_vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q_3(DELAY,924)
    redist121_vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q, xout => redist121_vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q_3_q, clk => clk, aclr => areset );

    -- redist120_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_2(DELAY,923)
    redist120_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist119_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_1_q, xout => redist120_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_2_q, clk => clk, aclr => areset );

    -- redist118_vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q_1(DELAY,921)
    redist118_vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q_1 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q, xout => redist118_vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q_1_q, clk => clk, aclr => areset );

    -- vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest(BITJOIN,82)@8
    vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q <= redist123_vCount_uid43_lzcShifterZ1_uid10_fxpToFPTest_q_5_q & redist122_vCount_uid50_lzcShifterZ1_uid10_fxpToFPTest_q_4_q & redist121_vCount_uid57_lzcShifterZ1_uid10_fxpToFPTest_q_3_q & redist120_vCount_uid64_lzcShifterZ1_uid10_fxpToFPTest_q_2_q & redist118_vCount_uid71_lzcShifterZ1_uid10_fxpToFPTest_q_1_q & vCount_uid78_lzcShifterZ1_uid10_fxpToFPTest_q;

    -- redist117_vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q_1(DELAY,920)
    redist117_vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q_1 : dspba_delay
    GENERIC MAP ( width => 6, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q, xout => redist117_vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q_1_q, clk => clk, aclr => areset );

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
                WHEN "0" => vCountFinal_uid87_lzcShifterZ1_uid10_fxpToFPTest_q <= redist117_vCount_uid83_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
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

    -- l_uid17_fxpToFPTest_merged_bit_select(BITSELECT,800)@11
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
    expFracR_uid23_fxpToFPTest_BitSelect_for_b_b <= expFracR_uid23_fxpToFPTest_BitExpansion_for_b_q(38 downto 0);

    -- redist18_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_4(DELAY,821)
    redist18_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist17_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_3_q, xout => redist18_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_4_q, clk => clk, aclr => areset );

    -- redist20_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_4(DELAY,823)
    redist20_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist19_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_3_q, xout => redist20_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_4_q, clk => clk, aclr => areset );

    -- redist22_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_4(DELAY,825)
    redist22_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist21_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_3_q, xout => redist22_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_4_q, clk => clk, aclr => areset );

    -- redist24_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_4(DELAY,827)
    redist24_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist23_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_3_q, xout => redist24_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_4_q, clk => clk, aclr => areset );

    -- redist26_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_4(DELAY,829)
    redist26_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist25_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_3_q, xout => redist26_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_4_q, clk => clk, aclr => areset );

    -- redist28_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_4(DELAY,831)
    redist28_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist27_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_3_q, xout => redist28_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_4_q, clk => clk, aclr => areset );

    -- redist30_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_4(DELAY,833)
    redist30_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist29_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_3_q, xout => redist30_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_4_q, clk => clk, aclr => areset );

    -- redist32_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_4(DELAY,835)
    redist32_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist31_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_3_q, xout => redist32_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_4_q, clk => clk, aclr => areset );

    -- redist34_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_4(DELAY,837)
    redist34_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist33_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_3_q, xout => redist34_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_4_q, clk => clk, aclr => areset );

    -- redist36_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_4(DELAY,839)
    redist36_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist35_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_3_q, xout => redist36_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_4_q, clk => clk, aclr => areset );

    -- redist38_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_4(DELAY,841)
    redist38_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist37_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_3_q, xout => redist38_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_4_q, clk => clk, aclr => areset );

    -- redist40_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_4(DELAY,843)
    redist40_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist39_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_3_q, xout => redist40_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_4_q, clk => clk, aclr => areset );

    -- redist42_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_4(DELAY,845)
    redist42_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist41_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_3_q, xout => redist42_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_4_q, clk => clk, aclr => areset );

    -- redist44_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_4(DELAY,847)
    redist44_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist43_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_3_q, xout => redist44_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_4_q, clk => clk, aclr => areset );

    -- redist46_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_4(DELAY,849)
    redist46_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist45_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_3_q, xout => redist46_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_4_q, clk => clk, aclr => areset );

    -- redist48_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_4(DELAY,851)
    redist48_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist47_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_3_q, xout => redist48_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_4_q, clk => clk, aclr => areset );

    -- redist50_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_4(DELAY,853)
    redist50_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist49_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_3_q, xout => redist50_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_4_q, clk => clk, aclr => areset );

    -- redist52_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_4(DELAY,855)
    redist52_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist51_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_3_q, xout => redist52_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_4_q, clk => clk, aclr => areset );

    -- redist54_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_4(DELAY,857)
    redist54_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist53_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_3_q, xout => redist54_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_4_q, clk => clk, aclr => areset );

    -- redist56_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_4(DELAY,859)
    redist56_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist55_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_3_q, xout => redist56_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_4_q, clk => clk, aclr => areset );

    -- redist58_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_4(DELAY,861)
    redist58_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist57_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_3_q, xout => redist58_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_4_q, clk => clk, aclr => areset );

    -- redist60_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_4(DELAY,863)
    redist60_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist59_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_3_q, xout => redist60_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_4_q, clk => clk, aclr => areset );

    -- redist62_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_4(DELAY,865)
    redist62_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist61_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_3_q, xout => redist62_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_4_q, clk => clk, aclr => areset );

    -- redist64_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_4(DELAY,867)
    redist64_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist63_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_3_q, xout => redist64_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_4_q, clk => clk, aclr => areset );

    -- redist66_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_4(DELAY,869)
    redist66_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist65_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_3_q, xout => redist66_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_4_q, clk => clk, aclr => areset );

    -- redist68_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_4(DELAY,871)
    redist68_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist67_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_3_q, xout => redist68_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_4_q, clk => clk, aclr => areset );

    -- redist70_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_4(DELAY,873)
    redist70_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist69_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_3_q, xout => redist70_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_4_q, clk => clk, aclr => areset );

    -- redist72_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_4(DELAY,875)
    redist72_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist71_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_3_q, xout => redist72_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_4_q, clk => clk, aclr => areset );

    -- redist74_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_4(DELAY,877)
    redist74_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist73_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_3_q, xout => redist74_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_4_q, clk => clk, aclr => areset );

    -- redist76_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_4(DELAY,879)
    redist76_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist75_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_3_q, xout => redist76_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_4_q, clk => clk, aclr => areset );

    -- redist78_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_4(DELAY,881)
    redist78_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist77_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_3_q, xout => redist78_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_4_q, clk => clk, aclr => areset );

    -- redist80_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_4(DELAY,883)
    redist80_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist79_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_3_q, xout => redist80_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_4_q, clk => clk, aclr => areset );

    -- redist82_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_4(DELAY,885)
    redist82_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist81_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_3_q, xout => redist82_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_4_q, clk => clk, aclr => areset );

    -- redist84_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_4(DELAY,887)
    redist84_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist83_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_3_q, xout => redist84_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_4_q, clk => clk, aclr => areset );

    -- redist86_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_4(DELAY,889)
    redist86_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist85_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_3_q, xout => redist86_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_4_q, clk => clk, aclr => areset );

    -- redist88_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_4(DELAY,891)
    redist88_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist87_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_3_q, xout => redist88_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_4_q, clk => clk, aclr => areset );

    -- redist90_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_4(DELAY,893)
    redist90_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist89_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_3_q, xout => redist90_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_4_q, clk => clk, aclr => areset );

    -- redist92_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_4(DELAY,895)
    redist92_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist91_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_3_q, xout => redist92_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_4_q, clk => clk, aclr => areset );

    -- redist94_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_4(DELAY,897)
    redist94_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist93_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_3_q, xout => redist94_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_4_q, clk => clk, aclr => areset );

    -- expFracR_uid23_fxpToFPTest_BitSelect_for_a_BitJoin_for_b(BITJOIN,293)@12
    expFracR_uid23_fxpToFPTest_BitSelect_for_a_BitJoin_for_b_q <= redist18_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p39_q_4_q & redist20_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p38_q_4_q & redist22_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p37_q_4_q & redist24_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p36_q_4_q & redist26_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p35_q_4_q & redist28_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p34_q_4_q & redist30_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p33_q_4_q & redist32_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p32_q_4_q & redist34_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p31_q_4_q & redist36_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p30_q_4_q & redist38_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p29_q_4_q & redist40_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p28_q_4_q & redist42_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p27_q_4_q & redist44_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p26_q_4_q & redist46_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p25_q_4_q & redist48_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p24_q_4_q & redist50_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p23_q_4_q & redist52_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p22_q_4_q & redist54_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p21_q_4_q & redist56_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p20_q_4_q & redist58_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p19_q_4_q & redist60_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p18_q_4_q & redist62_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p17_q_4_q & redist64_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p16_q_4_q & redist66_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p15_q_4_q & redist68_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p14_q_4_q & redist70_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p13_q_4_q & redist72_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p12_q_4_q & redist74_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p11_q_4_q & redist76_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p10_q_4_q & redist78_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p9_q_4_q & redist80_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p8_q_4_q & redist82_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p7_q_4_q & redist84_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p6_q_4_q & redist86_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p5_q_4_q & redist88_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p4_q_4_q & redist90_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p3_q_4_q & redist92_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p2_q_4_q & redist94_vStagei_uid82_lzcShifterZ1_uid10_fxpToFPTest_p1_q_4_q;

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
    expFracR_uid23_fxpToFPTest_p1_of_2_c(0) <= expFracR_uid23_fxpToFPTest_p1_of_2_o(39);
    expFracR_uid23_fxpToFPTest_p1_of_2_q <= expFracR_uid23_fxpToFPTest_p1_of_2_o(38 downto 0);

    -- expFracR_uid23_fxpToFPTest_BitSelect_for_b_tessel1_0(BITSELECT,294)
    expFracR_uid23_fxpToFPTest_BitSelect_for_b_tessel1_0_b <= STD_LOGIC_VECTOR(expFracR_uid23_fxpToFPTest_UpperBits_for_b_q(64 downto 38));

    -- expFracR_uid23_fxpToFPTest_BitExpansion_for_a(BITJOIN,98)@11
    expFracR_uid23_fxpToFPTest_BitExpansion_for_a_q <= l_uid17_fxpToFPTest_merged_bit_select_d & expFracRnd_uid16_fxpToFPTest_q;

    -- expFracR_uid23_fxpToFPTest_BitSelect_for_a(BITSELECT,103)@11
    expFracR_uid23_fxpToFPTest_BitSelect_for_a_c <= STD_LOGIC_VECTOR(expFracR_uid23_fxpToFPTest_BitExpansion_for_a_q(65 downto 39));

    -- redist113_expFracR_uid23_fxpToFPTest_BitSelect_for_a_c_2(DELAY,916)
    redist113_expFracR_uid23_fxpToFPTest_BitSelect_for_a_c_2 : dspba_delay
    GENERIC MAP ( width => 27, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => expFracR_uid23_fxpToFPTest_BitSelect_for_a_c, xout => redist113_expFracR_uid23_fxpToFPTest_BitSelect_for_a_c_2_q, clk => clk, aclr => areset );

    -- expFracR_uid23_fxpToFPTest_p2_of_2(ADD,106)@13 + 1
    expFracR_uid23_fxpToFPTest_p2_of_2_cin <= expFracR_uid23_fxpToFPTest_p1_of_2_c;
    expFracR_uid23_fxpToFPTest_p2_of_2_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((27 downto 27 => redist113_expFracR_uid23_fxpToFPTest_BitSelect_for_a_c_2_q(26)) & redist113_expFracR_uid23_fxpToFPTest_BitSelect_for_a_c_2_q) & '1');
    expFracR_uid23_fxpToFPTest_p2_of_2_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0" & expFracR_uid23_fxpToFPTest_BitSelect_for_b_tessel1_0_b) & expFracR_uid23_fxpToFPTest_p2_of_2_cin(0));
    expFracR_uid23_fxpToFPTest_p2_of_2_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            expFracR_uid23_fxpToFPTest_p2_of_2_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            expFracR_uid23_fxpToFPTest_p2_of_2_o <= STD_LOGIC_VECTOR(SIGNED(expFracR_uid23_fxpToFPTest_p2_of_2_a) + SIGNED(expFracR_uid23_fxpToFPTest_p2_of_2_b));
        END IF;
    END PROCESS;
    expFracR_uid23_fxpToFPTest_p2_of_2_q <= expFracR_uid23_fxpToFPTest_p2_of_2_o(27 downto 1);

    -- redist112_expFracR_uid23_fxpToFPTest_p1_of_2_q_1(DELAY,915)
    redist112_expFracR_uid23_fxpToFPTest_p1_of_2_q_1 : dspba_delay
    GENERIC MAP ( width => 39, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => expFracR_uid23_fxpToFPTest_p1_of_2_q, xout => redist112_expFracR_uid23_fxpToFPTest_p1_of_2_q_1_q, clk => clk, aclr => areset );

    -- expFracR_uid23_fxpToFPTest_BitJoin_for_q(BITJOIN,107)@14
    expFracR_uid23_fxpToFPTest_BitJoin_for_q_q <= expFracR_uid23_fxpToFPTest_p2_of_2_q & redist112_expFracR_uid23_fxpToFPTest_p1_of_2_q_1_q;

    -- expR_uid25_fxpToFPTest(BITSELECT,24)@14
    expR_uid25_fxpToFPTest_b <= STD_LOGIC_VECTOR(expFracR_uid23_fxpToFPTest_BitJoin_for_q_q(65 downto 53));

    -- expR_uid37_fxpToFPTest(BITSELECT,36)@14
    expR_uid37_fxpToFPTest_in <= expR_uid25_fxpToFPTest_b(10 downto 0);
    expR_uid37_fxpToFPTest_b <= expR_uid37_fxpToFPTest_in(10 downto 0);

    -- redist124_expR_uid37_fxpToFPTest_b_1(DELAY,927)
    redist124_expR_uid37_fxpToFPTest_b_1 : dspba_delay
    GENERIC MAP ( width => 11, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => expR_uid37_fxpToFPTest_b, xout => redist124_expR_uid37_fxpToFPTest_b_1_q, clk => clk, aclr => areset );

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

    -- redist126_inIsZero_uid12_fxpToFPTest_q_5(DELAY,929)
    redist126_inIsZero_uid12_fxpToFPTest_q_5 : dspba_delay
    GENERIC MAP ( width => 1, depth => 4, reset_kind => "ASYNC" )
    PORT MAP ( xin => inIsZero_uid12_fxpToFPTest_q, xout => redist126_inIsZero_uid12_fxpToFPTest_q_5_q, clk => clk, aclr => areset );

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
    udfOrInZero_uid32_fxpToFPTest_q <= udf_uid26_fxpToFPTest_n or redist126_inIsZero_uid12_fxpToFPTest_q_5_q;

    -- excSelector_uid33_fxpToFPTest(BITJOIN,32)@15
    excSelector_uid33_fxpToFPTest_q <= ovf_uid28_fxpToFPTest_n & udfOrInZero_uid32_fxpToFPTest_q;

    -- expRPostExc_uid38_fxpToFPTest(MUX,37)@15
    expRPostExc_uid38_fxpToFPTest_s <= excSelector_uid33_fxpToFPTest_q;
    expRPostExc_uid38_fxpToFPTest_combproc: PROCESS (expRPostExc_uid38_fxpToFPTest_s, redist124_expR_uid37_fxpToFPTest_b_1_q, expZ_uid36_fxpToFPTest_q, expInf_uid27_fxpToFPTest_q)
    BEGIN
        CASE (expRPostExc_uid38_fxpToFPTest_s) IS
            WHEN "00" => expRPostExc_uid38_fxpToFPTest_q <= redist124_expR_uid37_fxpToFPTest_b_1_q;
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

    -- redist125_fracR_uid24_fxpToFPTest_b_1(DELAY,928)
    redist125_fracR_uid24_fxpToFPTest_b_1 : dspba_delay
    GENERIC MAP ( width => 52, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => fracR_uid24_fxpToFPTest_b, xout => redist125_fracR_uid24_fxpToFPTest_b_1_q, clk => clk, aclr => areset );

    -- excSelector_uid29_fxpToFPTest(LOGICAL,28)@15
    excSelector_uid29_fxpToFPTest_q <= redist126_inIsZero_uid12_fxpToFPTest_q_5_q or ovf_uid28_fxpToFPTest_n or udf_uid26_fxpToFPTest_n;

    -- fracRPostExc_uid31_fxpToFPTest(MUX,30)@15
    fracRPostExc_uid31_fxpToFPTest_s <= excSelector_uid29_fxpToFPTest_q;
    fracRPostExc_uid31_fxpToFPTest_combproc: PROCESS (fracRPostExc_uid31_fxpToFPTest_s, redist125_fracR_uid24_fxpToFPTest_b_1_q, fracZ_uid30_fxpToFPTest_q)
    BEGIN
        CASE (fracRPostExc_uid31_fxpToFPTest_s) IS
            WHEN "0" => fracRPostExc_uid31_fxpToFPTest_q <= redist125_fracR_uid24_fxpToFPTest_b_1_q;
            WHEN "1" => fracRPostExc_uid31_fxpToFPTest_q <= fracZ_uid30_fxpToFPTest_q;
            WHEN OTHERS => fracRPostExc_uid31_fxpToFPTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- outRes_uid39_fxpToFPTest(BITJOIN,38)@15
    outRes_uid39_fxpToFPTest_q <= redist128_signX_uid6_fxpToFPTest_b_15_q & expRPostExc_uid38_fxpToFPTest_q & fracRPostExc_uid31_fxpToFPTest_q;

    -- xOut(GPOUT,4)@15
    q <= outRes_uid39_fxpToFPTest_q;

END normal;
