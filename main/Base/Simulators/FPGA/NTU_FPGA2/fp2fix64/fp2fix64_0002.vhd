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

-- VHDL created from fp2fix64_0002
-- VHDL created on Tue Oct 14 06:28:54 2025


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

entity fp2fix64_0002 is
    port (
        a : in std_logic_vector(63 downto 0);  -- float64_m52
        q : out std_logic_vector(54 downto 0);  -- sfix55_en53
        clk : in std_logic;
        areset : in std_logic
    );
end fp2fix64_0002;

architecture normal of fp2fix64_0002 is

    attribute altera_attribute : string;
    attribute altera_attribute of normal : architecture is "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name PHYSICAL_SYNTHESIS_REGISTER_DUPLICATION ON; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007";
    
    signal GND_q : STD_LOGIC_VECTOR (0 downto 0);
    signal VCC_q : STD_LOGIC_VECTOR (0 downto 0);
    signal cstAllOWE_uid6_fpToFxPTest_q : STD_LOGIC_VECTOR (10 downto 0);
    signal cstZeroWF_uid7_fpToFxPTest_q : STD_LOGIC_VECTOR (51 downto 0);
    signal cstAllZWE_uid8_fpToFxPTest_q : STD_LOGIC_VECTOR (10 downto 0);
    signal exp_x_uid9_fpToFxPTest_b : STD_LOGIC_VECTOR (10 downto 0);
    signal frac_x_uid10_fpToFxPTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal excZ_x_uid11_fpToFxPTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excZ_x_uid11_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid12_fpToFxPTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid12_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsNotZero_uid14_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_x_uid15_fpToFxPTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_x_uid15_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_x_uid16_fpToFxPTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_x_uid16_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invExcXZ_uid22_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal oFracX_uid23_fpToFxPTest_q : STD_LOGIC_VECTOR (52 downto 0);
    signal signX_uid25_fpToFxPTest_b : STD_LOGIC_VECTOR (0 downto 0);
    signal ovfExpVal_uid26_fpToFxPTest_q : STD_LOGIC_VECTOR (11 downto 0);
    signal ovfExpRange_uid27_fpToFxPTest_a : STD_LOGIC_VECTOR (13 downto 0);
    signal ovfExpRange_uid27_fpToFxPTest_b : STD_LOGIC_VECTOR (13 downto 0);
    signal ovfExpRange_uid27_fpToFxPTest_o : STD_LOGIC_VECTOR (13 downto 0);
    signal ovfExpRange_uid27_fpToFxPTest_n : STD_LOGIC_VECTOR (0 downto 0);
    signal udfExpVal_uid28_fpToFxPTest_q : STD_LOGIC_VECTOR (10 downto 0);
    signal udf_uid29_fpToFxPTest_a : STD_LOGIC_VECTOR (13 downto 0);
    signal udf_uid29_fpToFxPTest_b : STD_LOGIC_VECTOR (13 downto 0);
    signal udf_uid29_fpToFxPTest_o : STD_LOGIC_VECTOR (13 downto 0);
    signal udf_uid29_fpToFxPTest_n : STD_LOGIC_VECTOR (0 downto 0);
    signal ovfExpVal_uid30_fpToFxPTest_q : STD_LOGIC_VECTOR (10 downto 0);
    signal shiftValE_uid31_fpToFxPTest_a : STD_LOGIC_VECTOR (13 downto 0);
    signal shiftValE_uid31_fpToFxPTest_b : STD_LOGIC_VECTOR (13 downto 0);
    signal shiftValE_uid31_fpToFxPTest_o : STD_LOGIC_VECTOR (13 downto 0);
    signal shiftValE_uid31_fpToFxPTest_q : STD_LOGIC_VECTOR (12 downto 0);
    signal shiftValRaw_uid32_fpToFxPTest_in : STD_LOGIC_VECTOR (5 downto 0);
    signal shiftValRaw_uid32_fpToFxPTest_b : STD_LOGIC_VECTOR (5 downto 0);
    signal maxShiftCst_uid33_fpToFxPTest_q : STD_LOGIC_VECTOR (5 downto 0);
    signal shiftOutOfRange_uid34_fpToFxPTest_a : STD_LOGIC_VECTOR (14 downto 0);
    signal shiftOutOfRange_uid34_fpToFxPTest_b : STD_LOGIC_VECTOR (14 downto 0);
    signal shiftOutOfRange_uid34_fpToFxPTest_o : STD_LOGIC_VECTOR (14 downto 0);
    signal shiftOutOfRange_uid34_fpToFxPTest_n : STD_LOGIC_VECTOR (0 downto 0);
    signal shiftVal_uid35_fpToFxPTest_s : STD_LOGIC_VECTOR (0 downto 0);
    signal shiftVal_uid35_fpToFxPTest_q : STD_LOGIC_VECTOR (5 downto 0);
    signal zPadd_uid36_fpToFxPTest_q : STD_LOGIC_VECTOR (1 downto 0);
    signal shifterIn_uid37_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal maxPosValueS_uid39_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal maxNegValueS_uid40_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal zRightShiferNoStickyOut_uid41_fpToFxPTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal xXorSignE_uid42_fpToFxPTest_b : STD_LOGIC_VECTOR (55 downto 0);
    signal xXorSignE_uid42_fpToFxPTest_q : STD_LOGIC_VECTOR (55 downto 0);
    signal d0_uid43_fpToFxPTest_q : STD_LOGIC_VECTOR (2 downto 0);
    signal sPostRndFull_uid44_fpToFxPTest_a : STD_LOGIC_VECTOR (56 downto 0);
    signal sPostRndFull_uid44_fpToFxPTest_b : STD_LOGIC_VECTOR (56 downto 0);
    signal sPostRndFull_uid44_fpToFxPTest_o : STD_LOGIC_VECTOR (56 downto 0);
    signal sPostRndFull_uid44_fpToFxPTest_q : STD_LOGIC_VECTOR (56 downto 0);
    signal sPostRnd_uid45_fpToFxPTest_in : STD_LOGIC_VECTOR (55 downto 0);
    signal sPostRnd_uid45_fpToFxPTest_b : STD_LOGIC_VECTOR (54 downto 0);
    signal sPostRnd_uid46_fpToFxPTest_in : STD_LOGIC_VECTOR (57 downto 0);
    signal sPostRnd_uid46_fpToFxPTest_b : STD_LOGIC_VECTOR (56 downto 0);
    signal rndOvfPos_uid47_fpToFxPTest_a : STD_LOGIC_VECTOR (58 downto 0);
    signal rndOvfPos_uid47_fpToFxPTest_b : STD_LOGIC_VECTOR (58 downto 0);
    signal rndOvfPos_uid47_fpToFxPTest_o : STD_LOGIC_VECTOR (58 downto 0);
    signal rndOvfPos_uid47_fpToFxPTest_c : STD_LOGIC_VECTOR (0 downto 0);
    signal ovfPostRnd_uid48_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal muxSelConc_uid49_fpToFxPTest_q : STD_LOGIC_VECTOR (2 downto 0);
    signal muxSel_uid50_fpToFxPTest_q : STD_LOGIC_VECTOR (1 downto 0);
    signal maxNegValueU_uid51_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal finalOut_uid52_fpToFxPTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal finalOut_uid52_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal eq0_uid56_fracXIsZero_uid13_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq1_uid59_fracXIsZero_uid13_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq2_uid62_fracXIsZero_uid13_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq3_uid65_fracXIsZero_uid13_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq4_uid68_fracXIsZero_uid13_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq5_uid71_fracXIsZero_uid13_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq6_uid74_fracXIsZero_uid13_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq7_uid77_fracXIsZero_uid13_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal eq8_uid80_fracXIsZero_uid13_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid81_fracXIsZero_uid13_fpToFxPTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid81_fracXIsZero_uid13_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid82_fracXIsZero_uid13_fpToFxPTest_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev0_uid82_fracXIsZero_uid13_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal and_lev1_uid83_fracXIsZero_uid13_fpToFxPTest_q : STD_LOGIC_VECTOR (0 downto 0);
    signal rightShiftStage0Idx1Rng16_uid86_rightShiferNoStickyOut_uid38_fpToFxPTest_b : STD_LOGIC_VECTOR (38 downto 0);
    signal rightShiftStage0Idx1Pad16_uid87_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (15 downto 0);
    signal rightShiftStage0Idx1_uid88_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal rightShiftStage0Idx2Rng32_uid89_rightShiferNoStickyOut_uid38_fpToFxPTest_b : STD_LOGIC_VECTOR (22 downto 0);
    signal rightShiftStage0Idx2Pad32_uid90_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (31 downto 0);
    signal rightShiftStage0Idx2_uid91_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal rightShiftStage0Idx3Rng48_uid92_rightShiferNoStickyOut_uid38_fpToFxPTest_b : STD_LOGIC_VECTOR (6 downto 0);
    signal rightShiftStage0Idx3Pad48_uid93_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (47 downto 0);
    signal rightShiftStage0Idx3_uid94_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal rightShiftStage1Idx1Rng4_uid97_rightShiferNoStickyOut_uid38_fpToFxPTest_b : STD_LOGIC_VECTOR (50 downto 0);
    signal rightShiftStage1Idx1Pad4_uid98_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (3 downto 0);
    signal rightShiftStage1Idx1_uid99_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal rightShiftStage1Idx2Rng8_uid100_rightShiferNoStickyOut_uid38_fpToFxPTest_b : STD_LOGIC_VECTOR (46 downto 0);
    signal rightShiftStage1Idx2Pad8_uid101_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (7 downto 0);
    signal rightShiftStage1Idx2_uid102_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal rightShiftStage1Idx3Rng12_uid103_rightShiferNoStickyOut_uid38_fpToFxPTest_b : STD_LOGIC_VECTOR (42 downto 0);
    signal rightShiftStage1Idx3Pad12_uid104_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (11 downto 0);
    signal rightShiftStage1Idx3_uid105_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal rightShiftStage2Idx1Rng1_uid108_rightShiferNoStickyOut_uid38_fpToFxPTest_b : STD_LOGIC_VECTOR (53 downto 0);
    signal rightShiftStage2Idx1_uid110_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal rightShiftStage2Idx2Rng2_uid111_rightShiferNoStickyOut_uid38_fpToFxPTest_b : STD_LOGIC_VECTOR (52 downto 0);
    signal rightShiftStage2Idx2_uid113_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal rightShiftStage2Idx3Rng3_uid114_rightShiferNoStickyOut_uid38_fpToFxPTest_b : STD_LOGIC_VECTOR (51 downto 0);
    signal rightShiftStage2Idx3Pad3_uid115_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (2 downto 0);
    signal rightShiftStage2Idx3_uid116_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal rightShiftStage2_uid118_rightShiferNoStickyOut_uid38_fpToFxPTest_s : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStage2_uid118_rightShiferNoStickyOut_uid38_fpToFxPTest_q : STD_LOGIC_VECTOR (54 downto 0);
    signal c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_b : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_c : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_d : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_e : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_f : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_g : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_h : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_i : STD_LOGIC_VECTOR (5 downto 0);
    signal c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_j : STD_LOGIC_VECTOR (3 downto 0);
    signal z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_b : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_c : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_d : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_e : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_f : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_g : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_h : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_i : STD_LOGIC_VECTOR (5 downto 0);
    signal z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_j : STD_LOGIC_VECTOR (3 downto 0);
    signal rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_b : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_c : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_d : STD_LOGIC_VECTOR (1 downto 0);
    signal redist0_rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_c_1_q : STD_LOGIC_VECTOR (1 downto 0);
    signal redist1_rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_d_1_q : STD_LOGIC_VECTOR (1 downto 0);
    signal redist2_sPostRnd_uid45_fpToFxPTest_b_2_q : STD_LOGIC_VECTOR (54 downto 0);
    signal redist3_shiftValRaw_uid32_fpToFxPTest_b_1_q : STD_LOGIC_VECTOR (5 downto 0);
    signal redist4_udf_uid29_fpToFxPTest_n_6_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist5_ovfExpRange_uid27_fpToFxPTest_n_6_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist6_signX_uid25_fpToFxPTest_b_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist7_signX_uid25_fpToFxPTest_b_6_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist8_excN_x_uid16_fpToFxPTest_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist9_excI_x_uid15_fpToFxPTest_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist10_expXIsMax_uid12_fpToFxPTest_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist11_excZ_x_uid11_fpToFxPTest_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist12_frac_x_uid10_fpToFxPTest_b_2_q : STD_LOGIC_VECTOR (51 downto 0);

begin


    -- maxNegValueU_uid51_fpToFxPTest(CONSTANT,50)
    maxNegValueU_uid51_fpToFxPTest_q <= "0000000000000000000000000000000000000000000000000000000";

    -- maxNegValueS_uid40_fpToFxPTest(CONSTANT,39)
    maxNegValueS_uid40_fpToFxPTest_q <= "1000000000000000000000000000000000000000000000000000000";

    -- maxPosValueS_uid39_fpToFxPTest(CONSTANT,38)
    maxPosValueS_uid39_fpToFxPTest_q <= "0111111111111111111111111111111111111111111111111111111";

    -- d0_uid43_fpToFxPTest(CONSTANT,42)
    d0_uid43_fpToFxPTest_q <= "001";

    -- signX_uid25_fpToFxPTest(BITSELECT,24)@0
    signX_uid25_fpToFxPTest_b <= STD_LOGIC_VECTOR(a(63 downto 63));

    -- redist6_signX_uid25_fpToFxPTest_b_4(DELAY,128)
    redist6_signX_uid25_fpToFxPTest_b_4 : dspba_delay
    GENERIC MAP ( width => 1, depth => 4, reset_kind => "ASYNC" )
    PORT MAP ( xin => signX_uid25_fpToFxPTest_b, xout => redist6_signX_uid25_fpToFxPTest_b_4_q, clk => clk, aclr => areset );

    -- GND(CONSTANT,0)
    GND_q <= "0";

    -- rightShiftStage2Idx3Pad3_uid115_rightShiferNoStickyOut_uid38_fpToFxPTest(CONSTANT,114)
    rightShiftStage2Idx3Pad3_uid115_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= "000";

    -- rightShiftStage2Idx3Rng3_uid114_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,113)@3
    rightShiftStage2Idx3Rng3_uid114_rightShiferNoStickyOut_uid38_fpToFxPTest_b <= rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_q(54 downto 3);

    -- rightShiftStage2Idx3_uid116_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,115)@3
    rightShiftStage2Idx3_uid116_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage2Idx3Pad3_uid115_rightShiferNoStickyOut_uid38_fpToFxPTest_q & rightShiftStage2Idx3Rng3_uid114_rightShiferNoStickyOut_uid38_fpToFxPTest_b;

    -- zPadd_uid36_fpToFxPTest(CONSTANT,35)
    zPadd_uid36_fpToFxPTest_q <= "00";

    -- rightShiftStage2Idx2Rng2_uid111_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,110)@3
    rightShiftStage2Idx2Rng2_uid111_rightShiferNoStickyOut_uid38_fpToFxPTest_b <= rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_q(54 downto 2);

    -- rightShiftStage2Idx2_uid113_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,112)@3
    rightShiftStage2Idx2_uid113_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= zPadd_uid36_fpToFxPTest_q & rightShiftStage2Idx2Rng2_uid111_rightShiferNoStickyOut_uid38_fpToFxPTest_b;

    -- rightShiftStage2Idx1Rng1_uid108_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,107)@3
    rightShiftStage2Idx1Rng1_uid108_rightShiferNoStickyOut_uid38_fpToFxPTest_b <= rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_q(54 downto 1);

    -- rightShiftStage2Idx1_uid110_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,109)@3
    rightShiftStage2Idx1_uid110_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= GND_q & rightShiftStage2Idx1Rng1_uid108_rightShiferNoStickyOut_uid38_fpToFxPTest_b;

    -- rightShiftStage1Idx3Pad12_uid104_rightShiferNoStickyOut_uid38_fpToFxPTest(CONSTANT,103)
    rightShiftStage1Idx3Pad12_uid104_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= "000000000000";

    -- rightShiftStage1Idx3Rng12_uid103_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,102)@3
    rightShiftStage1Idx3Rng12_uid103_rightShiferNoStickyOut_uid38_fpToFxPTest_b <= rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_q(54 downto 12);

    -- rightShiftStage1Idx3_uid105_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,104)@3
    rightShiftStage1Idx3_uid105_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage1Idx3Pad12_uid104_rightShiferNoStickyOut_uid38_fpToFxPTest_q & rightShiftStage1Idx3Rng12_uid103_rightShiferNoStickyOut_uid38_fpToFxPTest_b;

    -- rightShiftStage1Idx2Pad8_uid101_rightShiferNoStickyOut_uid38_fpToFxPTest(CONSTANT,100)
    rightShiftStage1Idx2Pad8_uid101_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= "00000000";

    -- rightShiftStage1Idx2Rng8_uid100_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,99)@3
    rightShiftStage1Idx2Rng8_uid100_rightShiferNoStickyOut_uid38_fpToFxPTest_b <= rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_q(54 downto 8);

    -- rightShiftStage1Idx2_uid102_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,101)@3
    rightShiftStage1Idx2_uid102_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage1Idx2Pad8_uid101_rightShiferNoStickyOut_uid38_fpToFxPTest_q & rightShiftStage1Idx2Rng8_uid100_rightShiferNoStickyOut_uid38_fpToFxPTest_b;

    -- rightShiftStage1Idx1Pad4_uid98_rightShiferNoStickyOut_uid38_fpToFxPTest(CONSTANT,97)
    rightShiftStage1Idx1Pad4_uid98_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= "0000";

    -- rightShiftStage1Idx1Rng4_uid97_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,96)@3
    rightShiftStage1Idx1Rng4_uid97_rightShiferNoStickyOut_uid38_fpToFxPTest_b <= rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_q(54 downto 4);

    -- rightShiftStage1Idx1_uid99_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,98)@3
    rightShiftStage1Idx1_uid99_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage1Idx1Pad4_uid98_rightShiferNoStickyOut_uid38_fpToFxPTest_q & rightShiftStage1Idx1Rng4_uid97_rightShiferNoStickyOut_uid38_fpToFxPTest_b;

    -- rightShiftStage0Idx3Pad48_uid93_rightShiferNoStickyOut_uid38_fpToFxPTest(CONSTANT,92)
    rightShiftStage0Idx3Pad48_uid93_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= "000000000000000000000000000000000000000000000000";

    -- rightShiftStage0Idx3Rng48_uid92_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,91)@2
    rightShiftStage0Idx3Rng48_uid92_rightShiferNoStickyOut_uid38_fpToFxPTest_b <= shifterIn_uid37_fpToFxPTest_q(54 downto 48);

    -- rightShiftStage0Idx3_uid94_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,93)@2
    rightShiftStage0Idx3_uid94_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage0Idx3Pad48_uid93_rightShiferNoStickyOut_uid38_fpToFxPTest_q & rightShiftStage0Idx3Rng48_uid92_rightShiferNoStickyOut_uid38_fpToFxPTest_b;

    -- rightShiftStage0Idx2Pad32_uid90_rightShiferNoStickyOut_uid38_fpToFxPTest(CONSTANT,89)
    rightShiftStage0Idx2Pad32_uid90_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= "00000000000000000000000000000000";

    -- rightShiftStage0Idx2Rng32_uid89_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,88)@2
    rightShiftStage0Idx2Rng32_uid89_rightShiferNoStickyOut_uid38_fpToFxPTest_b <= shifterIn_uid37_fpToFxPTest_q(54 downto 32);

    -- rightShiftStage0Idx2_uid91_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,90)@2
    rightShiftStage0Idx2_uid91_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage0Idx2Pad32_uid90_rightShiferNoStickyOut_uid38_fpToFxPTest_q & rightShiftStage0Idx2Rng32_uid89_rightShiferNoStickyOut_uid38_fpToFxPTest_b;

    -- rightShiftStage0Idx1Pad16_uid87_rightShiferNoStickyOut_uid38_fpToFxPTest(CONSTANT,86)
    rightShiftStage0Idx1Pad16_uid87_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= "0000000000000000";

    -- rightShiftStage0Idx1Rng16_uid86_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,85)@2
    rightShiftStage0Idx1Rng16_uid86_rightShiferNoStickyOut_uid38_fpToFxPTest_b <= shifterIn_uid37_fpToFxPTest_q(54 downto 16);

    -- rightShiftStage0Idx1_uid88_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,87)@2
    rightShiftStage0Idx1_uid88_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage0Idx1Pad16_uid87_rightShiferNoStickyOut_uid38_fpToFxPTest_q & rightShiftStage0Idx1Rng16_uid86_rightShiferNoStickyOut_uid38_fpToFxPTest_b;

    -- cstAllZWE_uid8_fpToFxPTest(CONSTANT,7)
    cstAllZWE_uid8_fpToFxPTest_q <= "00000000000";

    -- exp_x_uid9_fpToFxPTest(BITSELECT,8)@0
    exp_x_uid9_fpToFxPTest_b <= a(62 downto 52);

    -- excZ_x_uid11_fpToFxPTest(LOGICAL,10)@0 + 1
    excZ_x_uid11_fpToFxPTest_qi <= "1" WHEN exp_x_uid9_fpToFxPTest_b = cstAllZWE_uid8_fpToFxPTest_q ELSE "0";
    excZ_x_uid11_fpToFxPTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excZ_x_uid11_fpToFxPTest_qi, xout => excZ_x_uid11_fpToFxPTest_q, clk => clk, aclr => areset );

    -- redist11_excZ_x_uid11_fpToFxPTest_q_2(DELAY,133)
    redist11_excZ_x_uid11_fpToFxPTest_q_2 : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excZ_x_uid11_fpToFxPTest_q, xout => redist11_excZ_x_uid11_fpToFxPTest_q_2_q, clk => clk, aclr => areset );

    -- invExcXZ_uid22_fpToFxPTest(LOGICAL,21)@2
    invExcXZ_uid22_fpToFxPTest_q <= not (redist11_excZ_x_uid11_fpToFxPTest_q_2_q);

    -- frac_x_uid10_fpToFxPTest(BITSELECT,9)@0
    frac_x_uid10_fpToFxPTest_b <= a(51 downto 0);

    -- redist12_frac_x_uid10_fpToFxPTest_b_2(DELAY,134)
    redist12_frac_x_uid10_fpToFxPTest_b_2 : dspba_delay
    GENERIC MAP ( width => 52, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => frac_x_uid10_fpToFxPTest_b, xout => redist12_frac_x_uid10_fpToFxPTest_b_2_q, clk => clk, aclr => areset );

    -- oFracX_uid23_fpToFxPTest(BITJOIN,22)@2
    oFracX_uid23_fpToFxPTest_q <= invExcXZ_uid22_fpToFxPTest_q & redist12_frac_x_uid10_fpToFxPTest_b_2_q;

    -- shifterIn_uid37_fpToFxPTest(BITJOIN,36)@2
    shifterIn_uid37_fpToFxPTest_q <= oFracX_uid23_fpToFxPTest_q & zPadd_uid36_fpToFxPTest_q;

    -- maxShiftCst_uid33_fpToFxPTest(CONSTANT,32)
    maxShiftCst_uid33_fpToFxPTest_q <= "110111";

    -- ovfExpVal_uid30_fpToFxPTest(CONSTANT,29)
    ovfExpVal_uid30_fpToFxPTest_q <= "01111111111";

    -- shiftValE_uid31_fpToFxPTest(SUB,30)@0 + 1
    shiftValE_uid31_fpToFxPTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((13 downto 11 => ovfExpVal_uid30_fpToFxPTest_q(10)) & ovfExpVal_uid30_fpToFxPTest_q));
    shiftValE_uid31_fpToFxPTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("000" & exp_x_uid9_fpToFxPTest_b));
    shiftValE_uid31_fpToFxPTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            shiftValE_uid31_fpToFxPTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            shiftValE_uid31_fpToFxPTest_o <= STD_LOGIC_VECTOR(SIGNED(shiftValE_uid31_fpToFxPTest_a) - SIGNED(shiftValE_uid31_fpToFxPTest_b));
        END IF;
    END PROCESS;
    shiftValE_uid31_fpToFxPTest_q <= shiftValE_uid31_fpToFxPTest_o(12 downto 0);

    -- shiftValRaw_uid32_fpToFxPTest(BITSELECT,31)@1
    shiftValRaw_uid32_fpToFxPTest_in <= shiftValE_uid31_fpToFxPTest_q(5 downto 0);
    shiftValRaw_uid32_fpToFxPTest_b <= shiftValRaw_uid32_fpToFxPTest_in(5 downto 0);

    -- redist3_shiftValRaw_uid32_fpToFxPTest_b_1(DELAY,125)
    redist3_shiftValRaw_uid32_fpToFxPTest_b_1 : dspba_delay
    GENERIC MAP ( width => 6, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => shiftValRaw_uid32_fpToFxPTest_b, xout => redist3_shiftValRaw_uid32_fpToFxPTest_b_1_q, clk => clk, aclr => areset );

    -- shiftOutOfRange_uid34_fpToFxPTest(COMPARE,33)@1 + 1
    shiftOutOfRange_uid34_fpToFxPTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((14 downto 13 => shiftValE_uid31_fpToFxPTest_q(12)) & shiftValE_uid31_fpToFxPTest_q));
    shiftOutOfRange_uid34_fpToFxPTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("000000000" & maxShiftCst_uid33_fpToFxPTest_q));
    shiftOutOfRange_uid34_fpToFxPTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            shiftOutOfRange_uid34_fpToFxPTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            shiftOutOfRange_uid34_fpToFxPTest_o <= STD_LOGIC_VECTOR(SIGNED(shiftOutOfRange_uid34_fpToFxPTest_a) - SIGNED(shiftOutOfRange_uid34_fpToFxPTest_b));
        END IF;
    END PROCESS;
    shiftOutOfRange_uid34_fpToFxPTest_n(0) <= not (shiftOutOfRange_uid34_fpToFxPTest_o(14));

    -- shiftVal_uid35_fpToFxPTest(MUX,34)@2
    shiftVal_uid35_fpToFxPTest_s <= shiftOutOfRange_uid34_fpToFxPTest_n;
    shiftVal_uid35_fpToFxPTest_combproc: PROCESS (shiftVal_uid35_fpToFxPTest_s, redist3_shiftValRaw_uid32_fpToFxPTest_b_1_q, maxShiftCst_uid33_fpToFxPTest_q)
    BEGIN
        CASE (shiftVal_uid35_fpToFxPTest_s) IS
            WHEN "0" => shiftVal_uid35_fpToFxPTest_q <= redist3_shiftValRaw_uid32_fpToFxPTest_b_1_q;
            WHEN "1" => shiftVal_uid35_fpToFxPTest_q <= maxShiftCst_uid33_fpToFxPTest_q;
            WHEN OTHERS => shiftVal_uid35_fpToFxPTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select(BITSELECT,121)@2
    rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_b <= shiftVal_uid35_fpToFxPTest_q(5 downto 4);
    rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_c <= shiftVal_uid35_fpToFxPTest_q(3 downto 2);
    rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_d <= shiftVal_uid35_fpToFxPTest_q(1 downto 0);

    -- rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest(MUX,95)@2 + 1
    rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_s <= rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_b;
    rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_s) IS
                WHEN "00" => rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= shifterIn_uid37_fpToFxPTest_q;
                WHEN "01" => rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage0Idx1_uid88_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
                WHEN "10" => rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage0Idx2_uid91_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
                WHEN "11" => rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage0Idx3_uid94_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
                WHEN OTHERS => rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- redist0_rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_c_1(DELAY,122)
    redist0_rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_c_1 : dspba_delay
    GENERIC MAP ( width => 2, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_c, xout => redist0_rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_c_1_q, clk => clk, aclr => areset );

    -- rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest(MUX,106)@3
    rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_s <= redist0_rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_c_1_q;
    rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_combproc: PROCESS (rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_s, rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_q, rightShiftStage1Idx1_uid99_rightShiferNoStickyOut_uid38_fpToFxPTest_q, rightShiftStage1Idx2_uid102_rightShiferNoStickyOut_uid38_fpToFxPTest_q, rightShiftStage1Idx3_uid105_rightShiferNoStickyOut_uid38_fpToFxPTest_q)
    BEGIN
        CASE (rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_s) IS
            WHEN "00" => rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage0_uid96_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
            WHEN "01" => rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage1Idx1_uid99_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
            WHEN "10" => rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage1Idx2_uid102_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
            WHEN "11" => rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage1Idx3_uid105_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
            WHEN OTHERS => rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- redist1_rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_d_1(DELAY,123)
    redist1_rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_d_1 : dspba_delay
    GENERIC MAP ( width => 2, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_d, xout => redist1_rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_d_1_q, clk => clk, aclr => areset );

    -- rightShiftStage2_uid118_rightShiferNoStickyOut_uid38_fpToFxPTest(MUX,117)@3 + 1
    rightShiftStage2_uid118_rightShiferNoStickyOut_uid38_fpToFxPTest_s <= redist1_rightShiftStageSel5Dto4_uid95_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_d_1_q;
    rightShiftStage2_uid118_rightShiferNoStickyOut_uid38_fpToFxPTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            rightShiftStage2_uid118_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (rightShiftStage2_uid118_rightShiferNoStickyOut_uid38_fpToFxPTest_s) IS
                WHEN "00" => rightShiftStage2_uid118_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage1_uid107_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
                WHEN "01" => rightShiftStage2_uid118_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage2Idx1_uid110_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
                WHEN "10" => rightShiftStage2_uid118_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage2Idx2_uid113_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
                WHEN "11" => rightShiftStage2_uid118_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage2Idx3_uid116_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
                WHEN OTHERS => rightShiftStage2_uid118_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= (others => '0');
            END CASE;
        END IF;
    END PROCESS;

    -- zRightShiferNoStickyOut_uid41_fpToFxPTest(BITJOIN,40)@4
    zRightShiferNoStickyOut_uid41_fpToFxPTest_q <= GND_q & rightShiftStage2_uid118_rightShiferNoStickyOut_uid38_fpToFxPTest_q;

    -- xXorSignE_uid42_fpToFxPTest(LOGICAL,41)@4
    xXorSignE_uid42_fpToFxPTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((55 downto 1 => redist6_signX_uid25_fpToFxPTest_b_4_q(0)) & redist6_signX_uid25_fpToFxPTest_b_4_q));
    xXorSignE_uid42_fpToFxPTest_q <= zRightShiferNoStickyOut_uid41_fpToFxPTest_q xor xXorSignE_uid42_fpToFxPTest_b;

    -- sPostRndFull_uid44_fpToFxPTest(ADD,43)@4 + 1
    sPostRndFull_uid44_fpToFxPTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((56 downto 56 => xXorSignE_uid42_fpToFxPTest_q(55)) & xXorSignE_uid42_fpToFxPTest_q));
    sPostRndFull_uid44_fpToFxPTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((56 downto 3 => d0_uid43_fpToFxPTest_q(2)) & d0_uid43_fpToFxPTest_q));
    sPostRndFull_uid44_fpToFxPTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            sPostRndFull_uid44_fpToFxPTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            sPostRndFull_uid44_fpToFxPTest_o <= STD_LOGIC_VECTOR(SIGNED(sPostRndFull_uid44_fpToFxPTest_a) + SIGNED(sPostRndFull_uid44_fpToFxPTest_b));
        END IF;
    END PROCESS;
    sPostRndFull_uid44_fpToFxPTest_q <= sPostRndFull_uid44_fpToFxPTest_o(56 downto 0);

    -- sPostRnd_uid45_fpToFxPTest(BITSELECT,44)@5
    sPostRnd_uid45_fpToFxPTest_in <= sPostRndFull_uid44_fpToFxPTest_q(55 downto 0);
    sPostRnd_uid45_fpToFxPTest_b <= sPostRnd_uid45_fpToFxPTest_in(55 downto 1);

    -- redist2_sPostRnd_uid45_fpToFxPTest_b_2(DELAY,124)
    redist2_sPostRnd_uid45_fpToFxPTest_b_2 : dspba_delay
    GENERIC MAP ( width => 55, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => sPostRnd_uid45_fpToFxPTest_b, xout => redist2_sPostRnd_uid45_fpToFxPTest_b_2_q, clk => clk, aclr => areset );

    -- redist7_signX_uid25_fpToFxPTest_b_6(DELAY,129)
    redist7_signX_uid25_fpToFxPTest_b_6 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => redist6_signX_uid25_fpToFxPTest_b_4_q, xout => redist7_signX_uid25_fpToFxPTest_b_6_q, clk => clk, aclr => areset );

    -- udfExpVal_uid28_fpToFxPTest(CONSTANT,27)
    udfExpVal_uid28_fpToFxPTest_q <= "01111001000";

    -- udf_uid29_fpToFxPTest(COMPARE,28)@0 + 1
    udf_uid29_fpToFxPTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((13 downto 11 => udfExpVal_uid28_fpToFxPTest_q(10)) & udfExpVal_uid28_fpToFxPTest_q));
    udf_uid29_fpToFxPTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("000" & exp_x_uid9_fpToFxPTest_b));
    udf_uid29_fpToFxPTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            udf_uid29_fpToFxPTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            udf_uid29_fpToFxPTest_o <= STD_LOGIC_VECTOR(SIGNED(udf_uid29_fpToFxPTest_a) - SIGNED(udf_uid29_fpToFxPTest_b));
        END IF;
    END PROCESS;
    udf_uid29_fpToFxPTest_n(0) <= not (udf_uid29_fpToFxPTest_o(13));

    -- redist4_udf_uid29_fpToFxPTest_n_6(DELAY,126)
    redist4_udf_uid29_fpToFxPTest_n_6 : dspba_delay
    GENERIC MAP ( width => 1, depth => 5, reset_kind => "ASYNC" )
    PORT MAP ( xin => udf_uid29_fpToFxPTest_n, xout => redist4_udf_uid29_fpToFxPTest_n_6_q, clk => clk, aclr => areset );

    -- sPostRnd_uid46_fpToFxPTest(BITSELECT,45)@5
    sPostRnd_uid46_fpToFxPTest_in <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((57 downto 57 => sPostRndFull_uid44_fpToFxPTest_q(56)) & sPostRndFull_uid44_fpToFxPTest_q));
    sPostRnd_uid46_fpToFxPTest_b <= STD_LOGIC_VECTOR(sPostRnd_uid46_fpToFxPTest_in(57 downto 1));

    -- rndOvfPos_uid47_fpToFxPTest(COMPARE,46)@5 + 1
    rndOvfPos_uid47_fpToFxPTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0000" & maxPosValueS_uid39_fpToFxPTest_q));
    rndOvfPos_uid47_fpToFxPTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((58 downto 57 => sPostRnd_uid46_fpToFxPTest_b(56)) & sPostRnd_uid46_fpToFxPTest_b));
    rndOvfPos_uid47_fpToFxPTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            rndOvfPos_uid47_fpToFxPTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            rndOvfPos_uid47_fpToFxPTest_o <= STD_LOGIC_VECTOR(SIGNED(rndOvfPos_uid47_fpToFxPTest_a) - SIGNED(rndOvfPos_uid47_fpToFxPTest_b));
        END IF;
    END PROCESS;
    rndOvfPos_uid47_fpToFxPTest_c(0) <= rndOvfPos_uid47_fpToFxPTest_o(58);

    -- ovfExpVal_uid26_fpToFxPTest(CONSTANT,25)
    ovfExpVal_uid26_fpToFxPTest_q <= "010000000000";

    -- ovfExpRange_uid27_fpToFxPTest(COMPARE,26)@0 + 1
    ovfExpRange_uid27_fpToFxPTest_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("000" & exp_x_uid9_fpToFxPTest_b));
    ovfExpRange_uid27_fpToFxPTest_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((13 downto 12 => ovfExpVal_uid26_fpToFxPTest_q(11)) & ovfExpVal_uid26_fpToFxPTest_q));
    ovfExpRange_uid27_fpToFxPTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            ovfExpRange_uid27_fpToFxPTest_o <= (others => '0');
        ELSIF (clk'EVENT AND clk = '1') THEN
            ovfExpRange_uid27_fpToFxPTest_o <= STD_LOGIC_VECTOR(SIGNED(ovfExpRange_uid27_fpToFxPTest_a) - SIGNED(ovfExpRange_uid27_fpToFxPTest_b));
        END IF;
    END PROCESS;
    ovfExpRange_uid27_fpToFxPTest_n(0) <= not (ovfExpRange_uid27_fpToFxPTest_o(13));

    -- redist5_ovfExpRange_uid27_fpToFxPTest_n_6(DELAY,127)
    redist5_ovfExpRange_uid27_fpToFxPTest_n_6 : dspba_delay
    GENERIC MAP ( width => 1, depth => 5, reset_kind => "ASYNC" )
    PORT MAP ( xin => ovfExpRange_uid27_fpToFxPTest_n, xout => redist5_ovfExpRange_uid27_fpToFxPTest_n_6_q, clk => clk, aclr => areset );

    -- cstZeroWF_uid7_fpToFxPTest(CONSTANT,6)
    cstZeroWF_uid7_fpToFxPTest_q <= "0000000000000000000000000000000000000000000000000000";

    -- c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select(BITSELECT,119)
    c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_b <= cstZeroWF_uid7_fpToFxPTest_q(5 downto 0);
    c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_c <= cstZeroWF_uid7_fpToFxPTest_q(11 downto 6);
    c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_d <= cstZeroWF_uid7_fpToFxPTest_q(17 downto 12);
    c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_e <= cstZeroWF_uid7_fpToFxPTest_q(23 downto 18);
    c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_f <= cstZeroWF_uid7_fpToFxPTest_q(29 downto 24);
    c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_g <= cstZeroWF_uid7_fpToFxPTest_q(35 downto 30);
    c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_h <= cstZeroWF_uid7_fpToFxPTest_q(41 downto 36);
    c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_i <= cstZeroWF_uid7_fpToFxPTest_q(47 downto 42);
    c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_j <= cstZeroWF_uid7_fpToFxPTest_q(51 downto 48);

    -- z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select(BITSELECT,120)@2
    z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_b <= redist12_frac_x_uid10_fpToFxPTest_b_2_q(5 downto 0);
    z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_c <= redist12_frac_x_uid10_fpToFxPTest_b_2_q(11 downto 6);
    z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_d <= redist12_frac_x_uid10_fpToFxPTest_b_2_q(17 downto 12);
    z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_e <= redist12_frac_x_uid10_fpToFxPTest_b_2_q(23 downto 18);
    z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_f <= redist12_frac_x_uid10_fpToFxPTest_b_2_q(29 downto 24);
    z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_g <= redist12_frac_x_uid10_fpToFxPTest_b_2_q(35 downto 30);
    z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_h <= redist12_frac_x_uid10_fpToFxPTest_b_2_q(41 downto 36);
    z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_i <= redist12_frac_x_uid10_fpToFxPTest_b_2_q(47 downto 42);
    z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_j <= redist12_frac_x_uid10_fpToFxPTest_b_2_q(51 downto 48);

    -- eq8_uid80_fracXIsZero_uid13_fpToFxPTest(LOGICAL,79)@2
    eq8_uid80_fracXIsZero_uid13_fpToFxPTest_q <= "1" WHEN z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_j = c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_j ELSE "0";

    -- eq7_uid77_fracXIsZero_uid13_fpToFxPTest(LOGICAL,76)@2
    eq7_uid77_fracXIsZero_uid13_fpToFxPTest_q <= "1" WHEN z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_i = c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_i ELSE "0";

    -- eq6_uid74_fracXIsZero_uid13_fpToFxPTest(LOGICAL,73)@2
    eq6_uid74_fracXIsZero_uid13_fpToFxPTest_q <= "1" WHEN z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_h = c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_h ELSE "0";

    -- and_lev0_uid82_fracXIsZero_uid13_fpToFxPTest(LOGICAL,81)@2 + 1
    and_lev0_uid82_fracXIsZero_uid13_fpToFxPTest_qi <= eq6_uid74_fracXIsZero_uid13_fpToFxPTest_q and eq7_uid77_fracXIsZero_uid13_fpToFxPTest_q and eq8_uid80_fracXIsZero_uid13_fpToFxPTest_q;
    and_lev0_uid82_fracXIsZero_uid13_fpToFxPTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev0_uid82_fracXIsZero_uid13_fpToFxPTest_qi, xout => and_lev0_uid82_fracXIsZero_uid13_fpToFxPTest_q, clk => clk, aclr => areset );

    -- eq5_uid71_fracXIsZero_uid13_fpToFxPTest(LOGICAL,70)@2
    eq5_uid71_fracXIsZero_uid13_fpToFxPTest_q <= "1" WHEN z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_g = c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_g ELSE "0";

    -- eq4_uid68_fracXIsZero_uid13_fpToFxPTest(LOGICAL,67)@2
    eq4_uid68_fracXIsZero_uid13_fpToFxPTest_q <= "1" WHEN z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_f = c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_f ELSE "0";

    -- eq3_uid65_fracXIsZero_uid13_fpToFxPTest(LOGICAL,64)@2
    eq3_uid65_fracXIsZero_uid13_fpToFxPTest_q <= "1" WHEN z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_e = c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_e ELSE "0";

    -- eq2_uid62_fracXIsZero_uid13_fpToFxPTest(LOGICAL,61)@2
    eq2_uid62_fracXIsZero_uid13_fpToFxPTest_q <= "1" WHEN z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_d = c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_d ELSE "0";

    -- eq1_uid59_fracXIsZero_uid13_fpToFxPTest(LOGICAL,58)@2
    eq1_uid59_fracXIsZero_uid13_fpToFxPTest_q <= "1" WHEN z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_c = c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_c ELSE "0";

    -- eq0_uid56_fracXIsZero_uid13_fpToFxPTest(LOGICAL,55)@2
    eq0_uid56_fracXIsZero_uid13_fpToFxPTest_q <= "1" WHEN z0_uid54_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_b = c0_uid55_fracXIsZero_uid13_fpToFxPTest_merged_bit_select_b ELSE "0";

    -- and_lev0_uid81_fracXIsZero_uid13_fpToFxPTest(LOGICAL,80)@2 + 1
    and_lev0_uid81_fracXIsZero_uid13_fpToFxPTest_qi <= eq0_uid56_fracXIsZero_uid13_fpToFxPTest_q and eq1_uid59_fracXIsZero_uid13_fpToFxPTest_q and eq2_uid62_fracXIsZero_uid13_fpToFxPTest_q and eq3_uid65_fracXIsZero_uid13_fpToFxPTest_q and eq4_uid68_fracXIsZero_uid13_fpToFxPTest_q and eq5_uid71_fracXIsZero_uid13_fpToFxPTest_q;
    and_lev0_uid81_fracXIsZero_uid13_fpToFxPTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => and_lev0_uid81_fracXIsZero_uid13_fpToFxPTest_qi, xout => and_lev0_uid81_fracXIsZero_uid13_fpToFxPTest_q, clk => clk, aclr => areset );

    -- and_lev1_uid83_fracXIsZero_uid13_fpToFxPTest(LOGICAL,82)@3
    and_lev1_uid83_fracXIsZero_uid13_fpToFxPTest_q <= and_lev0_uid81_fracXIsZero_uid13_fpToFxPTest_q and and_lev0_uid82_fracXIsZero_uid13_fpToFxPTest_q;

    -- cstAllOWE_uid6_fpToFxPTest(CONSTANT,5)
    cstAllOWE_uid6_fpToFxPTest_q <= "11111111111";

    -- expXIsMax_uid12_fpToFxPTest(LOGICAL,11)@0 + 1
    expXIsMax_uid12_fpToFxPTest_qi <= "1" WHEN exp_x_uid9_fpToFxPTest_b = cstAllOWE_uid6_fpToFxPTest_q ELSE "0";
    expXIsMax_uid12_fpToFxPTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => expXIsMax_uid12_fpToFxPTest_qi, xout => expXIsMax_uid12_fpToFxPTest_q, clk => clk, aclr => areset );

    -- redist10_expXIsMax_uid12_fpToFxPTest_q_3(DELAY,132)
    redist10_expXIsMax_uid12_fpToFxPTest_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => expXIsMax_uid12_fpToFxPTest_q, xout => redist10_expXIsMax_uid12_fpToFxPTest_q_3_q, clk => clk, aclr => areset );

    -- excI_x_uid15_fpToFxPTest(LOGICAL,14)@3 + 1
    excI_x_uid15_fpToFxPTest_qi <= redist10_expXIsMax_uid12_fpToFxPTest_q_3_q and and_lev1_uid83_fracXIsZero_uid13_fpToFxPTest_q;
    excI_x_uid15_fpToFxPTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excI_x_uid15_fpToFxPTest_qi, xout => excI_x_uid15_fpToFxPTest_q, clk => clk, aclr => areset );

    -- redist9_excI_x_uid15_fpToFxPTest_q_3(DELAY,131)
    redist9_excI_x_uid15_fpToFxPTest_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => excI_x_uid15_fpToFxPTest_q, xout => redist9_excI_x_uid15_fpToFxPTest_q_3_q, clk => clk, aclr => areset );

    -- fracXIsNotZero_uid14_fpToFxPTest(LOGICAL,13)@3
    fracXIsNotZero_uid14_fpToFxPTest_q <= not (and_lev1_uid83_fracXIsZero_uid13_fpToFxPTest_q);

    -- excN_x_uid16_fpToFxPTest(LOGICAL,15)@3 + 1
    excN_x_uid16_fpToFxPTest_qi <= redist10_expXIsMax_uid12_fpToFxPTest_q_3_q and fracXIsNotZero_uid14_fpToFxPTest_q;
    excN_x_uid16_fpToFxPTest_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "ASYNC" )
    PORT MAP ( xin => excN_x_uid16_fpToFxPTest_qi, xout => excN_x_uid16_fpToFxPTest_q, clk => clk, aclr => areset );

    -- redist8_excN_x_uid16_fpToFxPTest_q_3(DELAY,130)
    redist8_excN_x_uid16_fpToFxPTest_q_3 : dspba_delay
    GENERIC MAP ( width => 1, depth => 2, reset_kind => "ASYNC" )
    PORT MAP ( xin => excN_x_uid16_fpToFxPTest_q, xout => redist8_excN_x_uid16_fpToFxPTest_q_3_q, clk => clk, aclr => areset );

    -- ovfPostRnd_uid48_fpToFxPTest(LOGICAL,47)@6
    ovfPostRnd_uid48_fpToFxPTest_q <= redist8_excN_x_uid16_fpToFxPTest_q_3_q or redist9_excI_x_uid15_fpToFxPTest_q_3_q or redist5_ovfExpRange_uid27_fpToFxPTest_n_6_q or rndOvfPos_uid47_fpToFxPTest_c;

    -- muxSelConc_uid49_fpToFxPTest(BITJOIN,48)@6
    muxSelConc_uid49_fpToFxPTest_q <= redist7_signX_uid25_fpToFxPTest_b_6_q & redist4_udf_uid29_fpToFxPTest_n_6_q & ovfPostRnd_uid48_fpToFxPTest_q;

    -- muxSel_uid50_fpToFxPTest(LOOKUP,49)@6 + 1
    muxSel_uid50_fpToFxPTest_clkproc: PROCESS (clk, areset)
    BEGIN
        IF (areset = '1') THEN
            muxSel_uid50_fpToFxPTest_q <= "00";
        ELSIF (clk'EVENT AND clk = '1') THEN
            CASE (muxSelConc_uid49_fpToFxPTest_q) IS
                WHEN "000" => muxSel_uid50_fpToFxPTest_q <= "00";
                WHEN "001" => muxSel_uid50_fpToFxPTest_q <= "01";
                WHEN "010" => muxSel_uid50_fpToFxPTest_q <= "11";
                WHEN "011" => muxSel_uid50_fpToFxPTest_q <= "11";
                WHEN "100" => muxSel_uid50_fpToFxPTest_q <= "00";
                WHEN "101" => muxSel_uid50_fpToFxPTest_q <= "10";
                WHEN "110" => muxSel_uid50_fpToFxPTest_q <= "11";
                WHEN "111" => muxSel_uid50_fpToFxPTest_q <= "11";
                WHEN OTHERS => -- unreachable
                               muxSel_uid50_fpToFxPTest_q <= (others => '-');
            END CASE;
        END IF;
    END PROCESS;

    -- VCC(CONSTANT,1)
    VCC_q <= "1";

    -- finalOut_uid52_fpToFxPTest(MUX,51)@7
    finalOut_uid52_fpToFxPTest_s <= muxSel_uid50_fpToFxPTest_q;
    finalOut_uid52_fpToFxPTest_combproc: PROCESS (finalOut_uid52_fpToFxPTest_s, redist2_sPostRnd_uid45_fpToFxPTest_b_2_q, maxPosValueS_uid39_fpToFxPTest_q, maxNegValueS_uid40_fpToFxPTest_q, maxNegValueU_uid51_fpToFxPTest_q)
    BEGIN
        CASE (finalOut_uid52_fpToFxPTest_s) IS
            WHEN "00" => finalOut_uid52_fpToFxPTest_q <= redist2_sPostRnd_uid45_fpToFxPTest_b_2_q;
            WHEN "01" => finalOut_uid52_fpToFxPTest_q <= maxPosValueS_uid39_fpToFxPTest_q;
            WHEN "10" => finalOut_uid52_fpToFxPTest_q <= maxNegValueS_uid40_fpToFxPTest_q;
            WHEN "11" => finalOut_uid52_fpToFxPTest_q <= maxNegValueU_uid51_fpToFxPTest_q;
            WHEN OTHERS => finalOut_uid52_fpToFxPTest_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- xOut(GPOUT,4)@7
    q <= finalOut_uid52_fpToFxPTest_q;

END normal;
