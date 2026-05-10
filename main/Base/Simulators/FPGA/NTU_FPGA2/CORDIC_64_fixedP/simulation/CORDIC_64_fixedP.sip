set_global_assignment -entity "CORDIC_64_fixedP" -library "lib_CORDIC_64_fixedP" -name IP_TOOL_NAME "Qsys"
set_global_assignment -entity "CORDIC_64_fixedP" -library "lib_CORDIC_64_fixedP" -name IP_TOOL_VERSION "25.1"
set_global_assignment -entity "CORDIC_64_fixedP" -library "lib_CORDIC_64_fixedP" -name IP_TOOL_ENV "Qsys"
set_global_assignment -library "lib_CORDIC_64_fixedP" -name SPD_FILE [file join $::quartus(sip_path) "../CORDIC_64_fixedP.spd"]
set_global_assignment -library "lib_CORDIC_64_fixedP" -name MISC_FILE [file join $::quartus(sip_path) "../../CORDIC_64_fixedP.qsys"]

set_global_assignment -library "lib_CORDIC_64_fixedP" -name MISC_FILE [file join $::quartus(sip_path) "CORDIC_64_fixedP.v"]
set_global_assignment -library "lib_CORDIC_64_fixedP" -name MISC_FILE [file join $::quartus(sip_path) "submodules/dspba_library_package.vhd"]
set_global_assignment -library "lib_CORDIC_64_fixedP" -name MISC_FILE [file join $::quartus(sip_path) "submodules/dspba_library.vhd"]
set_global_assignment -library "lib_CORDIC_64_fixedP" -name MISC_FILE [file join $::quartus(sip_path) "submodules/CORDIC_64_fixedP_CORDIC_0.vhd"]
