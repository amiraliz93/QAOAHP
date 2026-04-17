set_global_assignment -entity "cordicR" -library "lib_cordicR" -name IP_TOOL_NAME "Qsys"
set_global_assignment -entity "cordicR" -library "lib_cordicR" -name IP_TOOL_VERSION "25.1"
set_global_assignment -entity "cordicR" -library "lib_cordicR" -name IP_TOOL_ENV "Qsys"
set_global_assignment -library "lib_cordicR" -name SPD_FILE [file join $::quartus(sip_path) "../cordicR.spd"]
set_global_assignment -library "lib_cordicR" -name MISC_FILE [file join $::quartus(sip_path) "../../cordicR.qsys"]

set_global_assignment -library "lib_cordicR" -name MISC_FILE [file join $::quartus(sip_path) "cordicR.v"]
set_global_assignment -library "lib_cordicR" -name MISC_FILE [file join $::quartus(sip_path) "submodules/dspba_library_package.vhd"]
set_global_assignment -library "lib_cordicR" -name MISC_FILE [file join $::quartus(sip_path) "submodules/dspba_library.vhd"]
set_global_assignment -library "lib_cordicR" -name MISC_FILE [file join $::quartus(sip_path) "submodules/cordicR_CORDIC_0.vhd"]
