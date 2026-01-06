set_global_assignment -entity "Cordic168" -library "lib_Cordic168" -name IP_TOOL_NAME "Qsys"
set_global_assignment -entity "Cordic168" -library "lib_Cordic168" -name IP_TOOL_VERSION "17.1"
set_global_assignment -entity "Cordic168" -library "lib_Cordic168" -name IP_TOOL_ENV "Qsys"
set_global_assignment -library "lib_Cordic168" -name SPD_FILE [file join $::quartus(sip_path) "../Cordic168.spd"]
set_global_assignment -library "lib_Cordic168" -name MISC_FILE [file join $::quartus(sip_path) "../../Cordic168.qsys"]

set_global_assignment -library "lib_Cordic168" -name MISC_FILE [file join $::quartus(sip_path) "Cordic168.v"]
set_global_assignment -library "lib_Cordic168" -name MISC_FILE [file join $::quartus(sip_path) "submodules/dspba_library_package.vhd"]
set_global_assignment -library "lib_Cordic168" -name MISC_FILE [file join $::quartus(sip_path) "submodules/dspba_library.vhd"]
set_global_assignment -library "lib_Cordic168" -name MISC_FILE [file join $::quartus(sip_path) "submodules/Cordic168_CORDIC_0.vhd"]
