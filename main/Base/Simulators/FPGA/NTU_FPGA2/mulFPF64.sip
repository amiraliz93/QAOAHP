set_global_assignment -entity "mulFPF64" -library "lib_mulFPF64" -name IP_TOOL_NAME "altera_fp_functions"
set_global_assignment -entity "mulFPF64" -library "lib_mulFPF64" -name IP_TOOL_VERSION "17.1"
set_global_assignment -entity "mulFPF64" -library "lib_mulFPF64" -name IP_TOOL_ENV "mwpim"
set_global_assignment -library "lib_mulFPF64" -name SPD_FILE [file join $::quartus(sip_path) "mulFPF64.spd"]

set_global_assignment -library "lib_mulFPF64" -name MISC_FILE [file join $::quartus(sip_path) "mulFPF64_sim/mulFPF64.vo"]
