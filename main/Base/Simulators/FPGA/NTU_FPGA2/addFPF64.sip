set_global_assignment -entity "addFPF64" -library "lib_addFPF64" -name IP_TOOL_NAME "altera_fp_functions"
set_global_assignment -entity "addFPF64" -library "lib_addFPF64" -name IP_TOOL_VERSION "17.1"
set_global_assignment -entity "addFPF64" -library "lib_addFPF64" -name IP_TOOL_ENV "mwpim"
set_global_assignment -library "lib_addFPF64" -name SPD_FILE [file join $::quartus(sip_path) "addFPF64.spd"]

set_global_assignment -library "lib_addFPF64" -name MISC_FILE [file join $::quartus(sip_path) "addFPF64_sim/addFPF64.vo"]
