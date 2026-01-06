set_global_assignment -entity "FP2Fix53" -library "lib_FP2Fix53" -name IP_TOOL_NAME "altera_fp_functions"
set_global_assignment -entity "FP2Fix53" -library "lib_FP2Fix53" -name IP_TOOL_VERSION "17.1"
set_global_assignment -entity "FP2Fix53" -library "lib_FP2Fix53" -name IP_TOOL_ENV "mwpim"
set_global_assignment -library "lib_FP2Fix53" -name SPD_FILE [file join $::quartus(sip_path) "FP2Fix53.spd"]

set_global_assignment -library "lib_FP2Fix53" -name MISC_FILE [file join $::quartus(sip_path) "FP2Fix53_sim/FP2Fix53.vo"]
