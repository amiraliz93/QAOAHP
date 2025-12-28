set_global_assignment -entity "fp2fix64" -library "lib_fp2fix64" -name IP_TOOL_NAME "altera_fp_functions"
set_global_assignment -entity "fp2fix64" -library "lib_fp2fix64" -name IP_TOOL_VERSION "17.1"
set_global_assignment -entity "fp2fix64" -library "lib_fp2fix64" -name IP_TOOL_ENV "mwpim"
set_global_assignment -library "lib_fp2fix64" -name SPD_FILE [file join $::quartus(sip_path) "fp2fix64.spd"]

set_global_assignment -library "lib_fp2fix64" -name MISC_FILE [file join $::quartus(sip_path) "fp2fix64_sim/fp2fix64.vo"]
