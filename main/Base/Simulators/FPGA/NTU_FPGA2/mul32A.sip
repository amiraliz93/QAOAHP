set_global_assignment -entity "mul32A" -library "lib_mul32A" -name IP_TOOL_NAME "altera_mult_add"
set_global_assignment -entity "mul32A" -library "lib_mul32A" -name IP_TOOL_VERSION "17.1"
set_global_assignment -entity "mul32A" -library "lib_mul32A" -name IP_TOOL_ENV "mwpim"
set_global_assignment -library "lib_mul32A" -name SPD_FILE [file join $::quartus(sip_path) "mul32A.spd"]

set_global_assignment -library "lib_mul32A" -name MISC_FILE [file join $::quartus(sip_path) "mul32A_sim/mul32A.v"]
