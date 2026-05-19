module I2C_Config (
    input  logic        iClk,
    input  logic        iRST_n,
    // I2C Controller Interface
    output logic        oStart,
    output logic [6:0]  oSlave_Addr,
    output logic [7:0]  oWord_Addr,
    output logic [7:0]  owdata,
    output logic        owcmd,
    input  logic [7:0]  iReadData,
    input  logic        iReadData_rdy,
    input  logic        iCONFIG_DONE,
    // Fan Control Interface
    input  logic [7:0]  Speed_Set,
    input  logic        Alert,
    input  logic        Alert_Clear,
    output logic [12:0] Speed_Detected,
    output logic [3:0]  Alert_Type
);

    // I2C Registers (Constants)
    localparam SLAVE_ADDR       = 7'h48; // 72 (Decimal)
    localparam REG_SET_RPM      = 8'h00;
    localparam REG_READ_RPS     = 8'h0C;
    localparam REG_READ_STATUS  = 8'h0A;
    localparam INIT_SIZE        = 5;

    // FSM States
    typedef enum logic [2:0] {
        ST_INIT,
        ST_IDLE,
        ST_WRITE_SPEED,
        ST_READ_DATA,
        ST_WAIT_DONE
    } state_t;

    state_t state, next_return_state;
    logic [3:0] init_index;
    logic [7:0] prev_speed;
    logic       config_done_prev;
    logic       config_done_edge;

    // CONFIG_DONE の立ち上がりエッジ検出
    always_ff @(posedge iClk or negedge iRST_n) begin
        if (!iRST_n) config_done_prev <= 1'b0;
        else         config_done_prev <= iCONFIG_DONE;
    end
    assign config_done_edge = iCONFIG_DONE && !config_done_prev;

    // 初期化データテーブル
    logic [15:0] init_rom [INIT_SIZE] = '{
        16'h004e, // 0: Default speed
        16'h022a, // 1: Config mode
        16'h04f5, // 2: GPIO Definition
        16'h0807, // 3: Alarm Enable
        16'h1602  // 4: Tachometer Count-Time
    };

    // --- メインステートマシン ---
    always_ff @(posedge iClk or negedge iRST_n) begin
        if (!iRST_n) begin
            state <= ST_INIT;
            init_index <= 4'd0;
            oStart <= 1'b0;
            oSlave_Addr <= SLAVE_ADDR;
            prev_speed <= 8'd0;
            Speed_Detected <= 13'd0;
            Alert_Type <= 4'd0;
        end else begin
            case (state)
                // 1. 初期化シーケンス
                ST_INIT: begin
                    if (init_index < INIT_SIZE) begin
                        oWord_Addr <= init_rom[init_index][15:8];
                        owdata     <= init_rom[init_index][7:0];
                        owcmd      <= 1'b1; // Write
                        oStart     <= 1'b1;
                        state      <= ST_WAIT_DONE;
                        next_return_state <= ST_INIT;
                        init_index <= init_index + 4'd1;
                    end else begin
                        state <= ST_IDLE;
                    end
                end

                // 2. 待機状態（変化監視）
                ST_IDLE: begin
                    oStart <= 1'b0;
                    if (Speed_Set != prev_speed) begin
                        state <= ST_WRITE_SPEED;
                    end else begin
                        state <= ST_READ_DATA;
                    end
                end

                // 3. スピード設定の書き込み
                ST_WRITE_SPEED: begin
                    oWord_Addr <= REG_SET_RPM;
                    owdata     <= Speed_Set;
                    owcmd      <= 1'b1;
                    oStart     <= 1'b1;
                    prev_speed <= Speed_Set;
                    state      <= ST_WAIT_DONE;
                    next_return_state <= ST_IDLE;
                end

                // 4. ステータスまたは回転数の読み出し
                ST_READ_DATA: begin
                    oWord_Addr <= Alert ? REG_READ_RPS : REG_READ_STATUS;
                    owcmd      <= 1'b0; // Read
                    oStart     <= 1'b1;
                    state      <= ST_WAIT_DONE;
                    next_return_state <= ST_IDLE;
                end

                // 5. I2C完了待ち
                ST_WAIT_DONE: begin
                    oStart <= 1'b0; // Startは1クロックだけ出す
                    if (config_done_edge) begin
                        // 読み出しデータの確定処理
                        if (owcmd == 1'b0) begin
                            if (oWord_Addr == REG_READ_RPS) begin
                                Speed_Detected <= (iReadData * 60) >> 1;
                            end else if (oWord_Addr == REG_READ_STATUS) begin
                                if (Alert_Clear) Alert_Type <= 4'd0;
                                else             Alert_Type <= iReadData[3:0];
                            end
                        end
                        state <= next_return_state;
                    end
                end

                default: state <= ST_IDLE;
            endcase
            
            // Alert_Clearは常に監視
            if (Alert_Clear) Alert_Type <= 4'd0;
        end
    end

endmodule
