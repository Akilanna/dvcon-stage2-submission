`timescale 1ns/1ps
module rankmlp_accel_tb;

  logic ap_clk = 0;
  always #10 ap_clk = ~ap_clk;   // 50 MHz = 20 ns period

  logic ap_rst_n = 0;
  initial begin
    repeat(10) @(posedge ap_clk);
    ap_rst_n = 1;
  end

  // Slave AXI4-Lite
  logic        s_axi_control_AWVALID = 0;
  logic        s_axi_control_AWREADY;
  logic [6:0]  s_axi_control_AWADDR  = 0;
  logic        s_axi_control_WVALID  = 0;
  logic        s_axi_control_WREADY;
  logic [31:0] s_axi_control_WDATA   = 0;
  logic [3:0]  s_axi_control_WSTRB   = 4'hF;
  logic        s_axi_control_BVALID;
  logic        s_axi_control_BREADY  = 1;
  logic [1:0]  s_axi_control_BRESP;
  logic        s_axi_control_ARVALID = 0;
  logic        s_axi_control_ARREADY;
  logic [6:0]  s_axi_control_ARADDR  = 0;
  logic        s_axi_control_RVALID;
  logic        s_axi_control_RREADY  = 1;
  logic [31:0] s_axi_control_RDATA;
  logic [1:0]  s_axi_control_RRESP;
  logic        interrupt;

  // Master AXI4 (64-bit data path)
  logic        m_axi_M_AXI_awvalid;
  logic        m_axi_M_AXI_awready;
  logic [11:0] m_axi_M_AXI_awid;
  logic [63:0] m_axi_M_AXI_awaddr;
  logic [7:0]  m_axi_M_AXI_awlen;
  logic [2:0]  m_axi_M_AXI_awsize;
  logic [1:0]  m_axi_M_AXI_awburst;
  logic [1:0]  m_axi_M_AXI_awlock;
  logic [3:0]  m_axi_M_AXI_awcache;
  logic [2:0]  m_axi_M_AXI_awprot;
  logic [3:0]  m_axi_M_AXI_awqos;
  logic [3:0]  m_axi_M_AXI_awregion;
  logic [0:0]  m_axi_M_AXI_awuser;
  
  logic        m_axi_M_AXI_wvalid;
  logic        m_axi_M_AXI_wready;
  logic [63:0] m_axi_M_AXI_wdata;
  logic [7:0]  m_axi_M_AXI_wstrb;
  logic        m_axi_M_AXI_wlast;
  logic [11:0] m_axi_M_AXI_wid;
  logic [0:0]  m_axi_M_AXI_wuser;

  logic        m_axi_M_AXI_bvalid;
  logic        m_axi_M_AXI_bready;
  logic [11:0] m_axi_M_AXI_bid;
  logic [1:0]  m_axi_M_AXI_bresp;
  logic [0:0]  m_axi_M_AXI_buser;

  logic        m_axi_M_AXI_arvalid;
  logic        m_axi_M_AXI_arready;
  logic [11:0] m_axi_M_AXI_arid;
  logic [63:0] m_axi_M_AXI_araddr;
  logic [7:0]  m_axi_M_AXI_arlen;
  logic [2:0]  m_axi_M_AXI_arsize;
  logic [1:0]  m_axi_M_AXI_arburst;
  logic [1:0]  m_axi_M_AXI_arlock;
  logic [3:0]  m_axi_M_AXI_arcache;
  logic [2:0]  m_axi_M_AXI_arprot;
  logic [3:0]  m_axi_M_AXI_arqos;
  logic [3:0]  m_axi_M_AXI_arregion;
  logic [0:0]  m_axi_M_AXI_aruser;

  logic        m_axi_M_AXI_rvalid;
  logic        m_axi_M_AXI_rready;
  logic [11:0] m_axi_M_AXI_rid;
  logic [63:0] m_axi_M_AXI_rdata;
  logic [1:0]  m_axi_M_AXI_rresp;
  logic        m_axi_M_AXI_rlast;
  logic [0:0]  m_axi_M_AXI_ruser;

  // DUT Instance
  ranking_mlp_top dut (
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    
    // Master
    .m_axi_M_AXI_AWVALID(m_axi_M_AXI_awvalid),
    .m_axi_M_AXI_AWREADY(m_axi_M_AXI_awready),
    .m_axi_M_AXI_AWADDR(m_axi_M_AXI_awaddr),
    .m_axi_M_AXI_AWID(m_axi_M_AXI_awid),
    .m_axi_M_AXI_AWLEN(m_axi_M_AXI_awlen),
    .m_axi_M_AXI_AWSIZE(m_axi_M_AXI_awsize),
    .m_axi_M_AXI_AWBURST(m_axi_M_AXI_awburst),
    .m_axi_M_AXI_AWLOCK(m_axi_M_AXI_awlock),
    .m_axi_M_AXI_AWCACHE(m_axi_M_AXI_awcache),
    .m_axi_M_AXI_AWPROT(m_axi_M_AXI_awprot),
    .m_axi_M_AXI_AWQOS(m_axi_M_AXI_awqos),
    .m_axi_M_AXI_AWREGION(m_axi_M_AXI_awregion),
    .m_axi_M_AXI_AWUSER(m_axi_M_AXI_awuser),
    .m_axi_M_AXI_WVALID(m_axi_M_AXI_wvalid),
    .m_axi_M_AXI_WREADY(m_axi_M_AXI_wready),
    .m_axi_M_AXI_WDATA(m_axi_M_AXI_wdata),
    .m_axi_M_AXI_WSTRB(m_axi_M_AXI_wstrb),
    .m_axi_M_AXI_WLAST(m_axi_M_AXI_wlast),
    .m_axi_M_AXI_WID(m_axi_M_AXI_wid),
    .m_axi_M_AXI_WUSER(m_axi_M_AXI_wuser),
    .m_axi_M_AXI_ARVALID(m_axi_M_AXI_arvalid),
    .m_axi_M_AXI_ARREADY(m_axi_M_AXI_arready),
    .m_axi_M_AXI_ARADDR(m_axi_M_AXI_araddr),
    .m_axi_M_AXI_ARID(m_axi_M_AXI_arid),
    .m_axi_M_AXI_ARLEN(m_axi_M_AXI_arlen),
    .m_axi_M_AXI_ARSIZE(m_axi_M_AXI_arsize),
    .m_axi_M_AXI_ARBURST(m_axi_M_AXI_arburst),
    .m_axi_M_AXI_ARLOCK(m_axi_M_AXI_arlock),
    .m_axi_M_AXI_ARCACHE(m_axi_M_AXI_arcache),
    .m_axi_M_AXI_ARPROT(m_axi_M_AXI_arprot),
    .m_axi_M_AXI_ARQOS(m_axi_M_AXI_arqos),
    .m_axi_M_AXI_ARREGION(m_axi_M_AXI_arregion),
    .m_axi_M_AXI_ARUSER(m_axi_M_AXI_aruser),
    .m_axi_M_AXI_RVALID(m_axi_M_AXI_rvalid),
    .m_axi_M_AXI_RREADY(m_axi_M_AXI_rready),
    .m_axi_M_AXI_RDATA(m_axi_M_AXI_rdata),
    .m_axi_M_AXI_RLAST(m_axi_M_AXI_rlast),
    .m_axi_M_AXI_RID(m_axi_M_AXI_rid),
    .m_axi_M_AXI_RUSER(m_axi_M_AXI_ruser),
    .m_axi_M_AXI_RRESP(m_axi_M_AXI_rresp),
    .m_axi_M_AXI_BVALID(m_axi_M_AXI_bvalid),
    .m_axi_M_AXI_BREADY(m_axi_M_AXI_bready),
    .m_axi_M_AXI_BRESP(m_axi_M_AXI_bresp),
    .m_axi_M_AXI_BID(m_axi_M_AXI_bid),
    .m_axi_M_AXI_BUSER(m_axi_M_AXI_buser),
    
    // Slave control
    .s_axi_control_AWVALID(s_axi_control_AWVALID),
    .s_axi_control_AWREADY(s_axi_control_AWREADY),
    .s_axi_control_AWADDR(s_axi_control_AWADDR),
    .s_axi_control_WVALID(s_axi_control_WVALID),
    .s_axi_control_WREADY(s_axi_control_WREADY),
    .s_axi_control_WDATA(s_axi_control_WDATA),
    .s_axi_control_WSTRB(s_axi_control_WSTRB),
    .s_axi_control_ARVALID(s_axi_control_ARVALID),
    .s_axi_control_ARREADY(s_axi_control_ARREADY),
    .s_axi_control_ARADDR(s_axi_control_ARADDR),
    .s_axi_control_RVALID(s_axi_control_RVALID),
    .s_axi_control_RREADY(s_axi_control_RREADY),
    .s_axi_control_RDATA(s_axi_control_RDATA),
    .s_axi_control_RRESP(s_axi_control_RRESP),
    .s_axi_control_BVALID(s_axi_control_BVALID),
    .s_axi_control_BREADY(s_axi_control_BREADY),
    .s_axi_control_BRESP(s_axi_control_BRESP),
    .interrupt(interrupt)
  );

  // AXI4 Memory Model (64-bit)
  axi4_mem_model mem_model (
    .aclk(ap_clk),
    .aresetn(ap_rst_n),
    
    .m_axi_awvalid(m_axi_M_AXI_awvalid),
    .m_axi_awready(m_axi_M_AXI_awready),
    .m_axi_awid(m_axi_M_AXI_awid),
    .m_axi_awaddr(m_axi_M_AXI_awaddr),
    .m_axi_awlen(m_axi_M_AXI_awlen),
    .m_axi_awsize(m_axi_M_AXI_awsize),
    .m_axi_awburst(m_axi_M_AXI_awburst),
    .m_axi_awlock(m_axi_M_AXI_awlock),
    .m_axi_awcache(m_axi_M_AXI_awcache),
    .m_axi_awprot(m_axi_M_AXI_awprot),
    .m_axi_awqos(m_axi_M_AXI_awqos),
    
    .m_axi_wvalid(m_axi_M_AXI_wvalid),
    .m_axi_wready(m_axi_M_AXI_wready),
    .m_axi_wdata(m_axi_M_AXI_wdata),
    .m_axi_wstrb(m_axi_M_AXI_wstrb),
    .m_axi_wlast(m_axi_M_AXI_wlast),
    
    .m_axi_bvalid(m_axi_M_AXI_bvalid),
    .m_axi_bready(m_axi_M_AXI_bready),
    .m_axi_bid(m_axi_M_AXI_bid),
    .m_axi_bresp(m_axi_M_AXI_bresp),
    
    .m_axi_arvalid(m_axi_M_AXI_arvalid),
    .m_axi_arready(m_axi_M_AXI_arready),
    .m_axi_arid(m_axi_M_AXI_arid),
    .m_axi_araddr(m_axi_M_AXI_araddr),
    .m_axi_arlen(m_axi_M_AXI_arlen),
    .m_axi_arsize(m_axi_M_AXI_arsize),
    .m_axi_arburst(m_axi_M_AXI_arburst),
    .m_axi_arlock(m_axi_M_AXI_arlock),
    .m_axi_arcache(m_axi_M_AXI_arcache),
    .m_axi_arprot(m_axi_M_AXI_arprot),
    .m_axi_arqos(m_axi_M_AXI_arqos),
    
    .m_axi_rvalid(m_axi_M_AXI_rvalid),
    .m_axi_rready(m_axi_M_AXI_rready),
    .m_axi_rid(m_axi_M_AXI_rid),
    .m_axi_rdata(m_axi_M_AXI_rdata),
    .m_axi_rresp(m_axi_M_AXI_rresp),
    .m_axi_rlast(m_axi_M_AXI_rlast)
  );

  task automatic axi_lite_write(input [31:0] addr, input [31:0] data);
    $display("[%0t ns] TB AXI LITE WRITE: addr=%h, data=%h", $time, addr, data);
    @(posedge ap_clk);
    s_axi_control_AWVALID = 1; s_axi_control_AWADDR = addr[6:0];
    s_axi_control_WVALID  = 1; s_axi_control_WDATA  = data;
    fork
      begin: aw_done
        wait(s_axi_control_AWREADY); @(posedge ap_clk);
        s_axi_control_AWVALID = 0;
      end
      begin: w_done
        wait(s_axi_control_WREADY);  @(posedge ap_clk);
        s_axi_control_WVALID  = 0;
      end
    join
    wait(s_axi_control_BVALID);
    @(posedge ap_clk);
  endtask

  task automatic axi_lite_read(input [31:0] addr, output [31:0] data);
    @(posedge ap_clk);
    s_axi_control_ARVALID = 1; s_axi_control_ARADDR = addr[6:0];
    wait(s_axi_control_ARREADY); @(posedge ap_clk);
    s_axi_control_ARVALID = 0;
    wait(s_axi_control_RVALID);
    data = s_axi_control_RDATA;
    $display("[%0t ns] TB AXI LITE READ: addr=%h, data=%h", $time, addr, data);
    @(posedge ap_clk);
  endtask

  initial begin
    wait(ap_rst_n);
    repeat(5) @(posedge ap_clk);

    $display("[%0t ns] --- SYSTEMVERILOG TESTBENCH START ---", $time);
    $display("[%0t ns] Configuring registers via AXI4-Lite...", $time);
    
    // Address offsets match Vitis HLS 2024.2 control register map
    axi_lite_write(32'h10, 32'h8000_0000);  // DDR_INPUT LSB (base of input vectors)
    axi_lite_write(32'h14, 32'h0000_0000);  // DDR_INPUT MSB
    axi_lite_write(32'h1C, 32'h8001_0000);  // DDR_OUTPUT LSB (base of output buffer)
    axi_lite_write(32'h20, 32'h0000_0000);  // DDR_OUTPUT MSB
    axi_lite_write(32'h28, 32'h00000014);  // N_OBJECTS = 20
    axi_lite_write(32'h30, 32'h0000_007B);  // TASK_ID = 123
    
    $display("[%0t ns] Configuration complete. Starting accelerator (ap_start)...", $time);
    axi_lite_write(32'h00, 32'h0000_0001);  // ap_ctrl: ap_start = 1

    $display("[%0t ns] Waiting for AXI Master burst transactions...", $time);

    begin
      logic [31:0] ctrl_val;
      int timeout = 0;
      ctrl_val = 0;
      while (ctrl_val[1] == 0 && timeout < 100000) begin
        repeat(50) @(posedge ap_clk);
        axi_lite_read(32'h00, ctrl_val);
        timeout++;
      end
      
      if (timeout >= 100000) begin
        $display("[%0t ns] ERROR: Timeout waiting for accelerator ap_done!", $time);
        $finish;
      end else begin
        $display("[%0t ns] Accelerator finished! (ap_done detected after %0d polls)", $time, timeout);
      end
    end

    // Verify output scores from DDR3 memory model
    begin
      shortreal expected [20];
      shortreal score;
      logic [63:0] raw_out;
      int passed = 1;

      expected[0] = -0.063477;
      expected[1] = 0.026367;
      expected[2] = 0.096680;
      expected[3] = 0.126953;
      expected[4] = -0.120117;
      expected[5] = -0.275391;
      expected[6] = -0.294922;
      expected[7] = -0.229492;
      expected[8] = -0.118164;
      expected[9] = -0.123047;
      expected[10] = -0.131836;
      expected[11] = -0.101562;
      expected[12] = -0.064453;
      expected[13] = -0.084961;
      expected[14] = 0.083984;
      expected[15] = 0.066406;
      expected[16] = 0.052734;
      expected[17] = -0.063477;
      expected[18] = 0.026367;
      expected[19] = 0.096680;

      $display("[%0t ns] --- Verification Results in RTL Simulation ---", $time);
      for (int i = 0; i < 20; i++) begin
        raw_out = mem_model.mem[mem_model.addr_to_idx(64'h8001_0000 + i*8)];
        score = shortreal'($signed(raw_out[15:0])) / 1024.0;
        $display("  Object %0d score: %f (Expected: %f)", i, score, expected[i]);
        if (score < expected[i] - 0.001 || score > expected[i] + 0.001) begin
          passed = 0;
        end
      end

      if (passed) begin
        $display("[%0t ns] RTL SIMULATION PASSED!", $time);
      end else begin
        $display("[%0t ns] ERROR: Score mismatch in RTL simulation!", $time);
      end
    end

    // Read and print performance counters from registers
    begin
      logic [31:0] cycles;
      logic [31:0] objects;
      axi_lite_read(32'h38, cycles);
      axi_lite_read(32'h48, objects);
      $display("Performance Telemetry Registers:");
      $display("  perf_total_cycles: %0d", cycles);
      $display("  perf_object_count: %0d", objects);
    end

    repeat(50) @(posedge ap_clk);
    $display("[%0t ns] --- SYSTEMVERILOG TESTBENCH COMPLETE ---", $time);
    $finish;
  end

endmodule
