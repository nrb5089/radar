======================================
Lab 2 - Schedulers and State Machines
======================================

Basic Pulse Controller
==========================

Below is an example of a state machine in Verilog for scheduling pulses at different times. This state machine generates a pulse for a duration of 5 clock cycles, then stays low for 10 clock cycles, and then repeats. It uses a counter to keep track of the number of clock cycles that have passed.

.. code-block:: verilog

    module PulseGenerator (
        input wire clk,
        input wire rst_n,
        output reg pulse
    );

        parameter IDLE = 2'b00, PULSE_HIGH = 2'b01, PULSE_LOW = 2'b10;
        reg [1:0] state, next_state;
        reg [4:0] counter;

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                state <= IDLE;
                counter <= 5'b00000;
                pulse <= 1'b0;
            end else begin
                state <= next_state;
                if (state == PULSE_HIGH)
                    pulse <= 1'b1;
                else
                    pulse <= 1'b0;

                if (counter == 5'b00000)
                    counter <= 5'b11111;
                else
                    counter <= counter - 1'b1;
            end
        end

        always @(*) begin
            next_state = state;
            case (state)
                IDLE: begin
                    if (counter == 5'b00000)
                        next_state = PULSE_HIGH;
                end

                PULSE_HIGH: begin
                    if (counter == 5'b00000)
                        next_state = PULSE_LOW;
                end

                PULSE_LOW: begin
                    if (counter == 5'b00000)
                        next_state = PULSE_HIGH;
                end
            endcase
        end

    endmodule


Multiple PRI Controller
===========================

This state machine has three states: ``IDLE``, ``PULSE_HIGH`` and ``PULSE_LOW``. It starts in the ``IDLE`` state and moves to ``PULSE_HIGH`` after 5 clock cycles, where it stays for 5 clock cycles before moving to ``PULSE_LOW``. It stays in ``PULSE_LOW`` for 10 clock cycles before returning to ``PULSE_HIGH``, and so on.  This is a basic example, and your specific requirements might need different adjustments, such as changing the number of clock cycles for each pulse or low period, or adding more states.  Next is a state machine with 8 different states for 8 different pulse repetition intervals. All pulses have the same width, which is 5 cycles in this case.

This state machine will transition from one state to another after each pulse repetition interval. For example, in the first state, it will generate a pulse, wait for 5 cycles, then generate another pulse. In the second state, it will generate a pulse, wait for 10 cycles, then generate another pulse, and so forth.

Here's an example of how you might code this:

.. code-block:: verilog

    module PulseGenerator (
        input wire clk,
        input wire rst_n,
        output reg pulse
    );

        // Define state IDs
        parameter [2:0] 
            STATE_5  = 3'b000,
            STATE_10 = 3'b001,
            STATE_15 = 3'b010,
            STATE_20 = 3'b011,
            STATE_25 = 3'b100,
            STATE_30 = 3'b101,
            STATE_35 = 3'b110,
            STATE_40 = 3'b111;

        // Current state
        reg [2:0] state;

        // Pulse width counter and interval counter
        reg [4:0] pulse_width_counter;
        reg [5:0] interval_counter;

        // FSM Logic
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                state <= STATE_5;
                pulse <= 1'b0;
                pulse_width_counter <= 5'd5;  // Pulse width is 5 cycles
                interval_counter <= 6'd5;  // First interval is 5 cycles
            end else begin
                // Handle pulse width counter
                if (pulse_width_counter > 1) begin
                    pulse <= 1'b1;
                    pulse_width_counter <= pulse_width_counter - 1'b1;
                end else begin
                    pulse <= 1'b0;
                    if (interval_counter > 1) 
                        interval_counter <= interval_counter - 1'b1;
                    else
                        // Move to the next state
                        case (state)
                            STATE_5:  begin state <= STATE_10; interval_counter <= 6'd10; end
                            STATE_10: begin state <= STATE_15; interval_counter <= 6'd15; end
                            STATE_15: begin state <= STATE_20; interval_counter <= 6'd20; end
                            STATE_20: begin state <= STATE_25; interval_counter <= 6'd25; end
                            STATE_25: begin state <= STATE_30; interval_counter <= 6'd30; end
                            STATE_30: begin state <= STATE_35; interval_counter <= 6'd35; end
                            STATE_35: begin state <= STATE_40; interval_counter <= 6'd40; end
                            STATE_40: begin state <= STATE_5;  interval_counter <= 6'd5;  end
                        endcase
                end
            end
        end
    endmodule


This is a reduced state machine, and in a real-world situation, you might need to add more features to handle corner cases, errors, or specific application requirements. Please remember to test this code in your environment as different applications may need different adjustments.   The scheduler generally has built in processing states (that may be done in parallel), here's an example of a state machine which goes through 8 pulse repetition intervals, moves to a data processing state, and repeats the entire sequence 4 times. After four complete sequences, it stays in the IDLE state until a reset signal is received.

.. code-block:: verilog

    module PulseGenerator (
        input wire clk,
        input wire rst_n,
        output reg pulse,
        output reg processing
    );

        // Define state IDs
        parameter [3:0] 
            IDLE     = 4'b0000,
            STATE_5  = 4'b0001,
            STATE_10 = 4'b0010,
            STATE_15 = 4'b0011,
            STATE_20 = 4'b0100,
            STATE_25 = 4'b0101,
            STATE_30 = 4'b0110,
            STATE_35 = 4'b0111,
            STATE_40 = 4'b1000,
            PROCESS  = 4'b1001;

        // Current state
        reg [3:0] state;

        // Pulse width counter and interval counter
        reg [4:0] pulse_width_counter;
        reg [6:0] interval_counter;

        // Repeat counter
        reg [2:0] repeat_counter;

        // FSM Logic
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                state <= IDLE;
                pulse <= 1'b0;
                processing <= 1'b0;
                pulse_width_counter <= 5'd5;  // Pulse width is 5 cycles
                interval_counter <= 7'd5;  // First interval is 5 cycles
                repeat_counter <= 3'd4;  // Repeat 4 times
            end else begin
                // Handle pulse width counter
                if (pulse_width_counter > 1) begin
                    pulse <= 1'b1;
                    pulse_width_counter <= pulse_width_counter - 1'b1;
                end else begin
                    pulse <= 1'b0;
                    if (interval_counter > 1) 
                        interval_counter <= interval_counter - 1'b1;
                    else
                        // Move to the next state
                        case (state)
                            IDLE:     begin state <= STATE_5; interval_counter <= 7'd5; end
                            STATE_5:  begin state <= STATE_10; interval_counter <= 7'd10; end
                            STATE_10: begin state <= STATE_15; interval_counter <= 7'd15; end
                            STATE_15: begin state <= STATE_20; interval_counter <= 7'd20; end
                            STATE_20: begin state <= STATE_25; interval_counter <= 7'd25; end
                            STATE_25: begin state <= STATE_30; interval_counter <= 7'd30; end
                            STATE_30: begin state <= STATE_35; interval_counter <= 7'd35; end
                            STATE_35: begin state <= STATE_40; interval_counter <= 7'd40; end
                            STATE_40: begin state <= PROCESS; end
                            PROCESS:  begin 
                                if (repeat_counter > 1) begin
                                    state <= STATE_5;
                                    interval_counter <= 7'd5;
                                    repeat_counter <= repeat_counter - 1'b1;
                                end else begin
                                    state <= IDLE;
                                end
                            end
                        endcase
                end
                processing <= (state == PROCESS);
            end
        end
    endmodule


In this example, a new ``PROCESS`` state is added to the state machine. After going through the 8 pulse states, the state machine moves to the ``PROCESS`` state. After staying in the ``PROCESS`` state for one cycle, it checks the repeat_counter. If the repeat_counter is greater than 1, it repeats the whole process by moving back to the first pulse state. Otherwise, it moves to the IDLE state. The processing output signal is high when the state machine is in the ``PROCESS`` state and low otherwise.

Nested Scheduling
===================

The main takeaway here is that there are timed intervals within timed intervals for scheduling, here's an example of what several processing intervals inside a ``SUBFRAME_A`` state that performs a *processing* operation 5 times. After the fifth time, it transitions to the ``SUBFRAME_B`` state:

.. code-block:: verilog

    module SubframeStateMachine (
        input wire clk,
        input wire rst_n,
        output reg processing
    );

        // Define state IDs
        parameter [2:0] 
            IDLE      = 3'b000,
            SUBFRAME_A = 3'b001,
            SUBFRAME_B = 3'b010;

        // Current state
        reg [2:0] state;

        // Processing counter
        reg [3:0] processing_counter;

        // FSM Logic
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                state <= IDLE;
                processing <= 1'b0;
                processing_counter <= 4'd0;
            end else begin
                // Move to the next state
                case (state)
                    IDLE: begin 
                        state <= SUBFRAME_A; 
                        processing_counter <= 4'd5;
                    end

                    SUBFRAME_A: begin
                        processing <= 1'b1;  // Perform processing
                        if (processing_counter > 1)
                            processing_counter <= processing_counter - 1'b1;
                        else
                            state <= SUBFRAME_B;
                    end

                    SUBFRAME_B: begin
                        processing <= 1'b0;  // No processing
                        // More logic can be added here for what needs to be done in SUBFRAME_B
                    end
                endcase
            end
        end
    endmodule


In this code, the state machine starts in an ``IDLE`` state. When it transitions to the ``SUBFRAME_A`` state, it performs some *processing* operation (represented by the *processing* output signal being set to 1) five times. After the fifth *processing* operation, the state machine transitions to the ``SUBFRAME_B`` state.

.. note::

    Much of this lab was autogenerated, but the code should reflect what type of logic needs to be implemented.