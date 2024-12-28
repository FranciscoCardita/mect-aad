clean:
	rm -f work-*.cf barrel_shifter_tb.vcd accumulator_tb.vcd

.PHONY: barrel_shifter_tb.vcd
barrel_shifter_tb.vcd:
	rm -f work-*.cf
	ghdl -i --std=08 barrel_shifter.vhd shift_slice.vhd barrel_shifter_tb.vhd
	ghdl -m --std=08 barrel_shifter_tb
	ghdl -r --std=08 barrel_shifter_tb --stop-time=1000ns --vcd=barrel_shifter_tb.vcd

.PHONY: accumulator_tb.vcd
accumulator_tb.vcd:
	rm -f work-*.cf
	ghdl -i --std=08 accumulator.vhd accumulator_tb.vhd adder_n.vhd barrel_shifter.vhd full_adder.vhd shift_slice.vhd triple_port_ram.vhd vector_register.vhd xor_gate_3.vhd and_gate_2.vhd or_gate_3.vhd
	ghdl -m --std=08 accumulator_tb
	ghdl -r --std=08 accumulator_tb --stop-time=1000ns --vcd=accumulator_tb.vcd