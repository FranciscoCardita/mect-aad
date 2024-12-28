library ieee;
use ieee.std_logic_1164.all;

entity accumulator is
    generic (
        ADDR_BITS       : integer range 2 to 8 := 4;
        DATA_BITS_LOG2  : integer range 1 to 4 := 3
    );
    port (
        clock      : in  std_logic;
        write_addr : in  std_logic_vector(ADDR_BITS-1 downto 0);
        write_inc  : in  std_logic_vector(2**DATA_BITS_LOG2-1 downto 0);
        write_shift: in  std_logic_vector(DATA_BITS_LOG2-1 downto 0);
        read_addr  : in  std_logic_vector(ADDR_BITS-1 downto 0);
        read_data  : out std_logic_vector(2**DATA_BITS_LOG2-1 downto 0)
    );
end accumulator;

architecture structural of accumulator is
    signal s_write_addr_stable : std_logic_vector(ADDR_BITS-1 downto 0);
    signal s_write_inc_stable  : std_logic_vector(2**DATA_BITS_LOG2-1 downto 0);
    signal s_write_shift_stable: std_logic_vector(DATA_BITS_LOG2-1 downto 0);
    signal s_shifted_write_inc : std_logic_vector(2**DATA_BITS_LOG2-1 downto 0);
    signal s_value_to_write    : std_logic_vector(2**DATA_BITS_LOG2-1 downto 0);
    signal s_aux_read_data     : std_logic_vector(2**DATA_BITS_LOG2-1 downto 0);
begin
    addr_reg : entity work.vector_register(behavioral)
        generic map (
            DATA_BITS => ADDR_BITS
        )
        port map (
            clock => clock,
            d     => write_addr,
            q     => s_write_addr_stable,
            en    => '1'
        );
    inc_reg : entity work.vector_register(behavioral)
        generic map (
            DATA_BITS => 2**DATA_BITS_LOG2
        )
        port map (
            clock => clock,
            d     => write_inc,
            q     => s_write_inc_stable,
            en    => '1'
        );
    shift_reg : entity work.vector_register(behavioral)
        generic map (
            DATA_BITS => DATA_BITS_LOG2
        )
        port map (
            clock => clock,
            d     => write_shift,
            q     => s_write_shift_stable,
            en    => '1'
        );
    shifter : entity work.barrel_shifter(behavioral)
        generic map (
            DATA_BITS_LOG2 => DATA_BITS_LOG2
        )
        port map (
            data_in  => s_write_inc_stable,
            shift    => s_write_shift_stable,
            data_out => s_shifted_write_inc
        );
    adder : entity work.adder_n(behavioral)
        generic map (
            N => 2**DATA_BITS_LOG2
        )
        port map (
            a    => s_aux_read_data,
            b    => s_shifted_write_inc,
            c_in => '0',
            s    => s_value_to_write,
            c_out=> open
        );
    memory : entity work.triple_port_ram(behavioral)
        generic map (
            ADDR_BITS => ADDR_BITS,
            DATA_BITS => 2**DATA_BITS_LOG2
        )
        port map (
            clock         => clock,
            write_addr    => s_write_addr_stable,
            write_data    => s_value_to_write,
            read_addr     => read_addr,
            read_data     => read_data,
            aux_read_addr => s_write_addr_stable,
            aux_read_data => s_aux_read_data
        );
end structural;
