library ieee;
use ieee.std_logic_1164.all;

entity accumulator is
    generic (
        ADDR_BITS       : integer range 2 to 8 := 4;
        DATA_BITS       : integer range 1 to 32
    );
    port (
        clock      : in  std_logic;
        write_addr : in  std_logic_vector(ADDR_BITS-1 downto 0);
        write_inc  : in  std_logic_vector(DATA_BITS-1 downto 0);
        read_addr  : in  std_logic_vector(ADDR_BITS-1 downto 0);
        read_data  : out std_logic_vector(DATA_BITS-1 downto 0)
    );
end accumulator;

architecture structural of accumulator is
    signal s_write_addr_stable : std_logic_vector(ADDR_BITS-1 downto 0);
    signal s_write_inc_stable  : std_logic_vector(DATA_BITS-1 downto 0);
    signal s_value_to_write    : std_logic_vector(DATA_BITS-1 downto 0);
    signal s_aux_read_data     : std_logic_vector(DATA_BITS-1 downto 0);
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
            DATA_BITS => DATA_BITS
        )
        port map (
            clock => clock,
            d     => write_inc,
            q     => s_write_inc_stable,
            en    => '1'
        );
    adder : entity work.adder_n(behavioral)
        generic map (
            N => DATA_BITS
        )
        port map (
            a    => s_aux_read_data,
            b    => s_write_inc_stable,
            c_in => '0',
            s    => s_value_to_write,
            c_out=> open
        );
    memory : entity work.triple_port_ram(behavioral)
        generic map (
            ADDR_BITS => ADDR_BITS,
            DATA_BITS => DATA_BITS
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
