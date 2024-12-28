library IEEE;
use     IEEE.std_logic_1164.all;

entity vector_register is
    generic
    (
        DATA_BITS : integer range 1 to 32
    );
    port(
        clock : in std_logic;
        d     : in std_logic_vector(DATA_BITS-1 downto 0);
        q     : out std_logic_vector(DATA_BITS-1 downto 0) := (others => '0');
        en    : in std_logic
    );
end vector_register;

architecture behavioral of vector_register is
    begin
    process(clock) is
    begin
        if rising_edge(clock) and en = '1' then
            q <= transport d after 10 ps;
        end if;
    end process;
end behavioral;