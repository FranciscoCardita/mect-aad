library IEEE;
use     IEEE.std_logic_1164.all;
use     IEEE.numeric_std.all;

entity barrel_shifter_tb is
end barrel_shifter_tb;

architecture stimulus of barrel_shifter_tb is
    constant DATA_BITS_LOG2 : positive := 3;
    signal s_data_in  : std_logic_vector(2**DATA_BITS_LOG2-1 downto 0);
    signal s_data_out : std_logic_vector(2**DATA_BITS_LOG2-1 downto 0);
    signal s_shift    : std_logic_vector(DATA_BITS_LOG2-1 downto 0);
begin
    uut : entity work.barrel_shifter(behavioral)
                generic map
                (
                    DATA_BITS_LOG2 => DATA_BITS_LOG2
                )
                port map
                (
                    data_in  => s_data_in,
                    shift    => s_shift,
                    data_out => s_data_out
                );

    signals_stim : process is
    begin
        s_data_in <= (0 => '1', others => '0');
        for_1 : for i in 0 to 2**DATA_BITS_LOG2-1 loop
            s_shift <= std_logic_vector(to_unsigned(i, DATA_BITS_LOG2));
            wait for 250 ps;
        end loop;
        s_data_in <= (others => '1');
        for_2 : for i in 0 to 2**DATA_BITS_LOG2-1 loop
            s_shift <= std_logic_vector(to_unsigned(i, DATA_BITS_LOG2));
            wait for 250 ps;
        end loop;
        wait;
    end process;
end stimulus;