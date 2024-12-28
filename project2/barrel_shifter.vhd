library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity barrel_shifter is
    generic (
        DATA_BITS_LOG2 : positive := 3
    );
    Port (
        data_in  : in  std_logic_vector((2**DATA_BITS_LOG2)-1 downto 0);
        shift    : in  std_logic_vector(DATA_BITS_LOG2-1 downto 0);
        data_out : out std_logic_vector((2**DATA_BITS_LOG2)-1 downto 0)
    );
end barrel_shifter;

architecture behavioral of barrel_shifter is
    type stage_array is array (0 to DATA_BITS_LOG2)
        of std_logic_vector((2**DATA_BITS_LOG2)-1 downto 0);
    signal shift_stages : stage_array;

begin
    shift_stages(0) <= data_in;

    gen_stages: for i in 0 to DATA_BITS_LOG2-1 generate
        stage_x: entity work.shift_slice
            generic map (
                DATA_BITS => 2**DATA_BITS_LOG2,
                SHIFT_AMOUNT => 2**i
            )
            port map (
                data_in  => shift_stages(i),
                data_out => shift_stages(i+1),
                sel      => shift(i)
            );
    end generate;

    data_out <= shift_stages(DATA_BITS_LOG2);

end behavioral;