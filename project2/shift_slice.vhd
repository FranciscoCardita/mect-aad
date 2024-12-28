library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity shift_slice is
    generic (
        DATA_BITS       : integer range 1 to 32 := 4;
        SHIFT_AMOUNT    : integer range 1 TO 32 := 1
    );
    Port (
            data_in     : in    std_logic_vector(DATA_BITS-1 downto 0);
            data_out    : out   std_logic_vector(DATA_BITS-1 downto 0);
            sel         : in    std_logic
        );
end shift_slice;

architecture Behavioral of shift_slice is
begin
    process(data_in, sel)
        variable zeros : std_logic_vector(SHIFT_AMOUNT-1 downto 0) := (others => '0');
    begin
        if sel = '0' then
            data_out <= data_in;
        else
            data_out <= data_in(DATA_BITS-1-SHIFT_AMOUNT downto 0) & zeros;
        end if;
    end process;
end Behavioral;