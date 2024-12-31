--
-- AAD 2024/2025, accumulator test bench
--

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity accumulator_tb is
end accumulator_tb;

architecture stimulus of accumulator_tb is
  constant ADDR_BITS  : positive := 4;
  constant DATA_BITS  : positive := 8;
  signal clock        : std_logic;
  signal s_write_addr : std_logic_vector(ADDR_BITS-1 downto 0);
  signal s_write_inc  : std_logic_vector(DATA_BITS-1 downto 0);
  signal s_read_addr  : std_logic_vector(ADDR_BITS-1 downto 0);
  signal s_read_data  : std_logic_vector(DATA_BITS-1 downto 0);
begin
  uut : entity work.accumulator(structural)
               generic map
               (
                 ADDR_BITS      => ADDR_BITS,
                 DATA_BITS      => DATA_BITS
               )
               port map
               (
                 clock       => clock,
                 write_addr  => s_write_addr,
                 write_inc   => s_write_inc,
                 read_addr   => s_read_addr,
                 read_data   => s_read_data
               );

  -- clock generation
  clock_stim : process is
  begin
    for cycle in 0 to 10+2+2**ADDR_BITS loop
      clock <= '0';
      wait for 500 ps;
      clock <= '1';
      wait for 500 ps;
    end loop;
  wait;
  end process;
  -- write port (10 clock cycles with activity)
  write_stim : process is
  begin
    s_write_addr <= std_logic_vector(to_unsigned(3, ADDR_BITS));
    s_write_inc  <= std_logic_vector(to_unsigned(7, DATA_BITS));
    wait for 1000 ps;

    s_write_addr <= std_logic_vector(to_unsigned(15, ADDR_BITS));
    s_write_inc  <= std_logic_vector(to_unsigned(48, DATA_BITS));
    wait for 1000 ps;

    s_write_addr <= std_logic_vector(to_unsigned(3, ADDR_BITS));
    s_write_inc  <= std_logic_vector(to_unsigned(2, DATA_BITS));
    wait for 1000 ps;

    s_write_addr <= std_logic_vector(to_unsigned(3, ADDR_BITS));
    s_write_inc  <= std_logic_vector(to_unsigned(7, DATA_BITS));
    wait for 1000 ps;

    s_write_addr <= std_logic_vector(to_unsigned(5, ADDR_BITS));
    s_write_inc  <= std_logic_vector(to_unsigned(128, DATA_BITS));
    wait for 1000 ps;

    s_write_addr <= std_logic_vector(to_unsigned(4, ADDR_BITS));
    s_write_inc  <= std_logic_vector(to_unsigned(64, DATA_BITS));
    wait for 1000 ps;

    s_write_addr <= std_logic_vector(to_unsigned(4, ADDR_BITS));
    s_write_inc  <= std_logic_vector(to_unsigned(0, DATA_BITS));
    wait for 1000 ps;

    s_write_addr <= std_logic_vector(to_unsigned(3, ADDR_BITS));
    s_write_inc  <= std_logic_vector(to_unsigned(16, DATA_BITS));
    wait for 1000 ps;

    s_write_addr <= std_logic_vector(to_unsigned(10, ADDR_BITS));
    s_write_inc  <= std_logic_vector(to_unsigned(144, DATA_BITS));
    wait for 1000 ps;

    s_write_addr <= std_logic_vector(to_unsigned(2, ADDR_BITS));
    s_write_inc  <= std_logic_vector(to_unsigned(34, DATA_BITS));
    wait for 1000 ps;

    s_write_addr <= std_logic_vector(to_unsigned(0, ADDR_BITS));
    s_write_inc  <= std_logic_vector(to_unsigned(0, DATA_BITS));
   wait;
  end process;
  -- read port (at the end, read the entire memory)
  read_stim : process is
  begin
    s_read_addr <= std_logic_vector(to_unsigned(3 ,ADDR_BITS));
    wait for (10+2)*1000 ps;
    for_1 : for i in 0 to 2**ADDR_BITS-1 loop
      s_read_addr <= std_logic_vector(to_unsigned(i,ADDR_BITS));
      wait for 1000 ps;
    end loop;
    wait;
  end process;
end stimulus;
