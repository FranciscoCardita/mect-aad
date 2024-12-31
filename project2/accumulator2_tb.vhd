--
-- AAD 2024/2025, accumulator2 test bench
--

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity accumulator2_tb is
end accumulator2_tb;

architecture stimulus of accumulator2_tb is
  constant ADDR_BITS      : positive := 4;
  constant DATA_BITS_LOG2 : positive := 3;
  constant DATA_BITS      : integer  := 2**DATA_BITS_LOG2;
  signal clock         : std_logic;
  signal s_write_addr  : std_logic_vector(        ADDR_BITS-1 downto 0);
  signal s_write_inc   : std_logic_vector(2**DATA_BITS_LOG2-1 downto 0);
  signal s_write_shift : std_logic_vector(   DATA_BITS_LOG2-1 downto 0);
  signal s_read_addr   : std_logic_vector(        ADDR_BITS-1 downto 0);
  signal s_read_data   : std_logic_vector(2**DATA_BITS_LOG2-1 downto 0);
begin
  uut : entity work.accumulator2(structural)
               generic map
               (
                 ADDR_BITS      => ADDR_BITS,
                 DATA_BITS_LOG2 => DATA_BITS_LOG2
               )
               port map
               (
                 clock       => clock,
                 write_addr  => s_write_addr,
                 write_inc   => s_write_inc,
                 write_shift => s_write_shift,
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
    -- T=0 --- a[3] = 7
    s_write_addr  <= std_logic_vector(to_unsigned(   3 ,ADDR_BITS));
    s_write_inc   <= std_logic_vector(to_unsigned(   7 ,2**DATA_BITS_LOG2));
    s_write_shift <= std_logic_vector(to_unsigned(   0 ,DATA_BITS_LOG2));
    wait for 1000 ps;
    -- T=1 --- a[15] = 48 (final)
    s_write_addr  <= std_logic_vector(to_unsigned(  15 ,ADDR_BITS));
    s_write_inc   <= std_logic_vector(to_unsigned(   3 ,2**DATA_BITS_LOG2));
    s_write_shift <= std_logic_vector(to_unsigned(   4 ,DATA_BITS_LOG2));
    wait for 1000 ps;
    -- T=2 --- a[ 3] = 9
    s_write_addr  <= std_logic_vector(to_unsigned(   3 ,ADDR_BITS));
    s_write_inc   <= std_logic_vector(to_unsigned(   1 ,2**DATA_BITS_LOG2));
    s_write_shift <= std_logic_vector(to_unsigned(   1 ,DATA_BITS_LOG2));
    wait for 1000 ps;
    -- T=3 --- a[ 3] = 16
    s_write_addr  <= std_logic_vector(to_unsigned(   3 ,ADDR_BITS));
    s_write_inc   <= std_logic_vector(to_unsigned(   7 ,2**DATA_BITS_LOG2));
    s_write_shift <= std_logic_vector(to_unsigned(   0 ,DATA_BITS_LOG2));
    wait for 1000 ps;
    -- T=4 --- a[ 5] = 128 (final)
    s_write_addr  <= std_logic_vector(to_unsigned(   5 ,ADDR_BITS));
    s_write_inc   <= std_logic_vector(to_unsigned(  64 ,2**DATA_BITS_LOG2));
    s_write_shift <= std_logic_vector(to_unsigned(   1 ,DATA_BITS_LOG2));
    wait for 1000 ps;
    -- T=5 --- a[ 4] = 64
    s_write_addr  <= std_logic_vector(to_unsigned(   4 ,ADDR_BITS));
    s_write_inc   <= std_logic_vector(to_unsigned(   1 ,2**DATA_BITS_LOG2));
    s_write_shift <= std_logic_vector(to_unsigned(   6 ,DATA_BITS_LOG2));
    wait for 1000 ps;
    -- T=6 --- a[ 4] = 64 (final)
    s_write_addr  <= std_logic_vector(to_unsigned(   4 ,ADDR_BITS));
    s_write_inc   <= std_logic_vector(to_unsigned(  16 ,2**DATA_BITS_LOG2));
    s_write_shift <= std_logic_vector(to_unsigned(   6 ,DATA_BITS_LOG2));
    wait for 1000 ps;
    -- T=7 --- a[ 3] = 32 (final)
    s_write_addr  <= std_logic_vector(to_unsigned(   3 ,ADDR_BITS));
    s_write_inc   <= std_logic_vector(to_unsigned(   1 ,2**DATA_BITS_LOG2));
    s_write_shift <= std_logic_vector(to_unsigned(   4 ,DATA_BITS_LOG2));
    wait for 1000 ps;
    -- T=8 --- a[10] = 144 (final)
    s_write_addr  <= std_logic_vector(to_unsigned(  10 ,ADDR_BITS));
    s_write_inc   <= std_logic_vector(to_unsigned( 100 ,2**DATA_BITS_LOG2));
    s_write_shift <= std_logic_vector(to_unsigned(   2 ,DATA_BITS_LOG2));
    wait for 1000 ps;
    -- T=9 --- a[ 2] = 34 (final)
    s_write_addr  <= std_logic_vector(to_unsigned(   2 ,ADDR_BITS));
    s_write_inc   <= std_logic_vector(to_unsigned(  17 ,2**DATA_BITS_LOG2));
    s_write_shift <= std_logic_vector(to_unsigned(   1 ,DATA_BITS_LOG2));
    wait for 1000 ps;
    -- done
    s_write_addr  <= std_logic_vector(to_unsigned(   0 ,ADDR_BITS));
    s_write_inc   <= std_logic_vector(to_unsigned(   0 ,2**DATA_BITS_LOG2));
    s_write_shift <= std_logic_vector(to_unsigned(   0 ,DATA_BITS_LOG2));
    wait;
  end process;
  -- read port (at the end, read the entire memory)
  read_stim : process is
  begin
    s_read_addr <= std_logic_vector(to_unsigned( 3 ,ADDR_BITS));
    wait for (10+2)*1000 ps;
    for_1 : for i in 0 to 2**ADDR_BITS-1 loop
      s_read_addr <= std_logic_vector(to_unsigned(i,ADDR_BITS));
      wait for 1000 ps;
    end loop;
    wait;
  end process;
end stimulus;