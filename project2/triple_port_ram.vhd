library IEEE;
use     IEEE.std_logic_1164.all;
use     IEEE.numeric_std.all;

entity triple_port_ram is
  generic
  (
    ADDR_BITS : integer range 2 to 16;
    DATA_BITS : integer range 1 to 32
  );
  port
  (
    clock        : in std_logic;
    -- write port
    write_addr : in  std_logic_vector(ADDR_BITS-1 downto 0);
    write_data : in  std_logic_vector(DATA_BITS-1 downto 0);
    -- read port
    read_addr  : in  std_logic_vector(ADDR_BITS-1 downto 0);
    read_data  : out std_logic_vector(DATA_BITS-1 downto 0) := (others => '0');
    -- aux read port
    aux_read_addr  : in  std_logic_vector(ADDR_BITS-1 downto 0);
    aux_read_data  : out std_logic_vector(DATA_BITS-1 downto 0) := (others => '0')
  );
end triple_port_ram;

architecture behavioral of triple_port_ram is
  type ram_type is array (0 to 2**ADDR_BITS-1) of std_logic_vector(DATA_BITS-1 downto 0);
  signal ram : ram_type := (others => (others => '0'));
begin
  process (clock) is
  begin
    if rising_edge(clock) then
      ram(to_integer(unsigned(write_addr))) <= write_data;
    end if;
  end process;
  read_data <= transport ram(to_integer(unsigned(read_addr))) after 200 ps;
  aux_read_data <= transport ram(to_integer(unsigned(aux_read_addr))) after 200 ps;
end behavioral;