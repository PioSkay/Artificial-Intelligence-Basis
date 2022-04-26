from game import game, mode

while True:
    try:
        m_mode = int(input("Please provide a gamemode\n 0 - AI vs AI\n 1 - AI vs Player as X\n 2 - AI vs Player as O\n 3 - Player vs Player\n"))
        if m_mode < 0 or m_mode > 4:
            raise ValueError
    except ValueError:
        print("Invalid input!")
    else:
        break
g_obj = game(mode(m_mode))
print(g_obj.play().name)