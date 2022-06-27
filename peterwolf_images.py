import pygame

def Im_create(flag):

    pygame.init()

    display_width = 1500
    display_height = 500

    gameDisplay = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption('Peter and the Wolf')

    black = (0, 0, 0)
    white = (255, 255, 255)

    clock = pygame.time.Clock()
    crashed = False
    PeteImg = pygame.image.load('res/Peter.png').convert()  #Include an alpha channel in the loaded image
    WolfImg = pygame.image.load('res/Wolf.png').convert()

    if flag == 1:
        PeteImg.set_alpha(256)
        WolfImg.set_alpha(256)
    elif flag == 0:
        PeteImg.set_alpha(32)
        WolfImg.set_alpha(32)

    def peter(x, y):
        gameDisplay.blit(PeteImg, (x, y))

    def wolf(x1, y1):
        gameDisplay.blit(WolfImg, (x1, y1))

    x = (display_width * 0.7)
    y = (display_height * 0.2)

    x1 = (display_width * 0.1)
    y1 = (display_width * 0.1)

    while not crashed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

        gameDisplay.fill(white)
        peter(x, y)
        wolf(x1, y1)

        pygame.display.update()
        clock.tick(60)