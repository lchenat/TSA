import pygame
import random
import numpy as np

#color_map = dict([
    #*[('grey-{}'.format(v), plt.cm.Greys(0.1 * v)) for v in range(1, 20)],
    #*[('purple-{}'.format(v), plt.cm.Purples(0.05 * v)) for v in range(1, 20)],
    #*[('blue-{}'.format(v), plt.cm.Blues(0.05 * v)) for v in range(1, 20)],
    #*[('green-{}'.format(v), plt.cm.Greens(0.05 * v)) for v in range(1, 20)],
    #*[('orange-{}'.format(v), plt.cm.Oranges(0.05 * v)) for v in range(1, 20)],
    #*[('red-{}'.format(v), plt.cm.Reds(0.05 * v)) for v in range(1, 20)],
#])

class GridDrawer:
    def __init__(self, color_list):
        self.color_list = np.asarray(color_list)

    # input: a 2-d index matrix
    # output: a 2-d rgb matrix
    def draw(self, indices, repeat=16):
        return np.uint8(255 * np.array(self.color_list[indices, :]).repeat(repeat, 0).repeat(repeat, 1))

class ImgSprite(pygame.sprite.Sprite):
    def __init__(self, rect_pos=(5, 5, 64, 64)):
        super(ImgSprite, self).__init__()
        self.image = None
        self.rect = pygame.Rect(*rect_pos)

    def update(self, image):
        if isinstance(image, str):
            self.image = pygame.image.load(image)
        else:
            self.image = pygame.surfarray.make_surface(image)

class Render(object):
    def __init__(self, size=(320, 320)):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.group = pygame.sprite.Group(ImgSprite()) # the group of all sprites

    def render(self, img):
        img = np.asarray(img).transpose(1, 0, 2)
        self.group.update(img)
        self.group.draw(self.screen)
        pygame.display.flip()
        e = pygame.event.poll()


