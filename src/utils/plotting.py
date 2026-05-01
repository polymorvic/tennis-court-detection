import matplotlib.pyplot as plt

from src.utils.common import ArrayLike

def display_img(img: ArrayLike, title: str = "") -> None:
    plt.imshow(img)
    plt.title(title)
    plt.show()