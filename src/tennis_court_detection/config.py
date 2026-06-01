import matplotlib.pyplot as plt


plt.rcParams['image.cmap'] = 'gray'


_DEBUG_MODE = False


def set_debug_mode(mode: bool) -> None:
    global _DEBUG_MODE
    _DEBUG_MODE = mode


def get_debug_mode() -> bool:
    global _DEBUG_MODE
    return _DEBUG_MODE