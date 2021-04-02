import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def imshow(input, vmax=None, vmin=None, title=None, cmap=None, figsize=None, fig=None, axes=None):
        if (axes is None) or (fig is None):
            fig, axes = plt.subplots(figsize=figsize)
        cmap = cmap if cmap != None else "rainbow"
        if vmax == None:
            vmax = np.max(input)
        if vmin == None:
            vmin = np.min(input)
        no_cbar_ticks = False
        im = axes.imshow(input, vmax=vmax, vmin=vmin, origin='lower', cmap=cmap)
        axes.set_title(title)
        axes.grid(False)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('left', size='5%', pad=0.1)
        axes.yaxis.set_ticks_position('right')
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cax.yaxis.set_ticks_position('left')
        if (axes is None) or (fig is None):
            plt.show()

class PatternGenerator():
    """
    임의로 생성된 pattern을 사용하려면
    pattern generator object를 생성하세요.
    pattern generator를 indexing하면 패턴을 생성할 수 있습니다.  
    """
    
    def __init__(self, shape, num_patterns):
        np.random.seed(seed=42)
        self.patterns = []
        for i in range(num_patterns):
            pattern_col = np.random.choice(2, shape[1])
            pattern = pattern_col * np.ones((shape))
            self.patterns.append(pattern)
            
    def __getitem__(self, idx):
        return self.patterns[idx]

    def __len__(self):
        return len(self.patterns)