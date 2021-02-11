# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 01:31:20 2020

@author: rowe1
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

import skimage
import skimage.feature
import skimage.viewer

class Edge_Detect(object):
    def __init__(self, file, rows = None, sigma = 1.0, low_threshold = None, high_threshold = None):
        self.image = skimage.io.imread(fname = file, as_gray = True)
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.rows = rows
        
        if self.rows is not None:
            '''Reshape image to (rows, columns) while maintaining the image's aspect ratio'''
            self.columns = int(self.rows*self.image.shape[1]//self.image.shape[0])
            self.image = skimage.transform.resize(self.image, (self.rows, self.columns))
        
        self.edges = skimage.feature.canny(image=self.image, sigma=1.0, low_threshold=None, high_threshold=None)
        
        if np.mean(self.edges) > 0.5:
            self.invert()
    
    def invert(self):
        '''Switch black to white and white to black'''
        R, C = len(self.edges), len(self.edges[0])
        for r in range(R):
            for c in range(C):
                self.edges[r][c] = 1 - self.edges[r][c]
        
    def show(self):
        plt.figure(dpi = 200)
        plt.imshow(self.image)
        plt.figure(dpi = 200)
        plt.imshow(self.edges)


if __name__ == '__main__':
    import glob
    files = glob.glob('./graphics/maze_images/*')
    file = files[8]
    plt.close('all')
    image = Edge_Detect(file, rows = 50)
    image.show()