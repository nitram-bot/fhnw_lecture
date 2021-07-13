#!/usr/bin/env python
# coding: utf-8

# 
# # Content with notebooks
# 
# You can also create content with Jupyter Notebooks. This means that you can include
# code blocks and their outputs in your book.
# 
# ## Markdown + notebooks
# 
# As it is markdown, you can embed images, HTML, etc into your posts!
# 
# ![](https://myst-parser.readthedocs.io/en/latest/_static/logo.png)
# ![](logo.png)
# 
# You can also $add_{math}$ and
# 
# $$
# math^{blocks}
# $$
# 
# or
# 
# $$
# \begin{aligned}
# \mbox{mean} la_{tex} \\ \\
# math blocks
# \end{aligned}
# $$
# 
# But make sure you \$Escape \$your \$dollar signs \$you want to keep!
# 
# ## MyST markdown
# 
# MyST markdown works in Jupyter Notebooks as well. For more information about MyST markdown, check
# out [the MyST guide in Jupyter Book](https://jupyterbook.org/content/myst.html),
# or see [the MyST markdown documentation](https://myst-parser.readthedocs.io/en/latest/).
# 
# ## Code blocks and outputs
# 
# Jupyter Book will also embed your code blocks and output in your book.
# For example, here's some sample Matplotlib code:

# In[1]:


from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt
import numpy as np
plt.ion()


# In[2]:


# Fixing random state for reproducibility
np.random.seed(19680801)

N = 10
data = [np.logspace(0, 1, 100) + np.random.randn(100) + ii for ii in range(N)]
data = np.array(data).T
cmap = plt.cm.coolwarm
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.5), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]

fig, ax = plt.subplots(figsize=(10, 5))
lines = ax.plot(data)
ax.legend(custom_lines, ['Cold', 'Medium', 'Hot']);


# There is a lot more that you can do with outputs (such as including interactive outputs)
# with your book. For more information about this, see [the Jupyter Book documentation](https://jupyterbook.org)

# Next, we can include the constant term $a$ into the vector $b$. This is done by adding an all-ones column to $\mathbf{X}$: 
#     
# \begin{equation*}
#      \begin{bmatrix}
#       y_1\\
#       y_2\\
#       .  \\
#       .  \\
#       .  \\
#       y_i
#     \end{bmatrix}
#     =
#     \begin{bmatrix}
#       1& x_{11} & x_{21} & x_{31} & \ldots & x_{p1}\\
#       1 &  x_{12} & x_{22} & x_{32} & \ldots & x_{p2}\\
#       &\ldots&\ldots&\ldots&\ldots&\ldots\\
#       &\ldots&\ldots&\ldots&\ldots&\ldots\\
#       1& x_{1i} & x_{2i} & x_{3i} & \ldots & x_{pi}
#     \end{bmatrix}
#     \cdot
#     \begin{bmatrix}
#       a\\
#       b_1\\
#       b_2\\
#       .\\
#       .\\
#       b_p
#     \end{bmatrix}
# \end{equation*}

# \begin{align*}
#     y_1&=a+b_1\cdot x_{11}+b_2\cdot x_{21}+\cdots + b_p\cdot x_{p1}\\
#     y_2&=a+b_1\cdot x_{12}+b_2\cdot x_{22}+\cdots + b_p\cdot x_{p2}\\
#     \ldots& \ldots\\
#     y_i&=a+b_1\cdot x_{1i}+b_2\cdot x_{2i}+\cdots + b_p\cdot x_{pi}\\
# \end{align*}
