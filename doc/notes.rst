Johnson Renormalization
=======================

| [B. R. Johnson JCP **69**, 4678 (1978)]


.. math:: \mathbf{T}_n = -\frac{2(\Delta R)^2\mu}{12\hbar^2} (E\mathbf{I}-\mathbf{V}_n)

.. math::

   \begin{array}{l}
     \mathbf{F}_n = \mathbf{W}_n \boldsymbol{\chi}_n = [\mathbf{I} - \mathbf{T}_n] \chi \\
     \left.  
      \begin{array}{l}
       \mathbf{F}_{n+1} - \mathbf{U}_n\mathbf{F}_n + \mathbf{F}_{n-1} = \mathbf{0} \\
       \mathbf{R}_n = \mathbf{F}_{n+1}\mathbf{F}^{-1}_n \\
      \end{array}
     \right\} 
     \Rightarrow 
      \begin{array}{l}
       \mathbf{R}_n = \mathbf{U}_n - \mathbf{R}^{-1}_{n-1} \\
       \mathbf{U}_n = 12\mathbf{W}^{-1}_n - 10\mathbf{I} \\
      \end{array}
   \end{array}


.. math::

   \begin{aligned}
   \mathbf{V} &= \left( \begin{array}{lllll}
                          V_{00} & V_{01} & V_{02} & \ldots & V_{0\text{n}} \\
                                 & V_{11} & V_{12} & \ldots & V_{1\text{n}} \\
                                 &        & \ddots &        &  \vdots       \\
                                 &        &        & \ddots & V_{\text{n}\text{n}}\\
                       \end{array}
                \right)\end{aligned}


:math:`V_{ii}` diabatic potential energy curves, :math:`V_{i j\neq i}`

off-diagonal coupling terms [H. Lefebvre Brion and R. W. Field table 2.2
page 39.]

| :math:`\Delta \Omega = 0` homogeneous
| :math:`\Delta \Omega = \pm 1` heterogenous - :math:`J` dependent.


 

Outward Solution
~~~~~~~~~~~~~~~~

| 

  .. math::

     \begin{array}{ll}
     \mathbf{R}_n = \mathbf{U}_n - \mathbf{R}^{-1}_{n-1} & 
     n = 1 \rightarrow m \text{ with}\ \mathbf{R}^{-1}_0 = 0 \\
     \mathbf{F}_n = \mathbf{R}^{-1}_n\mathbf{F}_{n+1} & n = m \rightarrow 0
     \text{ with}\ \mathbf{F}_\infty = \mathbf{W}_\infty \boldsymbol{\chi}_\infty 
     \end{array}


Except when :math:`\left| \mathbf{R}_n \right| \sim 0` then

  :math:`\mathbf{R}^{-1}_n` is not well defined.

| Use :math:`\mathbf{F}_n = \mathbf{U}_{n+1}\mathbf{F}_{n+1} - \mathbf{F}_{n+2}`



Inward Solution (:math:`\hat{\ }` - matrices)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  .. math::

     \begin{array}{ll}
     \hat{\mathbf{R}}_n = \mathbf{U}_n - \hat{\mathbf{R}}^{-1}_{n+1} &
     n = \infty \rightarrow m \text{ with}\ \hat{\mathbf{R}}^{-1}_\infty = 0 \\
     \mathbf{F}_n = \hat{\mathbf{R}}^{-1}_n \mathbf{F}_{n-1} &
     n = m \rightarrow \infty
     \ \text{ with}\ \mathbf{F}_0 = \mathbf{W}_0\boldsymbol{\chi}_0\\
     \end{array}


Except when :math:`\left| \mathbf{R}_n \right| \sim 0` then
  :math:`\mathbf{R}^{-1}_n` is not well defined.

| Use :math:`\mathbf{F}_n = \mathbf{U}_{n-1}\mathbf{F}_{n-1} - \mathbf{F}_{n-2}`


Multiple Open Channels
~~~~~~~~~~~~~~~~~~~~~~

| :math:`n_{\rm open}` linearly independent solutions:
|

  .. math::

     \mathbf{R}(R=\infty) =
     \begin{pmatrix}
     1       & 0  & \ldots & 0 \\
     0       & 1  & \ldots & 0 \\
     \vdots  & \vdots  & \ddots & \ldots & \vdots\\
     0       & 0  & \ldots & 1\\
     \end{pmatrix}
     \rightarrow \text{CSE} \rightarrow
     \boldsymbol{\chi}(R) =
     \begin{pmatrix}
        \chi_{00} & \chi_{01} & \chi_{02} & \ldots &
     \chi_{0N_{\text{open}}}\\
        \chi_{10} & \chi_{11} & \chi_{12} & \ldots \\
        \vdots    & \vdots    & \vdots    &        & \vdots \\
        \chi_{N_{\text{total}}0} & &  & \ldots &
     \chi_{N_{\text{total}}N_{\text{open}
     }} \\
     \end{pmatrix}

Normalization
-------------

| [Mies - Molecular Physics **14**, 953 (1980).]

:math:`\boldsymbol{\chi} = \mathbf{JA} + \mathbf{NB}`

:math:`\mathbf{F}^K = \boldsymbol{\chi} \mathbf{A}^{-1} = \mathbf{J} + \mathbf{NK}`

where

:math:`\mathbf{K} = \mathbf{BA}^{-1} = \mathbf{U}\tan \boldsymbol{\xi}
\hat{\mathbf{U}}`.

Physical solution

:math:`\mathbf{F}^S = \mathbf{F}^k\mathbf{U}\cos\boldsymbol{\xi}
e^{\text{i} \boldsymbol{\xi}} \hat{\mathbf{U}} = \text{i}e^{-\text{i}\mathbf{k}R} - \text{i}e^{\text{i}\mathbf{k}R}\mathbf{S}`

Determine matrices , by energy normalization of each wavefunction.

:math:`\chi_{ij} = \left( \frac{2\mu}{\hbar^2\pi} \right) ^{\frac{1}{2}}
\frac{1}{\sqrt{k}} \left[ J_i a_{ij} + N_i b_{ij} \right]` for
potential :math:`i`.
