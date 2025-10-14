.. qnm_content_site documentation master file, created by
   sphinx-quickstart on Wed Sep 24 10:07:08 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QNM Content Site
============================================

.. figure:: /front_page_figures/mode_content_0010.png
   :alt: Mode content plot for simulation 0010
   :align: center
   :width: 100%

This website contains links (on the left) to the mode content and amplitude stabilities of various target 
modes at a range of start times for each of the 13 publicly available Cauchy characteristic evolved (CCE) 
NR simulations provided by SXS (see `Moxon+ 2020 for more details on CCE <https://arxiv.org/abs/2007.01339>`_). 

The simulations can be accessed `here <https://data.black-holes.org/waveforms/extcce_catalog.html>`_. 

Each page also contains fits, residuals, and a comparison of the mass and spin posteriors to the NR values. 

The code used to produce the analyses and figures on this website is available `here <https://github.com/BGP-QNM-FITS/bgp_qnm_content>`_ 
and the methods are described in Dyer & Moore 2025. 

We use a fully Bayesian framework to fit the ringdown, model the uncertainty on the waveforms, and give posterior distributions on the QNM amplitudes, 
remnant mass, and remnant spin. Example posteriors for the (2,2,n,+) QNMs at 10M are shown below. 

.. figure:: /front_page_figures/corner_plot.png
   :alt: Mode content plot for simulation 0010
   :align: center
   :width: 100%

.. toctree::
   :maxdepth: 2
   :caption: Simulations
   :hidden:

   sim_0001
   sim_0002
   sim_0003
   sim_0004
   sim_0005
   sim_0006
   sim_0007
   sim_0008
   sim_0009
   sim_0010
   sim_0011
   sim_0012
   sim_0013
