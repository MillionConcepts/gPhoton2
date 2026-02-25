---
title: 'gPhoton2: A High-Performance Processing Pipeline for GALEX Data'
tags:
  - Python
  - astronomy
  - ultraviolet
  - photometry
  - time-domain
  - GALEX
authors:
  - name: Chase Million
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Michael St. Clair
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Rebecca Albach
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Zack Weinberg
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Million Concepts, Louisville, KY, USA
    index: 1
date: 19 February 2026
bibliography: paper.bib
---

# Summary

gPhoton2 is a Python pipeline that transforms raw data from the GALEX (Galaxy Evolution Explorer) ultraviolet mission into calibrated, science-ready products for astronomy. The pipeline retrieves telemetry and calibration files from the Mikulski Archive for Space Telescopes (MAST), unpacks and calibrates the data to produce time-tagged photon event lists, generates images at user-specified cadences and integration depths, and performs aperture photometry to extract lightcurves.

gPhoton2 is a complete rewrite of the gPhoton library [@million2016] (henceforth _gPhoton1_), achieving 1-3 orders of magnitude performance improvements through a variety of architecture and implementation improvements, including Numba JIT compilation [@lam2015numba], Apache Parquet columnar storage [@parquet], sparse matrix representations [@scipy], vectorized computations, and parallelization. Unlike gPhoton1, gPhoton2 implements source _detection_, in addition to measurement. The gPhoton2 pipeline can process an entire GALEX observation---about 0.5 square degrees, and containing 100s-1000s of sources---about 10x faster than gPhoton1 could generate a lightcurve of a _single_ source. This enables efficient reprocessing at the level of observation or catalogs.

# Statement of Need

Between 2003 and 2013, the GALEX mission acquired observations of ~100 million sources over ~2250 square degrees in two ultraviolet bands, Near-UV (NUV, 1750-2750 \AA) and Far-UV (FUV, 1350-1750 \AA) [@martin2005]. While GALEX's photon-counting micro-channel plate detectors recorded individual photon arrival times at 5-millisecond resolution, the standard mission pipeline [@morrissey2007] only produced integrated images, discarding temporal information and preventing systematic study of UV variability on timescales from seconds to years. The original mission pipeline is neither publicly available nor runnable.

gPhoton1 [@million2016] addressed this gap, enabling studies of stellar flares [@fleming2022; @jackman2023; @doyle2018], pulsating hot subdwarfs [@boudreaux2017], and other UV variability [@million2023gfcat; @davenport2018galex]. However, gPhoton1's architecture---optimized for targeted, single-source queries---made bulk processing of the full GALEX archive impractical, limiting UV time-domain studies to hand-selected subsets of sources (see State of the Field).

gPhoton2 enables efficient reprocessing of the full archive of GALEX direct imaging observations. Researchers studying rapid UV phenomena can now generate large numbers of lightcurves at arbitrary cadences without prohibitive time or compute costs. Researchers interested in integrated flux can apply alternative methodologies such as new background estimation algorithms or forced photometry.

# State of the field

gPhoton1 [@million2016], the direct predecessor to gPhoton2, addressed the gap left by the GALEX mission pipeline by preprocessing over a trillion photon events and storing them in a publicly accessible database at the Mikulski Archive for Space Telescopes (MAST). Clientside command-line scripts queried this database on demand to produce calibrated light curves and image stamps [@million2016userguide]. This design suited targeted, single-source queries well, but the combination of repeated database lookups and over-network data transfers imposed severe performance limits: producing a single lightcurve (with ~30 second cadence over a ~30 minute observation) typically required up to 30 minutes of wall time, making archive-scale bulk processing impractical.

The original GALEX mission pipeline [@morrissey2007] is neither publicly available nor runnable outside its original hardware environment, making community tools the only viable path to photon-level GALEX reprocessing.

Other space ultraviolet observatories---including Swift/UVOT, the XMM-Newton Optical Monitor, and Hubble Space Telescope UV instruments---have dedicated reduction pipelines specific to their respective instruments and data formats. No general-purpose UV photometry tool handles GALEX's photon-counting detector architecture or its calibration chain.

Mission-agnostic time-domain photometry tools such as Lightkurve [@lightkurve2018] were designed for optical, CCD-based missions (Kepler, TESS) and do not include the GALEX-specific calibration steps---walk and wiggle corrections, stim pulse removal, spatial linearity corrections, and aspect solution interpolation---required for scientifically valid UV photometry. gPhoton2 fills an analogous role for GALEX: a mission-specific pipeline that transforms raw photon data into calibrated, science-ready products.

The decision to rewrite rather than extend gPhoton1 was driven by the incompatibility of its architecture with bulk processing requirements. Efficient archive-scale reprocessing requires local rather than remote data access, vectorized rather than event-by-event computation, and image-based rather than photon-list photometry---changes that amount to a near-total rewrite of gPhoton1's core. The additional burden of maintaining backward compatibility with gPhoton1's interface, combined with its Python 2 codebase and accumulated technical debt, made a clean rewrite the more practical path.

# Software design

gPhoton2 employs a three-stage modular architecture separating calibration, image generation, and photometry. This design enables flexible reprocessing: users can regenerate images at different cadences or extract photometry for different source lists without repeating expensive calibration.

The first stage, `photonpipe`, retrieves raw telemetry from MAST and applies the full calibration chain: detector walk and wiggle corrections, stim pulse removal, spatial linearity corrections, post-CSP (Coarse Sun Point) adjustments, hotspot masking, and aspect solution interpolation. Calibrated photon events are stored as Apache Parquet files [@parquet], whose columnar format provides 3-5x compression over gzip-compressed FITS while enabling fast random access for downstream processing.

The second stage, `moviemaker`, bins calibrated photons into temporal image sequences at user-specified cadences. Sparse matrix representations [@scipy] handle the predominantly empty pixels in UV images, reducing memory requirements 10-100x compared to dense arrays. This enables processing deep observations on modest hardware while outputting standard FITS cubes for compatibility.

The third stage, `lightcurve`, performs source detection using DAOStarFinder [@photutils] and extracts aperture photometry with full exposure time and flag tracking. Independent per-frame detection enables robust photometry for transient or highly variable sources.

Performance-critical routines use Numba JIT compilation [@lam2015numba], achieving 50-200x speedups over interpreted Python without sacrificing readability. Memory usage and thread counts are configurable for operation from laptops to cloud nodes. Both programmatic and command-line interfaces support integration into larger workflows.

# Research impact statement

gPhoton2 was developed primarily to enable the GALEX Legacy Catalog (GLCAT) project, which aims to produce uniform time-domain photometry for all sources in the GALEX archive. The full archive of approximately 100,000 observations can now be processed in weeks rather than the decades that would be required with gPhoton1.

The gPhoton2 software is extremely flexible, and eases use of the GALEX data for a wide cross-section of scientific analyses. Although data inputs and outputs conform to generally accepted idioms of the Python scientific and astronomical software ecosystems (e.g. [@harris2020numpy; @pedregosa2011scikit; @astropy2022]) gPhoton2 is also highly specialized and optimized for _only_ working with GALEX data. No other publicly available software provides comparable functionality.

Beyond GLCAT, gPhoton2 enables systematic analysis of archival GALEX observations for stellar flares, white dwarf pulsations, eclipsing binaries, and AGN variability---studies previously limited to small hand-selected samples. The pipeline's arbitrary cadence support---from native 5-millisecond resolution to observation-integrated depths---serves science cases from rapid transient detection to long-term variability surveys.

gPhoton2 is publicly available on GitHub with documentation, example notebooks, and continuous integration testing. The software has been presented at the 243rd Meeting of the American Astronomical Society.

# AI usage disclosure

This paper was drafted with assistance from generative LLM tools. The gPhoton2 software and documentation may have been developed in part with the assistance of generative LLM tools. All software, documentation, and the content in this paper were reviewed, verified, and edited by the authors. Humans are accountable for all work products.

# Acknowledgements

This work was supported by NASA grants 80NSSC18K0084 and 80NSSC21K0421. We thank the MAST archive team at STScI for data access and support. This work made use of [Astropy](http://www.astropy.org), a community-developed core Python package and an ecosystem of tools and resources for astronomy [@astropy2013; @astropy2018; @astropy2022].

# References
