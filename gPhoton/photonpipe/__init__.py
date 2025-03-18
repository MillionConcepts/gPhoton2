"""
.. module:: PhotonPipe
   :synopsis: A recreation / port of key functionality of the GALEX mission
       execute_pipeline to generate calibrated and sky-projected photon-level data from
       raw spacecraft and detector telemetry. Generates time-tagged photon
       lists given mission-produced -raw6, -scst, and -asprta data.
"""

from .core import execute_photonpipe

__all__ = ["execute_photonpipe"]
