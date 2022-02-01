# OGGM massbalance_with_debris

This project is currently under active development. Please contact Seyedhamidreza Mojtabavi (mojtabavi@uni-bremen.de) for any questions.

Temperature index mass balance with debri covers on the elevation band flowline. Work in process!


![](/images/PROTECT.png)

# PROTECT project
- This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement 869304.

At the moment these options are available:
- to compute mass balance with debris on the elevation band flowline (Huss)
- to compute specific mass-balance with debris

## An introduction to debris-covered glaciers

**Author**: Lindsey Nicholson (adapted from Anne Maussion, [Atelier les Gros yeux](http://atelierlesgrosyeux.com/))

This graphic was designed for the paper from [Nicholson et al., 2021](https://doi.org/10.3389/feart.2021.662695).
It illustrates the processes of debris flux through a mountain glacier.

**Download**: [zip file](//images/glacier_debriscovered.zip)

### Image with english labels

![](/images/glacier_debriscovered_englishlabels.png)

Rock debris from the surrounding landscape is transported by glacier ice motion,
and some glaciers can develop into debris covered glaciers, with a layer of
rock rubble covering part of their ablation zone. This debris cover alters
the ablation rate of the glacier, and therefore its overall interaction with
a forcing climate.

# How to install/use !
<!-- structure as in https://github.com/fmaussion/scispack and oggm/oggm -->
the newest OGGM developer version has to be installed in order that MBdebris works:
e.g. do:

    $ conda create --name env_mb
    $ source activate env_mb
    $ git clone  https://github.com/OGGM/oggm.git
    $ cd oggm 
    $ pip install -e .
    $ git clone https://github.com/OGGM/massbalance-debris
    $ cd massbalance-debris
    $ pip install -e .

Test the installation via pytest while being in the massbalance-sandbox folder:

    $ pytest .

The MBsandbox package can be imported in python by

    >>> import MBdebris


# How to install/use in cluster environments/ Singularity and docker containers 
(please see https://docs.oggm.org/en/stable/practicalities.html#singularity-and-docker-containers)


    # All commands in the EOF block run inside of the container
    singularity exec /path/to/oggm/image/oggm_20191122.sif bash -s <<EOF
      set -e
      # Setup a fake home dir inside of our workdir, so we don't clutter the
      # actual shared homedir with potentially incompatible stuff
      export HOME="$OGGM_WORKDIR/fake_home"
      mkdir "\$HOME"
      # Create a venv that _does_ use system-site-packages, since everything is
      # already installed on the container. We cannot work on the container
      # itself, as the base system is immutable.
      python3 -m venv --system-site-packages "$OGGM_WORKDIR/oggm_env"
      source "$OGGM_WORKDIR/oggm_env/bin/activate"
      # OPTIONAL: make sure latest pip is installed
      pip install --upgrade pip setuptools
      # OPTIONAL: install massbalance-debris + install another OGGM version (here provided by its git commit hash) 
      pip install "git+https://github.com/OGGM/oggm.git@ce22ceb77f3f6ffc865be65964b568835617db0d"
      pip install "git+https://github.com/OGGM/massbalance-debris"
      # Finally, you can test OGGM with `pytest --pyargs oggm`, or run your script:
    YOUR_RUN_SCRIPT_HERE
    EOF


# inside of MBdebris

- ***MassBalance_with_Debris.py***: ***debris_to_gdir***, ***elevation_band_flowline_debris*** and ***mb_modules***
- **tests**: tests for different functions (Work in process)


# docs/*:

### simple use case: 

- ***how_to_use_MBdebris.ipynb***: shows mb with debri cover for different glaciers (Hintereisferner, Baltoro and Tipra glaciers)
    - plot debris thickness
    - plot debris correction factor
    - plot mass balance as a function of elevation (with and without debris)
    - plot MSB_specific_mb (with and without debris)
    - plot specific MB with reference mass-balance (WGMS)
