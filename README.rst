See full documentation at https://nrb5089-radar.readthedocs.io/en/latest/

Notes
------

The primary radar signal processing blocks are contained in ```core```, do not modify these.

Local user directory should be used for development of simulatio, ```sim``` is intended to be an example, see ```test_sim``` for example settings.

This repository is in work, current action items: 

- Combine ``test_sim.py``, ``sim.py`` and the classes for ``MonostaticRadar`` into a series of example files.
- Add tutorials from SignalProcessingTutorial repo.

- Add multiple IF stages to ``core.Waveform`` object.