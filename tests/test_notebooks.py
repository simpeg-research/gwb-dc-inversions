import os
import numpy as np
import testipynb
import os
import numpy as np
import testipynb
import unittest

NBDIR = os.path.sep.join(
    os.path.abspath(__file__).split(os.path.sep)[:-2] + ['notebooks']
)


IGNORE = [
    "DC-1d-parametric-inversion",
    "DC-1d-smooth-inversion",
    "DC-1d-sounding",
    "DC-plot-sounding-data",
    "DC-2d-sounding"
]

Test = testipynb.TestNotebooks(directory=NBDIR, timeout=2800)
Test.ignore = IGNORE
TestNotebooks = Test.get_tests()

if __name__ == "__main__":
    unittest.main()
