import numpy
import numpy.testing

hermcoefficients = [
    (numpy.array([1])),
    (numpy.array([0, 2])),
    (numpy.array([-2, 0, 4])),
    (numpy.array([0, -12, 0, 8])),
    (numpy.array([12, 0, -48, 0, 16])),
    (numpy.array([0, 120, 0, -160, 0, 32])),
    (numpy.array([-120, 0, 720, 0, -480, 0, 64])),
    (numpy.array([0, -1680, 0, 3360, 0, -1344, 0, 128])),
    (numpy.array([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256])),
    (numpy.array([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512])),
]
