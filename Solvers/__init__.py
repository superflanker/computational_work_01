from .Solvers import *
from .Ackley import *
from .Beale import *
from .Booth import *
from .Matyas import *
from .Rastrigin2D import *
from .Rastrigin4D import *
from .Rastrigin30D import *
from .Rosenbrock2D import *
from .Rosenbrock4D import *
from .Rosenbrock30D import *
from .common import *

constraints = {"ackley": {"def_func": get_ackley_defs,
                          "search_space": [-32, 32],
                          "design_space": [-0.05, 0.05],
                          "minimum": [0, 0],
                          "isoline_space": [-0.2, 0.2]},
               "beale": {"def_func": get_beale_defs,
                         "search_space": [-4.5, 4.5],
                         "design_space": [-1, 3.5],
                         "minimum": [3, 0.5],
                         "isoline_space": [-4.5, 4.5]},
               "booth": {"def_func": get_booth_defs,
                         "search_space": [-10, 10],
                         "design_space": [0, 4],
                         "minimum": [1, 3],
                         "isoline_space": [-10, 10]},
               "matyas": {"def_func": get_matyas_defs,
                          "search_space": [-10, 10],
                          "design_space": [-10, 10],
                          "minimum": [0, 0],
                          "isoline_space": [-10, 10]},
               "rastrigin2d": {"def_func": get_rastrigin2D_defs,
                               "search_space": [-5.12, 5.12],
                               "design_space": [-0.15, 0.15],
                               "minimum": [0] * 2,
                               "isoline_space": [-0.5, 0.5]},
               "rastrigin4d": {"def_func": get_rastrigin4D_defs,
                               "search_space": [-5.12, 5.12],
                               "design_space": [-0.15, 0.15],
                               "minimum": [0] * 4,
                               "isoline_space": [-0.5, 0.5]},
               "rastrigin30d": {"def_func": get_rastrigin30D_defs,
                                "search_space": [-5.12, 5.12],
                                "design_space": [-0.15, 0.15],
                                "minimum": [0] * 30,
                                "isoline_space": [-0.5, 0.5]},
               "rosenbrock2d": {"def_func": get_rosenbrock2D_defs,
                                "search_space": [-10, 10],
                                "design_space": [-1.5, 1.5],
                                "minimum": [1] * 2,
                                "isoline_space": [-3, 3]},
               "rosenbrock4d": {"def_func": get_rosenbrock4D_defs,
                                "search_space": [-10, 10],
                                "design_space": [-1.5, 1.5],
                                "minimum": [1] * 4,
                                "isoline_space": [-3, 3]},
               "rosenbrock30d": {"def_func": get_rosenbrock30D_defs,
                                 "search_space": [-10, 10],
                                 "design_space": [-1.5, 1.5],
                                 "minimum": [1] * 30,
                                 "isoline_space": [-3, 3]}}

function_names = {"ackley": "Ackley Function",
                  "beale": "Beale Function",
                  "booth": "Booth Function",
                  "matyas": "Matyas Function",
                  "rastrigin2d": "Rastrigin Function",
                  "rastrigin4d": "Rastrigin Function",
                  "rastrigin30d": "Rastrigin Function",
                  "rosenbrock2d": "Rosenbrock Function",
                  "rosenbrock4d": "Rosenbrock Function",
                  "rosenbrock30d": "Rosenbrock Function"}

algorithms = {"dfp": dfp,
              "bfgs": bfgs,
              "lbfgs": lbfgs,
              "lma": lma}
