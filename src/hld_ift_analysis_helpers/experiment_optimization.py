#command prompt
#set AN_SRC="D:\projects\HLD_parameter_determination\hld_ift_analysis_helpers\src\hld_ift_analysis_helpers"
#python %AN_SRC%\experiment_optimization.py

import sys
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_http/src")
from scipy.optimize import minimize
import numpy as np
import itertools

from hld_ift_http.solution import Solution, Solution_Component
from hld_ift_http.solution_repository import Solution_Repository

def unif_to_sum_1(x):
    """
        converts numpy array of shape (N,) with all values in interval (0,1) to
        a list with N+1 items that sum up to 1
        x1, x2, .., xN --> x1, (1 - x1)x2, (1 - x1)(1 - x2)x3, ... ,(1 - x1)(1 - x2)...(1 - x(N-1))xN, (1 - x1)(1 - x2)...(1 - x{N-1})(1 - xN)
    """
    temp = list(itertools.accumulate(x.tolist(), lambda a, b: a * (1 - b), initial = 1))
    temp2 = x.tolist() + [1]
    print(temp)
    print(temp2)
    print(len(temp))
    out = list(map(lambda x,y: x*y, temp, temp2))
    print(out)
    return out

def fun_001(x,y,z):
    return x + y + z

def fun_001_wrap(x):
    params = unif_to_sum_1(x)
    return fun_001(params[0], params[1], params[2])

arr = np.ones((2,)) * 0.5
params = unif_to_sum_1(arr)
print(params)

my_bounds = [(0,1), (0,1)]
print(arr)



out = minimize(fun_001_wrap, arr, bounds = my_bounds) 
print(out)
print(type(out))

# ---------- path 
#...\DATA_PATH    = "C:/Users/admin/Documents/Data/aikars/opentron/AOT_IB45__C07C16__NaCl"
#...\LOG_PATH     = "C:/Users/admin/Documents/Data/aikars/opentron/AOT_IB45__C07C16__NaCl/log.log"
#...\CONFIG_PATH  = "C:/Users/admin/Documents/Data/aikars/opentron/AOT_IB45__C07C16__NaCl/config"
#SOLUTION_REPOSITORY_PATH = "C:/Users/admin/Documents/Data/aikars/opentron/AOT_IB45__C07C16__NaCl/config/solution_repository.json"
SOLUTION_REPOSITORY_PATH = "//huckdfs-srv.science.ru.nl/huckdfs/RobotLab/Storage-Miscellaneous/aigars/temp/AOT_IB-45_C7C16_NaCl/config/solution_repository.json"

# ............................................................ solution repository
rep = Solution_Repository.fromJSON(file = SOLUTION_REPOSITORY_PATH)

surfactant_lbl = "aerosol_ib-45"
nacl_lbl = "NaCl"

surfactant_stock_name = "44perc_IB-45_in_wt"
nacl_stock_name = "30g_NaCl_in_100mL_water_20250225"

m_v_surfactant_stock_target = 0.2
m_v_nacl_stock_target = 0.077

water_stock = rep.items["water"]
nacl_stock =  rep.items[nacl_stock_name]
suf_stock = rep.items[surfactant_stock_name]

def m_v_concentration(solution: Solution, component: str):
   return solution.ro *  solution.components[component].w if component in solution.components.keys() else 0

def m_v_surf(solution):
    return m_v_concentration(solution, surfactant_lbl)

def m_v_nacl(solution):
    return m_v_concentration(solution, nacl_lbl)

def optimization_target(solution, cNaCl_0 = 0, cSurfactant_0 = 0): 
    return abs(m_v_nacl(solution) - cNaCl_0) + abs(m_v_surf(solution) - cSurfactant_0)

def make_mix_virtual_fn(solutions: list, cNaCl_0, cSurfactant_0):
    qty_type_lst = list(itertools.repeat("volume", len(solutions)))
    return lambda x: optimization_target(Solution.combine(unif_to_sum_1(x), qty_type_lst, solutions, None), cNaCl_0, cSurfactant_0)


stock_lst = [suf_stock, nacl_stock, water_stock]
def optimize_stock_report(stock_lst, nacl_trg, surf_trg):
    fn = make_mix_virtual_fn(stock_lst, nacl_trg, surf_trg)
    x0 = np.ones((2,)) * 0.5
    my_bounds = [(0,1), (0,1)]
    out = minimize(fn, x0, bounds = my_bounds) 
    print(out)
    print(type(out))
    print(f'x: {out.x}')
    print(f'x_converted: {unif_to_sum_1(out.x)}')
    print(f'success: {out.success}')

    recipe = unif_to_sum_1(out.x)
    sol = Solution.combine(recipe,
                list(itertools.repeat("volume", len(recipe))),
                stock_lst,
                None)

    print(f'solution_created:\n{str(sol)}')
    print(f'nacl: {m_v_nacl(sol)}, surf: {m_v_surf(sol)}')
                
optimize_stock_report(stock_lst, m_v_nacl_stock_target, m_v_surfactant_stock_target)
optimize_stock_report(stock_lst, 0, m_v_surfactant_stock_target)
optimize_stock_report(stock_lst, m_v_nacl_stock_target, 0)
optimize_stock_report(stock_lst, 0, 0)


