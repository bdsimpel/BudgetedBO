from typing import Callable, Dict, List, Optional

import os
import sys

from budgeted_bo_trial import budgeted_bo_trial


def experiment_manager(
    problem: str,
    algos: List[str],
    algos_params: List[Dict],
    restart: bool,
    first_trial: int, 
    last_trial: int,
    get_objective_cost_function: Callable,
    input_dim: int,
    n_init_evals: int,
    budget: float,
) -> None:

    for i, algo in enumerate(algos):
        for trial in range(first_trial, last_trial + 1):

            get_objective_cost_function_output = get_objective_cost_function(seed=trial)

            if len(get_objective_cost_function_output) == 1:
                objective_cost_function = get_objective_cost_function_output[0]
                objective_function = None
                cost_function = None
            elif len(get_objective_cost_function_output) == 2:
                objective_cost_function = None
                objective_function = get_objective_cost_function_output[0]
                cost_function = get_objective_cost_function_output[1]

            budgeted_bo_trial(
                problem=problem,
                algo=algo,
                algo_params=algos_params[i],
                trial=trial,
                restart=restart,
                objective_function=objective_function,
                cost_function=cost_function,
                objective_cost_function=objective_cost_function,
                input_dim=input_dim,
                n_init_evals=n_init_evals,
                budget=budget,
            )
            