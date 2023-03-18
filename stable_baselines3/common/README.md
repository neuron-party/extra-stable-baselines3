# extra files 

`research_method_1.py`: Modified on_policy_algorithm.py file. Backward training algorithm for hard levels ONLY, i.e no probability sampling. Paired with init_research_method_1, where each trajectory for each level has its own set of boundaries. <br>
`research_method_2.py`: Modified on_policy_algorithm.py file. Backward training algorithm for hard levels ONLY. Paired with init_research_method_2, where all trajectories for a single level have the same set of boundaries. <br>
`research_method_3.py`: Modified on_policy_algorithm.py file. Backward training for the entire level distribution, i.e hard and easy levels. Paired with init_research_method_2, where all trajectories for a single level have the same set of boundaries. <br>

### future files
* modified on policy algorithm for sampling boundaries, where the starting state for a trajectory is anywhere in the range of [boundary_1, boundary_2] rather than enforcing the same boundary repeatedly until it can move on.