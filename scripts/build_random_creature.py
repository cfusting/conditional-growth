from grow.entities.conditional_growth_genome import ConditionalGrowthGenome
import os
from grow.function_approximators.mlp import MLPGrowthFunction 
from grow.utils.tensor_to_cdata import tensor_to_cdata, add_cdata_to_xml
import argparse


parser = argparse.ArgumentParser(
    description="Generate a VXD file for the voxcraft simulation."
)
parser.add_argument("--growth-iterations", type=int)
parser.add_argument("--search-radius", type=int)
parser.add_argument("--max-voxels", type=int)
parser.add_argument("--record-history", 
                    action="store_true")
parser.add_argument("--sample", action="store_true")

args = parser.parse_args()

l_system = ConditionalGrowthGenome(
    growth_iterations=args.growth_iterations,
    search_radius=args.search_radius,
    max_voxels=args.max_voxels,
)

# The output size for the growth function is the
# number of possible configurations that can be grown.
input_size = len(l_system.materials)
output_size = len(l_system.configuration_map)

growth_function = MLPGrowthFunction(input_size, output_size, sample=args.sample)

# Grow the creature.
l_system.expand(growth_function)

# Prepare creature for simulation.
X = l_system.to_tensor()
C = tensor_to_cdata(X)

# Change working directory to scipts to use relative paths.
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
robot_path = "../data/robot.vxd"

add_cdata_to_xml(C, X.shape[0], X.shape[1], X.shape[2], robot_path,
                 record_history=args.record_history)
