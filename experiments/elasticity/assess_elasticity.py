import argparse
import numpy as np
import subprocess
from time import time
from grow.utils.output import get_voxel_positions
from grow.utils.tensor_to_cdata import tensor_to_cdata, add_cdata_to_xml
from grow.utils.simulation import write_configs_to_base
from grow.utils.fitness import has_fallen

parser = argparse.ArgumentParser(
    description=(
        "Determine whether an elasticity / density "
        "ratio causes a structure to collapse."
    )
)

parser.add_argument("--num-voxels", type=int, nargs="+")
parser.add_argument("--elastic-mod", type=float,  nargs="+")
parser.add_argument("--density", type=float, nargs="+")
parser.add_argument("--structure", type=str, nargs="+")
parser.add_argument("--time", type=int)
parser.add_argument("--sim-build-path", type=str)
parser.add_argument("--template-data-folder-path", type=str)
parser.add_argument("--output-path", type=str)
args = parser.parse_args()


def run_simulation(data_folder_path, output_file_path):
    print(data_folder_path)
    run_command = f"./voxcraft-sim -i {data_folder_path} -o {output_file_path}"
    subprocess.run(
        run_command.split(),
        cwd=args.sim_build_path,
    )


def generate_robot(structure, num_voxels, file_path):
    if structure == "cube":
        n = int(np.floor(np.cbrt(num_voxels)))
        X = np.ones((n, n, n))
    else:
        raise ValueError(f"Invalid structure: {structure}")
    C = tensor_to_cdata(X)
    add_cdata_to_xml(
        C, X.shape[0], X.shape[1], X.shape[2], robot_path, record_history=False
    )


output_folder_path = f"{args.output_path}/{time()}"
for n in args.num_voxels:
    for e in args.elastic_mod:
        for d in args.density:
            for s in args.structure:
                print(
                    f"Testing for stability: Number of Voxels: {n}, "
                    "Elastic_Mod: {e}, Density: {d}, Structure: {s}"
                )

                output_folder_path += f"_{n}_{e}_{d}_{s}"
                data_folder_path = f"{output_folder_path}/data"
                base_path = f"{data_folder_path}/base.vxa"
                robot_path = f"{data_folder_path}/robot.vxd"
                output_file_path = f"{output_folder_path}/output.xml"

                subprocess.run(f"mkdir -p {output_folder_path}".split())
                subprocess.run(
                    f"cp -R {args.template_data_folder_path} {data_folder_path}".split()
                )

                write_configs_to_base(base_path, e, d, args.time)
                generate_robot(s, n, robot_path)
                run_simulation(data_folder_path, output_file_path)
                initial_positions, final_positions = get_voxel_positions(
                    output_file_path
                )
                stable = not has_fallen(initial_positions, final_positions)

                print(f"{output_file_path} is stable: {stable}")
